"""Training / evaluation / bootstrapping loops, with full logging.

This module holds the model trainers:
  * :class:`Trainer`        - NAEA (TransE + neighbourhood-aware attention).
  * :class:`BootEATrainer`  - BootEA (AlignE + swapping + MWGM bootstrapping).
  * :class:`AliNetTrainer`  - AliNet (gated multi-hop GNN, full-batch).
  * :class:`KECGTrainer`    - KECG (shared GAT cross-graph + TransE, alternating).
  * :class:`GCNAlignTrainer`- GCN-Align (shared 2-layer GCN, margin L1 loss, full-batch).
  * :class:`JAPETrainer`    - JAPE (TransE SE on merged-seed graph + attribute AE).

The :class:`Trainer` wires together the data samplers, the model and the
metrics, and persists everything the config asks for:

  * ``training.txt``  : every log line (handled by the logger in config.py)
  * ``loss.csv``      : per-epoch loss components
  * ``metrics.csv``   : per-evaluation MRR and Hit@k (both directions plus avg)
  * ``model.pt`` / ``model_best.pt`` : checkpoints (state_dict plus config)
  * ``embeddings.pt`` : final entity and relation embedding tables
  * ``loss_curve.png`` / ``metrics_curve.png`` : figures

It also implements BootEA-style bootstrapping: periodically the model's own
high-confidence mutual nearest neighbours (on the unlabelled test pool, gold
labels are never read) are added to the alignment training set.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .data import (AlignSampler, DBP15K, DynamicTripleSampler, Swapper,
                   TripleSampler, kg_of_entity)
from .models.naea import NAEA, alignment_loss, margin_ranking_loss
from .models.bootea import BootEA, alignment_pull_loss, limit_based_triple_loss
from .models.alinet import AliNet, alinet_align_loss, alinet_limit_loss, alinet_relation_loss
from .models.kecg import KECG, kecg_cg_loss, kecg_ke_loss
from .models.gcnalign import GCNAlign, gcnalign_loss
from .models.jape import JAPE, jape_se_loss
from .models.dgmc import DGMC
from .models.mraea import MRAEA, mraea_align_loss
from .models.rrea import RREA
from .utils import metrics as M
from .utils.plotting import plot_loss_curves, plot_metric_curves, set_modern_dark_style


class Trainer:
    def __init__(self, cfg, data: DBP15K, model: NAEA, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.triple_sampler = TripleSampler(data, self.device, neg=cfg.train.neg_samples)
        # seed alignment sampler is FIXED to the gold seed pairs and never changes
        self.align_sampler = AlignSampler(data, self.device, neg=cfg.train.neg_samples)
        # pseudo sampler holds the (re-computed-each-round) bootstrapped pairs
        self.pseudo_sampler = None
        self.n_pseudo = 0

        opt = cfg.train.optimizer.lower()
        params = model.parameters()
        if opt == "adam":
            self.optimizer = torch.optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        self.scheduler = None
        if str(cfg.train.get("lr_schedule", "none")).lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.02)

        # KG1 / KG2 entity id tensors (for hard-negative candidate pools)
        self.kg1_ents = torch.from_numpy(data.kg1_ents).to(self.device)
        self.kg2_ents = torch.from_numpy(data.kg2_ents).to(self.device)

        # test tensors (gold order preserved so column i is the gold of row i)
        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).to(self.device)

        # history
        self.loss_hist = []          # list of dicts
        self.metric_hist = []        # list of dicts
        self.best_mrr = -1.0
        self.best_epoch = -1
        self.no_improve = 0

    # ------------------------------------------------------------------ #
    #  One training epoch (joint TransE + alignment loss)
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int):
        self.model.train()
        cfg = self.cfg.train
        # cycle alignment batches alongside the (usually more numerous) triple batches
        align_batches = list(self.align_sampler.batches(cfg.align_batch_size))
        n_align = len(align_batches)
        pseudo_batches = (list(self.pseudo_sampler.batches(cfg.align_batch_size))
                          if self.pseudo_sampler is not None else [])
        n_pseudo = len(pseudo_batches)
        pw = self.cfg.train.bootstrap.pseudo_weight

        tot = {"kge": 0.0, "align": 0.0, "pseudo": 0.0, "loss": 0.0}
        n_steps = 0
        pbar = tqdm(
            self.triple_sampler.batches(cfg.batch_size),
            total=(len(self.triple_sampler) + cfg.batch_size - 1) // cfg.batch_size,
            desc=f"epoch {epoch:>4}", leave=False, ncols=100,
        )
        for i, (pos, neg, _kg) in enumerate(pbar):
            self.optimizer.zero_grad()

            pos_s = self.model.transe_score(pos)
            neg_s = self.model.transe_score(neg)
            kge = margin_ranking_loss(pos_s, neg_s, cfg.margin_kge)

            _pos, (e1, e2, neg_l, neg_r) = align_batches[i % n_align]
            align = alignment_loss(self.model, e1, e2, neg_l, neg_r,
                                   cfg.align_pos_margin, cfg.align_neg_margin, cfg.align_neg_weight)

            loss = kge + cfg.align_loss_weight * align

            pseudo_val = 0.0
            if n_pseudo:
                _p, (pe1, pe2, pnl, pnr) = pseudo_batches[i % n_pseudo]
                pseudo = alignment_loss(self.model, pe1, pe2, pnl, pnr,
                                        cfg.align_pos_margin, cfg.align_neg_margin, cfg.align_neg_weight)
                loss = loss + pw * pseudo
                pseudo_val = pseudo.item()

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            tot["kge"] += kge.item()
            tot["align"] += align.item()
            tot["pseudo"] += pseudo_val
            tot["loss"] += loss.item()
            n_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", kge=f"{kge.item():.3f}",
                             align=f"{align.item():.3f}", pseudo=f"{pseudo_val:.3f}")

        return {k: v / max(1, n_steps) for k, v in tot.items()}

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ec = self.cfg.eval
        zl = self.model.encode_all(self.test_left, chunk=self.cfg.model.get("encode_chunk", 4096))
        zr = self.model.encode_all(self.test_right, chunk=self.cfg.model.get("encode_chunk", 4096))
        res = M.evaluate_alignment(
            zl, zr,
            hits_at=tuple(ec.hits_at), metric=ec.metric, csls_k=ec.csls_k,
            chunk=ec.eval_chunk, direction=ec.direction,
        )
        return res

    # ------------------------------------------------------------------ #
    #  Bootstrapping (self-training)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def bootstrap(self):
        """RE-LABEL the unlabelled pool with confident pseudo-pairs (BootEA-style).

        The pseudo-label set is recomputed from scratch every round (it is
        not accumulated), so mistakes made early, when the model was weak,
        are dropped or revised as the model improves. This is the key difference
        from the naive variant that collapses.

        Procedure (gold labels are never read):
          1. encode the unlabelled test-pool entities of KG1 and KG2;
          2. score with CSLS (reduces hubness for higher matching precision);
          3. keep only mutual nearest neighbours;
          4. resolve to a one-to-one matching greedily by confidence (MWGM);
          5. accept a pair only if its raw cosine is at least ``threshold``.

        Returns ``(n_pseudo, accepted)``.
        """
        bs = self.cfg.train.bootstrap
        self.model.eval()
        left, right = self.test_left, self.test_right
        zl = torch.nn.functional.normalize(self.model.encode_all(left), dim=-1)
        zr = torch.nn.functional.normalize(self.model.encode_all(right), dim=-1)
        cos = zl @ zr.t()                                        # (n, n) cosine

        # CSLS adjustment for the matching (better precision than raw cosine)
        k = self.cfg.eval.csls_k
        r_t = cos.topk(min(k, cos.shape[0]), dim=0).values.mean(0)   # per target (col)
        r_s = cos.topk(min(k, cos.shape[1]), dim=1).values.mean(1)   # per source (row)
        csls = 2 * cos - r_t.unsqueeze(0) - r_s.unsqueeze(1)

        idx = torch.arange(len(left), device=self.device)
        best_r = csls.argmax(1)                                  # best right for each left
        best_l = csls.argmax(0)                                  # best left for each right
        mutual = best_l[best_r] == idx                           # mutual nearest neighbours
        cos_conf = cos[idx, best_r]                              # raw cosine of the candidate
        keep = mutual & (cos_conf >= bs.threshold)

        cand = idx[keep]
        if len(cand) == 0:
            self.pseudo_sampler = None
            self.n_pseudo = 0
            return 0, 0

        # greedy one-to-one (MWGM): take highest-confidence pairs, each side once
        order = torch.argsort(cos_conf[cand], descending=True)
        cand = cand[order][: bs.max_add]
        cand_l = cand.tolist()
        cand_r = best_r[cand].tolist()
        used_r, pairs = set(), []
        for li, ri in zip(cand_l, cand_r):
            if ri in used_r:
                continue
            used_r.add(ri)
            pairs.append((int(left[li]), int(right[ri])))

        pairs = np.array(pairs, dtype=np.int64)
        if self.pseudo_sampler is None:
            self.pseudo_sampler = AlignSampler(self.data, self.device, neg=self.cfg.train.neg_samples)
        self.pseudo_sampler.set_pairs(pairs)               # REPLACE (no accumulation)
        self.n_pseudo = len(pairs)
        return self.n_pseudo, len(pairs)

    # ------------------------------------------------------------------ #
    #  Hard (nearest-neighbour) negatives for the seed alignment loss
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_hard_negatives(self):
        """Recompute each seed pair's nearest cross-KG candidates (epsilon-truncated).

        For every seed pair ``(e1, e2)`` we store the ``C`` nearest KG2 entities
        to ``e1`` (to corrupt the right) and the ``C`` nearest KG1 entities to
        ``e2`` (to corrupt the left). Negatives sampled from these confusable
        pools make the margin loss directly separate hard cases for better Hit@1.
        """
        C = self.cfg.train.hard_negatives.num_candidates
        self.model.eval()
        pairs = self.align_sampler.pairs                         # (S,2)
        left, right = pairs[:, 0], pairs[:, 1]

        z_left = torch.nn.functional.normalize(self.model.encode_all(left), dim=-1)
        z_right = torch.nn.functional.normalize(self.model.encode_all(right), dim=-1)
        z_kg1 = torch.nn.functional.normalize(self.model.encode_all(self.kg1_ents), dim=-1)
        z_kg2 = torch.nn.functional.normalize(self.model.encode_all(self.kg2_ents), dim=-1)

        def nearest(zq, zpool, pool_ids):
            out = torch.empty((zq.shape[0], C), dtype=torch.long, device=self.device)
            for s in range(0, zq.shape[0], 1024):
                sim = zq[s:s + 1024] @ zpool.t()
                top = sim.topk(C, dim=1).indices
                out[s:s + 1024] = pool_ids[top]
            return out

        hard_r = nearest(z_left, z_kg2, self.kg2_ents)           # nearest KG2 to each e1
        hard_l = nearest(z_right, z_kg1, self.kg1_ents)          # nearest KG1 to each e2
        self.align_sampler.set_hard_negatives(hard_r, hard_l)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, name: str, epoch: int, res=None):
        path = self.run_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "config": self.cfg.to_plain(),
                "metrics": res,
            },
            path,
        )
        return path

    def save_embeddings(self):
        path = self.run_dir / self.cfg.logging.embeddings_name
        torch.save(
            {
                "ent_emb": self.model.ent_emb.weight.detach().cpu(),
                "rel_emb": self.model.rel_emb.weight.detach().cpu(),
            },
            path,
        )
        return path

    def _append_csv(self, name: str, row: dict, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #
    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax)
            fig.tight_layout()
            fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve)
            plt.close(fig)

        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout()
            fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve)
            plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def fit(self):
        cfg = self.cfg
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.data.num_entities} "
                      f"relations={self.data.num_relations} triples={len(self.triple_sampler)} "
                      f"seed_pairs={len(self.align_sampler)} test_pairs={len(self.test_left)}")
        boot = cfg.train.bootstrap
        t0 = time.time()

        for epoch in range(1, cfg.train.epochs + 1):
            losses = self.train_epoch(epoch)
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss", "kge", "align", "pseudo"])

            msg = (f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f} "
                   f"(kge={losses['kge']:.4f} align={losses['align']:.4f} pseudo={losses['pseudo']:.4f})")

            # hard-negative refresh (epsilon-truncated nearest-neighbour sampling)
            hn = cfg.train.get("hard_negatives", None)
            if hn and hn.enabled and epoch >= hn.start_epoch and (epoch - hn.start_epoch) % hn.refresh_every == 0:
                self.refresh_hard_negatives()
                msg += " | hard-neg refreshed"

            # bootstrapping: re-label the unlabelled pool from scratch
            if boot.enabled and epoch >= boot.start_epoch and (epoch - boot.start_epoch) % boot.every == 0:
                n_pseudo, accepted = self.bootstrap()
                msg += f" | bootstrap: {accepted} pseudo-pairs (beta={boot.pseudo_weight})"

            if self.scheduler is not None:
                self.scheduler.step()

            self.log.info(msg)

            # evaluation
            if epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs:
                res = self.evaluate()
                self.log.info("           " + M.format_metrics(res))
                ref = res.get("avg", res.get("l2r"))
                row = {"epoch": epoch, **{k: ref[k] for k in ref}}
                self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                         **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
                self._append_csv(cfg.logging.metrics_csv, row)
                self.plot_curves()

                if ref["MRR"] > self.best_mrr:
                    self.best_mrr = ref["MRR"]; self.best_epoch = epoch
                    self.no_improve = 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1

                # early stopping: avoids the long degradation tail and saves compute
                patience = cfg.train.get("early_stop_patience", 0)
                if patience and self.no_improve >= patience:
                    self.log.info(f"           early stop: no MRR improvement for "
                                  f"{patience} evals (best={self.best_mrr:.4f} @ {self.best_epoch}).")
                    break

        # final artefacts
        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.save_embeddings()
        self.plot_curves()
        dt = time.time() - t0
        self.log.info(f"Done in {dt/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


# =========================================================================== #
#  BootEA trainer
# =========================================================================== #
class BootEATrainer:
    """BootEA training loop (Sun et al., IJCAI 2018).

    Components:
      * **AlignE embedding** : limit-based TransE loss over a (mutable) triple set
        with epsilon-truncated negative sampling (nearest same-KG neighbours).
      * Alignment by swapping: seed-aligned pairs generate aligned triples
        (added once to the triple set) that calibrate the two embedding spaces.
      * Contrastive alignment loss: a limit-based, hard-negative contrastive
        objective on the labelled pairs (seed plus bootstrapped). This is the
        reliable driver of Hit@1 and corresponds to BootEA's alignment likelihood.
      * Editable MWGM bootstrapping: recomputed mutual 1-to-1 matching each
        round feeds the pseudo alignment term (down-weighted, never accumulated).

    Note: swapping is restricted to the gold seed pairs (built once) so that
    erroneous bootstrapped pairs never corrupt the triple set, which is the cause
    of the collapse seen with naive swap-on-pseudo-labels.
    """

    def __init__(self, cfg, data: DBP15K, model: BootEA, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.kg_of_ent = kg_of_entity(data)
        self.kg1_ents = torch.from_numpy(data.kg1_ents).to(self.device)
        self.kg2_ents = torch.from_numpy(data.kg2_ents).to(self.device)

        # triple sampler over base + seed-swapped triples (epsilon-truncated negatives)
        self.triple_sampler = DynamicTripleSampler(
            self.device, self.kg_of_ent, data.kg1_ents, data.kg2_ents, neg=cfg.train.neg_samples)
        self.swapper = Swapper(data)
        self.base_triples = np.concatenate([data.triples1, data.triples2], axis=0)
        self.labeled = data.train_pairs.copy()      # grows via bootstrapping (if swap_pseudo)
        self._rebuild_triples(self.labeled)

        # alignment: seed sampler (with hard negatives) + recomputed pseudo sampler
        self.align_sampler = AlignSampler(data, self.device, neg=cfg.train.neg_samples)
        self.pseudo_sampler = None
        self.n_pseudo = 0

        params = model.parameters()
        if cfg.train.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = None
        if str(cfg.train.get("lr_schedule", "none")).lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.02)

        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).to(self.device)

        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0

    # ------------------------------------------------------------------ #
    #  Triple set : base + aligned (swapped) triples from labelled pairs
    # ------------------------------------------------------------------ #
    def _rebuild_triples(self, pairs):
        """Set the triple set to base + swapped triples generated from ``pairs``.

        With ``swapping.swap_pseudo`` the labelled set grows via bootstrapping, so
        confident bootstrapped pairs also calibrate the embeddings (BootEA's core
        idea). Recomputed from scratch each round (editable) so wrong pairs are
        dropped as the matching is revised.
        """
        triples = self.base_triples
        if self.cfg.train.swapping.enabled and len(pairs):
            swapped = self.swapper.generate(pairs, cap_per_role=self.cfg.train.swapping.cap_per_role)
            if len(swapped):
                triples = np.concatenate([triples, swapped], axis=0)
        self.triple_sampler.set_triples(triples)
        return len(triples)

    # ------------------------------------------------------------------ #
    #  epsilon-truncated triple negatives (nearest same-KG neighbours)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_truncated_candidates(self):
        C = self.cfg.train.eps_truncated.num_candidates
        self.model.eval()
        N = self.data.num_entities
        cand = torch.zeros((N, C), dtype=torch.long, device=self.device)
        for ids in (self.kg1_ents, self.kg2_ents):
            z = torch.nn.functional.normalize(self.model.encode_all(ids), dim=-1)
            for s in range(0, len(ids), 1024):
                sim = z[s:s + 1024] @ z.t()
                top = sim.topk(C + 1, dim=1).indices[:, 1:]      # drop self
                cand[ids[s:s + 1024]] = ids[top]
        self.triple_sampler.set_candidates(cand)

    # ------------------------------------------------------------------ #
    #  Hard (nearest cross-KG) negatives for the alignment loss
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_hard_negatives(self):
        C = self.cfg.train.hard_negatives.num_candidates
        self.model.eval()
        pairs = self.align_sampler.pairs
        left, right = pairs[:, 0], pairs[:, 1]
        z_left = torch.nn.functional.normalize(self.model.encode_all(left), dim=-1)
        z_right = torch.nn.functional.normalize(self.model.encode_all(right), dim=-1)
        z_kg1 = torch.nn.functional.normalize(self.model.encode_all(self.kg1_ents), dim=-1)
        z_kg2 = torch.nn.functional.normalize(self.model.encode_all(self.kg2_ents), dim=-1)

        def nearest(zq, zpool, pool_ids):
            out = torch.empty((zq.shape[0], C), dtype=torch.long, device=self.device)
            for s in range(0, zq.shape[0], 1024):
                top = (zq[s:s + 1024] @ zpool.t()).topk(C, dim=1).indices
                out[s:s + 1024] = pool_ids[top]
            return out

        hard_r = nearest(z_left, z_kg2, self.kg2_ents)
        hard_l = nearest(z_right, z_kg1, self.kg1_ents)
        self.align_sampler.set_hard_negatives(hard_r, hard_l)

    # ------------------------------------------------------------------ #
    #  One training epoch
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int):
        self.model.train()
        c = self.cfg.train
        align_batches = list(self.align_sampler.batches(c.align_batch_size))
        n_align = max(1, len(align_batches))
        pseudo_batches = (list(self.pseudo_sampler.batches(c.align_batch_size))
                          if self.pseudo_sampler is not None else [])
        n_pseudo = len(pseudo_batches)
        pw = c.bootstrap.pseudo_weight

        tot = {"loss": 0.0, "kge": 0.0, "align": 0.0, "pseudo": 0.0}
        steps = 0
        pbar = tqdm(self.triple_sampler.batches(c.batch_size),
                    total=(len(self.triple_sampler) + c.batch_size - 1) // c.batch_size,
                    desc=f"epoch {epoch:>4}", leave=False, ncols=100)
        for i, (pos, neg) in enumerate(pbar):
            self.optimizer.zero_grad()
            kge = limit_based_triple_loss(
                self.model.triple_score(pos), self.model.triple_score(neg),
                c.pos_margin_kge, c.neg_margin_kge, c.neg_weight_kge)

            _p, (e1, e2, nl, nr) = align_batches[i % n_align]
            align = alignment_loss(self.model, e1, e2, nl, nr,
                                   c.align_pos_margin, c.align_neg_margin, c.align_neg_weight)
            loss = kge + c.align_loss_weight * align

            pseudo_val = 0.0
            if n_pseudo:
                _q, (pe1, pe2, pnl, pnr) = pseudo_batches[i % n_pseudo]
                pseudo = alignment_loss(self.model, pe1, pe2, pnl, pnr,
                                        c.align_pos_margin, c.align_neg_margin, c.align_neg_weight)
                loss = loss + pw * pseudo
                pseudo_val = pseudo.item()

            loss.backward()
            if c.grad_clip and c.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            self.optimizer.step()

            tot["kge"] += kge.item(); tot["align"] += align.item()
            tot["pseudo"] += pseudo_val; tot["loss"] += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", kge=f"{kge.item():.3f}",
                             align=f"{align.item():.3f}", pseudo=f"{pseudo_val:.3f}")
        return {k: v / max(1, steps) for k, v in tot.items()}

    # ------------------------------------------------------------------ #
    #  Editable MWGM bootstrapping -> pseudo alignment pairs (recomputed)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def bootstrap(self):
        bs = self.cfg.train.bootstrap
        self.model.eval()
        left, right = self.test_left, self.test_right
        zl = torch.nn.functional.normalize(self.model.encode_all(left), dim=-1)
        zr = torch.nn.functional.normalize(self.model.encode_all(right), dim=-1)
        cos = zl @ zr.t()
        k = self.cfg.eval.csls_k
        r_t = cos.topk(min(k, cos.shape[0]), dim=0).values.mean(0)
        r_s = cos.topk(min(k, cos.shape[1]), dim=1).values.mean(1)
        csls = 2 * cos - r_t.unsqueeze(0) - r_s.unsqueeze(1)

        idx = torch.arange(len(left), device=self.device)
        best_r = csls.argmax(1)
        best_l = csls.argmax(0)
        mutual = best_l[best_r] == idx
        conf = cos[idx, best_r]
        keep = mutual & (conf >= bs.threshold)
        cand = idx[keep]
        if len(cand) == 0:
            self.pseudo_sampler = None; self.n_pseudo = 0
            return 0
        order = torch.argsort(conf[cand], descending=True)
        cand = cand[order][: bs.max_add]
        used_r, pairs = set(), []
        for li, ri in zip(cand.tolist(), best_r[cand].tolist()):
            if ri in used_r:
                continue
            used_r.add(ri)
            pairs.append((int(left[li]), int(right[ri])))
        pairs = np.array(pairs, dtype=np.int64)
        if self.pseudo_sampler is None:
            self.pseudo_sampler = AlignSampler(self.data, self.device, neg=self.cfg.train.neg_samples)
        self.pseudo_sampler.set_pairs(pairs)              # REPLACE (editable, no accumulation)
        self.n_pseudo = len(pairs)

        # optionally feed confident pairs to the embedding via swapping (BootEA core)
        if self.cfg.train.swapping.get("swap_pseudo", False):
            seed = {int(a): int(b) for a, b in self.data.train_pairs}
            for a, b in pairs:
                seed.setdefault(int(a), int(b))
            self.labeled = np.array(list(seed.items()), dtype=np.int64)
            self._rebuild_triples(self.labeled)
        return len(pairs)

    # ------------------------------------------------------------------ #
    #  Evaluation / persistence / plots
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ec = self.cfg.eval
        zl = self.model.encode_all(self.test_left)
        zr = self.model.encode_all(self.test_right)
        return M.evaluate_alignment(zl, zr, hits_at=tuple(ec.hits_at), metric=ec.metric,
                                    csls_k=ec.csls_k, chunk=ec.eval_chunk, direction=ec.direction)

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        torch.save({"ent_emb": self.model.ent_emb.weight.detach().cpu(),
                    "rel_emb": self.model.rel_emb.weight.detach().cpu()},
                   self.run_dir / self.cfg.logging.embeddings_name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss", "kge", "align", "pseudo"))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def fit(self):
        cfg = self.cfg
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.data.num_entities} "
                      f"relations={self.data.num_relations} base_triples={len(self.base_triples)} "
                      f"triples(+seed-swap)={len(self.triple_sampler)} "
                      f"seed_pairs={len(self.data.train_pairs)} test_pairs={len(self.test_left)}")
        boot, eps, hn = cfg.train.bootstrap, cfg.train.eps_truncated, cfg.train.hard_negatives
        t0 = time.time()

        for epoch in range(1, cfg.train.epochs + 1):
            losses = self.train_epoch(epoch)
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss", "kge", "align", "pseudo"])
            msg = (f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f} "
                   f"(kge={losses['kge']:.4f} align={losses['align']:.4f} pseudo={losses['pseudo']:.4f})")

            if hn.enabled and epoch >= hn.start_epoch and (epoch - hn.start_epoch) % hn.refresh_every == 0:
                self.refresh_hard_negatives()
                msg += " | hard-neg refreshed"
            if eps.enabled and epoch >= eps.start_epoch and (epoch - eps.start_epoch) % eps.refresh_every == 0:
                self.refresh_truncated_candidates()
                msg += " | eps-trunc refreshed"
            if boot.enabled and epoch >= boot.start_epoch and (epoch - boot.start_epoch) % boot.every == 0:
                added = self.bootstrap()
                msg += f" | bootstrap: {added} pseudo-pairs (beta={boot.pseudo_weight})"

            if self.scheduler is not None:
                self.scheduler.step()
            self.log.info(msg)

            if epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs:
                res = self.evaluate()
                self.log.info("           " + M.format_metrics(res))
                ref = res.get("avg", res.get("l2r"))
                self._append_csv(cfg.logging.metrics_csv, {"epoch": epoch, **{k: ref[k] for k in ref}})
                self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                         **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
                self.plot_curves()
                if ref["MRR"] > self.best_mrr:
                    self.best_mrr, self.best_epoch, self.no_improve = ref["MRR"], epoch, 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1
                patience = cfg.train.get("early_stop_patience", 0)
                if patience and self.no_improve >= patience:
                    self.log.info(f"           early stop: no MRR improvement for {patience} evals "
                                  f"(best={self.best_mrr:.4f} @ {self.best_epoch}).")
                    break

        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.save_embeddings()
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


# =========================================================================== #
#  AliNet trainer (full-batch gated multi-hop GNN)
# =========================================================================== #
class AliNetTrainer:
    """AliNet training loop (Sun et al., AAAI 2020).

    AliNet is a **full-batch** GNN: every epoch the gated multi-hop network
    encodes *all* entities once (``model.forward_all()``), then a limit-based
    contrastive alignment loss is applied to all seed pairs with epsilon-truncated
    (nearest cross-KG) negatives. No TransE triples, no swapping; bootstrapping
    is optional (off by default, matching the base paper).
    """

    def __init__(self, cfg, data: DBP15K, model: AliNet, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.kg1_ents = torch.from_numpy(data.kg1_ents).to(self.device)
        self.kg2_ents = torch.from_numpy(data.kg2_ents).to(self.device)
        self.seed_l = torch.from_numpy(data.train_pairs[:, 0]).to(self.device)
        self.seed_r = torch.from_numpy(data.train_pairs[:, 1]).to(self.device)
        self.hard_r = None          # (S, C) nearest KG2 to each seed_l
        self.hard_l = None          # (S, C) nearest KG1 to each seed_r
        self.pseudo = None          # optional bootstrapped pairs (S2, 2)

        # triples + per-KG entity pools for the relation-aware (TransE) loss
        self.kg_of_ent = torch.as_tensor(kg_of_entity(data), device=self.device)
        self.triples = torch.from_numpy(np.concatenate([data.triples1, data.triples2], 0)).to(self.device)

        params = model.parameters()
        if cfg.train.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = None
        if str(cfg.train.get("lr_schedule", "none")).lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.02)

        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).to(self.device)

        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0

    # ------------------------------------------------------------------ #
    #  epsilon-truncated nearest cross-KG negatives for the seed (and pseudo) pairs
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_hard_negatives(self):
        C = self.cfg.train.hard_negatives.num_candidates
        self.model.eval()
        z = self.model.forward_all()
        left = self.seed_l if self.pseudo is None else torch.cat([self.seed_l, self.pseudo[:, 0]])
        right = self.seed_r if self.pseudo is None else torch.cat([self.seed_r, self.pseudo[:, 1]])
        z_kg2 = z[self.kg2_ents]
        z_kg1 = z[self.kg1_ents]

        def nearest(zq, zpool, pool_ids):
            out = torch.empty((zq.shape[0], C), dtype=torch.long, device=self.device)
            for s in range(0, zq.shape[0], 2048):
                top = (zq[s:s + 2048] @ zpool.t()).topk(C, dim=1).indices
                out[s:s + 2048] = pool_ids[top]
            return out

        self.hard_r = nearest(z[left], z_kg2, self.kg2_ents)
        self.hard_l = nearest(z[right], z_kg1, self.kg1_ents)

    def _sample_neg(self, table, gold, pool):
        """One random candidate per row of ``table`` (avoid the gold target)."""
        sel = torch.randint(table.shape[1], (table.shape[0], 1), device=self.device)
        out = torch.gather(table, 1, sel).squeeze(1)
        clash = out == gold
        if clash.any():
            out[clash] = pool[torch.randint(len(pool), (int(clash.sum()),), device=self.device)]
        return out

    # ------------------------------------------------------------------ #
    #  One full-batch training epoch
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int):
        self.model.train()
        c = self.cfg.train
        z = self.model.forward_all()

        left = self.seed_l if self.pseudo is None else torch.cat([self.seed_l, self.pseudo[:, 0]])
        right = self.seed_r if self.pseudo is None else torch.cat([self.seed_r, self.pseudo[:, 1]])
        n = c.neg_samples
        e1 = left.repeat_interleave(n)
        e2 = right.repeat_interleave(n)
        B = e1.shape[0]
        rand_r = self.kg2_ents[torch.randint(len(self.kg2_ents), (B,), device=self.device)]
        rand_l = self.kg1_ents[torch.randint(len(self.kg1_ents), (B,), device=self.device)]
        if self.hard_r is not None:
            hard_r = self._sample_neg(self.hard_r.repeat_interleave(n, 0), e2, self.kg2_ents)
            hard_l = self._sample_neg(self.hard_l.repeat_interleave(n, 0), e1, self.kg1_ents)
            strat = str(c.hard_negatives.get("strategy", "hard")).lower()
            if strat == "mixed":
                # gently mix random + hard negatives (hard alone tends to scatter the tail)
                ratio = c.hard_negatives.get("hard_ratio", 0.5)
                use_hard = torch.rand(B, device=self.device) < ratio
                neg_r = torch.where(use_hard, hard_r, rand_r)
                neg_l = torch.where(use_hard, hard_l, rand_l)
            else:
                neg_r, neg_l = hard_r, hard_l
        else:
            neg_r, neg_l = rand_r, rand_l

        if str(c.get("loss", "ranking")).lower() == "limit":
            align = alinet_limit_loss(z, e1, e2, neg_l, neg_r,
                                      c.align_pos_margin, c.align_neg_margin, c.get("align_neg_weight", 1.0))
        else:
            align = alinet_align_loss(z, e1, e2, neg_l, neg_r, c.align_margin)

        # relation-aware (TransE) loss on a sampled batch of triples -> anchors all entities
        rel_val = 0.0
        rcfg = c.get("relation", None)
        if rcfg and rcfg.enabled and self.model.rel_emb is not None:
            M_ = self.triples.shape[0]
            bs = min(rcfg.batch_size, M_)
            idx = torch.randint(M_, (bs,), device=self.device)
            pos = self.triples[idx]
            # corrupt the tail with a random entity from the SAME KG as the true tail
            kg_t = self.kg_of_ent[pos[:, 2]]
            rt2 = self.kg2_ents[torch.randint(len(self.kg2_ents), (bs,), device=self.device)]
            rt1 = self.kg1_ents[torch.randint(len(self.kg1_ents), (bs,), device=self.device)]
            neg_t = torch.where(kg_t == 0, rt1, rt2)
            rel = alinet_relation_loss(z, self.model.rel_emb, pos, neg_t, rcfg.margin)
            rel_val = rel.item()
            loss = align + rcfg.weight * rel
        else:
            loss = align

        self.optimizer.zero_grad()
        loss.backward()
        if c.grad_clip and c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
        self.optimizer.step()
        return {"loss": loss.item(), "align": align.item(), "rel": rel_val}

    # ------------------------------------------------------------------ #
    #  Optional bootstrapping (editable MWGM) -> extra labelled pairs
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def bootstrap(self):
        bs = self.cfg.train.bootstrap
        self.model.eval()
        z = self.model.forward_all()
        zl = z[self.test_left]; zr = z[self.test_right]
        cos = zl @ zr.t()
        k = self.cfg.eval.csls_k
        r_t = cos.topk(min(k, cos.shape[0]), dim=0).values.mean(0)
        r_s = cos.topk(min(k, cos.shape[1]), dim=1).values.mean(1)
        csls = 2 * cos - r_t.unsqueeze(0) - r_s.unsqueeze(1)
        idx = torch.arange(len(self.test_left), device=self.device)
        best_r = csls.argmax(1); best_l = csls.argmax(0)
        keep = (best_l[best_r] == idx) & (cos[idx, best_r] >= bs.threshold)
        cand = idx[keep]
        if len(cand) == 0:
            self.pseudo = None
            return 0
        order = torch.argsort(cos[cand, best_r[cand]], descending=True)
        cand = cand[order][: bs.max_add]
        used, pairs = set(), []
        for li, ri in zip(cand.tolist(), best_r[cand].tolist()):
            if ri in used:
                continue
            used.add(ri); pairs.append((int(self.test_left[li]), int(self.test_right[ri])))
        self.pseudo = torch.tensor(pairs, dtype=torch.long, device=self.device)
        return len(pairs)

    # ------------------------------------------------------------------ #
    #  Evaluation / persistence / plots
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ec = self.cfg.eval
        z = self.model.forward_all()
        return M.evaluate_alignment(z[self.test_left], z[self.test_right],
                                    hits_at=tuple(ec.hits_at), metric=ec.metric,
                                    csls_k=ec.csls_k, chunk=ec.eval_chunk, direction=ec.direction)

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        with torch.no_grad():
            z = self.model.forward_all().detach().cpu()
        torch.save({"entity_repr": z, "ent_emb": self.model.ent_emb.weight.detach().cpu()},
                   self.run_dir / self.cfg.logging.embeddings_name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss",))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def fit(self):
        cfg = self.cfg
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.data.num_entities} "
                      f"2hop_edges={self.model.e_dst.numel()} rep_dim={self.model.out_dim} "
                      f"seed_pairs={len(self.seed_l)} test_pairs={len(self.test_left)}")
        boot, hn = cfg.train.bootstrap, cfg.train.hard_negatives
        t0 = time.time()

        for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="AliNet", ncols=100):
            losses = self.train_epoch(epoch)
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss", "align", "rel"])
            msg = f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f}"

            if hn.enabled and epoch >= hn.start_epoch and (epoch - hn.start_epoch) % hn.refresh_every == 0:
                self.refresh_hard_negatives()
                msg += " | hard-neg refreshed"
            if boot.enabled and epoch >= boot.start_epoch and (epoch - boot.start_epoch) % boot.every == 0:
                added = self.bootstrap()
                msg += f" | bootstrap: {added} pairs"
            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs:
                self.log.info(msg)
                res = self.evaluate()
                self.log.info("           " + M.format_metrics(res))
                ref = res.get("avg", res.get("l2r"))
                self._append_csv(cfg.logging.metrics_csv, {"epoch": epoch, **{k: ref[k] for k in ref}})
                self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                         **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
                self.plot_curves()
                if ref["MRR"] > self.best_mrr:
                    self.best_mrr, self.best_epoch, self.no_improve = ref["MRR"], epoch, 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1
                patience = cfg.train.get("early_stop_patience", 0)
                if patience and self.no_improve >= patience:
                    self.log.info(f"           early stop: no MRR improvement for {patience} evals "
                                  f"(best={self.best_mrr:.4f} @ {self.best_epoch}).")
                    break

        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.save_embeddings()
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


# =========================================================================== #
#  KECG trainer (shared GAT cross-graph + TransE knowledge embedding)
# =========================================================================== #
class KECGTrainer:
    """KECG training loop (Li et al., EMNLP 2019).

    Alternates two objectives that share the entity embeddings:
      * even epochs - **Cross-Graph** triplet margin loss on the GAT output, with
        **NNS** (nearest-neighbour) hard negatives refreshed every ``update_num`` epochs;
      * odd epochs  - **Knowledge-Embedding** TransE margin loss on the triples.
    Evaluation uses the GAT output embeddings.
    """

    def __init__(self, cfg, data: DBP15K, model: KECG, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.kg1_ents = torch.from_numpy(data.kg1_ents).to(self.device)
        self.kg2_ents = torch.from_numpy(data.kg2_ents).to(self.device)
        self.kg_of_ent = torch.as_tensor(kg_of_entity(data), device=self.device)
        self.seed_l = torch.from_numpy(data.train_pairs[:, 0]).to(self.device)
        self.seed_r = torch.from_numpy(data.train_pairs[:, 1]).to(self.device)
        self.triples = torch.from_numpy(np.concatenate([data.triples1, data.triples2], 0)).to(self.device)
        self.hard_r = None          # (S, k_CG) NNS negatives for seed_l (in KG2)
        self.hard_l = None          # (S, k_CG) NNS negatives for seed_r (in KG1)

        opt = cfg.train.optimizer.lower()
        if opt == "adagrad":
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif opt == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).to(self.device)
        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0

    # ------------------------------------------------------------------ #
    #  NNS : nearest cross-KG negatives for the cross-graph triplet loss
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_nns(self):
        """KECG NNS: for each seed, the K nearest OTHER seed entities of the SAME KG.

        Negatives come from the (small, clean) seed set via within-KG proximity
        (right-to-right / left-to-left), excluding self. This separates confusable
        seed targets without scattering the whole cross-KG geometry (drawing arbitrary
        opposite-KG nearest neighbours collapses the space and was the bug).
        """
        k = self.cfg.train.k_cg
        self.model.eval()
        z = self.model.forward_all()
        zr, zl = z[self.seed_r], z[self.seed_l]                 # (S, d)
        idx_r = torch.cdist(zr, zr).topk(k + 1, largest=False).indices[:, 1:]   # drop self (dist 0)
        idx_l = torch.cdist(zl, zl).topk(k + 1, largest=False).indices[:, 1:]
        self.hard_r = self.seed_r[idx_r]                        # (S, k) KG2 negs near each gold-right
        self.hard_l = self.seed_l[idx_l]                        # (S, k) KG1 negs near each gold-left

    # ------------------------------------------------------------------ #
    #  One epoch (alternating CG / KE)
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int):
        self.model.train()
        c = self.cfg.train
        if epoch % 2 == 0:                                   # ---- Cross-Graph (GAT alignment)
            z = self.model.forward_all()
            n = c.k_cg
            e1 = self.seed_l.repeat_interleave(n); e2 = self.seed_r.repeat_interleave(n)
            if self.hard_r is not None:                          # KECG: all K NNS negatives per seed
                neg_r = self.hard_r.reshape(-1)                  # KG2 negs (near each gold-right)
                neg_l = self.hard_l.reshape(-1)                  # KG1 negs (near each gold-left)
            else:
                neg_r = self.kg2_ents[torch.randint(len(self.kg2_ents), (e1.shape[0],), device=self.device)]
                neg_l = self.kg1_ents[torch.randint(len(self.kg1_ents), (e1.shape[0],), device=self.device)]
            loss = kecg_cg_loss(z, e1, e2, neg_l, neg_r, c.margin_cg)
            tag = "CG"
        else:                                                # ---- Knowledge Embedding (TransE)
            bs = min(c.get("ke_batch_size", 20000), self.triples.shape[0])
            idx = torch.randint(self.triples.shape[0], (bs,), device=self.device)
            pos = self.triples[idx]
            k = c.k_ke
            pos_r = pos.repeat(k, 1); neg = pos_r.clone()
            corrupt_head = torch.rand(pos_r.shape[0], device=self.device) < 0.5
            kg_h = self.kg_of_ent[pos_r[:, 0]]; kg_t = self.kg_of_ent[pos_r[:, 2]]
            rh = torch.where(kg_h == 0, self.kg1_ents[torch.randint(len(self.kg1_ents), (pos_r.shape[0],), device=self.device)],
                             self.kg2_ents[torch.randint(len(self.kg2_ents), (pos_r.shape[0],), device=self.device)])
            rt = torch.where(kg_t == 0, self.kg1_ents[torch.randint(len(self.kg1_ents), (pos_r.shape[0],), device=self.device)],
                             self.kg2_ents[torch.randint(len(self.kg2_ents), (pos_r.shape[0],), device=self.device)])
            neg[:, 0] = torch.where(corrupt_head, rh, neg[:, 0])
            neg[:, 2] = torch.where(~corrupt_head, rt, neg[:, 2])
            z = self.model.forward_all()
            loss = kecg_ke_loss(z, self.model.rel_emb, pos_r, neg, c.margin_ke)
            tag = "KE"

        self.optimizer.zero_grad(); loss.backward()
        if c.grad_clip and c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
        self.optimizer.step()
        return {"loss": loss.item(), tag: loss.item()}

    # ------------------------------------------------------------------ #
    #  Evaluation / persistence / plots
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ec = self.cfg.eval
        z = self.model.forward_all()
        return M.evaluate_alignment(z[self.test_left], z[self.test_right],
                                    hits_at=tuple(ec.hits_at), metric=ec.metric,
                                    csls_k=ec.csls_k, chunk=ec.eval_chunk, direction=ec.direction)

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        with torch.no_grad():
            z = self.model.forward_all().detach().cpu()
        torch.save({"entity_repr": z, "ent_emb": self.model.ent_emb.weight.detach().cpu(),
                    "rel_emb": self.model.rel_emb.weight.detach().cpu()},
                   self.run_dir / self.cfg.logging.embeddings_name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss",))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def fit(self):
        cfg = self.cfg
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.data.num_entities} "
                      f"edges={self.model.e_dst.numel()} rep_dim={self.model.out_dim} "
                      f"seed_pairs={len(self.seed_l)} test_pairs={len(self.test_left)}")
        t0 = time.time()
        upd = cfg.train.update_num
        use_nns = cfg.train.get("use_nns", True)             # NNS hard negs (decline-prone) vs random

        for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="KECG", ncols=100):
            if use_nns and epoch >= 2 and (epoch % upd == 0):  # refresh NNS hard negatives
                self.refresh_nns()
            losses = self.train_epoch(epoch)
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss", "CG", "KE"])

            if epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs:
                self.log.info(f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f}")
                res = self.evaluate()
                self.log.info("           " + M.format_metrics(res))
                ref = res.get("avg", res.get("l2r"))
                self._append_csv(cfg.logging.metrics_csv, {"epoch": epoch, **{k: ref[k] for k in ref}})
                self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                         **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
                self.plot_curves()
                if ref["MRR"] > self.best_mrr:
                    self.best_mrr, self.best_epoch, self.no_improve = ref["MRR"], epoch, 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1
                patience = cfg.train.get("early_stop_patience", 0)
                if patience and self.no_improve >= patience:
                    self.log.info(f"           early stop: no MRR improvement for {patience} evals "
                                  f"(best={self.best_mrr:.4f} @ {self.best_epoch}).")
                    break

        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.save_embeddings()
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


# =========================================================================== #
#  GCN-Align trainer (shared 2-layer GCN, margin L1 loss, full-batch)
# =========================================================================== #
class GCNAlignTrainer:
    """GCN-Align training loop (Wang et al., EMNLP 2018).

    Full-batch: each epoch the shared GCN encodes all entities once
    (``forward_all()``), then a margin-based L1 alignment loss is applied to the
    seed pairs with negatives sampled uniformly from ALL entities (both sides
    corrupted). Evaluation ranks by L1 distance. Structure channel (SE) only.
    """

    def __init__(self, cfg, data: DBP15K, model: GCNAlign, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.n_ent = data.num_entities
        self.seed_l = torch.from_numpy(data.train_pairs[:, 0]).to(self.device)
        self.seed_r = torch.from_numpy(data.train_pairs[:, 1]).to(self.device)
        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).to(self.device)

        opt = cfg.train.optimizer.lower()
        if opt == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif opt == "adagrad":
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0

    def train_epoch(self, epoch: int):
        self.model.train()
        c = self.cfg.train
        z = self.model.forward_all()
        k = c.k
        e1 = self.seed_l.repeat_interleave(k)
        e2 = self.seed_r.repeat_interleave(k)
        # negatives uniformly from ALL entities (both corruption directions)
        neg_r = torch.randint(self.n_ent, (e1.shape[0],), device=self.device)
        neg_l = torch.randint(self.n_ent, (e1.shape[0],), device=self.device)
        loss = gcnalign_loss(z, e1, e2, neg_l, neg_r, c.margin)
        self.optimizer.zero_grad(); loss.backward()
        if c.grad_clip and c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
        self.optimizer.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ec = self.cfg.eval
        z = self.model.forward_all()
        return M.evaluate_alignment(z[self.test_left], z[self.test_right],
                                    hits_at=tuple(ec.hits_at), metric=ec.metric,
                                    csls_k=ec.csls_k, chunk=ec.eval_chunk, direction=ec.direction)

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        with torch.no_grad():
            z = self.model.forward_all().detach().cpu()
        torch.save({"entity_repr": z, "ent_emb": self.model.ent_emb.weight.detach().cpu()},
                   self.run_dir / self.cfg.logging.embeddings_name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss",))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    def fit(self):
        cfg = self.cfg
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.data.num_entities} "
                      f"rep_dim={self.model.out_dim} seed_pairs={len(self.seed_l)} "
                      f"test_pairs={len(self.test_left)}")
        t0 = time.time()
        for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="GCN-Align", ncols=100):
            losses = self.train_epoch(epoch)
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss"])
            if epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs:
                self.log.info(f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f}")
                res = self.evaluate()
                self.log.info("           " + M.format_metrics(res))
                ref = res.get("avg", res.get("l2r"))
                self._append_csv(cfg.logging.metrics_csv, {"epoch": epoch, **{k: ref[k] for k in ref}})
                self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                         **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
                self.plot_curves()
                if ref["MRR"] > self.best_mrr:
                    self.best_mrr, self.best_epoch, self.no_improve = ref["MRR"], epoch, 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1
                patience = cfg.train.get("early_stop_patience", 0)
                if patience and self.no_improve >= patience:
                    self.log.info(f"           early stop (best={self.best_mrr:.4f} @ {self.best_epoch}).")
                    break
        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.save_embeddings()
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


# =========================================================================== #
#  JAPE trainer (TransE structure embedding on merged-seed graph + attribute AE)
# =========================================================================== #
class JAPETrainer:
    """JAPE training loop (Sun et al., ISWC 2017).

    SE: TransE on the merged graph (aligned seeds share ids => the two KGs are
    bridged) with margin loss + corrupted-triple negatives. AE: a cross-KG,
    TF-IDF-weighted attribute bag whose cosine similarity is combined with the SE
    cosine at alignment time (``sim = beta*SE + (1-beta)*AE``). Reports SE-only and SE+AE.
    """

    def __init__(self, cfg, data: DBP15K, model: JAPE, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.n_ent = data.num_entities
        self.triples = torch.from_numpy(np.concatenate([data.triples1, data.triples2], 0)).to(self.device)
        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).to(self.device)

        opt = cfg.train.optimizer.lower()
        if opt == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif opt == "adagrad":
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        # ---- attribute channel (AE): precompute TF-IDF test-set cosine sim ----
        self.beta = cfg.train.get("beta", 0.9)
        self.ae_sim = None
        if cfg.train.get("use_attributes", True):
            from .data import load_jape_attributes
            bag, A = load_jape_attributes(Path(cfg._project_root) / cfg.data.root,
                                          cfg.data.lang, cfg.data.fold, data.ent_uri)
            bag = bag.to(self.device)
            df = torch.sparse.sum(bag, dim=0).to_dense()                  # (A,)
            idf = torch.log(self.n_ent / (df + 1.0))
            bl = bag.index_select(0, self.test_left).to_dense() * idf     # (T, A) tf-idf
            br = bag.index_select(0, self.test_right).to_dense() * idf
            bl = torch.nn.functional.normalize(bl, dim=-1)
            br = torch.nn.functional.normalize(br, dim=-1)
            self.ae_sim = bl @ br.t()                                     # (T, T) cosine
            self.log.info(f"AE attribute channel: {A} attributes, beta={self.beta}")

        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0

    def train_epoch(self, epoch: int):
        self.model.train()
        c = self.cfg.train
        order = torch.randperm(self.triples.shape[0], device=self.device)
        tot, steps = 0.0, 0
        for s in tqdm(range(0, self.triples.shape[0], c.batch_size),
                      desc=f"epoch {epoch:>4}", leave=False, ncols=100):
            pos = self.triples[order[s:s + c.batch_size]]
            k = c.neg_samples
            pos_r = pos.repeat(k, 1)
            neg = pos_r.clone()
            ch = torch.rand(pos_r.shape[0], device=self.device) < 0.5
            rnd = torch.randint(self.n_ent, (pos_r.shape[0],), device=self.device)
            neg[:, 0] = torch.where(ch, rnd, neg[:, 0])
            neg[:, 2] = torch.where(~ch, rnd, neg[:, 2])
            loss = jape_se_loss(self.model, pos, neg, c.margin)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            tot += loss.item(); steps += 1
        return {"loss": tot / max(1, steps)}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ec = self.cfg.eval
        zl = torch.nn.functional.normalize(self.model.encode_all(self.test_left), dim=-1)
        zr = torch.nn.functional.normalize(self.model.encode_all(self.test_right), dim=-1)
        se = zl @ zr.t()                                                  # SE cosine
        res = {}
        res["SE"] = M._rank_metrics(se, tuple(ec.hits_at), ec.eval_chunk)
        if self.ae_sim is not None:
            # JAPE: fuse SE + AE similarity, then CSLS-normalise (cross-domain
            # similarity local scaling) so neither channel's hubs dominate.
            k = int(ec.get("csls_k", 5))
            comb = self.beta * se + (1 - self.beta) * self.ae_sim
            rs = comb.topk(k, dim=1).values.mean(1)
            cs = comb.topk(k, dim=0).values.mean(0)
            comb = 2 * comb - rs[:, None] - cs[None, :]
            res["SE+AE"] = M._rank_metrics(comb, tuple(ec.hits_at), ec.eval_chunk)
        return res

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        torch.save({"ent_emb": self.model.ent_emb.weight.detach().cpu(),
                    "rel_emb": self.model.rel_emb.weight.detach().cpu()},
                   self.run_dir / self.cfg.logging.embeddings_name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss",))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    def fit(self):
        cfg = self.cfg
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.data.num_entities} "
                      f"triples={len(self.triples)} test_pairs={len(self.test_left)} "
                      f"(merged-seed graph)")
        t0 = time.time()
        for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="JAPE", ncols=100):
            losses = self.train_epoch(epoch)
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss"])
            if epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs:
                res = self.evaluate()
                self.log.info(f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f}")
                self.log.info("           " + M.format_metrics(res))
                ref = res.get("SE+AE", res["SE"])                          # track SE+AE if available
                self._append_csv(cfg.logging.metrics_csv, {"epoch": epoch, **{k: ref[k] for k in ref}})
                self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                         **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
                self.plot_curves()
                if ref["MRR"] > self.best_mrr:
                    self.best_mrr, self.best_epoch, self.no_improve = ref["MRR"], epoch, 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1
                patience = cfg.train.get("early_stop_patience", 0)
                if patience and self.no_improve >= patience:
                    self.log.info(f"           early stop (best={self.best_mrr:.4f} @ {self.best_epoch}).")
                    break
        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.save_embeddings()
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


class DGMCTrainer:
    """Deep Graph Matching Consensus training loop (Fey et al., ICLR 2020).

    Single graph-pair, full-batch. Two phases following the official example:
    epochs ``1..refine_after`` optimise only the local feature matching
    (``num_steps=0``); afterwards the consensus refinement is switched on and the
    feature GNN is detached. Loss is the sparse NLL of the gold target within the
    top-k candidates; evaluation reports Hits@1 / Hits@10 on the test matchings.
    """

    def __init__(self, cfg, data, model: DGMC, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device
        dev = self.device

        from .data import build_dgmc_adj
        self.x1 = data.x1.to(dev)
        self.x2 = data.x2.to(dev)
        if cfg.model.get("normalize_features", True):       # unit-norm summed-name features
            self.x1 = torch.nn.functional.normalize(self.x1, dim=1)
            self.x2 = torch.nn.functional.normalize(self.x2, dim=1)
        self.ai1, self.ao1 = (a.to(dev) for a in build_dgmc_adj(data.edge_index1, data.x1.size(0)))
        self.ai2, self.ao2 = (a.to(dev) for a in build_dgmc_adj(data.edge_index2, data.x2.size(0)))
        self.train_y = data.train_y.to(dev)
        self.test_y = data.test_y.to(dev)

        opt = cfg.train.optimizer.lower()
        Opt = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad,
               "sgd": torch.optim.SGD}.get(opt, torch.optim.Adam)
        self.optimizer = Opt(model.parameters(), lr=cfg.train.lr,
                             weight_decay=cfg.train.get("weight_decay", 0.0))

        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0   # "mrr" == Hits@1 here

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        S_idx, _, S_L = self.model(self.x1, self.ai1, self.ao1,
                                   self.x2, self.ai2, self.ao2, self.train_y)
        loss = self.model.loss(S_idx, S_L, self.train_y)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        S_idx, _, S_L = self.model(self.x1, self.ai1, self.ao1,
                                   self.x2, self.ai2, self.ao2)
        res = {}
        for k in self.cfg.eval.hits_at:
            res[f"Hit@{k}"] = self.model.hits_at_k(int(k), S_idx, S_L, self.test_y)
        res["MRR"] = self.model.mrr(S_idx, S_L, self.test_y)
        return res

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        pass

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss",))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    def fit(self):
        cfg = self.cfg
        refine_after = cfg.train.get("refine_after", 100)
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | src={self.x1.size(0)} tgt={self.x2.size(0)} "
                      f"feat={self.x1.size(1)} train_y={self.train_y.size(1)} "
                      f"test_y={self.test_y.size(1)}")
        self.log.info(f"Phase 1 (1..{refine_after}): local feature matching (num_steps=0).")
        t0 = time.time()
        for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="DGMC", ncols=100):
            if epoch == refine_after + 1:
                self.model.num_steps = int(cfg.model.num_steps)
                self.model.detach = bool(cfg.train.get("detach_refine", True))
                self.log.info(f"Phase 2 (>{refine_after}): consensus refinement "
                              f"(num_steps={self.model.num_steps}, detach psi_1).")
            losses = self.train_epoch()
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss"])

            if epoch % cfg.eval.every == 0 or epoch > refine_after or epoch == cfg.train.epochs:
                res = self.evaluate()
                msg = " ".join(f"{k}={v:.4f}" for k, v in res.items())
                self.log.info(f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f} | {msg}")
                self._append_csv(cfg.logging.metrics_csv, {"epoch": epoch, **res})
                self.metric_hist.append({"epoch": epoch, **res})
                self.plot_curves()
                h1 = res.get("Hit@1", 0.0)
                if h1 > self.best_mrr:
                    self.best_mrr, self.best_epoch, self.no_improve = h1, epoch, 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best Hit@1={h1:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1
        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best Hit@1={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


class MRAEATrainer:
    """MRAEA training loop (Mao et al., WSDM 2020).

    Full-batch graph attention over the whole entity graph. Each epoch samples a
    fresh set of positive pairs (repeated up to ``batch_size``) plus two random
    negatives per pair and takes one L1-margin gradient step. Alignment is read
    out by cosine similarity. Optional bi-directional iterative bootstrapping adds
    high-confidence mutual nearest neighbours to the training set during training.
    """

    def __init__(self, cfg, data: DBP15K, model: MRAEA, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.train_pairs = torch.from_numpy(data.train_pairs).long().to(self.device)
        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).long().to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).long().to(self.device)
        self.batch_size = model.N

        opt = cfg.train.optimizer.lower()
        Opt = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad,
               "sgd": torch.optim.SGD}.get(opt, torch.optim.Adam)
        self.optimizer = Opt(model.parameters(), lr=cfg.train.lr,
                             weight_decay=cfg.train.get("weight_decay", 0.0))
        self.gamma = cfg.train.get("gamma", 3.0)

        # iterative (semi-supervised) bootstrapping state
        self.iterative = bool(cfg.train.get("iterative", False))
        self.added = None                                  # extra (l,r) seeds, as tensor

        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0

    def _current_train(self):
        if self.added is not None and self.added.numel():
            return torch.cat([self.train_pairs, self.added], dim=0)
        return self.train_pairs

    def _make_quad(self, train_pairs):
        n = train_pairs.size(0)
        reps = self.batch_size // max(1, n) + 1
        idx = train_pairs.repeat(reps, 1)
        perm = torch.randperm(idx.size(0), device=self.device)[:self.batch_size]
        pos = idx[perm]                                    # (B, 2)
        neg = torch.randint(self.model.N, pos.shape, device=self.device)  # (B, 2)
        return torch.cat([pos, neg], dim=-1)               # (B, 4) = l,r,neg_l,neg_r

    def train_epoch(self):
        self.model.train()
        quad = self._make_quad(self._current_train())
        emb = self.model()
        loss = mraea_align_loss(emb, quad, self.gamma)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def embed(self):
        self.model.eval()
        return self.model()

    @torch.no_grad()
    def evaluate(self):
        emb = self.embed()
        ec = self.cfg.eval
        return M.evaluate_alignment(emb[self.test_left], emb[self.test_right],
                                    hits_at=tuple(ec.hits_at), metric=ec.metric,
                                    chunk=ec.eval_chunk, direction=ec.direction)

    @torch.no_grad()
    def bootstrap(self, emb):
        """Bi-directional iterative: add mutual nearest neighbours as new seeds.

        Candidates are the *unaligned* entities of each KG (those not already in a
        training pair). A pair (l, r) is accepted when l is r's NN in KG1 and r is
        l's NN in KG2 (mutual), above an optional cosine threshold.
        """
        cur = self._current_train()
        used_l = set(cur[:, 0].tolist()); used_r = set(cur[:, 1].tolist())
        left = torch.tensor([e for e in self.data.kg1_ents if e not in used_l],
                            device=self.device)
        right = torch.tensor([e for e in self.data.kg2_ents if e not in used_r],
                             device=self.device)
        if left.numel() == 0 or right.numel() == 0:
            return
        zl = torch.nn.functional.normalize(emb[left], dim=-1)
        zr = torch.nn.functional.normalize(emb[right], dim=-1)
        sim = zl @ zr.t()                                  # (|L|, |R|)
        nn_lr = sim.argmax(dim=1)                          # best right for each left
        nn_rl = sim.argmax(dim=0)                          # best left for each right
        thr = self.cfg.train.get("iter_threshold", 0.0)
        rows = torch.arange(left.numel(), device=self.device)
        mutual = nn_rl[nn_lr] == rows                      # l -> r -> l
        ok = mutual & (sim[rows, nn_lr] >= thr)
        new = torch.stack([left[rows[ok]], right[nn_lr[ok]]], dim=1)
        if new.numel():
            self.added = new if self.added is None else torch.cat([self.added, new], 0)
            self.log.info(f"           [iter] added {new.size(0)} mutual-NN seeds "
                          f"(total extra {self.added.size(0)})")

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        with torch.no_grad():
            torch.save({"ent_emb": self.embed().detach().cpu()},
                       self.run_dir / self.cfg.logging.embeddings_name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss",))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    def fit(self):
        cfg = self.cfg
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.model.N} "
                      f"rels(x2)={self.model.R} edges={self.model.adj_index.size(1)} "
                      f"train={self.train_pairs.size(0)} test={self.test_left.numel()} "
                      f"iterative={self.iterative}")
        t0 = time.time()
        for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="MRAEA", ncols=100):
            losses = self.train_epoch()
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss"])

            if epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs:
                res = self.evaluate()
                self.log.info(f"epoch {epoch:>4}/{cfg.train.epochs} | loss={losses['loss']:.4f}")
                self.log.info("           " + M.format_metrics(res))
                ref = res.get("avg", res.get("l2r"))
                self._append_csv(cfg.logging.metrics_csv, {"epoch": epoch, **{k: ref[k] for k in ref}})
                self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                         **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
                self.plot_curves()
                if ref["MRR"] > self.best_mrr:
                    self.best_mrr, self.best_epoch, self.no_improve = ref["MRR"], epoch, 0
                    if cfg.logging.save_best:
                        self.save_checkpoint("model_best.pt", epoch, res)
                        self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")
                else:
                    self.no_improve += 1

            if self.iterative and epoch >= cfg.train.get("iter_start", 1000) \
                    and epoch % cfg.train.get("iter_every", 300) == 0:
                self.bootstrap(self.embed())

        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, cfg.train.epochs)
        self.save_embeddings()
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}


class RREATrainer:
    """RREA training loop (Mao et al., CIKM 2020).

    Full-batch relational-reflection graph attention. Training is **turn-based**:
    ``turns`` rounds of ``epoch_per_turn`` L1-margin steps; between rounds a
    bi-directional **CSLS mutual nearest-neighbour** bootstrap adds high-precision
    pseudo-anchors from the still-unaligned test entities (removed from the pool
    once matched). ``turns == 1`` reproduces the *basic* (non-semi) model.
    Alignment is read out with CSLS.
    """

    def __init__(self, cfg, data: DBP15K, model: RREA, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device

        self.train_pairs = torch.from_numpy(data.train_pairs).long().to(self.device)
        self.test_left = torch.from_numpy(data.test_pairs[:, 0]).long().to(self.device)
        self.test_right = torch.from_numpy(data.test_pairs[:, 1]).long().to(self.device)
        self.rest_left = self.test_left.clone()
        self.rest_right = self.test_right.clone()
        self.batch_size = model.N

        opt = cfg.train.optimizer.lower()
        wd = cfg.train.get("weight_decay", 0.0)
        if opt == "rmsprop":
            # match Keras RMSprop (rho=0.9, eps=1e-7), not PyTorch's defaults (alpha=0.99)
            self.optimizer = torch.optim.RMSprop(
                model.parameters(), lr=cfg.train.lr, weight_decay=wd,
                alpha=cfg.train.get("rms_alpha", 0.9), eps=cfg.train.get("rms_eps", 1e-7))
        else:
            Opt = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad,
                   "sgd": torch.optim.SGD}.get(opt, torch.optim.Adam)
            self.optimizer = Opt(model.parameters(), lr=cfg.train.lr, weight_decay=wd)
        self.gamma = cfg.train.get("gamma", 3.0)
        self.csls_k = cfg.train.get("csls_k", 10)

        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch = -1.0, -1

    def _make_quad(self):
        n = self.train_pairs.size(0)
        reps = self.batch_size // max(1, n) + 1
        idx = self.train_pairs.repeat(reps, 1)
        perm = torch.randperm(idx.size(0), device=self.device)[:self.batch_size]
        pos = idx[perm]
        neg = torch.randint(self.model.N, pos.shape, device=self.device)
        return torch.cat([pos, neg], dim=-1)

    def train_epoch(self):
        self.model.train()
        quad = self._make_quad()
        emb = self.model()
        loss = mraea_align_loss(emb, quad, self.gamma)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def embed(self):
        self.model.eval()
        return self.model()

    @torch.no_grad()
    def evaluate(self):
        emb = self.embed()
        ec = self.cfg.eval
        return M.evaluate_alignment(emb[self.test_left], emb[self.test_right],
                                    hits_at=tuple(ec.hits_at), metric=ec.metric,
                                    csls_k=ec.csls_k, chunk=ec.eval_chunk, direction=ec.direction)

    @torch.no_grad()
    def _csls_mutual(self, emb):
        """Return mutual CSLS nearest-neighbour pairs among the unaligned pool."""
        if self.rest_left.numel() == 0 or self.rest_right.numel() == 0:
            return None
        zl = torch.nn.functional.normalize(emb[self.rest_left], dim=-1)
        zr = torch.nn.functional.normalize(emb[self.rest_right], dim=-1)
        sim = zl @ zr.t()                                          # (|L|, |R|)
        k = min(self.csls_k, sim.size(0), sim.size(1))
        rl = sim.topk(k, dim=1).values.mean(1)
        rr = sim.topk(k, dim=0).values.mean(0)
        csls = 2 * sim - rl[:, None] - rr[None, :]
        a = csls.argmax(dim=1)                                     # best right for each left
        b = csls.argmax(dim=0)                                     # best left for each right
        rows = torch.arange(self.rest_left.numel(), device=self.device)
        mutual = b[a] == rows
        return rows[mutual], a[mutual]                             # positions in rest_left/right

    @torch.no_grad()
    def bootstrap(self):
        res = self._csls_mutual(self.embed())
        if res is None:
            return
        li, ri = res
        if li.numel() == 0:
            return
        new = torch.stack([self.rest_left[li], self.rest_right[ri]], dim=1)
        self.train_pairs = torch.cat([self.train_pairs, new], dim=0)
        keep_l = torch.ones(self.rest_left.numel(), dtype=torch.bool, device=self.device)
        keep_l[li] = False
        keep_r = torch.ones(self.rest_right.numel(), dtype=torch.bool, device=self.device)
        keep_r[ri] = False
        self.rest_left = self.rest_left[keep_l]
        self.rest_right = self.rest_right[keep_r]
        self.log.info(f"           [semi] +{new.size(0)} mutual-NN pairs "
                      f"(train={self.train_pairs.size(0)}, pool={self.rest_left.numel()})")

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def save_embeddings(self):
        with torch.no_grad():
            torch.save({"ent_emb": self.embed().detach().cpu()},
                       self.run_dir / self.cfg.logging.embeddings_name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=("loss",))
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / self.cfg.logging.plots.metrics_curve); plt.close(fig)

    def _eval_and_log(self, epoch, total, losses):
        res = self.evaluate()
        self.log.info(f"epoch {epoch:>4}/{total} | loss={losses['loss']:.4f}")
        self.log.info("           " + M.format_metrics(res))
        ref = res.get("avg", res.get("l2r"))
        self._append_csv(self.cfg.logging.metrics_csv, {"epoch": epoch, **{k: ref[k] for k in ref}})
        self.metric_hist.append({"epoch": epoch, "MRR": ref["MRR"],
                                 **{k: v for k, v in ref.items() if k.startswith("Hit@")}})
        self.plot_curves()
        if ref["MRR"] > self.best_mrr:
            self.best_mrr, self.best_epoch = ref["MRR"], epoch
            if self.cfg.logging.save_best:
                self.save_checkpoint("model_best.pt", epoch, res)
                self.log.info(f"           -> new best MRR={self.best_mrr:.4f} (saved model_best.pt)")

    def fit(self):
        cfg = self.cfg
        turns = cfg.train.get("turns", 5)
        ept = cfg.train.get("epoch_per_turn", 1200)
        total = turns * ept
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"Device: {self.device} | entities={self.model.N} rels(x2)={self.model.R} "
                      f"edges={self.model.adj_index.size(1)} train={self.train_pairs.size(0)} "
                      f"test={self.test_left.numel()} | turns={turns} x {ept}")
        t0 = time.time()
        epoch = 0
        for turn in range(turns):
            self.log.info(f"=== turn {turn + 1}/{turns} (train pairs={self.train_pairs.size(0)}) ===")
            for _ in tqdm(range(ept), desc=f"RREA t{turn+1}", ncols=100):
                epoch += 1
                losses = self.train_epoch()
                losses["epoch"] = epoch
                self.loss_hist.append(losses)
                self._append_csv(cfg.logging.loss_csv, losses, ["epoch", "loss"])
                if epoch % cfg.eval.every == 0 or epoch == total:
                    self._eval_and_log(epoch, total, losses)
            if turn < turns - 1:                      # bootstrap between turns
                self.bootstrap()

        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, total)
        self.save_embeddings()
        self.plot_curves()
        self.log.info(f"Done in {(time.time()-t0)/60:.1f} min. Best MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}
