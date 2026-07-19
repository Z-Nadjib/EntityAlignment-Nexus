"""Re-score a finished run under the unified evaluation protocol.

Every model is trained with its own hyper-parameters, but the numbers reported in
the paper must all be produced by *one* scoring rule. This module reloads a run's
best checkpoint, recomputes the entity representations, and scores the test pairs
with CSLS (``--csls_k``, default 10) averaged over both ranking directions,
whatever scoring the run itself happened to use.

No retraining is involved: only the forward pass and the ranking are recomputed.

Two methods are deliberate exceptions, and are reported as such in the paper:

* **JAPE** fuses a structural and an attribute similarity, so there is no single
  embedding to score. We apply the same CSLS correction to the fused similarity
  and evaluate it in both directions.
* **DGMC** produces a sparse top-k correspondence as part of the method itself,
  so its own matching is kept and simply reported.

Usage (from the ``code/`` directory)::

    python -m src.utils.rescore ../experiments/kecg_zh_en_0_3_20260718-044645
    python -m src.utils.rescore ../experiments/*/ --csv ../rescored.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path

import torch

from . import metrics as M
from .config import load_config


def _build(cfg, device, logger):
    """Rebuild data, model and trainer exactly as ``src.main`` does."""
    from ..data import load_dbp15k, load_dgmc_dbp15k
    from .. import main as m
    from ..trainer import (AliNetTrainer, BootEATrainer, DGMCTrainer, GCNAlignTrainer,
                           JAPETrainer, KECGTrainer, MRAEATrainer, RREATrainer, Trainer)

    name = str(cfg.experiment.get("model", "naea")).lower()
    root = Path(cfg._project_root) / cfg.data.root
    if name == "dgmc":
        data = load_dgmc_dbp15k(root, cfg.data.pair)
        return name, data, DGMCTrainer(cfg, data, m.build_dgmc(cfg, data, device), Path("."), logger)

    data = load_dbp15k(root, cfg.data.lang, cfg.data.fold, cfg.data.use_mtranse_format)
    builders = {
        "bootea": (m.build_bootea, BootEATrainer), "alinet": (m.build_alinet, AliNetTrainer),
        "kecg": (m.build_kecg, KECGTrainer), "gcnalign": (m.build_gcnalign, GCNAlignTrainer),
        "jape": (m.build_jape, JAPETrainer), "mraea": (m.build_mraea, MRAEATrainer),
        "rrea": (m.build_rrea, RREATrainer), "naea": (m.build_naea, Trainer),
    }
    build, Tr = builders.get(name, (m.build_naea, Trainer))
    return name, data, Tr(cfg, data, build(cfg, data, device), Path("."), logger)


@torch.no_grad()
def _jape_both_directions(trainer, csls_k, hits_at, chunk, use_csls=True):
    """JAPE: score the fused SE+AE similarity, evaluated in both directions."""
    trainer.model.eval()
    zl = torch.nn.functional.normalize(trainer.model.encode_all(trainer.test_left), dim=-1)
    zr = torch.nn.functional.normalize(trainer.model.encode_all(trainer.test_right), dim=-1)
    sim = zl @ zr.t()
    if getattr(trainer, "ae_sim", None) is not None:
        sim = trainer.beta * sim + (1 - trainer.beta) * trainer.ae_sim
    if use_csls:
        rs = sim.topk(csls_k, dim=1).values.mean(1)
        cs = sim.topk(csls_k, dim=0).values.mean(0)
        sim = 2 * sim - rs[:, None] - cs[None, :]                   # CSLS
    l2r = M._rank_metrics(sim, hits_at, chunk)
    r2l = M._rank_metrics(sim.t(), hits_at, chunk)
    return {"l2r": l2r, "r2l": r2l,
            "avg": {k: 0.5 * (l2r[k] + r2l[k]) for k in l2r}}


def rescore(run_dir: Path, csls_k: int = 10, device: str = "cpu", project_root: Path | None = None,
            metric: str = "csls", direction: str = "both"):
    """Return ``(model_name, metrics_dict)`` for one run under the unified protocol.

    ``metric`` / ``direction`` are exposed so that Section 4.2 (scoring choices)
    can be reproduced from this same entry point: pass ``metric="cosine"`` to get
    the plain-cosine baseline, or ``direction="l2r"`` / ``"r2l"`` for one side only.
    """
    import logging
    logger = logging.getLogger("rescore")
    logger.addHandler(logging.NullHandler())

    run_dir = Path(run_dir)
    root = Path(project_root) if project_root else run_dir.parent.parent
    cfg = load_config(run_dir / "config_used.yaml", project_root=root)

    # the whole point: one scoring rule for every method
    cfg.eval.metric = metric
    cfg.eval.csls_k = csls_k
    cfg.eval.direction = direction
    cfg.experiment.device = device

    dev = torch.device(device)
    name, _, trainer = _build(cfg, dev, logger)

    ckpt = run_dir / "model_best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        trainer.model.load_state_dict(state["model_state"])
    else:
        print(f"  ! no model_best.pt in {run_dir.name}, scoring the final weights", file=sys.stderr)

    # DGMC is built for phase 1 (num_steps=0); the trained checkpoint comes from
    # phase 2, so restore the consensus iterations before scoring.
    if name == "dgmc":
        trainer.model.num_steps = int(cfg.model.get("num_steps", 0))

    hits = tuple(cfg.eval.get("hits_at", (1, 5, 10)))
    chunk = cfg.eval.get("eval_chunk", 1024)
    if name == "jape":
        return name, _jape_both_directions(trainer, csls_k, hits, chunk, use_csls=(metric == "csls"))
    return name, trainer.evaluate()


def _headline(res: dict):
    """Pick the row a table should quote: averaged directions when available."""
    for key in ("avg", "SE+AE", "l2r"):
        if key in res and isinstance(res[key], dict):
            return res[key]
    return res                                    # DGMC returns flat Hit@k / MRR


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories under experiments/")
    ap.add_argument("--csls_k", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--csv", default=None, help="also write the table to this CSV")
    ap.add_argument("--metric", default="csls", choices=["csls", "cosine"],
                    help="scoring rule (Section 4.2 compares the two)")
    ap.add_argument("--direction", default="both", choices=["both", "l2r", "r2l"])
    args = ap.parse_args()

    rows = []
    print(f"{'model':10} {'Hit@1':>8} {'Hit@10':>8} {'MRR':>8}")
    for run in args.runs:
        try:
            name, res = rescore(run, args.csls_k, args.device, args.project_root,
                                metric=args.metric, direction=args.direction)
        except Exception as exc:                                   # keep going on the others
            print(f"{Path(run).name:10} FAILED: {exc}", file=sys.stderr)
            continue
        h = _headline(res)
        row = {"model": name, "run": Path(run).name,
               "Hit@1": h.get("Hit@1"), "Hit@10": h.get("Hit@10"), "MRR": h.get("MRR"),
               "Hit@1_l2r": res.get("l2r", {}).get("Hit@1"),
               "Hit@1_r2l": res.get("r2l", {}).get("Hit@1")}
        rows.append(row)
        d = ""
        if row["Hit@1_l2r"] is not None:
            d = f"  | l2r {row['Hit@1_l2r']:.4f}  r2l {row['Hit@1_r2l']:.4f}"
        print(f"{name:10} {row['Hit@1']:8.4f} {row['Hit@10']:8.4f} {row['MRR']:8.4f}{d}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten: {args.csv}")


if __name__ == "__main__":
    main()
