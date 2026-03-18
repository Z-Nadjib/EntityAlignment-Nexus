"""DBP15K data loading + neighbourhood construction for NAEA.

Data layout (the clean JAPE / MTransE split we use, ``<lang>/mtranse/<fold>/``):

    ent_ids_1 / ent_ids_2 : "<id>\\t<uri>"   entities of KG1 (e.g. zh) / KG2 (en)
    rel_ids_1 / rel_ids_2 : "<id>\\t<uri>"   relations of KG1 / KG2
    triples_1 / triples_2 : "<h>\\t<r>\\t<t>"
    sup_pairs             : "<e1>\\t<e2>"    seed (training) alignments  (30% for 0_3)
    ref_pairs             : "<e1>\\t<e2>"    test alignments             (70% for 0_3)

Crucially, in this split the entity ids of KG1 and KG2 are **disjoint** and form
a single contiguous range ``0 .. num_entities-1``; likewise relation ids. That
lets us use one shared embedding table and index it directly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Raw file readers
# --------------------------------------------------------------------------- #
def _read_id_uri(path: Path) -> dict[int, str]:
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                d[int(parts[0])] = parts[1]
    return d


def _read_triples(path: Path) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append((int(p[0]), int(p[1]), int(p[2])))
    return np.asarray(rows, dtype=np.int64)


def _read_pairs(path: Path) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                rows.append((int(p[0]), int(p[1])))
    return np.asarray(rows, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  Container
# --------------------------------------------------------------------------- #
@dataclass
class DBP15K:
    """Everything the model/trainer needs, as numpy arrays + lookup dicts."""

    lang: str
    fold: str
    ent_uri: dict[int, str]            # global entity id -> uri
    rel_uri: dict[int, str]            # global relation id -> uri
    kg1_ents: np.ndarray               # entity ids belonging to KG1
    kg2_ents: np.ndarray               # entity ids belonging to KG2
    triples1: np.ndarray               # (M1, 3)  KG1 triples
    triples2: np.ndarray               # (M2, 3)  KG2 triples
    train_pairs: np.ndarray            # (S, 2)   seed alignments
    test_pairs: np.ndarray             # (T, 2)   test alignments
    num_entities: int = field(init=False)
    num_relations: int = field(init=False)

    def __post_init__(self):
        self.num_entities = int(max(self.ent_uri) + 1)
        self.num_relations = int(max(self.rel_uri) + 1)

    @property
    def triples(self) -> np.ndarray:
        return np.concatenate([self.triples1, self.triples2], axis=0)

    def summary(self) -> dict:
        return {
            "lang": self.lang,
            "fold": self.fold,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "|KG1 ents|": len(self.kg1_ents),
            "|KG2 ents|": len(self.kg2_ents),
            "|KG1 triples|": len(self.triples1),
            "|KG2 triples|": len(self.triples2),
            "train_pairs": len(self.train_pairs),
            "test_pairs": len(self.test_pairs),
        }


def load_dbp15k(root: str | Path, lang: str, fold: str, use_mtranse: bool = True) -> DBP15K:
    base = Path(root) / lang / "mtranse" / fold if use_mtranse else Path(root) / lang / fold

    ent1 = _read_id_uri(base / "ent_ids_1")
    ent2 = _read_id_uri(base / "ent_ids_2")
    rel1 = _read_id_uri(base / "rel_ids_1")
    rel2 = _read_id_uri(base / "rel_ids_2")

    ent_uri = {**ent1, **ent2}
    rel_uri = {**rel1, **rel2}

    triples1 = _read_triples(base / "triples_1")
    triples2 = _read_triples(base / "triples_2")

    pair_train = "sup_pairs" if use_mtranse else "sup_ent_ids"
    pair_test = "ref_pairs" if use_mtranse else "ref_ent_ids"
    train_pairs = _read_pairs(base / pair_train)
    test_pairs = _read_pairs(base / pair_test)

    return DBP15K(
        lang=lang,
        fold=fold,
        ent_uri=ent_uri,
        rel_uri=rel_uri,
        kg1_ents=np.asarray(sorted(ent1), dtype=np.int64),
        kg2_ents=np.asarray(sorted(ent2), dtype=np.int64),
        triples1=triples1,
        triples2=triples2,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
    )


# --------------------------------------------------------------------------- #
#  Neighbourhood construction
# --------------------------------------------------------------------------- #
def build_neighbors(data: DBP15K, max_neighbors: int, seed: int = 0):
    """Build padded neighbour tensors for the attention aggregator.

    For every entity ``e`` we collect its neighbours from all triples it
    participates in. A neighbour is a *(relation, entity, direction)* triplet:

      * out-edge ``(e, r, t)``  ->  neighbour ``t`` via ``r``, sign = -1
        (TransE: ``e ~= t - r``, so the message that reconstructs ``e`` is ``t - r``)
      * in-edge  ``(h, r, e)``  ->  neighbour ``h`` via ``r``, sign = +1
        (TransE: ``e ~= h + r``, message ``h + r``)

    Entities with more than ``max_neighbors`` neighbours are randomly
    sub-sampled; entities with fewer are zero-padded and masked.

    Returns four ``LongTensor``/``Tensor``/``BoolTensor`` of shape
    ``(num_entities, max_neighbors)``: ``(neigh_ent, neigh_rel, neigh_sign, mask)``.
    """
    rng = random.Random(seed)
    N = data.num_entities
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(N)]

    for arr in (data.triples1, data.triples2):
        for h, r, t in arr:
            adj[h].append((int(r), int(t), -1))   # out-edge:  e=h, message t - r
            adj[t].append((int(r), int(h), +1))   # in-edge:   e=t, message h + r

    K = max_neighbors
    neigh_ent = np.zeros((N, K), dtype=np.int64)
    neigh_rel = np.zeros((N, K), dtype=np.int64)
    neigh_sign = np.zeros((N, K), dtype=np.float32)
    mask = np.zeros((N, K), dtype=bool)

    for e in range(N):
        nbrs = adj[e]
        if len(nbrs) > K:
            nbrs = rng.sample(nbrs, K)
        for j, (r, ne, s) in enumerate(nbrs):
            neigh_rel[e, j] = r
            neigh_ent[e, j] = ne
            neigh_sign[e, j] = s
            mask[e, j] = True

    return (
        torch.from_numpy(neigh_ent),
        torch.from_numpy(neigh_rel),
        torch.from_numpy(neigh_sign),
        torch.from_numpy(mask),
    )


# --------------------------------------------------------------------------- #
#  Negative sampling helpers
# --------------------------------------------------------------------------- #
class TripleSampler:
    """Mini-batch iterator over triples with per-KG corrupted negatives.

    Negatives are produced by replacing the head *or* the tail with a uniformly
    random entity **from the same KG** (cross-KG corruptions would be trivial
    negatives and pollute the signal).
    """

    def __init__(self, data: DBP15K, device, neg: int = 5):
        self.device = device
        self.neg = neg
        t1 = torch.from_numpy(data.triples1)
        t2 = torch.from_numpy(data.triples2)
        kg = torch.cat([torch.zeros(len(t1), dtype=torch.long),
                        torch.ones(len(t2), dtype=torch.long)])
        self.triples = torch.cat([t1, t2], 0).to(device)        # (M,3)
        self.kg = kg.to(device)                                  # (M,)
        self.kg1_ents = torch.from_numpy(data.kg1_ents).to(device)
        self.kg2_ents = torch.from_numpy(data.kg2_ents).to(device)

    def __len__(self):
        return len(self.triples)

    def _rand_same_kg(self, kg_flags: torch.Tensor) -> torch.Tensor:
        """A random entity from the same KG as each row of ``kg_flags``."""
        n = kg_flags.shape[0]
        r1 = self.kg1_ents[torch.randint(len(self.kg1_ents), (n,), device=self.device)]
        r2 = self.kg2_ents[torch.randint(len(self.kg2_ents), (n,), device=self.device)]
        return torch.where(kg_flags == 0, r1, r2)

    def batches(self, batch_size: int, shuffle: bool = True):
        M = len(self.triples)
        order = torch.randperm(M, device=self.device) if shuffle else torch.arange(M, device=self.device)
        for s in range(0, M, batch_size):
            idx = order[s:s + batch_size]
            pos = self.triples[idx]                              # (B,3)
            kg = self.kg[idx]                                    # (B,)
            B = pos.shape[0]
            # repeat for `neg` corruptions
            pos_r = pos.repeat(self.neg, 1)                      # (B*neg,3)
            kg_r = kg.repeat(self.neg)
            neg = pos_r.clone()
            corrupt_head = torch.rand(B * self.neg, device=self.device) < 0.5
            rnd = self._rand_same_kg(kg_r)
            neg[:, 0] = torch.where(corrupt_head, rnd, neg[:, 0])
            neg[:, 2] = torch.where(~corrupt_head, rnd, neg[:, 2])
            yield pos, neg, kg


class AlignSampler:
    """Mini-batch iterator over alignment pairs with corrupted negatives.

    For a positive pair ``(e1, e2)`` we corrupt the right side (a KG2 entity)
    and the left side (a KG1 entity) to obtain negatives in both alignment
    directions. ``set_pairs`` lets the trainer inject bootstrapped pseudo-pairs.

    **Hard (nearest-neighbour) negatives.** If ``set_hard_negatives`` has been
    called, negatives are drawn from each pair's pre-computed nearest cross-KG
    candidates (the confusable entities) instead of uniformly at random. This is
    the epsilon-truncated negative sampling of BootEA/NAEA, which sharpens Hit@1. Any
    candidate that equals the gold target is replaced by a random one.
    """

    def __init__(self, data: DBP15K, device, neg: int = 5):
        self.device = device
        self.neg = neg
        self.kg1_ents = torch.from_numpy(data.kg1_ents).to(device)
        self.kg2_ents = torch.from_numpy(data.kg2_ents).to(device)
        self.hard_r = None          # (S, C) nearest KG2 candidates per pair (corrupt right)
        self.hard_l = None          # (S, C) nearest KG1 candidates per pair (corrupt left)
        self.set_pairs(data.train_pairs)

    def set_pairs(self, pairs):
        if isinstance(pairs, np.ndarray):
            pairs = torch.from_numpy(pairs)
        self.pairs = pairs.to(self.device).long()
        self.hard_r = self.hard_l = None    # invalidate stale hard-negative tables

    def set_hard_negatives(self, hard_r, hard_l):
        """``hard_r``/``hard_l``: (S, C) LongTensors aligned with ``self.pairs``."""
        self.hard_r = hard_r.to(self.device)
        self.hard_l = hard_l.to(self.device)

    def __len__(self):
        return len(self.pairs)

    def _sample_hard(self, table, rows, n, gold):
        """Pick ``n`` candidates per row from ``table[rows]``, avoiding ``gold``."""
        cand = table[rows]                                       # (B, C)
        C = cand.shape[1]
        sel = torch.randint(C, (cand.shape[0], n), device=self.device)
        out = torch.gather(cand, 1, sel).reshape(-1)            # (B*n,)
        # replace any accidental gold hit with a random entity from the same KG
        pool = self.kg2_ents if table is self.hard_r else self.kg1_ents
        clash = out == gold
        if clash.any():
            out[clash] = pool[torch.randint(len(pool), (int(clash.sum()),), device=self.device)]
        return out

    def batches(self, batch_size: int, shuffle: bool = True):
        S = len(self.pairs)
        order = torch.randperm(S, device=self.device) if shuffle else torch.arange(S, device=self.device)
        for s in range(0, S, batch_size):
            idx = order[s:s + batch_size]
            pos = self.pairs[idx]                                # (B,2)
            B = pos.shape[0]
            n = self.neg
            e1 = pos[:, 0].repeat_interleave(n)
            e2 = pos[:, 1].repeat_interleave(n)
            if self.hard_r is not None and self.hard_l is not None:
                neg_r = self._sample_hard(self.hard_r, idx, n, e2)
                neg_l = self._sample_hard(self.hard_l, idx, n, e1)
            else:
                neg_r = self.kg2_ents[torch.randint(len(self.kg2_ents), (B * n,), device=self.device)]
                neg_l = self.kg1_ents[torch.randint(len(self.kg1_ents), (B * n,), device=self.device)]
            yield pos, (e1, e2, neg_l, neg_r)


# =========================================================================== #
#  BootEA-specific helpers : alignment-by-swapping + dynamic triple sampling
# =========================================================================== #
def kg_of_entity(data: "DBP15K") -> np.ndarray:
    """Return an array ``kg[e] in {0, 1}`` giving the KG each entity belongs to."""
    kg = np.zeros(data.num_entities, dtype=np.int64)
    kg[data.kg2_ents] = 1
    return kg


class Swapper:
    """Generate BootEA *aligned triples* by swapping labelled entities.

    For a labelled pair ``(a, b)`` we substitute ``b`` for ``a`` in each of
    ``a``'s triples and vice-versa, e.g. ``(a, r, t) -> (b, r, t)``. The swapped
    entities then share relational contexts, which pulls their embeddings
    together, which is BootEA's core alignment mechanism.

    Per-entity triple lists are precomputed once; :meth:`generate` is then cheap
    enough to be re-run every bootstrapping round on the (growing) labelled set.
    """

    def __init__(self, data: "DBP15K"):
        triples = np.concatenate([data.triples1, data.triples2], axis=0)
        self.triples = triples
        n = data.num_entities
        self.as_head = [[] for _ in range(n)]
        self.as_tail = [[] for _ in range(n)]
        for i, (h, _r, t) in enumerate(triples):
            self.as_head[int(h)].append(i)
            self.as_tail[int(t)].append(i)

    def generate(self, labeled_pairs, cap_per_role: int = 100) -> np.ndarray:
        """Return an ``(K, 3)`` array of swapped (h, r, t) triples for the pairs."""
        T = self.triples
        out = []
        for a, b in labeled_pairs:
            a, b = int(a), int(b)
            for src, dst in ((a, b), (b, a)):          # dst takes src's place
                for ti in self.as_head[src][:cap_per_role]:
                    out.append((dst, T[ti, 1], T[ti, 2]))
                for ti in self.as_tail[src][:cap_per_role]:
                    out.append((T[ti, 0], T[ti, 1], dst))
        if not out:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(out, dtype=np.int64)


class DynamicTripleSampler:
    """Mini-batch sampler over a *mutable* set of triples (BootEA / AlignE).

    Unlike :class:`TripleSampler` (which fixes the per-KG split), this sampler
    determines each corrupted entity's KG from a global ``kg_of_ent`` array, so
    it correctly handles **swapped** triples whose head and tail may live in
    different KGs. Negatives are uniform within the corrupted position's KG, or
    epsilon-truncated (drawn from that entity's nearest same-KG neighbours) when a
    candidate table has been provided via :meth:`set_candidates`.
    """

    def __init__(self, device, kg_of_ent: np.ndarray, kg1_ents, kg2_ents, neg: int = 5):
        self.device = device
        self.neg = neg
        self.kg_of_ent = torch.as_tensor(kg_of_ent, device=device).long()
        self.kg1_ents = torch.as_tensor(kg1_ents, device=device).long()
        self.kg2_ents = torch.as_tensor(kg2_ents, device=device).long()
        self.cand = None            # (N, C) nearest same-KG neighbours, optional
        self.triples = torch.empty((0, 3), dtype=torch.long, device=device)

    def set_triples(self, triples):
        if isinstance(triples, np.ndarray):
            triples = torch.from_numpy(triples)
        self.triples = triples.to(self.device).long()

    def set_candidates(self, cand):
        """``cand`` : (N, C) LongTensor of nearest same-KG neighbours per entity."""
        self.cand = cand.to(self.device) if cand is not None else None

    def __len__(self):
        return len(self.triples)

    def _rand_same_kg(self, ent_ids):
        kg = self.kg_of_ent[ent_ids]
        n = ent_ids.shape[0]
        r1 = self.kg1_ents[torch.randint(len(self.kg1_ents), (n,), device=self.device)]
        r2 = self.kg2_ents[torch.randint(len(self.kg2_ents), (n,), device=self.device)]
        return torch.where(kg == 0, r1, r2)

    def _trunc_same_kg(self, ent_ids):
        """Epsilon-truncated: a random one of each entity's nearest same-KG neighbours."""
        cand = self.cand[ent_ids]                              # (n, C)
        sel = torch.randint(cand.shape[1], (cand.shape[0], 1), device=self.device)
        return torch.gather(cand, 1, sel).squeeze(1)

    def batches(self, batch_size: int, shuffle: bool = True):
        M = len(self.triples)
        order = torch.randperm(M, device=self.device) if shuffle else torch.arange(M, device=self.device)
        for s in range(0, M, batch_size):
            idx = order[s:s + batch_size]
            pos = self.triples[idx]                            # (B,3)
            B = pos.shape[0]
            n = self.neg
            pos_r = pos.repeat(n, 1)                           # (B*n,3)
            neg = pos_r.clone()
            corrupt_head = torch.rand(B * n, device=self.device) < 0.5
            # entity currently occupying the position we corrupt
            orig = torch.where(corrupt_head, pos_r[:, 0], pos_r[:, 2])
            repl = self._trunc_same_kg(orig) if self.cand is not None else self._rand_same_kg(orig)
            neg[:, 0] = torch.where(corrupt_head, repl, neg[:, 0])
            neg[:, 2] = torch.where(~corrupt_head, repl, neg[:, 2])
            yield pos, neg


# =========================================================================== #
#  AliNet-specific helpers : 1-hop normalised adjacency + capped 2-hop edges
# =========================================================================== #
def build_alinet_graph(data: "DBP15K", max_two_hop: int = 10, seed: int = 0):
    """Build the graph structures AliNet needs (relation types ignored).

    Returns
    -------
    adj1 : torch.sparse_coo_tensor (N, N)
        Symmetrically normalised adjacency with self-loops,
        ``A_hat = D^{-1/2} (A + I) D^{-1/2}``, used for 1-hop GCN aggregation.
    two_hop : LongTensor (2, E)
        Edges ``[dst, src]`` where ``src`` is a (sampled, capped) 2-hop neighbour
        of ``dst`` (excluding 1-hop neighbours and self), used for the
        attention-based 2-hop aggregation.
    """
    rng = random.Random(seed)
    N = data.num_entities
    nbrs = [set() for _ in range(N)]
    for arr in (data.triples1, data.triples2):
        for h, _r, t in arr:
            h, t = int(h), int(t)
            if h != t:
                nbrs[h].add(t)
                nbrs[t].add(h)

    # ---- 1-hop normalised adjacency (with self-loops) ---------------------
    rows, cols = [], []
    for i in range(N):
        rows.append(i); cols.append(i)            # self-loop
        for j in nbrs[i]:
            rows.append(i); cols.append(j)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    deg = np.asarray([len(nbrs[i]) + 1 for i in range(N)], dtype=np.float64)  # +1 self-loop
    inv_sqrt = 1.0 / np.sqrt(deg)
    vals = (inv_sqrt[rows] * inv_sqrt[cols]).astype(np.float32)
    adj1 = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(vals), (N, N)
    ).coalesce()

    # ---- capped 2-hop edges ----------------------------------------------
    dst, src = [], []
    for i in range(N):
        one = nbrs[i]
        if not one:
            continue
        two = set()
        # bound exploration on hub nodes
        for j in list(one)[:64]:
            two.update(list(nbrs[j])[:64])
        two.discard(i)
        two -= one
        if not two:
            continue
        two = list(two)
        if len(two) > max_two_hop:
            two = rng.sample(two, max_two_hop)
        for k in two:
            dst.append(i); src.append(k)
    two_hop = torch.tensor([dst, src], dtype=torch.long) if dst else torch.zeros((2, 0), dtype=torch.long)
    return adj1, two_hop


# =========================================================================== #
#  KECG-specific helper : combined undirected graph edges (both KGs)
# =========================================================================== #
def build_kecg_graph(data: "DBP15K"):
    """Edges of the combined graph used by KECG's shared GAT.

    Both KGs are put in one graph (relation types ignored); edges are made
    undirected and self-loops are added. Returns ``edge_index`` of shape
    ``(2, E) = [dst, src]`` so attention aggregates ``src`` features into ``dst``.
    """
    N = data.num_entities
    seen = set()
    for arr in (data.triples1, data.triples2):
        for h, _r, t in arr:
            h, t = int(h), int(t)
            if h != t:
                seen.add((h, t)); seen.add((t, h))
    for i in range(N):
        seen.add((i, i))                                  # self-loops
    e = np.fromiter((x for edge in seen for x in edge), dtype=np.int64, count=2 * len(seen))
    e = e.reshape(-1, 2)                                   # (E, 2) = (dst, src)
    return torch.from_numpy(e.T.copy())                   # (2, E) = [dst, src]


# =========================================================================== #
#  GCN-Align helper : functionality-weighted, symmetrically-normalised adjacency
# =========================================================================== #
def build_gcnalign_adj(data: "DBP15K"):
    """Build GCN-Align's structure adjacency (Wang et al., EMNLP 2018).

    Each relation r gets a *functionality* ``fun(r)=#heads/#triples`` and inverse
    functionality ``ifun(r)=#tails/#triples``. For a triple ``(h, r, t)`` the edge
    weights accumulate ``M[h,t] += max(ifun(r), 0.3)`` and ``M[t,h] += max(fun(r), 0.3)``.
    Self-loops are added, then the matrix is symmetrically normalised
    ``D^{-1/2} (M+I) D^{-1/2}``. Returns a torch sparse tensor (N, N).
    """
    triples = np.concatenate([data.triples1, data.triples2], axis=0)
    cnt, head, tail = {}, {}, {}
    for h, r, t in triples:
        r = int(r)
        if r not in cnt:
            cnt[r] = 0; head[r] = set(); tail[r] = set()
        cnt[r] += 1; head[r].add(int(h)); tail[r].add(int(t))
    r2f = {r: len(head[r]) / cnt[r] for r in cnt}      # functionality
    r2if = {r: len(tail[r]) / cnt[r] for r in cnt}     # inverse functionality

    N = data.num_entities
    M = {}
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        M[(h, t)] = M.get((h, t), 0.0) + max(r2if[r], 0.3)
        M[(t, h)] = M.get((t, h), 0.0) + max(r2f[r], 0.3)
    for i in range(N):                                 # self-loops
        M[(i, i)] = M.get((i, i), 0.0) + 1.0

    rows = np.fromiter((i for (i, _j) in M), dtype=np.int64, count=len(M))
    cols = np.fromiter((j for (_i, j) in M), dtype=np.int64, count=len(M))
    vals = np.fromiter(M.values(), dtype=np.float64, count=len(M))
    deg = np.zeros(N, dtype=np.float64)
    np.add.at(deg, rows, vals)                          # row sums
    inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    norm_vals = (inv_sqrt[rows] * vals * inv_sqrt[cols]).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(norm_vals), (N, N)
    ).coalesce()


# =========================================================================== #
#  JAPE helpers : merged-seed format check + attribute bag (AE channel)
# =========================================================================== #
def load_jape_attributes(root: str | Path, lang: str, fold: str, ent_uri: dict[int, str]):
    """Build the entity-attribute bag for JAPE's attribute channel (AE).

    Reads ``<lang>/training_attrs_1`` and ``training_attrs_2`` (entity URI -> list
    of attribute URIs) and ``<lang>/sup_attr_pairs`` (aligned cross-KG attribute
    URIs). A shared attribute vocabulary is built, **merging aligned attributes**
    to the same id so the bags are cross-KG comparable. Returns a binary sparse
    tensor ``(N, A)`` (entity x attribute) plus the vocab size A.
    """
    base = Path(root) / lang
    uri2id = {u: i for i, u in ent_uri.items()}

    # pre-merge aligned attribute URIs (sup_attr_pairs: "<zh_attr>\t<en_attr>")
    attr2id: dict[str, int] = {}
    sap = base / "sup_attr_pairs"
    if sap.exists():
        with open(sap, "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    aid = attr2id.setdefault(p[0], len(attr2id))
                    attr2id[p[1]] = aid                         # alias -> same id

    rows, cols = [], []
    for fn in ("training_attrs_1", "training_attrs_2"):
        path = base / fn
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = uri2id.get(parts[0])
                if eid is None:
                    continue
                for a in parts[1:]:
                    aid = attr2id.setdefault(a, len(attr2id))
                    rows.append(eid); cols.append(aid)
    A = len(attr2id)
    N = len(ent_uri)
    idx = torch.tensor([rows, cols], dtype=torch.long)
    vals = torch.ones(len(rows), dtype=torch.float32)
    bag = torch.sparse_coo_tensor(idx, vals, (N, A)).coalesce()
    # binary (clamp accumulated duplicates to 1)
    bag = torch.sparse_coo_tensor(bag.indices(), bag.values().clamp(max=1.0), (N, A)).coalesce()
    return bag, A


# --------------------------------------------------------------------------- #
#  DGMC : DBP15K with entity-name word-embedding features (Fey et al. 2020)
# --------------------------------------------------------------------------- #
@dataclass
class DGMCData:
    """Two graphs (source/target) + name features for graph matching.

    Mirrors the layout consumed by Deep Graph Matching Consensus on DBP15K:
    each entity carries a node feature = SUM of the pre-trained word embeddings
    of its (translated) name; the two KGs stay **separate** with their own
    contiguous node ids; ``train_y`` / ``test_y`` are ``[2, M]`` matchings of
    local indices (row in source, col in target).
    """

    pair: str
    x1: torch.Tensor                  # (N1, F)  source node features
    x2: torch.Tensor                  # (N2, F)  target node features
    edge_index1: torch.Tensor         # (2, E1)  source edges (local ids)
    edge_index2: torch.Tensor         # (2, E2)  target edges (local ids)
    train_y: torch.Tensor             # (2, S)   seed matchings
    test_y: torch.Tensor              # (2, T)   test matchings
    names1: list = None               # source entity names by local idx (display only)
    names2: list = None               # target entity names by local idx (display only)

    @property
    def num_features(self) -> int:
        return self.x1.size(-1)

    def summary(self) -> dict:
        return {
            "pair": self.pair,
            "num_features": self.num_features,
            "|src nodes|": int(self.x1.size(0)),
            "|tgt nodes|": int(self.x2.size(0)),
            "|src edges|": int(self.edge_index1.size(1)),
            "|tgt edges|": int(self.edge_index2.size(1)),
            "train_y": int(self.train_y.size(1)),
            "test_y": int(self.test_y.size(1)),
        }


def _load_glove(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Read ``sub.glove.300d`` -> {word: vec} and the ``**UNK**`` fallback vec."""
    embs: dict[str, torch.Tensor] = {}
    unk = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            info = line.rstrip("\n").split(" ")
            if len(info) > 300:                       # "<word> v1 .. v300"
                embs[info[0]] = torch.tensor([float(x) for x in info[1:]])
            else:                                     # the trailing UNK line (300 floats)
                unk = torch.tensor([float(x) for x in info])
    if unk is None:
        unk = torch.zeros(300)
    embs["**UNK**"] = unk
    return embs, unk


def _load_dgmc_graph(triple_path: Path, feat_path: Path, embs, unk):
    """Build (x, edge_index, assoc) for one KG: summed name embeddings + edges."""
    # node features: id \t name  ->  sum of word vectors (SumEmbedding)
    feats: dict[int, torch.Tensor] = {}
    raw_name: dict[int, str] = {}
    with open(feat_path, "r", encoding="utf-8") as f:
        for line in f:
            info = line.rstrip("\n").split("\t")
            info = info if len(info) == 2 else info + ["**UNK**"]
            seq = info[1].lower().split()
            vecs = [embs.get(w, unk) for w in seq] or [unk]
            feats[int(info[0])] = torch.stack(vecs, 0).sum(0)         # (300,)
            raw_name[int(info[0])] = info[1]

    ids = sorted(feats)
    assoc = {orig: i for i, orig in enumerate(ids)}                  # orig id -> local idx
    x = torch.stack([feats[i] for i in ids], 0)                      # (N, 300)
    names = [raw_name[i] for i in ids]                              # by local idx

    rows, cols = [], []
    with open(triple_path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.split()
            h, t = int(h), int(t)
            if h in assoc and t in assoc:
                rows.append(assoc[h]); cols.append(assoc[t])
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return x, edge_index, assoc, names


def _read_ref(path: Path, assoc1, assoc2) -> torch.Tensor:
    """Read ``<src_orig>\\t<tgt_orig>`` gold pairs -> [2, M] of local indices."""
    rows, cols = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.split()
            a, b = int(a), int(b)
            if a in assoc1 and b in assoc2:
                rows.append(assoc1[a]); cols.append(assoc2[b])
    return torch.tensor([rows, cols], dtype=torch.long)


def load_dgmc_dbp15k(root: str | Path, pair: str) -> DGMCData:
    """Load the DGMC/GMNN DBP15K split (entity-name GloVe features).

    ``root`` points at the extracted ``DBP15K`` directory containing
    ``sub.glove.300d`` and per-pair ``triples_{1,2}``, ``id_features_{1,2}``,
    ``train.ref`` (4500) and ``test.ref`` (10500). Graph 1 is the source
    language (zh/ja/fr), graph 2 is English.
    """
    base = Path(root)
    embs, unk = _load_glove(base / "sub.glove.300d")
    pdir = base / pair
    x1, ei1, assoc1, names1 = _load_dgmc_graph(pdir / "triples_1", pdir / "id_features_1", embs, unk)
    x2, ei2, assoc2, names2 = _load_dgmc_graph(pdir / "triples_2", pdir / "id_features_2", embs, unk)
    train_y = _read_ref(pdir / "train.ref", assoc1, assoc2)
    test_y = _read_ref(pdir / "test.ref", assoc1, assoc2)
    return DGMCData(pair, x1, x2, ei1, ei2, train_y, test_y, names1, names2)


def build_dgmc_adj(edge_index: torch.Tensor, num_nodes: int):
    """Two row-normalised sparse adjacencies for RelConv's mean aggregation.

    ``adj_in @ X``  = mean of features over *incoming* neighbours (flow s->t),
    ``adj_out @ X`` = mean of features over *outgoing* neighbours (flow t->s).
    Parallel edges (the same pair linked by several relations) are **kept**, so a
    repeated neighbour is weighted by its multiplicity - matching PyG's mean
    aggregation over the raw triple edge list.
    """
    src, dst = edge_index[0], edge_index[1]

    def _norm(row, col):                               # row-normalised, multiplicity-aware
        idx = torch.stack([row, col])
        A = torch.sparse_coo_tensor(idx, torch.ones(row.numel()),
                                    (num_nodes, num_nodes)).coalesce()
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
        r, c = A.indices()
        val = A.values() / deg[r]
        return torch.sparse_coo_tensor(torch.stack([r, c]), val,
                                       (num_nodes, num_nodes)).coalesce()

    adj_in = _norm(dst, src)                            # i <- j for edges j->i
    adj_out = _norm(src, dst)                           # i <- j for edges i->j
    return adj_in, adj_out


# --------------------------------------------------------------------------- #
#  MRAEA : meta-relation-aware graph structures (Mao et al., WSDM 2020)
# --------------------------------------------------------------------------- #
def build_mraea_graph(data: "DBP15K"):
    """Build the sparse structures MRAEA's attention layer consumes.

    From the (disjoint-id) triples of both KGs:
      * **directed edges** : each triple ``(h, r, t)`` yields edge ``(h, t)`` with
        relation ``r`` and the inverse edge ``(t, h)`` with relation ``r + R`` (a
        relation and its inverse get distinct meta-relation embeddings; table size
        ``2R``). Parallel relations on the same pair are merged into one edge.
      * ``edge_rel`` : sparse ``(E, 2R)`` mapping each unique edge to its
        relation(s), value ``1/#relations`` (used to pool relation attention).
      * ``rel_adj`` : sparse ``(N, 2R)`` node->relation incidence (incoming rel ``r``
        in column ``r``, outgoing rel ``r`` in column ``R+r``) for the relation
        feature aggregation.
      * ``ent_adj`` : sparse ``(N, N)`` neighbour incidence + self-loops for the
        entity feature aggregation.
    Returns a dict of LongTensors / FloatTensors. The model applies softmax over
    the (ones-valued) ``rel_adj`` / ``ent_adj`` patterns, so only their indices
    matter; ``edge_rel`` keeps its averaged values.
    """
    N = data.num_entities
    R = data.num_relations
    triples = np.concatenate([data.triples1, data.triples2], axis=0)

    edge_rels: dict[tuple, list] = {}
    rel_pairs = set()                                   # (node, rel column in [0,2R))
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        edge_rels.setdefault((h, t), []).append(r)      # forward relation r
        edge_rels.setdefault((t, h), []).append(R + r)  # inverse relation R+r
        rel_pairs.add((t, r))                           # t has incoming relation r
        rel_pairs.add((h, R + r))                       # h has outgoing relation r

    edges = sorted(edge_rels.keys())                    # row-major (h, t) order
    eid = {e: i for i, e in enumerate(edges)}
    adj_index = torch.tensor(edges, dtype=torch.long).t().contiguous()   # (2, E)

    r_rows, r_cols, r_vals = [], [], []
    for e in edges:
        rels = edge_rels[e]
        w = 1.0 / len(rels)
        for rid in rels:
            r_rows.append(eid[e]); r_cols.append(rid); r_vals.append(w)
    edge_rel_index = torch.tensor([r_rows, r_cols], dtype=torch.long)
    edge_rel_val = torch.tensor(r_vals, dtype=torch.float32)

    rel_adj_index = torch.tensor(sorted(rel_pairs), dtype=torch.long).t().contiguous()

    ent_pairs = set(edges) | {(i, i) for i in range(N)}  # neighbours + self-loops
    ent_adj_index = torch.tensor(sorted(ent_pairs), dtype=torch.long).t().contiguous()

    return {
        "num_nodes": N, "num_rels": 2 * R, "num_edges": len(edges),
        "adj_index": adj_index,                          # (2, E)
        "edge_rel_index": edge_rel_index,                # (2, M)  -> (E, 2R)
        "edge_rel_val": edge_rel_val,                    # (M,)
        "rel_adj_index": rel_adj_index,                  # (2, P)  -> (N, 2R)
        "ent_adj_index": ent_adj_index,                  # (2, Q)  -> (N, N)
    }
