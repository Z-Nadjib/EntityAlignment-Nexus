"""The AliNet model.

AliNet (Sun, Wang, Hu, Li, Zhang, Qu, Li - "Knowledge Graph Alignment Network
with Gated Multi-hop Neighborhood Aggregation", AAAI 2020) aligns entities with
a GNN that mitigates the *neighbourhood non-isomorphism* across KGs by combining:

1. 1-hop aggregation: a GCN-style pass over the symmetrically-normalised
   adjacency ``A_hat`` (with self-loops):  ``g1 = A_hat (H W1)``.
2. 2-hop aggregation with attention: distant neighbours are noisier, so they
   are aggregated with a GAT-style attention over the (capped) 2-hop edges:
   ``alpha_ik = softmax_k(LeakyReLU(a1.W2 h_i + a2.W2 h_k))``, ``g2 = sum_k alpha_ik W2 h_k``.
3. Gating: a learned gate combines the two:
   ``out = sigmoid(g1 W_g + b) * g1 + (1 - sigmoid(...)) * g2``.

The final entity representation is the **concatenation** of the input embedding
and every layer's output (JK-style), L2-normalised, and used directly for
alignment / evaluation (distance in embedding space).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Scatter helpers (avoid a torch_scatter dependency)
# --------------------------------------------------------------------------- #
def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of ``scores`` grouped by ``index`` (each value in [0, n))."""
    mx = scores.new_full((n,), float("-inf")).index_reduce_(0, index, scores, "amax", include_self=True)
    mx = torch.nan_to_num(mx, neginf=0.0)
    s = (scores - mx[index]).exp()
    denom = torch.zeros(n, device=scores.device, dtype=scores.dtype).index_add_(0, index, s)
    return s / (denom[index] + 1e-16)


def scatter_add(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Sum rows of ``src`` into ``n`` buckets given by ``index``."""
    out = torch.zeros((n, src.shape[1]), device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


# --------------------------------------------------------------------------- #
#  Gated multi-hop layer
# --------------------------------------------------------------------------- #
class AliNetLayer(nn.Module):
    # Output is LINEAR (no ReLU): for entity alignment, a ReLU between propagation
    # layers destroys the structural signal (halves the features each layer) and
    # cripples generalisation. Linear/spectral propagation works far better here.
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)   # 1-hop transform
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)   # 2-hop transform
        self.a1 = nn.Parameter(torch.zeros(out_dim))       # attention (centre)
        self.a2 = nn.Parameter(torch.zeros(out_dim))       # attention (neighbour)
        self.gate = nn.Linear(out_dim, out_dim)            # gating on the 1-hop signal
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.a1.view(1, -1)); nn.init.xavier_uniform_(self.a2.view(1, -1))

    def forward(self, h, adj1, e_dst, e_src):
        n = h.shape[0]
        # 1-hop GCN aggregation
        g1 = torch.sparse.mm(adj1, self.W1(h))                       # (N, out)
        # 2-hop attention aggregation (memory-efficient over the edge list)
        wh = self.W2(h)                                              # (N, out)
        if e_dst.numel() > 0:
            # attention scores via NODE-level projections (avoids materialising E x out)
            s1 = (wh * self.a1).sum(-1)                              # (N,)
            s2 = (wh * self.a2).sum(-1)                              # (N,)
            score = self.leaky(s1[e_dst] + s2[e_src])               # (E,)
            alpha = scatter_softmax(score, e_dst, n)                # (E,)
            if self.dropout > 0 and self.training:
                alpha = F.dropout(alpha, p=self.dropout)
            # weighted sum in edge-chunks so peak memory is (chunk x out), not (E x out)
            g2 = torch.zeros_like(g1)
            chunk = 500_000
            for s in range(0, e_src.shape[0], chunk):
                sl = slice(s, s + chunk)
                g2 = g2.index_add(0, e_dst[sl], alpha[sl].unsqueeze(-1) * wh[e_src[sl]])
        else:
            g2 = torch.zeros_like(g1)
        gate = torch.sigmoid(self.gate(g1))            # gating is the only non-linearity
        return gate * g1 + (1.0 - gate) * g2           # linear propagation (no ReLU)


# --------------------------------------------------------------------------- #
#  AliNet
# --------------------------------------------------------------------------- #
class AliNet(nn.Module):
    def __init__(self, num_entities, adj1, two_hop, embed_dim=300,
                 layer_dims=(300, 300), dropout=0.0, init="xavier",
                 normalize_embeddings=True, num_relations=0):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.feat_dropout = dropout
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        # NB: xavier on a (num_entities, dim) table gives a tiny range and makes all
        # entities nearly identical (no discrimination => no generalisation). Init the
        # entity table with unit-scale normal vectors so they are spread on the sphere.
        nn.init.normal_(self.ent_emb.weight, std=1.0)
        with torch.no_grad():
            self.ent_emb.weight.data = F.normalize(self.ent_emb.weight.data, dim=-1)

        dims = [embed_dim] + list(layer_dims)
        self.layers = nn.ModuleList(
            [AliNetLayer(dims[i], dims[i + 1], dropout=dropout) for i in range(len(layer_dims))]
        )
        # graph structures are registered as buffers (move with .to(device), not trained)
        self.register_buffer("_adj1", adj1)
        self.register_buffer("e_dst", two_hop[0])
        self.register_buffer("e_src", two_hop[1])
        # JK = concat of the LAYER outputs only (NOT the raw input embedding).
        # Including the raw per-entity embedding lets the model memorise seeds via
        # their own identity and never generalise; the alignment signal must come
        # from neighbourhood aggregation, so we drop it from the representation.
        self.out_dim = sum(layer_dims)

        # relation embeddings live in the representation space (out_dim) for the
        # relation-aware (TransE-style) loss that anchors EVERY entity structurally.
        self.rel_emb = nn.Embedding(num_relations, self.out_dim) if num_relations else None
        if self.rel_emb is not None:
            nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward_all(self) -> torch.Tensor:
        """Compute the JK-concatenated (layer outputs) representation of ALL entities."""
        h = self.ent_emb.weight
        reps = []
        for layer in self.layers:
            if self.feat_dropout > 0 and self.training:
                h = F.dropout(h, p=self.feat_dropout)
            h = layer(h, self._adj1, self.e_dst, self.e_src)
            reps.append(h)
        z = torch.cat(reps, dim=-1)                    # (N, out_dim)
        return F.normalize(z, dim=-1) if self.normalize_embeddings else z


# --------------------------------------------------------------------------- #
#  Loss (limit-based contrastive on precomputed representations)
# --------------------------------------------------------------------------- #
def alinet_align_loss(z, e1, e2, neg_l, neg_r, margin):
    """**Margin-ranking** alignment loss on rows of a precomputed ``z`` (AliNet/GCN-Align).

    ``z`` is the full-graph representation (so the GNN runs once per step). All
    index tensors share the same length (``e1`` / ``e2`` are repeated ``neg`` times
    by the trainer to align with ``neg_r`` / ``neg_l``). A *relative* margin keeps a
    gradient flowing until each negative is at least ``margin`` farther than the
    gold pair. Unlike a limit-based loss it does not saturate to zero after the
    few seed pairs are separated, so the GNN keeps learning generalisable structure.

        L = mean( relu( margin + d(x,y) - d(x, y_neg) ) )   (plus symmetric left corruption)
    """
    d_pos = torch.norm(z[e1] - z[e2], p=2, dim=-1)
    d_neg_r = torch.norm(z[e1] - z[neg_r], p=2, dim=-1)     # corrupt right (KG2)
    d_neg_l = torch.norm(z[neg_l] - z[e2], p=2, dim=-1)     # corrupt left (KG1)
    loss_r = F.relu(margin + d_pos - d_neg_r).mean()
    loss_l = F.relu(margin + d_pos - d_neg_l).mean()
    return 0.5 * (loss_r + loss_l)


def alinet_relation_loss(z, rel_emb, pos, neg_t, margin):
    """Relation-aware (TransE-style) loss on the GNN representations ``z``.

    For a triple ``(h, r, t)`` the score is ``||z_h + r_r - z_t||`` with learnable
    relation vectors ``r_r``. A margin-ranking objective against tail-corrupted
    triples shapes ``z`` so that EVERY entity (not just the 11% seeds) gets a
    structural signal, the missing anchor that caps a pure entity-alignment GNN.
    This is AliNet's relation-aware component.
    """
    h = z[pos[:, 0]]
    r = rel_emb(pos[:, 1])
    t = z[pos[:, 2]]
    pos_s = torch.norm(h + r - t, p=2, dim=-1)
    neg_s = torch.norm(h + r - z[neg_t], p=2, dim=-1)
    return F.relu(margin + pos_s - neg_s).mean()


def alinet_limit_loss(z, e1, e2, neg_l, neg_r, pos_margin, neg_margin, neg_weight=1.0):
    """**Limit-based** (absolute-margin) alignment loss on rows of ``z``.

    Pulls aligned pairs below ``pos_margin`` and pushes negatives above
    ``neg_margin``. Unlike margin-ranking, the negative push is **bounded** (it
    stops once a negative is far enough), so it does not over-scatter the tail
    when combined with hard (epsilon-truncated) negatives, which is what wrecks the
    ranking loss here. Same recipe that worked for BootEA.
    """
    d_pos = torch.norm(z[e1] - z[e2], p=2, dim=-1)
    d_neg_r = torch.norm(z[e1] - z[neg_r], p=2, dim=-1)
    d_neg_l = torch.norm(z[neg_l] - z[e2], p=2, dim=-1)
    l_pos = F.relu(d_pos - pos_margin).mean()
    l_neg = 0.5 * (F.relu(neg_margin - d_neg_r).mean() + F.relu(neg_margin - d_neg_l).mean())
    return l_pos + neg_weight * l_neg
