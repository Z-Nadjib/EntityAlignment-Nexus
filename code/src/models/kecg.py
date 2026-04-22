"""The KECG model.

KECG (Li, Cao, Hou, Shi, Li, Chua - "Semi-supervised Entity Alignment via Joint
Knowledge Embedding Model and Cross-graph Model", EMNLP 2019) jointly trains two
models that **share the entity embedding table**:

1. **Cross-Graph model (CG)** - a multi-head **GAT with diagonal weights**, shared
   by both KGs (one combined graph). Aligned seed entities are pulled together
   (triplet margin loss) using the GAT output embeddings, with nearest-neighbour
   (NNS) hard negatives. The attention lets the shared GAT ignore unimportant
   neighbours and transfer structure across graphs.
2. **Knowledge-Embedding model (KE)** - a TransE energy ``||h + r - t||`` over the
   triples that shapes the *same* entity embeddings with relational constraints.

Training alternates the two objectives (CG on even epochs, KE on odd). Evaluation
uses the GAT output embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Scatter helpers (sparse edge softmax / sum, no torch_scatter dependency)
# --------------------------------------------------------------------------- #
def scatter_softmax(scores, index, n):
    mx = scores.new_full((n,), float("-inf")).index_reduce_(0, index, scores, "amax", include_self=True)
    mx = torch.nan_to_num(mx, neginf=0.0)
    s = (scores - mx[index]).exp()
    denom = torch.zeros(n, device=scores.device, dtype=scores.dtype).index_add_(0, index, s)
    return s / (denom[index] + 1e-16)


def scatter_add(src, index, n):
    out = torch.zeros((n, src.shape[1]), device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


# --------------------------------------------------------------------------- #
#  Multi-head GAT layer with diagonal weights (KECG / GCN-Align style)
# --------------------------------------------------------------------------- #
class DiagGATLayer(nn.Module):
    """GAT layer whose linear map is **diagonal** (element-wise scale per head).

    Diagonal weights keep the layer parameter-light (a vector per head, not a
    full matrix) and work well for entity alignment. Heads are averaged (the
    representation dimension is preserved). Propagation is linear (no ReLU): a
    ReLU between layers destroys the structural signal for EA.
    """

    def __init__(self, dim, n_heads, attn_dropout=0.0, combine="mean"):
        super().__init__()
        self.n_heads = n_heads
        self.combine = combine                                  # "concat" or "mean"
        self.w = nn.Parameter(torch.ones(n_heads, dim))        # diagonal transform per head
        self.a_dst = nn.Parameter(torch.zeros(n_heads, dim))   # attention (centre)
        self.a_src = nn.Parameter(torch.zeros(n_heads, dim))   # attention (neighbour)
        self.leaky = nn.LeakyReLU(0.2)
        self.attn_dropout = attn_dropout
        self.out_dim = dim * n_heads if combine == "concat" else dim
        nn.init.xavier_uniform_(self.a_dst); nn.init.xavier_uniform_(self.a_src)

    def forward(self, h, e_dst, e_src):
        n = h.shape[0]
        outs = []
        for hd in range(self.n_heads):
            g = h * self.w[hd]                                  # (N, dim) diagonal transform
            sd = (g * self.a_dst[hd]).sum(-1)                   # (N,)  centre score
            ss = (g * self.a_src[hd]).sum(-1)                   # (N,)  neighbour score
            score = -self.leaky(sd[e_dst] + ss[e_src])          # (E,)  KECG: softmax(-LeakyReLU)
            alpha = scatter_softmax(score, e_dst, n)            # (E,)
            if self.attn_dropout > 0 and self.training:
                alpha = F.dropout(alpha, p=self.attn_dropout)
            outs.append(scatter_add(alpha.unsqueeze(-1) * g[e_src], e_dst, n))
        if self.combine == "concat":
            return torch.cat(outs, dim=-1)                      # (N, dim*n_heads)
        return torch.stack(outs, 0).mean(0)                     # (N, dim)


# --------------------------------------------------------------------------- #
#  KECG
# --------------------------------------------------------------------------- #
class KECG(nn.Module):
    def __init__(self, num_entities, num_relations, edge_index, embed_dim=128,
                 n_layers=2, n_heads=2, attn_dropout=0.0, normalize_embeddings=False,
                 instance_normalization=False):
        super().__init__()
        import math
        self.normalize_embeddings = normalize_embeddings
        self.instance_normalization = instance_normalization
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        nn.init.normal_(self.ent_emb.weight, std=1.0 / math.sqrt(num_entities))
        nn.init.xavier_uniform_(self.rel_emb.weight)
        if self.instance_normalization:
            # InstanceNorm1d over the node dimension (KECG): standardises each feature
            # across entities; learnable affine. Stabilises the GAT and the dynamics.
            self.norm = nn.InstanceNorm1d(embed_dim, momentum=0.0, affine=True)

        # KECG GAT: diagonal multi-head attention, heads AVERAGED each layer
        # (dimension preserved), ELU between layers, final layer output as the rep.
        self.layers = nn.ModuleList(
            [DiagGATLayer(embed_dim, n_heads, attn_dropout, combine="mean") for _ in range(n_layers)])
        self.register_buffer("e_dst", edge_index[0])
        self.register_buffer("e_src", edge_index[1])
        self.out_dim = embed_dim

    # -- Cross-Graph: GAT output (used for alignment + evaluation) ---------- #
    def forward_all(self):
        # KECG: instance-norm the input (feature-wise over entities), stack GAT
        # layers with ELU between (none after the final), final layer = the rep.
        h = self.ent_emb.weight
        if self.instance_normalization:
            h = self.norm(h.t().unsqueeze(0)).squeeze(0).t()   # (N, d)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h, self.e_dst, self.e_src)
            if i < last:
                h = F.elu(h)
        return F.normalize(h, dim=-1) if self.normalize_embeddings else h

# --------------------------------------------------------------------------- #
#  Losses
# --------------------------------------------------------------------------- #
def kecg_cg_loss(z, e1, e2, neg_l, neg_r, margin):
    """Cross-graph triplet margin loss on GAT embeddings (both directions)."""
    d_pos = torch.norm(z[e1] - z[e2], p=2, dim=-1)
    d_neg_r = torch.norm(z[e1] - z[neg_r], p=2, dim=-1)
    d_neg_l = torch.norm(z[neg_l] - z[e2], p=2, dim=-1)
    return 0.5 * (F.relu(margin + d_pos - d_neg_r).mean() + F.relu(margin + d_pos - d_neg_l).mean())


def kecg_ke_loss(z, rel_emb, pos, neg, margin):
    """TransE margin-ranking loss on the **GAT outputs** (`z`).
    
    NOTE: Replicates a mathematical bug from the original paper's repository:
    `F.normalize(h+r-t, p=2).sum(1)`. Instead of standard L2 distance, it
    normalizes the error vector and sums its components.
    """
    if neg.shape[0] != pos.shape[0]:
        pos = pos.repeat(neg.shape[0] // pos.shape[0], 1)
        
    x_pos = F.normalize(z[pos[:, 0]] + rel_emb(pos[:, 1]) - z[pos[:, 2]], p=2, dim=-1)
    x_neg = F.normalize(z[neg[:, 0]] + rel_emb(neg[:, 1]) - z[neg[:, 2]], p=2, dim=-1)
    
    y = torch.ones(x_pos.size(0), 1, device=z.device)
    return F.margin_ranking_loss(x_pos.sum(1).view(-1, 1), x_neg.sum(1).view(-1, 1), y, margin=margin)

