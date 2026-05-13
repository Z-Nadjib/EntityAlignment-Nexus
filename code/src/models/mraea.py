"""The MRAEA model - Meta Relation Aware Entity Alignment (Mao et al., WSDM 2020).

Reference: "MRAEA: An Efficient and Robust Entity Alignment Approach for
Cross-lingual Knowledge Graph", https://dl.acm.org/doi/10.1145/3336191.3371804
Official Keras implementation: github.com/MaoXinn/MRAEA

MRAEA learns entity embeddings with a **meta-relation-aware graph attention**: a
node attends over its incoming/outgoing neighbours, and the attention logit of an
edge also depends on the **meta-relation** carried by that edge (a relation and
its inverse get distinct embeddings). Concretely, each entity starts from

    h^(0) = relu( [ mean_{neigh+self} ent_emb  ||  mean_{rel} rel_emb ] )

(the relation channel makes the representation relation-aware), and then ``depth``
shared graph-attention steps refine it; the attention weight of edge ``(i, j)`` is

    softmax_j  leakyrelu( a_r . rel(i,j) + a_s . h_i + a_n . h_j )

The outputs of every step are concatenated (JK). Trained with an L1 margin loss
against random negatives; aligned at test time by cosine similarity.

From-scratch PyTorch port (the official code is Keras/TF), specialised to the
DBP15K single graph-pair setting. The sparse structures come from
``data.build_mraea_graph``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MRAEA(nn.Module):
    def __init__(self, graph: dict, node_hidden=100, rel_hidden=100,
                 depth=2, attn_heads=2, dropout=0.3):
        super().__init__()
        self.N = graph["num_nodes"]
        self.R = graph["num_rels"]
        self.depth = depth
        self.heads = attn_heads
        self.dropout = dropout
        self.ent_F = node_hidden + rel_hidden

        self.ent_emb = nn.Embedding(self.N, node_hidden)
        self.rel_emb = nn.Embedding(self.R, rel_hidden)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # per-head attention kernels (shared across the `depth` steps), as in the
        # official layer: a_self, a_neigh over the entity feature (ent_F), a_rel
        # over the relation embedding (rel_hidden).
        self.k_self = nn.Parameter(torch.empty(attn_heads, self.ent_F, 1))
        self.k_neigh = nn.Parameter(torch.empty(attn_heads, self.ent_F, 1))
        self.k_rel = nn.Parameter(torch.empty(attn_heads, rel_hidden, 1))
        for p in (self.k_self, self.k_neigh, self.k_rel):
            nn.init.xavier_uniform_(p)

        # ---- static graph structures (registered so .to(device) moves them) ----
        self.register_buffer("adj_index", graph["adj_index"])           # (2, E)
        self.register_buffer("edge_rel_index", graph["edge_rel_index"]) # (2, M)
        self.register_buffer("edge_rel_val", graph["edge_rel_val"])     # (M,)

        # precompute the (static) uniform row-normalised aggregators: softmax over
        # a row of ones == 1 / row-degree.
        ei = graph["ent_adj_index"]
        ev = self._row_uniform(ei[0], self.N)
        self.register_buffer("ent_adj_index", ei)
        self.register_buffer("ent_adj_val", ev)
        ri = graph["rel_adj_index"]
        rv = self._row_uniform(ri[0], self.N)
        self.register_buffer("rel_adj_index", ri)
        self.register_buffer("rel_adj_val", rv)

    @staticmethod
    def _row_uniform(rows, n):
        deg = torch.zeros(n).index_add_(0, rows, torch.ones(rows.numel()))
        return 1.0 / deg[rows].clamp(min=1.0)

    def _sp(self, index, val, shape):
        return torch.sparse_coo_tensor(index, val, shape, device=index.device).coalesce()

    def forward(self):
        ent_emb, rel_emb = self.ent_emb.weight, self.rel_emb.weight
        N, R2 = self.N, self.R
        E = self.adj_index.size(1)

        ent_adj = self._sp(self.ent_adj_index, self.ent_adj_val, (N, N))
        rel_adj = self._sp(self.rel_adj_index, self.rel_adj_val, (N, R2))
        edge_rel = self._sp(self.edge_rel_index, self.edge_rel_val, (E, R2))

        ent_features = torch.sparse.mm(ent_adj, ent_emb)        # (N, node_hidden)
        rel_features = torch.sparse.mm(rel_adj, rel_emb)        # (N, rel_hidden)
        features = F.relu(torch.cat([ent_features, rel_features], dim=-1))   # (N, ent_F)
        outputs = [features]

        src, dst = self.adj_index[0], self.adj_index[1]
        for _ in range(self.depth):
            head_feats = []
            for head in range(self.heads):
                # per-edge meta-relation attention contribution
                rel_proj = rel_emb @ self.k_rel[head]                       # (R2, 1)
                rel_score = torch.sparse.mm(edge_rel, rel_proj).squeeze(-1)  # (E,)
                a_self = (features @ self.k_self[head]).squeeze(-1)          # (N,)
                a_neigh = (features @ self.k_neigh[head]).squeeze(-1)        # (N,)
                att_val = rel_score + a_self[src] + a_neigh[dst]            # (E,)
                att_val = F.leaky_relu(att_val)
                att = self._sp(self.adj_index, att_val, (N, N))
                att = torch.sparse.softmax(att, dim=1)
                head_feats.append(torch.sparse.mm(att, features))           # (N, ent_F)
            features = F.relu(torch.stack(head_feats, 0).mean(0))
            outputs.append(features)

        out = torch.cat(outputs, dim=-1)                        # (N, ent_F*(depth+1))
        return F.dropout(out, p=self.dropout, training=self.training)


def mraea_align_loss(emb, quad, gamma):
    """L1 margin loss: relu(g + d(l,r) - d(l,r-)) + relu(g + d(l,r) - d(l-,r)).

    ``quad`` is ``[B, 4]`` of indices ``[l, r, neg_for_l, neg_for_r]``; distances
    are L1 over the (unnormalised) embeddings.
    """
    l = emb[quad[:, 0]]; r = emb[quad[:, 1]]
    nl = emb[quad[:, 2]]; nr = emb[quad[:, 3]]

    def d(a, b):
        return (a - b).abs().sum(-1)

    pos = d(l, r)
    loss = F.relu(gamma + pos - d(l, nr)) + F.relu(gamma + pos - d(nl, r))
    return loss.sum() / l.size(0)
