"""The RREA model - Relational Reflection Entity Alignment (Mao et al., CIKM 2020).

Reference: "Relational Reflection Entity Alignment", https://arxiv.org/abs/2008.07962
Official Keras implementation: github.com/MaoXinn/RREA

RREA is the successor of MRAEA. Its key idea is the relational reflection
operator: when a node ``i`` aggregates a neighbour ``j`` connected by relation
``r``, the neighbour vector is reflected across the hyperplane orthogonal to the
(unit) relation vector ``r_hat``:

    reflect_r(h_j) = h_j - 2 (h_j . r_hat) r_hat

a Householder reflection, a norm- and orthogonality-preserving transform
(unlike MRAEA's additive relation term). The edge attention logit is

    a . [ h_i || reflect_r(h_j) || r_hat ]        (softmax over neighbours, no leakyrelu)

and the node aggregates the reflected neighbours. The shared graph-attention
encoder (``depth`` layers, JK-concat) is run on two initial features, one
built from neighbour entity embeddings and one from relation embeddings - whose
outputs are concatenated. Trained with an L1 margin loss; aligned by CSLS.

From-scratch PyTorch port (the official code is Keras/TF). The sparse graph
structures are exactly MRAEA's ``data.build_mraea_graph``; the loss is the shared
``mraea_align_loss``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RREA(nn.Module):
    def __init__(self, graph: dict, node_hidden=100, depth=2, attn_heads=1, dropout=0.3):
        super().__init__()
        self.N = graph["num_nodes"]
        self.R = graph["num_rels"]
        self.depth = depth
        self.heads = attn_heads
        self.dropout = dropout
        self.F = node_hidden

        self.ent_emb = nn.Embedding(self.N, node_hidden)
        self.rel_emb = nn.Embedding(self.R, node_hidden)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # per-(depth, head) attention kernel over [self || reflected_neigh || rel]
        self.attn = nn.Parameter(torch.empty(depth, attn_heads, 3 * node_hidden, 1))
        nn.init.xavier_uniform_(self.attn)

        self.register_buffer("adj_index", graph["adj_index"])            # (2, E)
        self.register_buffer("edge_rel_index", graph["edge_rel_index"])  # (2, M)
        self.register_buffer("edge_rel_val", graph["edge_rel_val"])      # (M,)
        ei = graph["ent_adj_index"]
        self.register_buffer("ent_adj_index", ei)
        self.register_buffer("ent_adj_val", self._row_uniform(ei[0], self.N))
        ri = graph["rel_adj_index"]
        self.register_buffer("rel_adj_index", ri)
        self.register_buffer("rel_adj_val", self._row_uniform(ri[0], self.N))

    @staticmethod
    def _row_uniform(rows, n):
        deg = torch.zeros(n).index_add_(0, rows, torch.ones(rows.numel()))
        return 1.0 / deg[rows].clamp(min=1.0)

    def _sp(self, index, val, shape):
        return torch.sparse_coo_tensor(index, val, shape, device=index.device).coalesce()

    @staticmethod
    def _segment_softmax(att, src, n):
        """Softmax of edge logits within each source-node group (edge order kept)."""
        gmax = torch.full((n,), float("-inf"), device=att.device)
        gmax = gmax.scatter_reduce(0, src, att, reduce="amax", include_self=True)
        att = (att - gmax[src]).exp()
        gsum = torch.zeros(n, device=att.device).index_add_(0, src, att)
        return att / gsum[src].clamp_min(1e-12)

    def _encode(self, features, rel_emb, edge_rel):
        """Run the relational-reflection attention encoder on one initial feature."""
        src, dst = self.adj_index[0], self.adj_index[1]
        features = F.relu(features)
        outputs = [features]
        for l in range(self.depth):
            head_feats = []
            for head in range(self.heads):
                rels = torch.sparse.mm(edge_rel, rel_emb)          # (E, F) per-edge relation
                rels = F.normalize(rels, dim=1)                    # unit relation vector r_hat
                neigh = features[dst]                              # (E, F)
                slf = features[src]                                # (E, F)
                bias = (neigh * rels).sum(1, keepdim=True) * rels  # (h_j . r_hat) r_hat
                neigh = neigh - 2.0 * bias                         # reflection
                logit = (torch.cat([slf, neigh, rels], dim=-1)
                         @ self.attn[l, head]).squeeze(-1)         # (E,)
                att = self._segment_softmax(logit, src, self.N)
                agg = torch.zeros(self.N, self.F, device=features.device)
                agg.index_add_(0, src, neigh * att.unsqueeze(-1))  # aggregate reflected neighs
                head_feats.append(agg)
            features = F.relu(torch.stack(head_feats, 0).mean(0))
            outputs.append(features)
        return torch.cat(outputs, dim=-1)                          # (N, F*(depth+1))

    def forward(self):
        ent_emb, rel_emb = self.ent_emb.weight, self.rel_emb.weight
        N, R2, E = self.N, self.R, self.adj_index.size(1)
        ent_adj = self._sp(self.ent_adj_index, self.ent_adj_val, (N, N))
        rel_adj = self._sp(self.rel_adj_index, self.rel_adj_val, (N, R2))
        edge_rel = self._sp(self.edge_rel_index, self.edge_rel_val, (E, R2))

        ent_feature = torch.sparse.mm(ent_adj, ent_emb)            # neighbour-entity feature
        rel_feature = torch.sparse.mm(rel_adj, rel_emb)            # relation feature
        out = torch.cat([self._encode(ent_feature, rel_emb, edge_rel),
                         self._encode(rel_feature, rel_emb, edge_rel)], dim=-1)
        return F.dropout(out, p=self.dropout, training=self.training)
