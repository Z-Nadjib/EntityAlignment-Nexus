"""The DGMC model - Deep Graph Matching Consensus (Fey et al., ICLR 2020).

Reference: "Deep Graph Matching Consensus", https://arxiv.org/abs/2001.09621
Official PyTorch-Geometric implementation: github.com/rusty1s/deep-graph-matching-consensus

DGMC aligns the nodes of two graphs in **two stages**:

1. **Local feature matching** (``psi_1``): a GNN embeds the nodes of each graph;
   the initial soft correspondence is ``S0 = softmax(top_k(h_s @ h_t^T))`` - a
   *sparse* ranking keeping only the ``k`` best target candidates per source node.
2. **Neighbourhood consensus** (``psi_2`` + an MLP): for ``num_steps`` iterations,
   random node "colourings" ``r_s`` are passed through the current correspondence
   to the target graph (``r_t = S^T r_s``), both are diffused by a second GNN, and
   the per-candidate disagreement ``D = psi_2(r_s) - psi_2(r_t)`` drives an additive
   update ``S_hat += MLP(D)``. This rewards correspondences whose neighbourhoods
   *agree*, propagating a matching consensus across local structure.

This is a from-scratch, dependency-free (no torch-geometric / torch-scatter)
re-implementation specialised to a **single graph pair** (batch size 1) - the
DBP15K entity-alignment setting - using the *sparse* top-k variant the paper uses
for large KGs. ``RelConv`` is the relation-agnostic GNN of the official code:
``out = root(x) + mean_{j->i} lin1(x_j) + mean_{i->j} lin2(x_j)``, realised with
two pre-built row-normalised adjacencies.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class RelConv(nn.Module):
    """Directed mean-aggregation conv (relation-agnostic), official RelConv.

    Needs two row-normalised sparse adjacencies: ``adj_in @ X`` averages incoming
    neighbours, ``adj_out @ X`` averages outgoing neighbours.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.root = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, adj_in, adj_out):
        return self.root(x) + torch.sparse.mm(adj_in, self.lin1(x)) \
                            + torch.sparse.mm(adj_out, self.lin2(x))


class RelCNN(nn.Module):
    """Stack of ``RelConv`` layers with ReLU/dropout and a JK-concat + final linear."""

    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        c = in_channels
        for _ in range(num_layers):
            self.convs.append(RelConv(c, out_channels))
            self.batch_norms.append(nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity())
            c = out_channels

        cat_dim = in_channels + num_layers * out_channels if cat else out_channels
        if lin:
            self.out_channels = out_channels
            self.final = nn.Linear(cat_dim, out_channels)
        else:
            self.out_channels = cat_dim
            self.final = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            if hasattr(bn, "reset_parameters"):
                bn.reset_parameters()
        if self.final is not None:
            self.final.reset_parameters()

    def forward(self, x, adj_in, adj_out):
        xs = [x]
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(xs[-1], adj_in, adj_out)
            h = bn(F.relu(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            xs.append(h)
        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        return self.final(x) if self.final is not None else x


class DGMC(nn.Module):
    """Deep Graph Matching Consensus (single graph-pair, sparse top-k variant)."""

    def __init__(self, psi_1: RelCNN, psi_2: RelCNN, num_steps: int, k: int = 10,
                 detach: bool = False, normalize: bool = True, temperature: float = 30.0):
        super().__init__()
        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        # L2-normalise the psi_1 embeddings and score with a temperature: summed
        # word-embedding features have wildly varying norms, so a raw inner product
        # under-uses them. Cosine plus temperature recovers the paper's
        # initial-matching quality (S_0 around 0.68 on zh_en).
        self.normalize = normalize
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(psi_2.out_channels, psi_2.out_channels),
            nn.ReLU(),
            nn.Linear(psi_2.out_channels, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        self.psi_2.reset_parameters()
        for m in self.mlp:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _soft(self, S_hat):
        """Temperature-scaled softmax over the candidate dimension.

        ``S_hat`` is kept at cosine scale so the additive consensus updates
        ``mlp(D)`` stay comparable to it; the temperature sharpens the softmax.
        """
        return (self.temperature * S_hat).softmax(dim=-1)

    @staticmethod
    def _include_gt(S_idx, y):
        """Ensure each ground-truth target is present in the candidate set ``S_idx``."""
        row, col = y[0], y[1]
        present = (S_idx[row] == col.view(-1, 1)).any(dim=-1)
        miss = ~present
        if miss.any():
            S_idx[row[miss], -1] = col[miss]            # overwrite last slot
        return S_idx

    def forward(self, x_s, adj_in_s, adj_out_s, x_t, adj_in_t, adj_out_t, y=None):
        h_s = self.psi_1(x_s, adj_in_s, adj_out_s)
        h_t = self.psi_1(x_t, adj_in_t, adj_out_t)
        if self.detach:
            h_s, h_t = h_s.detach(), h_t.detach()
        if self.normalize:
            h_s, h_t = F.normalize(h_s, dim=-1), F.normalize(h_t, dim=-1)

        N_s, N_t = h_s.size(0), h_t.size(0)
        R_in, R_out = self.psi_2.in_channels, self.psi_2.out_channels

        # ---- initial sparse correspondence: top-k by (cosine) similarity ----
        S_idx = (h_s @ h_t.t()).topk(self.k, dim=1)[1]      # (N_s, k)
        if self.training and y is not None:
            rnd = torch.randint(N_t, (N_s, self.k), device=S_idx.device)
            S_idx = torch.cat([S_idx, rnd], dim=-1)
            S_idx = self._include_gt(S_idx, y)
        k = S_idx.size(-1)

        S_hat = (h_s.unsqueeze(1) * h_t[S_idx]).sum(-1)     # (N_s, k) cosine scale
        S_0 = self._soft(S_hat)

        # ---- neighbourhood consensus refinement ----
        for _ in range(self.num_steps):
            S = self._soft(S_hat)                           # (N_s, k)
            r_s = torch.randn((N_s, R_in), dtype=h_s.dtype, device=h_s.device)
            # r_t[c] = sum_{i,slot : S_idx[i,slot]=c} S[i,slot] * r_s[i]
            contrib = (r_s.unsqueeze(1) * S.unsqueeze(-1)).reshape(-1, R_in)
            r_t = torch.zeros((N_t, R_in), dtype=h_s.dtype, device=h_s.device)
            r_t.index_add_(0, S_idx.reshape(-1), contrib)

            o_s = self.psi_2(r_s, adj_in_s, adj_out_s)
            o_t = self.psi_2(r_t, adj_in_t, adj_out_t)
            D = o_s.unsqueeze(1) - o_t[S_idx]               # (N_s, k, R_out)
            S_hat = S_hat + self.mlp(D).squeeze(-1)

        S_L = self._soft(S_hat)
        return S_idx, S_0, S_L

    # ----- loss / metrics on the sparse correspondence (idx, val) ----- #
    @staticmethod
    def loss(S_idx, S_val, y):
        """Negative log-likelihood of the ground-truth target within the candidates."""
        row, col = y[0], y[1]
        mask = S_idx[row] == col.view(-1, 1)                # (M, k)
        val = (S_val[row] * mask).sum(dim=-1)               # prob mass on gt (0 if absent)
        return -torch.log(val + EPS).mean()

    @staticmethod
    @torch.no_grad()
    def hits_at_k(k, S_idx, S_val, y):
        row, col = y[0], y[1]
        kk = min(k, S_val.size(-1))
        perm = S_val[row].argsort(dim=-1, descending=True)[:, :kk]
        pred = torch.gather(S_idx[row], -1, perm)           # (M, kk)
        correct = (pred == col.view(-1, 1)).any(dim=-1).sum().item()
        return correct / y.size(1)

    @staticmethod
    @torch.no_grad()
    def mrr(S_idx, S_val, y):
        """Mean reciprocal rank of the gold target within the top-k candidates.

        Candidates outside the sparse top-k count as rank infinity (rr = 0).
        """
        row, col = y[0], y[1]
        order = S_val[row].argsort(dim=-1, descending=True)  # (M, k) by score
        ranked = torch.gather(S_idx[row], -1, order)         # (M, k) target ids, best first
        match = ranked == col.view(-1, 1)                    # (M, k)
        has = match.any(dim=-1)                              # gold present among candidates?
        rank = match.float().argmax(dim=-1) + 1              # 1-based rank (first True)
        rr = torch.where(has, 1.0 / rank.float(), torch.zeros_like(rank, dtype=torch.float))
        return rr.mean().item()
