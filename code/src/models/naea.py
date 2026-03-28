"""The NAEA model.

NAEA (Zhu et al., IJCAI 2019) learns entity/relation embeddings with two
complementary signals:

1. Relation-level (knowledge) representation: a classic TransE energy
   ``f(h, r, t) = ||h + r - t||`` trained with a margin ranking loss. This
   captures the relational structure of each KG.

2. Neighbourhood-aware attentional representation: for an entity ``e`` we
   aggregate its neighbours ``(r_k, e_j)`` with a GAT-style attention. Each
   neighbour contributes a translation-consistent message
   ``m = e_j + sign * r_k`` (``sign = +1`` for in-edges, ``-1`` for out-edges,
   so ``m`` reconstructs ``e``). A shared linear map ``W`` and an attention
   vector ``a`` produce per-neighbour weights::

        alpha_k = softmax_k( LeakyReLU( a^T [ W e || W m_k ] ) )
        e_hat   = sigmoid( sum_k alpha_k * W m_k )

   The representation used for alignment and evaluation is the joint
   vector ``z = e + e_hat`` (self embedding plus neighbourhood embedding),
   L2-normalised.

The alignment loss pulls seed-aligned entities' ``z`` vectors together and
pushes random ones apart (margin ranking). Bootstrapping (handled in the
trainer) periodically adds high-confidence predicted pairs to the seed set.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NAEA(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        neigh_ent: torch.Tensor,
        neigh_rel: torch.Tensor,
        neigh_sign: torch.Tensor,
        neigh_mask: torch.Tensor,
        attn_heads: int = 1,
        attn_dropout: float = 0.0,
        init: str = "xavier",
        normalize_embeddings: bool = True,
        neighbor_message: str = "trans",
    ):
        super().__init__()
        self.dim = embed_dim
        self.normalize_embeddings = normalize_embeddings
        self.neighbor_message = neighbor_message
        self.attn_dropout = attn_dropout

        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)

        # attention parameters (single shared head by default; multi-head averaged)
        self.heads = max(1, attn_heads)
        self.W = nn.Linear(embed_dim, embed_dim * self.heads, bias=False)
        self.a_self = nn.Parameter(torch.zeros(self.heads, embed_dim))
        self.a_neigh = nn.Parameter(torch.zeros(self.heads, embed_dim))
        self.leaky = nn.LeakyReLU(0.2)

        # neighbour structure stored as (non-trainable) buffers so they move
        # with .to(device) and are saved/restored with the module.
        self.register_buffer("neigh_ent", neigh_ent)
        self.register_buffer("neigh_rel", neigh_rel)
        self.register_buffer("neigh_sign", neigh_sign)
        self.register_buffer("neigh_mask", neigh_mask)

        self._init_params(init)

    def _init_params(self, init: str):
        if init == "xavier":
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            nn.init.normal_(self.ent_emb.weight, std=0.02)
            nn.init.normal_(self.rel_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_self)
        nn.init.xavier_uniform_(self.a_neigh)

    # ------------------------------------------------------------------ #
    #  Relation-level (TransE) energy
    # ------------------------------------------------------------------ #
    def _ent(self, idx):
        e = self.ent_emb(idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    def _rel(self, idx):
        r = self.rel_emb(idx)
        return F.normalize(r, dim=-1) if self.normalize_embeddings else r

    def transe_score(self, triples: torch.Tensor) -> torch.Tensor:
        """``||h + r - t||_2`` for a batch of (h, r, t). Lower = more plausible."""
        h = self._ent(triples[:, 0])
        r = self._rel(triples[:, 1])
        t = self._ent(triples[:, 2])
        return torch.norm(h + r - t, p=2, dim=-1)

    # ------------------------------------------------------------------ #
    #  Neighbourhood-aware attentional encoder
    # ------------------------------------------------------------------ #
    def encode(self, ent_idx: torch.Tensor) -> torch.Tensor:
        """Joint neighbour-aware representation ``z = e + e_hat`` for ``ent_idx``.

        ``ent_idx`` : LongTensor of arbitrary shape ``(...)``.
        returns      : Tensor ``(..., dim)`` (L2-normalised if configured).
        """
        flat = ent_idx.reshape(-1)
        B = flat.shape[0]
        H, d = self.heads, self.dim

        e = self.ent_emb(flat)                                   # (B, d)
        ne = self.neigh_ent[flat]                                # (B, K)
        nr = self.neigh_rel[flat]                                # (B, K)
        sg = self.neigh_sign[flat].unsqueeze(-1)                 # (B, K, 1)
        msk = self.neigh_mask[flat]                              # (B, K)

        ent_n = self.ent_emb(ne)                                 # (B, K, d)
        rel_n = self.rel_emb(nr)                                 # (B, K, d)
        if self.neighbor_message == "trans":
            msg = ent_n + sg * rel_n                             # translation-consistent
        else:
            msg = ent_n                                          # entity-only neighbour

        Wself = self.W(e).view(B, H, d)                          # (B, H, d)
        Wmsg = self.W(msg).view(B, -1, H, d)                     # (B, K, H, d)
        K = Wmsg.shape[1]

        # attention logits: a^T[W e || W m]  -> split into self/neigh parts
        logit_self = (Wself * self.a_self).sum(-1).unsqueeze(1)  # (B, 1, H)
        logit_neigh = (Wmsg * self.a_neigh).sum(-1)              # (B, K, H)
        logit = self.leaky(logit_self + logit_neigh)             # (B, K, H)

        logit = logit.masked_fill(~msk.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(logit, dim=1)                      # (B, K, H)
        # entities with zero neighbours -> softmax of all -inf == nan; zero them
        alpha = torch.nan_to_num(alpha, nan=0.0)
        if self.attn_dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.attn_dropout)

        neigh_repr = (alpha.unsqueeze(-1) * Wmsg).sum(1)         # (B, H, d)
        neigh_repr = torch.tanh(neigh_repr).mean(1)              # average heads -> (B, d)

        z = e + neigh_repr
        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1)
        return z.view(*ent_idx.shape, d)

    @torch.no_grad()
    def encode_all(self, ent_ids: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Encode a (possibly large) set of entities in chunks; eval helper."""
        self.eval()
        outs = []
        for s in range(0, len(ent_ids), chunk):
            outs.append(self.encode(ent_ids[s:s + chunk]))
        return torch.cat(outs, 0)


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #
def margin_ranking_loss(pos_score, neg_score, margin: float) -> torch.Tensor:
    """``mean( relu(margin + pos - neg) )`` where *lower score = better*.

    ``pos_score`` : (B,)        ``neg_score`` : (B*neg,) reshaped to (B, neg).
    """
    neg_score = neg_score.view(pos_score.shape[0], -1)
    loss = F.relu(margin + pos_score.unsqueeze(1) - neg_score)
    return loss.mean()


def alignment_loss(model: NAEA, e1, e2, neg_l, neg_r,
                   pos_margin: float, neg_margin: float, neg_weight: float = 1.0) -> torch.Tensor:
    """Limit-based (absolute-margin) alignment loss, BootEA-style.

    With L2-normalised ``z`` the distance ``d`` lies in ``[0, 2]``, so a relative
    margin can never saturate and collapses the space (especially with hard
    negatives). The limit-based objective instead sets absolute targets that
    saturate:

      * pull positives below ``pos_margin``  : ``relu( d(z_e1,z_e2) - gamma1 )``
      * push negatives above ``neg_margin``  : ``relu( gamma2 - d(z_e1,z_negR) )`` (plus the left side)

    Once a negative is far enough (``d >= gamma2``) its gradient is zero, so there
    is no runaway repulsion, even for nearest-neighbour (hard) negatives.
    """
    z1 = model.encode(e1)
    z2 = model.encode(e2)
    zr = model.encode(neg_r)
    zl = model.encode(neg_l)
    d_pos = torch.norm(z1 - z2, p=2, dim=-1)                     # (B*neg,)
    d_neg_r = torch.norm(z1 - zr, p=2, dim=-1)
    d_neg_l = torch.norm(zl - z2, p=2, dim=-1)
    l_pos = F.relu(d_pos - pos_margin).mean()
    l_neg = 0.5 * (F.relu(neg_margin - d_neg_r).mean() + F.relu(neg_margin - d_neg_l).mean())
    return l_pos + neg_weight * l_neg
