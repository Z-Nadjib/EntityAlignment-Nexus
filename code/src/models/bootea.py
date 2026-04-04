"""The BootEA model.

BootEA (Sun, Hu, Zhang, Qu, "Bootstrapping Entity Alignment with Knowledge
Graph Embedding", IJCAI 2018) learns alignment-oriented KG embeddings:

1. Alignment-oriented embedding (AlignE). A TransE energy
   ``f(h, r, t) = ||h + r - t||`` trained with a limit-based objective
   (absolute margins) rather than a relative margin:

       O_e = sum_{pos triples} relu( f - gamma1 )  +  mu * sum_{neg triples} relu( gamma2 - f )

   Entity embeddings are constrained to the unit sphere (L2-normalised);
   relation embeddings are free. Negatives use epsilon-truncated sampling
   (corrupt with one of the entity's nearest same-KG neighbours).

2. Alignment by swapping. For a labelled pair ``(e1, e2)`` BootEA generates
   *aligned triples* by swapping ``e1`` and ``e2`` in each other's triples
   (handled in the trainer / data module). Because the swapped entities then
   share relational contexts, their embeddings are pulled together. This is
   BootEA's core alignment mechanism, complemented here by a light limit-based
   pull on the labelled pairs.

3. **Bootstrapping.** An editable, recomputed maximum-weight (1-to-1) matching
   over the unlabelled pool proposes new alignments each round (in the trainer).

The representation used for alignment and evaluation is simply the
(L2-normalised) entity embedding. There is no neighbourhood aggregation here
(that is what distinguishes BootEA from NAEA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BootEA(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        init: str = "xavier",
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.dim = embed_dim
        self.normalize_embeddings = normalize_embeddings
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        self._init_params(init)

    def _init_params(self, init: str):
        if init == "xavier":
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            nn.init.normal_(self.ent_emb.weight, std=0.02)
            nn.init.normal_(self.rel_emb.weight, std=0.02)

    # ------------------------------------------------------------------ #
    #  Embeddings
    # ------------------------------------------------------------------ #
    def _ent(self, idx):
        e = self.ent_emb(idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    def _rel(self, idx):
        # relations are left unconstrained in AlignE
        return self.rel_emb(idx)

    def triple_score(self, triples: torch.Tensor) -> torch.Tensor:
        """``||h + r - t||_2`` for a batch of (h, r, t). Lower = more plausible."""
        h = self._ent(triples[:, 0])
        r = self._rel(triples[:, 1])
        t = self._ent(triples[:, 2])
        return torch.norm(h + r - t, p=2, dim=-1)

    def encode(self, ent_idx: torch.Tensor) -> torch.Tensor:
        """Representation used for alignment / evaluation: the (normalised) entity embedding."""
        e = self.ent_emb(ent_idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    @torch.no_grad()
    def encode_all(self, ent_ids: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        self.eval()
        outs = [self.encode(ent_ids[s:s + chunk]) for s in range(0, len(ent_ids), chunk)]
        return torch.cat(outs, 0)


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #
def limit_based_triple_loss(pos_score, neg_score, pos_margin: float,
                            neg_margin: float, neg_weight: float) -> torch.Tensor:
    """AlignE limit-based objective (absolute margins).

      * positives pulled below ``pos_margin`` : ``relu( f - gamma1 )``
      * negatives pushed above ``neg_margin`` : ``relu( gamma2 - f )``
    """
    l_pos = F.relu(pos_score - pos_margin).mean()
    l_neg = F.relu(neg_margin - neg_score).mean()
    return l_pos + neg_weight * l_neg


def alignment_pull_loss(model: BootEA, e1, e2, pos_margin: float) -> torch.Tensor:
    """Light limit-based pull on labelled aligned pairs (complements swapping).

    Pulls ``d(z_e1, z_e2)`` below ``pos_margin`` (with normalised embeddings the
    distance is in ``[0, 2]``).
    """
    z1 = model.encode(e1)
    z2 = model.encode(e2)
    d = torch.norm(z1 - z2, p=2, dim=-1)
    return F.relu(d - pos_margin).mean()
