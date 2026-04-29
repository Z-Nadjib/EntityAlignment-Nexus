"""The JAPE model.

JAPE (Sun, Hu, Li - "Cross-lingual Entity Alignment via Joint Attribute-Preserving
Embedding", ISWC 2017) - the paper that introduced DBP15K.

* **Structure Embedding (SE)** : a TransE energy ``f(h, r, t) = ||h + r - t||`` over
  the triples of BOTH KGs put in ONE space. Seed alignments are encoded by giving
  aligned seed entities the **same id** (merged), so the two graphs are bridged and
  TransE propagates the alignment to the test entities. Trained with a margin loss
  and negative sampling; entity embeddings are L2-normalised.
* **Attribute Embedding (AE)** : entities are described by a (cross-KG, shared-vocab)
  bag of attributes; the attribute similarity refines/augments SE. Here the AE
  cosine similarity is combined with the SE similarity at alignment time
  (``sim = beta*SE + (1-beta)*AE``), which captures JAPE's attribute-preserving signal.

The representation used for alignment / evaluation is the (L2-normalised) entity
embedding; the trainer adds the AE similarity on top.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JAPE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=75,
                 init="xavier", normalize_embeddings=True):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        if init == "xavier":
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            nn.init.normal_(self.ent_emb.weight, std=0.02)
            nn.init.normal_(self.rel_emb.weight, std=0.02)

    def _ent(self, idx):
        e = self.ent_emb(idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    def triple_score(self, triples):
        """``||h + r - t||_2`` (L2-normalised entities, free relations)."""
        h = self._ent(triples[:, 0])
        r = self.rel_emb(triples[:, 1])
        t = self._ent(triples[:, 2])
        return torch.norm(h + r - t, p=2, dim=-1)

    def encode(self, idx):
        return self._ent(idx)

    @torch.no_grad()
    def encode_all(self, ids, chunk=8192):
        self.eval()
        return torch.cat([self.encode(ids[s:s + chunk]) for s in range(0, len(ids), chunk)], 0)


def jape_se_loss(model, pos, neg, margin):
    """TransE margin-ranking loss for SE (negatives = corrupted triples)."""
    pos_s = model.triple_score(pos)
    neg_s = model.triple_score(neg)
    pos_s = pos_s.repeat(neg.shape[0] // pos.shape[0]) if neg.shape[0] != pos.shape[0] else pos_s
    return F.relu(margin + pos_s - neg_s).mean()
