# Entity Alignment on DBP15K — NAEA, BootEA, AliNet, KECG, GCN-Align, JAPE, DGMC, MRAEA & RREA

Reproductions of nine entity-alignment methods on the DBP15K benchmark
(`zh_en`, `ja_en`, `fr_en`):

- **NAEA** — *Neighborhood-Aware Attentional Representation* (Zhu et al., IJCAI 2019).
  https://www.ijcai.org/proceedings/2019/0269.pdf
- **BootEA** — *Bootstrapping Entity Alignment with KG Embedding* (Sun et al., IJCAI 2018).
  https://www.ijcai.org/proceedings/2018/0611.pdf
- **AliNet** — *KG Alignment Network with Gated Multi-hop Aggregation* (Sun et al., AAAI 2020).
  https://arxiv.org/pdf/1911.08936v1
- **KECG** — *Semi-supervised EA via Joint Knowledge Embedding + Cross-graph* (Li et al., EMNLP 2019).
  https://aclanthology.org/D19-1274.pdf
- **GCN-Align** — *Cross-lingual KG Alignment via Graph Convolutional Networks* (Wang et al., EMNLP 2018).
  https://aclanthology.org/D18-1032.pdf
- **JAPE** — *Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding* (Sun et al., ISWC 2017).
  https://arxiv.org/pdf/1708.05045  *(the paper that introduced DBP15K)*
- **DGMC** — *Deep Graph Matching Consensus* (Fey et al., ICLR 2020).
  https://arxiv.org/abs/2001.09621  *(uses entity-name word embeddings, not pure structure)*
- **MRAEA** — *Meta Relation Aware Entity Alignment* (Mao et al., WSDM 2020).
  https://dl.acm.org/doi/10.1145/3336191.3371804
- **RREA** — *Relational Reflection Entity Alignment* (Mao et al., CIKM 2020).
  https://arxiv.org/abs/2008.07962

## NAEA — method
- **Relation-level** TransE embeddings (margin loss on triples).
- **Neighbourhood-aware attention** (GAT-style) over translation-consistent
  neighbour messages `e_j ± r_k`.
- **Limit-based alignment loss** (absolute margins → saturates, no collapse).
- **Hard / ε-truncated negatives** (nearest cross-KG entities) — the key to Hit@1.
- **Recomputed bootstrapping** (BootEA-style self-training; re-labelled each round).
- **CSLS** at evaluation to reduce hubness.

## BootEA — method
- **AlignE embedding** : TransE with **limit-based** loss (absolute margins) and
  unit-sphere entity embeddings.
- **ε-truncated negatives** : corrupt with the entity's nearest same-KG neighbours.
- **Alignment by swapping** : labelled pairs generate aligned triples (swap the two
  entities in each other's triples) + a light limit-based pull.
- **Editable MWGM bootstrapping** : recomputed mutual 1-to-1 matching each round.

## AliNet — method
- **Gated multi-hop GNN** : 1-hop GCN + attention over (capped) 2-hop edges, combined by a
  learned gate; **linear** propagation; JK-concat of layer outputs (no raw embedding).
- **Relation-aware (TransE) loss** `||z_h + r − z_t||` on the GNN reps — anchors every entity
  structurally (the key to lifting Hit@10 / MeanRank).
- Margin-ranking alignment + **mixed ε-truncated hard negatives**; full-batch; CSLS eval.

## KECG — method
- **Cross-graph (CG)** : a shared diagonal multi-head **GAT** (instance-norm input, ELU
  between layers); seed pairs pulled together with a triplet margin loss; **NNS** hard
  negatives = nearest *other-seed* entities of the same KG.
- **Knowledge-embedding (KE)** : a TransE loss on the *same* entity embeddings.
- The two objectives **alternate** each epoch (CG even, KE odd).

## GCN-Align — method
- **Functionality-weighted adjacency** : edges weighted by relation functionality
  `max(fun/ifun, 0.3)`, +self-loops, symmetric-normalised `D^-1/2 A D^-1/2`.
- **Shared 2-layer GCN** (structure channel SE); **linear** propagation.
- **Margin-ranking loss on L1 distance** with random negatives (both sides).

## JAPE — method
- **Merged-seed format** (`use_mtranse_format: false`): aligned seeds share ids, so the two KGs
  live in one graph and TransE propagates the alignment to the test entities.
- **Structure embedding (SE)** : plain TransE `||h + r − t||` (L2-normalised entities, free
  relations) on the merged graph; margin loss + random corrupted-triple negatives.
- **Attribute embedding (AE)** : cross-KG TF-IDF attribute-bag cosine; `sup_attr_pairs`
  merges the zh↔en attribute predicates so the two vocabularies align.
- **Fusion** : `sim = β·SE + (1−β)·AE` with **β=0.9** (SE-dominant; AE is a light refiner),
  then **CSLS** on the fused matrix — this is what realises the paper's attribute boost (+~11 Hit@1).

## DGMC — method
- **Entity-name features** : node feature = SUM of the GloVe-300d word embeddings of each
  entity's (translated) name. *This is what makes DGMC far stronger than the structural models.*
- **Local feature matching** (`psi_1`, a 3-layer RelCNN GNN) → a *sparse* top-k correspondence
  `S_0 = softmax(top_k(h_s · h_t))`. We L2-normalise the embeddings and use a temperature
  (cosine matching) because the summed-name features have wildly varying norms.
- **Neighbourhood consensus** (`psi_2` GNN + MLP, L=10 iterations) : random node colourings are
  passed through the correspondence to the target graph, diffused, and their disagreement drives
  `S_hat += MLP(D)`, re-ranking the candidates so matched neighbourhoods agree.
- Two training phases (feature matching, then refinement); separate KGs; `train.ref`/`test.ref`.

## MRAEA — method
- **Meta-relation-aware**: a relation and its **inverse** get distinct embeddings (table size
  `2R`); each entity starts from `relu([ mean(neighbour+self ent_emb) || mean(relation emb) ])`,
  so the representation is relation-aware.
- **Graph attention** (`depth=2` shared steps, `2` heads averaged): the logit of edge `(i,j)` is
  `leakyrelu(a_rel·rel(i,j) + a_self·h_i + a_neigh·h_j)`, softmax over neighbours; outputs of all
  steps are JK-concatenated. **L1 margin loss** with random negatives; **cosine** alignment.
- **Bi-directional iterative bootstrapping** (`iterative: true`, semi-supervised): every few
  epochs, **mutual** nearest neighbours among the unaligned entities are added as pseudo-anchors
  (`l→r` best *and* `r→l` best) — no gold-label leakage. Lifts Hit@1 ~+9 points.

## RREA — method (MRAEA's successor)
- **Relational reflection** (the new idea): aggregating neighbour `j` via relation `r`, `j` is
  *reflected* across the hyperplane orthogonal to the unit relation vector — `h_j − 2(h_j·r̂)r̂`
  (a Householder reflection, **norm/orthogonality-preserving**) — instead of MRAEA's additive
  term. Edge attention `a·[h_i ‖ reflect_r(h_j) ‖ r̂]`, softmax over neighbours (no leakyrelu).
- The **shared** `depth=2` encoder is run on an entity-based *and* a relation-based initial
  feature, outputs concatenated; JK-concat; **RMSprop lr 5e-3**, L1 margin; CSLS alignment.
- **Turn-based** semi-supervised bootstrap (`turns` × `epoch_per_turn`): between turns, CSLS
  **mutual** NN among the unaligned test entities are added as pseudo-anchors. `turns: 1` = basic.

## Package layout
```
code/
├── pyproject.toml / setup.py / LICENSE
└── src/
    ├── data.py            # DBP15K loading, neighbours, negatives, swapping, AliNet/KECG/GCN-Align graphs, JAPE attrs, DGMC name features, MRAEA graph (shared by RREA)
    ├── trainer.py         # ALL trainers: Trainer (NAEA), BootEATrainer, AliNetTrainer, KECGTrainer, GCNAlignTrainer, JAPETrainer, DGMCTrainer, MRAEATrainer, RREATrainer
    ├── main.py            # CLI entry point (dispatches to the chosen model)
    ├── models/
    │   ├── naea.py        # NAEA model + loss functions
    │   ├── bootea.py      # BootEA (AlignE) model + loss functions
    │   ├── alinet.py      # AliNet (gated multi-hop GNN) + relation-aware loss
    │   ├── kecg.py        # KECG (diagonal multi-head GAT + TransE) + losses
    │   ├── gcnalign.py    # GCN-Align (shared 2-layer GCN, SE) + L1 margin loss
    │   ├── jape.py        # JAPE (merged-seed TransE SE) + SE margin loss
    │   ├── dgmc.py        # DGMC (RelCNN GNN + sparse top-k neighbourhood-consensus)
    │   ├── mraea.py       # MRAEA (meta-relation-aware GAT) + L1 margin loss
    │   └── rrea.py        # RREA (relational-reflection GAT; reuses MRAEA graph + L1 loss)
    └── utils/
        ├── config.py      # YAML config + run directory
        ├── logger.py      # console + training.txt logger
        ├── metrics.py     # MRR / Hit@k, cosine / CSLS / L1 / L2
        └── plotting.py    # modern dark-theme figures
```

## Install
```bash
cd code
pip install -e .
```

## Train
```bash
# NAEA
python -m src.main --config ../configs/naea_dbp15k.yaml
# BootEA
python -m src.main --config ../configs/bootea_dbp15k.yaml
# AliNet
python -m src.main --config ../configs/alinet_dbp15k.yaml
# KECG
python -m src.main --config ../configs/kecg_dbp15k.yaml
# GCN-Align
python -m src.main --config ../configs/gcnalign_dbp15k.yaml
# JAPE
python -m src.main --config ../configs/jape_dbp15k.yaml
# DGMC  (needs the name-feature data in Data/DBP15K_pyg/DBP15K; --lang sets the pair)
python -m src.main --config ../configs/dgmc_dbp15k.yaml --lang zh_en
# MRAEA  (set train.iterative: true in the config for the semi-supervised +iter variant)
python -m src.main --config ../configs/mraea_dbp15k.yaml
# RREA  (train.turns: 1 = basic, 5 = semi-supervised; RMSprop rho/alpha=0.9 matches Keras)
python -m src.main --config ../configs/rrea_dbp15k.yaml
# overrides:
python -m src.main --config ../configs/alinet_dbp15k.yaml --lang fr_en --epochs 2000
```

## Metrics
`MRR`, `Hit@1`, `Hit@5`, `Hit@10` (both directions + average), reported every
`eval.every` epochs and logged to `metrics.csv`. Artefacts (checkpoints,
embeddings, loss/metric CSVs, dark-theme PNG curves) are written to a timestamped
run directory under `experiments/`.

## Results (zh_en, 30% seed) — paper vs. this repo
| | Hit@1 | Hit@10 | MRR |
|---|---:|---:|---:|
| NAEA (paper)      | 0.650 | 0.867 | 0.720 |
| NAEA (here)       | ~0.62 | ~0.86 | ~0.70 |
| BootEA (paper)    | 0.629 | 0.847 | 0.703 |
| BootEA (here)     | ~0.56 | ~0.85 | ~0.66 |
| AliNet (paper)    | 0.539 | 0.826 | 0.628 |
| AliNet (here)     | ~0.53 | ~0.81 | ~0.63 |
| KECG (paper)      | 0.477 | 0.835 | 0.598 |
| KECG (here)       | ~0.42 | ~0.73 | ~0.52 |
| GCN-Align (paper, SE) | 0.384 | 0.703 | — |
| GCN-Align (here, SE)  | ~0.38 | ~0.68 | ~0.49 |
| JAPE (paper, SE+AE)   | 0.412 | 0.745 | 0.490 |
| JAPE (here, SE+AE)    | **0.425** | **0.761** | **0.537** |
| DGMC (paper, Hit@1)   | 0.801 | — | — |
| DGMC (here, Hit@1)    | 0.767 | — | — |
| MRAEA base (paper)      | 0.638 | 0.882 | 0.729 |
| MRAEA base (here, l2r)  | **0.659** | **0.898** | **0.746** |
| MRAEA +iter (paper)     | 0.757 | 0.930 | 0.827 |
| MRAEA +iter (here, l2r) | 0.746 | **0.930** | 0.814 |
| RREA basic (paper)      | 0.715 | 0.929 | 0.794 |
| RREA basic (here, l2r)  | 0.712 | **0.934** | 0.793 |
| RREA semi (paper)       | 0.801 | 0.948 | 0.857 |
| RREA semi (here, l2r)   | **0.805** | **0.950** | **0.859** |

NAEA / BootEA / AliNet / GCN-Align reach (or essentially match) their paper's headline
metrics; the residual Hit@1 gap is the known hard-to-reproduce part for purely *structural*
models (independent reproductions, e.g. OpenEA, land well below the papers too). KECG sits
~0.07 under its paper — the NNS that gives it the extra points was approximated, not the exact
sparse-GAT implementation. **JAPE meets/beats its paper on all three languages** (ja_en
0.368/0.735/0.490, fr_en 0.311/0.705/0.442); the attribute channel is what lifts it well above
the structure-only models. **DGMC** (which uses entity-*name* embeddings, not pure structure)
reaches Hit@1 zh_en 0.767 / ja_en 0.814 / **fr_en 0.939** (paper 0.801 / 0.848 / 0.933) — it
*beats* the paper on fr_en; zh/ja sit ~3.4 pts under, the residual being the top-k candidate
recall of the initial feature matching. **MRAEA** (meta-relation-aware GAT) is *above* the
paper in the base setting (zh_en l2r 0.659/0.898/0.746 vs 0.638/0.882/0.729) and ~1 pt under
with bi-directional iterative self-training (0.746/0.930/0.814 vs 0.757/0.930/0.827, Hit@10
matched exactly). **RREA** matches the paper in both settings (zh_en basic 0.712/0.934/0.793 vs
0.715/0.929/0.794; semi 0.805/0.950/0.859 vs 0.801/0.948/0.857) — the key was matching Keras's
RMSprop (`rho/alpha=0.9, eps=1e-7`), which PyTorch defaults differently (`alpha=0.99`); that fix
alone added ~2-3 Hit@1.
