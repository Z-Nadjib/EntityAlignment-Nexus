# Getting started

This page takes you from a fresh clone to a trained model and its metrics.

## 1. Install

```bash
git clone https://github.com/Z-Nadjib/EntityAlignment-Nexus.git
cd EntityAlignment-Nexus/code
pip install -e .
```

This installs the `src` package and its dependencies (PyTorch, NumPy, pandas, scikit-learn,
matplotlib, seaborn, tqdm, PyYAML). Python **3.9+** and PyTorch **2.0+** are required. A GPU is
recommended for the GNN models but not mandatory.

!!! tip "Documentation extras"
    To build this site locally:
    ```bash
    pip install -r requirements-docs.txt
    mkdocs serve
    ```

## 2. Data layout

Most models use the **JAPE/MTransE split** under `Data/dbp15k/<lang>/mtranse/<fold>/`, where the
entity and relation ids of the two graphs are **disjoint** and contiguous (`0..N-1`), so a single
embedding table can be indexed directly.

```text
Data/dbp15k/zh_en/mtranse/0_3/
├── ent_ids_1 / ent_ids_2     # "<id>\t<uri>"  entities of KG1 / KG2
├── rel_ids_1 / rel_ids_2     # "<id>\t<uri>"  relations
├── triples_1 / triples_2     # "<h>\t<r>\t<t>"
├── sup_pairs                 # seed (train) alignments  (30% for fold 0_3)
└── ref_pairs                 # test alignments          (70%)
```

Two models use a different layout:

- **JAPE** uses the *high-level* merged-seed format (`use_mtranse_format: false`): aligned seeds
  share the same id, so the two KGs live in one graph and TransE can propagate the alignment.
- **DGMC** uses the *GMNN/PyG* split (`Data/DBP15K_pyg/DBP15K/`) with GloVe **entity-name**
  features, separate ids per graph, and `train.ref` / `test.ref` matchings.

## 3. Train a model

Every model is driven by one YAML file in `configs/`:

=== "RREA (best)"

    ```bash
    python -m src.main --config ../configs/rrea_dbp15k.yaml
    ```

=== "JAPE (attributes)"

    ```bash
    python -m src.main --config ../configs/jape_dbp15k.yaml
    ```

=== "DGMC (names)"

    ```bash
    python -m src.main --config ../configs/dgmc_dbp15k.yaml --lang zh_en
    ```

=== "MRAEA"

    ```bash
    python -m src.main --config ../configs/mraea_dbp15k.yaml
    ```

### Useful overrides

| Flag | Effect |
|------|--------|
| `--lang zh_en\|ja_en\|fr_en` | choose the language pair |
| `--fold 0_1 .. 0_5` | choose the seed split |
| `--epochs N` | override the number of epochs |
| `--device cuda\|cpu` | force a device |
| `--model <name>` | override the model dispatched from the config |

## 4. What a run produces

Each run creates a timestamped directory under `experiments/` (or the configured `output_dir`):

```text
<model>_<lang>_<fold>_<timestamp>/
├── config_used.yaml      # exact resolved config (reproducibility)
├── training.txt          # full training log
├── loss.csv              # per-epoch loss components
├── metrics.csv           # per-eval MRR / Hit@k (both directions + avg)
├── model_best.pt         # best checkpoint (by MRR)
├── embeddings.pt         # final entity / relation embeddings
├── loss_curve.png        # dark-theme curves
└── metrics_curve.png
```

## 5. Metrics

All models are evaluated with the same protocol (see [`utils/metrics.py`](https://github.com/Z-Nadjib/EntityAlignment-Nexus/blob/main/code/src/utils/metrics.py)):

- **Hit@k** - fraction of source entities whose gold target ranks in the top `k`.
- **MRR** - mean reciprocal rank of the gold target.
- **CSLS** - Cross-domain Similarity Local Scaling, which corrects *hubness* in
  high-dimensional spaces and typically lifts Hit@1 by several points.

$$\text{csls}(x, y) = 2\cos(x, y) - r_T(x) - r_S(y)$$

where $r_T(x)$ is the mean cosine of $x$ to its $k$ nearest targets and $r_S(y)$ the symmetric
quantity. Metrics are reported in both directions (left-to-right, right-to-left) and averaged.

## 6. Notebooks

If you would rather read a method than run it, open the matching notebook in `Notebook/`. Each
one is **self-contained** - it re-implements the data loader, model, metrics and trainer inline,
documented cell by cell, and reproduces the package results.
