"""NAEA on DBP15K: reusable engine.

Package layout
--------------
data          : DBP15K loading, neighbourhood construction, negative sampling,
                BootEA swapping and dynamic triple sampling, plus the graph
                builders for the GNN models.
trainer       : all model trainers (NAEA, BootEA, AliNet, KECG, GCN-Align,
                JAPE, DGMC, MRAEA, RREA).
main          : CLI entry point dispatching to the chosen model's trainer.
models/       : the nine entity-alignment models and their loss functions.
utils/        : config loading, logging, metrics (MRR/Hit@k, CSLS), modern plotting.
"""
from . import data, trainer
from . import models, utils

__all__ = ["data", "trainer", "models", "utils"]
