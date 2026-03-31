"""Command-line entry point: train NAEA or BootEA on DBP15K.

Wires together the data module and the appropriate trainer, driven by a YAML
config. The model is chosen by ``experiment.model`` in the config (``naea`` or
``bootea``) or overridden with ``--model``.

Examples
--------
    python -m src.main --config configs/naea_dbp15k.yaml
    python -m src.main --config configs/bootea_dbp15k.yaml
    python -m src.main --config configs/bootea_dbp15k.yaml --lang fr_en --epochs 300
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .data import (build_alinet_graph, build_gcnalign_adj, build_kecg_graph,
                   build_mraea_graph, build_neighbors, load_dbp15k, load_dgmc_dbp15k)
from .models.naea import NAEA
from .models.bootea import BootEA
from .models.alinet import AliNet
from .models.kecg import KECG
from .models.gcnalign import GCNAlign
from .models.jape import JAPE
from .models.dgmc import DGMC, RelCNN
from .models.mraea import MRAEA
from .models.rrea import RREA
from .trainer import (AliNetTrainer, BootEATrainer, DGMCTrainer, GCNAlignTrainer,
                      JAPETrainer, KECGTrainer, MRAEATrainer, RREATrainer, Trainer)
from .utils.config import load_config, make_run_dir
from .utils.logger import get_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train NAEA on DBP15K.")
    p.add_argument("--config", required=True, help="path to the YAML config")
    p.add_argument("--project-root", default=None, help="project root (default: parent of configs/)")
    p.add_argument("--lang", default=None, help="override data.lang (zh_en | ja_en | fr_en)")
    p.add_argument("--fold", default=None, help="override data.fold (0_1 .. 0_5)")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--device", default=None, help="override experiment.device (cuda | cpu)")
    p.add_argument("--model", default=None,
                   choices=["naea", "bootea", "alinet", "kecg", "gcnalign", "jape", "dgmc", "mraea", "rrea"],
                   help="override experiment.model")
    return p.parse_args()


def build_naea(cfg, data, device):
    """Instantiate NAEA from config + neighbourhood tensors."""
    ne, nr, ns, nm = build_neighbors(data, cfg.model.max_neighbors, seed=cfg.experiment.seed)
    return NAEA(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        embed_dim=cfg.model.embed_dim,
        neigh_ent=ne.to(device), neigh_rel=nr.to(device),
        neigh_sign=ns.to(device), neigh_mask=nm.to(device),
        attn_heads=cfg.model.attn_heads, attn_dropout=cfg.model.attn_dropout,
        init=cfg.model.init, normalize_embeddings=cfg.model.normalize_embeddings,
        neighbor_message=cfg.model.neighbor_message,
    ).to(device)


def build_bootea(cfg, data, device):
    """Instantiate BootEA (AlignE) from config."""
    return BootEA(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        embed_dim=cfg.model.embed_dim,
        init=cfg.model.init,
        normalize_embeddings=cfg.model.normalize_embeddings,
    ).to(device)


def build_alinet(cfg, data, device):
    """Instantiate AliNet (gated multi-hop GNN) + its graph structures."""
    adj1, two_hop = build_alinet_graph(data, max_two_hop=cfg.model.max_two_hop, seed=cfg.experiment.seed)
    return AliNet(
        num_entities=data.num_entities,
        adj1=adj1, two_hop=two_hop,
        embed_dim=cfg.model.embed_dim,
        layer_dims=tuple(cfg.model.layer_dims),
        dropout=cfg.model.get("dropout", 0.0),
        normalize_embeddings=cfg.model.normalize_embeddings,
        num_relations=data.num_relations,
    ).to(device)


def build_jape(cfg, data, device):
    """Instantiate JAPE (TransE structure embedding)."""
    return JAPE(
        num_entities=data.num_entities, num_relations=data.num_relations,
        embed_dim=cfg.model.embed_dim, init=cfg.model.init,
        normalize_embeddings=cfg.model.normalize_embeddings,
    ).to(device)


def build_dgmc(cfg, data, device):
    """Instantiate DGMC (two RelCNN GNNs + consensus) for graph matching."""
    m = cfg.model
    psi_1 = RelCNN(data.num_features, m.dim, m.num_layers, batch_norm=False,
                   cat=True, lin=True, dropout=m.get("dropout", 0.5))
    psi_2 = RelCNN(m.rnd_dim, m.rnd_dim, m.num_layers, batch_norm=False,
                   cat=True, lin=True, dropout=0.0)
    return DGMC(psi_1, psi_2, num_steps=0, k=m.k,             # phase 1 starts at num_steps=0
                normalize=m.get("normalize", True),
                temperature=m.get("temperature", 30.0)).to(device)


def build_mraea(cfg, data, device):
    """Instantiate MRAEA (meta-relation-aware GAT) + its sparse graph structures."""
    graph = build_mraea_graph(data)
    m = cfg.model
    return MRAEA(graph, node_hidden=m.node_hidden, rel_hidden=m.rel_hidden,
                 depth=m.depth, attn_heads=m.attn_heads, dropout=m.dropout).to(device)


def build_rrea(cfg, data, device):
    """Instantiate RREA (relational-reflection GAT) + its sparse graph structures."""
    graph = build_mraea_graph(data)                  # identical get_matrix as MRAEA
    m = cfg.model
    return RREA(graph, node_hidden=m.node_hidden, depth=m.depth,
                attn_heads=m.attn_heads, dropout=m.dropout).to(device)


def build_gcnalign(cfg, data, device):
    """Instantiate GCN-Align (shared 2-layer GCN, SE channel) + its adjacency."""
    adj = build_gcnalign_adj(data)
    return GCNAlign(
        num_entities=data.num_entities, adj=adj.to(device),
        embed_dim=cfg.model.embed_dim, n_layers=cfg.model.n_layers,
        dropout=cfg.model.get("dropout", 0.0), activation=cfg.model.get("activation", True),
        normalize_embeddings=cfg.model.get("normalize_embeddings", True),
    ).to(device)


def build_kecg(cfg, data, device):
    """Instantiate KECG (shared diagonal GAT + TransE) + its combined graph."""
    edge_index = build_kecg_graph(data)
    return KECG(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        edge_index=edge_index.to(device),
        embed_dim=cfg.model.embed_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        attn_dropout=cfg.model.get("attn_dropout", 0.0),
        normalize_embeddings=cfg.model.normalize_embeddings,
        instance_normalization=cfg.model.get("instance_normalization", False),
    ).to(device)


def main():
    args = parse_args()
    cfg = load_config(args.config, project_root=args.project_root)

    # CLI overrides
    if args.lang:
        cfg.data.lang = args.lang
    if args.fold:
        cfg.data.fold = args.fold
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.device:
        cfg.experiment.device = args.device
    if args.model:
        cfg.experiment.model = args.model
    model_name = str(cfg.experiment.get("model", "naea")).lower()
    if model_name == "dgmc" and args.lang:               # --lang overrides the pair
        cfg.data.pair = args.lang
    tag = cfg.data.pair if model_name == "dgmc" else f"{cfg.data.lang}_{cfg.data.fold}"
    cfg.experiment.name = f"{model_name}_{tag}"

    set_seed(cfg.experiment.seed)
    run_dir = make_run_dir(cfg)
    logger = get_logger(cfg, run_dir)

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Model: {model_name} | Device: {device}")

    if model_name == "dgmc":
        data = load_dgmc_dbp15k(Path(cfg._project_root) / cfg.data.root, cfg.data.pair)
        logger.info(f"Data: {data.summary()}")
        model = build_dgmc(cfg, data, device)
        trainer = DGMCTrainer(cfg, data, model, run_dir, logger)
        logger.info(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        history = trainer.fit()
        logger.info(f"Finished. Best Hit@1={history['best_mrr']:.4f} @ epoch {history['best_epoch']}.")
        return history

    data = load_dbp15k(Path(cfg._project_root) / cfg.data.root, cfg.data.lang,
                       cfg.data.fold, cfg.data.use_mtranse_format)
    logger.info(f"Data: {data.summary()}")

    if model_name == "bootea":
        model = build_bootea(cfg, data, device)
        trainer = BootEATrainer(cfg, data, model, run_dir, logger)
    elif model_name == "alinet":
        model = build_alinet(cfg, data, device)
        trainer = AliNetTrainer(cfg, data, model, run_dir, logger)
    elif model_name == "kecg":
        model = build_kecg(cfg, data, device)
        trainer = KECGTrainer(cfg, data, model, run_dir, logger)
    elif model_name == "gcnalign":
        model = build_gcnalign(cfg, data, device)
        trainer = GCNAlignTrainer(cfg, data, model, run_dir, logger)
    elif model_name == "jape":
        model = build_jape(cfg, data, device)
        trainer = JAPETrainer(cfg, data, model, run_dir, logger)
    elif model_name == "mraea":
        model = build_mraea(cfg, data, device)
        trainer = MRAEATrainer(cfg, data, model, run_dir, logger)
    elif model_name == "rrea":
        model = build_rrea(cfg, data, device)
        trainer = RREATrainer(cfg, data, model, run_dir, logger)
    else:
        model = build_naea(cfg, data, device)
        trainer = Trainer(cfg, data, model, run_dir, logger)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    history = trainer.fit()
    logger.info(f"Finished. Best MRR={history['best_mrr']:.4f} @ epoch {history['best_epoch']}.")
    return history


if __name__ == "__main__":
    main()
