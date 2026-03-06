"""
train.py — Training entry point.

Usage (via uv):
    uv run train --config configs/default.yaml
    uv run python -m gnn_vuln.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from loguru import logger

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset import VulnerabilityDataset
from gnn_vuln.models.gcn import GCNVulnDetector
from gnn_vuln.models.gat import GATVulnDetector
from gnn_vuln.utils import set_seed, setup_logging, save_checkpoint, get_device


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Config, in_channels: int) -> torch.nn.Module:
    arch = cfg.model.architecture.lower()
    if arch == "gcn":
        return GCNVulnDetector(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
        )
    elif arch == "gat":
        return GATVulnDetector(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim // cfg.model.heads,
            num_layers=cfg.model.num_layers,
            heads=cfg.model.heads,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Choose from: gcn, gat")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=-1)
        correct += (preds == batch.y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNN for vulnerability detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()
    set_seed(cfg.train.seed)
    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    # Dataset & splits
    logger.info("Loading dataset…")
    dataset = VulnerabilityDataset(root=str(cfg.data.processed_dir.parent))
    train_idx, val_idx, test_idx = dataset.get_splits(seed=cfg.train.seed)

    train_loader = DataLoader(dataset[train_idx], batch_size=cfg.train.batch_size, shuffle=True)
    val_loader   = DataLoader(dataset[val_idx],   batch_size=cfg.train.batch_size)
    test_loader  = DataLoader(dataset[test_idx],  batch_size=cfg.train.batch_size)

    in_channels = dataset[0].x.size(1)
    logger.info(f"Dataset: {len(dataset)} graphs | in_channels={in_channels}")

    # Model
    model = build_model(cfg, in_channels).to(device)
    logger.info(f"Model: {cfg.model.architecture.upper()} | params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = cfg.train.checkpoint_dir / f"best_{cfg.model.architecture}.pt"

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:03d}/{cfg.train.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, best_ckpt, epoch=epoch, val_loss=val_loss, val_acc=val_acc)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    # Final test evaluation
    _, test_acc = evaluate(model, test_loader, device)
    logger.info(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
