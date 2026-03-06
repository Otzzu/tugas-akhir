"""
evaluate.py — Evaluation and metrics reporting.

Usage:
    uv run evaluate --checkpoint checkpoints/best_gcn.pt --config configs/default.yaml
    uv run python -m gnn_vuln.evaluate --checkpoint checkpoints/best_gcn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset import VulnerabilityDataset
from gnn_vuln.train import build_model
from gnn_vuln.utils import get_device, load_checkpoint, setup_logging


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (true_labels, predicted_labels, class_probabilities).

    class_probabilities has shape [N, num_classes] — works for both binary
    (num_classes=2) and any multi-class setting.
    """
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(logits, dim=-1)          # [B, num_classes]
        preds = logits.argmax(dim=-1)                  # [B]
        all_y.extend(batch.y.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.append(probs.cpu().numpy())
    return np.array(all_y), np.array(all_pred), np.vstack(all_prob)


def plot_roc_curve(y_true, y_prob, save_path: Path, class_names: list[str] | None = None) -> None:
    """
    Plot ROC curve(s).

    - Binary (num_classes=2):  single curve using the positive-class column.
    - Multi-class (N>2):       one curve per class (OvR), plus macro-average.
    """
    num_classes = y_prob.shape[1]
    plt.figure(figsize=(8, 6))

    if num_classes == 2:
        # Binary
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    else:
        # Multi-class OvR
        from sklearn.preprocessing import label_binarize
        classes = list(range(num_classes))
        y_bin = label_binarize(y_true, classes=classes)
        from sklearn.metrics import roc_curve
        for i in classes:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            name = class_names[i] if class_names else f"Class {i}"
            auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_i:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Vulnerability Detection")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: Path, class_names: list[str] | None = None) -> None:
    num_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(5, num_classes)  # scale figure with number of classes
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    tick_labels = class_names if class_names else [str(i) for i in range(num_classes)]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN vulnerability detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()
    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    dataset = VulnerabilityDataset(root=str(cfg.data.processed_dir.parent))
    _, _, test_idx = dataset.get_splits(seed=cfg.train.seed)
    test_loader = DataLoader(dataset[test_idx], batch_size=cfg.train.batch_size)

    in_channels = dataset[0].x.size(1)
    model = build_model(cfg, in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)
    num_classes = y_prob.shape[1]
    is_binary = num_classes == 2

    # Resolve class names from dataset if available
    class_names = getattr(dataset, "class_names", None)

    # Print metrics
    print("\n" + "=" * 55)
    print("Classification Report")
    print("=" * 55)
    target_names = class_names or [str(i) for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # AUC-ROC
    if is_binary:
        auc = roc_auc_score(y_true, y_prob[:, 1])
        f1 = f1_score(y_true, y_pred, average="binary")
    else:
        # macro OvR for multi-class
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

    print(f"AUC-ROC (macro OvR): {auc:.4f}")
    print(f"F1 Score (macro)   : {f1:.4f}")
    print("=" * 55 + "\n")

    # Save plots
    results_dir = cfg.train.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(y_true, y_prob, results_dir / "roc_curve.png", class_names=class_names)
    plot_confusion_matrix(y_true, y_pred, results_dir / "confusion_matrix.png", class_names=class_names)


if __name__ == "__main__":
    main()
