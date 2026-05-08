"""
evaluate.py — Evaluation and metrics reporting.

Metrics computed
----------------
Function-level (VulLMGNN / LineVul / VulChecker style):
  Accuracy, Precision, Recall, F1 (macro for multi-class), AUC-ROC, AUC-PR

Statement-level localization (LineVul / WAVES style):
  Top-10 Accuracy  : ≥1 flaw line in the top-10 ranked statements per function
  IFA              : clean lines inspected before first flaw line  (lower = better)
  Effort@20%Recall : fraction of lines to inspect for 20% flaw recall (lower = better)
  Recall@K%LOC     : flaw recall when inspecting top K% of lines   (higher = better)

All outputs are saved to results/ for offline analysis:
  predictions.csv          per-sample function-level results
  localization_scores.csv  per-(function,line) MIL scores + flaw labels
  metrics_summary.json     all scalar metrics in one file
  roc_curve.png
  confusion_matrix.png
  pr_curve.png
  recall_at_loc_curve.png  (only when flaw GT exists)
  ifa_distribution.png     (only when flaw GT exists)

Usage:
    uv run evaluate --checkpoint checkpoints/<run_id>/best_lmgcn.pt --config configs/lmgcn/binary.yaml
    uv run evaluate --checkpoint checkpoints/<run_id>/best_lmgat.pt --config configs/lmgat/multiclass.yaml
    uv run evaluate --checkpoint checkpoints/<run_id>/best_lmgat_codebert.pt --config configs/lmgat_codebert/multiclass.yaml
    uv run evaluate --checkpoint checkpoints/<run_id>/best_lmgat_mcs.pt --config configs/lmgat_mcs/multiclass.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import DataLoader

from gnn_vuln.baselines import LINEVUL_FUNC, LINEVUL_LOC_ATTENTION  # stored for future use
from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from gnn_vuln.metrics import compute_all_localization_metrics, make_func_loc_result
import torch.nn as nn
from gnn_vuln.models.lmgcn import LMGCNVulnDetector
from gnn_vuln.models.registry import build_model
from gnn_vuln.utils import get_device, load_checkpoint, setup_logging


# ---------------------------------------------------------------------------
# Localization data extraction
# ---------------------------------------------------------------------------

def _extract_func_loc(
    scores_b: torch.Tensor,
    batch_idx: torch.Tensor,
    node_line: torch.Tensor,
    flaw_mask: torch.Tensor | None,
    b: int,
) -> dict:
    """
    Extract per-function localization data for graph b from a batched PyG Data.

    scores_b  : [n_stmts] binary scalar scores  OR  [n_stmts, num_classes]
                multiclass logits (one element from stmt_scores_list[b])
    batch_idx : [N_total] graph membership per node
    node_line : [N_total] source line number per node (-1 = unknown)
    flaw_mask : [N_total] 1 if node is on a flaw line, else 0 (None → all zeros)
    b         : graph index within this batch
    """
    if len(scores_b) == 0:
        return make_func_loc_result([], [], [])

    # Convert raw logits to a scalar vulnerability score per statement
    if scores_b.dim() == 2:
        # Multiclass stmt head: use 1 - p_benign as vulnerability score
        probs = torch.softmax(scores_b, dim=-1)
        scores_scalar = (1.0 - probs[:, 0]).cpu()   # [n_stmts]
    else:
        # Binary stmt head: sigmoid of raw logit
        scores_scalar = torch.sigmoid(scores_b).cpu()  # [n_stmts]

    graph_mask = batch_idx == b
    node_line_b = node_line[graph_mask]
    flaw_b = (
        flaw_mask[graph_mask]
        if flaw_mask is not None
        else torch.zeros_like(node_line_b)
    )

    valid = node_line_b >= 0
    if not valid.any():
        return make_func_loc_result([], [], [])

    lines_b = node_line_b[valid]
    flaw_b = flaw_b[valid]
    unique_lines = lines_b.unique(sorted=True)

    scores_list = scores_scalar.tolist()
    line_labels: list[int] = []
    for line in unique_lines:
        nodes_on_line = lines_b == line
        line_labels.append(int(flaw_b[nodes_on_line].any().item()))

    return make_func_loc_result(
        unique_lines.cpu().tolist(),
        scores_list,
        line_labels,
    )


# ---------------------------------------------------------------------------
# Prediction + localization loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions_and_localization(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Run inference over the full loader, collecting both function-level predictions
    and per-function statement-level localization data.

    Returns
    -------
    y_true      : [N] int array — ground-truth class labels
    y_pred      : [N] int array — predicted class labels
    y_prob      : [N, C] float  — softmax probabilities per class
    confidence  : [N] float     — max softmax probability (predicted-class confidence)
    loc_results : list[dict]    — one per function (see metrics.make_func_loc_result)
    """
    model.eval()
    all_y: list = []
    all_pred: list = []
    all_prob: list = []
    loc_results: list[dict] = []

    for batch in loader:
        batch = batch.to(device)
        node_line = getattr(batch, "node_line", None)
        flaw_mask = getattr(batch, "flaw_line_mask", None)
        edge_attr = getattr(batch, "edge_attr", None)

        if hasattr(model, "codebert"):
            func_input_ids = getattr(batch, "func_input_ids", None)
            func_attention_mask = getattr(batch, "func_attention_mask", None)
            out = model(
                batch.x, batch.edge_index, batch.batch, node_line, edge_attr,
                func_input_ids, func_attention_mask,
            )
        else:
            out = model(
                batch.x, batch.edge_index, batch.batch, node_line, edge_attr
            )

        if len(out) == 5:
            logit_func, _, _, stmt_scores_list, _ = out
        elif len(out) == 4:
            logit_func, _, _, stmt_scores_list = out
        else:
            logit_func, stmt_scores_list = out

        probs = torch.softmax(logit_func, dim=-1)
        preds = logit_func.argmax(dim=-1)

        all_y.extend(batch.y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.append(probs.cpu().numpy())

        n_graphs = int(batch.batch.max().item()) + 1
        for b in range(n_graphs):
            if stmt_scores_list is not None and node_line is not None:
                func_loc = _extract_func_loc(
                    stmt_scores_list[b], batch.batch, node_line, flaw_mask, b
                )
            else:
                func_loc = make_func_loc_result([], [], [])
            loc_results.append(func_loc)

    y_prob = np.vstack(all_prob)
    confidence = y_prob.max(axis=1)
    return np.array(all_y), np.array(all_pred), y_prob, confidence, loc_results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """ROC curve — single curve for binary, one-vs-rest per class for multi-class."""
    num_classes = y_prob.shape[1]
    plt.figure(figsize=(8, 6))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc_val = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    else:
        classes = list(range(num_classes))
        y_bin = label_binarize(y_true, classes=classes)
        for i in classes:
            if y_bin[:, i].sum() == 0:
                continue
            try:
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
                name = class_names[i] if class_names else f"Class {i}"
                plt.plot(fpr_i, tpr_i, label=f"{name} (AUC={auc_i:.3f})")
            except ValueError:
                continue

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Vulnerability Detection")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """Confusion matrix with all classes shown even if absent in test set."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig_size = max(5, num_classes)
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    labels = class_names if class_names else [str(i) for i in range(num_classes)]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix → {save_path}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """Precision-Recall curve — single for binary, OvR per class for multi-class."""
    num_classes = y_prob.shape[1]
    plt.figure(figsize=(8, 6))

    if num_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"AUC-PR = {pr_auc:.4f}")
    else:
        classes = list(range(num_classes))
        y_bin = label_binarize(y_true, classes=classes)
        for i in classes:
            if y_bin[:, i].sum() == 0:
                continue
            try:
                prec_i, rec_i, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                pr_auc_i = auc(rec_i, prec_i)
                name = class_names[i] if class_names else f"Class {i}"
                plt.plot(rec_i, prec_i, label=f"{name} (AUC={pr_auc_i:.3f})")
            except ValueError:
                continue

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Vulnerability Detection")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"PR curve → {save_path}")


def plot_recall_at_loc_curve(
    k_values: list[float],
    recall_values: list[float],
    save_path: Path,
) -> None:
    """Recall@K%LOC curve — fraction of flaw lines caught vs fraction of lines inspected."""
    k_pct = [k * 100 for k in k_values]
    rec_pct = [
        r * 100 if not (isinstance(r, float) and np.isnan(r)) else float("nan")
        for r in recall_values
    ]
    plt.figure(figsize=(8, 5))
    plt.plot(k_pct, rec_pct, marker="o", linewidth=2)
    plt.xlabel("% Lines Inspected (LOC%)")
    plt.ylabel("% Flaw Lines Found (Recall%)")
    plt.title("Recall@K%LOC — Statement-Level Localization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Recall@K%LOC curve → {save_path}")


def plot_ifa_distribution(ifa_values: list[float], save_path: Path) -> None:
    """IFA histogram — distribution of clean lines inspected before first flaw."""
    if not ifa_values:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(ifa_values, bins=30, edgecolor="black", alpha=0.75)
    plt.axvline(
        np.mean(ifa_values), color="red", linestyle="--",
        label=f"Mean IFA = {np.mean(ifa_values):.2f}",
    )
    plt.xlabel("IFA (clean lines inspected before first flaw line)")
    plt.ylabel("Number of Functions")
    plt.title("IFA Distribution — Statement-Level Localization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"IFA distribution → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _safe(v: object) -> object:
    """Convert NaN floats to None for JSON serialisation."""
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained vulnerability detector")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_*.pt checkpoint file")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML used during training (auto-detected from checkpoint dir if omitted)")
    args = parser.parse_args()

    # Auto-detect config from checkpoint's run directory (train.py copies config.yaml there)
    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        ckpt_dir = Path(args.checkpoint).parent
        auto = ckpt_dir / "config.yaml"
        if auto.exists():
            config_path = auto
            logger.info(f"Auto-detected config: {config_path}")
        else:
            logger.warning(f"No config.yaml in {ckpt_dir}, falling back to default.")

    cfg = (
        Config.from_yaml(config_path)
        if config_path and config_path.exists()
        else load_default_config()
    )
    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
    func_lm_source = getattr(cfg.model, "func_lm_source", "raw")

    func_lm = getattr(cfg.model, "func_lm", "") or pretrained_lm

    logger.info("Loading dataset…")
    dataset = CodeBERTGraphDataset(
        root=str(cfg.data.processed_dir.parent),
        max_nodes=cfg.data.max_nodes,
        embedder_device=cfg.train.device,
        mode=cfg.data.mode,
        source=getattr(cfg.data, "source", "bigvul"),
        pretrained_lm=pretrained_lm,
        func_lm=func_lm,
        add_func_tokens=add_func_tokens,
        func_lm_source=func_lm_source,
        top_cwe=getattr(cfg.data, "top_cwe", 0),
        cwe_list=getattr(cfg.data, "cwe_list", None),
        cwe_groups=getattr(cfg.data, "cwe_groups", None),
        filter_owasp_top10=getattr(cfg.data, "filter_owasp_top10", False),
        filter_top25=getattr(cfg.data, "filter_top25", False),
        max_per_class=getattr(cfg.data, "max_per_class", 0),
        resample_seed=getattr(cfg.data, "resample_seed", 42),
    )
    _, _, test_idx = dataset.get_splits(seed=cfg.train.seed)
    test_loader = DataLoader(dataset[test_idx], batch_size=cfg.train.batch_size)

    in_channels = dataset[0].x.size(1)
    model = build_model(cfg, in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    logger.info(f"Model loaded from {args.checkpoint}")

    logger.info("Running inference…")
    y_true, y_pred, y_prob, confidence, loc_results = get_predictions_and_localization(
        model, test_loader, device
    )

    num_classes = y_prob.shape[1]
    is_binary = num_classes == 2
    class_names: list[str] | None = getattr(dataset, "class_names", None)
    target_names = class_names or [str(i) for i in range(num_classes)]
    correct_mask = y_true == y_pred
    raw_funcs = getattr(dataset, "raw_funcs", None)  # parallel list of source strings

    # ── Function-level report ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Function-Level Classification Report")
    print("=" * 65)
    print(classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0,
    ))

    try:
        if is_binary:
            auc_roc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Restrict to classes present in test set — OvR fails when a class
            # has zero positive samples (common with many classes + small test set).
            # Renormalize sliced probs so they sum to 1 (sklearn requirement).
            present = np.unique(y_true)
            y_prob_present = y_prob[:, present]
            y_prob_present = y_prob_present / y_prob_present.sum(axis=1, keepdims=True)
            auc_roc = roc_auc_score(
                y_true, y_prob_present,
                multi_class="ovr", average="macro", labels=present,
            )
    except ValueError:
        auc_roc = float("nan")

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = float((y_true == y_pred).mean())

    print(f"AUC-ROC (macro OvR) : {auc_roc:.4f}")
    print(f"F1 Score (macro)    : {f1:.4f}")
    print(f"Accuracy            : {accuracy:.4f}")
    print("-" * 65)
    print(f"Confidence (mean)   : {confidence.mean():.4f}")
    if correct_mask.any():
        print(f"Confidence (correct): {confidence[correct_mask].mean():.4f}")
    if (~correct_mask).any():
        print(
            f"Confidence (wrong)  : {confidence[~correct_mask].mean():.4f}"
            f"  (n={int((~correct_mask).sum())})"
        )
    print("=" * 65)

    # ── Statement-level localization ────────────────────────────────────────
    loc_metrics = compute_all_localization_metrics(loc_results)
    n_gt = loc_metrics["num_funcs_with_flaw_gt"]

    print(f"\n{'=' * 65}")
    print(f"Statement-Level Localization  (functions with flaw GT: {n_gt})")
    print("=" * 65)
    if n_gt == 0:
        print("  No flaw-line ground truth found (flaw_line_mask all-zero).")
        print("  To enable localization metrics, ensure each CPG file has a")
        print("  sidecar .meta.json with a non-empty 'flaw_lines' list.")
    else:
        print(f"  Top-1  Accuracy    : {loc_metrics['top_1_accuracy']:.4f}")
        print(f"  Top-3  Accuracy    : {loc_metrics['top_3_accuracy']:.4f}")
        print(f"  Top-5  Accuracy    : {loc_metrics['top_5_accuracy']:.4f}")
        print(f"  Top-10 Accuracy    : {loc_metrics['top_10_accuracy']:.4f}")
        print(f"  IFA (mean)         : {loc_metrics['ifa_mean']:.2f}  (lower = better)")
        print(f"  Effort@20%Recall   : {loc_metrics['effort_at_20pct_recall']:.4f}  (lower = better)")
        print(f"  Recall@1%LOC       : {loc_metrics['recall_at_1pct_loc']:.4f}")
        print(f"  Recall@5%LOC       : {loc_metrics['recall_at_5pct_loc']:.4f}")
        print(f"  Recall@20%LOC      : {loc_metrics['recall_at_20pct_loc']:.4f}")

        # Show sample: top-3 suspicious lines with code for first 3 vulnerable functions
        print()
        print("  Sample — top-3 suspicious lines (first 3 vulnerable functions):")
        shown = 0
        for func_idx, (r, yt) in enumerate(zip(loc_results, y_true)):
            if int(yt) == 0 or shown >= 3:
                continue
            raw_func = ""
            if raw_funcs is not None:
                ds_idx = test_idx[func_idx]
                raw_func = raw_funcs[ds_idx] if ds_idx < len(raw_funcs) else ""
            src_lines = raw_func.splitlines() if raw_func else []
            print(f"  func {func_idx} (class={int(yt)}):")
            for ln, sc, lab in zip(
                r["ranked_line_numbers"][:3],
                r["ranked_scores"][:3],
                r["ranked_labels"][:3],
            ):
                code = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else "<no code>"
                marker = "FLAW" if lab else "    "
                print(f"    [{marker}] line {ln:4d} score={sc:.3f}  {code[:60]}")
            shown += 1
    print("=" * 65 + "\n")

    # ── Baseline comparison ─────────────────────────────────────────────────
    # NOTE: Direct comparison to published baselines (LineVul, WAVES, VulLMGNN)
    # is NOT valid here because:
    #   - They use full BigVul / different dataset splits
    #   - Localization denominators differ (all vulnerable vs flaw-GT-only subset)
    #   - Our task is multiclass (11 CWE classes); baselines are binary
    # Fair comparison requires running baselines on our exact dataset split.
    # See baselines.py for stored baseline numbers; comparison table will be
    # generated separately once LineVul / VulLMGNN are run on our split.
    print("\n" + "=" * 65)
    print("Baseline Comparison: DEFERRED")
    print("  Reason: different dataset splits and task definitions make")
    print("  direct comparison misleading. Run LineVul / VulLMGNN on the")
    print("  same split first. See baselines.py for stored reference numbers.")
    print("=" * 65 + "\n")

    # ── Save outputs ────────────────────────────────────────────────────────
    run_id = Path(args.checkpoint).parent.name
    results_dir = cfg.train.results_dir / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {results_dir}")

    # predictions.csv
    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence,
        "correct": correct_mask,
    })
    for i, name in enumerate(target_names):
        pred_df[f"prob_{name}"] = y_prob[:, i]
    csv_path = results_dir / "predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"predictions.csv → {csv_path}")

    # localization_scores.csv
    loc_rows: list[dict] = []
    for func_idx, (r, yt, yp) in enumerate(zip(loc_results, y_true, y_pred)):
        # Retrieve source code for this function
        raw_func = ""
        if raw_funcs is not None:
            ds_idx = test_idx[func_idx]
            raw_func = raw_funcs[ds_idx] if ds_idx < len(raw_funcs) else ""
        src_lines = raw_func.splitlines() if raw_func else []

        for ln, sc, lab in zip(r["line_numbers"], r["line_scores"], r["line_labels"]):
            code = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else ""
            loc_rows.append({
                "func_idx": func_idx,
                "y_true": int(yt),
                "y_pred": int(yp),
                "line_number": int(ln),
                "score": round(float(sc), 6),
                "is_flaw_line": int(lab),
                "code": code,
            })
    if loc_rows:
        loc_df = pd.DataFrame(loc_rows)
        loc_csv = results_dir / "localization_scores.csv"
        loc_df.to_csv(loc_csv, index=False)
        logger.info(f"localization_scores.csv → {loc_csv}")
    else:
        logger.warning("No localization data collected (node_line not in dataset).")

    # metrics_summary.json
    metrics_summary = {
        "function_level": {
            "accuracy": accuracy,
            "f1_macro": f1,
            "auc_roc_macro_ovr": _safe(auc_roc),
            "confidence_mean": float(confidence.mean()),
            "confidence_correct": (
                float(confidence[correct_mask].mean()) if correct_mask.any() else None
            ),
            "confidence_wrong": (
                float(confidence[~correct_mask].mean()) if (~correct_mask).any() else None
            ),
            "num_classes": num_classes,
            "num_test_samples": int(len(y_true)),
        },
        "localization": {
            "top_1_accuracy": _safe(loc_metrics["top_1_accuracy"]),
            "top_3_accuracy": _safe(loc_metrics["top_3_accuracy"]),
            "top_5_accuracy": _safe(loc_metrics["top_5_accuracy"]),
            "top_10_accuracy": _safe(loc_metrics["top_10_accuracy"]),
            "ifa_mean": _safe(loc_metrics["ifa_mean"]),
            "effort_at_20pct_recall": _safe(loc_metrics["effort_at_20pct_recall"]),
            "recall_at_1pct_loc": _safe(loc_metrics["recall_at_1pct_loc"]),
            "recall_at_5pct_loc": _safe(loc_metrics["recall_at_5pct_loc"]),
            "recall_at_20pct_loc": _safe(loc_metrics["recall_at_20pct_loc"]),
            "num_funcs_with_flaw_gt": n_gt,
        },
        "localization_curve": {
            "k_values": loc_metrics["recall_at_loc_curve_k"],
            "recall_values": [_safe(v) for v in loc_metrics["recall_at_loc_curve_v"]],
        },
        "ifa_distribution": loc_metrics["ifa_per_func"],
    }
    json_path = results_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"metrics_summary.json → {json_path}")

    # Plots
    plot_roc_curve(y_true, y_prob, results_dir / "roc_curve.png", class_names)
    plot_confusion_matrix(
        y_true, y_pred, num_classes, results_dir / "confusion_matrix.png", class_names
    )
    plot_pr_curve(y_true, y_prob, results_dir / "pr_curve.png", class_names)

    if n_gt > 0:
        plot_recall_at_loc_curve(
            loc_metrics["recall_at_loc_curve_k"],
            loc_metrics["recall_at_loc_curve_v"],
            results_dir / "recall_at_loc_curve.png",
        )
        plot_ifa_distribution(
            loc_metrics["ifa_per_func"],
            results_dir / "ifa_distribution.png",
        )

    logger.info(f"All results saved to {results_dir}/")


if __name__ == "__main__":
    main()
