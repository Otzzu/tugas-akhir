"""
evaluate.py — Evaluation entry point.

Usage:
    uv run evaluate --checkpoint checkpoints/<run_id>/best_*.pt
    uv run evaluate --checkpoint checkpoints/<run_id>/best_*.pt --config configs/...yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from gnn_vuln.evaluation.localize import LocalizationExtractor
from gnn_vuln.evaluation.plots import ResultPlotter
from gnn_vuln.metrics import LocalizationMetrics
from gnn_vuln.models.registry import build_model
from gnn_vuln.utils import get_device, load_checkpoint, setup_logging


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    End-to-end evaluation: runs inference, computes metrics, writes all outputs.

    Outputs saved to results_dir/:
      predictions.csv, localization_scores.csv, metrics_summary.json,
      roc_curve.png, confusion_matrix.png, pr_curve.png,
      recall_at_loc_curve.png, ifa_distribution.png
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: CodeBERTGraphDataset,
        test_idx: list[int],
        device: torch.device,
        results_dir: Path,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.test_idx = test_idx
        self.device = device
        self.results_dir = Path(results_dir)
        self.batch_size = batch_size
        self.checkpoint_path: str = ""

        self.class_names: list[str] | None = getattr(dataset, "class_names", None)
        self.raw_funcs = getattr(dataset, "raw_funcs", None)
        self._loader = DataLoader(dataset[test_idx], batch_size=batch_size)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run full evaluation. Returns metrics_summary dict."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running inference…")
        extractor = LocalizationExtractor(self.model, self._loader, self.device)
        y_true, y_pred, y_prob, confidence, loc_results = extractor.run()

        target_names = self.class_names or [str(i) for i in range(y_prob.shape[1])]
        correct_mask = y_true == y_pred

        func_metrics = self._function_level(y_true, y_pred, y_prob, confidence,
                                             correct_mask, target_names)
        loc_metrics  = LocalizationMetrics(loc_results)

        self._print_report(y_true, y_pred, target_names, func_metrics, loc_metrics,
                           loc_results, y_true, confidence, correct_mask)
        summary = self._save_all(y_true, y_pred, y_prob, confidence, correct_mask,
                                 target_names, loc_results, func_metrics, loc_metrics)
        return summary

    # ------------------------------------------------------------------
    # Function-level metrics
    # ------------------------------------------------------------------

    def _function_level(self, y_true, y_pred, y_prob, confidence, correct_mask, target_names) -> dict:
        n_classes = y_prob.shape[1]
        try:
            if n_classes == 2:
                auc_roc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                present = np.unique(y_true)
                y_p = y_prob[:, present]
                y_p = y_p / y_p.sum(axis=1, keepdims=True)
                auc_roc = roc_auc_score(y_true, y_p, multi_class="ovr",
                                        average="macro", labels=present)
        except ValueError:
            auc_roc = float("nan")

        return {
            "accuracy": float((y_true == y_pred).mean()),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "auc_roc_macro_ovr": auc_roc,
            "confidence_mean":    float(confidence.mean()),
            "confidence_correct": float(confidence[correct_mask].mean()) if correct_mask.any() else None,
            "confidence_wrong":   float(confidence[~correct_mask].mean()) if (~correct_mask).any() else None,
            "num_classes": n_classes,
            "num_test_samples": int(len(y_true)),
        }

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def _print_report(self, y_true, y_pred, target_names, func_metrics,
                      loc_metrics: LocalizationMetrics, loc_results, yt_arr,
                      confidence, correct_mask) -> None:
        n = len(target_names)
        print("\n" + "=" * 65)
        print("Function-Level Classification Report")
        print("=" * 65)
        print(classification_report(y_true, y_pred, labels=list(range(n)),
                                    target_names=target_names, zero_division=0))
        print(f"AUC-ROC (macro OvR) : {func_metrics['auc_roc_macro_ovr']:.4f}")
        print(f"F1 Score (macro)    : {func_metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted) : {func_metrics['f1_weighted']:.4f}")
        print(f"Accuracy            : {func_metrics['accuracy']:.4f}")
        print("=" * 65)

        n_gt = loc_metrics.num_funcs_with_flaw_gt
        print(f"\n{'=' * 65}")
        print(f"Statement-Level Localization  (functions with flaw GT: {n_gt})")
        print("=" * 65)
        if n_gt == 0:
            print("  No flaw-line ground truth found.")
        else:
            d = loc_metrics.to_dict()
            print(f"  Top-1  Accuracy    : {d['top_1_accuracy']:.4f}")
            print(f"  Top-3  Accuracy    : {d['top_3_accuracy']:.4f}")
            print(f"  Top-5  Accuracy    : {d['top_5_accuracy']:.4f}")
            print(f"  Top-10 Accuracy    : {d['top_10_accuracy']:.4f}")
            print(f"  IFA (mean)         : {d['ifa_mean']:.2f}")
            print(f"  Effort@20%Recall   : {d['effort_at_20pct_recall']:.4f}")
            print(f"  Recall@1%LOC       : {d['recall_at_1pct_loc']:.4f}")
            print(f"  Recall@5%LOC       : {d['recall_at_5pct_loc']:.4f}")
            print(f"  Recall@20%LOC      : {d['recall_at_20pct_loc']:.4f}")
            self._print_sample_lines(loc_results, yt_arr)
        print("=" * 65 + "\n")

    def _print_sample_lines(self, loc_results, y_true) -> None:
        print()
        print("  Sample — top-3 suspicious lines (first 3 vulnerable functions):")
        shown = 0
        for func_idx, (r, yt) in enumerate(zip(loc_results, y_true)):
            if int(yt) == 0 or shown >= 3:
                continue
            src_lines = self._get_src_lines(func_idx)
            print(f"  func {func_idx} (class={int(yt)}):")
            for ln, sc, lab in zip(r["ranked_line_numbers"][:3],
                                   r["ranked_scores"][:3], r["ranked_labels"][:3]):
                code   = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else "<no code>"
                marker = "FLAW" if lab else "    "
                print(f"    [{marker}] line {ln:4d} score={sc:.3f}  {code[:60]}")
            shown += 1

    def _get_src_lines(self, func_idx: int) -> list[str]:
        if self.raw_funcs is None:
            return []
        ds_idx = self.test_idx[func_idx]
        raw = self.raw_funcs[ds_idx] if ds_idx < len(self.raw_funcs) else ""
        return raw.splitlines() if raw else []

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def _safe(self, v) -> object:
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    def _save_all(self, y_true, y_pred, y_prob, confidence, correct_mask,
                  target_names, loc_results, func_metrics, loc_metrics: LocalizationMetrics) -> dict:
        rd = self.results_dir

        # predictions.csv
        pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred,
                                 "confidence": confidence, "correct": correct_mask})
        for i, name in enumerate(target_names):
            pred_df[f"prob_{name}"] = y_prob[:, i]
        pred_df.to_csv(rd / "predictions.csv", index=False)
        logger.info(f"predictions.csv → {rd/'predictions.csv'}")

        # localization_scores.csv
        loc_rows: list[dict] = []
        for func_idx, (r, yt, yp) in enumerate(zip(loc_results, y_true, y_pred)):
            src_lines = self._get_src_lines(func_idx)
            for ln, sc, lab in zip(r["line_numbers"], r["line_scores"], r["line_labels"]):
                code = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else ""
                loc_rows.append({"func_idx": func_idx, "y_true": int(yt), "y_pred": int(yp),
                                  "line_number": int(ln), "score": round(float(sc), 6),
                                  "is_flaw_line": int(lab), "code": code})
        if loc_rows:
            pd.DataFrame(loc_rows).to_csv(rd / "localization_scores.csv", index=False)
            logger.info(f"localization_scores.csv → {rd/'localization_scores.csv'}")
        else:
            logger.warning("No localization data collected (node_line not in dataset).")

        # metrics_summary.json
        d = loc_metrics.to_dict()
        n_gt = loc_metrics.num_funcs_with_flaw_gt
        summary = {
            "function_level": {k: self._safe(v) for k, v in func_metrics.items()},
            "localization": {
                "top_1_accuracy":         self._safe(d["top_1_accuracy"]),
                "top_3_accuracy":         self._safe(d["top_3_accuracy"]),
                "top_5_accuracy":         self._safe(d["top_5_accuracy"]),
                "top_10_accuracy":        self._safe(d["top_10_accuracy"]),
                "ifa_mean":               self._safe(d["ifa_mean"]),
                "effort_at_20pct_recall": self._safe(d["effort_at_20pct_recall"]),
                "recall_at_1pct_loc":     self._safe(d["recall_at_1pct_loc"]),
                "recall_at_5pct_loc":     self._safe(d["recall_at_5pct_loc"]),
                "recall_at_20pct_loc":    self._safe(d["recall_at_20pct_loc"]),
                "num_funcs_with_flaw_gt": n_gt,
            },
            "localization_curve": {
                "k_values":     d["recall_at_loc_curve_k"],
                "recall_values":[self._safe(v) for v in d["recall_at_loc_curve_v"]],
            },
            "ifa_distribution": d["ifa_per_func"],
        }
        with open(rd / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"metrics_summary.json → {rd/'metrics_summary.json'}")

        # Plots
        plotter = ResultPlotter(rd)
        plotter.plot_roc_curve(y_true, y_prob, self.class_names or target_names)
        plotter.plot_confusion_matrix(y_true, y_pred, target_names)
        plotter.plot_pr_curve(y_true, y_prob, self.class_names or target_names)
        if n_gt > 0:
            k_vals, recall_vals = loc_metrics.recall_at_loc_curve
            plotter.plot_recall_at_loc_curve(k_vals, recall_vals)
            plotter.plot_ifa_distribution(d["ifa_per_func"])

        # Copy config + training files from checkpoint dir to results dir
        ckpt_dir = Path(self.checkpoint_path).parent
        for fname in ("config.yaml", "training_log.csv", "training_summary.json", "training_curves.png"):
            src = ckpt_dir / fname
            if src.exists():
                import shutil as _shutil
                _shutil.copy2(src, rd / fname)
                logger.info(f"Copied {fname} → {rd/fname}")

        logger.info(f"All results saved to {rd}/")
        return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained vulnerability detector")
    parser.add_argument("--checkpoint", required=True, help="Path to best_*.pt checkpoint")
    parser.add_argument("--config", default=None, help="Config YAML (auto-detected if omitted)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else Path(args.checkpoint).parent / "config.yaml"
    cfg = Config.from_yaml(config_path) if config_path.exists() else load_default_config()
    if not (args.config or config_path.exists()):
        logger.warning("No config.yaml found, using defaults.")

    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    pretrained_lm    = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    func_lm          = getattr(cfg.model, "func_lm", "") or pretrained_lm
    add_func_tokens  = getattr(cfg.model, "add_func_tokens", False)
    func_lm_source   = getattr(cfg.model, "func_lm_source", "raw")

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
        filter_owasp=getattr(cfg.data, "filter_owasp", False),
        filter_top25_dangerous=getattr(cfg.data, "filter_top25_dangerous", False),
        max_per_class=getattr(cfg.data, "max_per_class", 0),
        resample_seed=getattr(cfg.data, "resample_seed", 42),
        func_max_length=getattr(cfg.model, "func_max_length", 512),
        storage=getattr(cfg.data, "storage", "inmemory"),
    )
    _, _, test_idx = dataset.get_splits(seed=cfg.train.seed)

    in_channels = dataset[0].x.size(1)
    model = build_model(cfg, in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    logger.info(f"Model loaded from {args.checkpoint}")

    run_id = Path(args.checkpoint).parent.name
    results_dir = cfg.train.results_dir / run_id

    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        test_idx=test_idx,
        device=device,
        results_dir=results_dir,
        batch_size=cfg.train.batch_size,
    )
    evaluator.checkpoint_path = args.checkpoint
    evaluator.run()


if __name__ == "__main__":
    main()
