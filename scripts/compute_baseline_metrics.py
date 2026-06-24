#!/usr/bin/env python
"""compute_baseline_metrics.py — turn a baseline's dumped predictions/scores into OUR metrics.

Baselines don't save predictions, and each tool reports metrics with its own definitions
(LOSVER weighted-F1, LOSVER/LineVul native Top-k, EDAT global ranking). To compare fairly we
re-run each baseline in eval-only mode, dump two simple CSVs, and feed them here so every number
uses the SAME definitions as our model's metrics_summary.json (gnn_vuln.metrics.LocalizationMetrics
+ sklearn macro/weighted F1).

Input formats
-------------
--classification CSV : columns `y_true,y_pred` (one row per function; ints or label strings)
--localization   CSV : columns `func_id,line_number,score,is_flaw`
                       (one row per (function, line); is_flaw in {0,1}; score = higher == more suspicious)

Usage
-----
PYTHONPATH=src python scripts/compute_baseline_metrics.py \
    --name LOSVER \
    --classification preds_losver.csv \
    --localization   loc_losver.csv \
    --out results/baselines/losver_recomputed_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gnn_vuln.metrics import LocalizationMetrics  # noqa: E402


def classification_metrics(csv_path: str) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score, precision_score, recall_score)
    df = pd.read_csv(csv_path)
    assert {"y_true", "y_pred"} <= set(df.columns), f"{csv_path} needs y_true,y_pred columns"
    yt, yp = df["y_true"].tolist(), df["y_pred"].tolist()
    # macro averages over classes PRESENT in y_true (F1 undefined for a class with no samples).
    # Passing labels= makes the denominator independent of stray predictions into absent classes;
    # the sklearn default unique(y_true ∪ y_pred) would silently divide by +1 for such a class.
    macro_labels = sorted(set(yt))
    out = {
        "n": len(yt),
        "accuracy":     round(accuracy_score(yt, yp), 4),
        "f1_macro":     round(f1_score(yt, yp, average="macro", labels=macro_labels, zero_division=0), 4),
        "f1_weighted":  round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
        "f1_micro":     round(f1_score(yt, yp, average="micro", zero_division=0), 4),
        "precision_macro": round(precision_score(yt, yp, average="macro", labels=macro_labels, zero_division=0), 4),
        "recall_macro":    round(recall_score(yt, yp, average="macro", labels=macro_labels, zero_division=0), 4),
        "precision_weighted": round(precision_score(yt, yp, average="weighted", zero_division=0), 4),
        "recall_weighted":    round(recall_score(yt, yp, average="weighted", zero_division=0), 4),
    }
    out["report"] = classification_report(yt, yp, zero_division=0)
    return out


def localization_metrics(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    need = {"func_id", "line_number", "score", "is_flaw"}
    assert need <= set(df.columns), f"{csv_path} needs columns {need}"
    func_results = []
    for _fid, g in df.groupby("func_id"):
        func_results.append(LocalizationMetrics.make_result(
            line_numbers=g["line_number"].tolist(),
            line_scores=g["score"].astype(float).tolist(),
            line_labels=g["is_flaw"].astype(int).tolist(),
        ))
    m = LocalizationMetrics(func_results).to_dict()
    # drop bulky per-func arrays for the summary
    for k in ("recall_at_loc_curve_k", "recall_at_loc_curve_v", "ifa_per_func"):
        m.pop(k, None)
    m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    m["num_funcs_with_flaw_gt"] = sum(1 for r in func_results if r["num_flaw_lines"] > 0)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline", help="label for the printout")
    ap.add_argument("--classification", help="CSV with y_true,y_pred")
    ap.add_argument("--localization", help="CSV with func_id,line_number,score,is_flaw")
    ap.add_argument("--out", help="write the metrics dict as JSON here")
    args = ap.parse_args()

    result: dict = {"name": args.name}
    if args.classification:
        result["classification"] = classification_metrics(args.classification)
    if args.localization:
        result["localization"] = localization_metrics(args.localization)
    if not args.classification and not args.localization:
        ap.error("give --classification and/or --localization")

    # pretty print
    print(f"===== {args.name} =====")
    if "classification" in result:
        c = result["classification"]
        print(f"[classification] n={c['n']}  acc={c['accuracy']}  "
              f"macro-F1={c['f1_macro']}  weighted-F1={c['f1_weighted']}  "
              f"macro-P={c['precision_macro']}  macro-R={c['recall_macro']}")
        print(c["report"])
    if "localization" in result:
        l = result["localization"]
        print(f"[localization] funcs_with_flaw={l['num_funcs_with_flaw_gt']}  "
              f"IFA={l['ifa_mean']}  Top-1={l['top_1_accuracy']}  Top-5={l['top_5_accuracy']}  "
              f"Top-10={l['top_10_accuracy']}  R@1%={l['recall_at_1pct_loc']}  "
              f"R@5%={l['recall_at_5pct_loc']}  R@20%={l['recall_at_20pct_loc']}  "
              f"Effort@20%R={l['effort_at_20pct_recall']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
