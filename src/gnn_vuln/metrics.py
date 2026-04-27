"""
metrics.py — Vulnerability localization metrics.

Adapted from:
  LineVul  (Fu & Tantithamthavorn, MSR 2022) — Top-10 Acc, IFA, Effort@K%Recall, Recall@K%LOC
  WAVES    (Ni et al., 2023)                  — same metrics for MIL-trained statement head
  VulLMGNN (Cao et al., ICSE 2023)            — Accuracy, Precision, Recall, F1, AUC-ROC
  VulChecker (Mirsky et al., USENIX 2023)     — Precision, Recall, F1

Localization metric definitions (applied to functions with flaw-line ground truth):
  Top-10 Accuracy  : fraction of functions where ≥1 flaw line is in the top-10 ranked lines
  IFA              : mean clean lines inspected before the first flaw line (lower = better)
  Effort@20%Recall : fraction of total lines (globally) to inspect to catch 20% of flaw lines
                     (lower = better; matches LineVul global-aggregation definition)
  Recall@K%LOC     : fraction of flaw lines caught when inspecting top-K% of all lines globally
                     (higher = better; K=1 is the hardest cutoff, K=20 is the standard one)

Only functions with at least one flaw-line ground-truth node (flaw_line_mask > 0) are
included in localization metrics. If no such functions exist, NaN is returned.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-function result builder
# ---------------------------------------------------------------------------

def make_func_loc_result(
    line_numbers: list[int],
    line_scores: list[float],
    line_labels: list[int],
) -> dict:
    """
    Build a canonical per-function localization result dict.

    Parameters
    ----------
    line_numbers : source line numbers (ascending, matching line_scores order)
    line_scores  : MIL sigmoid score per unique source line (same order as line_numbers)
    line_labels  : 1=flaw line, 0=clean (same order as line_numbers)

    Returns
    -------
    dict with fields:
      line_numbers, line_scores, line_labels  — original (line-number) order
      ranked_scores, ranked_labels, ranked_line_numbers  — sorted by score desc
      num_lines, num_flaw_lines
    """
    assert len(line_numbers) == len(line_scores) == len(line_labels)
    order = sorted(range(len(line_scores)), key=lambda i: line_scores[i], reverse=True)
    return {
        "line_numbers": line_numbers,
        "line_scores": line_scores,
        "line_labels": line_labels,
        "ranked_scores": [line_scores[i] for i in order],
        "ranked_labels": [line_labels[i] for i in order],
        "ranked_line_numbers": [line_numbers[i] for i in order],
        "num_lines": len(line_scores),
        "num_flaw_lines": sum(line_labels),
    }


# ---------------------------------------------------------------------------
# Individual localization metrics
# ---------------------------------------------------------------------------

def top_k_accuracy(func_results: list[dict], k: int) -> float:
    """
    Fraction of functions (with flaw GT) where ≥1 flaw line appears in top-k
    ranked statements. Matches LineVul/WAVES Top-k Accuracy definition.
    """
    correct = total = 0
    for r in func_results:
        if r["num_flaw_lines"] == 0:
            continue
        total += 1
        if 1 in r["ranked_labels"][:k]:
            correct += 1
    return correct / total if total > 0 else float("nan")


def top_10_accuracy(func_results: list[dict]) -> float:
    return top_k_accuracy(func_results, k=10)


def ifa_metric(func_results: list[dict]) -> tuple[float, list[float]]:
    """
    Initial False Alarm: clean lines inspected before the first flaw line.
    Matches LineVul's IFA definition.

    Returns
    -------
    (mean_IFA, per_function_IFA_list)
    """
    values: list[float] = []
    for r in func_results:
        if r["num_flaw_lines"] == 0:
            continue
        clean = 0
        for label in r["ranked_labels"]:
            if label == 1:
                break
            clean += 1
        values.append(float(clean))
    mean_ifa = float(np.mean(values)) if values else float("nan")
    return mean_ifa, values


def effort_at_k_recall(func_results: list[dict], k: float = 0.20) -> float:
    """
    Fraction of total lines (globally) to inspect to catch k-fraction of all flaw lines.

    All lines from all functions with flaw GT are ranked globally by score descending,
    matching LineVul's global Effort@K%Recall definition.
    """
    all_scored: list[tuple[float, int]] = []
    total_lines = total_flaw = 0
    for r in func_results:
        if r["num_flaw_lines"] == 0:
            continue
        for sc, lab in zip(r["ranked_scores"], r["ranked_labels"]):
            all_scored.append((sc, lab))
        total_lines += r["num_lines"]
        total_flaw += r["num_flaw_lines"]

    if total_flaw == 0 or total_lines == 0:
        return float("nan")

    all_scored.sort(key=lambda x: x[0], reverse=True)
    target = max(1, int(total_flaw * k))
    caught = inspected = 0
    for _, lab in all_scored:
        inspected += 1
        caught += lab
        if caught >= target:
            break
    return round(inspected / total_lines, 4)


def recall_at_k_loc(func_results: list[dict], k: float = 0.01) -> float:
    """
    Fraction of flaw lines caught when inspecting the top-K% of all lines globally.
    Lines from all functions with flaw GT ranked globally by score descending.
    Matches LineVul's Recall@K%LOC definition.
    """
    all_scored: list[tuple[float, int]] = []
    total_lines = total_flaw = 0
    for r in func_results:
        if r["num_flaw_lines"] == 0:
            continue
        for sc, lab in zip(r["ranked_scores"], r["ranked_labels"]):
            all_scored.append((sc, lab))
        total_lines += r["num_lines"]
        total_flaw += r["num_flaw_lines"]

    if total_flaw == 0 or total_lines == 0:
        return float("nan")

    all_scored.sort(key=lambda x: x[0], reverse=True)
    target_inspect = max(1, int(total_lines * k))
    caught = sum(lab for _, lab in all_scored[:target_inspect])
    return round(caught / total_flaw, 4)


def recall_at_k_loc_curve(
    func_results: list[dict],
    k_values: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """
    Compute Recall@K%LOC for a range of K values (for plotting the curve).
    Returns (k_values, recall_values).
    """
    if k_values is None:
        k_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    recalls = [recall_at_k_loc(func_results, k) for k in k_values]
    return k_values, recalls


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_localization_metrics(func_results: list[dict]) -> dict:
    """
    Compute all localization metrics from a list of per-function result dicts.

    Returns a dict of scalar metrics plus curve arrays for plotting.
    NaN is returned for any metric when no functions have flaw GT.
    """
    mean_ifa, ifa_list = ifa_metric(func_results)
    k_vals, recall_curve = recall_at_k_loc_curve(func_results)
    num_gt = sum(1 for r in func_results if r["num_flaw_lines"] > 0)

    return {
        # Comparable to LineVul Table 5 / WAVES Table 3
        "top_1_accuracy": top_k_accuracy(func_results, k=1),
        "top_3_accuracy": top_k_accuracy(func_results, k=3),
        "top_5_accuracy": top_k_accuracy(func_results, k=5),
        "top_10_accuracy": top_10_accuracy(func_results),
        "ifa_mean": mean_ifa,
        "effort_at_20pct_recall": effort_at_k_recall(func_results, k=0.20),
        "recall_at_1pct_loc": recall_at_k_loc(func_results, k=0.01),
        "recall_at_5pct_loc": recall_at_k_loc(func_results, k=0.05),
        "recall_at_20pct_loc": recall_at_k_loc(func_results, k=0.20),
        # Curve for plot
        "recall_at_loc_curve_k": k_vals,
        "recall_at_loc_curve_v": recall_curve,
        # For IFA distribution plot / diagnostics
        "ifa_per_func": ifa_list,
        "num_funcs_with_flaw_gt": num_gt,
    }
