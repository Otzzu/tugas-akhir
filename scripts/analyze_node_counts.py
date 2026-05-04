"""
scripts/analyze_node_counts.py — Node count statistics for raw CPG XML files.

Auto-discovers all datasets under --raw-dir (any subdirectory containing
benign/ or vulnerable/ with *.xml files).

Usage:
    uv run python scripts/analyze_node_counts.py
    uv run python scripts/analyze_node_counts.py --raw-dir data/raw --workers 8
    uv run python scripts/analyze_node_counts.py --datasets bigvul --out reports/nodes.md
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _count_nodes(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().count("<node ")
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(counts: list[int]) -> dict:
    arr = np.array(counts, dtype=np.float64)
    mode_val, mode_freq = Counter(counts).most_common(1)[0]
    pcts = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    return {
        "count":     len(arr),
        "mean":      float(arr.mean()),
        "std":       float(arr.std()),
        "min":       int(arr.min()),
        "p10":       int(pcts[0]),
        "p25":       int(pcts[1]),
        "median":    int(pcts[2]),
        "p75":       int(pcts[3]),
        "p90":       int(pcts[4]),
        "p95":       int(pcts[5]),
        "p99":       int(pcts[6]),
        "max":       int(arr.max()),
        "mode":      mode_val,
        "mode_freq": mode_freq,
        "over_500":  int((arr > 500).sum()),
        "over_1000": int((arr > 1000).sum()),
        "over_2000": int((arr > 2000).sum()),
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_stats(label: str, stats: dict) -> None:
    n = stats["count"]
    print(f"\n{'='*60}")
    print(f"  {label}  (n={n:,})")
    print(f"{'='*60}")
    print(f"  mean   : {stats['mean']:.1f}   std: {stats['std']:.1f}")
    print(f"  min    : {stats['min']}")
    print(f"  p10    : {stats['p10']}")
    print(f"  p25    : {stats['p25']}")
    print(f"  median : {stats['median']}")
    print(f"  p75    : {stats['p75']}")
    print(f"  p90    : {stats['p90']}")
    print(f"  p95    : {stats['p95']}")
    print(f"  p99    : {stats['p99']}")
    print(f"  max    : {stats['max']}")
    print(f"  mode   : {stats['mode']} (occurs {stats['mode_freq']}x)")
    print(f"  >500   : {stats['over_500']:,}  ({100*stats['over_500']/n:.1f}%)")
    print(f"  >1000  : {stats['over_1000']:,}  ({100*stats['over_1000']/n:.1f}%)")
    print(f"  >2000  : {stats['over_2000']:,}  ({100*stats['over_2000']/n:.1f}%)")


def print_histogram(counts: list[int], bins: list[int]) -> None:
    arr = np.array(counts)
    edges = bins + [max(int(arr.max()) + 1, bins[-1] + 1)]
    print("\n  Histogram:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (arr >= lo) & (arr < hi)
        c = int(mask.sum())
        bar = "#" * min(40, int(40 * c / len(arr)))
        lbl = f"[{lo:>5},{hi:>5})" if hi <= bins[-1] else f"[{lo:>5},  inf)"
        print(f"  {lbl}: {bar:<40} {c:>6} ({100*c/len(arr):5.1f}%)")


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def md_stats_table(stats: dict) -> str:
    n = stats["count"]
    pct = lambda k: f"{100*stats[k]/n:.1f}%"
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Count | {n:,} |",
        f"| Mean | {stats['mean']:.1f} |",
        f"| Std | {stats['std']:.1f} |",
        f"| Min | {stats['min']} |",
        f"| P10 | {stats['p10']} |",
        f"| P25 | {stats['p25']} |",
        f"| Median (P50) | {stats['median']} |",
        f"| P75 | {stats['p75']} |",
        f"| P90 | {stats['p90']} |",
        f"| P95 | {stats['p95']} |",
        f"| P99 | {stats['p99']} |",
        f"| Max | {stats['max']} |",
        f"| Mode | {stats['mode']} (×{stats['mode_freq']}) |",
        f"| > 500 | {stats['over_500']:,} ({pct('over_500')}) |",
        f"| > 1000 | {stats['over_1000']:,} ({pct('over_1000')}) |",
        f"| > 2000 | {stats['over_2000']:,} ({pct('over_2000')}) |",
    ]
    return "\n".join(lines)


def md_histogram(counts: list[int], bins: list[int]) -> str:
    arr = np.array(counts)
    edges = bins + [max(int(arr.max()) + 1, bins[-1] + 1)]
    lines = [
        "| Range | Count | % | Bar |",
        "|-------|-------|---|-----|",
    ]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (arr >= lo) & (arr < hi)
        c = int(mask.sum())
        pct = 100 * c / len(arr)
        bar = "█" * min(30, int(30 * c / len(arr)))
        lbl = f"[{lo}, {hi})" if hi <= bins[-1] else f"[{lo}, inf)"
        lines.append(f"| `{lbl}` | {c:,} | {pct:.1f}% | {bar} |")
    return "\n".join(lines)


def build_markdown(
    raw_dir: Path,
    datasets: list[str],
    splits: list[str],
    counts: dict[tuple[str, str], list[int]],
    bins: list[int],
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = [
        "# CPG Node Count Analysis",
        "",
        f"**Generated:** {ts}  ",
        f"**Raw dir:** `{raw_dir}`  ",
        f"**Datasets:** {', '.join(datasets)}",
        "",
    ]

    # Summary table across all dataset×split
    md += ["## Summary", "", "| Dataset | Split | Count | Mean | Median | P90 | P95 | P99 | Max |",
           "|---------|-------|-------|------|--------|-----|-----|-----|-----|"]
    for ds in datasets:
        for sp in splits:
            c = counts.get((ds, sp), [])
            if not c:
                continue
            s = compute_stats(c)
            md.append(
                f"| {ds} | {sp} | {s['count']:,} | {s['mean']:.0f} | {s['median']} "
                f"| {s['p90']} | {s['p95']} | {s['p99']} | {s['max']:,} |"
            )
        # combined row
        combined = []
        for sp in splits:
            combined.extend(counts.get((ds, sp), []))
        if combined:
            s = compute_stats(combined)
            md.append(
                f"| **{ds}** | **all** | **{s['count']:,}** | **{s['mean']:.0f}** "
                f"| **{s['median']}** | **{s['p90']}** | **{s['p95']}** "
                f"| **{s['p99']}** | **{s['max']:,}** |"
            )
    all_counts = [c for lst in counts.values() for c in lst]
    if all_counts:
        s = compute_stats(all_counts)
        md.append(
            f"| **ALL** | **all** | **{s['count']:,}** | **{s['mean']:.0f}** "
            f"| **{s['median']}** | **{s['p90']}** | **{s['p95']}** "
            f"| **{s['p99']}** | **{s['max']:,}** |"
        )

    # max_nodes recommendation — per dataset
    md += ["", "## max_nodes Recommendation", ""]
    PCTS = [("P75 (fast iter)", 75), ("P90 (balanced)", 90),
            ("P95 (recommended)", 95), ("P99 (max coverage)", 99)]
    for ds in datasets:
        vuln_counts = counts.get((ds, "vulnerable"), [])
        if not vuln_counts:
            continue
        arr = np.array(vuln_counts)
        md += [
            f"### {ds}",
            "",
            f"| Coverage Target | Suggested max_nodes | Notes |",
            f"|----------------|---------------------|-------|",
        ]
        for pct_label, pct_val in PCTS:
            val = int(np.percentile(arr, pct_val))
            md.append(f"| {pct_label} | {val} | truncates {100-pct_val}% of vuln graphs |")
        md.append("")
    # combined across all datasets
    all_vuln = []
    for ds in datasets:
        all_vuln.extend(counts.get((ds, "vulnerable"), []))
    if all_vuln and len(datasets) > 1:
        arr = np.array(all_vuln)
        md += [
            "### All datasets combined",
            "",
            "| Coverage Target | Suggested max_nodes | Notes |",
            "|----------------|---------------------|-------|",
        ]
        for pct_label, pct_val in PCTS:
            val = int(np.percentile(arr, pct_val))
            md.append(f"| {pct_label} | {val} | truncates {100-pct_val}% of vuln graphs |")
        md.append("")

    # Detailed sections per dataset × split
    md += ["", "---", ""]
    for ds in datasets:
        md += [f"## {ds.upper()}", ""]
        for sp in splits:
            c = counts.get((ds, sp), [])
            if not c:
                continue
            s = compute_stats(c)
            md += [
                f"### {ds} / {sp}",
                "",
                md_stats_table(s),
                "",
                md_histogram(c, bins),
                "",
            ]
        combined = []
        for sp in splits:
            combined.extend(counts.get((ds, sp), []))
        if combined:
            s = compute_stats(combined)
            md += [
                f"### {ds} — all splits combined",
                "",
                md_stats_table(s),
                "",
                md_histogram(combined, bins),
                "",
            ]

    # All combined
    if all_counts:
        s = compute_stats(all_counts)
        md += [
            "## All Datasets Combined",
            "",
            md_stats_table(s),
            "",
            md_histogram(all_counts, bins),
            "",
        ]

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Discovery + main
# ---------------------------------------------------------------------------

def discover_datasets(raw_dir: Path) -> list[str]:
    """Return subdirs of raw_dir that contain benign/ or vulnerable/ with XML files."""
    found = []
    for d in sorted(raw_dir.iterdir()):
        if not d.is_dir():
            continue
        has_xml = any(
            list((d / sp).glob("*.xml"))
            for sp in ("benign", "vulnerable")
            if (d / sp).exists()
        )
        if has_xml:
            found.append(d.name)
    return found


def collect_files(raw_dir: Path, dataset: str, split: str) -> list[str]:
    d = raw_dir / dataset / split
    if not d.exists():
        return []
    return [str(p) for p in d.glob("*.xml")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to analyze (default: auto-discover all in --raw-dir)"
    )
    parser.add_argument(
        "--out", default=None, type=Path,
        help="Output markdown path (default: reports/node_count_analysis.md)"
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    splits = ["benign", "vulnerable"]
    HIST_BINS = [0, 50, 100, 200, 500, 1000, 2000, 5000]

    datasets = args.datasets or discover_datasets(raw_dir)
    if not datasets:
        print(f"No datasets found in {raw_dir}")
        return
    print(f"Datasets: {datasets}")

    all_files: dict[tuple[str, str], list[str]] = {}
    for ds in datasets:
        for sp in splits:
            files = collect_files(raw_dir, ds, sp)
            all_files[(ds, sp)] = files
            print(f"  Found {len(files):>6} XML files — {ds}/{sp}")

    flat_files = [f for files in all_files.values() for f in files]
    print(f"\nTotal XML files: {len(flat_files):,} | Workers: {args.workers}")
    print("Counting nodes (fast string scan)…")

    with Pool(args.workers) as pool:
        flat_counts = pool.map(_count_nodes, flat_files)

    idx = 0
    counts: dict[tuple[str, str], list[int]] = {}
    for key, files in all_files.items():
        n = len(files)
        raw = flat_counts[idx : idx + n]
        counts[key] = [c for c in raw if c >= 0]
        idx += n

    # Console output
    for ds in datasets:
        for sp in splits:
            c = counts.get((ds, sp), [])
            if not c:
                continue
            print_stats(f"{ds.upper()} / {sp}", compute_stats(c))
            print_histogram(c, HIST_BINS)
        combined = []
        for sp in splits:
            combined.extend(counts.get((ds, sp), []))
        if combined:
            print_stats(f"{ds.upper()} — ALL", compute_stats(combined))
            print_histogram(combined, HIST_BINS)

    all_counts = [c for lst in counts.values() for c in lst]
    if all_counts and len(datasets) > 1:
        print_stats("ALL DATASETS COMBINED", compute_stats(all_counts))
        print_histogram(all_counts, HIST_BINS)

    # Markdown output
    out_path = args.out or Path("NODE_COUNT_ANALYSIS.md")
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    md = build_markdown(raw_dir, datasets, splits, counts, HIST_BINS)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown saved -> {out_path}")


if __name__ == "__main__":
    main()
