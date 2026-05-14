"""
scripts/analyze_func_lines.py — Function line stats from .meta.json + CPG XML.

Reports two metrics per function:
  1. rel_lines  — len(raw_func.splitlines()) from .meta.json  (relative, function length)
  2. max_abs_line — max lineNumber across all CPG nodes in .xml (absolute file line number)

max_abs_line verifies MAX_LINE=100_000 safety in StmtHead._score_vectorized:
    sid = batch * MAX_LINE + node_line
If any node_line >= MAX_LINE, scatter IDs collide → wrong stmt grouping.

Usage:
    uv run python scripts/analyze_func_lines.py
    uv run python scripts/analyze_func_lines.py --raw-dir data/raw --workers 4
    uv run python scripts/analyze_func_lines.py --datasets megavul
    uv run python scripts/analyze_func_lines.py --out reports/func_lines.md
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np


_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
_LINE_KEYS: set[str] = {"LINE_NUMBER", "lineNumber"}

LIMITS_REL  = [10, 20, 50, 100, 200, 500, 1_000]
LIMITS_ABS  = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
HIST_BINS_REL = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 500]
HIST_BINS_ABS = [0, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]


# ── Worker — returns (rel_lines, max_abs_line); -1 on error ──────────────────

def _analyze(xml_path: str) -> tuple[int, int]:
    try:
        p = Path(xml_path)

        # 1. Relative line count from meta.json
        rel = -1
        meta = p.parent / f"{p.stem}.meta.json"
        if meta.exists():
            with open(meta, encoding="utf-8", errors="replace") as f:
                d = json.load(f)
            raw_func = d.get("raw_func") or ""
            if raw_func:
                rel = len(raw_func.splitlines())

        # 2. Max absolute lineNumber from CPG XML nodes
        max_abs = -1
        tag_data = f"{{{_GRAPHML_NS}}}data"
        tag_key  = f"{{{_GRAPHML_NS}}}key"
        tree = ET.parse(p)
        root = tree.getroot()

        # Build key_id → attr_name map
        key_map: dict[str, str] = {}
        for k in root.iter(tag_key):
            kid   = k.get("id", "")
            aname = k.get("attr.name", kid)
            key_map[kid] = aname

        line_keys = {kid for kid, aname in key_map.items() if aname in _LINE_KEYS}

        for data_el in root.iter(tag_data):
            if data_el.get("key", "") in line_keys:
                try:
                    v = int(data_el.text or "")
                    if v > max_abs:
                        max_abs = v
                except (ValueError, TypeError):
                    pass

        return rel, max_abs
    except Exception:
        return -1, -1


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(values: list[int], limits: list[int]) -> dict:
    arr = np.array(values, dtype=np.float64)
    mode_val, mode_freq = Counter(values).most_common(1)[0]
    pcts = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    s = {
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
    }
    for lim in limits:
        s[f"over_{lim}"] = int((arr > lim).sum())
    return s


def print_stats(label: str, stats: dict, limits: list[int]) -> None:
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
    for lim in limits:
        print(f"  >{lim:<7}: {stats[f'over_{lim}']:,}  ({100*stats[f'over_{lim}']/n:.2f}%)")


def print_histogram(values: list[int], bins: list[int]) -> None:
    arr = np.array(values)
    edges = bins + [max(int(arr.max()) + 1, bins[-1] + 1)]
    print("\n  Histogram:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (arr >= lo) & (arr < hi)
        c = int(mask.sum())
        bar = "#" * min(40, int(40 * c / len(arr)))
        lbl = f"[{lo:>7},{hi:>7})" if hi <= bins[-1] else f"[{lo:>7},     inf)"
        print(f"  {lbl}: {bar:<40} {c:>6} ({100*c/len(arr):5.1f}%)")


def md_stats_table(stats: dict, limits: list[int], unit: str = "lines") -> str:
    n = stats["count"]
    rows = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Count | {n:,} |",
        f"| Mean | {stats['mean']:.1f} |",
        f"| Std | {stats['std']:.1f} |",
        f"| Min | {stats['min']} |",
        f"| P10 | {stats['p10']} |",
        f"| P25 | {stats['p25']} |",
        f"| Median | {stats['median']} |",
        f"| P75 | {stats['p75']} |",
        f"| P90 | {stats['p90']} |",
        f"| P95 | {stats['p95']} |",
        f"| P99 | {stats['p99']} |",
        f"| Max | {stats['max']} |",
        f"| Mode | {stats['mode']} (×{stats['mode_freq']}) |",
    ]
    for lim in limits:
        k = f"over_{lim}"
        rows.append(f"| > {lim:,} {unit} | {stats[k]:,} ({100*stats[k]/n:.2f}%) |")
    return "\n".join(rows)


def md_histogram(values: list[int], bins: list[int]) -> str:
    arr = np.array(values)
    edges = bins + [max(int(arr.max()) + 1, bins[-1] + 1)]
    rows = [
        "| Range | Count | % | Bar |",
        "|-------|-------|---|-----|",
    ]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (arr >= lo) & (arr < hi)
        c = int(mask.sum())
        pct = 100 * c / len(arr)
        bar = "█" * min(30, int(30 * c / len(arr)))
        lbl = f"[{lo}, {hi})" if hi <= bins[-1] else f"[{lo}, inf)"
        rows.append(f"| `{lbl}` | {c:,} | {pct:.1f}% | {bar} |")
    return "\n".join(rows)


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_datasets(raw_dir: Path) -> list[str]:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",  default="data/raw", type=Path)
    parser.add_argument("--workers",  type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--out",      default=None, type=Path)
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    splits = ["benign", "vulnerable"]

    datasets = args.datasets or discover_datasets(raw_dir)
    if not datasets:
        print(f"No datasets found in {raw_dir}")
        return
    print(f"Datasets : {datasets}")

    all_files: dict[tuple[str, str], list[str]] = {}
    for ds in datasets:
        for sp in splits:
            files = collect_files(raw_dir, ds, sp)
            all_files[(ds, sp)] = files
            print(f"  {len(files):>6} XML files — {ds}/{sp}")

    flat_files = [f for files in all_files.values() for f in files]
    print(f"\nTotal: {len(flat_files):,} files | Workers: {args.workers}")
    print("Parsing XML + meta.json ...")

    with Pool(processes=args.workers) as pool:
        flat_results = pool.map(_analyze, flat_files)

    idx = 0
    rel_counts: dict[tuple[str, str], list[int]] = {}
    abs_counts: dict[tuple[str, str], list[int]] = {}
    for key, files in all_files.items():
        n = len(files)
        chunk = flat_results[idx: idx + n]
        rel_counts[key] = [r for r, _ in chunk if r >= 0]
        abs_counts[key] = [a for _, a in chunk if a >= 0]
        idx += n

    # ── Console: relative lines ───────────────────────────────────────────────
    print("\n\n" + "█"*60)
    print("  METRIC 1: Function length (relative lines from raw_func)")
    print("█"*60)
    for ds in datasets:
        for sp in splits:
            c = rel_counts.get((ds, sp), [])
            if c:
                print_stats(f"{ds} / {sp}  [rel lines]", compute_stats(c, LIMITS_REL), LIMITS_REL)
                print_histogram(c, HIST_BINS_REL)
        combined = []
        for sp in splits:
            combined.extend(rel_counts.get((ds, sp), []))
        if combined:
            print_stats(f"{ds} — ALL  [rel lines]", compute_stats(combined, LIMITS_REL), LIMITS_REL)
            print_histogram(combined, HIST_BINS_REL)

    # ── Console: absolute line numbers ────────────────────────────────────────
    print("\n\n" + "█"*60)
    print("  METRIC 2: Max absolute lineNumber per function (CPG XML)")
    print("  (MAX_LINE=100,000 safety check for StmtHead scatter)")
    print("█"*60)
    all_abs: list[int] = []
    for ds in datasets:
        for sp in splits:
            c = abs_counts.get((ds, sp), [])
            if c:
                print_stats(f"{ds} / {sp}  [abs lineNumber]", compute_stats(c, LIMITS_ABS), LIMITS_ABS)
                print_histogram(c, HIST_BINS_ABS)
                all_abs.extend(c)
        combined = []
        for sp in splits:
            combined.extend(abs_counts.get((ds, sp), []))
        if combined:
            print_stats(f"{ds} — ALL  [abs lineNumber]", compute_stats(combined, LIMITS_ABS), LIMITS_ABS)
            print_histogram(combined, HIST_BINS_ABS)

    if all_abs:
        global_max = max(all_abs)
        safe = global_max < 100_000
        print(f"\n{'='*60}")
        print(f"  MAX_LINE=100,000 safe? : {'YES' if safe else 'NO  <-- COLLISION RISK'}")
        print(f"  Global max node_line   : {global_max:,}")
        print(f"  Suggested MAX_LINE     : {max(global_max * 2, 10_000):,}  (2x margin)")
        print(f"{'='*60}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = [
        "# CPG Function Line Analysis",
        "",
        f"**Generated:** {ts}  ",
        f"**Raw dir:** `{raw_dir}`  ",
        f"**Datasets:** {', '.join(datasets)}",
        "",
        "Two metrics per function:",
        "- **rel_lines**: `len(raw_func.splitlines())` — function length in source lines",
        "- **max_abs_line**: max `lineNumber` across all CPG nodes — absolute file line number",
        "",
        "> `max_abs_line` verifies `MAX_LINE=100_000` safety in `StmtHead._score_vectorized`:",
        "> `sid = batch * MAX_LINE + node_line`. If any `node_line >= MAX_LINE` → ID collision → wrong stmt grouping.",
        "",
    ]

    # MAX_LINE safety summary
    if all_abs:
        global_max = max(all_abs)
        safe = global_max < 100_000
        md += [
            "## MAX_LINE Safety Check",
            "",
            "| | |",
            "|---|---|",
            f"| Global max `node_line` | **{global_max:,}** |",
            f"| MAX_LINE=100,000 safe? | **{'YES ✓' if safe else 'NO ✗ — increase MAX_LINE'}** |",
            f"| Suggested MAX_LINE | **{max(global_max * 2, 10_000):,}** (2× margin) |",
            "",
        ]

    # Summary tables
    md += [
        "## Summary — Function Length (rel_lines)",
        "",
        "| Dataset | Split | Count | Mean | Median | P90 | P95 | P99 | Max |",
        "|---------|-------|-------|------|--------|-----|-----|-----|-----|",
    ]
    for ds in datasets:
        for sp in splits:
            c = rel_counts.get((ds, sp), [])
            if not c:
                continue
            s = compute_stats(c, LIMITS_REL)
            md.append(f"| {ds} | {sp} | {s['count']:,} | {s['mean']:.0f} | {s['median']} "
                      f"| {s['p90']} | {s['p95']} | {s['p99']} | {s['max']} |")
        combined = []
        for sp in splits:
            combined.extend(rel_counts.get((ds, sp), []))
        if combined:
            s = compute_stats(combined, LIMITS_REL)
            md.append(f"| **{ds}** | **all** | **{s['count']:,}** | **{s['mean']:.0f}** "
                      f"| **{s['median']}** | **{s['p90']}** | **{s['p95']}** "
                      f"| **{s['p99']}** | **{s['max']}** |")

    md += [
        "",
        "## Summary — Max Absolute lineNumber (max_abs_line)",
        "",
        "| Dataset | Split | Count | Mean | Median | P90 | P95 | P99 | Max | >100k |",
        "|---------|-------|-------|------|--------|-----|-----|-----|-----|-------|",
    ]
    for ds in datasets:
        for sp in splits:
            c = abs_counts.get((ds, sp), [])
            if not c:
                continue
            s = compute_stats(c, LIMITS_ABS)
            n = s["count"]
            md.append(f"| {ds} | {sp} | {n:,} | {s['mean']:.0f} | {s['median']} "
                      f"| {s['p90']} | {s['p95']} | {s['p99']} | {s['max']} "
                      f"| {100*s['over_100000']/n:.2f}% |")
        combined = []
        for sp in splits:
            combined.extend(abs_counts.get((ds, sp), []))
        if combined:
            s = compute_stats(combined, LIMITS_ABS)
            n = s["count"]
            md.append(f"| **{ds}** | **all** | **{n:,}** | **{s['mean']:.0f}** "
                      f"| **{s['median']}** | **{s['p90']}** | **{s['p95']}** "
                      f"| **{s['p99']}** | **{s['max']}** | **{100*s['over_100000']/n:.2f}%** |")

    # Detailed sections
    md += ["", "---", ""]
    for ds in datasets:
        md += [f"## {ds.upper()}", ""]
        for sp in splits:
            rc = rel_counts.get((ds, sp), [])
            ac = abs_counts.get((ds, sp), [])
            if rc:
                md += [f"### {ds} / {sp} — rel_lines", "",
                       md_stats_table(compute_stats(rc, LIMITS_REL), LIMITS_REL, "lines"), "",
                       md_histogram(rc, HIST_BINS_REL), ""]
            if ac:
                md += [f"### {ds} / {sp} — max_abs_line", "",
                       md_stats_table(compute_stats(ac, LIMITS_ABS), LIMITS_ABS, "abs line"), "",
                       md_histogram(ac, HIST_BINS_ABS), ""]

    out_path = args.out or Path("FUNC_LINE_ANALYSIS.md")
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown saved → {out_path}")


if __name__ == "__main__":
    main()
