"""
scripts/analyze_func_tokens.py — Token count statistics for function CPG XML files.

Extracts all CODE text from CPG nodes, tokenizes with a HuggingFace tokenizer,
and reports how many functions exceed common LM token limits (512, 1024, 2048).

Auto-discovers all datasets under --raw-dir (same layout as analyze_node_counts.py).

Usage:
    uv run python scripts/analyze_func_tokens.py
    uv run python scripts/analyze_func_tokens.py --raw-dir data/raw --workers 4
    uv run python scripts/analyze_func_tokens.py --tokenizer microsoft/unixcoder-base
    uv run python scripts/analyze_func_tokens.py --datasets bigvul --out reports/tokens.md
    uv run python scripts/analyze_func_tokens.py --char-estimate   # fast, no tokenizer
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))

import numpy as np

# ---------------------------------------------------------------------------
# Worker (tokenizer initialized once per process via Pool initializer)
# ---------------------------------------------------------------------------

_TOKENIZER = None


def _init_worker(tokenizer_name: str | None) -> None:
    global _TOKENIZER
    if tokenizer_name:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def _count_tokens(path: str) -> int:
    """
    Count tokens for a function. Uses raw_func from .meta.json when available
    (actual source text). Falls back to CPG reconstruction via build_func_text.
    """
    try:
        import json
        from pathlib import Path as _P
        p = _P(path)
        meta_path = p.parent / f"{p.stem}.meta.json"
        text = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            text = meta.get("raw_func") or None

        if text is None:
            from gnn_vuln.data.graph_builder_lm import parse_cpg, build_func_text
            cpg = parse_cpg(path, max_nodes=999_999)
            if cpg is None:
                return 0
            text = build_func_text(cpg)

        if not text:
            return 0
        if _TOKENIZER is not None:
            return len(_TOKENIZER.encode(text, add_special_tokens=False))
        return max(1, len(text) // 4)
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

LIMITS = [128, 256, 512, 1024, 2048]


def compute_stats(counts: list[int]) -> dict:
    arr = np.array(counts, dtype=np.float64)
    mode_val, mode_freq = Counter(counts).most_common(1)[0]
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
    for lim in LIMITS:
        s[f"over_{lim}"] = int((arr > lim).sum())
    return s


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
    for lim in LIMITS:
        k = f"over_{lim}"
        print(f"  >{lim:<5}: {stats[k]:,}  ({100*stats[k]/n:.1f}%)")


def print_histogram(counts: list[int], bins: list[int]) -> None:
    arr = np.array(counts)
    edges = bins + [max(int(arr.max()) + 1, bins[-1] + 1)]
    print("\n  Histogram (tokens):")
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
    ]
    for lim in LIMITS:
        k = f"over_{lim}"
        lines.append(f"| > {lim} tokens | {stats[k]:,} ({100*stats[k]/n:.1f}%) |")
    return "\n".join(lines)


def md_histogram(counts: list[int], bins: list[int]) -> str:
    arr = np.array(counts)
    edges = bins + [max(int(arr.max()) + 1, bins[-1] + 1)]
    lines = [
        "| Token Range | Count | % | Bar |",
        "|-------------|-------|---|-----|",
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
    tokenizer_name: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = [
        "# CPG Function Token Count Analysis",
        "",
        f"**Generated:** {ts}  ",
        f"**Raw dir:** `{raw_dir}`  ",
        f"**Tokenizer:** `{tokenizer_name}`  ",
        f"**Datasets:** {', '.join(datasets)}",
        "",
        "> Tokens extracted from `raw_func` in `.meta.json` (actual source text) when available,",
        "> falling back to CPG `CODE` node reconstruction. Encoded with the specified tokenizer (no special tokens).",
        "",
    ]

    # Summary table
    md += [
        "## Summary",
        "",
        "| Dataset | Split | Count | Mean | Median | P90 | P95 | P99 | >512 | >1024 |",
        "|---------|-------|-------|------|--------|-----|-----|-----|------|-------|",
    ]
    for ds in datasets:
        for sp in splits:
            c = counts.get((ds, sp), [])
            if not c:
                continue
            s = compute_stats(c)
            n = s["count"]
            md.append(
                f"| {ds} | {sp} | {n:,} | {s['mean']:.0f} | {s['median']} "
                f"| {s['p90']} | {s['p95']} | {s['p99']} "
                f"| {100*s['over_512']/n:.1f}% | {100*s['over_1024']/n:.1f}% |"
            )
        combined = []
        for sp in splits:
            combined.extend(counts.get((ds, sp), []))
        if combined:
            s = compute_stats(combined)
            n = s["count"]
            md.append(
                f"| **{ds}** | **all** | **{n:,}** | **{s['mean']:.0f}** "
                f"| **{s['median']}** | **{s['p90']}** | **{s['p95']}** "
                f"| **{s['p99']}** | **{100*s['over_512']/n:.1f}%** "
                f"| **{100*s['over_1024']/n:.1f}%** |"
            )
    all_counts = [c for lst in counts.values() for c in lst]
    if all_counts and len(datasets) > 1:
        s = compute_stats(all_counts)
        n = s["count"]
        md.append(
            f"| **ALL** | **all** | **{n:,}** | **{s['mean']:.0f}** "
            f"| **{s['median']}** | **{s['p90']}** | **{s['p95']}** "
            f"| **{s['p99']}** | **{100*s['over_512']/n:.1f}%** "
            f"| **{100*s['over_1024']/n:.1f}%** |"
        )

    # Token limit recommendation per dataset
    md += ["", "## Token Limit Recommendation (func_lm)", ""]
    PCTS = [("P75", 75), ("P90", 90), ("P95 (recommended)", 95), ("P99 (max coverage)", 99)]
    for ds in datasets:
        vuln_counts = counts.get((ds, "vulnerable"), [])
        if not vuln_counts:
            continue
        arr = np.array(vuln_counts)
        md += [
            f"### {ds}",
            "",
            "| Coverage Target | Suggested max_seq_len | Truncates |",
            "|----------------|-----------------------|-----------|",
        ]
        for pct_label, pct_val in PCTS:
            val = int(np.percentile(arr, pct_val))
            md.append(f"| {pct_label} | {val} | {100-pct_val}% of vuln funcs |")
        for lim in LIMITS:
            k = f"over_{lim}"
            s = compute_stats(list(arr.astype(int)))
            pct = 100 * s[k] / s["count"]
            md.append(f"| Truncated at {lim} | {lim} | {pct:.1f}% of vuln funcs |")
        md.append("")

    if all_counts and len(datasets) > 1:
        arr = np.array([c for ds in datasets for c in counts.get((ds, "vulnerable"), [])])
        if len(arr):
            md += [
                "### All datasets combined",
                "",
                "| Coverage Target | Suggested max_seq_len | Truncates |",
                "|----------------|-----------------------|-----------|",
            ]
            for pct_label, pct_val in PCTS:
                val = int(np.percentile(arr, pct_val))
                md.append(f"| {pct_label} | {val} | {100-pct_val}% of vuln funcs |")
            md.append("")

    # Detailed per-dataset sections
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

    if all_counts and len(datasets) > 1:
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
        "--tokenizer", default="microsoft/codebert-base",
        help="HuggingFace tokenizer name (default: microsoft/codebert-base)"
    )
    parser.add_argument(
        "--char-estimate", action="store_true",
        help="Skip tokenizer — estimate tokens as char_count // 4 (fast)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to analyze (default: auto-discover all in --raw-dir)"
    )
    parser.add_argument(
        "--out", default=None, type=Path,
        help="Output markdown path (default: FUNC_TOKEN_ANALYSIS.md)"
    )
    args = parser.parse_args()

    tokenizer_name = None if args.char_estimate else args.tokenizer
    mode_label = "char//4 estimate" if args.char_estimate else f"tokenizer: {tokenizer_name}"

    raw_dir: Path = args.raw_dir
    splits = ["benign", "vulnerable"]
    HIST_BINS = [0, 64, 128, 256, 512, 1024, 2048, 4096]

    datasets = args.datasets or discover_datasets(raw_dir)
    if not datasets:
        print(f"No datasets found in {raw_dir}")
        return
    print(f"Datasets: {datasets}")
    print(f"Mode: {mode_label}")

    all_files: dict[tuple[str, str], list[str]] = {}
    for ds in datasets:
        for sp in splits:
            files = collect_files(raw_dir, ds, sp)
            all_files[(ds, sp)] = files
            print(f"  Found {len(files):>6} XML files — {ds}/{sp}")

    flat_files = [f for files in all_files.values() for f in files]
    print(f"\nTotal XML files: {len(flat_files):,} | Workers: {args.workers}")
    print("Extracting CODE and counting tokens…")

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(tokenizer_name,),
    ) as pool:
        flat_counts = pool.map(_count_tokens, flat_files)

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

    out_path = args.out or Path("FUNC_TOKEN_ANALYSIS.md")
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    md = build_markdown(raw_dir, datasets, splits, counts, HIST_BINS, mode_label)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown saved -> {out_path}")


if __name__ == "__main__":
    main()
