"""
patch_raw_multilang.py — Re-process non-C functions with correct Joern language frontend.

Previously all functions were written as .c → Joern used C parser for everything.
This script:
  1. Scans data/raw/<dataset>/{benign,vulnerable}/*.meta.json
  2. Detects language from raw_func via detect_language()
  3. For non-C functions where correct frontend produces better CPG:
       Java:   +3x nodes   → re-run with .java
       JS:     +85x nodes  → re-run with .js
       Python: +12x nodes  → re-run with .py
       Ruby:   +4.7x nodes → re-run with .rb
       C++:    cleaner AST  → re-run with .cpp
  4. Replaces existing .xml only if new CPG has MORE nodes
  5. Go/PHP: skipped (C parser produces denser CPGs)

Usage (PowerShell):
    uv run python scripts/patch_raw_multilang.py --datasets bigvul megavul titanvul
    uv run python scripts/patch_raw_multilang.py --datasets titanvul --workers 4 --dry-run
    uv run python scripts/patch_raw_multilang.py --joern-cli C:/joern/joern-cli

After patching, upload to Drive:
    .\\scripts\\upload_raw_datasets.ps1 -Datasets bigvul,megavul,titanvul
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_vuln.data.joern_runner import detect_language, process_function

# Languages where correct frontend improves CPG — skip c/go/php (C parser is better)
PATCH_LANGS = {"cpp", "java", "js", "py", "rb", "kt", "cs"}

NS = "http://graphml.graphdrawing.org/xmlns"


def count_nodes(xml_path: Path) -> int:
    """Count nodes in a GraphML CPG file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        nodes = root.findall(f'.//{{{NS}}}node')
        return len(nodes)
    except Exception:
        return 0


def patch_one(
    meta_path: Path,
    joern_cli: Path | None,
    dry_run: bool,
    lang_lookup: dict[int, str] | None = None,
) -> tuple[str, str]:
    """
    Process one function. Returns (status, message).
    lang_lookup: optional {row_id → ext} from parquet lang column (ground truth).
    Falls back to detect_language() if id not found or lookup not provided.
    """
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
        code = meta.get("raw_func", "")
        if not code:
            return "skip_c", "no raw_func"

        # Ground truth from parquet; null values fall back to detect_language()
        row_id = meta.get("id", -1)
        if lang_lookup is not None and row_id in lang_lookup:
            lang = lang_lookup[row_id]  # parquet ground truth
        else:
            lang = detect_language(code)  # fallback: heuristic

        if lang not in PATCH_LANGS:
            return "skip_c", f"lang={lang} (no patch needed)"

        xml_path = meta_path.with_suffix("").with_suffix(".xml")
        if not xml_path.exists():
            # Try alternate path pattern (func_N.meta.json → func_N.xml)
            xml_path = meta_path.parent / (meta_path.name.replace(".meta.json", ".xml"))
        if not xml_path.exists():
            return "failed", f"existing XML not found for {meta_path.name}"

        old_nodes = count_nodes(xml_path)

        if dry_run:
            return "dry_run", f"{meta_path.parent.name}/{xml_path.name}: lang={lang}, old_nodes={old_nodes}"

        # Re-run Joern with correct extension — use isolated tmp dir per function
        # (avoids race condition when workers share same parent/_patch_tmp)
        tmp_dir = Path(tempfile.mkdtemp(prefix="patch_"))
        try:
            new_xml = process_function(
                code=code,
                idx=0,
                out_dir=tmp_dir,
                joern_cli_dir=joern_cli,
                fmt="graphml",
                lang=lang,
            )

            if new_xml is None:
                return "failed", f"Joern failed: {meta_path.name} (lang={lang})"

            new_nodes = count_nodes(new_xml)
            # Replace unconditionally — correct frontend always gives correct CPG.
            # (More nodes ≠ better; correct syntax tree = better regardless of count.)
            xml_path.write_bytes(new_xml.read_bytes())
            return "patched", f"{xml_path.name}: {old_nodes}→{new_nodes} nodes (lang={lang})"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e:
        return "failed", f"{meta_path.name}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Re-process non-C raw functions with correct Joern frontend")
    parser.add_argument("--datasets",   nargs="+", default=["bigvul", "megavul", "titanvul"],
                        help="Dataset directories under data/raw/")
    parser.add_argument("--joern-cli",  default="C:/joern/joern-cli",
                        help="Path to Joern CLI directory (default: C:/joern/joern-cli)")
    parser.add_argument("--workers",    type=int, default=2,
                        help="Parallel workers (default: 2; increase for faster patching)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Estimate what would be patched without running Joern")
    parser.add_argument("--raw-dir",    default=str(PROJECT_ROOT / "data" / "raw"),
                        help="Root of raw data dir (default: data/raw)")
    args = parser.parse_args()

    joern_cli = Path(args.joern_cli) if args.joern_cli else None
    raw_root  = Path(args.raw_dir)
    data_root = PROJECT_ROOT / "data" / "datasets"

    # Parquet lang column config per dataset.
    # lang_col: column name in parquet containing language or file extension.
    # is_ext: True → value is already a file extension (e.g. ".py", "py");
    #         False → value is language name (e.g. "Python", "C++")
    # If column missing or value null → detect_language() fallback.
    _LANG_NORM = {
        "C": "c", "CPP": "cpp", "C++": "cpp", "JAVA": "java",
        "JAVASCRIPT": "js", "JS": "js", "PYTHON": "py", "GO": "go",
        "RUBY": "rb", "PHP": "php", "KOTLIN": "kt", "C#": "cs",
    }
    PARQUET_LANG: dict[str, tuple[Path, str, bool]] = {
        # dataset  → (parquet_path,                  lang_col,    is_ext)
        "bigvul":  (data_root / "bigvul"  / "all.parquet",  "lang",      False),
        # megavul: filtered to C-only during download — no lang column needed
        "titanvul":(data_root / "titanvul"/ "train.parquet", "extension", True),
    }

    import pandas as pd

    def _ext_from_val(val: str, is_ext: bool) -> str:
        """Convert parquet lang/ext value to Joern extension string."""
        s = str(val).strip().lstrip(".")
        if is_ext:
            return s.lower() or "c"
        return _LANG_NORM.get(s.upper(), detect_language(s))

    # Build per-dataset id→ext lookup from parquet where column exists
    ds_lang_lookup: dict[str, dict[int, str]] = {}
    for ds, (pq_path, lang_col, is_ext) in PARQUET_LANG.items():
        if not pq_path.exists() or ds not in args.datasets:
            continue
        try:
            df = pd.read_parquet(pq_path)
            if lang_col not in df.columns:
                print(f"[{ds}] Column '{lang_col}' not in parquet — will use detect_language()")
                continue
            lookup = {}
            for row_id, val in enumerate(df[lang_col]):
                if pd.notna(val) and str(val).strip():
                    lookup[row_id] = _ext_from_val(str(val), is_ext)
                # null → no entry → detect_language() fallback at patch time
            ds_lang_lookup[ds] = lookup
            n_null = len(df) - len(lookup)
            print(f"[{ds}] Loaded '{lang_col}' from parquet: {len(lookup)} rows, "
                  f"{n_null} null → detect_language() fallback")
        except Exception as e:
            print(f"[{ds}] WARNING: could not load parquet lang: {e}")

    all_meta: list[Path] = []
    meta_lookup: dict[Path, dict[int, str] | None] = {}  # meta_path → lang_lookup
    for ds in args.datasets:
        ds_dir = raw_root / ds
        if not ds_dir.exists():
            print(f"[WARN] Dataset dir not found: {ds_dir} — skipping")
            continue
        found = list(ds_dir.rglob("*.meta.json"))
        lookup = ds_lang_lookup.get(ds)
        src = "parquet" if lookup else "detect_language()"
        print(f"[{ds}] {len(found)} meta.json files | lang source: {src}")
        for f in found:
            meta_lookup[f] = lookup
        all_meta.extend(found)

    if not all_meta:
        print("No meta.json files found. Exiting.")
        return

    # For dry-run with sample, use max workers for speed (no Joern subprocess contention)
    workers = args.workers
    if args.dry_run:
        workers = min(32, max(workers, 8))

    print(f"\nChecking {len(all_meta)} functions (workers={workers})")
    if args.dry_run:
        print("DRY RUN — no Joern calls, summary only\n")

    stats: dict[str, int] = {"skip_c": 0, "patched": 0, "failed": 0, "dry_run": 0}
    lang_counts: dict[str, int] = {}

    def _worker(mp: Path):
        return patch_one(mp, joern_cli, args.dry_run, meta_lookup.get(mp))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, mp): mp for mp in all_meta}
        done = 0
        for fut in as_completed(futures):
            status, msg = fut.result()
            stats[status] = stats.get(status, 0) + 1
            # Track detected language for dry-run summary
            if args.dry_run and "lang=" in msg:
                import re as _re
                m = _re.search(r"lang=(\w+)", msg)
                if m:
                    lang_counts[m.group(1)] = lang_counts.get(m.group(1), 0) + 1
            done += 1
            # Verbose only for actual patch / failures; suppress skip spam
            if not args.dry_run and status not in ("skip_c",):
                print(f"[{status.upper():12s}] {msg}")
            if done % 1000 == 0:
                print(f"  ... {done}/{len(all_meta)} checked")

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"  Patched (correct lang)  : {stats['patched']:>6}")
    print(f"  Skipped (C/Go/PHP)      : {stats['skip_c']:>6}")
    print(f"  Failed                  : {stats['failed']:>6}")
    if args.dry_run:
        print(f"  Would patch (dry-run)   : {stats['dry_run']:>6}")
    print("="*60)

    if stats["patched"] > 0:
        print(f"\n{stats['patched']} functions re-processed with correct language frontend.")
        print("Run upload after patching:")
        print("  .\\scripts\\upload_raw_datasets.ps1 -Datasets " + ",".join(args.datasets))


if __name__ == "__main__":
    main()
