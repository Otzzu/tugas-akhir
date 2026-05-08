"""
download_datasets.py
~~~~~~~~~~~~~~~~~~~~
Download BigVul, Devign, DiverseVul, MegaVul, TitanVul, and BenchVul
from HuggingFace Hub and save them as parquet files under data/datasets/.

Usage:
    uv run python scripts/download_datasets.py
    uv run python scripts/download_datasets.py --only devign
    uv run python scripts/download_datasets.py --only megavul
    uv run python scripts/download_datasets.py --only titanvul
    uv run python scripts/download_datasets.py --only benchvul
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "datasets"

DATASETS = {
    "bigvul": {
        "repo": "bstee615/bigvul",
        "splits": ["train"],
    },
    "devign": {
        "repo": "DetectVul/devign",
        "splits": ["train"],
    },
    "diversevul": {
        "repo": "bstee615/diversevul",
        "splits": ["train"],
    },
}


def download_one(name: str, cfg: dict) -> None:
    from datasets import load_dataset

    dest = OUT_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Downloading: {name}  ({cfg['repo']})")
    print(f"{'='*60}")

    ds = load_dataset(cfg["repo"])

    saved = []
    for split in ds.keys():
        out_file = dest / f"{split}.parquet"
        ds[split].to_parquet(str(out_file))
        n = len(ds[split])
        cols = ds[split].column_names
        print(f"  [{split}]  {n:,} rows  |  columns: {cols}")
        print(f"  -> saved to {out_file.relative_to(PROJECT_ROOT)}")
        saved.append(out_file)

    print(f"  Done — {len(saved)} file(s) written.")


def _extract_flaw_lines(func_before: str, diff_func: str | None, diff_line_info: dict | None) -> list[int]:
    """
    Extract 1-indexed flaw line numbers from MegaVul diff info.
    Tries unified diff first (more accurate), falls back to content matching.
    """
    if diff_func:
        # Parse unified diff: lines starting with '-' (not '---') are removed (vulnerable)
        import re as _re
        flaw: list[int] = []
        cur_line = 0
        for line in diff_func.splitlines():
            m = _re.match(r'^@@ -(\d+)', line)
            if m:
                cur_line = int(m.group(1))
                continue
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("-"):
                flaw.append(cur_line)
                cur_line += 1
            elif not line.startswith("+"):
                cur_line += 1
        return sorted(set(flaw))

    if diff_line_info and func_before:
        deleted = diff_line_info.get("deleted_lines", [])
        if not deleted:
            return []
        src_lines = func_before.splitlines()
        deleted_stripped = {d.strip() for d in deleted if d.strip()}
        return sorted({i + 1 for i, l in enumerate(src_lines) if l.strip() in deleted_stripped})

    return []


def download_megavul() -> None:
    """
    Convert local MegaVul JSON to pipeline-compatible parquet.

    Reads from:
        data/datasets/megavul/c_cpp/megavul.json   (2.5 GB, download from OneDrive)
        data/datasets/megavul/java/megavul.json     (optional)

    Creates train.parquet with columns:
        func_before, func_after, vul, CWE ID, CVE ID,
        language, flaw_lines, func_graph_path_before, func_graph_path

    func_graph_path_before / func_graph_path: absolute path to pre-built Joern graph.
    Used by prepare_dataset.py --format megavul to copy graphs without re-running Joern.
    """
    import pandas as pd
    import ijson
    from decimal import Decimal

    dest = OUT_DIR / "megavul"
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / "train.parquet"

    # Locate source JSON files (c_cpp required, java optional)
    json_files: list[Path] = []
    for sub in ["c_cpp", "java"]:
        jf = dest / sub / "megavul.json"
        if jf.exists():
            json_files.append(jf)
    if not json_files:
        print("  ERROR: No megavul.json found.")
        print("  Download from: https://1drv.ms/f/s!AtzrzuojQf5sgeISZ9zN_4owVnUn9g")
        print("  Place at: data/datasets/megavul/c_cpp/megavul.json")
        return

    _EXT_TO_LANG = {
        "c": "C", "h": "C", "cpp": "C++", "cc": "C++", "cxx": "C++", "hpp": "C++",
        "java": "Java", "kt": "Kotlin", "js": "JavaScript", "ts": "TypeScript",
        "py": "Python", "rb": "Ruby", "php": "PHP", "go": "Go",
        "cs": "C#", "swift": "Swift", "rs": "Rust",
    }

    print(f"\n{'='*60}")
    print("  Processing: MegaVul (local megavul.json)")
    print(f"{'='*60}")

    rows: list[dict] = []

    for jf in json_files:
        graph_dir = jf.parent / "megavul_graph"
        print(f"  Reading {jf} ...")
        with open(jf, "rb") as f:
            for rec in ijson.items(f, "item"):
                # Normalise Decimal (ijson returns Decimal for floats)
                is_vul = bool(rec.get("is_vul", False))
                cwe_ids: list = rec.get("cwe_ids") or []
                cwe = cwe_ids[0] if cwe_ids else ""
                cve_id = str(rec.get("cve_id") or "")
                file_path = str(rec.get("file_path") or "")
                ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                lang = _EXT_TO_LANG.get(ext, ext.capitalize() or "C")

                func_before = rec.get("func_before") or ""
                func_patched = rec.get("func") or ""
                diff_func = rec.get("diff_func") or None
                diff_line_info = rec.get("diff_line_info") or None
                graph_before = str(rec.get("func_graph_path_before") or "")
                graph_after  = str(rec.get("func_graph_path") or "")

                # Absolute graph paths (empty string if graph missing)
                abs_before = str(graph_dir / graph_before) if graph_before and (graph_dir / graph_before).exists() else ""
                abs_after  = str(graph_dir / graph_after)  if graph_after  and (graph_dir / graph_after).exists()  else ""

                if is_vul:
                    if not func_before:
                        continue
                    flaw_lines = _extract_flaw_lines(func_before, diff_func, diff_line_info)
                    rows.append({
                        "func_before": func_before,
                        "func_after": func_patched or None,
                        "vul": 1,
                        "CWE ID": cwe,
                        "CVE ID": cve_id,
                        "language": lang,
                        "flaw_lines": flaw_lines,
                        "func_graph_path": abs_before,  # graph for vulnerable version
                    })
                else:
                    # Non-vulnerable: func (benign code), no flaw lines
                    if not func_patched:
                        continue
                    rows.append({
                        "func_before": func_patched,
                        "func_after": None,
                        "vul": 0,
                        "CWE ID": "",
                        "CVE ID": cve_id,
                        "language": lang,
                        "flaw_lines": [],
                        "func_graph_path": abs_after,
                    })

    df = pd.DataFrame(rows)
    df.to_parquet(str(out_file), index=False)

    vuln_n   = (df["vul"] == 1).sum()
    benign_n = (df["vul"] == 0).sum()
    has_graph = (df["func_graph_path"] != "").sum()
    has_flaw  = df[df["vul"] == 1]["flaw_lines"].apply(len).gt(0).sum()
    print(f"  train.parquet: {len(df):,} rows (vuln={vuln_n:,} benign={benign_n:,})")
    print(f"  Pre-built graphs available: {has_graph:,}/{len(df):,}")
    print(f"  Vuln rows with flaw lines:  {has_flaw:,}/{vuln_n:,}")
    print(f"  -> {out_file.relative_to(PROJECT_ROOT)}")
    print("\n  Language distribution (vuln):")
    for lang, cnt in df[df["vul"] == 1]["language"].value_counts().head(10).items():
        print(f"    {lang}: {cnt:,}")
    print("\n  CWE distribution (top 15):")
    for cwe, cnt in df[df["vul"] == 1]["CWE ID"].value_counts().head(15).items():
        print(f"    {cwe}: {cnt}")
    print(f"\n  Use: prepare_dataset.py --format megavul --input {out_file.relative_to(PROJECT_ROOT)}")


def download_titanvul() -> None:
    """
    Download TitanVul (38,548 vulnerability-fix pairs, multilingual).

    Column schema: func_before, func_after, cve_id, cwe_id, extension, ...
    Every row is a vulnerable function paired with its fix — there is no
    explicit 'vul' label column.  We expand each row into:
      - one vulnerable sample  (func_before, vul=1, CWE from cwe_id)
      - one benign sample      (func_after,  vul=0)  when func_after differs

    Output columns (BigVul-compatible so prepare_dataset.py --format bigvul works):
        func_before, func_after, vul, CWE ID, CVE ID

    Note: filter to C/C++ only via the 'extension' field (.c / .cpp / .h).
    """
    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    dest = OUT_DIR / "titanvul"
    dest.mkdir(parents=True, exist_ok=True)
    out_file  = dest / "train.parquet"
    raw_file  = dest / "raw.parquet"

    print(f"\n{'='*60}")
    print("  Downloading: titanvul  (yikun-li/TitanVul)")
    print("  38,548 vuln-fix pairs — saving raw + expanded train parquets…")
    print(f"{'='*60}")

    ds = load_dataset("yikun-li/TitanVul", split="train")

    # raw.parquet — all original columns, no transformation
    pd.DataFrame(list(ds)).to_parquet(str(raw_file), index=False)
    print(f"  raw.parquet: {len(ds):,} rows (all columns) -> {raw_file.relative_to(PROJECT_ROOT)}")

    # train.parquet — expanded vuln+benign pairs, pipeline-compatible
    _EXT_TO_LANG = {
        # C family
        "c": "C", "h": "C",
        "cpp": "C++", "cc": "C++", "cxx": "C++", "hpp": "C++",
        # JVM
        "java": "Java", "kt": "Kotlin", "scala": "Scala",
        # Scripting
        "js": "JavaScript", "jsx": "JavaScript", "ts": "TypeScript",
        "py": "Python", "rb": "Ruby", "php": "PHP",
        "lua": "Lua", "m": "Objective-C",
        # Systems
        "go": "Go", "rs": "Rust", "swift": "Swift",
        # .NET
        "cs": "C#",
    }
    rows: list[dict] = []
    skipped_no_code = 0
    for row in tqdm(ds, desc="  processing", unit="row"):
        vuln_code  = row.get("func_before")
        fixed_code = row.get("func_after")
        if not vuln_code:
            skipped_no_code += 1
            continue

        cwe    = (row.get("cwe_id")    or "").strip()
        cve_id = (row.get("cve_id")    or "").strip()
        ext    = (row.get("extension") or "").strip().lstrip(".")
        lang   = _EXT_TO_LANG.get(ext.lower(), ext or "unknown")

        rows.append({"func_before": vuln_code, "func_after": fixed_code or None,
                     "vul": 1, "CWE ID": cwe, "CVE ID": cve_id,
                     "extension": ext, "language": lang})
        if fixed_code and fixed_code.strip() != vuln_code.strip():
            rows.append({"func_before": fixed_code, "func_after": None,
                         "vul": 0, "CWE ID": "", "CVE ID": cve_id,
                         "extension": ext, "language": lang})

    df = pd.DataFrame(rows)
    df.to_parquet(str(out_file), index=False)

    vuln_n   = (df["vul"] == 1).sum()
    benign_n = (df["vul"] == 0).sum()
    print(f"  Skipped {skipped_no_code:,} rows with missing code")
    print(f"  train.parquet: {len(df):,} rows (vuln={vuln_n:,} benign={benign_n:,})")
    print(f"  -> {out_file.relative_to(PROJECT_ROOT)}")
    print("\n  Language distribution:")
    for lang, cnt in df[df["vul"]==1]["language"].value_counts().head(10).items():
        print(f"    {lang}: {cnt:,}")
    print("\n  CWE distribution (top 15):")
    for cwe, cnt in df[df["vul"] == 1]["CWE ID"].value_counts().head(15).items():
        print(f"    {cwe}: {cnt}")
    print(f"\n  Use with prepare_dataset.py --format bigvul --input {out_file.relative_to(PROJECT_ROOT)}")


def download_benchvul() -> None:
    """
    Download BenchVul (1,050 rows) — a manually-verified benchmark for the
    Top 25 Most Dangerous CWEs.  50 vulnerable + 50 fixed samples per CWE.

    Column schema: cwe_id, cve_id, func_before, func_after, programming_language, ...
    Same expansion logic as TitanVul: each row -> vuln sample + benign sample.
    Output is BigVul-compatible (func_before / func_after / vul / CWE ID).

    Note: filter to C/C++ via the 'programming_language' field.
    """
    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    dest = OUT_DIR / "benchvul"
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / "train.parquet"

    print(f"\n{'='*60}")
    print("  Downloading: benchvul  (yikun-li/BenchVul)")
    print("  1,050 manually-verified vuln-fix pairs — expanding to vuln + benign rows…")
    print(f"{'='*60}")

    ds = load_dataset("yikun-li/BenchVul", split="train")

    rows: list[dict] = []
    skipped_lang = 0
    skipped_no_code = 0
    for raw in tqdm(ds, desc="  processing", unit="row"):

        vuln_code = raw.get("func_before")
        fixed_code = raw.get("func_after")
        if not vuln_code:
            skipped_no_code += 1
            continue

        cwe = (raw.get("cwe_id") or "").strip()
        cve_id = (raw.get("cve_id") or "").strip()

        # Vulnerable sample
        rows.append({
            "func_before": vuln_code,
            "func_after": fixed_code or None,
            "vul": 1,
            "CWE ID": cwe,
            "CVE ID": cve_id,
        })

        # Benign sample from the fixed version
        if fixed_code and fixed_code.strip() != vuln_code.strip():
            rows.append({
                "func_before": fixed_code,
                "func_after": None,
                "vul": 0,
                "CWE ID": "",
                "CVE ID": cve_id,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(str(out_file), index=False)

    vuln_n = (df["vul"] == 1).sum()
    benign_n = (df["vul"] == 0).sum()
    print(f"\n  Skipped {skipped_no_code:,} rows with missing code")
    print(f"  Saved {len(df):,} rows  (vulnerable={vuln_n:,}  benign={benign_n:,})")
    print(f"  -> {out_file.relative_to(PROJECT_ROOT)}")
    print("\n  CWE distribution (top 15):")
    cwe_counts = df[df["vul"] == 1]["CWE ID"].value_counts().head(15)
    for cwe, cnt in cwe_counts.items():
        print(f"    {cwe}: {cnt}")
    print(f"\n  Use with prepare_dataset.py --format bigvul --input {out_file.relative_to(PROJECT_ROOT)}")
    print("  Tip: BenchVul is small (1,050 rows) — best used as a test/eval set, not for training.")


def download_bigvul() -> None:
    """
    Download BigVul from bstee615/bigvul (train / val / test splits).

    Saves 8 files per dataset:
      raw_train.parquet  — all original HuggingFace columns, train split
      raw_val.parquet    — all original HuggingFace columns, val split
      raw_test.parquet   — all original HuggingFace columns, test split
      raw_all.parquet    — concat of all 3 raw splits
      train.parquet      — pipeline-compatible (+ normalized language column)
      val.parquet        — pipeline-compatible
      test.parquet       — pipeline-compatible
      all.parquet        — concat of all 3 pipeline splits
    """
    import pandas as pd
    from datasets import load_dataset

    _LANG_NORM = {"C": "C", "CPP": "C++", "C++": "C++"}

    dest = OUT_DIR / "bigvul"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Downloading: bigvul  (bstee615/bigvul)")
    print("  3 splits: train / val / test")
    print(f"{'='*60}")

    ds = load_dataset("bstee615/bigvul")

    raw_dfs: list[pd.DataFrame] = []
    pipe_dfs: list[pd.DataFrame] = []

    for split in ["train", "validation", "test"]:
        if split not in ds:
            print(f"  [{split}] not found — skipping")
            continue

        split_ds = ds[split]
        raw_df = pd.DataFrame(split_ds)

        # raw_{split}.parquet — exact HuggingFace download
        raw_path = dest / f"raw_{split}.parquet"
        raw_df.to_parquet(str(raw_path), index=False)
        print(f"  [{split}] raw: {len(raw_df):,} rows | cols: {list(raw_df.columns)}")
        print(f"    -> {raw_path.relative_to(PROJECT_ROOT)}")
        raw_dfs.append(raw_df)

        # {split}.parquet — add normalized language column, keep all columns
        pipe_df = raw_df.copy()
        if "lang" in pipe_df.columns:
            pipe_df["language"] = pipe_df["lang"].apply(
                lambda v: _LANG_NORM.get(str(v).strip().upper(), str(v)) if pd.notna(v) else ""
            )
        pipe_path = dest / f"{split}.parquet"
        pipe_df.to_parquet(str(pipe_path), index=False)
        n_vuln = int((pipe_df["vul"] == 1).sum()) if "vul" in pipe_df.columns else "?"
        print(f"  [{split}] pipeline: {len(pipe_df):,} rows (vuln={n_vuln})")
        print(f"    -> {pipe_path.relative_to(PROJECT_ROOT)}")
        pipe_dfs.append(pipe_df)

    # raw_all.parquet + all.parquet
    if raw_dfs:
        raw_all = pd.concat(raw_dfs, ignore_index=True)
        raw_all.to_parquet(str(dest / "raw_all.parquet"), index=False)
        print(f"\n  raw_all.parquet: {len(raw_all):,} rows -> {(dest/'raw_all.parquet').relative_to(PROJECT_ROOT)}")

    if pipe_dfs:
        all_df = pd.concat(pipe_dfs, ignore_index=True)
        all_df.to_parquet(str(dest / "all.parquet"), index=False)
        print(f"  all.parquet:     {len(all_df):,} rows -> {(dest/'all.parquet').relative_to(PROJECT_ROOT)}")
        if "lang" in all_df.columns:
            print("\n  Language distribution:")
            for lang, cnt in all_df["lang"].value_counts().items():
                print(f"    {lang}: {cnt:,}")
        print(f"\n  Use: prepare_dataset.py --format bigvul --input {(dest/'train.parquet').relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download vulnerability datasets from HuggingFace.")
    parser.add_argument(
        "--only", choices=[*DATASETS.keys(), "megavul", "titanvul", "benchvul", "bigvul"],
        help="Download only this dataset (default: all standard datasets)",
    )
    args = parser.parse_args()

    if args.only == "bigvul":
        download_bigvul()
        return

    if args.only == "megavul":
        download_megavul()
        return

    if args.only == "titanvul":
        download_titanvul()
        return

    if args.only == "benchvul":
        download_benchvul()
        return

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS

    print(f"Saving datasets to: {OUT_DIR}")
    for name, cfg in targets.items():
        try:
            download_one(name, cfg)
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}", file=sys.stderr)

    if not args.only:
        print("\nSkipping megavul  (run with --only megavul  to download separately — streams 672K rows).")
        print("Skipping titanvul (run with --only titanvul to download separately — 38K rows).")
        print("Skipping benchvul (run with --only benchvul to download separately — 1K rows, eval only).")

    print("\nAll done.")
    print(f"Files are in: {OUT_DIR}")
    print("\nNext step - generate CPG files with Joern:")
    print("  uv run python scripts/prepare_dataset.py \\")
    print("      --input data/datasets/devign/train.parquet \\")
    print("      --format devign --joern-cli /path/to/joern/joern-cli")


if __name__ == "__main__":
    main()
