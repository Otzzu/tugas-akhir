"""
merge_datasets.py
~~~~~~~~~~~~~~~~~
Merge BigVul + MegaVul + scraped dataset into one unified parquet
compatible with prepare_dataset.py --format bigvul.

Output schema (BigVul-compatible):
    func_before  str   function body (vulnerable or benign)
    func_after   str   patched version (used for flaw-line diff), or None
    vul          int   1 = vulnerable, 0 = benign
    CWE ID       str   e.g. "CWE-119", empty for benign rows

Deduplication: exact match on func_before text (SHA-256 hash).
Sources sharing the same CVE/commit may have identical functions.

Usage:
    uv run python scripts/merge_datasets.py
    uv run python scripts/merge_datasets.py --no-scrape   # skip scraped data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
OUT_DIR = DATA_DIR / "merged"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_bigvul() -> pd.DataFrame:
    path = DATA_DIR / "bigvul" / "train.parquet"
    if not path.exists():
        print("  [SKIP] BigVul not found — run download_datasets.py first")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "lang" in df.columns:
        before = len(df)
        df = df[df["lang"].isin(["C", "C++"])].copy()
        print(f"  BigVul: {before:,} total → {len(df):,} C/C++ rows")
    df["CWE ID"] = df["CWE ID"].fillna("").astype(str)
    df["vul"] = df["vul"].astype(int)
    df["cve_id"] = df["CVE ID"].fillna("").astype(str) if "CVE ID" in df.columns else ""
    df = df[["func_before", "func_after", "vul", "CWE ID", "cve_id"]].copy()
    df = df.dropna(subset=["func_before"])
    df["func_before"] = df["func_before"].astype(str)
    df["source"] = "bigvul"
    return df


def load_megavul() -> pd.DataFrame:
    path = DATA_DIR / "megavul" / "train.parquet"
    if not path.exists():
        print("  [SKIP] MegaVul not found — run download_datasets.py --only megavul first")
        return pd.DataFrame()
    # Need cve_id — re-read from original streaming is too slow, so check if saved
    # MegaVul parquet was saved without cve_id; add empty column
    df = pd.read_parquet(path)
    df["CWE ID"] = df["CWE ID"].fillna("").astype(str)
    df["vul"] = df["vul"].astype(int)
    df["cve_id"] = df["CVE ID"].fillna("").astype(str) if "CVE ID" in df.columns else ""
    df = df[["func_before", "func_after", "vul", "CWE ID", "cve_id"]].copy()
    df = df.dropna(subset=["func_before"])
    df["func_before"] = df["func_before"].astype(str)
    df["source"] = "megavul"
    print(f"  MegaVul: {len(df):,} rows")
    return df


def load_scraped() -> pd.DataFrame:
    path = DATA_DIR / "scrap" / "vulnerability_data_c_v1.json"
    if not path.exists():
        print("  [SKIP] Scraped data not found at data/datasets/scrap/vulnerability_data_c_v1.json")
        return pd.DataFrame()

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    entries = raw.get("data", [])

    rows = []
    for e in entries:
        func_before = e.get("func_before", "")
        func_after = e.get("func_after", None)
        if not func_before or len(func_before) < 20:
            continue
        cwe_ids = e.get("cwe_ids", [])
        cwe = cwe_ids[0] if cwe_ids else ""
        cve_id = e.get("cve_id", "")

        rows.append({
            "func_before": func_before,
            "func_after": func_after if func_after and func_after != func_before else None,
            "vul": 1,
            "CWE ID": cwe,
            "cve_id": cve_id,
            "source": "scraped",
        })

        if func_after and func_after != func_before and len(func_after) >= 20:
            rows.append({
                "func_before": func_after,
                "func_after": None,
                "vul": 0,
                "CWE ID": "",
                "cve_id": cve_id,
                "source": "scraped",
            })

    df = pd.DataFrame(rows)
    print(f"  Scraped: {len(df):,} rows ({(df['vul']==1).sum()} vuln + {(df['vul']==0).sum()} benign)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge vulnerability datasets into one parquet.")
    parser.add_argument("--no-scrape", action="store_true", help="Exclude scraped dataset")
    parser.add_argument("--no-bigvul", action="store_true", help="Exclude BigVul")
    parser.add_argument("--no-megavul", action="store_true", help="Exclude MegaVul")
    args = parser.parse_args()

    print("Loading datasets…")
    parts: list[pd.DataFrame] = []

    if not args.no_bigvul:
        df = load_bigvul()
        if not df.empty:
            parts.append(df)

    if not args.no_megavul:
        df = load_megavul()
        if not df.empty:
            parts.append(df)

    if not args.no_scrape:
        df = load_scraped()
        if not df.empty:
            parts.append(df)

    if not parts:
        print("No datasets loaded. Exiting.")
        return

    merged = pd.concat(parts, ignore_index=True)
    print(f"\nBefore dedup: {len(merged):,} rows")

    # Deduplicate strategy:
    # 1. Same CVE ID + same func hash → definite duplicate across datasets
    # 2. Same func hash (no CVE ID) → exact code duplicate
    # BigVul loaded first → kept as preferred source
    merged["_func_hash"] = merged["func_before"].apply(_hash)
    merged["_cve_norm"] = merged["cve_id"].str.strip().str.upper()

    before = len(merged)
    # Primary: deduplicate on (CVE ID, func hash) for rows with known CVE
    has_cve = merged["_cve_norm"] != ""
    cve_dupes = merged[has_cve].duplicated(subset=["_cve_norm", "_func_hash"], keep="first")
    # Secondary: deduplicate remaining by func hash alone
    no_cve_dupes = merged[~has_cve].duplicated(subset=["_func_hash"], keep="first")
    # Also catch cross-group exact func duplicates
    global_dupes = merged.duplicated(subset=["_func_hash"], keep="first")

    drop_mask = global_dupes  # simplest: just dedup on func hash (covers both cases)
    merged = merged[~drop_mask].copy()
    merged = merged.drop(columns=["_func_hash", "_cve_norm"])

    dupes = before - len(merged)
    print(f"Removed {dupes:,} duplicate functions ({100*dupes/before:.1f}%)")

    # Normalize unknown CWE labels — empty string, null, "CWE-Other" → "CWE-unknown"
    # Kept in dataset; prepare_dataset.py will drop them during multiclass vocab building
    unknown_mask = (
        merged["CWE ID"].fillna("").str.strip().eq("") |
        merged["CWE ID"].str.upper().eq("CWE-OTHER") |
        merged["CWE ID"].str.upper().eq("UNKNOWN")
    ) & (merged["vul"] == 1)
    n_normalized = unknown_mask.sum()
    merged.loc[unknown_mask, "CWE ID"] = "CWE-unknown"
    print(f"Normalized {n_normalized:,} unknown/other CWE labels → 'CWE-unknown'")

    # Stats
    vuln_n = (merged["vul"] == 1).sum()
    benign_n = (merged["vul"] == 0).sum()
    print(f"\nFinal: {len(merged):,} rows  (vulnerable={vuln_n:,}  benign={benign_n:,})")

    print("\nSource breakdown:")
    for src, grp in merged.groupby("source"):
        v = (grp["vul"] == 1).sum()
        b = (grp["vul"] == 0).sum()
        print(f"  {src}: {len(grp):,}  (vuln={v:,}  benign={b:,})")

    print("\nTop 15 CWEs (vulnerable rows):")
    cwe_counts = merged[merged["vul"] == 1]["CWE ID"].value_counts().head(15)
    for cwe, cnt in cwe_counts.items():
        print(f"  {cwe}: {cnt:,}")

    merged = merged.drop(columns=["source", "cve_id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "train.parquet"
    merged.to_parquet(str(out_file), index=False)
    print(f"\nSaved → {out_file.relative_to(PROJECT_ROOT)}")
    print("\nUse with:")
    print("  uv run python scripts/prepare_dataset.py \\")
    print("      --input data/datasets/merged/train.parquet \\")
    print("      --format bigvul \\")
    print("      --joern-cli C:/joern/joern-cli \\")
    print("      --out-dir data/raw \\")
    print("      --sample-per-class 2000 \\")
    print("      --workers 4")


if __name__ == "__main__":
    main()
