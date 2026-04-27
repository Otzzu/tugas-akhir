"""
download_datasets.py
~~~~~~~~~~~~~~~~~~~~
Download BigVul, Devign, DiverseVul, and MegaVul from HuggingFace Hub
and save them as parquet files under data/datasets/.

Usage:
    uv run python scripts/download_datasets.py
    uv run python scripts/download_datasets.py --only devign
    uv run python scripts/download_datasets.py --only megavul
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


def download_megavul() -> None:
    """
    Stream MegaVul (672K rows), filter to C rows with extracted function code,
    and save in BigVul-compatible column layout so prepare_dataset.py can consume
    it with --format bigvul.

    Output columns (same as BigVul):
        func_before  — vulnerable function body       (vul=1 rows)
                       fixed function body             (vul=0 rows)
        func_after   — fixed function body             (vul=1 rows, used for flaw-line diff)
                       None                            (vul=0 rows)
        vul          — 1 = vulnerable, 0 = benign
        CWE ID       — CWE string (e.g. "CWE-416"), empty for benign rows
    """
    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    dest = OUT_DIR / "megavul"
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / "train.parquet"

    print(f"\n{'='*60}")
    print("  Downloading: megavul  (hitoshura25/megavul)")
    print("  Streaming 672K rows — filtering C rows with function code…")
    print(f"{'='*60}")

    ds = load_dataset("hitoshura25/megavul", split="train", streaming=True)

    rows: list[dict] = []
    total_seen = 0
    for raw in tqdm(ds, desc="  streaming", unit="row"):
        total_seen += 1
        if raw.get("language") != "C":
            continue
        vuln_code = raw.get("vulnerable_code")
        fixed_code = raw.get("fixed_code")
        if not vuln_code:
            continue

        cwe = raw.get("cwe_id") or ""
        cve_id = raw.get("cve_id") or ""

        # Vulnerable sample
        rows.append({
            "func_before": vuln_code,
            "func_after": fixed_code or None,
            "vul": 1,
            "CWE ID": cwe,
            "CVE ID": cve_id,
        })

        # Benign sample from fixed code (patched = not vulnerable)
        if fixed_code and fixed_code != vuln_code:
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
    print(f"\n  Streamed {total_seen:,} rows total")
    print(f"  Saved {len(df):,} rows  (vulnerable={vuln_n:,}  benign={benign_n:,})")
    print(f"  -> {out_file.relative_to(PROJECT_ROOT)}")
    print("\n  CWE distribution (top 15):")
    cwe_counts = df[df["vul"] == 1]["CWE ID"].value_counts().head(15)
    for cwe, cnt in cwe_counts.items():
        print(f"    {cwe}: {cnt}")
    print(f"\n  Use with prepare_dataset.py --format bigvul --input {out_file.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download vulnerability datasets from HuggingFace.")
    parser.add_argument(
        "--only", choices=[*DATASETS.keys(), "megavul"],
        help="Download only this dataset (default: all)",
    )
    args = parser.parse_args()

    if args.only == "megavul":
        download_megavul()
        return

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS

    print(f"Saving datasets to: {OUT_DIR}")
    for name, cfg in targets.items():
        try:
            download_one(name, cfg)
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}", file=sys.stderr)

    if not args.only:
        print("\nSkipping megavul (run with --only megavul to download separately — streams 672K rows).")

    print("\nAll done.")
    print(f"Files are in: {OUT_DIR}")
    print("\nNext step - generate CPG files with Joern:")
    print("  uv run python scripts/prepare_dataset.py \\")
    print("      --input data/datasets/devign/train.parquet \\")
    print("      --format devign --joern-cli /path/to/joern/joern-cli")


if __name__ == "__main__":
    main()
