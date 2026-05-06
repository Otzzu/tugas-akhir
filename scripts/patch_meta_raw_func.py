"""
scripts/patch_meta_raw_func.py — Patch existing .meta.json files with raw_func and id.

Reconstructs the exact same sampling order as prepare_dataset.py (random_state=42)
and writes the original function source text and parquet row id into each existing .meta.json.

Run this ONCE per dataset to backfill raw_func/id without regenerating CPGs.

Usage:
    uv run python scripts/patch_meta_raw_func.py \
        --input data/datasets/bigvul/train.parquet \
        --format bigvul \
        --out-dir data/raw/bigvul \
        --sample-per-class 2000

    uv run python scripts/patch_meta_raw_func.py \
        --input data/datasets/megavul/train.parquet \
        --format megavul \
        --out-dir data/raw/megavul \
        --sample-per-class 2000

Arguments mirror prepare_dataset.py exactly so the sampling is reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Reuse public loaders from prepare_dataset.py (safe — guarded by __main__)
sys.path.insert(0, str(Path(__file__).parent))
from prepare_dataset import (
    load_bigvul,
    load_devign,
    load_diversevul,
)


def _load(path: Path, fmt: str, top_cwe: int = 10) -> pd.DataFrame:
    if fmt in ("bigvul", "megavul", "merged", "titanvul"):
        df, _ = load_bigvul(path, top_k_cwe=top_cwe)
        return df
    elif fmt == "devign":
        return load_devign(path)
    elif fmt == "diversevul":
        return load_diversevul(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw_func and id into .meta.json files")
    parser.add_argument("--input",            required=True, type=Path)
    parser.add_argument("--format",           required=True,
                        choices=["bigvul", "megavul", "merged", "titanvul", "devign", "diversevul"])
    parser.add_argument("--out-dir",          required=True, type=Path,
                        help="Same --out-dir used in prepare_dataset.py (e.g. data/raw/bigvul)")
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--idx-offset",       type=int, default=0)
    parser.add_argument("--top-cwe",          type=int, default=10,
                        help="Must match value used in prepare_dataset.py (default 10)")
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--dry-run",          action="store_true",
                        help="Show what would be patched without writing")
    parser.add_argument("--force",            action="store_true",
                        help="Overwrite files that already have id+raw_func (re-patch everything)")
    args = parser.parse_args()

    logger.info(f"Loading {args.format} from {args.input}")
    df = _load(args.input, args.format, top_cwe=args.top_cwe)

    if args.limit:
        df = df.head(args.limit)

    if args.sample_per_class:
        n = args.sample_per_class
        df = pd.concat(
            [g.sample(min(len(g), n), random_state=42) for _, g in df.groupby("label")],
            ignore_index=True,
        )

    logger.info(f"Rows after sampling: {len(df)}")

    benign_dir = args.out_dir / "benign"
    vuln_dir   = args.out_dir / "vulnerable"

    # Pre-scan dirs once → O(1) lookup per row instead of one glob per row
    logger.info("Pre-scanning CPG directories...")
    def _cpg_stems(d: Path) -> set[str]:
        if not d.exists():
            return set()
        return {
            f.stem for f in d.iterdir()
            if f.suffix != ".json" and ".meta." not in f.name
        }

    benign_stems = _cpg_stems(benign_dir)
    vuln_stems   = _cpg_stems(vuln_dir)
    logger.info(f"Found {len(benign_stems)} benign CPGs, {len(vuln_stems)} vuln CPGs")

    patched = 0
    skipped_no_xml = 0
    skipped_already = 0

    for local_idx, row in enumerate(df.itertuples(index=False)):
        idx = local_idx + args.idx_offset
        class_id = int(row.label)
        phys_dir = benign_dir if class_id == 0 else vuln_dir
        stems    = benign_stems if class_id == 0 else vuln_stems
        raw_func = str(row.code)
        row_id = int(row.id) if hasattr(row, "id") else -1

        if f"func_{idx}" not in stems:
            skipped_no_xml += 1
            continue

        meta_path = phys_dir / f"func_{idx}.meta.json"

        # Read existing meta or start fresh
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if "raw_func" in meta and "id" in meta and not args.force:
                skipped_already += 1
                continue
        else:
            meta = {}

        meta["id"] = row_id
        meta["raw_func"] = raw_func

        if args.dry_run:
            preview = raw_func[:60].replace("\n", "\\n")
            logger.info(f"  [DRY] func_{idx} → id={row_id} raw_func={preview!r}…")
        else:
            meta_path.write_text(json.dumps(meta))
            if patched % 500 == 0:
                logger.info(f"  patched={patched}  skipped_no_cpg={skipped_no_xml}  skipped_already={skipped_already}")

        patched += 1

    logger.info(
        f"\nDone. patched={patched}  "
        f"skipped_no_cpg={skipped_no_xml}  "
        f"skipped_already_had_both={skipped_already}"
    )
    if args.dry_run:
        logger.info("DRY RUN — nothing written")


if __name__ == "__main__":
    main()
