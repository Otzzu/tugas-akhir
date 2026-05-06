"""
patch_pt_add_raw_func.py — Inject raw_funcs list into an old .pt that lacks it.

Old .pt files were saved as (data, slices, class_names) — 3-tuple.
New .pt files are (data, slices, class_names, raw_funcs) — 4-tuple.

Strategy
--------
1. Load old .pt.
2. Scan all .meta.json files under data/raw/<source>/ to build {parquet_id → raw_func}.
3. Extract parquet_id per graph from data.parquet_id tensor + slices.
4. Build raw_funcs list in graph order via lookup.
5. Save as 4-tuple.

If a graph's parquet_id is -1 or not found in the meta map, raw_func = "" for
that graph (the _try_patch_from_existing path will then call build_func_text as
fallback when tokenising later).

Usage
-----
    uv run python scripts/patch_pt_add_raw_func.py \\
        --pt data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.pt \\
        --raw-dir data/raw/bigvul

    # In-place (overwrites .pt)
    uv run python scripts/patch_pt_add_raw_func.py \\
        --pt data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.pt \\
        --raw-dir data/raw/bigvul \\
        --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Build parquet_id → raw_func map from meta.json files
# ---------------------------------------------------------------------------

def build_meta_map(raw_dir: Path) -> dict[int, str]:
    """
    Scan benign/ and vulnerable/ under raw_dir for .meta.json files.
    Returns {parquet_id (int): raw_func (str)}.
    """
    meta_map: dict[int, str] = {}
    for subdir in ("benign", "vulnerable"):
        d = raw_dir / subdir
        if not d.is_dir():
            continue
        meta_files = sorted(d.glob("*.meta.json"))
        print(f"  Scanning {len(meta_files)} meta files in {d} …")
        for mf in tqdm(meta_files, desc=f"  {subdir}", unit="file", leave=False):
            try:
                with open(mf, encoding="utf-8") as f:
                    m = json.load(f)
                pid = int(m.get("id", -1))
                raw = m.get("raw_func", "") or ""
                if pid >= 0:
                    meta_map[pid] = raw
            except Exception as e:
                print(f"  Warning: failed to read {mf}: {e}", file=sys.stderr)
    return meta_map


# ---------------------------------------------------------------------------
# Extract per-graph parquet_ids from collated data
# ---------------------------------------------------------------------------

def extract_parquet_ids(data, slices: dict) -> list[int]:
    """
    Read data.parquet_id using slices to get one int per graph.
    Returns list of ints (length = number of graphs).
    """
    if "parquet_id" not in slices:
        return []

    pid_tensor: torch.Tensor = data.parquet_id  # shape [N_graphs] (each graph = 1 value)
    pid_slices: torch.Tensor = slices["parquet_id"]  # [N_graphs + 1]
    n = len(pid_slices) - 1

    result = []
    for i in range(n):
        s = int(pid_slices[i].item())
        e = int(pid_slices[i + 1].item())
        val = pid_tensor[s:e]
        result.append(int(val[0].item()) if len(val) > 0 else -1)
    return result


# ---------------------------------------------------------------------------
# Main patch logic
# ---------------------------------------------------------------------------

def patch(pt_path: Path, raw_dir: Path, out_path: Path) -> None:
    print(f"Loading  {pt_path}")
    result = torch.load(pt_path, weights_only=False)

    if len(result) == 4:
        data, slices, class_names, existing_raw_funcs = result
        if existing_raw_funcs:
            print(
                f"raw_funcs already present ({len(existing_raw_funcs)} entries). "
                "Use --force to overwrite."
            )
            if out_path != pt_path:
                print(f"Copying unchanged → {out_path}")
                torch.save(result, out_path)
            return
        print("  raw_funcs present but empty — will repopulate.")
    elif len(result) == 3:
        data, slices, class_names = result
        print("  Old 3-tuple format — will add raw_funcs.")
    elif len(result) == 2:
        data, slices = result
        class_names = None
        print("  Old 2-tuple format — will add raw_funcs.")
    else:
        print(f"ERROR: unexpected .pt tuple length {len(result)}", file=sys.stderr)
        sys.exit(1)

    # ── Build meta map ──────────────────────────────────────────────────────
    if not raw_dir.is_dir():
        print(f"ERROR: raw_dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning meta.json files in {raw_dir} …")
    meta_map = build_meta_map(raw_dir)
    print(f"  {len(meta_map)} parquet_id → raw_func entries loaded.")

    # ── Extract parquet_ids from .pt ────────────────────────────────────────
    parquet_ids = extract_parquet_ids(data, slices)

    if not parquet_ids:
        print(
            "ERROR: parquet_id not found in slices.\n"
            "This .pt was built before parquet_id was added — must rebuild from scratch.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_graphs = len(parquet_ids)
    print(f"  {n_graphs} graphs in .pt.")

    # ── Build raw_funcs list ────────────────────────────────────────────────
    raw_funcs: list[str] = []
    n_missing = 0
    n_negative = 0

    for pid in parquet_ids:
        if pid < 0:
            raw_funcs.append("")
            n_negative += 1
        elif pid in meta_map:
            raw_funcs.append(meta_map[pid])
        else:
            raw_funcs.append("")
            n_missing += 1

    n_found = n_graphs - n_missing - n_negative
    print(
        f"  Matched: {n_found}/{n_graphs}  "
        f"(missing={n_missing}, parquet_id=-1: {n_negative})"
    )
    if n_missing > 0:
        print(
            f"  Warning: {n_missing} graphs had no matching meta.json entry. "
            "raw_func=\"\" for those — tokenization will fall back to build_func_text."
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    if out_path == pt_path:
        print(f"Overwriting  {out_path}")
    else:
        print(f"Saving →  {out_path}")

    torch.save((data, slices, class_names, raw_funcs), out_path)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject raw_funcs list into old .pt files that lack it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pt", required=True, type=Path,
        metavar="FILE",
        help="Input .pt file to patch.",
    )
    parser.add_argument(
        "--raw-dir", required=True, type=Path,
        metavar="DIR",
        help="data/raw/<source> directory containing benign/ and vulnerable/ with .meta.json files.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        metavar="FILE",
        help="Output .pt path. Default: <input>_rawfunc.pt alongside input.",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the input file (mutually exclusive with --out).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite raw_funcs even if already present.",
    )
    args = parser.parse_args()

    if args.in_place and args.out:
        parser.error("--in-place and --out are mutually exclusive.")

    pt_path = args.pt.resolve()
    if not pt_path.exists():
        print(f"ERROR: not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    raw_dir = args.raw_dir.resolve()

    if args.in_place:
        out_path = pt_path
    elif args.out:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = pt_path.parent / f"{pt_path.stem}_rawfunc.pt"
        print(f"Auto output: {out_path}")

    # Handle --force: reload and clear existing raw_funcs before patch
    if args.force:
        result = torch.load(pt_path, weights_only=False)
        if len(result) == 4:
            data, slices, class_names, _ = result
            tmp = pt_path.parent / f"_patch_tmp_{pt_path.name}"
            torch.save((data, slices, class_names), tmp)
            pt_path_to_use = tmp
        else:
            pt_path_to_use = pt_path
    else:
        pt_path_to_use = pt_path

    patch(pt_path_to_use, raw_dir, out_path)

    if args.force and pt_path_to_use != pt_path:
        pt_path_to_use.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
