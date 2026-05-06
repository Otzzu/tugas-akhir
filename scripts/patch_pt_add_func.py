"""
patch_pt_add_func.py — Add func_input_ids / func_attention_mask to an existing .pt file.

The .pt must already contain raw_funcs (stored at build time by dataset_lm.py).
Tokenizes all raw functions with the given model, injects the tensors, updates
the PyG slices dict, and saves the result.

No re-embedding of nodes — the expensive CodeBERT node step is skipped entirely.

Usage
-----
    # Same func_lm as node LM (most common)
    uv run python scripts/patch_pt_add_func.py \\
        --pt data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_top10.pt \\
        --func-lm microsoft/unixcoder-base

    # Different func_lm (creates separate output; node embeddings unchanged)
    uv run python scripts/patch_pt_add_func.py \\
        --pt data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_top10.pt \\
        --func-lm Salesforce/codet5p-110m-embedding \\
        --out data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_live_codet5p_ft_top10.pt

    # In-place (overwrites input)
    uv run python scripts/patch_pt_add_func.py \\
        --pt data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.pt \\
        --func-lm microsoft/unixcoder-base \\
        --in-place

Slices
------
PyG InMemoryDataset stores each per-graph [1, max_length] func token tensor concatenated
along dim=0 into [N, max_length]. Slices = [0, 1, 2, ..., N] (one row per graph).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tokenise
# ---------------------------------------------------------------------------

def _tokenise(raw_funcs: list[str], func_lm: str, max_length: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(func_lm, trust_remote_code=True)

    ids_list, mask_list = [], []
    for i in tqdm(range(0, len(raw_funcs), batch_size), desc="tokenize", unit="batch"):
        batch = raw_funcs[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids_list.append(enc["input_ids"])
        mask_list.append(enc["attention_mask"])

    return torch.cat(ids_list, dim=0), torch.cat(mask_list, dim=0)  # [N, max_length]


# ---------------------------------------------------------------------------
# Auto output path
# ---------------------------------------------------------------------------

def _auto_out_path(pt_path: Path, func_lm: str) -> Path:
    """
    Derive a sensible output filename from the input stem.

    Rules (matching dataset_lm.py processed_file_names):
      - Insert _ft right before _top / _s<N>r<N> / _f<hash> suffixes (or at end).
      - If func_lm short name differs from node LM (embedded in filename), also
        insert _live_<func_short> before _ft.
    """
    stem = pt_path.stem  # e.g. lm_dataset_bigvul_multiclass_unixcoder-base_top10

    # Extract node LM short name: third segment after lm_dataset_<source>_<mode>_
    lm_short_m = re.match(r"lm_dataset_[^_]+_[^_]+_(.+?)(?:_top\d+|_s\d+r\d+|_f[0-9a-f]{8}|$)", stem)
    node_lm_short = lm_short_m.group(1) if lm_short_m else ""
    # Strip existing _live_* suffix from node_lm_short if present
    node_lm_short = re.sub(r"_live_.+$", "", node_lm_short)

    func_short = func_lm.split("/")[-1]

    # Remove any existing _ft to avoid doubling
    stem = re.sub(r"_ft(?=_|$)", "", stem)

    # Find insertion point for _ft (before _top / _s<N>r<N> / _f<hash8>)
    m = re.search(r"(_top\d+|_s\d+r\d+|_f[0-9a-f]{8})", stem)
    if m:
        insert_at = m.start()
    else:
        insert_at = len(stem)

    live_tag = f"_live_{func_short}" if func_short != node_lm_short else ""

    new_stem = stem[:insert_at] + live_tag + "_ft" + stem[insert_at:]
    return pt_path.parent / f"{new_stem}.pt"


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

def patch(
    pt_path: Path,
    func_lm: str,
    out_path: Path,
    max_length: int = 512,
    batch_size: int = 64,
    overwrite_tokens: bool = False,
) -> None:
    print(f"Loading  {pt_path}")
    result = torch.load(pt_path, weights_only=False)

    if len(result) == 4:
        data, slices, class_names, raw_funcs = result
    elif len(result) == 3:
        data, slices, class_names = result
        raw_funcs = None
    else:
        data, slices = result
        class_names = None
        raw_funcs = None

    if not raw_funcs:
        print(
            "ERROR: .pt has no raw_funcs list.\n"
            "Re-build the dataset from scratch with add_func_tokens=true — "
            "raw_funcs are populated during initial graph processing.",
            file=sys.stderr,
        )
        sys.exit(1)

    n = len(raw_funcs)

    if "func_input_ids" in slices and not overwrite_tokens:
        print(
            f"func_input_ids already present ({n} graphs). "
            "Use --overwrite to re-tokenize with a different func_lm."
        )
    else:
        print(f"Tokenizing {n} functions with {func_lm} (max_length={max_length})…")
        all_ids, all_mask = _tokenise(raw_funcs, func_lm, max_length, batch_size)

        data.func_input_ids      = all_ids    # [N, max_length]
        data.func_attention_mask = all_mask   # [N, max_length]

        # Each graph occupies exactly one row → slices = [0, 1, 2, ..., N]
        func_slice = torch.arange(n + 1, dtype=torch.long)
        slices["func_input_ids"]      = func_slice
        slices["func_attention_mask"] = func_slice.clone()

        print(f"  func_input_ids shape : {all_ids.shape}")
        print(f"  func_attention_mask  : {all_mask.shape}")

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
        description="Patch a processed .pt to add func_input_ids / func_attention_mask.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pt", required=True, type=Path,
        metavar="FILE",
        help="Input .pt file (must contain raw_funcs).",
    )
    parser.add_argument(
        "--func-lm", required=True,
        metavar="MODEL",
        help="HuggingFace model ID to use for function tokenization.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        metavar="FILE",
        help="Output .pt path. Default: auto-derived from input stem.",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the input file (mutually exclusive with --out).",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Tokenizer max sequence length (default: 512).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Tokenization batch size (default: 64).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-tokenize even if func_input_ids already exists in the .pt.",
    )
    args = parser.parse_args()

    if args.in_place and args.out:
        parser.error("--in-place and --out are mutually exclusive.")

    pt_path = args.pt.resolve()
    if not pt_path.exists():
        print(f"ERROR: not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    if args.in_place:
        out_path = pt_path
    elif args.out:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = _auto_out_path(pt_path, args.func_lm)
        print(f"Auto output: {out_path}")

    patch(pt_path, args.func_lm, out_path, args.max_length, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
