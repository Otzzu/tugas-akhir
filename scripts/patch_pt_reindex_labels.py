"""
patch_pt_reindex_labels.py — Reindex sparse data.y to contiguous 0..N-1.

Fixes .pt files built before the dataset_lm.py reindex fix was added.
Uses class_names (already in the .pt) + cwe_vocab.json to reconstruct the
old sparse index → new contiguous index mapping.

Only data.y is remapped. data.cwe_id is left unchanged — it stores the
original cwe_vocab.json index so predictions can be traced back to the vocab.

Usage:
    python scripts/patch_pt_reindex_labels.py \
        --pt data/processed/lm_dataset_titanvul_multiclass_...pt \
        --vocab data/raw/titanvul/cwe_vocab.json

    # Multiple files:
    python scripts/patch_pt_reindex_labels.py \
        --pt path/a.pt path/b.pt \
        --vocab data/raw/titanvul/cwe_vocab.json
"""

import argparse
import json
import shutil
from pathlib import Path

import torch


def build_remap(class_names: list[str], cwe_vocab: dict[str, int]) -> dict[int, int]:
    """
    class_names[new_i] = CWE name → look up old sparse index in cwe_vocab.
    Returns {old_sparse_idx: new_contiguous_idx}.
    """
    remap: dict[int, int] = {}
    for new_i, name in enumerate(class_names):
        if name not in cwe_vocab:
            raise KeyError(f"class_names entry '{name}' not found in cwe_vocab.json")
        old_idx = cwe_vocab[name]
        remap[old_idx] = new_i
    return remap


def apply_remap_tensor(t: torch.Tensor, remap: dict[int, int], sentinel: int = -1) -> torch.Tensor:
    """Remap values in tensor; sentinel values (e.g. -1) are left unchanged."""
    out = t.clone()
    for old, new in remap.items():
        out[t == old] = new
    return out


def patch_file(pt_path: Path, cwe_vocab: dict[str, int]) -> None:
    print(f"\n[{pt_path.name}]")
    result = torch.load(pt_path, weights_only=False)

    if len(result) == 4:
        data, slices, class_names, raw_funcs = result
    elif len(result) == 3:
        data, slices, class_names = result
        raw_funcs = None
    else:
        print(f"  SKIP — unexpected tuple length {len(result)}")
        return

    if class_names is None:
        print("  SKIP — class_names is None (binary mode)")
        return

    # Build remap
    remap = build_remap(class_names, cwe_vocab)
    print(f"  {len(class_names)} classes, remap: {dict(list(remap.items())[:8])}{'...' if len(remap) > 8 else ''}")

    # Check if already contiguous (no-op case)
    old_indices = sorted(remap.keys())
    new_indices = [remap[k] for k in old_indices]
    if old_indices == new_indices:
        print("  Already contiguous — no changes needed.")
        return

    # Remap data.y
    y_before = data.y.unique().tolist()
    data.y = apply_remap_tensor(data.y, remap)
    y_after = data.y.unique().tolist()
    print(f"  data.y: {sorted(y_before)[:10]}... -> {sorted(y_after)[:10]}...")

    # data.cwe_id left unchanged — keeps original cwe_vocab.json index for traceability

    # Backup + save
    backup = pt_path.with_suffix(".pt.bak")
    shutil.copy2(pt_path, backup)
    print(f"  Backup: {backup.name}")

    out = (data, slices, class_names, raw_funcs) if raw_funcs is not None else (data, slices, class_names)
    torch.save(out, pt_path)
    print(f"  Saved: {pt_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", nargs="+", required=True, help=".pt file(s) to patch")
    parser.add_argument("--vocab", required=True, help="Path to cwe_vocab.json")
    args = parser.parse_args()

    with open(args.vocab, encoding="utf-8") as f:
        cwe_vocab: dict[str, int] = json.load(f)
    print(f"Loaded vocab: {len(cwe_vocab)} entries from {args.vocab}")

    for pt in args.pt:
        patch_file(Path(pt), cwe_vocab)

    print("\nDone.")


if __name__ == "__main__":
    main()
