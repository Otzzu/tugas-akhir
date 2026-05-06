"""
patch_pt_deoffset_edge_index.py — Convert edge_index from global to local indices.

PyG 2.7.0 changed InMemoryDataset.get() to decrement=False — it no longer
subtracts the cumulative node offset when reconstructing individual graphs.
.pt files built with the old streaming collate stored GLOBAL edge_index
(local + cumulative_node_offset). This patch converts them to LOCAL indices.

Usage:
    python scripts/patch_pt_deoffset_edge_index.py \
        --pt data/processed/lm_dataset_titanvul_multiclass_...pt

    # Multiple files:
    python scripts/patch_pt_deoffset_edge_index.py \
        --pt path/a.pt path/b.pt
"""

import argparse
import shutil
from pathlib import Path

import torch


def is_already_local(data, slices) -> bool:
    """Return True if stored edge_index is already local (0..n_nodes-1 per graph)."""
    n_graphs = len(slices["y"]) - 1
    checks = min(50, n_graphs)
    for i in range(checks):
        n_nodes = int(slices["x"][i + 1]) - int(slices["x"][i])
        e0, e1 = int(slices["edge_index"][i]), int(slices["edge_index"][i + 1])
        if e1 > e0:
            ei_raw = data.edge_index[:, e0:e1]
            # Local indices must be in [0, n_nodes). If any exceed that, still global.
            if ei_raw.max().item() >= n_nodes or ei_raw.min().item() < 0:
                return False
    return True


def patch_file(pt_path: Path) -> None:
    print(f"\n[{pt_path.name}]")
    result = torch.load(pt_path, weights_only=False)

    if len(result) == 4:
        data, slices, class_names, raw_funcs = result
    elif len(result) == 3:
        data, slices, class_names = result
        raw_funcs = None
    elif len(result) == 2:
        data, slices = result
        class_names = raw_funcs = None
    else:
        print(f"  SKIP — unexpected tuple length {len(result)}")
        return

    if "edge_index" not in slices or "x" not in slices:
        print("  SKIP — no edge_index or x in slices")
        return

    if is_already_local(data, slices):
        print("  Already local indices — no changes needed.")
        return

    n_graphs = len(slices["y"]) - 1
    print(f"  {n_graphs} graphs — converting global -> local edge_index...")

    new_ei = data.edge_index.clone()
    for i in range(n_graphs):
        e0, e1 = int(slices["edge_index"][i]), int(slices["edge_index"][i + 1])
        if e1 > e0:
            node_off = int(slices["x"][i])
            new_ei[:, e0:e1] -= node_off

    # Verify
    bad = 0
    for i in range(n_graphs):
        n_nodes = int(slices["x"][i + 1]) - int(slices["x"][i])
        e0, e1 = int(slices["edge_index"][i]), int(slices["edge_index"][i + 1])
        if e1 > e0:
            ei = new_ei[:, e0:e1]
            if ei.max().item() >= n_nodes or ei.min().item() < 0:
                bad += 1
    if bad > 0:
        print(f"  ERROR: {bad} graphs still have bad edge_index after patch — aborting")
        return

    data.edge_index = new_ei

    backup = pt_path.with_suffix(".pt.bak2")
    shutil.copy2(pt_path, backup)
    print(f"  Backup: {backup.name}")

    if len(result) == 4:
        torch.save((data, slices, class_names, raw_funcs), pt_path)
    elif len(result) == 3:
        torch.save((data, slices, class_names), pt_path)
    else:
        torch.save((data, slices), pt_path)
    print(f"  Saved: {pt_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", nargs="+", required=True, help=".pt file(s) to patch")
    args = parser.parse_args()

    for pt in args.pt:
        patch_file(Path(pt))

    print("\nDone.")


if __name__ == "__main__":
    main()
