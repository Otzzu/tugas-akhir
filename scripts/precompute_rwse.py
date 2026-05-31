"""Precompute Random Walk Structural Encoding (RWSE) for all graphs in a lazy dataset.

Walks data/processed/<dataset>_graphs/*.pt, computes RWSE per graph, saves back.
Each graph gets a `rwse` attribute: [num_nodes, walk_length] tensor.

Usage:
    uv run python scripts/precompute_rwse.py \\
        --graphs-dir data/processed/lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_graphs \\
        --walk-length 16

Idempotent — skips graphs that already have `rwse` attribute (unless --force).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.utils import degree
from tqdm import tqdm


def compute_rwse(edge_index: torch.Tensor, num_nodes: int, walk_length: int = 16) -> torch.Tensor:
    """Diagonal of (D^-1 A)^k for k=1..walk_length. Returns [N, walk_length]."""
    if num_nodes == 0 or edge_index.numel() == 0:
        return torch.zeros(num_nodes, walk_length)
    row, col = edge_index[0], edge_index[1]
    deg_inv = 1.0 / degree(row, num_nodes=num_nodes).clamp(min=1.0)
    edge_w = deg_inv[row]
    adj = torch.sparse_coo_tensor(edge_index, edge_w, size=(num_nodes, num_nodes)).coalesce()
    pe = torch.zeros(num_nodes, walk_length)
    M_k = adj
    for k in range(walk_length):
        idx = M_k.indices()
        vals = M_k.values()
        self_mask = idx[0] == idx[1]
        if self_mask.any():
            pe[idx[0][self_mask], k] = vals[self_mask]
        if k < walk_length - 1:
            M_k = torch.sparse.mm(M_k, adj).coalesce()
    return pe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs-dir", required=True, type=Path,
                   help="Path to <dataset>_graphs/ dir (contains 0.pt, 1.pt, ...)")
    p.add_argument("--walk-length", type=int, default=16)
    p.add_argument("--force", action="store_true", help="Recompute even if rwse attr exists")
    args = p.parse_args()

    if not args.graphs_dir.exists():
        raise SystemExit(f"graphs dir not found: {args.graphs_dir}")

    pt_files = sorted(args.graphs_dir.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    pt_files = [f for f in pt_files if f.stem.isdigit()]
    print(f"Found {len(pt_files)} graph .pt files in {args.graphs_dir}")

    skipped = computed = 0
    for f in tqdm(pt_files, desc="RWSE"):
        g = torch.load(f, weights_only=False)
        if not args.force and hasattr(g, "rwse") and g.rwse is not None and g.rwse.shape[-1] == args.walk_length:
            skipped += 1
            continue
        n = g.x.size(0)
        g.rwse = compute_rwse(g.edge_index, n, walk_length=args.walk_length)
        torch.save(g, f)
        computed += 1

    print(f"Done. Computed: {computed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
