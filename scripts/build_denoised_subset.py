"""build_denoised_subset.py — derive a cpg14-denoised dataset from a built LAZY dataset.

Drops non-cpg14 edges (keeps AST / CFG / CDG / REACHING_DEF) then keeps the LARGEST
connected component. The bookkeeping edges (CONTAINS, CALL, DOMINATE, EVAL_TYPE, ...) are
what tie the synthetic stub methods (operators, external-call stubs, <global>, file/namespace)
to the function; dropping them disconnects those stubs into small components, so keeping the
largest component removes them along with ~6% peripheral nodes. Reuses node features verbatim
(NO re-embed) so this is cheap, like build_vuln_only_subset.py.

Measured on megavul (30-graph sample): ~41% of edges and ~1/3 of nodes are bookkeeping/synthetic.

Run (cloud, Linux):
    PYTHONPATH=src python scripts/build_denoised_subset.py \
        --processed-dir data/processed --ds-name <base_no_meta>
Then set  data.ds_name_suffix: _cpg14  in the config (model.num_classes unchanged).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from gnn_vuln.data.cpg.constants import EDGE_TYPES

CPG14 = ["AST", "CFG", "CDG", "REACHING_DEF"]


def denoise_graph(g, keep_idx: set[int]):
    """Keep only cpg14 edges, then the largest connected component. Returns (g, lost_flaw)."""
    N = int(g.num_nodes)
    E = g.edge_index.shape[1] if g.edge_index is not None else 0
    if N == 0 or E == 0:
        return g, False

    etype = g.edge_attr.argmax(dim=1)
    emask = torch.zeros(E, dtype=torch.bool)
    for k in keep_idx:
        emask |= etype == k
    ei = g.edge_index[:, emask]
    ea = g.edge_attr[emask]

    if ei.numel() == 0:
        keep = torch.zeros(N, dtype=torch.bool)
        keep[0] = True
    else:
        rows, cols = ei[0].numpy(), ei[1].numpy()
        adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        _, labels = connected_components(adj, directed=False)
        biggest = np.bincount(labels).argmax()
        keep = torch.from_numpy(labels == biggest)

    # did we drop a flaw-line node?
    lost_flaw = False
    fm = getattr(g, "flaw_line_mask", None)
    if torch.is_tensor(fm) and fm.shape[0] == N:
        lost_flaw = bool((fm.bool() & ~keep).any())

    idx_map = torch.full((N,), -1, dtype=torch.long)
    idx_map[keep] = torch.arange(int(keep.sum()))
    e_in = keep[ei[0]] & keep[ei[1]]
    g.edge_index = idx_map[ei[:, e_in]]
    g.edge_attr = ea[e_in]
    g.x = g.x[keep]
    for attr in ("node_line", "node_end_line", "flaw_line_mask"):
        v = getattr(g, attr, None)
        if torch.is_tensor(v) and v.shape[0] == N:
            setattr(g, attr, v[keep])
    return g, lost_flaw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--ds-name", required=True, help="base dataset name (no _meta.pt suffix)")
    ap.add_argument("--suffix", default="_cpg14")
    ap.add_argument("--keep-edges", nargs="+", default=CPG14,
                    help=f"edge types to keep (default cpg14: {CPG14})")
    ap.add_argument("--limit", type=int, default=0, help="process only first N graphs (debug)")
    a = ap.parse_args()

    pdir = Path(a.processed_dir)
    base_meta = pdir / f"{a.ds_name}_meta.pt"
    base_graphs = pdir / f"{a.ds_name}_graphs"
    if not base_meta.exists():
        sys.exit(f"missing meta: {base_meta}")
    if not base_graphs.exists():
        sys.exit(f"missing graphs dir: {base_graphs}")

    meta = torch.load(base_meta, weights_only=False)
    n = int(meta["n_graphs"])
    if a.limit:
        n = min(n, a.limit)
    keep_idx = {EDGE_TYPES.index(t) for t in a.keep_edges}
    print(f"keep edge types {a.keep_edges} -> idx {sorted(keep_idx)}", file=sys.stderr)

    dst = pdir / f"{a.ds_name}{a.suffix}_graphs"
    dst.mkdir(parents=True, exist_ok=True)

    labels: list[int] = []
    tn0 = tn1 = te0 = te1 = lost = 0
    for i in tqdm(range(n), desc="denoise", unit="g"):
        g = torch.load(base_graphs / f"{i}.pt", weights_only=False)
        tn0 += int(g.num_nodes); te0 += int(g.edge_index.shape[1])
        g, lf = denoise_graph(g, keep_idx)
        tn1 += int(g.num_nodes); te1 += int(g.edge_index.shape[1]); lost += int(lf)
        torch.save(g, dst / f"{i}.pt")
        labels.append(int(g.y))

    torch.save({"n_graphs": n, "class_names": list(meta["class_names"])},
               pdir / f"{a.ds_name}{a.suffix}_meta.pt")
    torch.save(torch.tensor(labels, dtype=torch.long), dst / "_labels.pt")

    print(f"DONE -> {a.ds_name}{a.suffix}", file=sys.stderr)
    print(f"  nodes {tn0} -> {tn1} ({100*tn1/tn0:.0f}% kept)  "
          f"edges {te0} -> {te1} ({100*te1/te0:.0f}% kept)", file=sys.stderr)
    print(f"  graphs that lost a flaw-line node: {lost}/{n} "
          f"({100*lost/max(n,1):.1f}%)", file=sys.stderr)
    print(f"set in config:  data.ds_name_suffix: {a.suffix}", file=sys.stderr)


if __name__ == "__main__":
    main()
