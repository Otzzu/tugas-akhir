"""CPG diameter analysis — measures long-range reach to judge over-squashing.

Diameter = longest shortest-path (undirected). A flat L-layer GNN only reaches L
hops, so diameter > L means message passing under-reaches → over-squashing, which
is exactly what a Graph-ViT / global-mixer would fix.

Approx via 2-sweep BFS (exact for trees, tight lower bound otherwise) on the
largest connected component. Reads data/graphs/megavul.hdf5 (benign + vulnerable).

Run: uv run python scripts/analyze_diameter.py
"""
from __future__ import annotations

import sys
import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

H5 = "data/graphs/megavul.hdf5"
GNN_LAYERS = 4  # your flat GAT reach


def approx_diameter(src, dst, n):
    """2-sweep BFS approx diameter on largest CC. Returns (diam, lcc_size)."""
    if n <= 1 or len(src) == 0:
        return 0, n
    r = np.concatenate([src, dst])
    c = np.concatenate([dst, src])
    A = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    ncomp, labels = connected_components(A, directed=False)
    if ncomp > 1:
        vals, counts = np.unique(labels, return_counts=True)
        big = vals[counts.argmax()]
        nodes = np.where(labels == big)[0]
        A = A[nodes][:, nodes]
        n = len(nodes)
        if n <= 1:
            return 0, n
    lcc = n
    # sweep 1: from node 0 → farthest u
    d0 = shortest_path(A, method="D", indices=0, unweighted=True)
    d0[np.isinf(d0)] = -1
    u = int(d0.argmax())
    # sweep 2: from u → max dist = approx diameter
    du = shortest_path(A, method="D", indices=u, unweighted=True)
    du[np.isinf(du)] = 0
    return int(du.max()), lcc


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else 0.0


def report_split(name, diam, nodes):
    diam = np.asarray(diam)
    nodes = np.asarray(nodes)
    n = len(diam)
    lines = [f"### {name}\n", "| Metric | Value |", "| --- | --- |",
             f"| Count | {n} |",
             f"| Mean | {diam.mean():.1f} |",
             f"| Median (P50) | {pct(diam,50):.0f} |",
             f"| P90 | {pct(diam,90):.0f} |",
             f"| P95 | {pct(diam,95):.0f} |",
             f"| P99 | {pct(diam,99):.0f} |",
             f"| Max | {diam.max()} |"]
    for thr in (4, 6, 8, 10, 16):
        c = int((diam > thr).sum())
        lines.append(f"| diameter > {thr} | {c} ({100*c/n:.1f}%) |")
    # diameter by node-count bucket (does size drive diameter?)
    lines += ["", "| Node bucket | Count | Mean diam | % diam>4 |",
              "| --- | --- | --- | --- |"]
    for lo, hi in [(0,100),(100,200),(200,500),(500,1000),(1000,10**9)]:
        m = (nodes >= lo) & (nodes < hi)
        if m.sum():
            db = diam[m]
            lab = f"[{lo}, {hi if hi<10**9 else 'inf'})"
            lines.append(f"| {lab} | {m.sum()} | {db.mean():.1f} | {100*(db>4).mean():.0f}% |")
    return "\n".join(lines) + "\n"


def main():
    f = h5py.File(H5, "r")
    out = {"benign": ([], []), "vulnerable": ([], [])}  # split -> (diam, nodes)
    for split in ("benign", "vulnerable"):
        if split not in f:
            continue
        g = f[split]
        keys = list(g.keys())
        for i, k in enumerate(keys):
            fn = g[k]
            n = fn["node_type"].shape[0]
            src = fn["edge_src"][...].astype(np.int64)
            dst = fn["edge_dst"][...].astype(np.int64)
            d, _ = approx_diameter(src, dst, n)
            out[split][0].append(d)
            out[split][1].append(n)
            if (i + 1) % 2000 == 0:
                print(f"  {split}: {i+1}/{len(keys)}", file=sys.stderr)
    f.close()

    all_d = out["benign"][0] + out["vulnerable"][0]
    all_n = out["benign"][1] + out["vulnerable"][1]

    md = ["# CPG Diameter Analysis (megavul.hdf5)\n",
          f"Undirected approx diameter (2-sweep BFS, largest CC). "
          f"Flat GNN reach = {GNN_LAYERS} hops → **diameter > {GNN_LAYERS} = over-squashing risk**.\n",
          report_split("megavul / all", all_d, all_n),
          report_split("megavul / benign", *out["benign"]),
          report_split("megavul / vulnerable", *out["vulnerable"])]
    txt = "\n".join(md)
    with open("DIAMETER_ANALYSIS.md", "w", encoding="utf-8") as fh:
        fh.write(txt)
    print("Wrote DIAMETER_ANALYSIS.md", file=sys.stderr)


if __name__ == "__main__":
    main()
