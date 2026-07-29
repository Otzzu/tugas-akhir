"""Render the FULL Joern CPG (repr=all, every edge type) of one function to PNG.

Shows what the model actually consumes, versus the readable cpg14 illustration.
Needs graphviz `dot` on PATH (GV below points at the portable extract) and Joern.

    uv run python scripts/render_full_cpg.py
"""
from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

GV = r"C:\Users\Otzzu\AppData\Local\Temp\gv\Graphviz-15.1.0-win64\bin"
if Path(GV).exists():
    os.environ["PATH"] = GV + os.pathsep + os.environ["PATH"]

import pandas  # noqa: F401  (load before torch on Windows)
import networkx as nx
import pydot

from gnn_vuln.data.joern_runner import process_function

FUNC = """static void copyIPv6IfDifferent(void * dest, const void * src)
{
\tif(dest != src) {
\t\tmemcpy(dest, src, sizeof(struct in6_addr));
\t}
}"""

OUT = Path("docs/laporan-individu/image/bab-3/cpg_full.png")
OUT_CPG14 = Path("docs/laporan-individu/image/bab-3/cpg14.png")
JOERN = Path("C:/joern/joern-cli")
# cpg14 = keep only AST/CFG/CDG/REACHING_DEF, drop Joern bookkeeping edges, then the
# largest connected component (matches scripts/build_denoised_subset.py).
CPG14_EDGES = {"AST", "CFG", "CDG", "REACHING_DEF"}

# distinct colors cycled over whatever edge types Joern emits
PALETTE = ["#000000", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79",
           "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#a55194"]


def _pick(attrs: dict, *names):
    for n in names:
        if n in attrs and attrs[n]:
            return attrs[n]
    return ""


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="fullcpg_"))
    gpath = process_function(code=FUNC, idx=0, out_dir=work, joern_cli_dir=JOERN, fmt="graphml")
    if gpath is None:
        raise SystemExit("Joern produced no CPG")
    G = nx.read_graphml(gpath)
    print(f"nodes {G.number_of_nodes()}  edges {G.number_of_edges()}")

    # find the edge-type attribute: the key whose values look like AST/CFG/...
    edge_key, sample = None, {}
    for _, _, d in G.edges(data=True):
        sample = d
        break
    for k, v in sample.items():
        if isinstance(v, str) and v.isupper():
            edge_key = k
            break
    print("edge sample attrs:", sample, "-> type key:", edge_key)

    types = Counter(d.get(edge_key, "?") for *_e, d in G.edges(data=True))
    print("edge types:", dict(types))
    color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(sorted(types))}

    def render(graph, out: Path) -> None:
        dot = pydot.Dot(graph_type="digraph", rankdir="TB", splines="true", fontname="Helvetica")
        for n, d in graph.nodes(data=True):
            ntype = _pick(d, "labelV", "label", "TYPE_FULL_NAME")
            code = _pick(d, "CODE", "NAME", "FULL_NAME")
            lab = f"{ntype}\\n{code}"[:60].replace('"', "'")
            dot.add_node(pydot.Node(str(n), label=lab, shape="box", fontsize="9"))
        for u, v, d in graph.edges(data=True):
            t = d.get(edge_key, "?")
            dot.add_edge(pydot.Edge(str(u), str(v), color=color[t], label=t, fontsize="7", fontcolor=color[t]))
        out.parent.mkdir(parents=True, exist_ok=True)
        dot.write_png(str(out))
        print("wrote", out, "| nodes", graph.number_of_nodes(), "edges", graph.number_of_edges())

    render(G, OUT)

    # cpg14: keep only AST/CFG/CDG/REACHING_DEF, then the largest weakly connected component.
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if d.get(edge_key) in CPG14_EDGES:
            H.add_edge(u, v, **d)
    H.remove_nodes_from(list(nx.isolates(H)))
    if H.number_of_nodes():
        biggest = max(nx.weakly_connected_components(H), key=len)
        H = H.subgraph(biggest).copy()
    render(H, OUT_CPG14)
    print("edge-type legend:", color)


if __name__ == "__main__":
    main()
