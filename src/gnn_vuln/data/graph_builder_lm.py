"""
graph_builder_lm.py
~~~~~~~~~~~~~~~~~~~
Builds PyG Data objects from Joern JSON CPGs using CodeBERT node embeddings.

Node feature vector (773D):
    [node_type_idx (1)] + [CodeBERT CLS token (768)] + [dist_features (3)] + [danger_api (1)]
Edge feature vector (7D one-hot):
    {AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF}

This module is only used during dataset preprocessing (CodeBERTGraphDataset.process).
The resulting Data objects are cached to disk so CodeBERT is only run once.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data

from gnn_vuln.data.node_embedder import (
    CODEBERT_DIM,
    NODE_FEAT_DIM,
    NODE_TYPE_TO_IDX,
    CodeBERTNodeEmbedder,
)

EDGE_TYPES = ["AST", "CFG", "CDG", "DDG", "PDG", "CALL", "REACHING_DEF"]
EDGE_TYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPES)}

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

# Known dangerous C/C++ APIs — used as a binary node feature
DANGEROUS_APIS = frozenset({
    "malloc", "free", "realloc", "calloc",
    "memcpy", "memmove", "memset", "memcmp",
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strlen",
    "sprintf", "snprintf", "vsprintf", "vsnprintf",
    "gets", "fgets", "scanf", "sscanf",
    "read", "write", "recv", "send",
    "open", "fopen", "popen",
    "system", "execl", "execv", "execve",
    "alloca",
})

_TOKEN_RE = re.compile(r'\b\w+\b')


def _parse_graphml(path: str | Path) -> dict:
    """
    Parse a Joern GraphML (.xml) export into the same dict structure
    that build_lm_graph_from_json expects:
        {"nodes": [{id, labelV, code, lineNumber, ...}], "edges": [{src, dst, label}]}

    Joern stores attributes with uppercase names (CODE, LINE_NUMBER, labelE).
    We normalise them here so the rest of graph_builder_lm.py is unchanged.
    """
    def _tag(name: str) -> str:
        return f"{{{_GRAPHML_NS}}}{name}"

    tree = ET.parse(path)
    root = tree.getroot()

    # key id -> attr.name  (e.g. "node__METHOD__CODE" -> "CODE")
    key_map: dict[str, str] = {}
    for key_el in root.iter(_tag("key")):
        kid = key_el.get("id", "")
        aname = key_el.get("attr.name", kid)
        key_map[kid] = aname

    nodes: list[dict] = []
    edges: list[dict] = []

    for graph_el in root.iter(_tag("graph")):
        for node_el in graph_el.findall(_tag("node")):
            nd: dict = {"id": node_el.get("id")}
            for data_el in node_el.findall(_tag("data")):
                kid = data_el.get("key", "")
                aname = key_map.get(kid, kid)
                val = data_el.text or ""
                # Normalise to what graph_builder_lm expects
                if aname == "labelV":
                    nd["labelV"] = val
                elif aname == "CODE":
                    nd["code"] = val
                elif aname == "LINE_NUMBER":
                    nd["lineNumber"] = val
                else:
                    nd[aname] = val
            nodes.append(nd)

        for edge_el in graph_el.findall(_tag("edge")):
            ed: dict = {
                "src": edge_el.get("source"),
                "dst": edge_el.get("target"),
            }
            for data_el in edge_el.findall(_tag("data")):
                kid = data_el.get("key", "")
                aname = key_map.get(kid, kid)
                val = data_el.text or ""
                # labelE -> label so _edge_attr() finds it
                ed["label" if aname == "labelE" else aname] = val
            edges.append(ed)

    return {"nodes": nodes, "edges": edges}


def _node_type_idx(attrs: dict) -> float:
    # Joern v4 GraphML uses 'labelV'; flat JSON uses 'label'
    label = str(attrs.get("labelV", attrs.get("label", "UNKNOWN")))
    return float(NODE_TYPE_TO_IDX.get(label, NODE_TYPE_TO_IDX["UNKNOWN"]))


def _edge_attr(edge_attrs: dict) -> list[float]:
    etype = edge_attrs.get("label", "AST")
    idx = EDGE_TYPE_TO_IDX.get(etype, 0)
    one_hot = [0.0] * len(EDGE_TYPES)
    one_hot[idx] = 1.0
    return one_hot


def _node_line(attrs: dict) -> int:
    """Extract source line number from a CPG node; -1 if absent."""
    ln = attrs.get("lineNumber", attrs.get("LINE_NUMBER", None))
    try:
        return int(ln) if ln is not None else -1
    except (ValueError, TypeError):
        return -1


def _bfs_distances(
    sources: list[str],
    adjacency: dict[str, list[str]],
) -> dict[str, int]:
    """Multi-source BFS on node-id graph. Returns {node_id: hop_distance}."""
    dist: dict[str, int] = {}
    queue: deque = deque()
    for s in sources:
        if s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        curr = queue.popleft()
        for nb in adjacency.get(curr, []):
            if nb not in dist:
                dist[nb] = dist[curr] + 1
                queue.append(nb)
    return dist


def _compute_distance_features(
    nodes: list[dict],
    edges: list[dict],
) -> torch.Tensor:
    """
    Compute 3 VulChecker-inspired structural distance features per node,
    all normalised to [0, 1] by dividing by (N-1):

        [0] distance_from_entry      hops from METHOD node (forward edges)
        [1] distance_to_exit         hops to RETURN/METHOD_RETURN (reverse edges)
        [2] distance_to_nearest_call hops to nearest CALL node (reverse edges)

    Unreachable nodes get distance 1.0 (clamped max).
    Returns (N, 3) float32 tensor.
    """
    n = len(nodes)
    norm = max(n - 1, 1)

    adj:  dict[str, list[str]] = {nd["id"]: [] for nd in nodes}
    radj: dict[str, list[str]] = {nd["id"]: [] for nd in nodes}
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if s in adj and d in adj:
            adj[s].append(d)
            radj[d].append(s)

    def _label(nd: dict) -> str:
        return str(nd.get("labelV", nd.get("label", "UNKNOWN"))).upper()

    entry_ids = [nd["id"] for nd in nodes if _label(nd) == "METHOD"] or [nodes[0]["id"]]
    exit_ids  = [nd["id"] for nd in nodes if _label(nd) in ("RETURN", "METHOD_RETURN")]
    call_ids  = [nd["id"] for nd in nodes if _label(nd) == "CALL"]

    d_entry = _bfs_distances(entry_ids, adj)
    d_exit  = _bfs_distances(exit_ids,  radj)
    d_call  = _bfs_distances(call_ids,  radj)

    feats = [
        [
            min(d_entry.get(nd["id"], n) / norm, 1.0),
            min(d_exit.get(nd["id"],  n) / norm, 1.0),
            min(d_call.get(nd["id"],  n) / norm, 1.0),
        ]
        for nd in nodes
    ]
    return torch.tensor(feats, dtype=torch.float)  # (N, 3)


def _dangerous_api_flags(codes: list[str]) -> torch.Tensor:
    """
    Returns (N, 1) float32 tensor: 1.0 if the node code contains a known
    dangerous API name (matched as a whole word), else 0.0.
    """
    flags = [
        1.0 if any(t in DANGEROUS_APIS for t in _TOKEN_RE.findall(code)) else 0.0
        for code in codes
    ]
    return torch.tensor(flags, dtype=torch.float).unsqueeze(1)  # (N, 1)


def parse_cpg(path: str | Path, max_nodes: int = 500) -> Optional[dict]:
    """
    Parse a Joern CPG file (.json or .xml/.graphml) and validate its size.

    Returns a dict with keys {nodes, edges, codes} ready for build_from_parsed(),
    or None if the file is empty or exceeds max_nodes.
    Separating parsing from embedding lets the caller batch-embed all graphs at once.
    """
    path = Path(path)
    if path.suffix.lower() in (".xml", ".graphml"):
        cpg = _parse_graphml(path)
    else:
        with open(path, encoding="utf-8") as f:
            cpg = json.load(f)

    nodes = cpg.get("nodes", [])
    edges = cpg.get("edges", [])

    if not nodes or len(nodes) > max_nodes:
        return None

    cpg["codes"] = [str(n.get("code", "")) for n in nodes]
    return cpg


def build_func_text(cpg: dict) -> str:
    """
    Reconstruct a readable function body from CPG node code snippets,
    sorted by source line number. Used to tokenize the full function for
    live CodeBERT injection (Architecture 3 / 4).
    """
    nodes = cpg["nodes"]
    codes = cpg["codes"]
    line_codes: dict[int, str] = {}
    for code, node in zip(codes, nodes):
        line = _node_line(node)
        if line >= 0 and code.strip() and line not in line_codes:
            line_codes[line] = code
    if line_codes:
        return "\n".join(line_codes[l] for l in sorted(line_codes))
    # Fallback: join first 50 code snippets (no line info available)
    return " ".join(c for c in codes[:50] if c.strip()) or "<empty>"


def build_from_parsed(
    cpg: dict,
    cls_feats: torch.Tensor,
    label: int,
    flaw_lines: list[int] | None = None,
    func_input_ids: torch.Tensor | None = None,
    func_attention_mask: torch.Tensor | None = None,
) -> Data:
    """
    Build a PyG Data object from a pre-parsed CPG and pre-computed CodeBERT embeddings.

    Parameters
    ----------
    cpg       : dict returned by parse_cpg() — contains nodes, edges, codes
    cls_feats : (N, 768) CodeBERT CLS embeddings for each node, already on CPU
    label     : graph-level class label
    flaw_lines: 1-indexed known vulnerable line numbers (empty → all-zero mask)

    Returns
    -------
    Data with x=[N,773], edge_index=[2,E], edge_attr=[E,7], y=[1],
    node_line=[N], flaw_line_mask=[N]
    """
    nodes = cpg["nodes"]
    edges = cpg["edges"]
    codes = cpg["codes"]

    node_ids = [n["id"] for n in nodes]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    type_idxs    = torch.tensor([_node_type_idx(n) for n in nodes], dtype=torch.float).unsqueeze(1)
    dist_feats   = _compute_distance_features(nodes, edges)  # (N, 3)
    danger_flags = _dangerous_api_flags(codes)               # (N, 1)

    x = torch.cat([type_idxs, cls_feats, dist_feats, danger_flags], dim=1)  # (N, 773)

    node_lines = [_node_line(n) for n in nodes]
    node_line  = torch.tensor(node_lines, dtype=torch.long)
    flaw_set   = set(flaw_lines) if flaw_lines else set()
    flaw_line_mask = torch.tensor(
        [1 if ln >= 0 and ln in flaw_set else 0 for ln in node_lines],
        dtype=torch.long,
    )

    src_list, dst_list, edge_attr_list = [], [], []
    for e in edges:
        src, dst = e.get("src"), e.get("dst")
        if src in node_idx and dst in node_idx:
            src_list.append(node_idx[src])
            dst_list.append(node_idx[dst])
            edge_attr_list.append(_edge_attr(e))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, len(EDGE_TYPES)), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        node_line=node_line,
        flaw_line_mask=flaw_line_mask,
    )
    if func_input_ids is not None:
        data.func_input_ids = func_input_ids        # [1, 512] — batches to [B, 512]
        data.func_attention_mask = func_attention_mask  # [1, 512]
    return data


def build_lm_graph_from_json(
    path: str | Path,
    label: int,
    embedder: CodeBERTNodeEmbedder,
    max_nodes: int = 500,
    embed_batch_size: int = 256,
    flaw_lines: list[int] | None = None,
) -> Optional[Data]:
    """Convenience wrapper: parse + embed + build in one call (single-graph use)."""
    cpg = parse_cpg(path, max_nodes)
    if cpg is None:
        return None
    codes = cpg["codes"]
    cls_parts = [
        embedder.embed_batch(codes[i : i + embed_batch_size])
        for i in range(0, len(codes), embed_batch_size)
    ]
    cls_feats = torch.cat(cls_parts, dim=0)
    return build_from_parsed(cpg, cls_feats, label, flaw_lines)
