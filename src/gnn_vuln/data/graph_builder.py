"""
graph_builder.py
~~~~~~~~~~~~~~~~
Converts Joern-exported Code Property Graphs (CPG) into
PyTorch Geometric `Data` objects ready for GNN training.

Joern outputs CPGs in JSON or GraphML format. This module supports
both. The resulting `Data` object contains:

  data.x          — node feature matrix  [N, node_feat_dim]
  data.edge_index — COO edge list        [2, E]
  data.edge_attr  — one-hot edge types   [E, num_edge_types]
  data.y          — graph-level label    scalar (0=benign, 1=vulnerable)
  data.num_nodes  — N

Usage
-----
    from gnn_vuln.data.graph_builder import build_graph_from_json
    data = build_graph_from_json("path/to/cpg.json", label=1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Edge types present in a CPG
EDGE_TYPES = ["AST", "CFG", "CDG", "DDG", "PDG", "CALL", "REACHING_DEF"]
EDGE_TYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPES)}


# ---------------------------------------------------------------------------
# Node feature extraction
# ---------------------------------------------------------------------------

# Simple keyword vocabulary for one-hot node features (expandable)
C_KEYWORDS = [
    "if", "else", "while", "for", "do", "return", "break", "continue",
    "switch", "case", "goto", "sizeof", "typedef", "struct", "union",
    "enum", "void", "int", "long", "short", "char", "float", "double",
    "unsigned", "signed", "const", "static", "extern", "register",
    "volatile", "auto", "NULL", "malloc", "free", "memcpy", "strcpy",
]
KEYWORD_TO_IDX = {kw: i for i, kw in enumerate(C_KEYWORDS)}
NODE_FEAT_DIM = len(C_KEYWORDS) + 4  # keywords + [has_literal, call_depth, lineno_norm, out_degree_norm]


def _get_attr(attrs: dict, *keys: str, default: str = "") -> str:
    """Attribute lookup supporting multiple possible key names.

    Handles:
      - Joern JSON (flat): 'code', 'label'
      - Joern GraphML v4:  'CODE', 'labelV' (node type), 'labelE' (edge type)
    """
    for k in keys:
        if k in attrs:
            return str(attrs[k])
        if k.upper() in attrs:
            return str(attrs[k.upper()])
        if k.lower() in attrs:
            return str(attrs[k.lower()])
    return default


def _node_features(node_attrs: dict, graph: nx.DiGraph) -> list[float]:
    """
    Extract a fixed-length feature vector from a CPG node's attributes.

    Handles:
      - Flat JSON format (lowercase):  code, label, lineNumber
      - Joern GraphML v4 (uppercase):  CODE, labelV, LINE_NUMBER
    """
    code = _get_attr(node_attrs, "code", "CODE")
    # Joern v4 GraphML uses 'labelV' for node type; flat JSON uses 'label'
    label = _get_attr(node_attrs, "labelV", "label", "LABEL")

    # Keyword one-hot
    kw_feat = [0.0] * len(C_KEYWORDS)
    for token in code.split():
        if token in KEYWORD_TO_IDX:
            kw_feat[KEYWORD_TO_IDX[token]] = 1.0

    # Structural features
    has_literal = float(any(c.isdigit() for c in code))
    lineno = float(_get_attr(node_attrs, "lineNumber", "LINE_NUMBER") or 0) / 1000.0
    is_call = float("CALL" in label)

    return kw_feat + [has_literal, is_call, lineno, 0.0]  # last: out_degree (filled later)


def _edge_attr(edge_attrs: dict) -> list[float]:
    """One-hot vector for edge type. Handles 'labelE' (Joern v4 GraphML) and 'label' (JSON)."""
    etype = _get_attr(edge_attrs, "labelE", "label", "EDGE_TYPE", default="AST")
    idx = EDGE_TYPE_TO_IDX.get(etype, 0)
    one_hot = [0.0] * len(EDGE_TYPES)
    one_hot[idx] = 1.0
    return one_hot


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_graph_from_networkx(
    g: nx.DiGraph,
    label: int,
    max_nodes: int = 500,
) -> Optional[Data]:
    """Convert a NetworkX DiGraph (CPG) to a PyG Data object."""
    if g.number_of_nodes() == 0 or g.number_of_nodes() > max_nodes:
        return None

    nodes = list(g.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Node features
    x = []
    for n in nodes:
        feats = _node_features(g.nodes[n], g)
        # Fill normalised out-degree
        feats[-1] = g.out_degree(n) / max(g.number_of_nodes(), 1)
        x.append(feats)

    x_tensor = torch.tensor(x, dtype=torch.float)

    # Edges
    src_list, dst_list, edge_attrs = [], [], []
    for u, v, attrs in g.edges(data=True):
        src_list.append(node_idx[u])
        dst_list.append(node_idx[v])
        edge_attrs.append(_edge_attr(attrs))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros((0, len(EDGE_TYPES)))

    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_graph_from_json(path: str | Path, label: int, max_nodes: int = 500) -> Optional[Data]:
    """
    Load a Joern-exported CPG JSON file and convert it to a PyG Data object.

    Expected JSON format (simplified Joern export):
    {
        "nodes": [{"id": "0", "label": "METHOD", "code": "int foo()", ...}, ...],
        "edges": [{"src": "0", "dst": "1", "label": "AST"}, ...]
    }
    """
    with open(path) as f:
        cpg = json.load(f)

    g = nx.DiGraph()
    for node in cpg.get("nodes", []):
        g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in cpg.get("edges", []):
        g.add_edge(edge["src"], edge["dst"], label=edge.get("label", "AST"))

    return build_graph_from_networkx(g, label=label, max_nodes=max_nodes)


def build_graph_from_graphml(path: str | Path, label: int, max_nodes: int = 500) -> Optional[Data]:
    """Load a Joern-exported CPG GraphML file."""
    g = nx.read_graphml(str(path))
    if not isinstance(g, nx.DiGraph):
        g = g.to_directed()
    return build_graph_from_networkx(g, label=label, max_nodes=max_nodes)
