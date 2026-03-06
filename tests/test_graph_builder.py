"""Tests for graph building utilities."""

import pytest
import torch
from gnn_vuln.data.graph_builder import build_graph_from_networkx
import networkx as nx


def _make_simple_cpg(n_nodes: int = 5) -> nx.DiGraph:
    """Build a tiny synthetic CPG-like graph for testing."""
    g = nx.DiGraph()
    for i in range(n_nodes):
        g.add_node(i, label="IDENTIFIER", code=f"var_{i}", lineNumber=i + 1)
    for i in range(n_nodes - 1):
        g.add_edge(i, i + 1, label="AST")
    return g


def test_build_graph_returns_data():
    g = _make_simple_cpg(5)
    data = build_graph_from_networkx(g, label=1)
    assert data is not None
    assert data.x.shape[0] == 5          # 5 nodes
    assert data.edge_index.shape[1] == 4  # 4 edges
    assert data.y.item() == 1


def test_build_graph_label_zero():
    g = _make_simple_cpg(3)
    data = build_graph_from_networkx(g, label=0)
    assert data.y.item() == 0


def test_build_graph_exceeds_max_nodes():
    g = _make_simple_cpg(600)
    data = build_graph_from_networkx(g, label=0, max_nodes=500)
    assert data is None  # should be filtered out


def test_node_feature_dim():
    g = _make_simple_cpg(4)
    data = build_graph_from_networkx(g, label=0)
    # Node feature dim = len(C_KEYWORDS) + 4
    from gnn_vuln.data.graph_builder import NODE_FEAT_DIM
    assert data.x.shape[1] == NODE_FEAT_DIM


def test_edge_attr_shape():
    g = _make_simple_cpg(4)
    data = build_graph_from_networkx(g, label=0)
    from gnn_vuln.data.graph_builder import EDGE_TYPES
    assert data.edge_attr.shape == (3, len(EDGE_TYPES))
