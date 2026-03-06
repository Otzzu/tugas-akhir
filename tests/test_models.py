"""Tests for GNN model architectures."""

import pytest
import torch
from torch_geometric.data import Data, Batch
from gnn_vuln.models.gcn import GCNVulnDetector
from gnn_vuln.models.gat import GATVulnDetector
from gnn_vuln.data.graph_builder import NODE_FEAT_DIM


def _make_batch(n_graphs: int = 4, n_nodes: int = 6, in_channels: int = NODE_FEAT_DIM):
    """Create a synthetic PyG Batch for testing."""
    data_list = []
    for i in range(n_graphs):
        x = torch.randn(n_nodes, in_channels)
        # Simple chain graph
        src = torch.arange(0, n_nodes - 1)
        dst = torch.arange(1, n_nodes)
        edge_index = torch.stack([src, dst], dim=0)
        y = torch.tensor([i % 2], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return Batch.from_data_list(data_list)


def test_gcn_forward():
    batch = _make_batch()
    model = GCNVulnDetector(in_channels=NODE_FEAT_DIM, hidden_dim=64, num_layers=2)
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (4, 2)


def test_gat_forward():
    batch = _make_batch()
    model = GATVulnDetector(in_channels=NODE_FEAT_DIM, hidden_dim=16, num_layers=2, heads=2)
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (4, 2)


def test_gcn_output_is_finite():
    batch = _make_batch()
    model = GCNVulnDetector(in_channels=NODE_FEAT_DIM)
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert torch.isfinite(logits).all()


def test_model_train_eval_mode():
    model = GCNVulnDetector(in_channels=NODE_FEAT_DIM)
    model.train()
    assert model.training
    model.eval()
    assert not model.training
