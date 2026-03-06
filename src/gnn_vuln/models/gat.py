"""
gat.py — Graph Attention Network for vulnerability detection.

Architecture
------------
  Input node features  →  GATConv × num_layers (multi-head)  →  Global mean pool  →  MLP head  →  2-class softmax

The attention mechanism allows the model to learn which neighbouring nodes
(e.g., AST children, CFG successors) are most relevant per graph.

References
----------
  Veličković et al. (2018): https://arxiv.org/abs/1710.10903
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATVulnDetector(nn.Module):
    """
    Graph Attention Network for graph-level binary classification.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_dim : int
        Hidden layer width (per head).
    num_layers : int
        Number of GAT message-passing layers.
    heads : int
        Number of attention heads (except in the last layer, which uses 1).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Last GAT layer — single head, no concat
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            out_dim = hidden_dim
        else:
            out_dim = hidden_dim * heads

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.classifier(x)
