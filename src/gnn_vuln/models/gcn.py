"""
gcn.py — Graph Convolutional Network baseline for vulnerability detection.

Architecture
------------
  Input node features  →  GCNConv × num_layers  →  Global mean pool  →  MLP head  →  2-class softmax

References
----------
  Kipf & Welling (2017): https://arxiv.org/abs/1609.02907
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNVulnDetector(nn.Module):
    """
    Simple GCN for graph-level binary classification (vulnerable / benign).

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Number of GCN message-passing layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : Node features [N, in_channels]
        edge_index  : COO edges     [2, E]
        batch       : Batch vector  [N] (maps each node to its graph index)

        Returns
        -------
        logits : [B, 2]  (raw unnormalised scores)
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Aggregate node embeddings → graph embedding
        x = global_mean_pool(x, batch)  # [B, hidden_dim]
        return self.classifier(x)       # [B, 2]
