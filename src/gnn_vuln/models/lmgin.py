"""
lmgin.py — CodeBERT node embeddings + GIN (Graph Isomorphism Network).

Architecture
------------
  Pre-computed node features (773D)
      → GINEConv × num_layers   (edge-aware, sum aggregation)
      → BatchNorm + ReLU + Dropout
      ↓
      ├─ FUNCTION HEAD
      │    global_mean_pool → MLP → logit_func  [B, num_classes]
      │
      └─ STATEMENT HEAD (WAVES-style MIL)
           Group nodes by source line within each graph
           max-pool + mean-pool per line → dual scorers → stmt_score

GIN vs GCN/GAT:
  - GINConv: h = MLP((1+ε)*h_v + Σ h_u)  — sum aggregation, most expressive
  - GATv2:   h = Σ α_uv * W*h_u           — weighted mean, attention-based
  - Theoretically GIN ≥ GCN in distinguishing graph structures (Xu et al. 2019)
  - GINEConv extends GIN to edge features: h_u + e_uv before aggregation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

NODE_FEAT_DIM = 773
EDGE_FEAT_DIM = 7

_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


def _gin_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
    )


class LMGINVulnDetector(nn.Module):
    """
    GIN vulnerability detector with CodeBERT node embeddings, edge-aware
    aggregation (GINEConv), and two output heads:
      1. Function-level: multi-class classification (binary or per-CWE).
      2. Statement-level: per-line vulnerability score for MIL localisation.

    Parameters
    ----------
    in_channels : int
        Node feature dimension (773 for CodeBERT-embedded CPGs).
    hidden_dim : int
        Hidden dimension for GIN MLPs and statement head.
    num_layers : int
        Number of GINEConv message-passing steps.
    dropout : float
        Dropout probability.
    num_classes : int
        Function-head output size.
    edge_dim : int
        Edge feature dimension (7 for CPG one-hot edge types).
    """

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
        edge_dim: int = EDGE_FEAT_DIM,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        # Edge feature projection: GINEConv expects edge_attr dim == node dim
        self.edge_proj = nn.Linear(edge_dim, in_channels)

        # ── Shared GIN encoder ──────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GINEConv(_gin_mlp(in_channels, hidden_dim, dropout), edge_dim=in_channels))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(_gin_mlp(hidden_dim, hidden_dim, dropout), edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Edge proj for subsequent layers (hidden_dim → hidden_dim)
        self.edge_proj_hidden = nn.Linear(edge_dim, hidden_dim)

        if use_skip:
            self.res_projs = nn.ModuleList()
            self.res_projs.append(nn.Linear(in_channels, hidden_dim, bias=False))
            for _ in range(num_layers - 1):
                self.res_projs.append(nn.Identity())

        # ── Head 1: function-level classifier ───────────────────────────────
        self.func_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ── Head 2: statement-level scorers (WAVES dual-channel) ────────────
        self.stmt_max_head = nn.Linear(hidden_dim, 1)
        self.stmt_mean_head = nn.Linear(hidden_dim, 1)

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            if edge_attr is not None:
                e = self.edge_proj(edge_attr) if i == 0 else self.edge_proj_hidden(edge_attr)
            else:
                e = None
            x = conv(x, edge_index, edge_attr=e)
            x = bn(x)
            if residual is not None:
                x = F.relu(x + residual)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def _statement_scores(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
    ) -> list[torch.Tensor]:
        device = h.device
        batch_size = int(batch.max().item()) + 1
        result: list[torch.Tensor] = []

        for b in range(batch_size):
            mask = batch == b
            h_b = h[mask]
            lines_b = node_line[mask]

            valid = lines_b >= 0
            if not valid.any():
                result.append(torch.zeros(0, device=device))
                continue

            h_b = h_b[valid]
            lines_b = lines_b[valid]

            unique_lines = lines_b.unique(sorted=True)
            scores: list[torch.Tensor] = []

            for line in unique_lines:
                node_mask = lines_b == line
                h_line = h_b[node_mask]
                h_max = h_line.max(dim=0).values
                h_mean = h_line.mean(dim=0)
                s_max = self.stmt_max_head(h_max).squeeze(-1)
                s_mean = self.stmt_mean_head(h_mean).squeeze(-1)
                scores.append(_ALPHA_MAX * s_max + _ALPHA_MEAN * s_mean)

            result.append(torch.stack(scores))

        return result

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        h = self._encode(x, edge_index, edge_attr)

        h_graph = global_mean_pool(h, batch)
        logit_func = self.func_head(h_graph)

        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None
            else None
        )

        return logit_func, stmt_scores
