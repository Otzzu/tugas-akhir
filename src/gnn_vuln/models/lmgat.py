"""
lmgat.py — CodeBERT node embeddings + GATv2 with edge features and dual output heads.

Architecture
------------
  Pre-computed node features (773D)
      → GATv2Conv × num_layers          (shared encoder, edge-aware attention)
      → BatchNorm + ReLU + Dropout
      ↓
      ├─ FUNCTION HEAD
      │    global_mean_pool → MLP → logit_func  [B, num_classes]
      │
      └─ STATEMENT HEAD (WAVES-style MIL)
           Group nodes by source line within each graph
           max-pool + mean-pool per line → dual scorers → stmt_score

Vs LM-GCN:
  - GCNConv replaced with GATv2Conv (Brody et al., ICLR 2022)
  - GATv2 fixes static attention in GAT v1 — attention is truly dynamic
    (pair-specific, not just source-node-specific)
  - edge_attr (7D one-hot: AST/CFG/CDG/DDG/PDG/CALL/REACHING_DEF) is injected
    into every attention score computation via edge_dim — edge type now shapes
    which CPG neighbours each node attends to

Node feature layout expected (773D):
  [node_type_idx (1)] + [CodeBERT CLS (768)] + [dist_features (3)] + [danger_api (1)]
Edge feature layout expected (7D):
  one-hot over {AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF}
Both built by gnn_vuln.data.graph_builder_lm / CodeBERTGraphDataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

NODE_FEAT_DIM = 773   # 1 (node_type) + 768 (CodeBERT CLS) + 3 (dist) + 1 (danger API)
EDGE_FEAT_DIM = 7     # one-hot: AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF

_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATVulnDetector(nn.Module):
    """
    GATv2 vulnerability detector with CodeBERT node embeddings, edge-aware
    attention, and two output heads:
      1. Function-level: multi-class classification (binary or per-CWE).
      2. Statement-level: per-line vulnerability score for MIL-based localisation.

    Parameters
    ----------
    in_channels : int
        Node feature dimension (773 for CodeBERT-embedded CPGs).
    hidden_dim : int
        Width of every GATv2 hidden layer and the statement head.
    num_layers : int
        Number of GATv2Conv message-passing steps.
    dropout : float
        Dropout probability after each layer (also used as attention dropout).
    num_classes : int
        Function-head output size (2 for binary detection).
    num_heads : int
        Number of attention heads per layer. Output is averaged (concat=False)
        so hidden_dim is preserved across layers.
    edge_dim : int
        Dimension of edge feature vectors (7 for CPG one-hot edge types).
        Set to None to disable edge feature injection.
    """

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
        num_heads: int = 4,
        edge_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()
        self.dropout = dropout

        # ── Shared GATv2 encoder ────────────────────────────────────────────
        # concat=False: each head outputs hidden_dim, then averaged →
        # layer output stays at hidden_dim regardless of num_heads.
        # edge_dim injects edge_attr into the attention linear transform so
        # AST / CFG / PDG edges produce different attention weights.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATv2Conv(
                in_channels, hidden_dim,
                heads=num_heads, concat=False, dropout=dropout,
                edge_dim=edge_dim,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, hidden_dim,
                    heads=num_heads, concat=False, dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

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

    # ── Shared encoder ───────────────────────────────────────────────────────

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [N, hidden_dim]

    # ── Statement-level head ─────────────────────────────────────────────────

    def _statement_scores(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Produce per-statement raw logits by grouping CPG nodes by source line.

        Parameters
        ----------
        h         : [N_total, hidden_dim]  GATv2 node embeddings
        batch     : [N_total]              graph index per node
        node_line : [N_total]              source line number per node (-1 = unknown)

        Returns
        -------
        List of length B. Each element is a 1-D float tensor [n_stmts_i].
        Empty tensor if no valid line numbers exist.
        """
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
                h_line = h_b[node_mask]  # [k_nodes, hidden]

                h_max = h_line.max(dim=0).values
                h_mean = h_line.mean(dim=0)

                s_max = self.stmt_max_head(h_max).squeeze(-1)
                s_mean = self.stmt_mean_head(h_mean).squeeze(-1)

                scores.append(_ALPHA_MAX * s_max + _ALPHA_MEAN * s_mean)

            result.append(torch.stack(scores))

        return result

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Parameters
        ----------
        x          : [N, in_channels]  pre-computed node features
        edge_index : [2, E]            COO edge list
        batch      : [N]               maps each node to its graph index
        node_line  : [N] or None       source line number per node
        edge_attr  : [E, edge_dim] or None  one-hot edge type features;
                                       if None the attention is node-only

        Returns
        -------
        logit_func  : [B, num_classes]
        stmt_scores : list of [n_stmts_i] | None
        """
        h = self._encode(x, edge_index, edge_attr)

        h_graph = global_mean_pool(h, batch)
        logit_func = self.func_head(h_graph)

        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None
            else None
        )

        return logit_func, stmt_scores
