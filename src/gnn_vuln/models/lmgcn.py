"""
lmgcn.py — CodeBERT node embeddings + GCN with dual output heads.

Architecture
------------
  Pre-computed CodeBERT node features (769D)
      → GCNConv × num_layers          (shared encoder)
      → BatchNorm + ReLU + Dropout
      ↓
      ├─ FUNCTION HEAD (existing)
      │    global_mean_pool → MLP → logit_func  [B, num_classes]
      │
      └─ STATEMENT HEAD (WAVES-style MIL, new)
           Group nodes by source line number within each graph
           max-pool(GCN emb per line) → Linear → score_max
           mean-pool(GCN emb per line) → Linear → score_mean
           stmt_score = 0.8 * score_max + 0.6 * score_mean
           → list of [n_stmts_i] raw logit tensors (one per graph)

Relationship to VulLMGNN (Cao et al. 2022):
  - Keeps  : CodeBERT per-node embeddings as input features
  - Changes : GatedGraphConv → GCNConv
  - Removes : Text branch + k-fusion
  - Adds    : Statement-level MIL head (WAVES, Ni et al. 2023)

Relationship to WAVES (Ni et al. 2023):
  - Replaces: Transformer + token→statement indicative matrix
  - With    : GCN encoder + CPG node→line grouping
  - Benefit : Statement embeddings are structure-aware (GCN captures AST/CFG/PDG context)

Node feature layout expected (773D):
  [node_type_idx (1)] + [CodeBERT CLS (768)] + [dist_features (3)] + [danger_api (1)]
Built by gnn_vuln.data.graph_builder_lm / CodeBERTGraphDataset.
Edge features (edge_attr, 7D one-hot) are present in the dataset but not used by GCNConv.

NOTE: If you add node_line to the dataset for the first time, delete the
processed cache (data/processed/lm_dataset.pt) and rerun to rebuild it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

NODE_FEAT_DIM = 773  # 1 (node_type) + 768 (CodeBERT CLS) + 3 (dist) + 1 (danger API)
CODEBERT_NODE_FEAT_DIM = NODE_FEAT_DIM  # backwards-compat alias

# Fusion weights for max and mean statement channels (from WAVES paper)
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGCNVulnDetector(nn.Module):
    """
    GCN vulnerability detector with CodeBERT node embeddings and two output heads:
      1. Function-level: binary vulnerable/benign classification.
      2. Statement-level: per-line vulnerability score for MIL-based localisation.

    Parameters
    ----------
    in_channels : int
        Node feature dimension (773 for CodeBERT-embedded CPGs).
    hidden_dim : int
        Width of every GCN hidden layer and the statement head.
    num_layers : int
        Number of GCNConv message-passing steps.
    dropout : float
        Dropout probability after each GCN layer.
    num_classes : int
        Function-head output size (2 for binary detection).
    """

    def __init__(
        self,
        in_channels: int = CODEBERT_NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout

        # ── Shared GCN encoder ──────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Head 1: function-level classifier ───────────────────────────────
        self.func_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ── Head 2: statement-level scorers (WAVES dual-channel) ────────────
        # Each head maps a single hidden-dim vector → scalar logit.
        self.stmt_max_head = nn.Linear(hidden_dim, 1)
        self.stmt_mean_head = nn.Linear(hidden_dim, 1)

    # ── Shared encoder ───────────────────────────────────────────────────────

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,  # ignored by GCNConv, kept for uniform interface
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
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

        Replaces WAVES' token→statement indicative matrix with
        CPG node→line grouping. GCN embeddings are already structure-aware
        (via message passing over AST/CFG/PDG edges), so each statement
        embedding captures both local token semantics and structural context.

        Parameters
        ----------
        h         : [N_total, hidden_dim]  GCN node embeddings (full batch)
        batch     : [N_total]              graph index per node
        node_line : [N_total]              source line number per node (-1 = unknown)

        Returns
        -------
        List of length B. Each element is a 1-D float tensor [n_stmts_i]
        containing raw logits (before sigmoid) for every unique source line
        in that graph. Empty tensor if no valid line numbers exist.
        """
        device = h.device
        batch_size = int(batch.max().item()) + 1
        result: list[torch.Tensor] = []

        for b in range(batch_size):
            mask = batch == b
            h_b = h[mask]              # [N_b, hidden]
            lines_b = node_line[mask]  # [N_b]

            # Keep only nodes with known line numbers
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

                h_max = h_line.max(dim=0).values   # [hidden]
                h_mean = h_line.mean(dim=0)         # [hidden]

                s_max = self.stmt_max_head(h_max).squeeze(-1)   # scalar
                s_mean = self.stmt_mean_head(h_mean).squeeze(-1) # scalar

                scores.append(_ALPHA_MAX * s_max + _ALPHA_MEAN * s_mean)

            result.append(torch.stack(scores))  # [n_unique_lines]

        return result

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,  # ignored by GCNConv
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Parameters
        ----------
        x          : [N, in_channels]  pre-computed node features
        edge_index : [2, E]            COO edge list
        batch      : [N]               maps each node to its graph index
        node_line  : [N] or None       source line number per node
        edge_attr  : [E, D] or None    edge features (unused by GCNConv)

        Returns
        -------
        logit_func  : [B, num_classes]
        stmt_scores : list of [n_stmts_i] | None
        """
        h = self._encode(x, edge_index)  # [N, hidden_dim]

        # Head 1 — function level
        h_graph = global_mean_pool(h, batch)  # [B, hidden_dim]
        logit_func = self.func_head(h_graph)   # [B, num_classes]

        # Head 2 — statement level (only when line numbers are available)
        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None
            else None
        )

        return logit_func, stmt_scores
