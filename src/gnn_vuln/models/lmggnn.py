"""
lmggnn.py — Architecture 9: LM-GAT-CodeBERT with GATv2 → GatedGraphConv

Arch3 (lmgat_codebert) with GATv2Conv replaced by GatedGraphConv (GGNN).
GatedGraphConv requires in_channels == out_channels, so node features are
projected to hidden_dim via input_proj before GGNN.

Fusion: concat(pool(h_gnn), lm_cls) → single func_head (same as Arch3).
Statement head: binary per-line scorer (same as Arch3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from transformers import AutoModel

NODE_FEAT_DIM = 773
_LM_DIM = 768
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGNNVulnDetector(nn.Module):
    """
    GatedGraphConv vulnerability detector with pre-computed node embeddings
    and a live fine-tuned LM branch for full-function context.

    Arch3 (lmgat_codebert) equivalent with GATv2Conv replaced by
    GatedGraphConv. input_proj maps 773D → hidden_dim before GGNN since
    GatedGraphConv requires in_channels == out_channels.

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace model for frozen node embeddings (preprocessing only).
    func_lm : str
        HuggingFace model for live LM branch. Falls back to pretrained_lm.
    in_channels : int
        Node feature dimension (773D).
    hidden_dim : int
        GatedGraphConv output dimension.
    num_layers : int
        GatedGraphConv steps.
    dropout : float
        Dropout probability.
    num_classes : int
        Output classes.
    """

    def __init__(
        self,
        pretrained_lm: str = "microsoft/codebert-base",
        func_lm: str = "",
        in_channels: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.3,
        num_classes: int = 11,
        **kwargs,  # absorb unused config keys (e.g. alpha from old config)
    ):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes

        # Project node features to hidden_dim (GatedGraphConv requires in==out)
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GatedGraphConv backbone
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

        # Live LM branch (fine-tuned)
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm)

        # Function head: concat(GNN pooled, LM CLS) → num_classes
        self.func_head = nn.Sequential(
            nn.Linear(hidden_dim + _LM_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Statement head: binary per-line suspicious score
        self.stmt_max_head = nn.Linear(hidden_dim, 1)
        self.stmt_mean_head = nn.Linear(hidden_dim, 1)

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
                s = (
                    _ALPHA_MAX  * self.stmt_max_head(h_max).squeeze(-1)
                    + _ALPHA_MEAN * self.stmt_mean_head(h_mean).squeeze(-1)
                )
                scores.append(s)
            result.append(torch.stack(scores))

        return result

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,  # unused: GGNN ignores edge_attr
        func_input_ids: torch.Tensor | None = None,
        func_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        B = int(batch.max().item()) + 1

        # ── GNN branch ───────────────────────────────────────────────────────
        h = self.ggnn(self.input_proj(x), edge_index)    # [N, hidden_dim]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_graph = global_mean_pool(h, batch)              # [B, hidden_dim]

        # ── LM branch ────────────────────────────────────────────────────────
        if func_input_ids is not None:
            lm_out = self.codebert(
                input_ids=func_input_ids,
                attention_mask=func_attention_mask,
            )
            cls = lm_out.last_hidden_state[:, 0, :]      # CLS [B, 768]
        else:
            cls = torch.zeros(B, _LM_DIM, device=x.device)

        # ── Function head ─────────────────────────────────────────────────────
        logit_func = self.func_head(torch.cat([h_graph, cls], dim=-1))

        # ── Statement scores for MIL loss ────────────────────────────────────
        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None else None
        )

        return logit_func, stmt_scores
