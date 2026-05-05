"""
lmgat_interp.py — Architecture 5: VulLMGNN-style explicit interpolation.

Correct implementation of VulLMGNN explicit stage (Cao et al. 2022):
    pred = λ * logit_gnn + (1 - λ) * logit_lm

Two independent branches:
    GNN branch  : GATv2 on CPG → global_mean_pool → MLP → logit_gnn [B, C]
    LM  branch  : CodeBERT CLS (live, full function text) → MLP → logit_lm [B, C]
    Final       : λ * logit_gnn + (1-λ) * logit_lm  (λ learned, sigmoid-bounded)

Statement localisation: uses GNN branch node embeddings only (structure-aware).

Contrast with Arch 3 (lmgat_codebert):
    Arch 3 concatenates GNN pooled + CodeBERT CLS → single func_head.
    Arch 5 keeps branches independent → interpolates final logits.
    Arch 5 matches the VulLMGNN paper design more faithfully.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoModel

from gnn_vuln.models._lm_utils import lm_hidden_dim, lm_pool

NODE_FEAT_DIM = 773
EDGE_FEAT_DIM = 7
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATInterpVulnDetector(nn.Module):
    """
    VulLMGNN-style explicit interpolation: GATv2 GNN branch + live CodeBERT
    LM branch, combined via learned λ interpolation.

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace model name for the live CodeBERT branch.
    in_channels : int
        Node feature dimension (773D for pre-computed CPG graphs).
    hidden_dim : int
        GATv2 hidden width and statement head width.
    num_layers : int
        Number of GATv2Conv message-passing steps.
    dropout : float
        Dropout probability.
    num_classes : int
        Number of output classes.
    num_heads : int
        GATv2 attention heads.
    edge_dim : int
        Edge feature dimension.
    init_lambda : float
        Initial value for λ (GNN weight). Sigmoid-bounded → (0, 1).
    """

    def __init__(
        self,
        pretrained_lm: str = "microsoft/codebert-base",
        func_lm: str = "",
        in_channels: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
        num_heads: int = 4,
        edge_dim: int = EDGE_FEAT_DIM,
        init_lambda: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        # ── Live LM branch ──────────────────────────────────────────────────
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm)
        self._lm_dim = lm_hidden_dim(self.codebert)
        self._is_enc_dec = getattr(self.codebert.config, "is_encoder_decoder", False)
        self.lm_head = nn.Sequential(
            nn.Linear(self._lm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ── GATv2 GNN branch ────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATv2Conv(in_channels, hidden_dim, heads=num_heads, concat=False,
                      dropout=dropout, edge_dim=edge_dim)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                          dropout=dropout, edge_dim=edge_dim)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.gnn_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ── Statement-level scorers (GNN branch only) ───────────────────────
        self.stmt_max_head = nn.Linear(hidden_dim, 1)
        self.stmt_mean_head = nn.Linear(hidden_dim, 1)

        # ── Learned interpolation weight λ ──────────────────────────────────
        # λ = sigmoid(lambda_logit) → bounded in (0, 1)
        # logit 0 → λ=0.5 (equal weight); positive → GNN dominates
        import math
        init_logit = math.log(init_lambda / (1.0 - init_lambda))
        self.lambda_logit = nn.Parameter(torch.tensor(init_logit))

    # ── GNN encoder ──────────────────────────────────────────────────────────

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    # ── Statement-level head (GNN branch) ────────────────────────────────────

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

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        func_input_ids: torch.Tensor | None = None,
        func_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Returns
        -------
        logit_func  : [B, num_classes]  interpolated prediction
        stmt_scores : list of [n_stmts_i] | None
        """
        # GNN branch
        h = self._encode(x, edge_index, edge_attr)
        h_graph = global_mean_pool(h, batch)
        logit_gnn = self.gnn_head(h_graph)

        # LM branch (requires func_input_ids)
        if func_input_ids is not None:
            ids = func_input_ids.squeeze(1) if func_input_ids.dim() == 3 else func_input_ids
            mask = func_attention_mask.squeeze(1) if func_attention_mask is not None and func_attention_mask.dim() == 3 else func_attention_mask
            cls = lm_pool(self.codebert, self._is_enc_dec, ids, mask)
            logit_lm = self.lm_head(cls)              # [B, num_classes]
        else:
            logit_lm = torch.zeros_like(logit_gnn)

        # Interpolate: λ*GNN + (1-λ)*LM
        lam = torch.sigmoid(self.lambda_logit)
        logit_func = lam * logit_gnn + (1.0 - lam) * logit_lm

        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None
            else None
        )

        return logit_func, stmt_scores
