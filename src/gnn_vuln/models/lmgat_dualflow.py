"""
lmgat_dualflow.py — Architecture 10: LM-GAT-DualFlow

Stage 1: GATv2 binary localization → per-node suspicion s_i  (Arch7 v1)
Stage 2: GATv2(concat[x, s_i]) → h_cls, then TWO parallel poolings:
    focal_emb   = suspicion-weighted pool(h_cls)   — highlights the flaw
    context_emb = global_mean_pool(h_cls)          — preserves surrounding logic
Stage 3: concat(focal_emb, context_emb, lm_emb) → MLP → CWE logits

Motivation: Arch7 v1 drops F1 because suspicion-weighted pool discards
safe-context nodes that carry CWE-discriminating structural information.
Adding context_emb restores that information without disturbing localization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool
from transformers import AutoModel

from gnn_vuln.models._lm_utils import lm_hidden_dim, lm_pool

NODE_FEAT_DIM = 773
EDGE_FEAT_DIM = 7
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATDualFlowVulnDetector(nn.Module):
    """
    Dual-flow sequential detector.

    Stage 1 (binary localization):
        frozen node features → GATv2 × num_layers → h_loc
        binary stmt head → s_i per node

    Stage 2 (dual-flow encoding):
        concat(x, s_i) → GATv2 × num_layers → h_cls
        focal_emb   = weighted_mean_pool(h_cls, s_i)   [B, hidden_dim]
        context_emb = global_mean_pool(h_cls)           [B, hidden_dim]

    Stage 3 (tri-modal fusion):
        live LM → CLS → lm_emb                         [B, 768]
        concat(focal_emb, context_emb, lm_emb)          [B, hidden_dim*2 + 768]
        → func_head MLP → logit_func                    [B, num_classes]
    """

    def __init__(
        self,
        pretrained_lm: str = "microsoft/unixcoder-base",
        func_lm: str = "",
        in_channels: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 11,
        num_heads: int = 4,
        edge_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes

        # ── Stage 1: Localization GNN (input: 773D) ──────────────────────────
        self.loc_convs = nn.ModuleList()
        self.loc_bns = nn.ModuleList()
        self.loc_convs.append(
            GATv2Conv(in_channels, hidden_dim, heads=num_heads, concat=False,
                      dropout=dropout, edge_dim=edge_dim)
        )
        self.loc_bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.loc_convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                          dropout=dropout, edge_dim=edge_dim)
            )
            self.loc_bns.append(nn.BatchNorm1d(hidden_dim))

        # Stage 1 binary stmt head
        self.loc_stmt_max  = nn.Linear(hidden_dim, 1)
        self.loc_stmt_mean = nn.Linear(hidden_dim, 1)

        # ── Stage 2: Classification GNN (input: 773D + 1D s_i = 774D) ────────
        self.cls_convs = nn.ModuleList()
        self.cls_bns = nn.ModuleList()
        self.cls_convs.append(
            GATv2Conv(in_channels + 1, hidden_dim, heads=num_heads, concat=False,
                      dropout=dropout, edge_dim=edge_dim)
        )
        self.cls_bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.cls_convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                          dropout=dropout, edge_dim=edge_dim)
            )
            self.cls_bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Stage 3: Live LM branch ───────────────────────────────────────────
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm, trust_remote_code=True)
        self._lm_dim = lm_hidden_dim(self.codebert)
        self._is_enc_dec = getattr(self.codebert.config, "is_encoder_decoder", False)

        # ── Stage 3: Function head (focal + context + lm) ────────────────────
        fusion_dim = hidden_dim * 2 + self._lm_dim
        self.func_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    # ── Shared GNN encoder ───────────────────────────────────────────────────

    def _encode(self, x, edge_index, edge_attr, convs, bns):
        for conv, bn in zip(convs, bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    # ── Stage 1: per-node suspicion ──────────────────────────────────────────

    def _node_suspicion(self, h_loc: torch.Tensor) -> torch.Tensor:
        raw = _ALPHA_MAX * self.loc_stmt_max(h_loc) + _ALPHA_MEAN * self.loc_stmt_mean(h_loc)
        return torch.sigmoid(raw).squeeze(-1)  # [N]

    # ── Stage 1: per-line statement scores for MIL loss ─────────────────────

    def _statement_scores(
        self,
        h_loc: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
    ) -> list[torch.Tensor]:
        device = h_loc.device
        batch_size = int(batch.max().item()) + 1
        result: list[torch.Tensor] = []

        for b in range(batch_size):
            mask = batch == b
            h_b = h_loc[mask]
            lines_b = node_line[mask]
            valid = lines_b >= 0
            if not valid.any():
                result.append(torch.zeros(0, device=device))
                continue
            h_b, lines_b = h_b[valid], lines_b[valid]
            unique_lines = lines_b.unique(sorted=True)
            scores: list[torch.Tensor] = []
            for line in unique_lines:
                nm = lines_b == line
                h_line = h_b[nm]
                s = (
                    _ALPHA_MAX  * self.loc_stmt_max(h_line.max(0).values)
                    + _ALPHA_MEAN * self.loc_stmt_mean(h_line.mean(0))
                )
                scores.append(s.squeeze(-1))
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
        B = int(batch.max().item()) + 1

        # ── Stage 1: Localization ─────────────────────────────────────────────
        h_loc = self._encode(x, edge_index, edge_attr, self.loc_convs, self.loc_bns)
        s_i = self._node_suspicion(h_loc)  # [N]

        # ── Stage 2: Classification GNN on x_aug ─────────────────────────────
        x_aug = torch.cat([x, s_i.unsqueeze(-1)], dim=-1)  # [N, 774]
        h_cls = self._encode(x_aug, edge_index, edge_attr, self.cls_convs, self.cls_bns)

        # Flow A: focal — suspicion-weighted mean pool
        s_w = s_i.unsqueeze(-1)                                   # [N, 1]
        focal_emb = global_add_pool(h_cls * s_w, batch)           # [B, 256]
        focal_emb = focal_emb / global_add_pool(s_w, batch).clamp(min=1e-6)

        # Flow B: context — uniform mean pool
        context_emb = global_mean_pool(h_cls, batch)              # [B, 256]

        # ── Stage 3: LM branch ───────────────────────────────────────────────
        if func_input_ids is not None:
            lm_emb = lm_pool(self.codebert, self._is_enc_dec, func_input_ids, func_attention_mask)
        else:
            lm_emb = torch.zeros(B, self._lm_dim, device=x.device)

        # ── Tri-modal fusion ─────────────────────────────────────────────────
        logit_func = self.func_head(
            torch.cat([focal_emb, context_emb, lm_emb], dim=-1)   # [B, 1280]
        )

        # ── Stage 1 stmt scores for MIL / ranking loss ───────────────────────
        stmt_scores = (
            self._statement_scores(h_loc, batch, node_line)
            if node_line is not None else None
        )

        return logit_func, stmt_scores
