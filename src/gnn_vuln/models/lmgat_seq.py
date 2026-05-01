"""
lmgat_seq.py — Architecture 7: Sequential Localization → Classification

Stage 1: GATv2 binary localization → per-node suspicion score s_i ∈ [0,1]
Stage 2: GATv2(concat[stage2_input, s_i]) + live LM → CWE classification

stage2_node_input controls what Stage 2 GNN receives:
  "raw"  : concat(x_frozen, s_i)  [N, 774] — original features + suspicion
  "loc"  : concat(h_loc, s_i)     [N, 257] — Stage 1 learned features + suspicion
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from transformers import AutoModel

NODE_FEAT_DIM = 773
EDGE_FEAT_DIM = 7
_LM_DIM = 768
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATSeqVulnDetector(nn.Module):
    """
    Sequential Localization → Classification detector.

    Stage 1 (binary localization):
        frozen node embeddings → GATv2Conv × num_layers → h_loc
        binary stmt head → s_i per node (suspicion score)
        returns stmt_scores for MIL localization loss

    Stage 2 (multiclass CWE classification):
        Branch 1 — GNN: concat(node_feat, s_i) → GATv2Conv → suspicion-weighted pool
        Branch 2 — LM:  full function text → live LM → CLS
        concat(graph_emb, lm_emb) → func_head → CWE logits
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
        stage2_node_input: str = "raw",
    ):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        # "loc": Stage 2 GNN receives h_loc (256D) + s_i
        # "raw": Stage 2 GNN receives x_frozen (773D) + s_i
        self._use_loc = (stage2_node_input == "loc")
        cls_in = (hidden_dim + 1) if self._use_loc else (in_channels + 1)

        # Stage 1: Localization GNN (input: original 773D)
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

        # Stage 1: Binary statement head (scalar suspicion per node/line)
        self.loc_stmt_max  = nn.Linear(hidden_dim, 1)
        self.loc_stmt_mean = nn.Linear(hidden_dim, 1)

        # Stage 2: Classification GNN (input dim depends on stage2_node_input)
        self.cls_convs = nn.ModuleList()
        self.cls_bns = nn.ModuleList()
        self.cls_convs.append(
            GATv2Conv(cls_in, hidden_dim, heads=num_heads, concat=False,
                      dropout=dropout, edge_dim=edge_dim)
        )
        self.cls_bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.cls_convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                          dropout=dropout, edge_dim=edge_dim)
            )
            self.cls_bns.append(nn.BatchNorm1d(hidden_dim))

        # Stage 2: Live LM branch (func_lm overrides pretrained_lm if set)
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm, use_safetensors=True)

        # Stage 2: Function head
        self.func_head = nn.Sequential(
            nn.Linear(hidden_dim + _LM_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── Shared GNN encoder ───────────────────────────────────────────────────

    def _encode(self, x, edge_index, edge_attr, convs, bns):
        for conv, bn in zip(convs, bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    # ── Stage 1: per-node suspicion score ───────────────────────────────────

    def _node_suspicion(self, h_loc: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid suspicion score per node: [N]"""
        raw = _ALPHA_MAX * self.loc_stmt_max(h_loc) + _ALPHA_MEAN * self.loc_stmt_mean(h_loc)
        return torch.sigmoid(raw).squeeze(-1)  # [N]

    # ── Stage 1: per-line statement scores for MIL loss ─────────────────────

    def _statement_scores(
        self,
        h_loc: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Binary scalar score per source line (for MIL localization loss)."""
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

        # ── Stage 1: Localization ────────────────────────────────────────────
        h_loc = self._encode(x, edge_index, edge_attr, self.loc_convs, self.loc_bns)
        s_i = self._node_suspicion(h_loc)  # [N]

        # ── Stage 2: Classification GNN on augmented features ────────────────
        stage2_base = h_loc if self._use_loc else x
        x_aug = torch.cat([stage2_base, s_i.unsqueeze(-1)], dim=-1)  # [N, 257 or 774]
        h_cls = self._encode(x_aug, edge_index, edge_attr, self.cls_convs, self.cls_bns)

        # Suspicion-weighted global pooling
        s_w = s_i.unsqueeze(-1)                                  # [N, 1]
        graph_emb = global_add_pool(h_cls * s_w, batch)          # [B, hidden_dim]
        weight_sum = global_add_pool(s_w, batch).clamp(min=1e-6) # [B, 1]
        graph_emb = graph_emb / weight_sum                        # [B, hidden_dim]

        # ── Stage 2: LM branch ───────────────────────────────────────────────
        if func_input_ids is not None:
            lm_out = self.codebert(
                input_ids=func_input_ids,
                attention_mask=func_attention_mask,
            )
            lm_emb = lm_out.last_hidden_state[:, 0, :]  # CLS [B, 768]
        else:
            lm_emb = torch.zeros(B, _LM_DIM, device=x.device)

        # ── Classification head ──────────────────────────────────────────────
        func_in = torch.cat([graph_emb, lm_emb], dim=-1)  # [B, hidden_dim + 768]
        logit_func = self.func_head(func_in)

        # ── Stage 1 stmt scores for MIL localization loss ───────────────────
        stmt_scores = (
            self._statement_scores(h_loc, batch, node_line)
            if node_line is not None else None
        )

        return logit_func, stmt_scores
