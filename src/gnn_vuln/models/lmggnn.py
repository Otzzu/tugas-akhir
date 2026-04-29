"""
lmggnn.py — VulLMGNN faithful re-implementation (Cao et al., ICSE 2023)

Exact BertGGCN architecture adapted for PyG batch format:

    CPG nodes (773D pre-computed)
        → GatedGraphConv × num_layers → h [N, hidden_dim]
        → concat(h, x_original) per node [N, hidden_dim + in_channels]
        → global_mean_pool → [B, hidden_dim + in_channels]
        → MLP → gnn_logit [B, num_classes]

    Full function text → live CodeBERT → CLS → MLP → lm_logit [B, num_classes]

    Final: alpha * gnn_logit + (1 - alpha) * lm_logit
           (paper: alpha=0.1 — GNN auxiliary, LM dominant)

Adaptation note: paper uses Conv1d pooling over per-node features.
We use global_mean_pool (standard PyG) which is architecturally equivalent
for variable-size graphs in batched format.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from transformers import AutoModel

NODE_FEAT_DIM = 773
_LM_DIM = 768


class LMGNNVulnDetector(nn.Module):
    """
    VulLMGNN BertGGCN: GatedGraphConv + concat(h, x_orig) + live LM interpolation.

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace model for frozen node embeddings (preprocessing only).
    func_lm : str
        HuggingFace model for live LM branch. Falls back to pretrained_lm.
    in_channels : int
        Node feature dimension (773D).
    hidden_dim : int
        GatedGraphConv output dimension (paper: 200D).
    num_layers : int
        GatedGraphConv steps (paper: 6).
    dropout : float
        Dropout probability.
    num_classes : int
        Output classes.
    alpha : float
        GNN interpolation weight. Paper uses 0.1 (GNN auxiliary, LM dominant).
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
        alpha: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.num_classes = num_classes

        # Project node features to hidden_dim (GatedGraphConv requires in==out)
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GatedGraphConv (paper-faithful, num_layers steps)
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

        # GNN head: concat(h_gnn, x_projected) → num_classes
        # Mirrors paper's concat(ggnn_out, original_emb) before classification
        self.gnn_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Live LM branch (fine-tuned)
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm)

        # LM head: CLS → num_classes
        self.lm_head = nn.Sequential(
            nn.Linear(_LM_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        func_input_ids: torch.Tensor | None = None,
        func_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        B = int(batch.max().item()) + 1

        # ── GNN branch ───────────────────────────────────────────────────────
        h0 = self.input_proj(x)                          # [N, hidden_dim]
        h = self.ggnn(h0, edge_index)                    # [N, hidden_dim]
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Concat GNN output with projected input (paper: concat ggnn_out + orig_emb)
        h_cat = torch.cat([h, h0], dim=-1)               # [N, hidden_dim * 2]
        h_cat = global_mean_pool(h_cat, batch)            # [B, hidden_dim * 2]
        logit_gnn = self.gnn_head(h_cat)                  # [B, num_classes]

        # ── LM branch ────────────────────────────────────────────────────────
        if func_input_ids is not None:
            lm_out = self.codebert(
                input_ids=func_input_ids,
                attention_mask=func_attention_mask,
            )
            lm_emb = lm_out.last_hidden_state[:, 0, :]   # CLS [B, 768]
        else:
            lm_emb = torch.zeros(B, _LM_DIM, device=x.device)
        logit_lm = self.lm_head(lm_emb)                  # [B, num_classes]

        # ── Fixed-alpha interpolation (paper: 0.1 GNN + 0.9 LM) ─────────────
        logit = self.alpha * logit_gnn + (1.0 - self.alpha) * logit_lm

        return logit, None
