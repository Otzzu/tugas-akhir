"""
lmgat_waves_seq.py — Architecture 8: WAVES localization → VulLMGNN classification

Stage 1 (WAVES-inspired, transformer-only):
    Group CPG nodes by source line → extract CodeBERT slice from frozen node features
    → Transformer encoder over statement sequence → per-statement suspicion score
    → MIL binary localization loss

Stage 2 (VulLMGNN-style, two branches):
    GNN branch : node features (773D) + s_i per node → GATv2Conv → pool
    LM  branch : full function text → live LM → CLS token
    concat(gnn_emb, lm_emb) → CWE head

Stage 1 → Stage 2 connection:
    s_i (per-statement suspicion) mapped to node level via node_line.
    Detached before Stage 2 so classification loss only updates Stage 2 params
    and localization loss only updates Stage 1 params.

Config keys:
    pretrained_lm        : node embedder LM (frozen, preprocessing only)
    func_lm              : live LM for Stage 2 function branch (fine-tuned)
    stmt_transformer_layers : Transformer encoder depth for Stage 1 (default 2)
    stmt_transformer_heads  : attention heads in Stage 1 transformer (default 4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoModel

from gnn_vuln.models._lm_utils import lm_hidden_dim, lm_pool

NODE_FEAT_DIM  = 773
EDGE_FEAT_DIM  = 7
# Pre-computed CodeBERT node features: indices 1..768 in the 773D vector.
# Transformer statement encoder operates on this fixed 768-dim slice.
_NODE_CB_DIM   = 768
_CB_START, _CB_END = 1, 769


class LMGATWavesSeqVulnDetector(nn.Module):
    """
    Arch8: WAVES-style transformer localization → VulLMGNN-style classification.

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace model used for frozen node embeddings (preprocessing only).
        Not loaded at runtime — just recorded for reference.
    func_lm : str
        HuggingFace model for the live Stage 2 LM branch. If empty, falls back
        to pretrained_lm.
    in_channels : int
        Node feature dimension (773D).
    hidden_dim : int
        GATv2 hidden width for Stage 2 GNN branch.
    num_layers : int
        GATv2 message-passing steps in Stage 2.
    dropout : float
        Dropout probability.
    num_classes : int
        Output classes (11 for 10-CWE + benign).
    num_heads : int
        Attention heads for GATv2Conv.
    edge_dim : int
        Edge feature dimension (7 for CPG edge types).
    stmt_transformer_layers : int
        Transformer encoder depth for Stage 1 statement encoding.
    stmt_transformer_heads : int
        Multi-head attention heads for Stage 1 transformer.
    """

    def __init__(
        self,
        pretrained_lm: str = "microsoft/codebert-base",
        func_lm: str = "",
        in_channels: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_classes: int = 11,
        num_heads: int = 4,
        edge_dim: int = EDGE_FEAT_DIM,
        stmt_transformer_layers: int = 2,
        stmt_transformer_heads: int = 4,
    ):
        super().__init__()
        self.dropout    = dropout
        self.num_classes = num_classes

        # ── Stage 1: WAVES-style transformer statement encoder ────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=_NODE_CB_DIM,
            nhead=stmt_transformer_heads,
            dim_feedforward=_NODE_CB_DIM * 2,
            dropout=dropout,
            batch_first=True,   # input: [batch, seq, feat]
        )
        self.stmt_transformer = nn.TransformerEncoder(
            enc_layer, num_layers=stmt_transformer_layers
        )
        self.stmt_score_head = nn.Linear(_NODE_CB_DIM, 1)  # binary suspicion logit

        # ── Stage 2 GNN branch (VulLMGNN-style) ──────────────────────────────
        # Input: 773D node features + 1D s_i = 774D
        self.gnn_convs = nn.ModuleList()
        self.gnn_bns   = nn.ModuleList()
        self.gnn_convs.append(
            GATv2Conv(in_channels + 1, hidden_dim, heads=num_heads, concat=False,
                      dropout=dropout, edge_dim=edge_dim)
        )
        self.gnn_bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.gnn_convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                          dropout=dropout, edge_dim=edge_dim)
            )
            self.gnn_bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Stage 2 LM branch (VulLMGNN-style, live fine-tuned) ──────────────
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm)
        self._lm_dim = lm_hidden_dim(self.codebert)
        self._is_enc_dec = getattr(self.codebert.config, "is_encoder_decoder", False)

        # ── Stage 2 function head ─────────────────────────────────────────────
        self.func_head = nn.Sequential(
            nn.Linear(hidden_dim + self._lm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── Stage 1 helpers ──────────────────────────────────────────────────────

    def _stage1(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns
        -------
        node_suspicion : [N] float  — per-node suspicion score (detached)
        stmt_scores    : list[tensor]  — per-statement raw logits for MIL loss
        """
        N = x.shape[0]
        device = x.device
        batch_size = int(batch.max().item()) + 1

        node_susp = torch.zeros(N, device=device)
        stmt_scores_list: list[torch.Tensor] = []

        for b in range(batch_size):
            mask      = batch == b
            x_b       = x[mask]           # [N_b, 773]
            lines_b   = node_line[mask]   # [N_b]
            valid     = lines_b >= 0

            if not valid.any():
                stmt_scores_list.append(torch.zeros(0, device=device))
                continue

            x_bv     = x_b[valid]
            lines_bv = lines_b[valid]
            # Global indices of valid nodes in this graph
            node_idx = mask.nonzero(as_tuple=True)[0][valid]

            unique_lines = lines_bv.unique(sorted=True)
            n_stmts = len(unique_lines)

            # Build statement embeddings from CodeBERT slice [n_stmts, 768]
            stmt_embs = torch.stack([
                x_bv[lines_bv == line, _CB_START:_CB_END].mean(0)
                for line in unique_lines
            ])  # [n_stmts, 768]

            # Transformer encoder: [1, n_stmts, 768] → [1, n_stmts, 768]
            out = self.stmt_transformer(stmt_embs.unsqueeze(0)).squeeze(0)  # [n_stmts, 768]

            # Per-statement suspicion logits and scores
            logits     = self.stmt_score_head(out).squeeze(-1)   # [n_stmts]
            suspicion  = torch.sigmoid(logits)                    # [n_stmts]

            stmt_scores_list.append(logits)

            # Map statement → node level via node_line (detach: Stage 2 is independent)
            for stmt_i, line in enumerate(unique_lines):
                nm = lines_bv == line
                node_susp[node_idx[nm]] = suspicion[stmt_i].detach()

        return node_susp, stmt_scores_list

    # ── Stage 2 GNN encoder ──────────────────────────────────────────────────

    def _gnn_encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        for conv, bn in zip(self.gnn_convs, self.gnn_bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

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

        # ── Stage 1: WAVES transformer localization ───────────────────────────
        stmt_scores = None
        if node_line is not None:
            node_suspicion, stmt_scores = self._stage1(x, batch, node_line)
        else:
            node_suspicion = torch.zeros(x.shape[0], device=x.device)

        # ── Stage 2 GNN branch ────────────────────────────────────────────────
        x_aug   = torch.cat([x, node_suspicion.unsqueeze(-1)], dim=-1)  # [N, 774]
        h       = self._gnn_encode(x_aug, edge_index, edge_attr)
        gnn_emb = global_mean_pool(h, batch)                             # [B, hidden_dim]

        # ── Stage 2 LM branch ────────────────────────────────────────────────
        if func_input_ids is not None:
            lm_emb = lm_pool(self.codebert, self._is_enc_dec, func_input_ids, func_attention_mask)
        else:
            lm_emb = torch.zeros(B, self._lm_dim, device=x.device)

        # ── Classification head ───────────────────────────────────────────────
        logit_func = self.func_head(torch.cat([gnn_emb, lm_emb], dim=-1))

        return logit_func, stmt_scores
