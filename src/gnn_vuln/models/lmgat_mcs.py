"""
lmgat_mcs.py — Architecture 4: LM-GAT v2 + Live CodeBERT + Multiclass Statement Head

WAVES-style strict sync: the function prediction is derived directly from
statement scores, eliminating the desync root cause.

Full function text (func_input_ids / func_attention_mask in Data)
    → CodeBERT (LIVE, FINE-TUNED, lr=2e-5)
    → CLS token [B, 768] ────────────────────────────────────────────────────┐
                                                                              │
CPG nodes (773D pre-computed, frozen)                                         │
    → GATv2Conv × num_layers (lr=1e-3)                                       │
    → BatchNorm + ReLU + Dropout                                              │
    ↓                                                                         │
    stmt_head (MULTICLASS):                                                   │
        group nodes by source line                                            │
        max-pool(h_line)  → Linear(hidden_dim, num_classes) → score_max_j    │
        mean-pool(h_line) → Linear(hidden_dim, num_classes) → score_mean_j   │
        stmt_score_j = 0.8*score_max_j + 0.6*score_mean_j                    │
        → stmt_scores [n_stmts, num_classes]                                  │
              ↓                                                               │
        max(stmt_scores, dim=0) [num_classes] ──── concat ────────────────────┘
                                                       ↓
                                              [num_classes + 768]
                                                       ↓
                                               MLP → logit_func [B, num_classes]

Key property: func_logit is derived from max_pool(stmt_scores) + CodeBERT CLS.
Function and statement predictions are in strict sync by construction —
the function CWE prediction cannot disagree with the statement CWE scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from transformers import AutoModel

from gnn_vuln.models._lm_utils import lm_hidden_dim, lm_pool

NODE_FEAT_DIM = 773
EDGE_FEAT_DIM = 7
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATMCSVulnDetector(nn.Module):
    """
    GATv2 detector with multiclass statement head and strict func/stmt sync.

    The statement head outputs [n_stmts, num_classes] — each statement gets a
    per-CWE-class score vector instead of a binary suspicious/not scalar.
    The function-level prediction is then derived by max-pooling statement
    scores across all lines, then concatenating with the CodeBERT CLS token.

    This eliminates the independent func_head, so the function prediction is
    always mathematically consistent with the statement-level scores.

    MIL loss is multiclass CE: for a function of class c, the top-k statements
    (ranked by their score for class c) are pushed toward class c.

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace model name for the live CodeBERT branch.
    in_channels : int
        Node feature dimension (773D for pre-computed CPG graphs).
    hidden_dim : int
        GATv2 hidden width.
    num_layers : int
        Number of GATv2Conv message-passing steps.
    dropout : float
        Dropout probability.
    num_classes : int
        Number of output classes (11 for 10-CWE + benign multiclass).
    num_heads : int
        GATv2 attention heads (concat=False → output stays hidden_dim).
    edge_dim : int
        Edge feature dimension (7 for one-hot CPG edge types).
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
        add_self_loops: bool = True,
        use_skip: bool = False,
        matryoshka_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_skip = use_skip
        self._matryoshka_dim = matryoshka_dim

        # ── Live fine-tuned LM for full-function context ─────────────────────
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm, trust_remote_code=True)
        self._lm_dim = lm_hidden_dim(self.codebert, matryoshka_dim)
        self._is_enc_dec = getattr(self.codebert.config, "is_encoder_decoder", False)

        # ── Shared GATv2 encoder ─────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels, hidden_dim,
                heads=num_heads, concat=False, dropout=dropout,
                edge_dim=edge_dim, add_self_loops=add_self_loops,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, hidden_dim,
                    heads=num_heads, concat=False, dropout=dropout,
                    edge_dim=edge_dim, add_self_loops=add_self_loops,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = nn.ModuleList()
            self.res_projs.append(nn.Linear(in_channels, hidden_dim, bias=False))
            for _ in range(num_layers - 1):
                self.res_projs.append(nn.Identity())

        # ── Statement head: multiclass score per line ────────────────────────
        self.stmt_max_head = nn.Linear(hidden_dim, num_classes)
        self.stmt_mean_head = nn.Linear(hidden_dim, num_classes)

        # ── Function head: max_pool(stmt_scores) + CodeBERT CLS → num_classes
        # Input: [B, num_classes + 768]
        self.func_head = nn.Sequential(
            nn.Linear(num_classes + self._lm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── Shared encoder ───────────────────────────────────────────────────────

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            if residual is not None:
                x = F.relu(x + residual)
            else:
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
        Produce per-statement multiclass logit vectors by grouping CPG nodes
        by source line.

        Returns list of length B; each element is [n_stmts_i, num_classes].
        Empty graphs get shape [0, num_classes].
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
                result.append(torch.zeros(0, self.num_classes, device=device))
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
                s_max = self.stmt_max_head(h_max)    # [num_classes]
                s_mean = self.stmt_mean_head(h_mean)  # [num_classes]
                scores.append(_ALPHA_MAX * s_max + _ALPHA_MEAN * s_mean)
            result.append(torch.stack(scores))  # [n_stmts, num_classes]

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
        Parameters
        ----------
        x                  : [N, in_channels]
        edge_index         : [2, E]
        batch              : [N]
        node_line          : [N]
        edge_attr          : [E, edge_dim]
        func_input_ids     : [B, 512]
        func_attention_mask: [B, 512]

        Returns
        -------
        logit_func  : [B, num_classes]
        stmt_scores : list of [n_stmts_i, num_classes] | None
        """
        h = self._encode(x, edge_index, edge_attr)
        B = int(batch.max().item()) + 1

        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None else None
        )

        # ── CodeBERT CLS for full-function context ───────────────────────────
        if func_input_ids is not None:
            cls = lm_pool(self.codebert, self._is_enc_dec, func_input_ids, func_attention_mask, matryoshka_dim=self._matryoshka_dim)
        else:
            cls = torch.zeros(B, self._lm_dim, device=h.device)

        # ── Derive function prediction from stmt_scores + CLS ────────────────
        if stmt_scores is not None:
            # Max-pool across statements for each class → [B, num_classes]
            stmt_max = torch.stack([
                scores.max(dim=0).values if scores.shape[0] > 0
                else torch.zeros(self.num_classes, device=h.device)
                for scores in stmt_scores
            ])
        else:
            stmt_max = torch.zeros(B, self.num_classes, device=h.device)

        func_in = torch.cat([stmt_max, cls], dim=-1)  # [B, num_classes + 768]
        logit_func = self.func_head(func_in)

        return logit_func, stmt_scores
