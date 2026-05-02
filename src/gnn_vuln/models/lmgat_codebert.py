"""
lmgat_codebert.py — Architecture 3: LM-GAT v2 + Live Fine-tuned CodeBERT

Full function text (func_input_ids / func_attention_mask stored in Data)
    → CodeBERT (LIVE, FINE-TUNED, lr=2e-5)
    → CLS token [B, 768] ────────────────────────────────────────────┐
                                                                      │
CPG nodes (773D pre-computed, frozen — practical limitation)          │
    → GATv2Conv × num_layers (lr=1e-3)                               │
    → BatchNorm + ReLU + Dropout                                      │
    ↓                                                                 │
    global_mean_pool(h) [B, hidden_dim] ─── concat ──────────────────┘
                                                ↓
                                       [B, hidden_dim + 768]
                                                ↓
                                         MLP → logit_func [B, num_classes]
    └── stmt_head: group nodes by source line
              max-pool + mean-pool → dual scorers → stmt_scores [n_stmts]  (binary)

Training recipe:
    Two AdamW param groups — CodeBERT lr=2e-5, GNN lr=1e-3.
    Linear warmup + decay scheduler. Gradient clipping max_norm=1.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoModel

NODE_FEAT_DIM = 773   # 1 (node_type) + 768 (CodeBERT CLS) + 3 (dist) + 1 (danger API)
EDGE_FEAT_DIM = 7     # one-hot: AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF
_CODEBERT_DIM = 768
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATCodeBERTVulnDetector(nn.Module):
    """
    GATv2 vulnerability detector with pre-computed node embeddings AND a
    live fine-tuned CodeBERT branch for full-function context.

    The GNN encoder produces statement-level representations from the CPG.
    The live CodeBERT branch tokenises the full function text and provides
    its CLS token as a global function summary.  Both are concatenated before
    the function-level classifier, giving the model both local (CPG node) and
    global (full function) context.

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
        Dropout probability (applied after each GATv2 layer and in func_head).
    num_classes : int
        Number of output classes (2 for binary, 11 for 10-CWE multiclass).
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
        num_classes: int = 2,
        num_heads: int = 4,
        edge_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()
        self.dropout = dropout

        # ── Live fine-tuned LM for full-function context ─────────────────────
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm, use_safetensors=True)

        # ── Shared GATv2 encoder ─────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels, hidden_dim,
                heads=num_heads, concat=False, dropout=dropout, edge_dim=edge_dim,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, hidden_dim,
                    heads=num_heads, concat=False, dropout=dropout, edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Function head: concat(GNN pooled, CodeBERT CLS) → num_classes ───
        self.func_head = nn.Sequential(
            nn.Linear(hidden_dim + _CODEBERT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ── Statement head: binary suspicious-or-not score per line ─────────
        self.stmt_max_head = nn.Linear(hidden_dim, 1)
        self.stmt_mean_head = nn.Linear(hidden_dim, 1)

    # ── Shared encoder ───────────────────────────────────────────────────────

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
        return x  # [N, hidden_dim]

    # ── Statement-level head ─────────────────────────────────────────────────

    def _statement_scores(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Group CPG nodes by source line and produce one binary score per line.

        Returns list of length B; each element is [n_stmts_i] float tensor.
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
        Parameters
        ----------
        x                  : [N, in_channels]  pre-computed node features
        edge_index         : [2, E]            COO edge list
        batch              : [N]               graph index per node
        node_line          : [N]               source line per node (-1 = unknown)
        edge_attr          : [E, edge_dim]     one-hot edge type features
        func_input_ids     : [B, 512]          tokenized full function text
        func_attention_mask: [B, 512]          attention mask for func_input_ids

        Returns
        -------
        logit_func  : [B, num_classes]
        stmt_scores : list of [n_stmts_i] | None
        """
        h = self._encode(x, edge_index, edge_attr)
        h_graph = global_mean_pool(h, batch)  # [B, hidden_dim]

        B = h_graph.size(0)
        if func_input_ids is not None:
            if getattr(self.codebert.config, "is_encoder_decoder", False):
                cb_out = self.codebert.encoder(
                    input_ids=func_input_ids,
                    attention_mask=func_attention_mask,
                )
            else:
                cb_out = self.codebert(
                    input_ids=func_input_ids,
                    attention_mask=func_attention_mask,
                )
            cls = cb_out.last_hidden_state[:, 0, :]  # [B, 768]
        else:
            cls = torch.zeros(B, _CODEBERT_DIM, device=h_graph.device)

        func_in = torch.cat([h_graph, cls], dim=-1)  # [B, hidden_dim + 768]
        logit_func = self.func_head(func_in)

        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None else None
        )
        return logit_func, stmt_scores
