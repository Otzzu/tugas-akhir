"""
lmgat_codebert_mtl.py — Architecture: LM-GAT v2 + Live CodeBERT + MTL heads

Multi-Task Learning with three output heads on a shared encoder:

  Full function text → CodeBERT (live, lr=2e-5) → CLS [B, 768] ─────────┐
                                                                           │
  CPG nodes (773D pre-computed)                                            │
      → GATv2Conv × num_layers (lr=1e-3)                                  │
      → BatchNorm + ReLU + Dropout                                         │
      → global_mean_pool → h_graph [B, hidden_dim] ── concat ─────────────┘
                                                          ↓
                                                fused [B, hidden_dim + 768]
                                                          │
                     ┌────────────────────────────────────┤
                     │                                    │
                     ▼                                    ▼
              binary_head                           group_head
              [B, 2]                                [B, num_groups]
              (safe / vuln)                         (coarse CWE family)
                                                          │
                                              softmax + detach (routing hint)
                                                          │
                                              concat(fused, group_probs)
                                                          ↓
                                                    cwe_head
                                                 [B, num_classes]
                                                 (fine CWE or group ID)

  + Statement head: per-line binary score for MIL localisation (unchanged)

Losses (computed in train.py):
    loss = cwe_loss
         + group_loss_weight * group_loss
         + binary_loss_weight * binary_loss
         [+ mil_weight * mil_loss]

Labels required per Data object:
    batch.y        : CWE class ID (0 = benign) — primary target
    batch.group_id : group class ID (0 = benign) — auxiliary coarse target
    binary label derived as (batch.y > 0).long() inside train.py

Forward signature differs from base lmgat_codebert:
    returns (logit_cwe, logit_group, logit_binary, stmt_scores)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoModel

NODE_FEAT_DIM = 773
EDGE_FEAT_DIM = 7
_CODEBERT_DIM = 768
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATCodeBERTMTLVulnDetector(nn.Module):
    """
    GATv2 + live CodeBERT with three MTL output heads:
      1. binary_head  — safe vs. vulnerable (gradient from ALL samples)
      2. group_head   — coarse CWE family (VulANalyzeR + MulVul coarse routing)
      3. cwe_head     — fine-grained CWE class conditioned on group probs

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace model name for pre-computed node embeddings (frozen).
    func_lm : str
        HuggingFace model name for the live CodeBERT branch. Falls back to
        pretrained_lm if empty.
    in_channels : int
        Node feature dimension (773D).
    hidden_dim : int
        GATv2 hidden width.
    num_layers : int
        Number of GATv2Conv message-passing steps.
    dropout : float
        Dropout probability.
    num_classes : int
        Fine-grained CWE head output size (e.g. 11 for top-10 CWE + benign).
    num_groups : int
        Coarse group head output size (e.g. 16 for benign + 15 CWE families).
    num_heads : int
        GATv2 attention heads.
    edge_dim : int
        Edge feature dimension (7 for one-hot CPG edge types).
    add_self_loops : bool
        Whether GATv2Conv adds self-loops (node attends to itself within
        aggregation). Default True. When edge_dim is set, PyG pads self-loop
        edge features with zeros.
    use_skip : bool
        Whether to add residual skip connections around each GATv2 layer.
        Layer 0 uses a learned linear projection (in_channels → hidden_dim);
        subsequent layers use identity. Prevents oversmoothing in deep stacks
        and preserves the CodeBERT node embeddings through aggregation.
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
        num_groups: int = 16,
        num_heads: int = 4,
        edge_dim: int = EDGE_FEAT_DIM,
        use_group_cond: bool = True,
        add_self_loops: bool = True,
        use_skip: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.use_group_cond = use_group_cond
        self.use_skip = use_skip

        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm, use_safetensors=True)

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

        # ── Residual projections (only created when use_skip=True) ───────────
        if use_skip:
            self.res_projs = nn.ModuleList()
            self.res_projs.append(nn.Linear(in_channels, hidden_dim, bias=False))
            for _ in range(num_layers - 1):
                self.res_projs.append(nn.Identity())

        fused_dim = hidden_dim + _CODEBERT_DIM

        # ── Head 1: binary — safe vs. vulnerable ────────────────────────────
        self.binary_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        # ── Head 2: group — coarse CWE family ───────────────────────────────
        self.group_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_groups),
        )

        # ── Head 3: CWE — fine-grained, conditioned on group probs ──────────
        # Input: concat(fused, softmax(group_logits).detach())
        self.cwe_head = nn.Sequential(
            nn.Linear(fused_dim + num_groups, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ── Statement head: binary score per line (MIL localisation) ────────
        self.stmt_max_head = nn.Linear(hidden_dim, 1)
        self.stmt_mean_head = nn.Linear(hidden_dim, 1)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """
        Parameters
        ----------
        x                   : [N, in_channels]
        edge_index          : [2, E]
        batch               : [N]
        node_line           : [N]  source line per node (-1 = unknown)
        edge_attr           : [E, edge_dim]
        func_input_ids      : [B, seq_len]
        func_attention_mask : [B, seq_len]

        Returns
        -------
        logit_cwe    : [B, num_classes]   fine-grained CWE logits (primary)
        logit_group  : [B, num_groups]    coarse group logits (auxiliary)
        logit_binary : [B, 2]             binary safe/vuln logits (auxiliary)
        stmt_scores  : list of [n_stmts_i] | None
        """
        h = self._encode(x, edge_index, edge_attr)
        h_graph = global_mean_pool(h, batch)  # [B, hidden_dim]

        B = h_graph.size(0)
        if func_input_ids is not None:
            cb_out = self.codebert(
                input_ids=func_input_ids,
                attention_mask=func_attention_mask,
            )
            cls = cb_out.last_hidden_state[:, 0, :]  # [B, 768]
        else:
            cls = torch.zeros(B, _CODEBERT_DIM, device=h_graph.device)

        fused = torch.cat([h_graph, cls], dim=-1)  # [B, hidden_dim + 768]

        logit_binary = self.binary_head(fused)  # [B, 2]
        logit_group = self.group_head(fused)    # [B, num_groups]

        # CWE head: conditioned on group probs when use_group_cond=True,
        # otherwise zeros (no group PT needed; avoids random-noise conditioning)
        if self.use_group_cond:
            group_probs = F.softmax(logit_group.detach(), dim=-1)
        else:
            group_probs = torch.zeros(fused.size(0), self.num_groups, device=fused.device)
        cwe_in = torch.cat([fused, group_probs], dim=-1)       # [B, fused_dim + num_groups]
        logit_cwe = self.cwe_head(cwe_in)                      # [B, num_classes]

        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None else None
        )

        return logit_cwe, logit_group, logit_binary, stmt_scores
