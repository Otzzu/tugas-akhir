"""
lmrgcn.py — LM-RGCN: Relational GCN + live LM branch

Same structure as lmgat_codebert (Arch3) but replaces GATv2Conv with RGCNConv.
RGCNConv uses a separate weight matrix per CPG edge relation instead of attention,
naturally modelling the 7 distinct CPG edge types (AST, CFG, CDG, DDG, PDG, CALL,
REACHING_DEF) without needing one-hot edge_attr padding.

  Full function text → live LM → CLS [B, lm_dim] ──────────────────────┐
                                                                         │
  CPG nodes (773D pre-computed)                                          │
      → RGCNConv × num_layers  (relation-specific weight per edge type)  │
      → BatchNorm + ReLU + Dropout                                        │
      → global_mean_pool → h_graph [B, hidden_dim] ── concat ───────────┘
                                                         ↓
                                                fused [B, hidden_dim + lm_dim]
                                                         ↓
                                              MLP → logit [B, num_classes]

  + Statement head: per-line binary score for MIL localisation.

Edge type mapping (one-hot edge_attr → integer edge_type):
  argmax of edge_attr [E, 7] → edge_type [E] in {0..6}
  0=AST, 1=CFG, 2=CDG, 3=DDG, 4=PDG, 5=CALL, 6=REACHING_DEF
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from transformers import AutoConfig, AutoModel

from gnn_vuln.models._lm_utils import lm_hidden_dim, lm_pool

NODE_FEAT_DIM = 773
NUM_RELATIONS = 7   # CPG edge types
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMRGCNVulnDetector(nn.Module):
    """
    Relational GCN vulnerability detector with a live LM branch.

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace ID for pre-computed node embeddings (frozen, affects dataset).
    func_lm : str
        HuggingFace ID for the live LM text branch. Falls back to pretrained_lm.
    in_channels : int
        Node feature dimension (773D).
    hidden_dim : int
        RGCN hidden width.
    num_layers : int
        Number of RGCNConv message-passing steps.
    dropout : float
    num_classes : int
        Output classes (11 for top-10 CWE + benign).
    num_relations : int
        Number of CPG edge relation types (7).
    num_bases : int | None
        Basis decomposition for RGCN weight matrices. None = full matrices.
        Use e.g. 4 to reduce parameters when num_relations is large.
    use_skip : bool
        Residual connections around each RGCN layer.
    matryoshka_dim : int | None
        Truncate LM output to this dimension (for Matryoshka embedding models).
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
        num_relations: int = NUM_RELATIONS,
        num_bases: int | None = None,
        use_skip: bool = False,
        matryoshka_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip
        self._matryoshka_dim = matryoshka_dim

        # ── Live LM branch ────────────────────────────────────────────────────
        _func_lm = func_lm if func_lm else pretrained_lm
        _lm_cfg = AutoConfig.from_pretrained(_func_lm, trust_remote_code=True)
        if not hasattr(_lm_cfg, "is_decoder"):
            _lm_cfg.is_decoder = False
        self.codebert = AutoModel.from_pretrained(_func_lm, config=_lm_cfg, trust_remote_code=True)
        if hasattr(self.codebert, "gradient_checkpointing_enable"):
            self.codebert.gradient_checkpointing_enable()
        self._lm_dim = lm_hidden_dim(self.codebert, matryoshka_dim)
        self._is_enc_dec = getattr(self.codebert.config, "is_encoder_decoder", False)

        # ── Relational GCN encoder ────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_dim, num_relations, num_bases=num_bases))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = nn.ModuleList()
            self.res_projs.append(nn.Linear(in_channels, hidden_dim, bias=False))
            for _ in range(num_layers - 1):
                self.res_projs.append(nn.Identity())

        # ── Function head ─────────────────────────────────────────────────────
        fused_dim = hidden_dim + self._lm_dim
        self.func_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ── Statement head (MIL localisation) ────────────────────────────────
        self.stmt_max_head  = nn.Linear(hidden_dim, 1)
        self.stmt_mean_head = nn.Linear(hidden_dim, 1)

    # ── Encoder ──────────────────────────────────────────────────────────────

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        # one-hot [E, 7] → integer edge_type [E]
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_type = edge_attr.argmax(dim=-1)
        else:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index, edge_type=edge_type)
            x = bn(x)
            if residual is not None:
                x = F.relu(x + residual)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    # ── Statement head ────────────────────────────────────────────────────────

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
            h_b, lines_b = h_b[valid], lines_b[valid]
            unique_lines = lines_b.unique(sorted=True)
            scores: list[torch.Tensor] = []
            for line in unique_lines:
                nm = lines_b == line
                h_line = h_b[nm]
                s = (
                    _ALPHA_MAX  * self.stmt_max_head(h_line.max(dim=0).values)
                    + _ALPHA_MEAN * self.stmt_mean_head(h_line.mean(dim=0))
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
        h = self._encode(x, edge_index, edge_attr)
        h_graph = global_mean_pool(h, batch)   # [B, hidden_dim]

        B = h_graph.size(0)
        if func_input_ids is not None:
            cls = lm_pool(self.codebert, self._is_enc_dec, func_input_ids, func_attention_mask,
                          matryoshka_dim=self._matryoshka_dim)
        else:
            cls = torch.zeros(B, self._lm_dim, device=h_graph.device)

        fused = torch.cat([h_graph, cls], dim=-1)
        logit = self.func_head(fused)

        stmt_scores = (
            self._statement_scores(h, batch, node_line)
            if node_line is not None else None
        )
        return logit, stmt_scores
