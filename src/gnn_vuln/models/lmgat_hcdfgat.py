"""
lmgat_hcdfgat.py — Architecture: HC-DFGAT (Hierarchical Contrastive Dual-Flow GAT)

Extends LMGATDualFlow with MTL output heads and a returned Z_combined embedding
for hierarchical supervised contrastive regularisation.

                          Stage 1: Localization GNN (773D input)
                              → per-node suspicion s_i [N]
                              → stmt_scores (MIL / ranking)

                          Stage 2: Classification GNN (774D = 773 + s_i)
                              focal_emb   = suspicion-weighted pool [B, 256]
                              context_emb = global mean pool        [B, 256]

                          Stage 3: Live UniXcoder CLS                [B, 768]

    Z_combined = concat(focal_emb, context_emb, lm_emb)             [B, 1280]
                                    │
          ┌─────────────────────────┤────────────────────────┐
          │                         │                        │
          ▼                         ▼                        │
    binary_head               group_head                     │
    [B, 2]                    [B, num_groups]                │
                                    │                        │
                          softmax + detach                   │
                                    │                        │
                          concat(Z_combined, group_probs)    │
                                    ▼                        │
                               cwe_head                      │
                            [B, num_classes]  ←──────────────┘

Returns: (logit_cwe, logit_group, logit_binary, stmt_scores, z_combined)

Losses (computed in train.py):
    total = LIVABLE_CE(cwe)
          + group_loss_weight  * CE(group)
          + binary_loss_weight * CE(binary)
          + mil_weight         * MIL(stmt_scores)
          + rank_loss_weight   * RankLoss(stmt_scores)
          + supcon_weight      * HierarchicalSupCon(z_combined)
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


class LMGATHCDFGATVulnDetector(nn.Module):
    """
    HC-DFGAT: dual-flow GATv2 encoder + live UniXcoder + three MTL heads.

    Parameters
    ----------
    pretrained_lm : str
        HuggingFace ID for frozen node embeddings (used as fallback for func_lm).
    func_lm : str
        HuggingFace ID for the live UniXcoder branch.
    in_channels : int
        Node feature dimension (773D pre-computed).
    hidden_dim : int
        GATv2 hidden width.
    num_layers : int
        Number of GATv2Conv layers per GNN stage.
    dropout : float
    num_classes : int
        Fine-grained CWE head output size (e.g. 11 = top-10 CWE + benign).
    num_groups : int
        Coarse group head output size (e.g. 16 = 15 CWE groups + benign).
    num_heads : int
        GATv2 attention heads.
    edge_dim : int
        Edge feature dimension (7 for one-hot CPG edge types).
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
        num_groups: int = 16,
        num_heads: int = 4,
        edge_dim: int = EDGE_FEAT_DIM,
        use_group_cond: bool = True,
        add_self_loops: bool = True,
        use_skip: bool = False,
        matryoshka_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.use_group_cond = use_group_cond
        self.use_skip = use_skip
        self._matryoshka_dim = matryoshka_dim

        # ── Stage 1: Localization GNN (input: 773D) ──────────────────────────
        self.loc_convs = nn.ModuleList()
        self.loc_bns = nn.ModuleList()
        self.loc_convs.append(
            GATv2Conv(in_channels, hidden_dim, heads=num_heads, concat=False,
                      dropout=dropout, edge_dim=edge_dim, add_self_loops=add_self_loops)
        )
        self.loc_bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.loc_convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                          dropout=dropout, edge_dim=edge_dim, add_self_loops=add_self_loops)
            )
            self.loc_bns.append(nn.BatchNorm1d(hidden_dim))

        self.loc_stmt_max  = nn.Linear(hidden_dim, 1)
        self.loc_stmt_mean = nn.Linear(hidden_dim, 1)

        # ── Stage 2: Classification GNN (input: 773D + 1D s_i = 774D) ────────
        self.cls_convs = nn.ModuleList()
        self.cls_bns = nn.ModuleList()
        self.cls_convs.append(
            GATv2Conv(in_channels + 1, hidden_dim, heads=num_heads, concat=False,
                      dropout=dropout, edge_dim=edge_dim, add_self_loops=add_self_loops)
        )
        self.cls_bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.cls_convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                          dropout=dropout, edge_dim=edge_dim, add_self_loops=add_self_loops)
            )
            self.cls_bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.loc_res_projs = nn.ModuleList()
            self.loc_res_projs.append(nn.Linear(in_channels, hidden_dim, bias=False))
            for _ in range(num_layers - 1):
                self.loc_res_projs.append(nn.Identity())
            self.cls_res_projs = nn.ModuleList()
            self.cls_res_projs.append(nn.Linear(in_channels + 1, hidden_dim, bias=False))
            for _ in range(num_layers - 1):
                self.cls_res_projs.append(nn.Identity())

        # ── Stage 3: Live LM branch ───────────────────────────────────────────
        _func_lm = func_lm if func_lm else pretrained_lm
        self.codebert = AutoModel.from_pretrained(_func_lm, trust_remote_code=True)
        self._lm_dim = lm_hidden_dim(self.codebert, matryoshka_dim)
        self._is_enc_dec = getattr(self.codebert.config, "is_encoder_decoder", False)

        # ── MTL heads ────────────────────────────────────────────────────────
        fused_dim = hidden_dim * 2 + self._lm_dim

        self.binary_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.group_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_groups),
        )

        # CWE head conditioned on group probs (detached coarse-to-fine routing)
        self.cwe_head = nn.Sequential(
            nn.Linear(fused_dim + num_groups, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    # ── Shared GNN encoder ───────────────────────────────────────────────────

    def _encode(self, x, edge_index, edge_attr, convs, bns, res_projs=None):
        for i, (conv, bn) in enumerate(zip(convs, bns)):
            residual = res_projs[i](x) if res_projs is not None else None
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            if residual is not None:
                x = F.relu(x + residual)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    # ── Stage 1 helpers ──────────────────────────────────────────────────────

    def _node_suspicion(self, h_loc: torch.Tensor) -> torch.Tensor:
        raw = _ALPHA_MAX * self.loc_stmt_max(h_loc) + _ALPHA_MEAN * self.loc_stmt_mean(h_loc)
        return torch.sigmoid(raw).squeeze(-1)  # [N]

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
    ) -> tuple[
        torch.Tensor,           # logit_cwe    [B, num_classes]
        torch.Tensor,           # logit_group  [B, num_groups]
        torch.Tensor,           # logit_binary [B, 2]
        list[torch.Tensor] | None,  # stmt_scores
        torch.Tensor,           # z_combined   [B, 1280]
    ]:
        B = int(batch.max().item()) + 1

        # ── Stage 1: Localization ─────────────────────────────────────────────
        h_loc = self._encode(x, edge_index, edge_attr, self.loc_convs, self.loc_bns,
                             self.loc_res_projs if self.use_skip else None)
        s_i = self._node_suspicion(h_loc)  # [N]

        # ── Stage 2: Classification GNN with augmented node features ──────────
        x_aug = torch.cat([x, s_i.unsqueeze(-1)], dim=-1)  # [N, 774]
        h_cls = self._encode(x_aug, edge_index, edge_attr, self.cls_convs, self.cls_bns,
                             self.cls_res_projs if self.use_skip else None)

        # Focal flow: suspicion-weighted mean pool
        s_w = s_i.unsqueeze(-1)
        focal_emb = global_add_pool(h_cls * s_w, batch)
        focal_emb = focal_emb / global_add_pool(s_w, batch).clamp(min=1e-6)  # [B, 256]

        # Context flow: uniform mean pool
        context_emb = global_mean_pool(h_cls, batch)  # [B, 256]

        # ── Stage 3: Live LM ─────────────────────────────────────────────────
        if func_input_ids is not None:
            lm_emb = lm_pool(self.codebert, self._is_enc_dec, func_input_ids, func_attention_mask, matryoshka_dim=self._matryoshka_dim)
        else:
            lm_emb = torch.zeros(B, self._lm_dim, device=x.device)

        # ── Z_combined: trimodal fusion ───────────────────────────────────────
        z_combined = torch.cat([focal_emb, context_emb, lm_emb], dim=-1)  # [B, 1280]

        # ── MTL heads ─────────────────────────────────────────────────────────
        logit_binary = self.binary_head(z_combined)    # [B, 2]
        logit_group  = self.group_head(z_combined)     # [B, num_groups]

        # CWE head: conditioned on group probs when use_group_cond=True,
        # otherwise zeros (no group PT needed; avoids random-noise conditioning)
        if self.use_group_cond:
            group_probs = F.softmax(logit_group.detach(), dim=-1)
        else:
            group_probs = torch.zeros(z_combined.size(0), self.num_groups, device=x.device)
        cwe_in = torch.cat([z_combined, group_probs], dim=-1)       # [B, 1280+num_groups]
        logit_cwe = self.cwe_head(cwe_in)                           # [B, num_classes]

        # ── Stage 1 stmt scores (MIL / ranking) ──────────────────────────────
        stmt_scores = (
            self._statement_scores(h_loc, batch, node_line)
            if node_line is not None else None
        )

        return logit_cwe, logit_group, logit_binary, stmt_scores, z_combined
