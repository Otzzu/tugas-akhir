"""lmgat_interp.py — Arch6: GATv2Conv + live LM with learned λ interpolation."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import GATEncoder
from gnn_vuln.models.heads import SmallFuncHead, FuncHead, StmtHead

NODE_FEAT_DIM = 773

class LMGATInterpVulnDetector(VulnDetectorBase):
    def __init__(self, pretrained_lm="microsoft/unixcoder-base", func_lm="",
                 in_channels=NODE_FEAT_DIM, hidden_dim=256, num_layers=4,
                 dropout=0.3, num_classes=11, num_heads=4, edge_dim=7,
                 init_lambda=0.5, add_self_loops=False, use_skip=False,
                 matryoshka_dim=None):
        super().__init__()
        self._build_lm_branch(pretrained_lm, func_lm, matryoshka_dim)
        self.encoder  = GATEncoder(in_channels, hidden_dim, num_layers, num_heads, dropout, edge_dim, add_self_loops, use_skip)
        self.gnn_head = SmallFuncHead(hidden_dim, hidden_dim, num_classes, dropout)
        self.lm_head  = nn.Sequential(
            nn.Linear(self._lm_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.stmt_head   = StmtHead(hidden_dim)
        # Learned λ: sigmoid(lambda_logit) ∈ (0, 1)
        import math
        init_logit = math.log(init_lambda / (1.0 - init_lambda))
        self.lambda_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None):
        h = self.encoder(x, edge_index, edge_attr)
        logit_gnn = self.gnn_head(global_mean_pool(h, batch))

        B = logit_gnn.size(0)
        if func_input_ids is not None:
            # Handle optional 3D tensor (batch added externally)
            ids  = func_input_ids.squeeze(1)  if func_input_ids.dim() == 3  else func_input_ids
            mask = func_attention_mask.squeeze(1) if func_attention_mask is not None and func_attention_mask.dim() == 3 else func_attention_mask
            cls = self._lm_embed(ids, mask, B, x.device)
            logit_lm = self.lm_head(cls)
        else:
            logit_lm = torch.zeros_like(logit_gnn)

        lam = torch.sigmoid(self.lambda_logit)
        logit = lam * logit_gnn + (1.0 - lam) * logit_lm

        stmt_scores = self.stmt_head.score(h, batch, node_line) if node_line is not None else None
        return logit, stmt_scores

    @classmethod
    def from_config(cls, cfg, in_channels, **kwargs):
        pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/unixcoder-base")
        func_lm = getattr(cfg.model, "func_lm", "") or pretrained_lm
        return cls(
            pretrained_lm=pretrained_lm, func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            init_lambda=getattr(cfg.model, "init_lambda", 0.5),
            add_self_loops=getattr(cfg.model, "add_self_loops", False),
            use_skip=getattr(cfg.model, "use_skip", False),
            matryoshka_dim=getattr(cfg.model, "matryoshka_dim", None),
        )
