"""lmgin.py — LM-GIN: GINEConv + frozen node embeddings."""
from __future__ import annotations
import torch
import torch.nn as nn
from gnn_vuln.models._lm_utils import det_global_mean_pool as global_mean_pool
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import GINEncoder
from gnn_vuln.models.heads import SmallFuncHead, StmtHead

NODE_FEAT_DIM = 773

class LMGINVulnDetector(VulnDetectorBase):
    def __init__(self, in_channels=NODE_FEAT_DIM, hidden_dim=256, num_layers=4,
                 dropout=0.3, num_classes=11, edge_dim=7, use_skip=False):
        super().__init__()
        self.encoder   = GINEncoder(in_channels, hidden_dim, num_layers, dropout, edge_dim, use_skip)
        self.func_head = SmallFuncHead(hidden_dim, hidden_dim, num_classes, dropout)
        self.stmt_head = StmtHead(hidden_dim)

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None):
        h = self.encoder(x, edge_index, edge_attr)
        logit = self.func_head(global_mean_pool(h, batch))
        stmt_scores = self.stmt_head.score(h, batch, node_line) if node_line is not None else None
        return logit, stmt_scores

    @classmethod
    def from_config(cls, cfg, in_channels, **kwargs):
        return cls(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            use_skip=getattr(cfg.model, "use_skip", False),
        )
