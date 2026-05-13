"""lmgat_codebert.py — Arch3: GATv2Conv + live LM (fine-tuned)."""
from __future__ import annotations
import torch
import torch.nn as nn
from gnn_vuln.models._lm_utils import det_global_mean_pool as global_mean_pool
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import GATEncoder
from gnn_vuln.models.heads import FuncHead, StmtHead

NODE_FEAT_DIM = 773

class LMGATCodeBERTVulnDetector(VulnDetectorBase):
    def __init__(self, pretrained_lm="microsoft/unixcoder-base", func_lm="",
                 in_channels=NODE_FEAT_DIM, hidden_dim=256, num_layers=4,
                 dropout=0.3, num_classes=11, num_heads=4, edge_dim=7,
                 add_self_loops=False, use_skip=False, matryoshka_dim=None,
                 func_chunk_size=0, func_chunk_stride=0,
                 localization_encoder="gnn", use_flash_attention=False, compile_lm=False,
                 use_grad_checkpoint=True):
        super().__init__()
        self._build_lm_branch(pretrained_lm, func_lm, matryoshka_dim, func_chunk_size, func_chunk_stride, use_flash_attention, compile_lm, use_grad_checkpoint)
        self._loc_enc = localization_encoder
        self.encoder   = GATEncoder(in_channels, hidden_dim, num_layers, num_heads, dropout, edge_dim, add_self_loops, use_skip)
        self.func_head = FuncHead(hidden_dim + self._lm_dim, hidden_dim, num_classes, dropout)
        lm_dim = self._lm_dim if localization_encoder in ("lm", "both") else 0
        self.stmt_head = StmtHead(hidden_dim, lm_dim=lm_dim, localization_encoder=localization_encoder)

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None):
        h = self.encoder(x, edge_index, edge_attr)
        h_graph = global_mean_pool(h, batch)
        if self._loc_enc != "gnn":
            lm_emb, lm_hidden = self._lm_embed_full(func_input_ids, func_attention_mask, h_graph.size(0), x.device)
        else:
            lm_emb = self._lm_embed(func_input_ids, func_attention_mask, h_graph.size(0), x.device)
            lm_hidden = None
        logit = self.func_head(torch.cat([h_graph, lm_emb], dim=-1))
        stmt_scores = (
            self.stmt_head.score(h, batch, node_line, lm_hidden, func_token_lines)
            if node_line is not None else None
        )
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
            add_self_loops=getattr(cfg.model, "add_self_loops", False),
            use_skip=getattr(cfg.model, "use_skip", False),
            matryoshka_dim=getattr(cfg.model, "matryoshka_dim", None),
            func_chunk_size=getattr(cfg.model, "func_chunk_size", 0),
            func_chunk_stride=getattr(cfg.model, "func_chunk_stride", 0),
            localization_encoder=getattr(cfg.model, "localization_encoder", "gnn"),
            use_flash_attention=getattr(cfg.train, "use_flash_attention", False),
            compile_lm=getattr(cfg.train, "compile_lm", False),
            use_grad_checkpoint=getattr(cfg.model, "use_grad_checkpoint", True),
        )
