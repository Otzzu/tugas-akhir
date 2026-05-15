"""lmgat_codebert.py — Arch3: GATv2Conv + live LM (fine-tuned)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import GATEncoder
from gnn_vuln.models.heads import FuncHead, StmtHead
from gnn_vuln.models.cross_task import build_cross_task, loc_proto_pool

NODE_FEAT_DIM = 773

class LMGATCodeBERTVulnDetector(VulnDetectorBase):
    def __init__(self, pretrained_lm="microsoft/unixcoder-base", func_lm="",
                 in_channels=NODE_FEAT_DIM, hidden_dim=256, num_layers=4,
                 dropout=0.3, num_classes=11, num_heads=4, edge_dim=7,
                 add_self_loops=False, use_skip=False, matryoshka_dim=None,
                 func_chunk_size=0, func_chunk_stride=0,
                 localization_encoder="gnn", use_flash_attention=False, compile_lm=False,
                 use_grad_checkpoint=True,
                 stmt_both_mode="concat", stmt_lm_alpha=0.5,
                 cross_task_method="none", graph_pool="mean"):
        super().__init__()
        self._build_lm_branch(pretrained_lm, func_lm, matryoshka_dim, func_chunk_size, func_chunk_stride, use_flash_attention, compile_lm, use_grad_checkpoint)
        self._loc_enc = localization_encoder
        self.encoder   = GATEncoder(in_channels, hidden_dim, num_layers, num_heads, dropout, edge_dim, add_self_loops, use_skip)
        # Graph-level pooling: mean or gated attention
        assert graph_pool in ("mean", "attention"), \
            f"graph_pool must be mean|attention, got {graph_pool!r}"
        self._graph_pool = graph_pool
        self.attn_pool = (
            AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
            if graph_pool == "attention" else None
        )
        self.func_head = FuncHead(hidden_dim + self._lm_dim, hidden_dim, num_classes, dropout)
        lm_dim = self._lm_dim if localization_encoder in ("lm", "both") else 0
        self.stmt_head = StmtHead(hidden_dim, lm_dim=lm_dim, localization_encoder=localization_encoder,
                                  both_mode=stmt_both_mode, lm_alpha=stmt_lm_alpha)
        self._cross_task_method = cross_task_method
        self.cross_task = build_cross_task(
            cross_task_method, hidden_dim + self._lm_dim, hidden_dim, num_classes,
            self._lm_dim, localization_encoder, num_heads,
        )

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None):
        h = self.encoder(x, edge_index, edge_attr)
        h_graph = (
            self.attn_pool(h, batch) if self.attn_pool is not None
            else global_mean_pool(h, batch)
        )
        B = h_graph.size(0)
        if self._loc_enc != "gnn":
            lm_emb, lm_hidden = self._lm_embed_full(func_input_ids, func_attention_mask, B, x.device)
        else:
            lm_emb = self._lm_embed(func_input_ids, func_attention_mask, B, x.device)
            lm_hidden = None
        fused = torch.cat([h_graph, lm_emb], dim=-1)

        ct = self._cross_task_method
        if ct == "none" or node_line is None:
            logit = self.func_head(fused)
            stmt_scores = (
                self.stmt_head.score(h, batch, node_line, lm_hidden, func_token_lines)
                if node_line is not None else None
            )
            return logit, stmt_scores

        if ct == "direct":
            logit_base = self.func_head(fused)
            stmt_base  = self.stmt_head.score(h, batch, node_line, lm_hidden, func_token_lines)
            # loc → cls: per-graph stmt suspicion summary biases the logit
            stmt_summary = torch.stack([
                s.mean() if s.numel() > 0 else fused.new_zeros(()) for s in stmt_base
            ]).unsqueeze(1)                                              # [B, 1]
            logit = self.cross_task.cls_from_loc(logit_base, stmt_summary.detach())
            # cls → loc: vuln confidence gates the stmt scores
            vuln_conf = 1.0 - F.softmax(logit_base.detach(), dim=-1)[:, 0]   # [B]
            stmt_scores = self.cross_task.loc_from_cls(stmt_base, vuln_conf)
            return logit, stmt_scores

        # film | cross_attention | self_attention — feature-level conditioning
        loc_proto = loc_proto_pool(h, batch, node_line, lm_hidden, func_token_lines,
                                   self._loc_enc, B)
        if ct in ("cross_attention", "self_attention"):
            fused_mod, stmt_cond = self.cross_task(fused, loc_proto.detach(), h, batch, B,
                                                   lm_hidden, func_token_lines)
        else:  # film
            fused_mod, stmt_cond = self.cross_task(fused, loc_proto.detach())
        logit = self.func_head(fused_mod)
        stmt_scores = self.stmt_head.score(h, batch, node_line, lm_hidden, func_token_lines, cond=stmt_cond)
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
            stmt_both_mode=getattr(cfg.model, "stmt_both_mode", "concat"),
            stmt_lm_alpha=getattr(cfg.model, "stmt_lm_alpha", 0.5),
            cross_task_method=getattr(cfg.model, "cross_task_method", "none"),
            graph_pool=getattr(cfg.model, "graph_pool", "mean"),
        )
