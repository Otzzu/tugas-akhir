"""lmgat_codebert.py — Arch3: GATv2Conv + live LM (fine-tuned)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import GATEncoder
from gnn_vuln.models.heads import FuncHead, ThinFuncHead, StmtHead
from gnn_vuln.models.cross_task import build_cross_task, statement_features

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
                 cross_task_method="none", graph_pool="mean",
                 mmoe_task_encoder=False, cross_task_residual=True):
        super().__init__()
        self._build_lm_branch(pretrained_lm, func_lm, matryoshka_dim, func_chunk_size, func_chunk_stride, use_flash_attention, compile_lm, use_grad_checkpoint)
        self._loc_enc = localization_encoder
        self.encoder   = GATEncoder(in_channels, hidden_dim, num_layers, num_heads, dropout, edge_dim, add_self_loops, use_skip)
        # Graph-level pooling: mean | meanmax | attention
        assert graph_pool in ("mean", "meanmax", "attention"), \
            f"graph_pool must be mean|meanmax|attention, got {graph_pool!r}"
        self._graph_pool = graph_pool
        self.attn_pool = (
            AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
            if graph_pool == "attention" else None
        )
        # Thin head only for in-path MMOE (residual off + mmoe): MMOE's
        # task encoder + shared experts do the adaptation → head can be thin.
        # Attention methods don't carry that adaptation depth → keep fat head.
        _fused_dim = hidden_dim + self._lm_dim
        if cross_task_method == "mmoe" and not cross_task_residual:
            self.func_head = ThinFuncHead(_fused_dim, num_classes)
        else:
            self.func_head = FuncHead(_fused_dim, hidden_dim, num_classes, dropout)
        lm_dim = self._lm_dim if localization_encoder in ("lm", "both") else 0
        self.stmt_head = StmtHead(hidden_dim, lm_dim=lm_dim, localization_encoder=localization_encoder,
                                  both_mode=stmt_both_mode, lm_alpha=stmt_lm_alpha)
        self._cross_task_method = cross_task_method
        self.cross_task = build_cross_task(
            cross_task_method, hidden_dim + self._lm_dim, hidden_dim, num_classes,
            self._lm_dim, localization_encoder, num_heads,
            mmoe_task_encoder=mmoe_task_encoder, residual=cross_task_residual,
        )

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None):
        h = self.encoder(x, edge_index, edge_attr)
        if self._graph_pool == "attention":
            h_graph = self.attn_pool(h, batch)
        elif self._graph_pool == "meanmax":
            h_graph = 0.8 * global_max_pool(h, batch) + 0.6 * global_mean_pool(h, batch)
        else:
            h_graph = global_mean_pool(h, batch)
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

        # cross_attention | self_attention | mmoe — per-statement loc conditioning.
        # statement_features uses the SAME sid formula as StmtHead → cond [S,
        # loc_dim] aligns directly with StmtHead's statements.
        loc_feats, stmt_graph, _ = statement_features(
            h, batch, node_line, lm_hidden, func_token_lines, self._loc_enc,
        )
        fused_mod, stmt_cond = self.cross_task(
            fused, loc_feats.detach(), stmt_graph, h, batch, B,
            lm_hidden, func_token_lines,
        )
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
            mmoe_task_encoder=getattr(cfg.model, "mmoe_task_encoder", False),
            cross_task_residual=getattr(cfg.model, "cross_task_residual", True),
            graph_pool=getattr(cfg.model, "graph_pool", "mean"),
        )
