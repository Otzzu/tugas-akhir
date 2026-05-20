"""lmgat_codebert.py — Unified GATv2 + (optional) live LM.

live_lm modes:
  - none          : GNN only (replaces old lmgat). fused = h_graph. No LM forwards.
  - func          : func-level [CLS] (sliding window if func_chunk_size>0). Default.
  - func_and_line : func-level [CLS] for cls + per-line LM forward for localization
                    (EDAT-style line isolation). Reuses func_input_ids.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import build_gnn_encoder
from gnn_vuln.models.heads import FuncHead, ThinFuncHead, StmtHead
from gnn_vuln.models.cross_task import build_cross_task, statement_features, _LineLevelEncoder
from gnn_vuln.models._lm_utils import scatter_lines_to_tokens, _PERLINE_MAX_LINE

NODE_FEAT_DIM = 773

_VALID_LIVE_LM = ("none", "func", "func_and_line", "line")


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
                 mmoe_task_encoder=False, cross_task_residual=True,
                 mmoe_loc_transformer=False, live_lm="func",
                 gnn_model="gat", num_relations=7, num_bases=None,
                 codet5p_raw_encoder=False, codet5p_normalize_per_token=False,
                 normalize_gnn_output=False):
        super().__init__()
        self._normalize_gnn_output = normalize_gnn_output
        assert live_lm in _VALID_LIVE_LM, \
            f"live_lm must be one of {_VALID_LIVE_LM}, got {live_lm!r}"
        self._live_lm = live_lm
        # When no live LM, localization must be gnn-only (lm/both need LM hidden).
        if live_lm == "none":
            assert localization_encoder == "gnn", (
                f"live_lm='none' requires localization_encoder='gnn', "
                f"got {localization_encoder!r}. Live LM hidden states are unavailable."
            )
            assert cross_task_method == "none", (
                f"live_lm='none' requires cross_task_method='none', "
                f"got {cross_task_method!r}. Cross-task methods need LM features."
            )
            self._lm_dim = 0
        else:
            self._build_lm_branch(
                pretrained_lm, func_lm, matryoshka_dim,
                func_chunk_size, func_chunk_stride,
                use_flash_attention, compile_lm, use_grad_checkpoint,
                lm_per_line=(live_lm == "func_and_line"),
                codet5p_raw_encoder=codet5p_raw_encoder,
                codet5p_normalize_per_token=codet5p_normalize_per_token,
            )
        # Line-level transformer (live_lm=line): contextualizes per-line LM
        # embeddings across the function. Classification = meanmax pool of its
        # output; localization = its per-line output. No whole-function forward.
        self.line_encoder = (
            _LineLevelEncoder(self._lm_dim, self._lm_dim, num_layers=2, num_heads=num_heads)
            if live_lm == "line" else None
        )
        self._loc_enc = localization_encoder
        self.encoder = build_gnn_encoder(
            gnn_model, in_channels, hidden_dim, num_layers, dropout,
            num_heads=num_heads, edge_dim=edge_dim, add_self_loops=add_self_loops,
            use_skip=use_skip, num_relations=num_relations, num_bases=num_bases,
        )
        # Graph-level pooling: mean | meanmax | attention | dualflow
        assert graph_pool in ("mean", "meanmax", "attention", "dualflow"), \
            f"graph_pool must be mean|meanmax|attention|dualflow, got {graph_pool!r}"
        self._graph_pool = graph_pool
        self.attn_pool = (
            AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
            if graph_pool == "attention" else None
        )
        # dualflow: per-node suspicion head → focal (suspicion-weighted) + context (mean)
        self.node_susp = nn.Linear(hidden_dim, 1) if graph_pool == "dualflow" else None
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
            mmoe_loc_transformer=mmoe_loc_transformer,
        )

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None):
        h = self.encoder(x, edge_index, edge_attr)
        if self._graph_pool == "attention":
            h_graph = self.attn_pool(h, batch)
        elif self._graph_pool == "meanmax":
            h_graph = 0.8 * global_max_pool(h, batch) + 0.6 * global_mean_pool(h, batch)
        elif self._graph_pool == "dualflow":
            # focal: per-node suspicion-weighted pool + context: mean pool
            s = torch.sigmoid(self.node_susp(h))                      # [N, 1]
            focal = global_add_pool(h * s, batch) / global_add_pool(s, batch).clamp(min=1e-6)
            h_graph = focal + global_mean_pool(h, batch)
        else:
            h_graph = global_mean_pool(h, batch)
        B = h_graph.size(0)
        # Per-node GNN features for localization (optionally unit-normed, symmetric to F6 per_token norm).
        h_loc = F.normalize(h, dim=-1) if self._normalize_gnn_output else h
        # ── LM branch ─────────────────────────────────────────────────────────
        if self._live_lm == "none":
            # GNN-only: fused = h_graph. Skip all LM forwards, stmt head GNN-only.
            logit = self.func_head(h_graph)
            stmt_scores = (
                self.stmt_head.score(h_loc, batch, node_line)
                if node_line is not None else None
            )
            return logit, stmt_scores

        if self._live_lm == "line":
            # Hierarchical: per-line LM forward → line transformer (cross-line
            # context). Classification = meanmax pool; localization = per-line.
            # No whole-function forward — function length is unbounded.
            line_cls, uniq_sid, _, _ = self._lm_embed_per_line_raw(
                func_input_ids, func_token_lines,
            )
            line_graph = (uniq_sid // _PERLINE_MAX_LINE).long()
            line_ctx = self.line_encoder(line_cls, line_graph, B)        # [n, lm_dim]
            lm_emb = (0.8 * global_max_pool(line_ctx, line_graph, size=B)
                      + 0.6 * global_mean_pool(line_ctx, line_graph, size=B))
            lm_hidden = scatter_lines_to_tokens(
                line_ctx, uniq_sid, func_token_lines, B, func_input_ids.size(1),
            )
        elif self._loc_enc != "gnn":
            lm_emb, lm_hidden = self._lm_embed_full(
                func_input_ids, func_attention_mask, B, x.device, func_token_lines,
            )
        else:
            lm_emb = self._lm_embed(func_input_ids, func_attention_mask, B, x.device)
            lm_hidden = None
        if self._normalize_gnn_output:
            h_graph = F.normalize(h_graph, dim=-1)
        fused = torch.cat([h_graph, lm_emb], dim=-1)

        ct = self._cross_task_method
        if ct == "none" or node_line is None:
            logit = self.func_head(fused)
            stmt_scores = (
                self.stmt_head.score(h_loc, batch, node_line, lm_hidden, func_token_lines)
                if node_line is not None else None
            )
            return logit, stmt_scores

        # cross_attention | self_attention | mmoe — per-statement loc conditioning.
        # statement_features uses the SAME sid formula as StmtHead → cond [S,
        # loc_dim] aligns directly with StmtHead's statements.
        loc_feats, stmt_graph, _ = statement_features(
            h_loc, batch, node_line, lm_hidden, func_token_lines, self._loc_enc,
        )
        fused_mod, stmt_cond = self.cross_task(
            fused, loc_feats.detach(), stmt_graph, h_loc, batch, B,
            lm_hidden, func_token_lines,
        )
        logit = self.func_head(fused_mod)
        stmt_scores = self.stmt_head.score(h_loc, batch, node_line, lm_hidden, func_token_lines, cond=stmt_cond)
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
            mmoe_loc_transformer=getattr(cfg.model, "mmoe_loc_transformer", False),
            live_lm=getattr(cfg.model, "live_lm", "func"),
            gnn_model=getattr(cfg.model, "gnn_model", "gat"),
            num_relations=getattr(cfg.model, "num_relations", 7),
            num_bases=getattr(cfg.model, "num_bases", None),
            codet5p_raw_encoder=getattr(cfg.model, "codet5p_raw_encoder", False),
            codet5p_normalize_per_token=getattr(cfg.model, "codet5p_normalize_per_token", False),
            normalize_gnn_output=getattr(cfg.model, "normalize_gnn_output", False),
        )
