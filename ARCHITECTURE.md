# Architecture Options

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ **CURRENT** | Currently implemented and in use |
| ✔ Implemented | Previously implemented, superseded |
| 🎯 Target | Next planned implementation |

---

## Architecture 1 — LM-GCN

**Status:** ✔ Implemented (superseded by LM-GAT v2)
**Config:** `configs/lmgcn/binary.yaml`, `configs/lmgcn/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgcn.py`

```
CPG nodes (773D pre-computed, frozen CodeBERT per node)
    → GCNConv × num_layers
    → BatchNorm + ReLU + Dropout
    ↓
    ├── global_mean_pool(h) → MLP → logit_func [B, num_classes]
    └── stmt_head: group nodes by line → max-pool + mean-pool per line
              → dual linear scorers → stmt_scores [n_stmts]

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores, y, k)
```

**Node features (773D):** `[node_type (1)] + [CodeBERT CLS per node (768)] + [dist_features (3)] + [danger_api (1)]`
**Edge features:** computed but ignored by GCNConv

**Limitations:**
- GCNConv treats all edge types equally (AST/CFG/PDG same weight)
- Frozen CodeBERT cannot adapt to vulnerability detection
- No full-function context

---

## Architecture 2 — LM-GAT v2 with Edge Features

**Status:** ✅ **CURRENT**
**Config:** `configs/lmgat/binary.yaml`, `configs/lmgat/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgat.py`

```
CPG nodes (773D pre-computed, frozen CodeBERT per node)
CPG edges (7D one-hot: AST/CFG/CDG/DDG/PDG/CALL/REACHING_DEF)
    → GATv2Conv × num_layers
        heads=4, concat=False → output stays 256D
        edge_dim=7: edge type shapes attention weights
        add_self_loops=True
    → BatchNorm + ReLU + Dropout
    ↓
    ├── global_mean_pool(h) [B, 256] → MLP → logit_func [B, num_classes]
    └── stmt_head: group nodes by source line
              max-pool(h_line) → Linear → s_max
              mean-pool(h_line) → Linear → s_mean
              stmt_score = 0.8 * s_max + 0.6 * s_mean → stmt_scores [n_stmts]

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores, y, k)
     + rank_loss_weight * RankingLoss(stmt_scores, flaw_line_mask)
```

**Key improvements over LM-GCN:**
- GATv2Conv: truly dynamic pair-specific attention (Brody et al., ICLR 2022)
- edge_dim=7: AST/CFG/PDG edges produce different attention weights
- Pairwise ranking loss: directly optimizes IFA and Effort@20%
- Inverse-frequency class weights: fixes CWE collapse on minority classes

**Limitations:**
- Frozen CodeBERT node features cannot adapt to vulnerability detection
- No full-function context (each node sees only its own 1–10 token snippet)
- func_head and stmt_head are independent paths — can disagree
- Optimizer: plain Adam, no gradient clipping, ReduceLROnPlateau

---

## Architecture 3 — LM-GAT v2 + Live Fine-tuned CodeBERT

**Status:** ✔ Implemented
**Config:** `configs/lmgat_codebert/binary.yaml`, `configs/lmgat_codebert/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgat_codebert.py`

```
Full function text (stored as input_ids/attention_mask in Data object)
    → CodeBERT (LIVE, FINE-TUNED, lr=2e-5)
    → CLS token [B, 768] ────────────────────────────────────────────┐
                                                                      ↓
CPG nodes (773D pre-computed, frozen — practical limitation)       concat [B, 256+768]
    → GATv2Conv × num_layers (lr=1e-3)                                ↓
    → BatchNorm + ReLU + Dropout                              MLP → logit_func [B, num_classes]
    ↓
    ├── global_mean_pool(h) [B, 256] ──────────────────────────────────┘
    └── stmt_head: group nodes by source line
              max-pool + mean-pool per line
              → dual scorers → stmt_scores [n_stmts]   (binary: suspicious or not)

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores, y, k)
     + rank_loss_weight * RankingLoss(stmt_scores, flaw_line_mask)
```

**Key design decisions:**
- Binary is treated as multiclass with num_classes=2 — one architecture for both modes
- Statement head always binary (suspicious or not); function head always multiclass
- CodeBERT fine-tuned for full-function context only — running CodeBERT per node during training is infeasible (100 nodes/graph × batch → too slow)
- Node features stay pre-computed (same limitation as current)
- Sync between stmt_scores and logit_func is implicit through shared GNN encoder h

**Training recipe changes from current:**
```python
# Two optimizer param groups
optimizer = AdamW([
    {"params": model.codebert.parameters(),      "lr": 2e-5, "weight_decay": 0.01},
    {"params": model.gnn_and_head_parameters(),  "lr": 1e-3, "weight_decay": 1e-4},
])
# Linear warmup + decay (replaces ReduceLROnPlateau)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=total_steps*0.1, ...)
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Data changes required:**
- `graph_builder_lm.py`: tokenize full function text, store `func_input_ids [1, 512]` and `func_attention_mask [1, 512]` in Data object
- `dataset_lm.py`: add CodeBERT tokenizer during `process()`, propagate new fields
- No CPG regeneration needed — only the Data object processing changes

**Model changes required:**
- `lmgat.py`: add `self.codebert`, update `func_head` input dim to `hidden_dim + 768`, update `forward` to accept and run `func_input_ids/attention_mask`
- `train.py`: AdamW param groups, linear warmup scheduler, gradient clipping, pass func tokens through `_forward`
- `evaluate.py`: pass `func_input_ids/mask` through evaluate loop

**Batch size (VRAM):**
| Mode | Current | With Live CodeBERT |
|------|---------|-------------------|
| binary (num_classes=2) | 16 | 8 |
| multiclass (num_classes=11) | 6 | 4 |

**Expected improvements:**
| Metric | LM-GAT v2 (current) | LM-GAT + Live CodeBERT |
|--------|--------------------|-----------------------|
| Function F1 (binary) | ~0.64 | ~0.80+ |
| IFA | ~8.78 | ~5–7 |
| Effort@20% | ~0.121 | ~0.10 |
| Training time/epoch | ~2 min | ~8–15 min |
| VRAM (multiclass) | ~1 GB | ~3–4 GB |

---

## Architecture 4 — LM-GAT v2 + Live CodeBERT + Multiclass Statement Head

**Status:** ✔ Implemented
**Config:** `configs/lmgat_mcs/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgat_mcs.py`

Extends Architecture 3 by making the statement head multiclass instead of binary,
then deriving function-level prediction directly from statement scores (WAVES-style strict sync).
No separate function head — function prediction is always mathematically derived from statement scores.

```
Full function text (input_ids/attention_mask in Data object)
    → CodeBERT (LIVE, FINE-TUNED, lr=2e-5)
    → CLS token [B, 768] ──────────────────────────────────────────────────────────┐
                                                                                    │
CPG nodes (773D pre-computed, frozen — practical limitation)                        │
    → GATv2Conv × num_layers (lr=1e-3)                                             │
    → BatchNorm + ReLU + Dropout                                                    │
    ↓                                                                               │
    stmt_head (MULTICLASS):                                                         │
        group nodes by source line                                                  │
        max-pool(h_line)  → Linear(256, num_classes) → score_max_j                 │
        mean-pool(h_line) → Linear(256, num_classes) → score_mean_j                │
        stmt_score_j = 0.8 * score_max_j + 0.6 * score_mean_j                      │
        → stmt_scores [n_stmts, num_classes]                                        │
              ↓                                                                     │
        max_pool(stmt_scores, dim=0) [num_classes] ─── concat ─────────────────────┘
                                                           ↓
                                                [num_classes + 768]
                                                           ↓
                                                    MLP → func_logit [B, num_classes]

Loss = CE(func_logit, y, class_weight)
     + mil_weight * MulticlassMIL(stmt_scores, y, k)
```

**Key design decisions:**

**Strict sync (WAVES-style):** `func_logit` is derived from `max_pool(stmt_scores)` + CodeBERT CLS.
Function prediction cannot disagree with statement scores by construction.

**Multiclass MIL pseudo-labels:**
- Benign function (label=0): top-k stmts by max-class score → pseudo-label = 0
- CWE-X function (label=c): top-k stmts → pseudo-label = c (same CWE as the function)
- All flaw lines in a function share the same CWE → pseudo-labels are coherent

**Function-level aggregation:** `max(stmt_scores, dim=0)` takes the highest confidence
per class across all statements. Combined with CodeBERT CLS for richer class discrimination.

**No separate func_head:** function prediction depends directly on statement scores
plus CodeBERT CLS. Removing the independent func_head eliminates the desync root cause entirely.

**Semantic note:** CWE is a function-level concept, but pushing it to statement level is
partially justified — some CWEs have characteristic line-level patterns in the CPG
(e.g. CWE-416: `free()` call, CWE-119: unbounded `memcpy`). For CWEs where the
vulnerability spans multiple lines, the GNN's message passing captures inter-line context.

**Comparison with Architecture 6:**

| | Arch 3 (separate func_head) | Arch 4 (stmt-derived func) |
|--|-----------------------------|-----------------------------|
| Func/stmt sync | Implicit (shared encoder) | Strict (func derived from stmt) |
| Statement output | Binary scalar (suspicious?) | Multiclass vector (which CWE?) |
| Localization info | "Line X is suspicious" | "Line X looks like CWE-119" |
| Training difficulty | Moderate | Harder (11-class MIL) |
| Convergence risk | Lower | Higher (minority CWE MIL) |
| No func_head needed | No | Yes — simpler model |

**Additional change vs Architecture 6:**
- `stmt_max_head` and `stmt_mean_head`: `Linear(hidden_dim, 1)` → `Linear(hidden_dim, num_classes)`
- MIL loss: binary BCE → multiclass CE
- Class weights applied in MIL loss too, not just func CE
- `func_head` input: `Linear(num_classes + 768, ...)` instead of `Linear(hidden_dim + 768, ...)`

---

## Architecture 5 — LM-GIN (Graph Isomorphism Network)

**Status:** ✔ Implemented
**Config:** `configs/lmgin/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgin.py`

```
CPG nodes (773D pre-computed, frozen CodeBERT per node)
CPG edges (7D one-hot)
    → edge_proj: Linear(7, 773) [first layer] / Linear(7, 256) [subsequent]
    → GINEConv × num_layers
        MLP((1+ε)*h_v + Σ_{u∈N(v)} (h_u + e_uv))  ← sum aggregation + edge features
    → BatchNorm + ReLU + Dropout
    ↓
    ├── global_mean_pool(h) → MLP → logit_func [B, num_classes]
    └── stmt_head: group nodes by source line
              max-pool + mean-pool per line → dual scorers → stmt_scores [n_stmts]

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores, y, k)
     + rank_loss_weight * RankingLoss(stmt_scores, flaw_line_mask)
```

**Key design decisions:**
- GINEConv = GIN + edge features: `h_u + e_uv` before aggregation (Hu et al. 2020)
- Sum aggregation: more expressive than mean (GCN) or weighted mean (GAT) for graph isomorphism
- Frozen CodeBERT: same as Arch 1/2 — no live fine-tuning, uses existing processed `.pt`
- Edge features projected to node dim before GINEConv (required by GINEConv API)

**Comparison with LM-GAT:**
| | LM-GAT (Arch 2) | LM-GIN (Arch 5) |
|--|-----------------|-----------------|
| Aggregation | Weighted mean (attention) | Sum |
| Edge handling | Attention weight shaped by edge | Edge added to neighbor before sum |
| Expressiveness | GATv2 dynamic attention | GIN: most expressive (Xu et al. 2019) |
| Interpretability | Attention weights visible | No attention weights |

---

## Architecture 6 — LM-GAT + VulLMGNN Explicit Interpolation

**Status:** ✔ Implemented
**Config:** `configs/lmgat_interp/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgat_interp.py`

Correct implementation of VulLMGNN explicit stage (Cao et al. 2022).
Two fully independent branches combined via learned interpolation weight λ.

```
Full function text (input_ids/attention_mask in Data object)
    → CodeBERT (LIVE, FINE-TUNED, lr=1e-5)
    → CLS token [B, 768]
    → lm_head MLP → logit_lm [B, num_classes]   ─────────────────────┐
                                                                        │ λ = sigmoid(λ_logit) learned
CPG nodes (773D pre-computed, frozen)                                   │
    → GATv2Conv × num_layers (lr=1e-3)                                 │
    → BatchNorm + ReLU + Dropout                                        │
    ↓                                                                   │
    global_mean_pool(h) → gnn_head MLP → logit_gnn [B, num_classes] ──┤
                                                                        ↓
                                                    λ * logit_gnn + (1-λ) * logit_lm
                                                         ↓
                                                 logit_func [B, num_classes]

    └── stmt_head (GNN branch only):
              group nodes by source line
              max-pool + mean-pool per line → dual scorers → stmt_scores [n_stmts]

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores, y, k)
     + rank_loss_weight * RankingLoss(stmt_scores, flaw_line_mask)
```

**Key design decisions:**
- λ is a learned scalar (sigmoid-bounded to (0,1)) — model learns how much to trust GNN vs LM
- init_lambda=0.5: equal weight at initialization, model adjusts during training
- Branches are fully independent: GNN and CodeBERT optimize toward the same logit space separately
- Statement head uses GNN embeddings only (structure-aware localization)
- Uses same `_ft` processed `.pt` as Arch 3 (requires `add_func_tokens: true`)

**Contrast with Arch 3 (lmgat_codebert):**
| | Arch 3 (concat) | Arch 6 (interpolation) |
|--|-----------------|------------------------|
| GNN + LM fusion | Concat → single MLP | Separate heads → λ-blend logits |
| Paper alignment | Wrong (not VulLMGNN) | Correct VulLMGNN explicit stage |
| Branch independence | Shared func_head couples branches | Fully independent |
| λ interpretable | No | Yes: how much model trusts GNN vs LM |

---

## Architecture 7 — LM-GAT-Seq (Sequential Localization → Classification)

**Status:** ✔ Implemented
**Config:** `configs/lmgat_seq/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgat_seq.py`

Two independent GATv2 stacks: Stage 1 produces a per-node suspicion score `s_i`, which augments
node features before Stage 2. Stage 2 combines suspicion-weighted GNN pooling + live LM for CWE
classification.

```
CPG nodes (773D pre-computed, frozen)
CPG edges (7D one-hot)

── Stage 1: Binary Localization ─────────────────────────────────────────────
    → GATv2Conv × num_layers (heads=4, concat=False) → h_loc [N, 256]
    → BatchNorm + ReLU + Dropout
    → binary stmt head: sigmoid(0.8*max_head(h_loc) + 0.6*mean_head(h_loc))
    → s_i ∈ [0,1] per node                                          [N]

── Stage 2: Classification ───────────────────────────────────────────────────
    stage2_base = h_loc (256D) if stage2_node_input="loc"
               or x_frozen (773D) if stage2_node_input="raw"
    x_aug = concat(stage2_base, s_i)                    [N, 257 or 774]
    → GATv2Conv × num_layers (heads=4, concat=False) → h_cls [N, 256]
    → BatchNorm + ReLU + Dropout
    → suspicion-weighted pool:
          graph_emb = global_add_pool(h_cls * s_i) / sum(s_i)     [B, 256]

Full function text → live LM → CLS token                          [B, 768]
    concat(graph_emb, lm_emb)                                      [B, 1024]
    → MLP → logit_func                                             [B, num_classes]

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores_stage1, y, k)          # binary BCE on s_i
     + rank_loss_weight * RankingLoss(stmt_scores_stage1, flaw_line_mask)
```

**Key design decisions:**
- Stage 1 and Stage 2 share the same graph — Stage 1 gradient flows into `s_i` during training
- `stage2_node_input="raw"`: Stage 2 gets full 773D features + noisy-but-informative s_i gradient
- `stage2_node_input="loc"`: Stage 2 gets compact 256D h_loc + s_i — smoother but loses raw signal
- Suspicion-weighted pooling: high-suspicion nodes contribute more to graph_emb than uniform mean
- No detachment: classification loss backpropagates through Stage 2 GNN; localization loss through Stage 1

**Experimental finding:**
- `stage2_node_input="raw"` (v1, original) outperforms `"loc"` (v2, tuned) on all metrics
- Raw noisy s_i gradient is more informative than clean but compressed h_loc features

| Run | Config | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260429_121124 | v1: stage2=raw, mil=0.3, rank=0.1, lr=0.001 | **0.4554** | **0.8610** | **7.34** | **0.356** | **0.0855** | **0.387** |
| 20260429_135046 | v2: stage2=loc, mil=0.1, rank=0.0, lr=0.0005 | 0.3857 | 0.8018 | 12.13 | 0.182 | 0.1177 | 0.294 |

---

## Architecture 8 — LM-GAT-WAVES-Seq (Transformer Localization → GATv2 + LM Classification)

**Status:** ✔ Implemented
**Config:** `configs/lmgat_waves_seq/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgat_waves_seq.py`

Stage 1 uses a Transformer encoder (WAVES-inspired) over CPG statement embeddings to produce
per-statement suspicion scores. Stage 2 uses GATv2 + live LM (VulLMGNN-style) for classification.
Stages are **decoupled** — s_i is detached before Stage 2.

```
CPG nodes (773D pre-computed, frozen)
CPG edges (7D one-hot)

── Stage 1: WAVES-style Transformer Localization ────────────────────────────
    Group nodes by source line → mean-pool CodeBERT slice (indices 1:769)
    per-stmt embeddings [n_stmts, 768]
    → TransformerEncoder(d_model=768, nhead=4, layers=2, dim_ff=1536)
    → Linear(768, 1) → sigmoid → suspicion [n_stmts]
    → broadcast to node level via node_line mapping
    s_i ∈ [0,1] per node (DETACHED — Stage 2 sees no Stage 1 gradient)     [N]

── Stage 2 GNN Branch ────────────────────────────────────────────────────────
    x_aug = concat(x_frozen, s_i)                              [N, 774]
    → GATv2Conv × num_layers (heads=4, concat=False) → h [N, 256]
    → BatchNorm + ReLU + Dropout
    → global_mean_pool(h) → gnn_emb                           [B, 256]

── Stage 2 LM Branch ─────────────────────────────────────────────────────────
    Full function text → live LM → CLS → lm_emb               [B, 768]

    concat(gnn_emb, lm_emb)                                    [B, 1024]
    → MLP → logit_func                                         [B, num_classes]

Loss_Stage1 = mil_weight * MIL(stmt_logits, y, k)
            + rank_loss_weight * RankingLoss(stmt_logits, flaw_line_mask)
Loss_Stage2 = CE(logit_func, y, class_weight)
Loss        = Loss_Stage1 + Loss_Stage2
```

**Key design decisions:**
- Transformer operates on statement-level (sequence of lines), not node-level — no CPG topology in Stage 1
- Detach prevents classification gradient from corrupting localization transformer
- VulLMGNN-style global_mean_pool in Stage 2 (vs suspicion-weighted pool in Arch7)

**Experimental finding:**
- Transformer localization fails on CPG: Top-1=0.096 (worst of all models), IFA=13.72
- Without CPG structure, transformer cannot identify vulnerable lines from CodeBERT embeddings alone
- Classification still reasonable (F1=0.4305) — Stage 2 GATv2+LM is competent despite bad s_i

| Run | Config | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260429_125637 | stmt_transformer_layers=2, heads=4 | 0.4305 | 0.8357 | 13.72 | 0.096 | 0.1394 | 0.245 |

---

## Architecture 9 — LM-GGNN (LM-GAT-CodeBERT with GATv2 → GatedGraphConv)

**Status:** ✔ Implemented
**Config:** `configs/lmggnn/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmggnn.py`

Arch3 (LM-GAT-CodeBERT) variant with GATv2Conv swapped for GatedGraphConv. The fusion strategy
also changed from concat → single MLP (Arch3) to two separate heads interpolated via fixed alpha,
which aligns with the BertGGCN design (Cao et al. ICSE 2023). GatedGraphConv requires equal
in/out dims so node features are projected before GGNN. Residual concat(h_gnn, h0) before
pooling preserves original projected features alongside GGNN-refined features.

**No statement head** — localization capability removed entirely (GGNN returns `(logit, None)`).

```
CPG nodes (773D pre-computed, frozen)
    → input_proj: Linear(773, 256)                               [N, 256]
    → GatedGraphConv(out=256, num_layers=6) → h                  [N, 256]
    → Dropout
    → global_mean_pool(h) → h_graph                             [B, 256]

Full function text → live LM → CLS → cls                        [B, 768]

    concat(h_graph, cls)                                         [B, 1024]
    → func_head MLP(1024 → 256 → num_classes) → logit_func       [B, num_classes]

    └── stmt_head: group nodes by source line
              max-pool(h_line) → Linear(256,1) → s_max
              mean-pool(h_line) → Linear(256,1) → s_mean
              stmt_score = 0.8 * s_max + 0.6 * s_mean → stmt_scores [n_stmts]

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores, y, k)
     + rank_loss_weight * RankingLoss(stmt_scores, flaw_line_mask)
```

**Differences from Arch3 (LM-GAT-CodeBERT):**
| | Arch3 | Arch9 (LM-GGNN) |
|--|-------|-----------------|
| GNN backbone | GATv2Conv × 4 | GatedGraphConv × 6 |
| Edge features | 7D one-hot, shapes attention | Not used (GGNN ignores edge_attr) |
| GNN + LM fusion | concat(pool, cls) → single MLP | **same** |
| Statement head | Binary MIL | **same** |
| BatchNorm | Yes (after each GATv2) | No (GGNN gating handles normalisation) |
| Node projection | None (GATv2 accepts 773D) | input_proj: 773D → 256D (GGNN requires in==out) |

**Key design decisions:**
- `input_proj` maps 773D → 256D before GGNN — GatedGraphConv requires `in_channels == out_channels`
- `**kwargs` in `__init__` absorbs deprecated `alpha` from old checkpoint configs
- No BatchNorm after GGNN — GRU gating inside GatedGraphConv provides normalisation

**Results (20260429_203915, old implementation without stmt_head — Dataset v2):**

| F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|
| 0.3519 | 0.8053 | N/A | N/A | N/A | N/A |

> Old run used interpolation fusion without stmt_head — needs retrain with corrected implementation.

---

## Architecture 10 — LM-GAT-DualFlow (Context-Preserving Sequential Router)

**Status:** 🎯 Target
**Config:** `configs/lmgat_dualflow/multiclass.yaml`
**Model:** `src/gnn_vuln/models/lmgat_dualflow.py`

Extends Arch7 v1 (sequential localization) by adding a second parallel pooling stream in Stage 2.
Arch7's suspicion-weighted pool isolates the flaw but discards safe-context nodes carrying
CWE-discriminating structural information. DualFlow restores this by concatenating a standard
`global_mean_pool` embedding alongside the focal embedding before the classifier.

```
CPG nodes (773D pre-computed, frozen)
CPG edges (7D one-hot)

── Stage 1: Binary Localization (identical to Arch7 v1) ─────────────────────
    → GATv2Conv × num_layers (heads=4, concat=False) → h_loc [N, 256]
    → BatchNorm + ReLU + Dropout
    → binary stmt head: sigmoid(0.8*max_head + 0.6*mean_head)
    → s_i ∈ [0,1] per node                                          [N]

── Stage 2: Dual-Flow Graph Encoding ─────────────────────────────────────────
    x_aug = concat(x_frozen, s_i)                                   [N, 774]
    → GATv2Conv × num_layers (heads=4, concat=False) → h_cls        [N, 256]
    → BatchNorm + ReLU + Dropout

    Flow A — focal_emb:   weighted_mean_pool(h_cls, s_i)            [B, 256]
    Flow B — context_emb: global_mean_pool(h_cls)                   [B, 256]

── Stage 3: Tri-Modal Fusion ─────────────────────────────────────────────────
    Full function text → live UniXcoder → CLS → lm_emb              [B, 768]

    concat(focal_emb, context_emb, lm_emb)                          [B, 1280]
    → MLP(1280 → 512 → num_classes) → logit_func                    [B, num_classes]

Loss = CE(logit_func, y, class_weight)
     + mil_weight * MIL(stmt_scores_stage1, y, k)
     + rank_loss_weight * RankingLoss(stmt_scores_stage1, flaw_line_mask)
```

**Design rationale vs Arch7 v1:**
| | Arch7 v1 | Arch10 DualFlow |
|--|----------|-----------------|
| Stage 1 | GATv2 binary localization | **identical** |
| Stage 2 GNN | GATv2(concat[x, s_i]) | **identical** |
| Stage 2 pooling | focal only (weighted mean) | focal + context (both) |
| Fusion input | [256 + 768] = 1024D | [256 + 256 + 768] = **1280D** |
| func_head | MLP(1024 → 256 → C) | MLP(1280 → 512 → C) |
| Localization loss | Stage 1 MIL + ranking | **identical** |

**Why localization should be preserved:**
Stage 1 and its MIL/ranking losses are unchanged — `s_i` gradient path is identical to Arch7 v1.
Adding `context_emb` to Stage 3 input does not affect Stage 1 parameters.

**Why F1 should improve over Arch7 v1:**
`focal_emb` concentrates on high-suspicion nodes; for complex CWEs (race condition, missing auth)
the classifier needs surrounding safe-context nodes to resolve the CWE category.
`context_emb` provides the full CPG structural layout, giving the MLP all three signals:
flaw shape (focal), code structure (context), semantic text (LM).

**Results:** pending (needs cloud training run).
