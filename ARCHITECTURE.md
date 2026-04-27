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
