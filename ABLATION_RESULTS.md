# Ablation Results

## Phase 1 — Setup

Dataset: MegaVul Top-25 CWEs, max 1600 per class, 26 classes (25 CWE + benign), UniXcoder-base embeddings, seed=42.
Shared: `focal_loss_gamma=2.0`, `epoch_adaptive_weights=true`, `use_class_weights=true`, `patience=25`, `early_stop_metric=f1`.

| Run | Run ID | Config | Architecture | Localization | Batch (eff.) | LM Fine-tuned |
|-----|--------|--------|---|---|---|---|
| A1 | `20260513_202125_lmgat_multiclass` | `A1_lmgat.yaml` | lmgat (frozen) | GNN | 32 (16×2) | No |
| A2 | `20260513_210613_lmgat_codebert_multiclass` | `A2_lmgat_codebert_gnn.yaml` | lmgat_codebert | GNN | 32 (8×4) | Yes (lm_lr=1e-5) |
| A3 | `20260514_063713_lmgat_codebert_multiclass` | `A3_lmgat_codebert_lm.yaml` | lmgat_codebert | LM | 32 (16×2) | Yes (lm_lr=1e-5) |
| A4 | `20260514_102721_lmgat_codebert_multiclass` | `A4_lmgat_codebert_both.yaml` | lmgat_codebert | GNN+LM | 32 (16×2) | Yes (lm_lr=1e-5) |

---

## Function-Level Classification

All metrics measured on the best-val-F1 checkpoint.

| Run | Val F1 | Test F1 | Test Acc | AUC-ROC | Conf. mean | Epochs |
|-----|---|---|---|---|---|---|
| A1 | 0.458 | 0.471 | 0.510 | 0.884 | 0.765 | 55 |
| A2 | 0.532 | 0.494 | 0.500 | 0.907 | 0.698 | 54 |
| A3 | 0.548 | 0.495 | 0.517 | 0.913 | 0.801 | 76 |
| A4 | **0.550** | **0.504** | 0.507 | 0.899 | 0.813 | 74 |

## Val-Test F1 Gap

Both Val F1 and Test F1 come from the same checkpoint (best-val-F1 epoch). Gap = Val F1 − Test F1.

| Run | Val F1 | Test F1 | Gap | Gap % |
|-----|--------|---------|-----|-------|
| A1 | 0.458 | 0.471 | -0.013 | -2.8% |
| A2 | 0.532 | 0.494 | 0.038 | 7.1% |
| A3 | 0.548 | 0.495 | 0.053 | 9.7% |
| A4 | 0.550 | 0.504 | 0.046 | 8.4% |

## Statement-Level Localization

| Run | IFA ↓ | Top-1 ↑ | Top-3 ↑ | Top-5 ↑ | Top-10 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|-----|---|---|---|---|---|---|---|---|
| A1 | 1.49 | 0.804 | 0.914 | 0.941 | 0.966 | 0.195 | 0.394 | 0.052 |
| A2 | **0.89** | **0.874** | 0.936 | 0.959 | 0.977 | 0.217 | 0.401 | **0.039** |
| A3 | 1.33 | 0.818 | **0.939** | **0.969** | **0.988** | 0.197 | **0.451** | 0.052 |
| A4 | 1.26 | 0.794 | 0.917 | 0.959 | 0.978 | **0.207** | 0.431 | 0.047 |

## Loss Dynamics

Loss at best-F1 epoch. Test loss = unweighted CE recomputed from `predictions.csv`. Val loss was weighted during training — scales differ, watch relative gap.

| Run | Best Epoch | Train Loss | Val Loss | Test Loss | Min Val Loss | Min VL Epoch |
|-----|---|---|---|---|---|---|
| A1 | 30 | 0.150 | 2.357 | 2.469 | 1.670 | 9 |
| A2 | 29 | 0.326 | 1.700 | 1.906 | 1.433 | 12 |
| A3 | 51 | 0.276 | 2.329 | 2.475 | 1.400 | 7 |
| A4 | 49 | 0.149 | 2.430 | 2.769 | 1.374 | 7 |

### Gap Signals

| Run | Train/Val Ratio | Val→Test Gap | F1-Loss Divergence (epochs) | Val Loss Drift (best − min) |
|-----|---|---|---|---|
| A1 | 0.064 | +0.112 | 21 (9→30) | +0.687 |
| A2 | 0.192 | +0.206 | 17 (12→29) | +0.267 |
| A3 | 0.118 | +0.146 | **44** (7→51) | +0.929 |
| A4 | 0.061 | **+0.339** | **42** (7→49) | **+1.056** |

**How to read:**

- **Train/Val Ratio < 0.1** → train memorized
- **Val→Test Gap > 0.2** → val-set bias
- **F1-Loss Divergence > 30 epochs** → loss config wrong
- **Val Loss Drift > 0.8** → past calibration peak

## Loss Function Ablation (A4)

Same architecture (lmgat_codebert, localization=both) — only loss config varies.

| Variant | Run ID | Loss Config |
|---|---|---|
| A4 | `20260514_102721_lmgat_codebert_multiclass` | focal γ=2.0 + epoch_adaptive, wd=1e-4, patience=25 |
| A4-L1 | `20260514_174326_lmgat_codebert_multiclass` | no focal + epoch_adaptive + label_smoothing=0.1, wd=1e-3, cosine, patience=15 |
| A4-L2 | `20260515_052704_lmgat_codebert_multiclass` | LIVABLE two-branch (focal+LSCE), wd=1e-3, cosine, patience=15 (early-stopped — T schedule cut short) |

### Classification

| Variant | Val F1 | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. mean | Epochs |
|---|---|---|---|---|---|---|---|
| A4 | 0.550 | 0.504 | 0.507 | 0.503 | 0.899 | 0.813 | 74 |
| A4-L1 | **0.560** | **0.519** | 0.518 | 0.517 | **0.915** | **0.630** | 31 |
| A4-L2 | **0.561** | 0.475 | **0.540** | **0.526** | 0.904 | 0.757 | 43 |

### Localization

| Variant | IFA ↓ | Top-1 ↑ | Top-3 ↑ | Top-5 ↑ | Top-10 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|---|---|
| A4 | 1.26 | 0.794 | 0.917 | 0.959 | 0.978 | 0.207 | 0.431 | 0.047 |
| A4-L1 | **0.789** | **0.887** | **0.955** | 0.965 | 0.984 | 0.238 | 0.403 | 0.031 |
| A4-L2 | 0.867 | 0.817 | 0.928 | 0.949 | 0.980 | **0.256** | **0.476** | **0.029** |

A4-L1 (drop focal) → best Test F1 + best localization precision + best calibration.
A4-L2 (LIVABLE) → best Test Acc + best localization coverage; lower macro F1 (tail-class collapse — LIVABLE has no class-frequency weighting). Note: early-stopped at epoch 43/100, so the T time-shift never completed its tail→head curriculum.

### LIVABLE — early-stop vs fixed-epoch

LIVABLE T-schedule needs the full run. A4-L2-fixed disables early stopping.

| Variant | Run ID | Epochs | Test F1 | Test Acc | IFA ↓ | R@20%LOC ↑ |
|---|---|---|---|---|---|---|
| A4-L2 (early-stop) | `20260515_052704_lmgat_codebert_multiclass` | 43 | 0.475 | 0.540 | 0.867 | 0.476 |
| A4-L2-fixed | `20260515_084125_lmgat_codebert_multiclass` | 75 | **0.497** | **0.550** | 1.277 | **0.492** |

Full schedule recovered macro F1 (0.475→0.497) — tail classes got the late head-focus phase. Still below A4-L1's 0.519.

## Localization Fusion Ablation (A4 both, epoch_adaptive loss)

`localization_encoder=both` — varies how GNN + LM statement features combine (`stmt_both_mode`).
Baseline = A4-L1 (concat). weighted = score-level `(1-α)·gnn + α·lm`. gated = per-statement learned gate.

| Variant | Run ID | stmt_both_mode | Epochs |
|---|---|---|---|
| concat | `20260514_174326_lmgat_codebert_multiclass` | concat | 31 |
| gated | `20260515_120709_lmgat_codebert_multiclass` | gated | 38 |
| weighted α=0.3 | `20260515_135412_lmgat_codebert_multiclass` | weighted (GNN-leaning) | 34 |
| weighted α=0.5 | `20260515_165955_lmgat_codebert_multiclass` | weighted (balanced) | 39 |
| weighted α=0.7 | `20260515_152942_lmgat_codebert_multiclass` | weighted (LM-leaning) | 32 |

### Classification

| Variant | Val F1 | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. mean |
|---|---|---|---|---|---|---|
| concat | 0.560 | **0.519** | 0.518 | 0.517 | 0.915 | 0.630 |
| gated | 0.542 | 0.483 | 0.526 | 0.525 | 0.895 | 0.643 |
| weighted α=0.3 | 0.559 | 0.480 | 0.533 | 0.533 | 0.909 | 0.646 |
| weighted α=0.5 | 0.539 | 0.518 | **0.539** | 0.538 | 0.916 | 0.662 |
| weighted α=0.7 | 0.561 | 0.515 | 0.515 | 0.514 | **0.919** | 0.635 |

### Localization

| Variant | IFA ↓ | Top-1 ↑ | Top-3 ↑ | Top-5 ↑ | Top-10 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|---|---|
| concat | 0.789 | 0.887 | 0.955 | 0.965 | 0.984 | 0.238 | 0.403 | 0.031 |
| gated | 1.138 | 0.851 | 0.934 | **0.966** | 0.980 | 0.230 | **0.422** | 0.033 |
| weighted α=0.3 | **0.644** | 0.876 | 0.950 | 0.965 | **0.985** | **0.241** | 0.414 | **0.029** |
| weighted α=0.5 | 1.007 | **0.890** | 0.944 | 0.959 | 0.975 | 0.228 | 0.400 | 0.032 |
| weighted α=0.7 | 0.947 | 0.832 | 0.939 | 0.960 | 0.981 | 0.211 | 0.357 | 0.044 |

No fusion beats concat on macro F1 (α=0.5 ties). Fusion lifts accuracy (α=0.5 → 0.539). weighted α=0.3 (GNN-leaning) → best IFA 0.644 — a localization-precision knob. gated underperforms. Concat = best all-rounder for classification + localization.

## Graph Pooling Ablation (A4 both, epoch_adaptive loss)

`localization_encoder=both`, `stmt_both_mode=concat` — varies `graph_pool` (function
classification representation): mean pool vs gated attention pool.

| Variant | Run ID | graph_pool | Epochs |
|---|---|---|---|
| mean | `20260514_174326_lmgat_codebert_multiclass` | mean | 31 |
| attention | `20260515_235912_lmgat_codebert_multiclass` | attention | 50 |

| Variant | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@20%LOC ↑ |
|---|---|---|---|---|---|---|---|---|---|
| mean | **0.519** | 0.518 | 0.517 | **0.915** | 0.630 | **0.789** | **0.887** | 0.965 | 0.403 |
| attention | 0.437 | 0.522 | 0.523 | 0.895 | 0.625 | 1.253 | 0.805 | 0.943 | **0.439** |

Attention pool collapses macro F1 (−0.082) while weighted-F1/accuracy stay flat —
the learnable gate over-parameterizes and overfits tail classes (few samples).
Localization also worse (IFA 0.789→1.253). Mean pool (parameter-free) retained.

## Phase 2 — Cross-Task Ablation (A4 both concat, epoch_adaptive loss)

Bidirectional cross-task between localization (stmt_head) and classification
(func_head). Zero-init residual gates (ReZero/ControlNet style) — module starts
as a no-op, baseline-equivalent at init.
**Baseline B1 = A4-L1** (`20260514_174326`, the best Phase 1 model: A4 both concat,
epoch_adaptive loss, no cross-task).

| ID | Run ID | cross_task_method | Epochs |
|---|---|---|---|
| B1 | `20260514_174326_lmgat_codebert_multiclass` | none (= A4-L1 baseline) | 31 |
| B2 | `20260515_211228_lmgat_codebert_multiclass` | cross_attention | 58 |
| B3 | `20260516_055225_lmgat_codebert_multiclass` | self_attention | 63 |
| B4 | `20260516_101335_lmgat_codebert_multiclass` | mmoe | 40 |

### Classification

| ID | Method | Val F1 | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. |
|---|---|---|---|---|---|---|---|
| B1 | none (A4-L1) | 0.560 | **0.519** | 0.518 | 0.517 | 0.915 | 0.630 |
| B2 | cross_attention | 0.526 | 0.468 | 0.507 | 0.505 | 0.909 | 0.462 |
| B3 | self_attention | 0.548 | 0.488 | **0.556** | **0.555** | **0.919** | 0.466 |
| B4 | mmoe | 0.553 | 0.497 | 0.541 | 0.541 | 0.907 | 0.625 |

### Localization

| ID | Method | IFA ↓ | Top-1 ↑ | Top-3 ↑ | Top-5 ↑ | Top-10 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|---|---|---|
| B1 | none (A4-L1) | **0.789** | **0.887** | 0.955 | 0.965 | 0.984 | **0.238** | **0.403** | **0.031** |
| B2 | cross_attention | 1.309 | 0.830 | 0.943 | 0.962 | 0.978 | 0.086 | 0.197 | 0.205 |
| B3 | self_attention | 1.195 | 0.808 | 0.944 | **0.978** | **0.991** | 0.089 | 0.210 | 0.186 |
| B4 | mmoe | 0.837 | 0.873 | 0.950 | 0.965 | 0.981 | 0.151 | 0.273 | 0.106 |

Both cross-task **attention** methods (B2, B3) hurt — localization collapsed
(R@20%LOC halved, 0.403→0.20), the `stmt_cond` conditioning corrupts per-statement
features. cross_attention worse everywhere; self_attention trades tail for head
(acc/F1-w up, macro F1 down).

**MMOE (B4) is the least harmful** — IFA 0.837 (near baseline 0.789, vs 1.2-1.3 for
attention), R@20%LOC 0.273 (still down from 0.403 but far above the attention
methods' 0.20). Shared-expert routing degrades localization less than feature
conditioning. Still: macro F1 0.497 < baseline 0.519, R@20%LOC below baseline —
**no cross-task method beats the baseline.** Zero-init gate prevented divergence
across all three but couldn't make coupling helpful.

## Training Efficiency

GPU: RTX 5070 Ti

| Run | Params | Epoch Time | Total Time (hr) | VRAM Peak |
|-----|--------|-----------|------------|-----------|
| A1 | 3.5M | 48s | 0.73 | 4.2 GB |
| A2 | 129.6M | 216s | 3.24 | 4.2 GB |
| A3 | 129.6M | 175s | 3.70 | 6.0 GB |
| A4 | 129.6M | 176s | 3.62 | 4.8 GB |
| A4-L1 | 129.6M | 161s | 1.39 | 6.9 GB |
| A4-L2 | 129.6M | 162s | 1.93 | 7.7 GB |
| A4-L2-fixed | 129.6M | 162s | 3.37 | — |
| gated | 129.6M | 163s | 1.72 | — |
| weighted α=0.3 | 129.6M | 162s | 1.53 | — |
| weighted α=0.5 | 129.6M | 162s | 1.75 | — |
| weighted α=0.7 | 129.6M | 162s | 1.44 | — |
| attn pool | 129.6M | 162s | 2.25 | 7.4 GB |
| B2 cross_attn | 129.6M | 169s | 2.72 | — |
| B3 self_attn | 129.6M | 245s | 4.29 | — |
| B4 mmoe | 129.6M | 165s | 1.83 | — |
