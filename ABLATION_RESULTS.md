# Ablation Results

Dataset: MegaVul Top-25 CWEs, max 1600 per class, 26 classes (25 CWE + benign),
UniXcoder-base embeddings, seed=42. GPU: RTX 5070 Ti.

Phase structure:
- **Phase 1 — Encoder & Localization**: live-LM vs frozen, localization encoder
- **Phase 2 — GNN+LM Localization Fusion**: how GNN + LM statement features combine
- **Phase 3 — Loss Function**: focal / epoch-adaptive / LIVABLE tuning
- **Phase 4 — Graph Pooling**: mean / attention / meanmax
- **Phase 5 — Multi-Task / Cross-Task**: bidirectional cross-task coupling

---

# Phase 1 — Encoder & Localization

`configs/ablation/phase1/` — varies architecture (frozen vs live LM) and
`localization_encoder` (GNN nodes / LM tokens / both). Shared loss config:
`focal_loss_gamma=2.0`, `epoch_adaptive_weights=true`, `patience=25`.

| Run | Run ID | Config | Architecture | Localization | LM Fine-tuned |
|-----|--------|--------|---|---|---|
| A1 | `20260513_202125_lmgat_multiclass` | `A1_lmgat.yaml` | lmgat (frozen) | GNN | No |
| A2 | `20260513_210613_lmgat_codebert_multiclass` | `A2_lmgat_codebert_gnn.yaml` | lmgat_codebert | GNN | Yes (lm_lr=1e-5) |
| A3 | `20260514_063713_lmgat_codebert_multiclass` | `A3_lmgat_codebert_lm.yaml` | lmgat_codebert | LM | Yes (lm_lr=1e-5) |
| A4 | `20260514_102721_lmgat_codebert_multiclass` | `A4_lmgat_codebert_both.yaml` | lmgat_codebert | GNN+LM | Yes (lm_lr=1e-5) |

## Function-Level Classification

| Run | Val F1 | Test F1 | Test Acc | AUC-ROC | Conf. mean | Epochs |
|-----|---|---|---|---|---|---|
| A1 | 0.458 | 0.471 | 0.510 | 0.884 | 0.765 | 55 |
| A2 | 0.532 | 0.494 | 0.500 | 0.907 | 0.698 | 54 |
| A3 | 0.548 | 0.495 | 0.517 | 0.913 | 0.801 | 76 |
| A4 | **0.550** | **0.504** | 0.507 | 0.899 | 0.813 | 74 |

## Val-Test F1 Gap

Both from the same best-val-F1 checkpoint. Gap = Val F1 − Test F1.

| Run | Val F1 | Test F1 | Gap | Gap % |
|-----|--------|---------|-----|-------|
| A1 | 0.458 | 0.471 | -0.013 | -2.8% |
| A2 | 0.532 | 0.494 | 0.038 | 7.1% |
| A3 | 0.548 | 0.495 | 0.053 | 9.7% |
| A4 | 0.550 | 0.504 | 0.046 | 8.4% |

A1 (frozen LM) has no gap; live-LM runs (A2-A4) show 7-10% — the live LM overfits.

## Statement-Level Localization

| Run | IFA ↓ | Top-1 ↑ | Top-3 ↑ | Top-5 ↑ | Top-10 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|-----|---|---|---|---|---|---|---|---|
| A1 | 1.49 | 0.804 | 0.914 | 0.941 | 0.966 | 0.195 | 0.394 | 0.052 |
| A2 | **0.89** | **0.874** | 0.936 | 0.959 | 0.977 | 0.217 | 0.401 | **0.039** |
| A3 | 1.33 | 0.818 | **0.939** | **0.969** | **0.988** | 0.197 | **0.451** | 0.052 |
| A4 | 1.26 | 0.794 | 0.917 | 0.959 | 0.978 | **0.207** | 0.431 | 0.047 |

GNN localization (A2) = precise (best IFA/Top-1). LM localization (A3) = coverage
(best R@20%LOC). Both (A4) = best classification F1, localization a compromise.

## Loss Dynamics (A1-A4)

Loss at best-F1 epoch. Test loss = unweighted CE recomputed from `predictions.csv`.
Val loss was weighted during training — scales differ, watch relative gap.

| Run | Best Epoch | Train Loss | Val Loss | Test Loss | Min Val Loss | Min VL Epoch |
|-----|---|---|---|---|---|---|
| A1 | 30 | 0.150 | 2.357 | 2.469 | 1.670 | 9 |
| A2 | 29 | 0.326 | 1.700 | 1.906 | 1.433 | 12 |
| A3 | 51 | 0.276 | 2.329 | 2.475 | 1.400 | 7 |
| A4 | 49 | 0.149 | 2.430 | 2.769 | 1.374 | 7 |

A3/A4 show ~44-epoch F1-loss divergence (val_loss min at ep 7, F1 peak at ep 49-51)
— symptom of the stacked loss, addressed in Phase 3.

**Phase 1 winner: A4 (localization=both).**

---

# Phase 2 — GNN+LM Localization Fusion

`configs/ablation/phase2/` — base A4 (localization=both). Varies `stmt_both_mode`
(how GNN + LM statement features combine). concat = baseline. weighted =
score-level `(1-α)·gnn + α·lm`. gated = per-statement learned gate.

| ID | Run ID | Config | stmt_both_mode |
|---|---|---|---|
| B0 | (= A4-L1 baseline, `20260514_174326`) | — | concat |
| B1 | `20260515_120709_lmgat_codebert_multiclass` | `B1_both_gated.yaml` | gated |
| B2 | `20260515_135412_lmgat_codebert_multiclass` | `B2_both_weighted_a03.yaml` | weighted (GNN-leaning) |
| B3 | `20260515_165955_lmgat_codebert_multiclass` | `B3_both_weighted_a05.yaml` | weighted (balanced) |
| B4 | `20260515_152942_lmgat_codebert_multiclass` | `B4_both_weighted_a07.yaml` | weighted (LM-leaning) |

| ID | Variant | Test F1 | Test Acc | F1-w | IFA ↓ | Top-1 ↑ | R@20%LOC ↑ |
|---|---|---|---|---|---|---|---|
| B0 | concat | **0.519** | 0.518 | 0.517 | 0.789 | 0.887 | 0.403 |
| B1 | gated | 0.483 | 0.526 | 0.525 | 1.138 | 0.851 | **0.422** |
| B2 | weighted α=0.3 | 0.480 | 0.533 | 0.533 | **0.644** | 0.876 | 0.414 |
| B3 | weighted α=0.5 | 0.518 | **0.539** | 0.538 | 1.007 | **0.890** | 0.400 |
| B4 | weighted α=0.7 | 0.515 | 0.515 | 0.514 | 0.947 | 0.832 | 0.357 |

No fusion beats concat on macro F1 (α=0.5 ties). weighted α=0.3 (GNN-leaning) →
best IFA — a localization-precision knob.

**Phase 2 winner: concat.**

---

# Phase 3 — Loss Function

`configs/ablation/phase3/` — fixes the stacked-loss problem from Phase 1.
Architecture held at A4 (localization=both, concat).

| Variant | Run ID | Loss Config |
|---|---|---|
| A4 (Phase 1 base) | `20260514_102721_lmgat_codebert_multiclass` | focal γ=2.0 + epoch_adaptive, wd=1e-4, patience=25 |
| A4-L1 | `20260514_174326_lmgat_codebert_multiclass` | no focal + epoch_adaptive + label_smoothing=0.1, wd=1e-3, cosine, patience=15 |
| A4-L2 | `20260515_052704_lmgat_codebert_multiclass` | LIVABLE two-branch (focal+LSCE), wd=1e-3, cosine, patience=15 |
| A4-L2-fixed | `20260515_084125_lmgat_codebert_multiclass` | A4-L2, no early stopping (full T-schedule) |

## Classification

| Variant | Val F1 | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. | Epochs |
|---|---|---|---|---|---|---|---|
| A4 | 0.550 | 0.504 | 0.507 | 0.503 | 0.899 | 0.813 | 74 |
| A4-L1 | **0.560** | **0.519** | 0.518 | 0.517 | **0.915** | **0.630** | 31 |
| A4-L2 | **0.561** | 0.475 | 0.540 | 0.526 | 0.904 | 0.757 | 43 |
| A4-L2-fixed | — | 0.497 | **0.550** | — | — | — | 75 |

## Localization

| Variant | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|
| A4 | 1.26 | 0.794 | 0.959 | 0.207 | 0.431 | 0.047 |
| A4-L1 | **0.789** | **0.887** | 0.965 | 0.238 | 0.403 | **0.031** |
| A4-L2 | 0.867 | 0.817 | 0.949 | **0.256** | 0.476 | 0.029 |
| A4-L2-fixed | 1.277 | — | — | — | **0.492** | — |

A4-L1 (drop focal, add label smoothing) → best Test F1 + best localization
precision + best calibration. A4-L2 (LIVABLE) → best accuracy but lower macro F1
(tail-class collapse — LIVABLE rebalances via focal branch only, no class-frequency
weighting). A4-L2-fixed (full T-schedule, no early stop) recovered macro F1
0.475→0.497 — still below A4-L1.

**Phase 3 winner: A4-L1 (no focal + label smoothing). Baseline for Phases 4-5.**

Note: 5 exploratory loss runs (`20260514_145017/160914/191041/214622/234550`,
focal-off + livable on/off probes on frozen + live LM) predate the clean A4-L1/L2
runs — superseded, kept for reference only.

---

# Phase 4 — Graph Pooling

`configs/ablation/phase4/` — base A4-L1 (Phase 3 winner). Varies `graph_pool`
(function classification representation): mean / gated attention / meanmax
(0.8·max + 0.6·mean).

| Variant | Run ID | graph_pool | Epochs |
|---|---|---|---|
| mean | (= A4-L1, `20260514_174326`) | mean | 31 |
| attention | `20260515_235912_lmgat_codebert_multiclass` | attention | 50 |
| meanmax | `20260516_125619_lmgat_codebert_multiclass` | meanmax | 48 |

| Variant | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@20%LOC ↑ |
|---|---|---|---|---|---|---|---|---|---|
| mean | **0.519** | 0.518 | 0.517 | **0.915** | 0.630 | 0.789 | 0.887 | 0.965 | 0.403 |
| attention | 0.437 | 0.522 | 0.523 | 0.895 | 0.625 | 1.253 | 0.805 | 0.943 | 0.439 |
| meanmax | 0.517 | **0.538** | **0.539** | 0.911 | 0.502 | **0.644** | **0.900** | **0.982** | **0.487** |

Attention pool collapses macro F1 (−0.082) — the learnable gate over-parameterizes
and overfits tail classes. Mean and meanmax tie on macro F1 (0.519 vs 0.517) — but
**meanmax wins everywhere else**: best accuracy, F1-w, and all localization metrics.
Parameter-free (the max channel sharpens the peak signal without the attention
gate's overfit).

**Phase 4 winner: meanmax.**

---

# Phase 5 — Multi-Task / Cross-Task

`configs/ablation/phase5/` — bidirectional cross-task between localization
(stmt_head) and classification (func_head). Zero-init residual gates
(ReZero/ControlNet style) — module starts as a no-op.
**Baseline E0 = A4-L1** (Phase 3 winner, no cross-task).

| ID | Run ID | Config | cross_task_method | Epochs |
|---|---|---|---|---|
| E0 | `20260514_174326_lmgat_codebert_multiclass` | — | none (= A4-L1 baseline) | 31 |
| E1 | `20260515_211228_lmgat_codebert_multiclass` | `E1_crossattn.yaml` | cross_attention | 58 |
| E2 | `20260516_055225_lmgat_codebert_multiclass` | `E2_selfattn.yaml` | self_attention | 63 |
| E3 | `20260516_101335_lmgat_codebert_multiclass` | `E3_mmoe.yaml` | mmoe | 40 |

(`E4_mmoe_taskenc.yaml`, `E5_mmoe_taskenc_thin.yaml` — pending runs.)

## Classification

| ID | Method | Val F1 | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. |
|---|---|---|---|---|---|---|---|
| E0 | none (A4-L1) | 0.560 | **0.519** | 0.518 | 0.517 | 0.915 | 0.630 |
| E1 | cross_attention | 0.526 | 0.468 | 0.507 | 0.505 | 0.909 | 0.462 |
| E2 | self_attention | 0.548 | 0.488 | **0.556** | **0.555** | **0.919** | 0.466 |
| E3 | mmoe | 0.553 | 0.497 | 0.541 | 0.541 | 0.907 | 0.625 |

## Localization

| ID | Method | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|---|
| E0 | none (A4-L1) | **0.789** | **0.887** | 0.965 | **0.238** | **0.403** | **0.031** |
| E1 | cross_attention | 1.309 | 0.830 | 0.962 | 0.086 | 0.197 | 0.205 |
| E2 | self_attention | 1.195 | 0.808 | **0.978** | 0.089 | 0.210 | 0.186 |
| E3 | mmoe | 0.837 | 0.873 | 0.965 | 0.151 | 0.273 | 0.106 |

Cross-task **attention** methods (E1, E2) hurt — localization collapsed (R@20%LOC
halved). MMOE (E3) is the least harmful — IFA near baseline, R@20%LOC above the
attention methods. Still: **no cross-task method beats the E0 baseline.** Per-statement
cross-task + line-level transformer encoder variants (E4, E5) pending re-run.

**Phase 5 winner (so far): E0 baseline — cross-task not yet beneficial.**

---

# Training Efficiency

| Run | Params | Epoch Time | Total Time (hr) | VRAM Peak |
|-----|--------|-----------|------------|-----------|
| A1 | 3.5M | 48s | 0.73 | 4.2 GB |
| A2 | 129.6M | 216s | 3.24 | 4.2 GB |
| A3 | 129.6M | 175s | 3.70 | 6.0 GB |
| A4 | 129.6M | 176s | 3.62 | 4.8 GB |
| A4-L1 | 129.6M | 161s | 1.39 | 6.9 GB |
| A4-L2 | 129.6M | 162s | 1.93 | 7.7 GB |
| A4-L2-fixed | 129.6M | 162s | 3.37 | — |
| fusion gated | 129.6M | 163s | 1.72 | — |
| fusion weighted α=0.3 | 129.6M | 162s | 1.53 | — |
| fusion weighted α=0.5 | 129.6M | 162s | 1.75 | — |
| fusion weighted α=0.7 | 129.6M | 162s | 1.44 | — |
| pool attention | 129.6M | 162s | 2.25 | 7.4 GB |
| pool meanmax | 129.6M | 162s | 2.16 | — |
| B2 cross_attn | 129.6M | 169s | 2.72 | — |
| B3 self_attn | 129.6M | 245s | 4.29 | — |
| B4 mmoe | 129.6M | 165s | 1.83 | — |
