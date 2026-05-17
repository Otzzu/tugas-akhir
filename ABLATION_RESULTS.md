# Ablation Results

Dataset: MegaVul Top-25 CWEs, max 1600 per class, 26 classes (25 CWE + benign),
UniXcoder-base embeddings, seed=42. GPU: RTX 5070 Ti.

Phase structure:
- **Phase 1 — Encoder & Localization**: live-LM vs frozen, localization encoder
- **Phase 2 — GNN+LM Localization Fusion**: how GNN + LM statement features combine
- **Phase 3 — Loss Function**: focal / epoch-adaptive / LIVABLE tuning
- **Phase 4 — Graph Pooling**: mean / attention / meanmax / dualflow
- **Phase 5 — Multi-Task / Cross-Task**: bidirectional cross-task coupling
- **Phase 6 — Language Model**: node_lm / func_lm choice (UniXcoder / CodeT5+)

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
(0.8·max + 0.6·mean) / dualflow (suspicion-weighted focal + mean context).

| Variant | Run ID | graph_pool | Epochs |
|---|---|---|---|
| mean | (= A4-L1, `20260514_174326`) | mean | 31 |
| attention | `20260515_235912_lmgat_codebert_multiclass` | attention | 50 |
| meanmax | `20260516_125619_lmgat_codebert_multiclass` | meanmax | 48 |
| dualflow | `20260517_013824_lmgat_codebert_multiclass` | dualflow | 38 |

| Variant | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@20%LOC ↑ |
|---|---|---|---|---|---|---|---|---|---|
| mean | **0.519** | 0.518 | 0.517 | **0.915** | 0.630 | 0.789 | 0.887 | 0.965 | 0.403 |
| attention | 0.437 | 0.522 | 0.523 | 0.895 | 0.625 | 1.253 | 0.805 | 0.943 | 0.439 |
| meanmax | 0.517 | **0.538** | **0.539** | 0.911 | 0.502 | **0.644** | **0.900** | **0.982** | **0.487** |
| dualflow | 0.496 | 0.528 | 0.528 | 0.896 | 0.667 | 0.717 | 0.886 | 0.971 | 0.417 |

Attention pool collapses macro F1 (−0.082) — the learnable gate over-parameterizes
and overfits tail classes. Mean and meanmax tie on macro F1 (0.519 vs 0.517) — but
**meanmax wins everywhere else**: best accuracy, F1-w, and all localization metrics.
Parameter-free (the max channel sharpens the peak signal without the attention
gate's overfit). dualflow (learned per-node suspicion → focal + context) lands
mid-pack: macro F1 0.496 (below both mean and meanmax), localization between mean
and meanmax — its learned suspicion gate carries the same overfit risk as the
attention gate, just milder. Parameter-free meanmax still wins.

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
| E1 | `20260516_213244_lmgat_codebert_multiclass` | `E1_crossattn.yaml` | cross_attention | 31 |
| E2 | `20260516_185818_lmgat_codebert_multiclass` | `E2_selfattn.yaml` | self_attention | 55 |
| E3 | — | `E3_mmoe.yaml` | mmoe | _pending_ |
| E4 | `20260516_152322_lmgat_codebert_multiclass` | `E4_mmoe_taskenc.yaml` | mmoe + task encoder | 40 |
| E5 | `20260516_171751_lmgat_codebert_multiclass` | `E5_mmoe_taskenc_thin.yaml` | mmoe + task encoder + thin head | 35 |
| E6 | `20260517_042939_lmgat_codebert_multiclass` | `E6_crossattn_noresidual.yaml` | cross_attention, residual off | 92 |
| E7 | `20260517_084804_lmgat_codebert_multiclass` | `E7_selfattn_noresidual.yaml` | self_attention, residual off | 58 |

> **Earlier cross-task results removed.** A prior set of E1/E2/E3 runs was
> trained **before the per-statement line-level cross-task code was correct**
> (they collapsed the localization view to a per-graph vector instead of
> conditioning each statement). Those metrics were invalid and deleted.
> Invalidated run IDs (kept on disk, do not report): `20260515_211228`,
> `20260516_055225`, `20260516_101335`.
> The E1/E2/E4/E5 results below are from corrected-code runs.

## Classification

| ID | Method | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. |
|---|---|---|---|---|---|---|
| E0 | none (A4-L1) | 0.519 | 0.518 | 0.517 | 0.915 | 0.630 |
| E1 | cross_attention | **0.530** | 0.532 | 0.533 | **0.919** | 0.615 |
| E2 | self_attention | 0.504 | **0.538** | **0.537** | 0.897 | 0.606 |
| E4 | mmoe + task encoder | 0.479 | 0.535 | 0.535 | 0.883 | 0.620 |
| E5 | mmoe + taskenc + thin | 0.480 | 0.509 | 0.509 | 0.835 | 0.658 |
| E6 | cross_attention, residual off | 0.375 | 0.377 | 0.379 | 0.882 | 0.300 |
| E7 | self_attention, residual off | 0.414 | 0.433 | 0.433 | 0.863 | 0.443 |

## Localization

| ID | Method | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|---|
| E0 | none (A4-L1) | 0.789 | **0.887** | 0.965 | 0.238 | 0.403 | 0.031 |
| E1 | cross_attention | **0.717** | 0.823 | **0.971** | 0.209 | 0.381 | 0.045 |
| E2 | self_attention | 0.792 | 0.858 | 0.969 | 0.089 | 0.285 | 0.134 |
| E4 | mmoe + task encoder | 0.785 | 0.848 | 0.968 | 0.244 | 0.411 | 0.031 |
| E5 | mmoe + taskenc + thin | 1.165 | 0.846 | 0.962 | **0.295** | **0.453** | **0.018** |
| E6 | cross_attention, residual off | 1.337 | 0.745 | 0.959 | 0.165 | 0.315 | 0.075 |
| E7 | self_attention, residual off | 1.253 | 0.839 | 0.943 | 0.207 | 0.434 | 0.046 |

With the corrected line-level code, **cross_attention (E1) beats the E0 baseline**
on macro F1 (0.530 vs 0.519) — and also best IFA + AUC-ROC. E2 self_attention →
best accuracy / F1-w but lower macro F1. MMOE variants (E4, E5) collapse macro F1;
E5 (thin head) trades classification away for the best localization coverage
(R@20%LOC 0.453, Effort@20%R 0.018).

**Residual off (E6, E7) collapses hard** — macro F1 0.375 / 0.414, far below every
residual-on variant. `cross_task_residual=false` does in-path replace
(`fused_mod = cross`), discarding the original fused representation entirely — the
model must route everything through the freshly-init cross-task module from
scratch. The zero-init residual gate is load-bearing: it lets the cross-task
signal grow from a baseline-safe no-op instead of overwriting it. E3 plain mmoe
still pending.

**Phase 5 winner: E1 cross_attention — only method to beat the baseline F1; the
zero-init residual gate is essential (E6/E7 confirm in-path replace fails).**

---

# Phase 6 — Language Model (node_lm / func_lm)

`configs/ablation/phase6/` — varies the two language models with
`localization_encoder=both` and `graph_pool=meanmax` (Phase 4 winner) fixed
(matching the F1 baseline), isolating each LM's effect. CodeT5+ per-token
states for the `both` localizer come from its internal T5 encoder
(`lm_full_codet5p`, projected to 256-dim).

- **node_lm** (`pretrained_lm`) — frozen, builds node features in the .pt cache
- **func_lm** (`func_lm`) — live, fine-tuned function-level branch

| ID | Config | node_lm | func_lm | .pt build config |
|---|---|---|---|---|
| F1 | — (= Phase 4 meanmax, `20260516_125619`) | UniXcoder | UniXcoder | `node-unixcoder_func-unixcoder` |
| F2 | `F2_node-codet5p_func-unixcoder.yaml` | CodeT5+ | UniXcoder | `node-codet5p_func-unixcoder` |
| F3 | `F3_node-unixcoder_func-codet5p.yaml` | UniXcoder | CodeT5+ | `node-unixcoder_func-codet5p` |
| F4 | `F4_node-codet5p_func-codet5p.yaml` | CodeT5+ | CodeT5+ | `node-codet5p_func-codet5p` |

F1 (both UniXcoder, localization=gnn, **meanmax**) is identical to the Phase 4
**meanmax** run — no re-run needed, it serves as the baseline. (Not A2 — A2 uses
`graph_pool=mean`; using the meanmax run keeps F1 on the same pool as F2/F3/F4 so
the comparison isolates the LM, not the pool.)

CodeT5+ = `Salesforce/codet5p-110m-embedding` — pooled-tensor output, **256-dim**
(not 768; `lm_hidden_dim` probes it, `in_channels` adapts from the .pt).
Each combo needs its own .pt build (node features + func tokenizer differ).
Any config with CodeT5+ as func_lm (F3, F4) uses `func_max_length=512` and
`use_flash_attention=false` — CodeT5+ caps at 512 tokens, no flash_attention_2.

| ID | Test F1 | Test Acc | F1-w | AUC-ROC | IFA ↓ | Top-1 ↑ | R@20%LOC ↑ |
|---|---|---|---|---|---|---|---|
| F1 (= meanmax) | 0.517 | 0.538 | 0.539 | 0.911 | 0.644 | 0.900 | 0.487 |
| F2 | _pending_ | | | | | | |
| F3 | _pending_ | | | | | | |
| F4 | _pending_ | | | | | | |

All F-configs use the Phase 3 winner loss (no focal + label_smoothing 0.1 + cosine,
wd 1e-3, patience 15) — same as F1's meanmax baseline and phases 4-5.

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
