# Ablation Results

> **Metric source rule:** all classification + localization values come from `metrics_summary.json`.
> `training_summary.json` used ONLY for epoch time, total time, VRAM, params, GPU.
> Use `/ablation-metrics <run_id> [phase_dir]` to extract formatted rows — never copy by hand.

Dataset: MegaVul Top-25 CWEs, max 1600 per class, 26 classes (25 CWE + benign),
UniXcoder-base embeddings, seed=42. GPU: RTX 5070 Ti (Phase 1–7, Phase 9 I2/I3); RTX 5090 for Phase 8 reruns (H2/H3) and Phase 9 I4.

Phase structure:

- **Phase 1 — Encoder & Localization**: live-LM vs frozen, localization encoder
- **Phase 2 — GNN+LM Localization Fusion**: how GNN + LM statement features combine
- **Phase 3 — Loss Function**: focal / epoch-adaptive / LIVABLE tuning
- **Phase 4 — Graph Pooling**: mean / attention / meanmax / dualflow
- **Phase 5 — Multi-Task / Cross-Task**: bidirectional cross-task coupling
- **Phase 6 — Language Model**: node_lm / func_lm choice (UniXcoder / CodeT5+)
- **Phase 7 — GNN Dimension**: hidden_dim vs func_lm dim alignment (50/50 vs 25/75 GNN/LM fused)
- **Phase 8 — Sliding Window Coverage**: extend func_max_length to 5120 via chunk/stride variants
- **Phase 9 — Line-Level Encoder**: hierarchical encoding — per-line LM → cross-line transformer; frozen vs live LM
- **Phase 10 — Language Model (func_lm alternatives)**: encoder-only LMs with longer native context (ModernBERT-base)
- **Phase 11 — SupCon Loss Ablation**: supervised contrastive loss with CWE hierarchy distance matrix weighting
- **Phase 12 — ULMFiT Fine-tuning**: discriminative LRs (LLRD) + gradual unfreezing to curb val-loss divergence

---

# Phase 1 — Encoder & Localization

`configs/ablation/phase1/` — varies architecture (frozen vs live LM) and
`localization_encoder` (GNN nodes / LM tokens / both). Shared loss config:
`focal_loss_gamma=2.0`, `epoch_adaptive_weights=true`, `patience=25`.

| Run | Run ID                                      | Config                        | Architecture   | Localization | LM Fine-tuned    |
| --- | ------------------------------------------- | ----------------------------- | -------------- | ------------ | ---------------- |
| A1  | `20260513_202125_lmgat_multiclass`          | `A1_lmgat.yaml`               | lmgat (frozen) | GNN          | No               |
| A2  | `20260513_210613_lmgat_codebert_multiclass` | `A2_lmgat_codebert_gnn.yaml`  | lmgat_codebert | GNN          | Yes (lm_lr=1e-5) |
| A3  | `20260514_063713_lmgat_codebert_multiclass` | `A3_lmgat_codebert_lm.yaml`   | lmgat_codebert | LM           | Yes (lm_lr=1e-5) |
| A4  | `20260514_102721_lmgat_codebert_multiclass` | `A4_lmgat_codebert_both.yaml` | lmgat_codebert | GNN+LM       | Yes (lm_lr=1e-5) |

## Function-Level Classification

| Run | Val F1    | Test F1   | Test Acc | AUC-ROC | Conf. mean | Epochs |
| --- | --------- | --------- | -------- | ------- | ---------- | ------ |
| A1  | 0.458     | 0.471     | 0.510    | 0.884   | 0.765      | 55     |
| A2  | 0.532     | 0.494     | 0.500    | 0.907   | 0.698      | 54     |
| A3  | 0.548     | 0.495     | 0.517    | 0.913   | 0.801      | 76     |
| A4  | **0.550** | **0.504** | 0.507    | 0.899   | 0.813      | 74     |

## Val-Test F1 Gap

Both from the same best-val-F1 checkpoint. Gap = Val F1 − Test F1.

| Run | Val F1 | Test F1 | Gap    | Gap % |
| --- | ------ | ------- | ------ | ----- |
| A1  | 0.458  | 0.471   | -0.013 | -2.8% |
| A2  | 0.532  | 0.494   | 0.038  | 7.1%  |
| A3  | 0.548  | 0.495   | 0.053  | 9.7%  |
| A4  | 0.550  | **0.504**| 0.046  | 8.4%  |

A1 (frozen LM) has no gap; live-LM runs (A2-A4) show 7-10% — the live LM overfits.

## Statement-Level Localization

| Run | IFA ↓    | Top-1 ↑   | Top-3 ↑   | Top-5 ↑   | Top-10 ↑  | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --- | -------- | --------- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| A1  | 1.49     | 0.804     | 0.914     | 0.941     | 0.966     | 0.195     | 0.394      | 0.052         |
| A2  | **0.89** | **0.874** | 0.936     | 0.959     | 0.977     | 0.217     | 0.401      | **0.039**     |
| A3  | 1.33     | 0.818     | **0.939** | **0.969** | **0.988** | 0.197     | **0.451**  | 0.052         |
| A4  | 1.26     | 0.794     | 0.917     | 0.959     | 0.978     | **0.207** | 0.431      | 0.047         |

GNN localization (A2) = precise (best IFA/Top-1). LM localization (A3) = coverage
(best R@20%LOC). Both (A4) = best classification F1, localization a compromise.

## Loss Dynamics (A1-A4)

Loss at best-F1 epoch. Test loss = unweighted CE recomputed from `predictions.csv`.
Val loss was weighted during training — scales differ, watch relative gap.

| Run | Best Epoch | Train Loss | Val Loss | Test Loss | Min Val Loss | Min VL Epoch |
| --- | ---------- | ---------- | -------- | --------- | ------------ | ------------ |
| A1  | 30         | 0.150      | 2.357    | 2.469     | 1.670        | 9            |
| A2  | 29         | 0.326      | 1.700    | 1.906     | 1.433        | 12           |
| A3  | 51         | 0.276      | 2.329    | 2.475     | 1.400        | 7            |
| A4  | 49         | 0.149      | 2.430    | 2.769     | 1.374        | 7            |

A3/A4 show ~44-epoch F1-loss divergence (val_loss min at ep 7, F1 peak at ep 49-51)
— symptom of the stacked loss, addressed in Phase 3.

**Phase 1 winner: A4 (localization=both).**

---

# Phase 2 — GNN+LM Localization Fusion

`configs/ablation/phase2/` — base A4 (localization=both). Varies `stmt_both_mode`
(how GNN + LM statement features combine). concat = baseline. weighted =
score-level `(1-α)·gnn + α·lm`. gated = per-statement learned gate.

| ID  | Run ID                                      | Config                      | stmt_both_mode         |
| --- | ------------------------------------------- | --------------------------- | ---------------------- |
| B0  | (= C1 baseline, `20260514_174326`)          | —                           | concat                 |
| B1  | `20260515_120709_lmgat_codebert_multiclass` | `B1_both_gated.yaml`        | gated                  |
| B2  | `20260515_135412_lmgat_codebert_multiclass` | `B2_both_weighted_a03.yaml` | weighted (GNN-leaning) |
| B3  | `20260515_165955_lmgat_codebert_multiclass` | `B3_both_weighted_a05.yaml` | weighted (balanced)    |
| B4  | `20260515_152942_lmgat_codebert_multiclass` | `B4_both_weighted_a07.yaml` | weighted (LM-leaning)  |

| ID  | Variant        | Test F1   | Test Acc  | F1-w  | IFA ↓     | Top-1 ↑   | R@20%LOC ↑ |
| --- | -------------- | --------- | --------- | ----- | --------- | --------- | ---------- |
| B0  | concat         | 0.519     | 0.518     | 0.517 | 0.789     | 0.887     | 0.403      |
| B1  | gated          | 0.502     | 0.526     | 0.525 | 1.138     | 0.851     | **0.422**  |
| B2  | weighted α=0.3 | 0.480     | 0.533     | 0.533 | **0.644** | 0.876     | 0.414      |
| B3  | weighted α=0.5 | **0.539** | **0.539** | 0.538 | 1.007     | **0.890** | 0.400      |
| B4  | weighted α=0.7 | 0.515     | 0.515     | 0.514 | 0.947     | 0.832     | 0.357      |

No fusion beats concat on macro F1 (α=0.5 ties). weighted α=0.3 (GNN-leaning) →
best IFA — a localization-precision knob.

**Phase 2 winner: concat.**

---

# Phase 3 — Loss Function

`configs/ablation/phase3/` — fixes the stacked-loss problem from Phase 1.
Architecture held at A4 (localization=both, concat).

| Variant           | Run ID                                      | Loss Config                                                                   |
| ----------------- | ------------------------------------------- | ----------------------------------------------------------------------------- |
| A4 (Phase 1 base) | `20260514_102721_lmgat_codebert_multiclass` | focal γ=2.0 + epoch_adaptive, wd=1e-4, patience=25                            |
| C1                | `20260514_174326_lmgat_codebert_multiclass` | no focal + epoch_adaptive + label_smoothing=0.1, wd=1e-3, cosine, patience=15 |
| C2                | `20260515_052704_lmgat_codebert_multiclass` | LIVABLE two-branch (focal+LSCE), wd=1e-3, cosine, patience=15                 |
| C2-fixed          | `20260515_084125_lmgat_codebert_multiclass` | C2, no early stopping (full T-schedule)                                       |

## Classification

| Variant  | Val F1    | Test F1   | Test Acc  | F1-w  | AUC-ROC   | Conf.     | Epochs |
| -------- | --------- | --------- | --------- | ----- | --------- | --------- | ------ |
| A4       | 0.550     | 0.504     | 0.507     | 0.503 | 0.899     | 0.813     | 74     |
| C1       | **0.560** | **0.519** | 0.518     | 0.517 | **0.915** | **0.630** | 31     |
| C2       | **0.561** | 0.475     | 0.529     | 0.526 | 0.904     | 0.757     | 43     |
| C2-fixed | —         | 0.497     | **0.550** | —     | —         | —         | 75     |

## Localization

| Variant  | IFA ↓     | Top-1 ↑   | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| -------- | --------- | --------- | ------- | --------- | ---------- | ------------- |
| A4       | 1.26      | 0.794     | 0.959   | 0.207     | 0.431      | 0.047         |
| C1       | **0.789** | **0.887** | 0.965   | 0.238     | 0.403      | **0.031**     |
| C2       | 0.867     | 0.817     | 0.949   | **0.256** | 0.476      | 0.029         |
| C2-fixed | 1.277     | —         | —       | —         | **0.492**  | —             |

C1 (drop focal, add label smoothing) → best Test F1 + best localization
precision + best calibration. C2 (LIVABLE) → best accuracy but lower macro F1
(tail-class collapse — LIVABLE rebalances via focal branch only, no class-frequency
weighting). C2-fixed (full T-schedule, no early stop) recovered macro F1
0.475→0.497 — still below C1.

**Phase 3 winner: C1 (no focal + label smoothing). Baseline for Phases 4-5.**

Note: 5 exploratory loss runs (`20260514_145017/160914/191041/214622/234550`,
focal-off + livable on/off probes on frozen + live LM) predate the clean C1/L2
runs — superseded, kept for reference only.

---

# Phase 4 — Graph Pooling

`configs/ablation/phase4/` — base C1 (Phase 3 winner). Varies `graph_pool`
(function classification representation): mean / gated attention / meanmax
(0.8·max + 0.6·mean) / dualflow (suspicion-weighted focal + mean context).

| Variant   | Run ID                                      | graph_pool | Epochs |
| --------- | ------------------------------------------- | ---------- | ------ |
| mean      | (= C1, `20260514_174326`)                   | mean       | 31     |
| attention | `20260515_235912_lmgat_codebert_multiclass` | attention  | 50     |
| meanmax   | `20260516_125619_lmgat_codebert_multiclass` | meanmax    | 48     |
| dualflow  | `20260517_013824_lmgat_codebert_multiclass` | dualflow   | 38     |
| cnn       | `20260609_190812_lmgat_codebert_multiclass` | cnn        | 37     |

| Variant   | Test F1   | Test Acc  | F1-w      | AUC-ROC   | Conf. | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@20%LOC ↑ |
| --------- | --------- | --------- | --------- | --------- | ----- | --------- | --------- | --------- | ---------- |
| mean      | 0.519     | 0.518     | 0.517     | **0.915** | 0.630 | 0.789     | 0.887     | 0.965     | 0.403      |
| attention | 0.455     | 0.522     | 0.523     | 0.895     | 0.625 | 1.253     | 0.805     | 0.943     | 0.439      |
| meanmax   | **0.537** | **0.538** | **0.539** | 0.911     | 0.502 | **0.644** | **0.900** | **0.982** | **0.487**  |
| dualflow  | 0.496     | 0.528     | 0.528     | 0.896     | 0.667 | 0.717     | 0.886     | 0.971     | 0.417      |
| cnn       | 0.511     | 0.535     | 0.532     | 0.903     | 0.610 | 0.927     | 0.873     | 0.959     | 0.414      |

Attention pool collapses macro F1 (−0.082) — learnable gate overfits tail classes. Mean and meanmax tie on macro F1 but **meanmax wins everywhere else**: best accuracy, F1-w, and all localization metrics. dualflow lands mid-pack (F1 0.496) with same overfit risk as attention but milder. cnn pool (D4, late rerun) F1 0.511 ≈ meanmax with best IFA (0.927) but doesn't beat meanmax on F1.

**Phase 4 winner: meanmax.**

---

# Phase 5 — Multi-Task / Cross-Task

`configs/ablation/phase5/` — bidirectional cross-task between localization
(stmt_head) and classification (func_head). Zero-init residual gates
(ReZero/ControlNet style) — module starts as a no-op.
**Baseline E0 = C1** (Phase 3 winner, no cross-task).
All E-series runs use `graph_pool=mean` (same as E0/C1 baseline) — Phase 4
meanmax was found separately. Cross-task comparison is internally consistent.

| ID  | Run ID                                      | Config                         | cross_task_method               | Epochs    |
| --- | ------------------------------------------- | ------------------------------ | ------------------------------- | --------- |
| E0  | `20260514_174326_lmgat_codebert_multiclass` | —                              | none (= C1 baseline)            | 31        |
| E1  | `20260516_213244_lmgat_codebert_multiclass` | `E1_crossattn.yaml`            | cross_attention                 | 31        |
| E2  | `20260516_185818_lmgat_codebert_multiclass` | `E2_selfattn.yaml`             | self_attention                  | 55        |
| E3  | —                                           | `E3_mmoe.yaml`                 | mmoe                            | _pending_ |
| E4  | `20260516_152322_lmgat_codebert_multiclass` | `E4_mmoe_taskenc.yaml`         | mmoe + task encoder             | 40        |
| E5  | `20260516_171751_lmgat_codebert_multiclass` | `E5_mmoe_taskenc_thin.yaml`    | mmoe + task encoder + thin head | 35        |
| E6  | `20260517_042939_lmgat_codebert_multiclass` | `E6_crossattn_noresidual.yaml` | cross_attention, residual off   | 92        |
| E7  | `20260517_084804_lmgat_codebert_multiclass` | `E7_selfattn_noresidual.yaml`  | self_attention, residual off    | 58        |

> **Earlier cross-task results removed.** A prior set of E1/E2/E3 runs was
> trained **before the per-statement line-level cross-task code was correct**
> (they collapsed the localization view to a per-graph vector instead of
> conditioning each statement). Those metrics were invalid and deleted.
> Invalidated run IDs (kept on disk, do not report): `20260515_211228`,
> `20260516_055225`, `20260516_101335`.
> The E1/E2/E4/E5 results below are from corrected-code runs.

## Classification

| ID  | Method                        | Test F1   | Test Acc  | F1-w      | AUC-ROC   | Conf. |
| --- | ----------------------------- | --------- | --------- | --------- | --------- | ----- |
| E0  | none (C1)                     | 0.519     | 0.518     | 0.517     | 0.915     | 0.630 |
| E1  | cross_attention               | **0.530** | 0.532     | 0.533     | **0.919** | 0.615 |
| E2  | self_attention                | 0.524     | **0.538** | **0.537** | 0.897     | 0.606 |
| E4  | mmoe + task encoder           | 0.479     | 0.535     | 0.535     | 0.883     | 0.620 |
| E5  | mmoe + taskenc + thin         | 0.480     | 0.509     | 0.509     | 0.835     | 0.658 |
| E6  | cross_attention, residual off | 0.390     | 0.377     | 0.379     | 0.882     | 0.300 |
| E7  | self_attention, residual off  | 0.430     | 0.433     | 0.433     | 0.863     | 0.443 |

## Localization

| ID  | Method                        | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --- | ----------------------------- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| E0  | none (C1)                     | 0.789     | **0.887** | 0.965     | 0.238     | 0.403      | 0.031         |
| E1  | cross_attention               | **0.717** | 0.823     | **0.971** | 0.209     | 0.381      | 0.045         |
| E2  | self_attention                | 0.792     | 0.858     | 0.969     | 0.089     | 0.285      | 0.134         |
| E4  | mmoe + task encoder           | 0.785     | 0.848     | 0.968     | 0.244     | 0.411      | 0.031         |
| E5  | mmoe + taskenc + thin         | 1.165     | 0.846     | 0.962     | **0.295** | **0.453**  | **0.018**     |
| E6  | cross_attention, residual off | 1.337     | 0.745     | 0.959     | 0.165     | 0.315      | 0.075         |
| E7  | self_attention, residual off  | 1.253     | 0.839     | 0.943     | 0.207     | 0.434      | 0.046         |

**E1 cross_attention beats E0 baseline** (F1 0.530 vs 0.519) — also best IFA + AUC-ROC. E2 self_attention → best accuracy/F1-w but lower macro F1. MMOE variants collapse macro F1; E5 thin head trades classification for best localization coverage (R@20% 0.453, Effort 0.018). **Residual off (E6/E7) collapses hard** (F1 0.375/0.414) — zero-init gate is load-bearing; `cross_task_residual=false` discards original fused representation entirely. E3 plain mmoe still pending.

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

| ID  | Config                                       | node_lm   | func_lm                     | .pt build config                | Run ID            | Epochs |
| --- | -------------------------------------------- | --------- | --------------------------- | ------------------------------- | ----------------- | ------ |
| F1  | — (= Phase 4 meanmax, `20260516_125619`)     | UniXcoder | UniXcoder                   | `node-unixcoder_func-unixcoder` | `20260516_125619` | 48     |
| F2  | `F2_node-codet5p_func-unixcoder.yaml`        | CodeT5+   | UniXcoder                   | `node-codet5p_func-unixcoder`   | `20260518_021722` | 48     |
| F3  | `F3_node-unixcoder_func-codet5p.yaml`        | UniXcoder | CodeT5+                     | `node-unixcoder_func-codet5p`   | `20260517_171638` | 67     |
| F4  | `F4_node-codet5p_func-codet5p.yaml`          | CodeT5+   | CodeT5+                     | `node-codet5p_func-codet5p`     | `20260518_103922` | 66     |
| F5  | `F5_node-unixcoder_func-codet5p-raw.yaml`    | UniXcoder | CodeT5+ raw (768-dim `<s>`) | `node-unixcoder_func-codet5p`   | `20260519_154339` | 54     |
| F6  | `F6_node-unixcoder_func-codet5p-normed.yaml` | UniXcoder | CodeT5+ proj+norm per-token | `node-unixcoder_func-codet5p`   | `20260519_224619` | 34     |

F1 (both UniXcoder, localization=gnn, **meanmax**) is identical to the Phase 4
**meanmax** run — no re-run needed, it serves as the baseline. (Not A2 — A2 uses
`graph_pool=mean`; using the meanmax run keeps F1 on the same pool as F2/F3/F4 so
the comparison isolates the LM, not the pool.)

CodeT5+ = `Salesforce/codet5p-110m-embedding` — pooled-tensor output, **256-dim**
(not 768; `lm_hidden_dim` probes it, `in_channels` adapts from the .pt).
Each combo needs its own .pt build (node features + func tokenizer differ).
Any config with CodeT5+ as func_lm (F3, F4) uses `func_max_length=512` and
`use_flash_attention=false` — CodeT5+ caps at 512 tokens, no flash_attention_2.

Bolds = best among F2–F7 (new configs; F1 is unchanged baseline).

| ID                      | Test F1   | Test Acc  | F1-w      | AUC-ROC   | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ----------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| F1 (= meanmax)          | **0.517** | 0.538     | 0.539     | 0.911     | 0.644     | 0.900     | 0.982     | 0.269     | 0.487      | 0.025         |
| F2 (node=CT5+)          | 0.502     | 0.554     | 0.552     | **0.906** | 0.745     | 0.899     | 0.982     | 0.232     | 0.415      | 0.032         |
| F3 (func=CT5+)          | 0.444     | 0.517     | 0.523     | 0.897     | 0.512     | **0.946** | **0.985** | **0.374** | **0.586**  | **0.016**     |
| F4 (both=CT5+)          | 0.475     | 0.548     | 0.552     | 0.833     | 0.991     | 0.857     | 0.958     | 0.254     | 0.436      | 0.024         |
| F5 (func=CT5+ raw 768)  | 0.499     | **0.568** | **0.566** | 0.897     | 1.186     | 0.925     | 0.977     | 0.308     | 0.547      | 0.022         |
| F6 (func=CT5+ norm 256) | 0.459     | 0.520     | 0.520     | 0.905     | 0.734     | 0.934     | 0.982     | 0.353     | 0.560      | **0.016**     |
| F7 (both normed)        | 0.484     | 0.557     | 0.556     | 0.887     | **0.508** | 0.927     | 0.980     | 0.319     | 0.570      | 0.020         |

All F-configs use Phase 3 winner loss (no focal + label_smoothing 0.1 + cosine, wd 1e-3, patience 15).

**F2 (node=CodeT5+)** — best accuracy (0.554) and AUC (0.906) among new configs; localization weaker than F1 (IFA 0.745 vs 0.644, R@20% 0.415 vs 0.487).

**F3 (func=CodeT5+)** — best localization overall (IFA 0.512, Top-1 0.946, R@20% 0.586) at cost of worst macro F1 (0.444).

**F4 (both=CodeT5+)** — CodeT5+ at both levels doesn't compound gains; AUC collapses (0.833, worst) and localization worse than either single-LM variant.

**F5 (func=CT5+ raw 768-dim)** — best accuracy (0.568) but localization collapses (IFA 1.186, worst): `[GNN-256 | LM-768]` concat drowns GNN statement signal (LM at 75% of dim).

**F6 (func=CT5+ proj+norm 256-dim)** — unit norm partially recovers localization vs F5 (IFA 0.734) but F3 (unnormalized) still wins (IFA 0.512); normalization removes useful amplitude info.

**F7 (GNN+LM both normed)** — marginal IFA gain over F3 (0.508 vs 0.512) but Top-1/R@20% worse; F3 remains best localization overall.

**Key finding:** LM dim relative to GNN dim governs the trade-off — raw 768-dim (F5) maximises classification, projected 256-dim (F3) best for localization; normalization neutral-to-harmful.

**Phase 6 winner: F1 (UniXcoder both) for balanced performance. F3 for localization
priority. F7 for IFA-only if that metric is the objective.**

---

# Phase 7 — GNN Dimension

`configs/ablation/phase7/` — tests whether aligning `hidden_dim` with the func_lm
embedding dimension (768 for UniXcoder) improves GNN/LM balance in the fused
representation. All configs use UniXcoder for both node_lm and func_lm, meanmax pool,
localization=both, concat fusion — isolating only the hidden_dim change.

- **G1** = F1 baseline (`20260516_125619`): hidden_dim=256, fused=1024, GNN 25% / LM 75%
- **G2** = `G2_dim768_equal.yaml`: hidden_dim=768, fused=1536, GNN 50% / LM 50%

| ID  | Run ID                                      | Config                     | hidden_dim | fused_dim | GNN% | LM% | Epochs |
| --- | ------------------------------------------- | -------------------------- | ---------- | --------- | ---- | --- | ------ |
| G1  | `20260516_125619_lmgat_codebert_multiclass` | — (= Phase 4 meanmax / F1) | 256        | 1024      | 25%  | 75% | 48     |
| G2  | `20260520_132730_lmgat_codebert_multiclass` | `G2_dim768_equal.yaml`     | 768        | 1536      | 50%  | 50% | 34     |

## Classification

| ID  | Test F1   | Test Acc  | F1-w      | AUC-ROC   | Conf. | Epochs |
| --- | --------- | --------- | --------- | --------- | ----- | ------ |
| G1  | **0.537** | 0.538     | 0.539     | 0.911     | 0.502 | 48     |
| G2  | 0.529     | **0.582** | **0.579** | **0.914** | 0.569 | 34     |

## Statement-Level Localization

| ID  | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| G1  | **0.644** | **0.900** | **0.982** | **0.269** | **0.487**  | **0.025**     |
| G2  | 1.410     | 0.747     | 0.962     | 0.186     | 0.427      | 0.056         |

G2 beats G1 on all classification metrics (+1.2pp F1, +4.4pp Acc, +0.003 AUC) but localization collapses (IFA 0.644→1.410, Top-1 0.900→0.747) — ~32% of MegaVul functions exceed 1024 tokens and G2 truncates them, degrading per-node signal. Also 2.5× slower and +3 GB VRAM.

**Phase 7 winner: G2 for classification (best F1 to date: 0.529). G1 for localization.
G2 used as Phase 8 baseline — Phase 8 will fix the truncation via sliding window.**

---

# Phase 8 — Sliding Window Coverage

`configs/ablation/phase8/` — base H1 = G2 (hidden_dim=768, func_max_length=1024, no sliding).
Sliding window extends func coverage to func_max_length=5120 (covers MegaVul P95 = 4326 tokens with CLS/SEP overhead).
Chunk=1024 = UniXcoder max. Classification aggregation: mean-pool over real tokens per window,
averaged across windows. Localization aggregation: per-token mean across overlapping windows.
H2 and H3 results are from reruns with fixed `lm_full_windowed` (mean-pool CLS across windows)
and ml5120 dataset on RTX 5090. Earlier H2/H3 runs with ml1024 never activated sliding window
(fast path always triggered when func_max_length=max_length=1024).

| ID    | Config                                                            | chunk | stride | max_len | Max windows     | Run ID            | Epochs |
| ----- | ----------------------------------------------------------------- | ----- | ------ | ------- | --------------- | ----------------- | ------ |
| H1    | — (= G2)                                                          | —     | —      | 1024    | 1               | `20260520_132730` | 34     |
| H2    | `H2_unixcoder_sliding_chunk1024_stride512.yaml`                   | 1024  | 512    | 5120    | 9               | `20260525_104032` | 30     |
| H3    | `H3_unixcoder_sliding_chunk1024_stride1024.yaml`                  | 1024  | 1024   | 5120    | 5               | `20260525_125031` | 31     |
| H4    | `H4_unixcoder_sliding_chunk1024_stride1024_winattn.yaml`          | 1024  | 1024   | 5120    | 5+attn          | `20260527_121315` | 22     |
| H5    | `H5_unixcoder_sliding_chunk1024_stride512_winattn.yaml`           | 1024  | 512    | 5120    | 9+attn          | `20260528_062323` | 34     |
| H6    | `H6_unixcoder_sliding_chunk1024_stride1024_winattn_hidden.yaml`   | 1024  | 1024   | 5120    | 5+attn+hidden   | `20260528_085945` | 31     |
| H7    | `H7_unixcoder_sliding_chunk1024_stride512_winattn_centerw.yaml`   | 1024  | 512    | 5120    | 9+attn+cw       | `20260528_063142` | 34     |
| H8    | `H8_unixcoder_sliding_chunk1024_stride512_winattn_crosswin.yaml`  | 1024  | 512    | 5120    | 9+attn+crosswin | `20260528_094016` | 41     |
| H9    | `H9_unixcoder_sliding_chunk1024_stride1024_winattn_crosswin.yaml` | 1024  | 1024   | 5120    | 5+attn+crosswin | `20260610_122559` | 55     |
| H10   | `H10_unixcoder_sliding_chunk1024_stride1024_winmixer.yaml`        | 1024  | 1024   | 5120    | 5+mixer         | `20260610_152801` | 26     |
| H10n  | `H10_nine.yaml` (node+func unixcoder-base-nine)                   | 1024  | 1024   | 5120    | 5+mixer         | `20260614_131244` | 43     |
| H10fn | `H10_funcnine.yaml` (node base, func unixcoder-base-nine)         | 1024  | 1024   | 5120    | 5+mixer         | `20260614_155019` | 40     |
| H10w0 | `H10_..._winmixer_w0.yaml` (num_workers 0 rerun)                  | 1024  | 1024   | 5120    | 5+mixer         | `20260617_143140` | 33     |
| H10-vo | `vulnonly/H10_vulnonly.yaml` (25-class vuln-only)                | 1024  | 1024   | 5120    | 5+mixer         | `20260618_040527` | 39     |

## Classification

| ID        | Test F1   | Test Acc | F1-w  | AUC-ROC   | Conf. | Epochs |
| --------- | --------- | -------- | ----- | --------- | ----- | ------ |
| H1 (= G2) | 0.529     | 0.582    | 0.579 | 0.914     | 0.569 | 34     |
| H2        | 0.459     | 0.508    | 0.507 | 0.890     | 0.588 | 30     |
| H3        | 0.528     | 0.529    | 0.533 | 0.895     | 0.587 | 31     |
| H4        | 0.520     | 0.563    | 0.560 | **0.927** | 0.695 | 22     |
| H5        | 0.443     | 0.522    | 0.522 | 0.885     | 0.589 | 34     |
| H6        | 0.513     | 0.532    | 0.533 | 0.903     | 0.607 | 31     |
| H7        | 0.485     | 0.524    | 0.525 | 0.896     | 0.618 | 34     |
| H8        | 0.520     | 0.536    | 0.538 | 0.898     | 0.584 | 41     |
| H9        | 0.490     | 0.520    | 0.539 | 0.897     | 0.479 | 55     |
| H10       | 0.534     | 0.538    | 0.537 | 0.879     | 0.607 | 26     |
| H10n      | 0.544     | 0.545    | 0.544 | 0.873     | 0.535 | 43     |
| H10fn     | 0.518     | 0.545    | 0.543 | 0.873     | 0.518 | 40     |
| H10w0     | 0.500     | 0.529    | 0.527 | 0.884     | 0.549 | 33     |
| H10-vo    | **0.565** | 0.593    | 0.592 | 0.896     | 0.539 | 39     |

## Statement-Level Localization

| ID        | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --------- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| H1 (= G2) | 1.410     | 0.747     | 0.962     | 0.186     | 0.427      | 0.056         |
| H2        | 1.025     | 0.876     | 0.971     | 0.193     | 0.395      | 0.052         |
| H3        | 1.047     | 0.873     | **0.978** | **0.221** | 0.442      | **0.041**     |
| H4        | 1.063     | 0.855     | 0.971     | 0.187     | 0.430      | 0.054         |
| H5        | 1.220     | 0.827     | 0.969     | 0.206     | 0.424      | 0.047         |
| H6        | **1.034** | 0.823     | 0.969     | 0.204     | **0.445**  | 0.048         |
| H7        | 2.804     | 0.669     | 0.846     | 0.143     | 0.410      | 0.076         |
| H8        | 1.069     | **0.912** | 0.977     | 0.167     | 0.400      | 0.068         |
| H9        | 1.391     | 0.905     | 0.974     | 0.146     | 0.387      | 0.072         |
| H10       | 1.180     | 0.798     | 0.975     | 0.212     | 0.432      | 0.045         |
| H10n      | 0.946     | 0.814     | 0.971     | 0.204     | 0.432      | 0.049         |
| H10fn     | 1.328     | 0.807     | 0.963     | 0.190     | 0.383      | 0.056         |
| H10w0     | 1.001     | 0.867     | 0.980     | 0.230     | 0.444      | 0.037         |
| H10-vo    | 2.333     | 0.761     | 0.931     | 0.200     | 0.431      | 0.050         |

Both H2 and H3 use ml5120 dataset with fixed `lm_full_windowed` (mean-pool CLS across windows),
so sliding window is genuinely active for functions exceeding 1024 tokens (~32% of MegaVul vuln functions).

**H2 (stride=512, 9 windows)** — classification degrades (F1 0.459, −0.070 vs H1): boundary regions appear in 2 consecutive windows, causing noisy mean-pool CLS. Localization improves over H1 (IFA 1.025 vs 1.410).

**H3 (stride=1024, 5 windows)** — ties H1 on classification (F1 0.528, −0.001); non-overlapping avoids boundary noise, cleaner per-window CLS. Beats H2 on R@5%/R@20%/Effort and is 1.7× faster (150s vs 250s/epoch).

H2/H3 both below G1 localization baseline (IFA 0.644) — mean aggregation blurs single-pass per-token signal.

**H4 (window attn pool, stride=1024)** — learned `Linear(768→1)` weights per-window CLS; best AUC 0.927 (vs H1 0.914). F1 0.520 is just below H3 (0.528) and H1 baseline (0.529) — window attn improves ranking but not macro F1 vs H3. Localization slightly worse than H3 (IFA 1.063 vs 1.047).

**H5 (attn pool + stride=512, 9 windows)** — overlap boundary noise dominates (F1 0.443, −0.077 vs H4); attn pool can't fully recover. Localization also worse than H4/H2 (IFA 1.220).

**H6 (attn weights to localization path, stride=1024)** — improves localization over H4 (IFA 1.034 vs 1.063, R@20% 0.445 vs 0.430); classification comparable (F1 0.513 vs H4 0.520, −0.007). Attention weights applied to per-token hidden states concentrate stmt_head on high-relevance windows.

**H7 (center weighting, stride=512)** — hurts both tasks (F1 0.485, −0.035 vs H4; IFA 2.804, worst): CWE-identifying tokens often land at window-edge positions, which center weighting systematically discounts.

**H8 (cross-window attention, stride=512)** — H5 + CrossWindowAttn: Q=per-token hidden [B,L,768], K/V=window CLS [B,W,768]; each token attends over all window summaries for global context. Improves over H5 on both classification (F1 0.520 vs 0.443, +0.077) and localization (IFA 1.069 vs 1.220, Top-1 0.912 vs 0.827, best Top-1 overall). Ties H4 on F1 (0.520) but worse than H3 (0.528).

**H9 (cross-window attention, stride=1024)** — H8 with non-overlap windows (5 vs 9). Classification drops vs H8 (F1 0.490 vs 0.520, −0.030) and localization too (IFA 1.391 vs 1.069, Top-1 0.905 vs 0.912) — CrossWindowAttn benefits from H8's overlapping windows giving each token richer cross-window context; non-overlap starves it of that signal. Worst F1 in phase 8 besides H2/H5.

**H10 (MLP-Mixer over window CLS, stride=1024)** — replaces softmax window-attn pool with WindowMixerPool (token-mix + channel-mix over 5 fixed window CLS). New phase-8 best F1 (0.534, +0.005 vs H1 0.529) and best Test Acc (0.538) with fewest epochs (26, early-stopped) — non-overlap + fixed window count is exactly the regime MLP-Mixer needs (ordered, fixed-length token set). Localization regresses vs H1 (IFA 1.180 vs 1.410 is actually better than H1, but Top-1 0.798 vs H1 0.747 better too — yet both well below H3/H6/H8); mixer pool has no localization-side branch (cross_window_attn=false), so stmt_head gets no extra signal from the mixer.

**H10n (node+func unixcoder-base-nine)** — new phase-8 best F1 **0.544** (+0.010 vs H10 base 0.534), best Acc 0.545, and best localization of the H10 family (IFA 0.946, Top-1 0.814) — both beat base. The C-aware nine LM **helps here** because H10's sliding window makes func_lm active, and nine is applied consistently to **both** node + func. This overturns the N48-nine verdict (node-only nine, func inert: 0.484, worse): nine pays off only when func_lm actually consumes it. AUC 0.873 is slightly below base (0.879). 211s/ep, 2.53 hr.

**H10fn (node unixcoder-base, func unixcoder-base-nine)** — F1 **0.518** (−0.016 vs base, −0.026 vs H10n) and worst localization of the three (IFA 1.328). Mixing base node embeddings with nine func tokens **hurts** — the win needs nine on both sides (H10n), not a split. Higher raw Prec/Rec (0.585/0.594) but lower macro F1 and confidence (0.518). Takeaway: use nine **consistently** (node+func) with an active func_lm, or stay fully on base — the mixed config is the worst of both.

**Phase 8 winner: H10 — new best F1 (0.534) and Test Acc (0.538), beating H1 baseline (0.529/0.582 resp. — note H1 Acc still highest overall at 0.582) with the fewest epochs (26). H4 best AUC (0.927). H6 best localization IFA (1.034). H8 best Top-1 (0.912). H3 best Effort (0.041). H9 confirms CrossWindowAttn needs H8's overlapping windows — non-overlap (H9) regresses both tasks vs H8. H4/H10 candidates for Phase 11+ baseline depending on task priority (ranking vs macro-F1).**

**H10-vo (25-class vuln-only)** — H10 (mixer LM-aggregation) on the vuln-only dataset for the apples-to-apples baseline compare. Test F1 **0.565** — among our vuln-only trio it ranks N48-vo **0.601** (GNN-only) > H10-vo **0.565** > O1-vo **0.552** (hybrid join), so the LM-aggregation model edges out the join here. Vs the vuln-only baselines: below LOSVER (0.580) and VulExplainer (0.576), above LIVABLE (0.047); Acc 0.593 ≈ VulExplainer 0.595. **Not directly comparable to the 26-class H-series above** (different label space, macro over 25 vs 26). Localization IFA 2.333 is the worst of the trio (N48-vo 0.474, O1-vo 1.95). Only N48-vo beats all baselines on macro-F1 — the GNN-only block remains our multiclass headline. 147.5M, 188s/ep, 2.04 hr on RTX 5090.

---

# Phase 9 — Line-Level Encoder

`configs/ablation/phase9/` — base I1 = H1 = G2 (hidden_dim=768, meanmax, localization=both, no sliding window).
Varies the LM encoding strategy: whole-function forward (I1) vs hierarchical per-line LM forward
→ `_LineLevelEncoder` (2-layer cross-line transformer) for cross-line context.
Classification = meanmax pool of line_encoder output [B, 768].
Localization = per-line encoder output scattered back to token positions.

- **I1** = H1 = G2 baseline (live_lm=func, whole-function forward)
- **I2** = live_lm=line, freeze_func_lm=true, precompute_line_cls=true
  - Frozen UniXcoder → per-line [CLS] precomputed from raw_func (all lines, no global truncation)
  - Only GNN + line_encoder + heads trained; no lm_lr group
- **I3** = live_lm=line, freeze_func_lm=false, no precompute
  - Per-line LM forward each batch; LM weights updated jointly with line_encoder

| ID  | Run ID                                      | Config                      | live_lm | freeze_func_lm | precompute | Line ctx | Epochs |
| --- | ------------------------------------------- | --------------------------- | ------- | -------------- | ---------- | -------- | ------ |
| I1  | `20260520_132730` (= H1/G2)                 | —                           | func    | No             | —          | —        | 34     |
| I2  | `20260527_093837_lmgat_codebert_multiclass` | `I2_line_encoder.yaml`      | line    | Yes            | Yes        | —        | 23†    |
| I3  | `20260527_102049_lmgat_codebert_multiclass` | `I3_line_encoder_live.yaml` | line    | No             | No         | —        | 27†    |
| I4  | `20260525_141857_lmgat_codebert_multiclass` | `I4_line_ctx5.yaml`         | line    | Yes            | Yes        | ±5       | 52     |

## Classification

† = classification collapsed (predicts majority class only; metrics not comparable).

| ID           | Test F1   | Test Acc  | F1-w      | AUC-ROC   | Conf.  | Epochs |
| ------------ | --------- | --------- | --------- | --------- | ------ | ------ |
| I1 (= H1/G2) | **0.529** | **0.582** | **0.579** | **0.914** | 0.569  | 34     |
| I2†          | 0.156†    | 0.328†    | 0.285†    | 0.832†    | 0.260† | 23     |
| I3†          | 0.012†    | 0.149†    | 0.040†    | 0.766†    | 0.567† | 27     |
| I4           | 0.375     | 0.414     | 0.410     | 0.877     | 0.381  | 52     |

## Statement-Level Localization

| ID           | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ------------ | --------- | --------- | --------- | --------- | ---------- | ------------- |
| I1 (= H1/G2) | 1.410     | 0.747     | 0.962     | 0.186     | 0.427      | 0.056         |
| I2†          | 1.438     | 0.713     | 0.940     | 0.237     | **0.506**  | 0.037         |
| I3†          | **1.201** | **0.896** | **0.978** | 0.165     | 0.400      | 0.066         |
| I4           | 3.009     | 0.609     | 0.857     | 0.179     | 0.443      | 0.059         |

**I2 (frozen LM + line encoder)** — classification severely degraded (F1 0.156): precomputed fixed CLS can't adapt to 26-class CWE task; best val F1 0.149 at ep 23. Localization survives (R@20% 0.506, best in Phase 9).

**I3 (live LM + line encoder)** — classification also collapsed (F1 0.010): per-line forward loses global function-level context; cross-line transformer can't recover 26-class CWE semantics. Top-1 0.896 (best in Phase 9).

**I4 (±5 line context)** — partially recovers classification (F1 0.375) but localization degrades (IFA 3.009); broader context widens recall slightly at high IFA cost.

**Phase 9 finding: line-level LM encoding is incompatible with 26-class CWE classification.** All variants (frozen/live) collapse — whole-function CLS is necessary. Negative result; confirms sliding window as better path.

---

# Phase 10 — Language Model (func_lm alternatives)

`configs/ablation/phase10/` — base J1 = I1 = H1 = G2 (hidden_dim=768, meanmax, localization=both,
UniXcoder func_lm, func_max_length=1024, no sliding window). Varies `func_lm` to test alternative
encoder-only LMs with longer native context. All configs use `live_lm=func, freeze_func_lm=false`.

- **J1** = H1 = G2 baseline (func_lm=UniXcoder, 1024-token, 1024-chunk sliding)
- **J3** = func_lm=ModernBERT-base (8192-token native RoPE, alternating local-global attention)

| ID  | Run ID                                      | Config                    | func_lm         | func_max_length | Epochs |
| --- | ------------------------------------------- | ------------------------- | --------------- | --------------- | ------ |
| J1  | `20260520_132730` (= H1/G2)                 | —                         | UniXcoder       | 1024            | 34     |
| J3  | `20260528_045419_lmgat_codebert_multiclass` | `J3_modernbert_base.yaml` | ModernBERT-base | 5120            | 55     |

## Classification

| ID           | Val F1 | Test F1   | Test Acc  | F1-w      | AUC-ROC   | Conf.     | Epochs |
| ------------ | ------ | --------- | --------- | --------- | --------- | --------- | ------ |
| J1 (= H1/G2) | —      | **0.529** | **0.582** | **0.579** | **0.914** | **0.569** | 34     |
| J3           | 0.386  | 0.393     | 0.426     | 0.422     | 0.847     | 0.397     | 55     |

## Statement-Level Localization

| ID           | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ------------ | --------- | --------- | --------- | --------- | ---------- | ------------- |
| J1 (= H1/G2) | 1.410     | 0.747     | 0.962     | 0.186     | **0.427**  | 0.056         |
| J3           | **0.950** | **0.814** | **0.978** | **0.209** | 0.422      | **0.046**     |

**J3 (ModernBERT-base, 5120-token native)** — localization improves (IFA 0.950 vs 1.410, −33%; Top-1 0.814 vs 0.747) but classification collapses (F1 0.378, −0.151; AUC 0.847 vs 0.914). Alternating local/global attention fails to produce function-level CWE semantics that UniXcoder's full bidirectional attention computes in a single pass.

**Phase 10 finding: ModernBERT-base is a localization/classification trade-off, not an improvement.** UniXcoder with sliding window remains best overall — H10 (window mixer, F1=0.534) best classification, H4 (winattn, AUC=0.927) best AUC. (Note: earlier text here cited "H4, F1=0.554" — that 0.554 was actually F2's Test Acc from Phase 6, an unrelated copy-paste error; H4's actual Test F1=0.520.)

---

# Phase 11 — SupCon Loss Ablation

`configs/ablation/phase11/` — base K1 = H4 (window attn pool, stride=1024, hidden_dim=768, ml5120).
Varies supervised contrastive loss (SupCon) with CWE hierarchy distance matrix weighting.
All K-configs use `supcon_use_distance_matrix=true`, `cwe_dist_matrix=data/cwe/cwe_distance_matrix.json`.
L_self = NT-Xent self-supervised collapse prevention (two dropout views of same function embedding).
Loss: `L = L_CE + supcon_weight·L_SupCon(matrix) + supcon_self_weight·L_self`.

| ID  | Run ID                                      | Config                                      | supcon_weight | weight_fn                        | L_self weight | Epochs |
| --- | ------------------------------------------- | ------------------------------------------- | ------------- | -------------------------------- | ------------- | ------ |
| K1  | `20260527_121315` (= H4)                    | —                                           | 0             | —                                | 0             | 22     |
| K2  | `20260528_142050_lmgat_codebert_multiclass` | `K2_unixcoder_winattn_supcon_w02.yaml`      | 0.2           | linear                           | 0.2           | 33     |
| K5  | `20260528_160806_lmgat_codebert_multiclass` | `K5_supcon_group.yaml`                      | 0.2           | group (intragroup_only)          | 0.2           | 30     |
| K6  | `20260528_175315_lmgat_codebert_multiclass` | `K6_unixcoder_winattn_supcon_balanced.yaml` | 0.2           | linear (balanced sampler 8cls×4) | 0.2           | 32     |

## Classification

| ID        | Val F1 | Test F1 | Test Acc | F1-w  | AUC-ROC | Conf. | Epochs |
| --------- | ------ | ------- | -------- | ----- | ------- | ----- | ------ |
| K1 (= H4) | 0.573  | **0.520**| 0.563    | 0.560 | 0.927   | 0.695 | 22     |
| K2        | 0.502  | 0.479   | 0.527    | 0.527 | 0.874   | 0.607 | 33     |
| K5        | 0.521  | 0.504   | 0.550    | 0.551 | 0.897   | 0.608 | 30     |
| K6        | 0.513  | 0.500   | 0.503    | 0.504 | 0.892   | 0.579 | 32     |

## Statement-Level Localization

| ID        | IFA ↓  | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --------- | ------ | ------- | ------- | --------- | ---------- | ------------- |
| K1 (= H4) | 1.063  | 0.855   | 0.971   | 0.187     | 0.430      | 0.054         |
| K2        | 12.167 | 0.253   | 0.526   | 0.055     | 0.207      | 0.194         |
| K5        | 12.934 | 0.264   | 0.540   | 0.061     | 0.200      | 0.200         |
| K6        | 12.935 | 0.235   | 0.512   | 0.048     | 0.196      | 0.203         |

K2 — SupCon with dist-matrix linear weighting + L_self (w=0.2 each) collapses both classification (F1 0.461, −0.059 vs K1) and localization (IFA 12.167, Top-1 0.253). SupCon loss at w=0.2 appears to overwhelm the classification signal with 26-class CWE embedding geometry constraints.

---

# Phase 12 — ULMFiT Fine-tuning

`configs/ablation/phase12/` — base H4 (UniXcoder, winattn pool, stride=1024, dim=768, ml5120). ULMFiT to curb the live-LM val-loss divergence: M1 = LLRD (discriminative per-layer LR, decay 0.95), M2 = gradual unfreezing (progressive layer-unfreeze schedule), M3 = both combined.

| ID  | Run ID                                      | Config                          | Technique         | Epochs |
| --- | ------------------------------------------- | ------------------------------- | ----------------- | ------ |
| M1  | `20260609_130107_lmgat_codebert_multiclass` | `M1_llrd_only.yaml`             | LLRD (decay 0.95) | 25     |
| M2  | `20260609_142900_lmgat_codebert_multiclass` | `M2_gradual_unfreeze.yaml`      | gradual unfreeze  | 58     |
| M3  | `20260609_171605_lmgat_codebert_multiclass` | `M3_llrd_gradual_combined.yaml` | LLRD + gradual    | 40     |

## Classification

| ID  | Val F1 | Test F1 | Test Acc | F1-w  | AUC-ROC | Conf. | Epochs |
| --- | ------ | ------- | -------- | ----- | ------- | ----- | ------ |
| M1  | 0.498  | 0.493   | 0.514    | 0.511 | 0.900   | 0.569 | 25     |
| M2  | 0.494  | **0.532**| 0.531    | 0.541 | 0.890   | 0.484 | 58     |
| M3  | 0.524  | 0.455   | 0.514    | 0.513 | 0.910   | 0.557 | 40     |

## Statement-Level Localization

| ID  | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --- | ----- | ------- | ------- | --------- | ---------- | ------------- |
| M1  | 1.003 | 0.868   | 0.978   | 0.206     | 0.423      | 0.047         |
| M2  | 1.187 | 0.848   | 0.974   | 0.229     | 0.454      | 0.039         |
| M3  | 1.045 | 0.857   | 0.977   | 0.210     | 0.422      | 0.045         |

Base H4 = Test F1 0.520. **M2 (gradual) = best ULMFiT (F1 0.532, +0.012 vs H4)** but slowest (58 ep) and worst IFA (1.187). M1 (LLRD) 0.493 and M3 (combined) 0.455 both regress. **M3 is the tell: highest Val F1 (0.524) but lowest Test F1 (0.455) — widest val/test gap, the opposite of ULMFiT's intent.** Net: ULMFiT ≈ F1-neutral, doesn't beat H1/H4; val-loss divergence is a calibration issue, not an F1 lever.

---

# Phase 13 — Join Best GNN + Best LM

`configs/ablation/phase13/` — join the best GNN-only block (N48: jknet pool + gnn_plus + elu + ffn + skip) with the best Phase 8 LM aggregation (H10: UniXcoder sliding-window stride1024 + WindowMixerPool, localization=both concat). Tests whether the two architectural winners stack. O1 = hidden_dim 768 (from H10) → jknet pool 4×768=3072D dominates the 768D LM 4:1 (GNN-dominant fusion). O2 (pending) = hidden_dim 256 (faithful N48) → 1024D GNN balanced vs 768D LM.

| Run        | Run ID                                      | Config                             | hidden | jknet pool | fused | GNN:LM |
| ---------- | ------------------------------------------- | ---------------------------------- | ------ | ---------- | ----- | ------ |
| O1         | `20260612_131926_lmgat_codebert_multiclass` | `O1_join_n48gnn_h10lm.yaml`        | 768    | 3072D      | 3840  | 4:1    |
| O2         | `20260613_081400_lmgat_codebert_multiclass` | `O2_join_n48gnn_h10lm_dim256.yaml` | 256    | 1024D      | 1792  | 1.3:1  |
| O1-nine    | `20260615_080653_lmgat_codebert_multiclass` | `O1_unixcoder_nine.yaml`           | 768    | 3072D      | 3840  | 4:1    |
| O2-nine    | `20260615_113136_lmgat_codebert_multiclass` | `O2_unixcoder_nine.yaml`           | 256    | 1024D      | 1792  | 1.3:1  |
| O1-nine-lm | `20260615_131334_lmgat_codebert_multiclass` | `O1_unixcoder_nine_lm.yaml`        | 768    | 3072D      | 3840  | 4:1    |
| O3 (cRT)   | `20260617_061505_lmgat_codebert_multiclass` | `O3_crt_o1.yaml`                   | 768    | 3072D       | 3840  | 4:1    |
| O4 (proj)  | `20260617_064203_lmgat_codebert_multiclass` | `O4_o1_balanced_proj.yaml`         | 768    | 768D (proj) | 1536  | 1:1    |
| O1-w0      | `20260617_113601_lmgat_codebert_multiclass` | `O1_join_n48gnn_h10lm_w0.yaml`     | 768    | 3072D       | 3840  | 4:1    |
| O1-vo      | `20260617_234136_lmgat_codebert_multiclass` | `vulnonly/O1_vulnonly.yaml`        | 768    | 3072D       | 3840  | 4:1    |
| O1-vo-jk   | `20260618_135742_lmgat_codebert_multiclass` | `vulnonly/O1_vulnonly_jkloc.yaml`  | 768    | 3072D       | 3840  | 4:1    |

**Nine variants** (O1-nine, O2-nine, O1-nine-lm) — same O1/O2 join, node+func LM swapped from `unixcoder-base` to `unixcoder-base-nine` (9 langs incl. C/C++). O1-nine/O2-nine swap **both** node + func LM; O1-nine-lm swaps **func_lm only** (node LM stays base, isolating the func-LM effect since H10's sliding-window encoder is the actual func_lm consumer). Tests whether the C-aware LM helps the join, after N48-nine (node-LM only) was flat-to-worse.

## Classification

| Run        | Val F1 | Test F1 | Test Acc | F1-w  | Prec  | Rec   | Prec-w | Rec-w | AUC-ROC | Conf. | Epochs |
| ---------- | ------ | ------- | -------- | ----- | ----- | ----- | ------ | ----- | ------- | ----- | ------ |
| O1         | 0.534  | 0.545   | 0.494    | 0.544 | 0.512 | 0.542 | 0.659  | 0.455 | 0.877   | 0.332 | 77     |
| O2         | 0.513  | 0.477   | 0.520    | 0.518 | 0.519 | 0.555 | 0.559  | 0.556 | 0.890   | 0.554 | 25     |
| O1-nine    | 0.518  | 0.464   | 0.527    | 0.526 | 0.511 | 0.521 | 0.572  | 0.538 | 0.887   | 0.495 | 40     |
| O2-nine    | 0.498  | 0.494   | 0.512    | 0.512 | 0.464 | 0.501 | 0.541  | 0.521 | 0.877   | 0.553 | 30     |
| O1-nine-lm | 0.537  | 0.509   | 0.535    | 0.536 | 0.510 | 0.549 | 0.547  | 0.540 | 0.910   | 0.550 | 31     |
| O3 (cRT)   | 0.546  | 0.520   | 0.535    | 0.529 | 0.570 | 0.513 | 0.547  | 0.541 | 0.868   | 0.860 | 30     |
| O4 (proj)  | 0.520  | 0.498   | 0.514    | 0.514 | 0.527 | 0.552 | 0.542  | 0.531 | 0.908   | 0.566 | 20     |
| O1-w0      | 0.529  | 0.524   | 0.525    | 0.526 | 0.533 | 0.532 | 0.578  | 0.500 | 0.872   | 0.434 | 47     |
| O1-vo      | 0.551  | **0.575**| 0.581    | 0.577 | 0.561 | 0.580 | 0.573  | 0.563 | 0.914   | 0.617 | 24     |
| O1-vo-jk   | 0.511  | 0.554   | 0.552    | 0.547 | 0.495 | 0.543 | 0.580  | 0.570 | 0.900   | 0.545 | 26     |

## Statement-Level Localization

| Run        | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ---------- | ----- | ------- | ------- | --------- | ---------- | ------------- |
| O1         | 0.707 | 0.878   | 0.966   | 0.258     | 0.439      | 0.027         |
| O2         | 0.980 | 0.871   | 0.972   | 0.230     | 0.447      | 0.036         |
| O1-nine    | 0.775 | 0.873   | 0.977   | 0.216     | 0.417      | 0.043         |
| O2-nine    | 0.720 | 0.873   | 0.972   | 0.201     | 0.422      | 0.049         |
| O1-nine-lm | 1.328 | 0.829   | 0.978   | 0.219     | 0.429      | 0.043         |
| O3 (cRT)   | 0.706 | 0.880   | 0.966   | 0.259     | 0.439      | 0.027         |
| O4 (proj)  | 0.608 | 0.871   | 0.978   | 0.236     | 0.397      | 0.034         |
| O1-w0      | 0.580 | 0.845   | 0.972   | 0.237     | 0.438      | 0.035         |
| O1-vo      | 1.950 | 0.888   | 0.959   | 0.234     | 0.428      | 0.037         |
| O1-vo-jk   | 1.620 | 0.731   | 0.930   | 0.199     | 0.430      | 0.050         |

**O1 (N48 GNN block + H10 LM aggregation, GNN-dominant 768)** — Test F1 0.524, tied with N48 (0.525), **below H10 (0.534)**. The two bests **do NOT stack** — the join lands at the GNN level, not above. Likely cause: jknet at hidden 768 makes the GNN pool 3072D drown the 768D LM 4:1 in the fused vector, so classification tracks the GNN branch (N48-like) and loses H10's LM edge. **Best F1-w of all runs (0.544)** with very high weighted precision (0.659) — strong on head classes — but macro 0.524 means the tail is weaker, and Val 0.534 vs Test 0.524 shows an overfit gap. Motivates **O2 (hidden 256, balanced 1024:768 fusion)** to stop the GNN dominating. 156.5M params, 20.7 GB (matched the ~20-24 GB estimate), 9 hr / 77 ep on RTX 5000 Ada.

**O2 (balanced fusion, hidden 256 = faithful N48)** — Test F1 **0.477**, **below O1 (0.524) and H10 (0.534)** — the balance hypothesis FAILED: shrinking the GNN to 256 (1024:768 fusion) did NOT beat O1's GNN-dominant 768. It DID improve the head-class metrics (Acc 0.520 vs O1 0.494, Rec 0.555, AUC 0.890 vs 0.877, Conf 0.554) and slashed cost (131M params, **9.9 GB** vs O1 20.7, 339s vs 422s) — but macro-F1 dropped 0.047, so the smaller GNN lost tail-class discrimination. Caveat: early-stopped at **25 epochs** (vs O1 77) — possibly undertrained; a longer-patience rerun might recover. Net: the O-series join does not beat H10 0.534 at either fusion balance — combining the two architectural winners is F1-neutral-to-negative, the LM aggregation (H10) alone is the better model. **O-series closed: join ≤ H10.**

**O1-vo (vuln-only 25-class) — fair head-to-head vs vuln-only baselines.** Same O1 join (jknet 768 + H10 LM, localization=both concat) but benign dropped, `num_classes 25`, ml5120, `num_workers 0` — matched to the vuln-only baseline label space (LOSVER, VulExplainer, LIVABLE). Test F1 **0.552** (Acc 0.581, F1-w 0.577, AUC 0.914, Conf 0.617). Vs the vuln-only baselines: above VulExplainer macro **0.576**? no — **below** LOSVER 0.580 and VulExplainer 0.576 on macro, but **higher accuracy** (0.581 vs LOSVER 0.620 — LOSVER still leads acc) and the best confidence. Key contrast: the **GNN-only N48-vo scores 0.601 macro** (ABLATION_GNN_ONLY.md), so on vuln-only the join again **does not beat the GNN-only block** (0.552 < 0.601), same pattern as the 26-class O1≤N48. The hybrid's value is localization (LM line scores), not macro-F1 — N48-vo is the multiclass headline, O1-vo the localization-capable variant. Early stop at 24 epochs. 156.5M params, 195s/ep, 1.31 hr on RTX 5090.

**Nine LM swap (O1-nine, O2-nine, O1-nine-lm)** — swapping the node+func LM to `unixcoder-base-nine` (C/C++ aware) does **not** help the join. O1-nine **0.464** vs O1 base **0.524** (−0.060) — the full node+func swap clearly hurts. O1-nine-lm **0.509** (func_lm-only swap, node stays base) is the least harmful of the trio and posts the **best AUC of the whole O-series (0.910)** plus highest val F1 (0.537), but still trails base O1 (0.524) and H10 (0.534) on macro-F1, with the worst localization IFA (1.328 vs base 0.707). O2-nine **0.494** edges out O2 base **0.477** (+0.017) at hidden 256, but O2 base was undertrained (25 ep) so this is not a clean win, and it stays below O1. Net: the C-aware LM is flat-to-worse everywhere — consistent with N48-nine — and the func-only swap (nine-lm) is strictly better than the full swap when nine is used at all. **Stay on `unixcoder-base`; do not adopt nine for the headline.**

---

# Training Efficiency

| Run                      | GPU             | Params | Epoch Time | Total Time (hr) | VRAM Peak |
| ------------------------ | --------------- | ------ | ---------- | --------------- | --------- |
| A1                       | RTX 5070 Ti     | 3.5M   | 48s        | 0.73            | 4.2 GB    |
| A2                       | RTX 5070 Ti     | 129.6M | 216s       | 3.24            | 4.2 GB    |
| A3                       | RTX 5070 Ti     | 129.6M | 175s       | 3.70            | 6.0 GB    |
| A4                       | RTX 5070 Ti     | 129.6M | 176s       | 3.62            | 4.8 GB    |
| B1 fusion gated          | RTX 5070 Ti     | 129.6M | 163s       | 1.72            | 7.8 GB    |
| B2 fusion weighted α=0.3 | RTX 5070 Ti     | 129.6M | 162s       | 1.53            | 7.3 GB    |
| B3 fusion weighted α=0.5 | RTX 5070 Ti     | 129.6M | 162s       | 1.75            | 7.4 GB    |
| B4 fusion weighted α=0.7 | RTX 5070 Ti     | 129.6M | 162s       | 1.44            | 7.1 GB    |
| C1                       | RTX 5070 Ti     | 129.6M | 161s       | 1.39            | 6.9 GB    |
| C2                       | RTX 5070 Ti     | 129.6M | 162s       | 1.93            | 7.7 GB    |
| C2-fixed                 | RTX 5070 Ti     | 129.6M | 162s       | 3.37            | 7.7 GB    |
| D1 pool attention        | RTX 5070 Ti     | 129.6M | 162s       | 2.25            | 7.4 GB    |
| D2 pool meanmax          | RTX 5070 Ti     | 129.6M | 162s       | 2.16            | 7.2 GB    |
| D3 pool dualflow         | RTX 5070 Ti     | 129.6M | 161s       | 1.71            | 7.8 GB    |
| E1 cross_attn            | RTX 5070 Ti     | 131.7M | 166s       | 1.44            | 7.0 GB    |
| E2 self_attn             | RTX 5070 Ti     | 131.0M | 164s       | 2.51            | 7.9 GB    |
| E4 mmoe+taskenc          | RTX 5070 Ti     | 131.4M | 166s       | 1.85            | 7.9 GB    |
| E5 mmoe taskenc thin     | RTX 5070 Ti     | 131.1M | 166s       | 1.61            | 7.7 GB    |
| E6 crossattn no-resid    | RTX 5070 Ti     | 131.7M | 166s       | 4.24            | 8.6 GB    |
| E7 selfattn no-resid     | RTX 5070 Ti     | 131.0M | 164s       | 2.64            | 7.0 GB    |
| F2 node=CT5+             | RTX 5070 Ti     | 128.6M | 159s       | 2.13            | 7.1 GB    |
| F3 func=CT5+             | RTX 5070 Ti     | 138.0M | 476s       | 8.87            | 9.9 GB    |
| F4 both=CT5+             | RTX 5070 Ti     | 137.0M | 323s       | 5.94            | 9.1 GB    |
| F5 CT5+ raw              | RTX 5070 Ti     | 138.2M | 457s       | 6.86            | 8.4 GB    |
| F6 CT5+ norm             | RTX 5070 Ti     | 138.0M | 460s       | 4.35            | 8.7 GB    |
| F7 both normed           | RTX 5070 Ti     | 138.0M | 460s       | 8.96            | 8.9 GB    |
| G2 dim=768               | RTX 5070 Ti     | 146.9M | 409s       | 3.87            | 10.2 GB   |
| H2 sliding stride512     | RTX 5090        | 146.9M | 250s       | 2.09            | 18.6 GB   |
| H3 sliding stride1024    | RTX 5090        | 146.9M | 150s       | 1.30            | 17.1 GB   |
| H4 winattn stride1024    | RTX 5090        | 146.9M | 211s       | 1.29            | 18.0 GB   |
| H5 winattn stride512     | RTX PRO 6000 Bk | 146.9M | 265s       | 2.51            | 19.9 GB   |
| H6 winattn hidden        | RTX PRO 6000 Bk | 146.9M | 162s       | 1.40            | 17.1 GB   |
| H7 centerweight s512     | RTX 5090        | 146.9M | 328s       | 3.10            | 19.8 GB   |
| H8 crosswin s512         | RTX PRO 6000 Bk | 149.3M | 332s       | 3.78            | 24.2 GB   |
| H9 crosswin s1024        | RTX 5090        | 149.3M | 196s       | 3.00            | 21.4 GB   |
| H10 winmixer s1024       | RTX 5090        | 147.5M | 192s       | 1.39            | 17.1 GB   |
| H10n node+func nine      | RTX 5090        | 147.5M | 211s       | 2.53            | 20.9 GB   |
| H10fn func nine only     | RTX 5090        | 147.5M | 211s       | 2.35            | 21.6 GB   |
| I2 line frozen           | RTX 5090        | 161.7M | 95s        | 0.61            | 18.3 GB   |
| I3 line live             | RTX 5090        | 161.7M | 205s       | 1.66            | 20.1 GB   |
| I4 line ctx±5 frozen     | RTX 5090        | 161.7M | 84s        | 1.22            | 17.4 GB   |
| J3 ModernBERT            | RTX 6000 Bk     | 170.0M | 74s        | 1.14            | 21.3 GB   |
| K2 supcon w0.2           | RTX 5090        | 147.3M | 190s       | 1.75            | 19.7 GB   |
| K5 supcon group          | RTX 5090        | 147.3M | 190s       | 1.59            | 17.5 GB   |
| K6 supcon balanced       | RTX 5090        | 147.3M | 188s       | 1.68            | 17.6 GB   |
| M1 ulmfit llrd           | RTX 5090        | 146.9M | 189s       | 1.31            | 18.9 GB   |
| M2 ulmfit gradual        | RTX 5090        | 146.9M | 171s       | 2.76            | 17.5 GB   |
| M3 ulmfit combined       | RTX 5090        | 146.9M | 163s       | 1.81            | 21.6 GB   |
| D4 pool cnn              | RTX 5090        | 130.8M | 92s        | 0.95            | 8.1 GB    |
| O1 join n48gnn h10lm     | RTX 5000 Ada    | 156.5M | 422s       | 9.02            | 20.7 GB   |
| O2 join dim256 balanced  | RTX 5000 Ada    | 131.3M | 339s       | 2.36            | 9.9 GB    |
| O1-nine node+func nine   | RTX 5090        | 156.5M | 195s       | 2.17            | 22.8 GB   |
| O2-nine node+func nine   | RTX 5090        | 131.3M | 173s       | 1.45            | 10.7 GB   |
| O1-nine-lm func nine     | RTX 5090        | 156.5M | 195s       | 1.68            | 18.6 GB   |
| O3 cRT on O1             | RTX 5090        | 156.5M | 50s        | 0.42            | 8.8 GB    |
| O4 O1 balanced proj      | RTX 5090        | 158.8M | 195s       | 1.08            | 25.3 GB   |
| O1-w0 num_workers0       | RTX 5090        | 156.5M | 217s       | 2.84            | 20.1 GB   |
| O1-vo 25-class           | RTX 5090        | 156.5M | 195s       | 1.31            | 24.4 GB   |
| O1-vo-jk 25-class        | RTX 5090        | 156.5M | 186s       | 1.35            | 20.4 GB   |
| H10-vo 25-class          | RTX 5090        | 147.5M | 188s       | 2.04            | 20.1 GB   |
| H10-w0 num_workers0      | RTX 5090        | 147.5M | 209s       | 1.92            | 18.6 GB   |

