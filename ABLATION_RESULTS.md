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
| A4  | 0.550  | 0.504   | 0.046  | 8.4%  |

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
| B0  | (= A4-L1 baseline, `20260514_174326`)       | —                           | concat                 |
| B1  | `20260515_120709_lmgat_codebert_multiclass` | `B1_both_gated.yaml`        | gated                  |
| B2  | `20260515_135412_lmgat_codebert_multiclass` | `B2_both_weighted_a03.yaml` | weighted (GNN-leaning) |
| B3  | `20260515_165955_lmgat_codebert_multiclass` | `B3_both_weighted_a05.yaml` | weighted (balanced)    |
| B4  | `20260515_152942_lmgat_codebert_multiclass` | `B4_both_weighted_a07.yaml` | weighted (LM-leaning)  |

| ID  | Variant        | Test F1   | Test Acc  | F1-w  | IFA ↓     | Top-1 ↑   | R@20%LOC ↑ |
| --- | -------------- | --------- | --------- | ----- | --------- | --------- | ---------- |
| B0  | concat         | **0.519** | 0.518     | 0.517 | 0.789     | 0.887     | 0.403      |
| B1  | gated          | 0.483     | 0.526     | 0.525 | 1.138     | 0.851     | **0.422**  |
| B2  | weighted α=0.3 | 0.480     | 0.533     | 0.533 | **0.644** | 0.876     | 0.414      |
| B3  | weighted α=0.5 | 0.518     | **0.539** | 0.538 | 1.007     | **0.890** | 0.400      |
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
| A4-L1             | `20260514_174326_lmgat_codebert_multiclass` | no focal + epoch_adaptive + label_smoothing=0.1, wd=1e-3, cosine, patience=15 |
| A4-L2             | `20260515_052704_lmgat_codebert_multiclass` | LIVABLE two-branch (focal+LSCE), wd=1e-3, cosine, patience=15                 |
| A4-L2-fixed       | `20260515_084125_lmgat_codebert_multiclass` | A4-L2, no early stopping (full T-schedule)                                    |

## Classification

| Variant     | Val F1    | Test F1   | Test Acc  | F1-w  | AUC-ROC   | Conf.     | Epochs |
| ----------- | --------- | --------- | --------- | ----- | --------- | --------- | ------ |
| A4          | 0.550     | 0.504     | 0.507     | 0.503 | 0.899     | 0.813     | 74     |
| A4-L1       | **0.560** | **0.519** | 0.518     | 0.517 | **0.915** | **0.630** | 31     |
| A4-L2       | **0.561** | 0.475     | 0.529     | 0.526 | 0.904     | 0.757     | 43     |
| A4-L2-fixed | —         | 0.497     | **0.550** | —     | —         | —         | 75     |

## Localization

| Variant     | IFA ↓     | Top-1 ↑   | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ----------- | --------- | --------- | ------- | --------- | ---------- | ------------- |
| A4          | 1.26      | 0.794     | 0.959   | 0.207     | 0.431      | 0.047         |
| A4-L1       | **0.789** | **0.887** | 0.965   | 0.238     | 0.403      | **0.031**     |
| A4-L2       | 0.867     | 0.817     | 0.949   | **0.256** | 0.476      | 0.029         |
| A4-L2-fixed | 1.277     | —         | —       | —         | **0.492**  | —             |

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

| Variant   | Run ID                                      | graph_pool | Epochs |
| --------- | ------------------------------------------- | ---------- | ------ |
| mean      | (= A4-L1, `20260514_174326`)                | mean       | 31     |
| attention | `20260515_235912_lmgat_codebert_multiclass` | attention  | 50     |
| meanmax   | `20260516_125619_lmgat_codebert_multiclass` | meanmax    | 48     |
| dualflow  | `20260517_013824_lmgat_codebert_multiclass` | dualflow   | 38     |

| Variant   | Test F1   | Test Acc  | F1-w      | AUC-ROC   | Conf. | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@20%LOC ↑ |
| --------- | --------- | --------- | --------- | --------- | ----- | --------- | --------- | --------- | ---------- |
| mean      | **0.519** | 0.518     | 0.517     | **0.915** | 0.630 | 0.789     | 0.887     | 0.965     | 0.403      |
| attention | 0.437     | 0.522     | 0.523     | 0.895     | 0.625 | 1.253     | 0.805     | 0.943     | 0.439      |
| meanmax   | 0.517     | **0.538** | **0.539** | 0.911     | 0.502 | **0.644** | **0.900** | **0.982** | **0.487**  |
| dualflow  | 0.496     | 0.528     | 0.528     | 0.896     | 0.667 | 0.717     | 0.886     | 0.971     | 0.417      |

Attention pool collapses macro F1 (−0.082) — learnable gate overfits tail classes. Mean and meanmax tie on macro F1 but **meanmax wins everywhere else**: best accuracy, F1-w, and all localization metrics. dualflow lands mid-pack (F1 0.496) with same overfit risk as attention but milder.

**Phase 4 winner: meanmax.**

---

# Phase 5 — Multi-Task / Cross-Task

`configs/ablation/phase5/` — bidirectional cross-task between localization
(stmt_head) and classification (func_head). Zero-init residual gates
(ReZero/ControlNet style) — module starts as a no-op.
**Baseline E0 = A4-L1** (Phase 3 winner, no cross-task).
All E-series runs use `graph_pool=mean` (same as E0/A4-L1 baseline) — Phase 4
meanmax was found separately. Cross-task comparison is internally consistent.

| ID  | Run ID                                      | Config                         | cross_task_method               | Epochs    |
| --- | ------------------------------------------- | ------------------------------ | ------------------------------- | --------- |
| E0  | `20260514_174326_lmgat_codebert_multiclass` | —                              | none (= A4-L1 baseline)         | 31        |
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
| E0  | none (A4-L1)                  | 0.519     | 0.518     | 0.517     | 0.915     | 0.630 |
| E1  | cross_attention               | **0.530** | 0.532     | 0.533     | **0.919** | 0.615 |
| E2  | self_attention                | 0.504     | **0.538** | **0.537** | 0.897     | 0.606 |
| E4  | mmoe + task encoder           | 0.479     | 0.535     | 0.535     | 0.883     | 0.620 |
| E5  | mmoe + taskenc + thin         | 0.480     | 0.509     | 0.509     | 0.835     | 0.658 |
| E6  | cross_attention, residual off | 0.375     | 0.377     | 0.379     | 0.882     | 0.300 |
| E7  | self_attention, residual off  | 0.414     | 0.433     | 0.433     | 0.863     | 0.443 |

## Localization

| ID  | Method                        | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --- | ----------------------------- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| E0  | none (A4-L1)                  | 0.789     | **0.887** | 0.965     | 0.238     | 0.403      | 0.031         |
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
| F1 (= meanmax)          | 0.517     | 0.538     | 0.539     | 0.911     | 0.644     | 0.900     | 0.982     | 0.269     | 0.487      | 0.025         |
| F2 (node=CT5+)          | **0.502** | 0.554     | 0.552     | **0.906** | 0.745     | 0.899     | 0.982     | 0.232     | 0.415      | 0.032         |
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
| G1  | 0.517     | 0.538     | 0.539     | 0.911     | 0.502 | 48     |
| G2  | **0.529** | **0.582** | **0.579** | **0.914** | 0.569 | 34     |

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

| ID  | Config                                                           | chunk | stride | max_len | Max windows     | Run ID            | Epochs |
| --- | ---------------------------------------------------------------- | ----- | ------ | ------- | --------------- | ----------------- | ------ |
| H1  | — (= G2)                                                         | —     | —      | 1024    | 1               | `20260520_132730` | 34     |
| H2  | `H2_unixcoder_sliding_chunk1024_stride512.yaml`                  | 1024  | 512    | 5120    | 9               | `20260525_104032` | 30     |
| H3  | `H3_unixcoder_sliding_chunk1024_stride1024.yaml`                 | 1024  | 1024   | 5120    | 5               | `20260525_125031` | 31     |
| H4  | `H4_unixcoder_sliding_chunk1024_stride1024_winattn.yaml`         | 1024  | 1024   | 5120    | 5+attn          | `20260527_121315` | 22     |
| H5  | `H5_unixcoder_sliding_chunk1024_stride512_winattn.yaml`          | 1024  | 512    | 5120    | 9+attn          | `20260528_062323` | 34     |
| H6  | `H6_unixcoder_sliding_chunk1024_stride1024_winattn_hidden.yaml`  | 1024  | 1024   | 5120    | 5+attn+hidden   | `20260528_085945` | 31     |
| H7  | `H7_unixcoder_sliding_chunk1024_stride512_winattn_centerw.yaml`  | 1024  | 512    | 5120    | 9+attn+cw       | `20260528_063142` | 34     |
| H8  | `H8_unixcoder_sliding_chunk1024_stride512_winattn_crosswin.yaml` | 1024  | 512    | 5120    | 9+attn+crosswin | `20260528_094016` | 41     |

## Classification

| ID        | Test F1 | Test Acc | F1-w  | AUC-ROC   | Conf. | Epochs |
| --------- | ------- | -------- | ----- | --------- | ----- | ------ |
| H1 (= G2) | 0.529   | 0.582    | 0.579 | 0.914     | 0.569 | 34     |
| H2        | 0.459   | 0.508    | 0.507 | 0.890     | 0.588 | 30     |
| H3        | 0.528   | 0.529    | 0.533 | 0.895     | 0.587 | 31     |
| H4        | 0.520   | 0.563    | 0.560 | **0.927** | 0.695 | 22     |
| H5        | 0.443   | 0.522    | 0.522 | 0.885     | 0.589 | 34     |
| H6        | 0.513   | 0.532    | 0.533 | 0.903     | 0.607 | 31     |
| H7        | 0.485   | 0.524    | 0.525 | 0.896     | 0.618 | 34     |
| H8        | 0.520   | 0.536    | 0.538 | 0.898     | 0.584 | 41     |

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

**Phase 8 winner: H4 — best AUC 0.927 (vs H1 0.914, +0.013). F1 0.520 ties H8 but below H3 (0.528) and H1 baseline (0.529) — window attn improves ranking quality, not macro F1. H6 best localization IFA (1.034). H8 best Top-1 (0.912) + strongest classification recovery over H5. H3 best Effort (0.041). H4 baseline for Phase 11+.**

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
| J3           | 0.386  | 0.378     | 0.426     | 0.422     | 0.847     | 0.397     | 55     |

## Statement-Level Localization

| ID           | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ------------ | --------- | --------- | --------- | --------- | ---------- | ------------- |
| J1 (= H1/G2) | 1.410     | 0.747     | 0.962     | 0.186     | **0.427**  | 0.056         |
| J3           | **0.950** | **0.814** | **0.978** | **0.209** | 0.422      | **0.046**     |

**J3 (ModernBERT-base, 5120-token native)** — localization improves (IFA 0.950 vs 1.410, −33%; Top-1 0.814 vs 0.747) but classification collapses (F1 0.378, −0.151; AUC 0.847 vs 0.914). Alternating local/global attention fails to produce function-level CWE semantics that UniXcoder's full bidirectional attention computes in a single pass.

**Phase 10 finding: ModernBERT-base is a localization/classification trade-off, not an improvement.** UniXcoder with sliding window (H4, F1=0.554) remains best overall.

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
| K1 (= H4) | 0.573  | 0.520   | 0.563    | 0.560 | 0.927   | 0.695 | 22     |
| K2        | 0.502  | 0.461   | 0.527    | 0.527 | 0.874   | 0.607 | 33     |
| K5        | 0.521  | 0.484   | 0.550    | 0.551 | 0.897   | 0.608 | 30     |
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

# GNN-only Ablation (N series)

`configs/ablation/gnn_only/` — `live_lm=none` (no LM forward, GNN-only mode). Base = A1 + Phase 3 L1 loss settings (drop focal γ=0, label_smoothing=0.1, cosine, wd=1e-3, patience=15). Varies graph pooling, residual, GNN block order. Purpose: isolate GNN architectural effects without LM noise.

| ID  | Run ID                                      | Config                                              | graph_pool | use_skip | block_style                        |
| --- | ------------------------------------------- | --------------------------------------------------- | ---------- | -------- | ---------------------------------- |
| N1  | `20260530_194019_lmgat_codebert_multiclass` | `N1_a1_l1.yaml`                                     | mean       | false    | resnet                             |
| N2  | `20260530_203000_lmgat_codebert_multiclass` | `N2_a1_l1_meanmax.yaml`                             | meanmax    | false    | resnet                             |
| N3  | `20260530_213022_lmgat_codebert_multiclass` | `N3_a1_l1_cnn.yaml`                                 | cnn        | false    | resnet                             |
| N4  | `20260530_225801_lmgat_codebert_multiclass` | `N4_a1_l1_meanmax_residual.yaml`                    | meanmax    | true     | resnet                             |
| N5  | `20260531_001117_lmgat_codebert_multiclass` | `N5_a1_l1_gnn_plus.yaml`                            | meanmax    | true     | gnn_plus                           |
| N6  | `20260531_033903_lmgat_codebert_multiclass` | `N6_a1_l1_gnn_plus_graphnorm.yaml`                  | meanmax    | true     | gnn_plus + GraphNorm               |
| N7  | `20260531_042425_lmgat_codebert_multiclass` | `N7_a1_l1_gnn_plus_elu.yaml`                        | meanmax    | true     | gnn_plus + ELU                     |
| N8  | `20260531_064326_lmgat_codebert_multiclass` | `N8_a1_l1_gnn_plus_graphnorm_elu.yaml`              | meanmax    | true     | gnn_plus + GraphNorm + ELU         |
| N9  | `20260531_081742_lmgat_codebert_multiclass` | `N9_a1_l1_gnn_plus_elu_ffn.yaml`                    | meanmax    | true     | gnn_plus + ELU + FFN               |
| N10 | `20260531_110214_lmgat_codebert_multiclass` | `N10_a1_l1_gnn_plus_elu_ffn_pe.yaml`                | meanmax    | true     | gnn_plus + ELU + FFN + RWSE-32 PE  |
| N11 | `20260531_142518_lmgat_codebert_multiclass` | `N11_a1_l1_gnn_plus_elu_dim512.yaml`                | meanmax    | true     | gnn_plus + ELU + hidden_dim=512    |
| N12 | `20260531_154612_lmgat_codebert_multiclass` | `N12_a1_l1_gnn_plus_elu_dim768.yaml`                | meanmax    | true     | gnn_plus + ELU + hidden_dim=768    |
| N13 | `20260531_144339_lmgat_codebert_multiclass` | `N13_a1_l1_gnn_plus_elu_balo.yaml`                  | meanmax    | true     | gnn_plus + ELU + BalO init         |
| N14 | `20260531_212716_lmgat_codebert_multiclass` | `N14_a1_l1_gnn_plus_elu_dim512_balo.yaml`           | meanmax    | true     | N11 dim=512 + N13 BalO             |
| N15 | `20260531_204352_lmgat_codebert_multiclass` | `N15_a1_l1_gnn_plus_elu_ffn_linhead.yaml`           | meanmax    | true     | N9 FFN + linear func head (GNN+)   |
| N16 | `20260601_055452_lmgat_codebert_multiclass` | `N16_a1_l1_gnn_plus_elu_ffn_linhead_balo.yaml`      | meanmax    | true     | N15 + BalO init                    |
| N17 | `20260601_065912_lmgat_codebert_multiclass` | `N17_a1_l1_gnn_plus_elu_ffn_meanpool.yaml`          | mean       | true     | N15 + mean pool                    |
| N18 | `20260601_055453_lmgat_codebert_multiclass` | `N18_a1_l1_gnn_plus_elu_ffn_addpool.yaml`           | add        | true     | N15 + add pool                     |
| N19 | `20260601_075846_lmgat_codebert_multiclass` | `N19_a1_l1_gnn_plus_elu_ffn_maxpool.yaml`           | max        | true     | N15 + max pool                     |
| N20 | `20260601_110545_lmgat_codebert_multiclass` | `N20_a1_l1_gnn_plus_elu_ffn_linhead_ginit.yaml`     | meanmax    | true     | N15 + G-Init (Kelesis 2024)        |
| N21 | `20260601_115551_lmgat_codebert_multiclass` | `N21_a1_l1_gnn_plus_elu_ffn_linhead_lsuv.yaml`      | meanmax    | true     | N15 + LSUV init (Mishkin 2016)     |
| N22 | `20260601_144252_lmgat_codebert_multiclass` | `N22_a1_l1_gnn_plus_elu_ffn_linhead_L3.yaml`        | meanmax    | true     | N15 + num_layers=3 (depth-1)       |
| N23 | `20260601_153557_lmgat_codebert_multiclass` | `N23_a1_l1_gnn_plus_elu_ffn_linhead_L5.yaml`        | meanmax    | true     | N15 + num_layers=5 (depth+1)       |
| N24 | `20260601_165353_lmgat_codebert_multiclass` | `N24_a1_l1_gnn_plus_elu_ffn_linhead_L6.yaml`        | meanmax    | true     | N15 + num_layers=6 (depth+2)       |
| N25 | `20260601_210541_lmgat_codebert_multiclass` | `N25_a1_l1_gnn_plus_elu_ffn_linhead_attnpool.yaml`  | attention  | true     | N15 + attention pool               |
| N26 | `20260601_220205_lmgat_codebert_multiclass` | `N26_a1_l1_gnn_plus_elu_ffn_linhead_crossattn.yaml` | meanmax    | true     | N15 + cross-task attention         |
| N27 | `20260601_225416_lmgat_codebert_multiclass` | `N27_a1_l1_gnn_plus_elu_ffn_linhead_kendall.yaml`   | meanmax    | true     | N15 + Kendall uncertainty MTL      |
| N28 | `20260601_232916_lmgat_codebert_multiclass` | `N28_a1_l1_gnn_plus_elu_ffn_linhead_pcgrad.yaml`    | meanmax    | true     | N15 + PCGrad (Yu 2020)             |
| N29 | `20260602_133832_lmgat_codebert_multiclass` | `N29_a1_l1_gnn_plus_elu_ffn_linhead_diagnose.yaml`  | meanmax    | true     | N15 + MTL diagnostics (num_workers=0 rerun) |
| N30 | `20260602_153635_lmgat_codebert_multiclass` | `N30_a1_l1_gnn_plus_elu_ffn_linhead_dualflow.yaml`  | dualflow   | true     | N15 + dualflow pool (num_workers=0 rerun)      |
| N31 | `20260602_160417_lmgat_codebert_multiclass` | `N31_a1_l1_gnn_plus_elu_ffn_linhead_heads2.yaml`    | meanmax    | true     | N15 + heads=2 (num_workers=0 rerun)            |
| N32 | `20260602_155136_lmgat_codebert_multiclass` | `N32_a1_l1_gnn_plus_elu_ffn_linhead_heads8.yaml`    | meanmax    | true     | N15 + heads=8 GATv2 default (num_workers=0 rerun) |
| N33 | `20260602_170600_lmgat_codebert_multiclass` | `N33_a1_l1_gnn_plus_elu_ffn_linhead_heads16.yaml`   | meanmax    | true     | N15 + heads=16 (num_workers=0 rerun)           |
| N34 | `20260603_121533_lmgat_codebert_multiclass` | `N34_a1_l1_gnn_plus_elu_ffn_linhead_norank.yaml`    | meanmax    | true     | N15 + rank_loss_weight=0 (drop rank)           |
| N35 | `20260603_124325_lmgat_codebert_multiclass` | `N35_a1_l1_gnn_plus_elu_ffn_linhead_rank01.yaml`    | meanmax    | true     | N15 + rank_loss_weight=0.1 (halve rank)        |
| N36 | `20260603_131456_lmgat_codebert_multiclass` | `N36_a1_l1_gnn_plus_elu_ffn_linhead_pcgrad_enc.yaml`| meanmax    | true     | N15 + PCGrad encoder-only (N28b fix)           |

## Classification

Macro = unweighted mean across 26 classes (each CWE = equal weight).
Weighted = mean weighted by class support (frequent CWEs dominate).
For vuln detection: **macro recall** is primary — measures how well we catch each CWE type.

| ID  | Val F1    | Test F1   | Test Acc  | F1-w      | Prec      | Rec       | Prec-w    | Rec-w     | AUC-ROC   | Conf.     | Epochs |
| --- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------ |
| N1  | 0.490     | 0.433     | 0.469     | 0.466     | 0.395     | 0.459     | 0.458     | 0.461     | 0.843     | **0.583** | 63     |
| N2  | 0.486     | 0.457     | 0.490     | 0.487     | 0.500     | 0.476     | 0.511     | 0.498     | 0.856     | 0.267     | 76     |
| N3  | 0.392     | 0.420     | 0.445     | 0.447     | 0.413     | 0.462     | 0.467     | 0.455     | 0.831     | 0.489     | 99     |
| N4  | 0.523     | 0.472     | 0.501     | 0.499     | 0.477     | 0.477     | 0.512     | 0.503     | 0.884     | 0.145     | 92     |
| N5  | 0.525     | 0.468     | 0.491     | 0.498     | 0.469     | 0.515     | 0.515     | 0.475     | 0.891     | 0.421     | 63     |
| N6  | **0.531** | 0.447     | 0.509     | 0.509     | 0.433     | 0.500     | 0.535     | 0.500     | **0.903** | 0.471     | 55     |
| N7  | 0.505     | 0.450     | **0.510** | **0.511** | 0.505     | 0.473     | **0.543** | **0.527** | 0.902     | 0.482     | 55     |
| N8  | 0.492     | 0.469     | 0.493     | 0.489     | 0.444     | 0.484     | 0.526     | 0.501     | 0.893     | 0.524     | 44     |
| N9  | 0.479     | 0.391     | 0.445     | 0.454     | 0.404     | 0.405     | 0.478     | 0.437     | 0.863     | 0.368     | 72     |
| N10 | 0.465     | 0.391     | 0.465     | 0.463     | 0.419     | 0.458     | 0.493     | 0.471     | 0.872     | 0.409     | 59     |
| N11 | 0.508     | 0.480     | 0.491     | 0.494     | 0.502     | 0.481     | 0.510     | 0.500     | 0.892     | 0.514     | 36     |
| N12 | 0.494     | 0.440     | 0.500     | 0.500     | 0.427     | 0.471     | 0.511     | 0.501     | 0.899     | 0.507     | 49     |
| N13 | 0.509     | 0.471     | 0.502     | 0.497     | 0.477     | 0.447     | **0.543** | 0.495     | 0.901     | 0.532     | 44     |
| N14 | 0.521     | 0.470     | 0.493     | 0.483     | 0.460     | 0.507     | 0.522     | 0.490     | 0.901     | 0.430     | 64     |
| N15 | 0.500     | **0.523** | 0.487     | 0.483     | 0.462     | 0.510     | 0.489     | 0.482     | 0.879     | 0.337     | 78     |
| N16 | 0.514     | 0.504     | 0.481     | 0.480     | 0.521     | **0.551** | 0.489     | 0.484     | 0.889     | 0.334     | 77     |
| N17 | 0.453     | 0.400     | 0.447     | 0.447     | 0.408     | 0.470     | 0.468     | 0.452     | 0.871     | 0.425     | 80     |
| N18 | 0.328     | 0.270     | 0.328     | 0.307     | 0.272     | 0.446     | 0.346     | 0.319     | 0.852     | 0.260     | 74     |
| N19 | 0.474     | 0.484     | 0.489     | 0.488     | 0.495     | 0.484     | 0.507     | 0.489     | 0.896     | 0.292     | 61     |
| N20 | 0.506     | 0.448     | 0.485     | 0.483     | **0.551** | 0.501     | 0.498     | 0.489     | 0.896     | 0.356     | 58     |
| N21 | 0.493     | 0.445     | 0.467     | 0.458     | 0.502     | 0.482     | 0.491     | 0.473     | 0.891     | 0.336     | 60     |
| N22 | 0.493     | 0.451     | 0.491     | 0.486     | 0.515     | 0.470     | 0.499     | 0.494     | 0.891     | 0.373     | 63     |
| N23 | 0.480     | 0.439     | 0.449     | 0.446     | 0.478     | 0.483     | 0.479     | 0.469     | 0.892     | 0.344     | 67     |
| N24 | 0.469     | 0.472     | 0.473     | 0.469     | 0.517     | 0.486     | 0.477     | 0.467     | 0.883     | 0.344     | 69     |
| N25 | 0.474     | 0.405     | 0.443     | 0.443     | 0.417     | 0.447     | 0.476     | 0.469     | 0.889     | 0.428     | 67     |
| N26 | 0.475     | 0.429     | 0.487     | 0.481     | 0.510     | 0.486     | 0.491     | 0.472     | 0.893     | 0.374     | 55     |
| N27 | 0.255     | 0.249     | 0.401     | 0.376     | 0.261     | 0.241     | 0.364     | 0.377     | 0.865     | 0.217     | 42     |
| N28 | 0.483     | 0.449     | 0.475     | 0.471     | 0.467     | 0.445     | 0.486     | 0.466     | 0.886     | 0.392     | 57     |
| N29 | 0.519     | 0.514     | 0.481     | 0.480     | 0.528     | 0.525     | 0.493     | 0.479     | 0.891     | 0.323     | 86     |
| N30 | 0.465     | 0.412     | 0.446     | 0.440     | 0.453     | 0.455     | 0.461     | 0.443     | 0.872     | 0.496     | 32     |
| N31 | 0.497     | 0.443     | 0.491     | 0.487     | 0.454     | 0.402     | 0.494     | 0.472     | 0.886     | 0.344     | 75     |
| N32 | 0.470     | 0.426     | 0.442     | 0.442     | 0.409     | 0.450     | 0.478     | 0.459     | 0.889     | 0.298     | 53     |
| N33 | 0.471     | 0.433     | 0.456     | 0.456     | 0.411     | 0.496     | 0.481     | 0.471     | 0.889     | 0.314     | 55     |
| N34 | 0.470     | 0.413     | 0.445     | 0.444     | 0.462     | 0.434     | 0.479     | 0.443     | 0.889     | 0.371     | 49     |
| N35 | 0.489     | 0.457     | 0.471     | 0.460     | 0.525     | 0.469     | 0.506     | 0.470     | 0.888     | 0.405     | 53     |
| N36 | 0.469     | 0.410     | 0.425     | 0.426     | 0.350     | 0.472     | 0.441     | 0.415     | 0.889     | 0.241     | 82     |

## Statement-Level Localization

| ID  | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| N1  | 1.199     | 0.845     | 0.955     | 0.227     | 0.410      | 0.037         |
| N2  | 1.312     | 0.799     | 0.934     | 0.232     | 0.430      | 0.034         |
| N3  | 1.618     | 0.807     | 0.933     | 0.222     | 0.430      | 0.039         |
| N4  | 1.672     | 0.714     | 0.931     | 0.242     | 0.449      | 0.033         |
| N5  | 0.697     | 0.835     | 0.969     | 0.257     | 0.478      | 0.032         |
| N6  | 0.654     | 0.829     | 0.965     | 0.248     | 0.480      | 0.029         |
| N7  | 0.474     | 0.918     | 0.985     | **0.274** | 0.485      | **0.023**     |
| N8  | 0.543     | 0.861     | 0.974     | 0.262     | 0.480      | 0.029         |
| N9  | 0.372     | 0.930     | 0.985     | 0.255     | 0.468      | 0.027         |
| N10 | 0.515     | 0.890     | 0.981     | 0.243     | 0.451      | 0.030         |
| N11 | 0.695     | 0.864     | 0.974     | 0.245     | 0.457      | 0.031         |
| N12 | 0.401     | 0.906     | 0.984     | 0.264     | 0.467      | 0.025         |
| N13 | 0.539     | 0.889     | 0.978     | 0.263     | 0.481      | 0.028         |
| N14 | 0.606     | 0.874     | 0.978     | 0.263     | **0.491**  | 0.027         |
| N15 | 0.299     | 0.940     | **0.988** | 0.271     | 0.475      | 0.024         |
| N16 | 0.385     | 0.911     | 0.981     | 0.240     | 0.451      | 0.036         |
| N17 | **0.220** | 0.928     | **0.988** | 0.257     | 0.464      | 0.029         |
| N18 | 0.485     | 0.895     | 0.978     | **0.285** | 0.488      | **0.023**     |
| N19 | 0.435     | 0.895     | 0.980     | 0.238     | 0.458      | 0.034         |
| N20 | 0.351     | 0.941     | **0.988** | 0.248     | 0.440      | 0.029         |
| N21 | 0.524     | 0.883     | 0.980     | 0.214     | 0.437      | 0.043         |
| N22 | 0.423     | 0.871     | 0.980     | 0.198     | 0.413      | 0.053         |
| N23 | 0.351     | 0.936     | 0.987     | 0.261     | 0.447      | 0.027         |
| N24 | 0.369     | 0.921     | 0.982     | 0.227     | 0.423      | 0.040         |
| N25 | 0.322     | **0.950** | 0.985     | 0.242     | 0.444      | 0.031         |
| N26 | 0.370     | 0.928     | 0.984     | 0.252     | 0.439      | 0.029         |
| N27 | 0.602     | 0.865     | 0.966     | 0.253     | **0.491**  | 0.031         |
| N28 | 0.613     | 0.884     | 0.977     | 0.238     | 0.449      | 0.034         |
| N29 | 0.271     | 0.934     | 0.987     | 0.262     | 0.449      | 0.027         |
| N30 | 0.375     | 0.940     | 0.984     | 0.225     | 0.423      | 0.038         |
| N31 | 0.529     | 0.889     | 0.977     | 0.231     | 0.455      | 0.039         |
| N32 | 0.508     | 0.915     | 0.984     | 0.246     | 0.458      | 0.032         |
| N33 | 0.490     | 0.924     | 0.982     | 0.260     | 0.456      | 0.026         |
| N34 | 9.826     | 0.269     | 0.593     | 0.076     | 0.224      | 0.175         |
| N35 | 0.581     | 0.852     | 0.974     | 0.200     | 0.438      | 0.050         |
| N36 | 0.423     | 0.909     | 0.981     | 0.256     | 0.466      | 0.029         |

---

# Training Efficiency

| Run                   | GPU             | Params | Epoch Time | Total Time (hr) | VRAM Peak |
| --------------------- | --------------- | ------ | ---------- | --------------- | --------- |
| A1                    | RTX 5070 Ti     | 3.5M   | 48s        | 0.73            | 4.2 GB    |
| A2                    | RTX 5070 Ti     | 129.6M | 216s       | 3.24            | 4.2 GB    |
| A3                    | RTX 5070 Ti     | 129.6M | 175s       | 3.70            | 6.0 GB    |
| A4                    | RTX 5070 Ti     | 129.6M | 176s       | 3.62            | 4.8 GB    |
| A4-L1                 | RTX 5070 Ti     | 129.6M | 161s       | 1.39            | 6.9 GB    |
| A4-L2                 | RTX 5070 Ti     | 129.6M | 162s       | 1.93            | 7.7 GB    |
| A4-L2-fixed           | RTX 5070 Ti     | 129.6M | 162s       | 3.37            | 7.7 GB    |
| fusion gated          | RTX 5070 Ti     | 129.6M | 163s       | 1.72            | 7.8 GB    |
| fusion weighted α=0.3 | RTX 5070 Ti     | 129.6M | 162s       | 1.53            | 7.3 GB    |
| fusion weighted α=0.5 | RTX 5070 Ti     | 129.6M | 162s       | 1.75            | 7.4 GB    |
| fusion weighted α=0.7 | RTX 5070 Ti     | 129.6M | 162s       | 1.44            | 7.1 GB    |
| pool attention        | RTX 5070 Ti     | 129.6M | 162s       | 2.25            | 7.4 GB    |
| pool meanmax          | RTX 5070 Ti     | 129.6M | 162s       | 2.16            | 7.2 GB    |
| B2 cross_attn         | RTX 5070 Ti     | 129.6M | 169s       | 2.72            | 7.1 GB    |
| B3 self_attn          | RTX 5070 Ti     | 129.6M | 245s       | 4.29            | 7.3 GB    |
| B4 mmoe               | RTX 5070 Ti     | 129.6M | 165s       | 1.83            | 7.9 GB    |
| F4 both=CT5+          | RTX 5070 Ti     | 137.0M | 323s       | 5.94            | 9.1 GB    |
| F5 CT5+ raw           | RTX 5070 Ti     | 138.2M | 457s       | 6.86            | 8.4 GB    |
| F6 CT5+ norm          | RTX 5070 Ti     | 138.0M | 460s       | 4.35            | 8.7 GB    |
| F7 both normed        | RTX 5070 Ti     | 138.0M | 460s       | 8.96            | 8.9 GB    |
| G2 dim=768            | RTX 5070 Ti     | 146.9M | 409s       | 3.87            | 10.2 GB   |
| H2 sliding stride512  | RTX 5090        | 146.9M | 250s       | 2.09            | 18.6 GB   |
| H3 sliding stride1024 | RTX 5090        | 146.9M | 150s       | 1.30            | 17.1 GB   |
| H4 winattn stride1024 | RTX 5090        | 146.9M | 211s       | 1.29            | 18.0 GB   |
| H5 winattn stride512  | RTX PRO 6000 Bk | 146.9M | 265s       | 2.51            | 19.9 GB   |
| H6 winattn hidden     | RTX PRO 6000 Bk | 146.9M | 162s       | 1.40            | 17.1 GB   |
| H7 centerweight s512  | RTX 5090        | 146.9M | 328s       | 3.10            | 19.8 GB   |
| H8 crosswin s512      | RTX PRO 6000 Bk | 149.3M | 332s       | 3.78            | 24.2 GB   |
| I2 line frozen        | RTX 5090        | 161.7M | 95s        | 0.61            | 18.3 GB   |
| I3 line live          | RTX 5090        | 161.7M | 205s       | 1.66            | 20.1 GB   |
| I4 line ctx±5 frozen  | RTX 5090        | 161.7M | 84s        | 1.22            | 17.4 GB   |
| J3 ModernBERT         | RTX 6000 Bk     | 170.0M | 74s        | 1.14            | 21.3 GB   |
| K2 supcon w0.2        | RTX 5090        | 147.3M | 190s       | 1.75            | 19.7 GB   |
| K5 supcon group       | RTX 5090        | 147.3M | 190s       | 1.59            | 17.5 GB   |
| K6 supcon balanced    | RTX 5090        | 147.3M | 188s       | 1.68            | 17.6 GB   |
| N1 a1+l1 mean         | RTX A4500       | 3.5M   | 47s        | 0.82            | 11.0 GB   |
| N2 a1+l1 meanmax      | RTX A4500       | 3.5M   | 47s        | 1.00            | 9.2 GB    |
| N3 a1+l1 cnn          | RTX A4500       | 4.7M   | 53s        | 1.45            | 9.1 GB    |
| N4 a1+l1 meanmax+skip | RTX A4500       | 3.7M   | 47s        | 1.21            | 9.6 GB    |
| N5 a1+l1 gnn_plus     | RTX A4500       | 3.7M   | 47s        | 0.83            | 11.0 GB   |
| N6 N5+GraphNorm       | RTX A4500       | 3.7M   | 49s        | 0.75            | 9.5 GB    |
| N7 N5+ELU             | RTX A4500       | 3.7M   | 47s        | 0.72            | 9.3 GB    |
| N8 N5+GraphNorm+ELU   | RTX A4500       | 3.7M   | 50s        | 0.61            | 10.7 GB   |
| N9 N7+FFN             | RTX A4500       | 4.8M   | 50s        | 1.01            | 10.3 GB   |
| N10 N9+RWSE-32 PE     | RTX A4500       | 4.9M   | 52s        | 0.85            | 12.4 GB   |
| N11 N7+dim512         | RTX A6000       | 10.7M  | 95s        | 0.95            | 31.2 GB   |
| N12 N7+dim768         | RTX A6000       | 21.0M  | 118s       | 1.61            | 29.8 GB   |
| N13 N7+BalO init      | RTX A5000       | 3.7M   | 57s        | 0.70            | 10.5 GB   |
| N14 N11+BalO          | RTX A6000       | 10.7M  | 87s        | 1.55            | 17.5 GB   |
| N15 N9+linear head    | RTX A4500       | 4.7M   | 54s        | 1.17            | 8.8 GB    |
| N16 N15+BalO          | RTX A4500       | 4.7M   | 50s        | 1.07            | 12.4 GB   |
| N17 N15+mean pool     | RTX A4500       | 4.7M   | 50s        | 1.10            | 10.2 GB   |
| N18 N15+add pool      | RTX 4060 Ti     | 4.7M   | 100s       | 2.05            | 10.8 GB   |
| N19 N15+max pool      | RTX 4060 Ti     | 4.7M   | 98s        | 1.67            | 9.8 GB    |
| N20 N15+G-Init        | RTX A4500       | 4.7M   | 51s        | 0.83            | 10.5 GB   |
| N21 N15+LSUV          | RTX A4500       | 4.7M   | 51s        | 0.84            | 10.1 GB   |
| N22 N15+L=3           | RTX A4500       | 3.9M   | 50s        | 0.88            | 9.1 GB    |
| N23 N15+L=5           | RTX A4500       | 5.5M   | 69s        | 1.29            | 11.2 GB   |
| N24 N15+L=6           | RTX A4500       | 6.4M   | 79s        | 1.52            | 12.9 GB   |
| N25 N15+attn pool     | RTX A4500       | 4.7M   | 50s        | 0.93            | 9.3 GB    |
| N26 N15+cross-attn    | RTX A4500       | 5.6M   | 56s        | 0.86            | 9.8 GB    |
| N27 N15+Kendall MTL   | RTX A4500       | 4.7M   | 49s        | 0.58            | 9.5 GB    |
| N28 N15+PCGrad        | RTX A4500       | 4.7M   | 98s        | 1.56            | 9.1 GB    |
| N29 N15+MTL diag (w0) | RTX A4500       | 4.7M   | 58s        | 1.38            | 11.2 GB   |
| N30 N15+dualflow (w0) | RTX A4500       | 4.7M   | 51s        | 0.45            | 10.9 GB   |
| N31 N15+heads=2 (w0)  | RTX A4500       | 3.0M   | 34s        | 0.71            | 5.3 GB    |
| N32 N15+heads=8 (w0)  | RTX A6000       | 8.2M   | 83s        | 1.23            | 22.1 GB   |
| N33 N15+heads=16 (w0) | RTX A6000       | 15.0M  | 131s       | 2.00            | 26.3 GB   |
| N34 N15+rank=0        | RTX 5070 Ti     | 4.7M   | 33s        | 0.46            | 10.4 GB   |
| N35 N15+rank=0.1      | RTX 5070 Ti     | 4.7M   | 35s        | 0.52            | 11.5 GB   |
| N36 N15+PCGrad enc    | RTX 5070 Ti     | 4.7M   | 77s        | 1.76            | 10.3 GB   |
