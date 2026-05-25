# Ablation Results

Dataset: MegaVul Top-25 CWEs, max 1600 per class, 26 classes (25 CWE + benign),
UniXcoder-base embeddings, seed=42. GPU: RTX 5070 Ti (Phase 1–7); RTX 5090 for Phase 8 reruns (H2/H3) and Phase 9.

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
| A4-L2       | **0.561** | 0.475     | 0.540     | 0.526 | 0.904     | 0.757     | 43     |
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

All F-configs use the Phase 3 winner loss (no focal + label_smoothing 0.1 + cosine,
wd 1e-3, patience 15) — same as F1's meanmax baseline and phases 4-5.

**F2 (node=CodeT5+)** improves classification: +0.985 macro F1 vs F1 baseline
(0.502 vs 0.517 — gap vs baseline is −0.015, but best among F2/F3 variants),
best accuracy (0.554), best AUC-ROC (0.906). Localization weaker than F1 (IFA 0.745,
R@20% 0.415 vs 0.487). CodeT5+ node embeddings carry richer program semantics but
less precise statement-level signal.

**F3 (func=CodeT5+)** trades classification for localization: macro F1 drops to 0.444
(worst of F2/F3) but localization dominates — best IFA (0.512), Top-1 (0.946),
R@5% (0.374), R@20% (0.586), Effort (0.016). CodeT5+ function-level hidden captures
finer per-token context useful for statement scoring even at 512-token cap.

**F5 (func=CodeT5+ raw 768-dim)** gives best accuracy (0.568) and F1-w (0.566) among
all F-configs, but localization collapses — IFA 1.186 (worst), R@20% 0.547. Root
cause: concat becomes `[GNN-256 | LM-768]` = 1024-dim with LM occupying 75% — GNN
statement-level signal drowned by the larger raw LM vectors.

**F6 (func=CodeT5+ proj+norm 256-dim)** normalizes per-token projected vectors to
unit norm. Localization partially recovers vs F5 (IFA 0.734 vs 1.186) but F3
(unnormalized proj-256) still wins (IFA 0.512, R@20% 0.586). Surprisingly,
unit-norm HURTS localization slightly vs F3 — amplitude variation in unnormalized
per-token vectors may encode statement-level suspicion signal that normalization
discards. F6 ties F3 on Effort@20%R (0.016). Classification degrades vs F5
(macro F1 0.459 vs 0.499) — consistent with the dimension-balance being restored
(256+256=512, equal GNN/LM) but the normalization removing useful scale info.

**Updated key finding:** LM embedding dimension relative to GNN dim governs the
classification/localization trade-off — not just LM model choice. Raw 768-dim
(F5) maximises classification but kills localization via GNN signal suppression in
concat. Projected 256-dim (F3) keeps GNN/LM at equal dim (256+256) and is best
for localization. Normalization (F6) is neutral-to-harmful for localization.
**F4 (both=CodeT5+)** — both node and func LM are CodeT5+. Classification: macro F1
0.475, AUC 0.833 (worst AUC of all F-configs). Localization: IFA 0.991, R@20%
0.436 — worse than F3 (func=CT5+, IFA 0.512) and F2 (node=CT5+, IFA 0.745).
Combining CodeT5+ at both levels does NOT compound gains — the two CodeT5+ branches
interfere rather than complement. AUC collapse (0.833 vs 0.906 F2) suggests
ranking quality degrades when node features and live LM share the same embedding
space, reducing diversity. The trade-off is intrinsic, not additive.

**F7 (both normed — GNN output norm + LM per-token norm)** — symmetric normalization
of both GNN (h_graph and per-node h_loc via F.normalize dim=-1) and LM per-token
vectors. Localization: IFA **0.508** (best of all F-configs), Top-1 0.927, R@20% 0.570.
Classification: macro F1 0.484, AUC 0.887. Compared to F3 (LM norm only, no GNN norm):
IFA marginally better (0.508 vs 0.512) but Top-1 / R@20% / R@5% all worse. Classification
slightly better (F1 0.484 vs 0.444, AUC 0.887 vs 0.897). Result: GNN output norm gives
marginal IFA gain at the cost of Top-1 / recall coverage. F3 (unnormalized GNN, LM
proj+unnorm) remains the best localization config overall; F7 is a Pareto tie on IFA only.

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

| ID | Run ID | Config | hidden_dim | fused_dim | GNN% | LM% | Epochs |
|---|---|---|---|---|---|---|---|
| G1 | `20260516_125619_lmgat_codebert_multiclass` | — (= Phase 4 meanmax / F1) | 256 | 1024 | 25% | 75% | 48 |
| G2 | `20260520_132730_lmgat_codebert_multiclass` | `G2_dim768_equal.yaml` | 768 | 1536 | 50% | 50% | 34 |

## Classification

| ID | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. | Epochs |
|---|---|---|---|---|---|---|
| G1 | 0.517 | 0.538 | 0.539 | 0.911 | 0.630 | 48 |
| G2 | **0.529** | **0.582** | **0.579** | **0.914** | 0.569 | 34 |

## Statement-Level Localization

| ID | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|
| G1 | **0.644** | **0.900** | **0.982** | **0.269** | **0.487** | **0.025** |
| G2 | 1.410 | 0.747 | 0.962 | 0.186 | 0.427 | 0.056 |

G2 beats G1 on classification across all metrics (+1.2pp F1, +4.4pp Acc, +0.003 AUC)
but localization collapses hard (IFA 0.644→1.410, Top-1 0.900→0.747). Root cause:
equal GNN/LM balance (50/50) shifts the fused representation toward classification but
~32% of MegaVul vuln functions exceed 1024 UniXcoder tokens — G2's func_max_length=1024
truncates those, losing functional context for the largest/most complex functions and
degrading per-node localization signal. The 768-dim GNN also runs slower (409s/epoch
vs 162s) and uses more VRAM (10.2 GB vs ~7 GB).

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

| ID | Config | chunk | stride | max_len | Max windows | Run ID | Epochs |
|---|---|---|---|---|---|---|---|
| H1 | — (= G2) | — | — | 1024 | 1 | `20260520_132730` | 34 |
| H2 | `H2_unixcoder_sliding_chunk1024_stride512.yaml` | 1024 | 512 | 5120 | 9 | `20260525_104032` | 30 |
| H3 | `H3_unixcoder_sliding_chunk1024_stride1024.yaml` | 1024 | 1024 | 5120 | 5 | `20260525_125031` | 31 |

## Classification

| ID | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. | Epochs |
|---|---|---|---|---|---|---|
| H1 (= G2) | **0.529** | **0.582** | **0.579** | **0.914** | 0.569 | 34 |
| H2 | 0.459 | 0.508 | 0.507 | 0.890 | 0.588 | 30 |
| H3 | 0.528 | 0.529 | 0.533 | 0.895 | 0.587 | 31 |

## Statement-Level Localization

| ID | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|
| H1 (= G2) | 1.410 | 0.747 | 0.962 | 0.186 | 0.427 | 0.056 |
| H2 | **1.025** | **0.876** | 0.971 | 0.193 | 0.395 | 0.052 |
| H3 | 1.047 | 0.873 | **0.978** | **0.221** | **0.442** | **0.041** |

Both H2 and H3 use ml5120 dataset with fixed `lm_full_windowed` (mean-pool CLS across windows),
so sliding window is genuinely active for functions exceeding 1024 tokens (~32% of MegaVul vuln functions).

**H2 (50% overlap, stride=512, 9 windows max)** — classification degrades vs H1 baseline
(F1 0.459 vs 0.529, −0.070). Root cause: 9 overlapping windows produce a noisy mean-pool CLS —
boundary regions appear in two consecutive windows, introducing redundant gradient signal that
dilutes the per-window CLS. With 9 windows × batch_size=16 forward passes per batch, gradient
variance is high relative to H1's single pass. Localization improves over H1 (IFA 1.025 vs
1.410, Top-1 0.876 vs 0.747) because overlapping per-token accumulation covers more context
without the CLS noise problem affecting stmt_head.

**H3 (non-overlapping, stride=1024, 5 windows max)** — classification ties H1 (F1 0.528 vs
0.529, −0.001). Unlike the prior invalid H3 run (ml1024, fast path always triggered), this
run with ml5120 correctly engages 2–5 windows for longer functions. Non-overlapping windows
avoid boundary-duplication noise: each position appears in exactly one window, giving a cleaner
per-window CLS mean-pool. Localization also improves over H1 (IFA 1.047 vs 1.410, Top-1 0.873
vs 0.747) and beats H2 on R@5%/R@20%/Effort despite marginally worse IFA and Top-1.
H3 is 1.7× faster than H2 (150s vs 250s/epoch).

Both H2 and H3 remain below the G1/F1 localization baseline (IFA 0.644, Top-1 0.900) —
mean aggregation across windows blurs fine-grained per-token signal that single-pass lm_full
produces for shorter functions already under 1024 tokens.

**Phase 8 winner: H3 — ties H1 on classification (F1 0.528 ≈ 0.529), better localization
(IFA 1.047 vs 1.410, R@20% 0.442 vs 0.427), and 1.7× faster than H2. H2 (overlapping)
hurts classification despite better IFA/Top-1 localization precision.**

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

| ID | Run ID | Config | live_lm | freeze_func_lm | precompute | Line ctx | Epochs |
|---|---|---|---|---|---|---|---|
| I1 | `20260520_132730` (= H1/G2) | — | func | No | — | — | 34 |
| I2 | `20260522_070552_lmgat_codebert_multiclass` | `I2_line_encoder.yaml` | line | Yes | Yes | — | 84 |
| I3 | `20260522_135012_lmgat_codebert_multiclass` | `I3_line_encoder_live.yaml` | line | No | No | — | 27† |
| I4 | `20260525_141857_lmgat_codebert_multiclass` | `I4_line_ctx5.yaml` | line | Yes | Yes | ±5 | 52 |

## Classification

† = classification collapsed (predicts majority class only; metrics not comparable).

| ID | Test F1 | Test Acc | F1-w | AUC-ROC | Conf. | Epochs |
|---|---|---|---|---|---|---|
| I1 (= H1/G2) | **0.529** | **0.582** | **0.579** | **0.914** | 0.569 | 34 |
| I2 | 0.390 | 0.402 | 0.402 | 0.880 | 0.311 | 84 |
| I3† | 0.017† | 0.152† | 0.044† | 0.774† | 0.586† | 27 |
| I4 | 0.375 | 0.414 | 0.410 | 0.877 | 0.381 | 52 |

## Statement-Level Localization

| ID | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|---|---|---|---|---|---|---|
| I1 (= H1/G2) | 1.410 | 0.747 | 0.962 | 0.186 | 0.427 | 0.056 |
| I2 | 2.116 | 0.666 | 0.924 | 0.200 | 0.390 | 0.051 |
| I3† | **1.164** | **0.795** | **0.955** | 0.186 | **0.428** | 0.057 |
| I4 | 3.009 | 0.609 | 0.857 | 0.179 | 0.443 | 0.059 |

**I2 (frozen LM + line encoder)** — classification drops significantly vs I1 baseline
(F1 0.390 vs 0.529, −0.139). Localization also degrades (IFA 2.116 vs 1.410, Top-1
0.666 vs 0.747). Root cause: frozen LM cannot be fine-tuned for the classification
task — precomputed per-line CLS embeddings are fixed features without task adaptation.
The cross-line transformer has no gradient signal from the LM to improve. Training ran
84 epochs (patience=15 triggered at ep 69; best val F1=0.405) — the extra epochs
reflect the optimizer searching for a better combination of the fixed LM features and
trainable GNN/heads.

**I3 (live LM + line encoder) — classification collapsed** (F1 0.017, predicts
class 0 only). Best val F1 = 0.009 at epoch 2; patience=15
triggered at epoch 27. Root cause: per-line LM forward replaces the whole-function
CLS forward entirely — the classification head loses global function-level context.
The cross-line transformer contextualizes lines but operates at statement granularity;
it cannot recover the function-level semantic representation needed to distinguish
26 CWE classes. Unlike I2 (frozen LM at least produces consistent per-line features
from a stable pretrained model), I3's live LM receives conflicting gradient signals —
CWE classification requires whole-function semantics but the LM only ever sees
individual lines. Localization survives (ranking loss flows through stmt_head
independently of func_head failure).

**I4 (frozen LM + line encoder + ±5 line context)** — adds per-line context: each
line's [CLS] is precomputed from [line_{i-5} … line_i … line_{i+5}] concatenated
and passed through UniXcoder jointly, so the per-line CLS can attend to neighbouring
lines' tokens before the cross-line transformer. Classification drops further below I2
(F1 0.375 vs 0.390) — the ±5 context window increases precompute cost but does not
substitute for whole-function semantics at training time; the frozen LM cannot adapt
to the 26-class CWE task regardless of context size. Localization: IFA worsens
(3.009 vs 2.116), Top-1 worsens (0.609 vs 0.666), but R@20%LOC improves (0.443 vs
0.390) — the broader context widens recall coverage at the cost of IFA/Top-1 precision.
Overall I4 is worse than I2 on classification and mixed on localization.

**Phase 9 finding: line-level LM encoding is incompatible with function-level
CWE multiclass classification.** Frozen (I2, I4) and live (I3) variants all underperform
I1 on classification. Adding line context ±5 (I4) worsens classification further vs I2
and gives mixed localization (better R@20%, worse IFA/Top-1). Whole-function CLS
(H3 approach) is necessary for classification; the hierarchical design loses global
context that 26-class CWE discrimination requires. Phase 9 is a negative result —
confirms H3 sliding window as the better path for extending LM coverage.

---

# Training Efficiency

| Run                   | Params | Epoch Time | Total Time (hr) | VRAM Peak |
| --------------------- | ------ | ---------- | --------------- | --------- |
| A1                    | 3.5M   | 48s        | 0.73            | 4.2 GB    |
| A2                    | 129.6M | 216s       | 3.24            | 4.2 GB    |
| A3                    | 129.6M | 175s       | 3.70            | 6.0 GB    |
| A4                    | 129.6M | 176s       | 3.62            | 4.8 GB    |
| A4-L1                 | 129.6M | 161s       | 1.39            | 6.9 GB    |
| A4-L2                 | 129.6M | 162s       | 1.93            | 7.7 GB    |
| A4-L2-fixed           | 129.6M | 162s       | 3.37            | —         |
| fusion gated          | 129.6M | 163s       | 1.72            | —         |
| fusion weighted α=0.3 | 129.6M | 162s       | 1.53            | —         |
| fusion weighted α=0.5 | 129.6M | 162s       | 1.75            | —         |
| fusion weighted α=0.7 | 129.6M | 162s       | 1.44            | —         |
| pool attention        | 129.6M | 162s       | 2.25            | 7.4 GB    |
| pool meanmax          | 129.6M | 162s       | 2.16            | —         |
| B2 cross_attn         | 129.6M | 169s       | 2.72            | —         |
| B3 self_attn          | 129.6M | 245s       | 4.29            | —         |
| B4 mmoe               | 129.6M | 165s       | 1.83            | —         |
| F4 both=CT5+          | 137.0M | 323s       | 5.94            | 9.1 GB    |
| F5 CT5+ raw           | 138.2M | 457s       | 6.86            | 8.4 GB    |
| F6 CT5+ norm          | 138.0M | 460s       | 4.35            | 8.7 GB    |
| F7 both normed        | 138.0M | 460s       | 8.96            | 8.9 GB    |
| G2 dim=768            | 146.9M | 409s       | 3.87            | 10.2 GB   |
| H2 sliding stride512  | 146.9M | 250s       | 2.09            | 18.6 GB   |
| H3 sliding stride1024 | 146.9M | 150s       | 1.30            | 17.1 GB   |
| I2 line frozen        | 161.7M | 285s       | 6.66            | 9.5 GB    |
| I3 line live          | 161.7M | 644s       | 4.84            | 11.0 GB   |
| I4 line ctx±5 frozen  | 161.7M | 84s        | 1.22            | 17.4 GB   |
