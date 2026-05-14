# Phase 1 Ablation — Deep Analysis

## Experiment Design

All runs use the same dataset: MegaVul Top-25 CWEs, max 1600 per class, 26 classes (25 CWE + benign), UniXcoder-base embeddings, seed=42.

| Run | Config | Architecture | Localization Source | Batch (effective) | LM Fine-tuned? |
|-----|--------|---|---|---|---|
| A1 | `A1_lmgat.yaml` | lmgat (frozen) | GNN nodes | 32 (16×2) | No |
| A2 | `A2_lmgat_codebert_gnn.yaml` | lmgat_codebert | GNN nodes | 32 (8×4) | Yes (lm_lr=1e-5) |
| A3 | `A3_lmgat_codebert_lm.yaml` | lmgat_codebert | LM hidden states | 32 (16×2) | Yes (lm_lr=1e-5) |
| A4 | `A4_lmgat_codebert_both.yaml` | lmgat_codebert | GNN + LM concat | 32 (16×2) | Yes (lm_lr=1e-5) |

All share: focal_loss_gamma=2.0, livable_loss=true, use_class_weights=true, patience=25, early_stop_metric=f1.

---

## Results Summary

### Function-Level Classification

| Run | Val F1 (best) | Test F1 | Test Acc | AUC-ROC | Confidence (mean) | Epochs |
|-----|---|---|---|---|---|---|
| A1 (frozen) | 0.458 | 0.471 | 0.510 | 0.884 | 0.765 | 55 |
| A2 (GNN loc) | 0.532 | 0.494 | 0.500 | 0.907 | 0.698 | 54 |
| A3 (LM loc) | 0.548 | 0.495 | 0.517 | 0.913 | 0.801 | 76 |
| A4 (both loc) | **0.550** | **0.504** | 0.507 | 0.899 | 0.813 | 74 |

### Statement-Level Localization

| Run | IFA ↓ | Top-1 ↑ | Top-3 ↑ | Top-5 ↑ | Top-10 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
|-----|---|---|---|---|---|---|---|---|
| A1 (frozen) | 1.49 | 0.804 | 0.914 | 0.941 | 0.966 | 0.195 | 0.394 | 0.052 |
| A2 (GNN loc) | **0.89** | **0.874** | 0.936 | 0.959 | 0.977 | 0.217 | 0.401 | **0.039** |
| A3 (LM loc) | 1.33 | 0.818 | **0.939** | **0.969** | **0.988** | 0.197 | **0.451** | 0.052 |
| A4 (both loc) | 1.26 | 0.794 | 0.917 | 0.959 | 0.978 | **0.207** | 0.431 | 0.047 |

---

## Deep Analysis

### 1. The Val-Test F1 Gap (Overfitting Signal)

| Run | Val F1 | Test F1 | Gap | Gap % |
|-----|--------|---------|-----|-------|
| A1 | 0.458 | 0.471 | -0.013 (test > val!) | — |
| A2 | 0.532 | 0.494 | 0.038 | 7.1% |
| A3 | 0.548 | 0.495 | 0.053 | 9.7% |
| A4 | 0.550 | 0.504 | 0.046 | 8.4% |

**Key finding:** A1 (frozen LM) has NO gap — test actually slightly exceeds val. All live-LM runs (A2-A4) show 7-10% gap. This confirms the live LM is overfitting to the training set.

**Why:** The live LM has 129M parameters being fine-tuned on ~1600×26 = ~41K samples. That's a very low sample-to-parameter ratio. The LM memorizes training patterns that don't generalize.

**Implication:** Weight decay (currently 1e-4) is too low for the LM branch. Standard LM fine-tuning uses 1e-2 to 1e-3. This is Tier 4.2 from `LOSS_F1_GAP.md`.

### 2. Training Dynamics — Loss-F1 Divergence

From A4's training log (best run):

| Epoch | Train Loss | Val Loss | Val F1 | Observation |
|-------|-----------|----------|--------|-------------|
| 1 | 2.81 | 2.22 | 0.063 | Cold start |
| 10 | 0.74 | 1.51 | 0.463 | Healthy learning |
| 20 | 0.39 | 1.74 | 0.478 | Val loss rising, F1 still improving |
| 33 | 0.23 | 2.09 | **0.528** | F1 peak region |
| 49 | 0.15 | 2.43 | **0.550** | Best F1 — val loss 10× train loss! |
| 74 | 0.09 | 3.32 | 0.462 | Patience exhausted, F1 collapsed |

**The pattern is clear:**
- Train loss drops monotonically: 2.81 → 0.09 (97% reduction)
- Val loss: drops to 1.37 (epoch 7), then RISES continuously to 3.32
- Val F1: rises to 0.55 (epoch 49), then drops
- Best F1 occurs when val_loss is already 2.43 — far from its minimum of 1.37

**This is exactly the loss-F1 gap from `LOSS_F1_GAP.md` §1.** The model becomes overconfident on training data (train_loss → 0.09) while val_loss explodes. But F1 keeps improving because the argmax decisions are still getting better even as probabilities become miscalibrated.

### 3. A2 vs A3 vs A4 — Localization Encoder Comparison

The key ablation question: where should localization features come from?

| Source | IFA | Top-1 | R@20%LOC | Classification F1 |
|--------|-----|-------|----------|-------------------|
| GNN only (A2) | **0.89** | **0.874** | 0.401 | 0.494 |
| LM only (A3) | 1.33 | 0.818 | **0.451** | 0.495 |
| Both (A4) | 1.26 | 0.794 | 0.431 | **0.504** |

**Interpretation:**
- **GNN is better for precise localization** (IFA, Top-1) — it knows exactly which node/line is suspicious
- **LM is better for coverage** (R@20%LOC) — it captures broader vulnerability patterns across more lines
- **Both gives best classification** but localization is a compromise between the two

**Why GNN wins on IFA/Top-1:** GNN operates on CPG nodes which map directly to source lines. Each node has structural context (data flow, control flow). The GNN can pinpoint "this specific node is the vulnerability" because it sees the graph structure.

**Why LM wins on R@20%LOC:** The LM sees the full function text and can identify vulnerability-related patterns (unsafe API calls, missing checks) even in lines that aren't directly connected in the CPG. It catches more flaw lines but is less precise about which one is THE flaw.

### 4. A2's Lower Confidence — A Feature, Not a Bug

A2 has the lowest confidence (0.698 vs 0.76-0.85 for others). But it also has the best localization. Why?

A2 uses batch_size=8 (vs 16 for others) with grad_accum=4. This means:
- Each forward pass sees only 8 samples
- More gradient noise per step
- Acts as implicit regularization
- Model is less confident but more calibrated

The lower confidence + better localization suggests A2 is less overfit. Its val_loss at best epoch (1.70) is lower than A4's (2.43), confirming better calibration.

### 5. Comparison with Literature

| Metric | A2 (best loc) | A4 (best F1) | LineVul (MSR 2022) | EDAT (2025) |
|--------|---|---|---|---|
| IFA ↓ | **0.89** | 1.26 | 4.56 | 2.79 |
| Top-5 Acc ↑ | 0.959 | 0.959 | 0.65 (Top-10) | 0.60 (Top-5) |
| R@20%LOC ↑ | 0.401 | 0.431 | — | 0.65 |
| F1 (multiclass) | 0.494 | 0.504 | — (binary) | 0.73 |

**Localization is already very strong.** IFA=0.89 means on average less than 1 clean line before finding the first flaw. This beats LineVul (4.56) and EDAT (2.79) significantly.

**Classification needs improvement.** F1=0.50 vs EDAT's 0.73. But note: EDAT uses CodeT5 (larger model), adversarial augmentation, and cross-task attention. Your model is simpler and still has room for improvement via loss stack fixes and MTL.

### 6. The Stacked Loss Problem — Confirmed

All runs use focal_loss_gamma=2.0 + livable_loss=true + use_class_weights=true. From `LOSS_F1_GAP.md` §2.5.2, this is LIVABLE's Table 2 worst row (class-balanced focal = 54.77% accuracy).

The val_loss explosion (1.37 → 3.32 in A4) is consistent with the triple-stacking diagnosis:
- Focal amplifies uncertain samples
- LIVABLE ramps weights on rare classes over time
- Class weights add another multiplier
- Combined: rare class predictions cause massive loss spikes on validation

---

## Recommendations Based on Results

### Immediate (Phase 2)

1. **Fix loss stack** — drop focal, keep livable + class_weights + add label_smoothing=0.1
2. **Increase weight_decay to 1e-3** for LM branch — the 7-10% val-test gap is too large
3. **Use A2's batch config** (batch=8, accum=4) for all runs — implicit regularization helps

### For localization encoder choice

- If priority is **precise localization** (IFA, Top-1): use `localization_encoder: gnn` (A2)
- If priority is **coverage** (R@20%LOC): use `localization_encoder: lm` (A3)
- If priority is **classification F1**: use `localization_encoder: both` (A4)

For thesis: report A4 (both) as the main model since it gives best classification, and note that GNN-only localization gives best IFA.

### For next architecture (MTL / HC-DFGAT)

- Start from A4's config (localization_encoder: both)
- Add MTL heads (binary + group) — expected +10 F1 points based on EDAT's MTL-only numbers
- Fix loss stack first before adding SupCon
- Consider SupCon only after MTL is validated

---

## Training Efficiency

| Run | Params | Epoch Time | Total Time | VRAM Peak |
|-----|--------|-----------|------------|-----------|
| A1 (frozen) | 3.5M | 48s | 44 min | 4.2 GB |
| A2 (live, bs=8) | 129.6M | 216s | 3.2 hr | 4.2 GB |
| A3 (live, bs=16) | 129.6M | 175s | 3.7 hr | 6.0 GB |
| A4 (live, bs=16) | 129.6M | 176s | 3.6 hr | 4.8 GB |

A2 is slower per epoch (216s vs 175s) because batch_size=8 means more forward passes per epoch. But it trains fewer epochs (54 vs 74-76) due to earlier convergence, so total time is similar.

A1 is 4.5× faster than live-LM runs — useful for quick ablation experiments on loss configs.
