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

## Training Efficiency

GPU: RTX 5070 Ti

| Run | Params | Epoch Time | Total Time (hr) | VRAM Peak |
|-----|--------|-----------|------------|-----------|
| A1 | 3.5M | 48s | 0.73 | 4.2 GB |
| A2 | 129.6M | 216s | 3.24 | 4.2 GB |
| A3 | 129.6M | 175s | 3.70 | 6.0 GB |
| A4 | 129.6M | 176s | 3.62 | 4.8 GB |
