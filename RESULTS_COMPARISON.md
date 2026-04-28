# Results Comparison — All Experiments

Generated: 2026-04-28

Metric direction: F1↑ AUC-ROC↑ IFA↓ Top-1↑ Effort@20%↓ Recall@20%loc↑

---

## Binary Classification (2 classes)

> Note: binary not primary goal — included for reference only.

| Folder | Model / Config | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260426_030543_lmgcn_binary | **Arch1 — LM-GCN** frozen CodeBERT | 0.674 | 0.778 | 8.78 | 0.236 | 0.154 | 0.247 |
| 20260428_154150_lmgat_binary | **Arch2 — LM-GAT** frozen GraphCodeBERT | 0.540 | 0.810 | 6.18 | 0.346 | 0.123 | 0.268 |

---

## Multiclass Classification (11 CWE classes) — Primary Goal

### Arch1 — LM-GCN (frozen CodeBERT, GCNConv)

| Folder | Description | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260426_002451_lmgcn_multiclass | LM-GCN multiclass, CodeBERT frozen | 0.209 | 0.742 | 8.65 | 0.272 | 0.162 | 0.232 |

---

### Arch2 — LM-GAT (frozen LM, GATv2Conv)

| Folder | Description | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260426_144901_lmgat_multiclass | LM-GAT v1, CodeBERT frozen | 0.172 | 0.711 | 8.62 | 0.329 | 0.121 | 0.297 |
| 20260426_181253_lmgat_multiclass | LM-GAT v2, CodeBERT frozen, tuned hp | 0.224 | 0.726 | 8.44 | 0.322 | 0.104 | 0.314 |
| 20260427_091241_lmgat_multiclass | LM-GAT v3, **GraphCodeBERT frozen** | 0.135 | 0.696 | 7.63 | 0.315 | 0.127 | 0.283 |

> Frozen GraphCodeBERT (v3) worse than frozen CodeBERT (v2) — frozen GraphCodeBERT embeddings not suited for this task without fine-tuning.

---

### Arch3 — LM-GAT-CodeBERT (live LM fine-tuned, GATv2Conv + CodeBERT joint)

| Folder | Description | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260427_012529_lmgat_codebert_multiclass | Arch3 v1, CodeBERT live, lm_lr=2e-5, batch=4 | 0.204 | 0.696 | 10.73 | 0.235 | 0.185 | 0.217 |
| 20260427_062921_lmgat_codebert_multiclass | Arch3 v2, CodeBERT live, lm_lr=4e-5, batch=16 | 0.193 | 0.686 | 9.20 | 0.315 | 0.142 | 0.294 |
| **20260427_075727_lmgat_codebert_multiclass** | **Arch3 v3, GraphCodeBERT live, lm_lr=1e-5, batch=16** | **0.259** | **0.738** | **6.12** | **0.398** | **0.106** | **0.372** |

> **Best localization model.** GraphCodeBERT + lower lm_lr (1e-5) gives best IFA and Top-1. Higher lm_lr (4e-5) causes instability/overfitting.

---

### Arch4 — LM-GAT-MCS (live LM + multi-scale context, GATv2Conv + pooling)

| Folder | Description | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260427_053340_lmgat_mcs_multiclass | Arch4 v1, CodeBERT live, lm_lr=2e-5, batch=16 | 0.207 | 0.721 | 14.23 | 0.173 | 0.291 | 0.132 |
| **20260427_103516_lmgat_mcs_multiclass** | **Arch4 v2, GraphCodeBERT live, lm_lr=1e-5, batch=32** | **0.272** | **0.761** | 12.76 | 0.225 | 0.149 | 0.268 |

> **Best F1/AUC model.** Arch4 v1 (CodeBERT) worst IFA of all — MCS pooling hurts localization. v2 (GraphCodeBERT) fixes classification but localization still poor vs Arch3.

---

### Arch5 — LM-GIN (frozen CodeBERT, GINEConv — theoretically most expressive)

| Folder | Description | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260427_175127_lmgin_multiclass | LM-GIN, CodeBERT frozen, GINEConv x4 | 0.107 | 0.645 | 8.40 | 0.339 | 0.139 | 0.278 |

> Underperforms all architectures. GINEConv sum aggregation not better than GAT attention for vulnerability localization.

---

### Arch6 — LM-GAT-Interp (live CodeBERT + GATv2, learned λ interpolation)

| Folder | Description | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260427_195648_lmgat_interp_multiclass | LM-GAT-Interp, CodeBERT live, λ=0.5 init | 0.160 | 0.704 | 8.87 | 0.318 | **0.101** | 0.310 |

> Mid-range performance. Best Effort@20% (0.101) — interpolation improves ranking efficiency. F1 lower than Arch3/4 likely because separate LM + GNN heads compete.

---

## Full Multiclass Comparison (sorted by F1)

| Rank | Model | Folder | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|---|
| 1 | **Arch4 v2** GraphCodeBERT live | 20260427_103516_lmgat_mcs | **0.272** | **0.761** | 12.76 | 0.225 | 0.149 | 0.268 |
| 2 | **Arch3 v3** GraphCodeBERT live | 20260427_075727_lmgat_codebert | 0.259 | 0.738 | **6.12** | **0.398** | 0.106 | **0.372** |
| 3 | Arch2 v2 CodeBERT frozen | 20260426_181253_lmgat | 0.224 | 0.726 | 8.44 | 0.322 | **0.104** | 0.314 |
| 4 | Arch3 v1 CodeBERT live | 20260427_012529_lmgat_codebert | 0.204 | 0.696 | 10.73 | 0.235 | 0.185 | 0.217 |
| 5 | Arch4 v1 CodeBERT live | 20260427_053340_lmgat_mcs | 0.207 | 0.721 | 14.23 | 0.173 | 0.291 | 0.132 |
| 6 | Arch1 LM-GCN CodeBERT frozen | 20260426_002451_lmgcn | 0.209 | 0.742 | 8.65 | 0.272 | 0.162 | 0.232 |
| 7 | Arch3 v2 CodeBERT live | 20260427_062921_lmgat_codebert | 0.193 | 0.686 | 9.20 | 0.315 | 0.142 | 0.294 |
| 8 | Arch6 GAT-Interp CodeBERT live | 20260427_195648_lmgat_interp | 0.160 | 0.704 | 8.87 | 0.318 | **0.101** | 0.310 |
| 9 | Arch2 v1 CodeBERT frozen | 20260426_144901_lmgat | 0.172 | 0.711 | 8.62 | 0.329 | 0.121 | 0.297 |
| 10 | Arch2 v3 GraphCodeBERT frozen | 20260427_091241_lmgat | 0.135 | 0.696 | 7.63 | 0.315 | 0.127 | 0.283 |
| 11 | Arch5 LM-GIN CodeBERT frozen | 20260427_175127_lmgin | 0.107 | 0.645 | 8.40 | 0.339 | 0.139 | 0.278 |

---

## Key Takeaways

1. **Best classification (F1/AUC):** Arch4 v2 (GraphCodeBERT live, F1=0.272, AUC=0.761)
2. **Best localization (IFA/Top-1):** Arch3 v3 (GraphCodeBERT live, IFA=6.12, Top-1=0.398)
3. **Best trade-off:** Arch3 v3 — strong on both F1 (0.259) and all localization metrics
4. **GraphCodeBERT > CodeBERT** only when fine-tuned live; frozen GraphCodeBERT hurts
5. **lm_lr sensitivity:** 1e-5 stable; 4e-5 causes instability; 2e-5 borderline
6. **GIN (Arch5) underperforms** — attention-based aggregation superior to sum for this task
7. **Arch4 MCS pooling** boosts F1 at expense of localization (IFA worst: 12.76–14.23)
8. **Arch6 interpolation** best Effort@20% (0.101) but lower F1 — competitive for ranking efficiency

---

## Missing / Pending Results

- Arch3 binary (GraphCodeBERT) — not yet trained
- Arch4 binary (GraphCodeBERT) — not yet trained
- LM-GAT binary (GraphCodeBERT, cloud) — not yet trained
