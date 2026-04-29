# Results Comparison — All Experiments

Generated: 2026-04-29 (updated 2026-04-29 with cloud UniXcoder runs)

Metric direction: F1↑ AUC-ROC↑ IFA↓ Top-1↑ Effort@20%↓ Recall@20%loc↑

---

## Binary Classification (2 classes)

> Note: binary not primary goal — included for reference only.

| Folder | Model / Config | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260426_030543_lmgcn_binary | **Arch1 — LM-GCN** frozen CodeBERT | 0.674 | 0.778 | 8.78 | 0.236 | 0.154 | 0.247 |
| 20260428_154150_lmgat_binary | **Arch2 — LM-GAT** frozen GraphCodeBERT | 0.540 | 0.810 | 6.18 | 0.346 | 0.123 | 0.268 |
| 20260428_141150_lmgin_binary | **Arch5 — LM-GIN** frozen GraphCodeBERT | 0.546 | 0.793 | 6.41 | 0.399 | 0.114 | 0.293 |
| 20260428_152917_lmgat_interp_binary | **Arch6 — LM-GAT-Interp** live GraphCodeBERT | **0.650** | **0.812** | **6.13** | 0.365 | 0.117 | **0.304** |

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
| **20260429_091918_lmgat_codebert_multiclass** | **Arch3 v4, UniXcoder live, lm_lr=1e-5, batch=16 (cloud RTX 4090)** | **0.4115** | **0.8562** | 7.72 | 0.366 | 0.103 | 0.340 |

> **Best localization model.** GraphCodeBERT + lower lm_lr (1e-5) gives best IFA and Top-1. Higher lm_lr (4e-5) causes instability/overfitting.
> **UniXcoder (v4) dominates F1/AUC** — F1=0.4115 vs 0.259 (+59%); AUC=0.856 vs 0.738. Localization slightly weaker (IFA=7.72 vs 6.12) — UniXcoder prioritises classification over localisation.

---

### Arch4 — LM-GAT-MCS (live LM + multi-scale context, GATv2Conv + pooling)

| Folder | Description | F1↑ | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
|---|---|---|---|---|---|---|---|
| 20260427_053340_lmgat_mcs_multiclass | Arch4 v1, CodeBERT live, lm_lr=2e-5, batch=16 | 0.207 | 0.721 | 14.23 | 0.173 | 0.291 | 0.132 |
| **20260427_103516_lmgat_mcs_multiclass** | **Arch4 v2, GraphCodeBERT live, lm_lr=1e-5, batch=32** | **0.272** | **0.761** | 12.76 | 0.225 | 0.149 | 0.268 |
| **20260429_095918_lmgat_mcs_multiclass** | **Arch4 v3, UniXcoder live, lm_lr=1e-5, batch=16 (cloud RTX 4090)** | **0.5791** | **0.8977** | 12.74 | 0.221 | 0.110 | 0.308 |

> **Best F1/AUC model.** Arch4 v1 (CodeBERT) worst IFA of all — MCS pooling hurts localization. v2 (GraphCodeBERT) fixes classification but localization still poor vs Arch3.
> **UniXcoder (v3) new overall best F1=0.5791, AUC=0.8977** — more than doubles GraphCodeBERT F1. IFA pattern unchanged (MCS pooling still hurts localisation).

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
| 1 | **Arch4 v3** UniXcoder live | 20260429_095918_lmgat_mcs | **0.5791** | **0.8977** | 12.74 | 0.221 | 0.110 | 0.308 |
| 2 | **Arch3 v4** UniXcoder live | 20260429_091918_lmgat_codebert | **0.4115** | **0.8562** | 7.72 | 0.366 | 0.103 | 0.340 |
| 3 | Arch4 v2 GraphCodeBERT live | 20260427_103516_lmgat_mcs | 0.272 | 0.761 | 12.76 | 0.225 | 0.149 | 0.268 |
| 4 | Arch3 v3 GraphCodeBERT live | 20260427_075727_lmgat_codebert | 0.259 | 0.738 | **6.12** | **0.398** | 0.106 | **0.372** |
| 5 | Arch2 v2 CodeBERT frozen | 20260426_181253_lmgat | 0.224 | 0.726 | 8.44 | 0.322 | **0.101** | 0.314 |
| 6 | Arch1 LM-GCN CodeBERT frozen | 20260426_002451_lmgcn | 0.209 | 0.742 | 8.65 | 0.272 | 0.162 | 0.232 |
| 7 | Arch4 v1 CodeBERT live | 20260427_053340_lmgat_mcs | 0.207 | 0.721 | 14.23 | 0.173 | 0.291 | 0.132 |
| 8 | Arch3 v1 CodeBERT live | 20260427_012529_lmgat_codebert | 0.204 | 0.696 | 10.73 | 0.235 | 0.185 | 0.217 |
| 9 | Arch3 v2 CodeBERT live | 20260427_062921_lmgat_codebert | 0.193 | 0.686 | 9.20 | 0.315 | 0.142 | 0.294 |
| 10 | Arch2 v1 CodeBERT frozen | 20260426_144901_lmgat | 0.172 | 0.711 | 8.62 | 0.329 | 0.121 | 0.297 |
| 11 | Arch6 GAT-Interp CodeBERT live | 20260427_195648_lmgat_interp | 0.160 | 0.704 | 8.87 | 0.318 | **0.101** | 0.310 |
| 12 | Arch2 v3 GraphCodeBERT frozen | 20260427_091241_lmgat | 0.135 | 0.696 | 7.63 | 0.315 | 0.127 | 0.283 |
| 13 | Arch5 LM-GIN CodeBERT frozen | 20260427_175127_lmgin | 0.107 | 0.645 | 8.40 | 0.339 | 0.139 | 0.278 |

---

## Key Takeaways

1. **Best classification (F1/AUC):** Arch4 v3 (UniXcoder live, F1=0.5791, AUC=0.8977) — +113% over previous best
2. **Best localization (IFA/Top-1):** Arch3 v3 (GraphCodeBERT live, IFA=6.12, Top-1=0.398)
3. **Best trade-off:** Arch3 v4 (UniXcoder live) — F1=0.4115, AUC=0.856, IFA=7.72 — strong on both
4. **UniXcoder >> GraphCodeBERT >> CodeBERT** for classification when fine-tuned live
5. **Frozen LM:** CodeBERT > GraphCodeBERT frozen (GraphCodeBERT embeddings not suited without fine-tuning)
6. **lm_lr sensitivity:** 1e-5 stable; 4e-5 causes instability; 2e-5 borderline
7. **GIN (Arch5) underperforms** — attention-based aggregation superior to sum for this task
8. **Arch4 MCS pooling** boosts F1 at expense of localization (IFA consistently worst: 12.74–14.23)
9. **Arch6 interpolation** best Effort@20% (0.101) but lower F1 — competitive for ranking efficiency
10. **UniXcoder does not fix Arch4 localisation weakness** — IFA 12.74 same pattern as GraphCodeBERT; MCS pooling is the bottleneck

---

## Missing / Pending Results

- Arch3/4 binary GraphCodeBERT variants — not yet trained
- Arch7 (lmgat_seq) UniXcoder — training in progress on cloud
- Arch8 (lmgat_waves_seq) UniXcoder — training in progress on cloud
- VulLMGNN / lmggnn UniXcoder — training locally

---

## Public Baseline Comparison — Status

**Direct comparison to published numbers is not valid at this stage.** Reasons:

### Classification (F1, AUC-ROC)

| Baseline | Reported F1 | Our Task | Why Not Comparable |
|---|---|---|---|
| LineVul (Fu et al. 2022) | ~0.91 | Binary on full BigVul | We use 11-class on 2000/class sample — F1 is naturally lower for multiclass |
| VulLMGNN (Cao et al. 2022) | ~0.70 | Binary on BigVul/Devign | Binary vs multiclass; different data split |

> **Note:** Multiclass F1=0.272 (11 classes) vs binary F1=0.91 is not a regression — they measure different tasks. Shown here for reference only.

### Localization (IFA, Top-1, Effort@20%)

| Baseline | IFA↓ | Top-1↑ | Effort@20%↓ | Why Not Comparable |
|---|---|---|---|---|
| LineVul (Fu et al. 2022) | ~4.0 | ~0.50 | ~0.08 | Full BigVul dataset; different functions in test set |
| WAVES | N/A | reported | reported | No source code available — cannot reproduce or verify |

> **Key issue:** Even though both LineVul and our models use BigVul, the exact sampled functions and train/test splits differ. Localization metrics are sensitive to which functions appear in the test set.

### Plan for Fair Comparison

To produce a valid comparison table:
1. **Run LineVul** on our exact dataset split (same 2000/class sample, same 70/15/15 split, same random seed)
2. **Run VulLMGNN** on our exact dataset split
3. Present all models — ours + baselines — evaluated on the same test set
4. **WAVES**: skip — no public source code, replication not feasible

This will make the comparison fair and publishable.
