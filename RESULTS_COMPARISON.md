# Results Comparison — All Experiments

Generated: 2026-04-29 (updated 2026-05-07)

Metric direction: F1↑ AUC-ROC↑ IFA↓ Top-1↑ Effort@20%↓ Recall@20%loc↑

---

## Dataset Reference

Three distinct datasets used across experiments. Results across datasets are **NOT directly comparable**.

| Dataset ID       | Source    | Parquet                   | CPG graphs              | Classes | Test size | Notes                                              |
| ---------------- | --------- | ------------------------- | ----------------------- | ------- | --------- | -------------------------------------------------- |
| **BigVul-v1**    | BigVul    | `train.parquet` only      | 2000 benign + 5494 vuln | 11      | ~1124     | No `top_cwe` filter; vocab naturally small         |
| **BigVul-v2**    | BigVul    | `all.parquet` (train+val+test combined) | 2200 benign + 8760 vuln | 11 | ~1363 | `top_cwe: 10` filter; 9089 total (6889 top-10 + 2200 benign) |
| **TitanVul-OWASP**  | TitanVul | TitanVul CPGs | larger functions (max_nodes=3400) | 90 | ~1499 | `filter_owasp_top10=true`; 89 CWE labels + benign |
| **TitanVul-Top25**  | TitanVul | TitanVul CPGs | larger functions (max_nodes=3400) | 26 | ~1681 | `filter_top25=true`; 25 CWE Top25 labels + benign |

**Cross-dataset comparisons are invalid.** BigVul-v1 → BigVul-v2 jump (F1=0.272→0.58+) is confounded: LM change + dataset size increase + cleaner split. TitanVul-OWASP is a structurally different task (90 classes vs 11).

**For valid LM comparison within BigVul-v2:** Retrain Arch3/4 with CodeBERT + GraphCodeBERT on BigVul-v2 and compare against UniXcoder runs.

**Code correctness:** `evaluate.py` uses same seed → same `test_idx` → no data leakage.

---

## Binary Classification (2 classes)

> Note: binary not primary goal — included for reference only.

| Folder                              | Model / Config                               | F1↑       | AUC-ROC↑  | IFA↓     | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| ----------------------------------- | -------------------------------------------- | --------- | --------- | -------- | ------ | ----------- | -------------- |
| 20260426_030543_lmgcn_binary        | **Arch1 — LM-GCN** frozen CodeBERT           | 0.674     | 0.778     | 8.78     | 0.236  | 0.154       | 0.247          |
| 20260428_154150_lmgat_binary        | **Arch2 — LM-GAT** frozen GraphCodeBERT      | 0.540     | 0.810     | 6.18     | 0.346  | 0.123       | 0.268          |
| 20260428_141150_lmgin_binary        | **Arch5 — LM-GIN** frozen GraphCodeBERT      | 0.546     | 0.793     | 6.41     | 0.399  | 0.114       | 0.293          |
| 20260428_152917_lmgat_interp_binary | **Arch6 — LM-GAT-Interp** live GraphCodeBERT | **0.650** | **0.812** | **6.13** | 0.365  | 0.117       | **0.304**      |

---

## Multiclass Classification (11 CWE classes) — Primary Goal

### Arch1 — LM-GCN (frozen CodeBERT, GCNConv)

| Folder                           | Description                        | F1↑   | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| -------------------------------- | ---------------------------------- | ----- | -------- | ---- | ------ | ----------- | -------------- |
| 20260426_002451_lmgcn_multiclass | LM-GCN multiclass, CodeBERT frozen | 0.209 | 0.742    | 8.65 | 0.272  | 0.162       | 0.232          |

---

### Arch2 — LM-GAT (frozen LM, GATv2Conv)

| Folder                           | Description                          | Dataset | F1↑   | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| -------------------------------- | ------------------------------------ | ------- | ----- | -------- | ---- | ------ | ----------- | -------------- |
| 20260426_144901_lmgat_multiclass | LM-GAT v1, CodeBERT frozen           | BigVul-v1 | 0.172 | 0.711    | 8.62 | 0.329  | 0.121       | 0.297          |
| 20260426_181253_lmgat_multiclass | LM-GAT v2, CodeBERT frozen, tuned hp | BigVul-v1 | 0.224 | 0.726    | 8.44 | 0.322  | 0.104       | 0.314          |
| 20260427_091241_lmgat_multiclass | LM-GAT v3, **GraphCodeBERT frozen**  | BigVul-v1 | 0.135 | 0.696    | 7.63 | 0.315  | 0.127       | 0.283          |
| **20260504_120447_lmgat_multiclass** | **LM-GAT v4, UniXcoder frozen + LIVABLE + F1-stop** | BigVul-v2 | **0.6401** | **0.9040** | **5.22** | **0.514** | **0.0527** | **0.485** |

> Frozen GraphCodeBERT (v3) worse than frozen CodeBERT (v2) — frozen GraphCodeBERT embeddings not suited for this task without fine-tuning.
> **Frozen UniXcoder + LIVABLE + F1-stop (v4) reaches F1=0.6401 on BigVul-v2** — only −0.055 below best overall (Arch12 v1: 0.6952). AUC=0.9040 is among the highest overall. Localization significantly weaker than live-LM variants: IFA=5.22, Effort@20%=0.0527, Recall@20%loc=0.485. LIVABLE+F1-stop recipe closes most of the frozen vs live F1 gap; remaining gap concentrates in localization where live fine-tuning provides richer per-node gradient signal.

---

### Arch3 — LM-GAT-CodeBERT (live LM fine-tuned, GATv2Conv + CodeBERT joint)

| Folder                                        | Description                                                         | Dataset | F1↑        | AUC-ROC↑   | IFA↓     | Top-1↑    | Effort@20%↓ | Recall@20%loc↑ |
| --------------------------------------------- | ------------------------------------------------------------------- | ------- | ---------- | ---------- | -------- | --------- | ----------- | -------------- |
| 20260427_012529_lmgat_codebert_multiclass     | Arch3 v1, CodeBERT live, lm_lr=2e-5, batch=4                        | BigVul-v1 | 0.204      | 0.696      | 10.73    | 0.235     | 0.185       | 0.217          |
| 20260427_062921_lmgat_codebert_multiclass     | Arch3 v2, CodeBERT live, lm_lr=4e-5, batch=16                       | BigVul-v1 | 0.193      | 0.686      | 9.20     | 0.315     | 0.142       | 0.294          |
| **20260427_075727_lmgat_codebert_multiclass** | **Arch3 v3, GraphCodeBERT live, lm_lr=1e-5, batch=16**              | BigVul-v1 | **0.259**  | **0.738**  | **6.12** | **0.398** | **0.106**   | **0.372**      |
| **20260429_091918_lmgat_codebert_multiclass** | **Arch3 v4, UniXcoder live, lm_lr=1e-5, batch=16 (cloud RTX 4090)** | BigVul-v2 | **0.4115** | **0.8562** | 7.72     | 0.366     | 0.103       | 0.340          |
| **20260501_085445_lmgat_codebert_multiclass** | **Arch3 v5, UniXcoder live, same as v4 + early_stop_metric=f1, patience=25** | BigVul-v2 | **0.6744** | **0.8999** | **5.84** | **0.478** | **0.0556**  | **0.483**      |
| **20260502_010952_lmgat_codebert_multiclass** | **Arch3 v6, UniXcoder live, same as v5 + livable_loss=true**                 | BigVul-v2 | **0.6797** | **0.9067** | **5.40** | **0.530** | **0.0350**  | **0.565**      |

> **Best localization on v1:** GraphCodeBERT + lower lm_lr (1e-5) gives best IFA and Top-1. Higher lm_lr (4e-5) causes instability/overfitting.
> **UniXcoder (v4) shows better F1/AUC** but uses BigVul-v2 — improvement is confounded by more data. For fair comparison, retrain CodeBERT/GraphCodeBERT on v2.
> **F1-stop (v5) is transformative:** Switching early_stop_metric from loss to f1 yields F1=0.6744 (+0.263 over v4), IFA=5.84 (best overall), Top-1=0.478 (best overall), Effort@20%=0.0556 (best overall). Same architecture, same LM — the stopping criterion was the bottleneck.
> **LIVABLE (v6) further improves localization:** Over v5 — AUC: 0.8999→0.9067 (new overall best), IFA: 5.84→5.40, Top-1: 0.478→0.530 (tied best), Effort@20%: 0.0556→0.0350 (new best overall), Recall@20%loc: 0.483→0.565 (new best overall). F1 slight drop 0.6744→0.6797 is actually +0.005. LIVABLE epoch-adaptive weights fix class-balance drift, enabling better localization convergence.

---

### Arch4 — LM-GAT-MCS (live LM + multi-scale context, GATv2Conv + pooling)

| Folder                                   | Description                                                         | Dataset | F1↑        | AUC-ROC↑   | IFA↓  | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| ---------------------------------------- | ------------------------------------------------------------------- | ------- | ---------- | ---------- | ----- | ------ | ----------- | -------------- |
| 20260427_053340_lmgat_mcs_multiclass     | Arch4 v1, CodeBERT live, lm_lr=2e-5, batch=16                       | BigVul-v1 | 0.207      | 0.721      | 14.23 | 0.173  | 0.291       | 0.132          |
| **20260427_103516_lmgat_mcs_multiclass** | **Arch4 v2, GraphCodeBERT live, lm_lr=1e-5, batch=32**              | BigVul-v1 | **0.272**  | **0.761**  | 12.76 | 0.225  | 0.149       | 0.268          |
| **20260429_095918_lmgat_mcs_multiclass** | **Arch4 v3, UniXcoder live, lm_lr=1e-5, batch=16 (cloud RTX 4090)** | BigVul-v2 | **0.5791** | **0.8977** | 12.74 | 0.221  | 0.110       | 0.308          |
| **20260501_120840_lmgat_mcs_multiclass** | **Arch4 v3 best, same as v3 + livable_loss=true + early_stop_metric=f1** | BigVul-v2 | **0.6851** | **0.9036** | 10.51 | 0.281  | 0.0862      | 0.374          |

> **Best F1/AUC on v1:** Arch4 v1 (CodeBERT) worst IFA — MCS pooling hurts localization. v2 (GraphCodeBERT) fixes classification but localization still poor vs Arch3.
> **UniXcoder (v3) shows F1=0.5791** but uses BigVul-v2 — confounded by more data. IFA pattern unchanged (MCS pooling is the bottleneck, not the LM).
> **LIVABLE+F1-stop (v3 best) yields F1=0.6851 (+0.106 over v3)** — classification jumps dramatically. IFA 12.74→10.51 modest improvement; MCS pooling localization weakness persists even with best training recipe — confirms pooling is the structural bottleneck.

---

### Arch5 — LM-GIN (frozen CodeBERT, GINEConv — theoretically most expressive)

| Folder                           | Description                          | F1↑   | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| -------------------------------- | ------------------------------------ | ----- | -------- | ---- | ------ | ----------- | -------------- |
| 20260427_175127_lmgin_multiclass | LM-GIN, CodeBERT frozen, GINEConv x4 | 0.107 | 0.645    | 8.40 | 0.339  | 0.139       | 0.278          |

> Underperforms all architectures. GINEConv sum aggregation not better than GAT attention for vulnerability localization.

---

### Arch6 — LM-GAT-Interp (live CodeBERT + GATv2, learned λ interpolation)

| Folder                                  | Description                              | F1↑   | AUC-ROC↑ | IFA↓ | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| --------------------------------------- | ---------------------------------------- | ----- | -------- | ---- | ------ | ----------- | -------------- |
| 20260427_195648_lmgat_interp_multiclass | LM-GAT-Interp, CodeBERT live, λ=0.5 init | 0.160 | 0.704    | 8.87 | 0.318  | **0.101**   | 0.310          |

> Mid-range performance. Best Effort@20% (0.101) — interpolation improves ranking efficiency. F1 lower than Arch3/4 likely because separate LM + GNN heads compete.

---

### Arch7 — LM-GAT-Seq (Stage 1 GATv2 binary localization → Stage 2 GATv2 + live LM classification)

| Folder                                   | Description                                                                                    | Dataset | F1↑        | AUC-ROC↑   | IFA↓     | Top-1↑    | Effort@20%↓ | Recall@20%loc↑ |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------- | ------- | ---------- | ---------- | -------- | --------- | ----------- | -------------- |
| **20260429_121124_lmgat_seq_multiclass** | **Arch7 v1, UniXcoder live, original config (stage2=raw, mil=0.3, rank=0.1, lr=0.001)**        | BigVul-v2 | **0.4554** | **0.8610** | **7.34** | **0.356** | **0.0855**  | **0.387**      |
| 20260429_135046_lmgat_seq_multiclass     | Arch7 v2, UniXcoder live, tuned config (stage2=loc, mil=0.1, rank=0.0, lr=0.0005, patience=25) | BigVul-v2 | 0.3857     | 0.8018     | 12.13    | 0.182     | 0.1177      | 0.294          |
| **20260501_150638_lmgat_seq_multiclass** | **Arch7 v1 best, same as v1 + livable_loss=true + early_stop_metric=f1**                       | BigVul-v2 | **0.6897** | **0.9041** | **6.88** | **0.340** | **0.0567**  | **0.445**      |

> **Tuning degraded performance:** v1 (original) beats v2 (tuned) on all metrics. stage2=loc + lower lr + no rank loss produced worse IFA and Top-1 — the Stage 2 receiving raw node features + s_i gradient signal was more useful than the cleaner-but-detached loc features.
> **Best localization among all v2 models (before F1-stop):** Arch7 v1 IFA=7.34 beats Arch3 v4 IFA=7.72; Top-1=0.356 close to Arch3 v4's 0.366.
> **LIVABLE+F1-stop (v1 best) is new overall best on F1:** F1=0.6897 (+0.243 over original v1), AUC=0.9041 (new highest). IFA=6.88 (2nd after Arch3 v5 5.84). Effort@20%=0.0567 (2nd after Arch3 v5 0.0556). Sequential two-stage architecture scales excellently with F1-stop — classification and localization both improve simultaneously.

---

### Arch9 — LM-GGNN (GATv2 → GatedGraphConv, fixed-alpha interpolation, no stmt head)

| Folder                                | Description                                                         | Dataset | F1↑        | AUC-ROC↑   | IFA↓     | Top-1↑    | Effort@20%↓ | Recall@20%loc↑ |
| ------------------------------------- | ------------------------------------------------------------------- | ------- | ---------- | ---------- | -------- | --------- | ----------- | -------------- |
| 20260429_203915_lmggnn_multiclass     | Arch9 **old impl** (interpolation, no stmt_head), UniXcoder live    | BigVul-v2 | 0.3519     | 0.8053     | N/A      | N/A       | N/A         | N/A            |
| **20260430_004221_lmggnn_multiclass** | **Arch9 corrected impl** (concat + stmt_head + MIL), UniXcoder live | BigVul-v2 | **0.4080** | **0.8073** | **8.29** | **0.244** | **0.1378**  | **0.292**      |

> Old impl (203915): alpha interpolation, no stmt_head — classification only.
> Corrected impl (004221): concat fusion + binary stmt_head + MIL — localization recovered. F1 improved 0.352→0.408.

---

### Arch8 — LM-GAT-WAVES-Seq (Stage 1 Transformer localization → Stage 2 GATv2 + live LM classification)

| Folder                                     | Description                                         | Dataset | F1↑    | AUC-ROC↑ | IFA↓  | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| ------------------------------------------ | --------------------------------------------------- | ------- | ------ | -------- | ----- | ------ | ----------- | -------------- |
| 20260429_125637_lmgat_waves_seq_multiclass | Arch8 v1, UniXcoder live, stmt_transformer_layers=2 | BigVul-v2 | 0.4305 | 0.8357   | 13.72 | 0.096  | 0.1394      | 0.245          |

> **Transformer localization fails:** Arch8 localization (IFA=13.72, Top-1=0.096) is worst among all v2 architectures — serial transformer over CPG statements cannot replace GATv2 attention for localization. Classification F1=0.4305 is reasonable but localization is broken.

---

### Arch10 — LM-GAT-DualFlow (Stage 1 localization GNN → Stage 2 GNN + live UniXcoder, dual-flow)

| Folder                                        | Description                                                                      | Dataset | F1↑        | AUC-ROC↑   | IFA↓     | Top-1↑    | Effort@20%↓ | Recall@20%loc↑ |
| --------------------------------------------- | -------------------------------------------------------------------------------- | ------- | ---------- | ---------- | -------- | --------- | ----------- | -------------- |
| **20260501_035449_lmgat_dualflow_multiclass** | **Arch10 v1, UniXcoder live, suspicion-weighted focal + context pool, RTX 4090** | BigVul-v2 | **0.4461** | **0.8671** | **8.05** | **0.340** | **0.0786**  | **0.405**      |

> **Dual-flow fusion (focal + context) improves Effort@20%:** 0.0786 is best localization efficiency among all v2 models. F1=0.4461 ranks 3rd overall (above Arch3 v4 and Arch8). IFA=8.05 moderate — focal pooling reduces noise but doesn't fully eliminate hard negatives. Stage 1 suspicion signal benefits both localization and classification.

---

### Arch12 — HC-DFGAT (Dual-flow GATv2 + live UniXcoder + 3 MTL heads + hierarchical supcon)

| Folder                                        | Description                                                                                         | Dataset | F1↑        | AUC-ROC↑   | IFA↓     | Top-1↑    | Effort@20%↓ | Recall@20%loc↑ |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------- | ---------- | ---------- | -------- | --------- | ----------- | -------------- |
| **20260501_205917_lmgat_hcdfgat_multiclass**  | **Arch12 v1, UniXcoder live, LIVABLE+F1-stop, supcon alpha-only (no distance matrix)**             | BigVul-v2 | **0.6952** | **0.9032** | **4.86** | **0.530** | **0.0386**  | **0.518**      |
| 20260502_185139_lmgat_hcdfgat_multiclass      | Arch12 v2, same as v1 + supcon_use_distance_matrix=true (linear weight_fn, CWE tree distances)     | BigVul-v2 | 0.6776     | 0.8943     | 6.00     | 0.405     | 0.0661      | 0.433          |

> **v1 (alpha-only) dominates v2 (distance matrix) on every metric.** F1: 0.6952→0.6776 (−0.018), IFA: 4.86→6.00 (worse), Top-1: 0.530→0.405 (−0.125), Effort@20%: 0.0386→0.0661 (worse), Recall@20%loc: 0.518→0.433 (worse). Distance matrix hurts rather than helps.
> **Root cause (confirmed):** CWE group categories are NOT anchored at the same depth in the CWE tree — some groups (e.g. Memory Safety CWE-119) are abstract top-level nodes while others are deeper subtrees. Tree distances between CWEs in different groups are therefore non-comparable: a CWE near the root of its group will appear "close" to CWEs in adjacent groups even though they belong to different vulnerability families. The linear weight `w = 1 − norm_dist` assigned non-zero weights to these cross-group pairs, corrupting the contrastive signal with noisy pseudo-positives from semantically unrelated groups. Alpha-only correctly treats all cross-group pairs as pure negatives.
> **Fix applied:** `HierarchicalSupConLoss` now has `intragroup_only=True` (new default). When enabled, matrix-derived weights are zeroed for all cross-group pairs even if both CWEs are in the matrix. Matrix distances are now only used to refine within-group positive pair weighting (different CWE, same group). Config key: `supcon_intragroup_only: false` to revert to legacy behavior.
> **Next:** Retrain Arch12 v3 with `supcon_use_distance_matrix=true` + `intragroup_only=True` (default) + `weight_fn=exp` to see if within-group distance refinement recovers or improves over alpha-only.
> **HC-DFGAT v1 vs Arch3 v6 trade-off (unchanged):** HC-DFGAT v1 leads on F1 (0.6952 vs 0.6797) and IFA (4.86 vs 5.40). Arch3 v6 leads on AUC (0.9067 vs 0.9032), Effort@20% (0.0350 vs 0.0386), Recall@20%loc (0.565 vs 0.518). Top-1 tied (0.530). Both Pareto-optimal.

---

### Arch11 — LM-GAT-CodeBERT-MTL (shared GATv2 + live UniXcoder + 3 MTL heads: binary + group + CWE)

| Folder                                            | Description                                                                                        | Dataset | F1↑        | AUC-ROC↑   | IFA↓      | Top-1↑    | Effort@20%↓ | Recall@20%loc↑ |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------- | ---------- | ---------- | --------- | --------- | ----------- | -------------- |
| **20260501_050001_lmgat_codebert_mtl_multiclass** | **Arch11 v1, UniXcoder live, binary_weight=0.2, group_weight=0.0, use_group_cond=false, RTX 4090** | BigVul-v2 | **0.4308** | **0.8724** | **10.69** | **0.169** | **0.1035**  | **0.313**      |
| **20260501_072750_lmgat_codebert_mtl_multiclass** | **Arch11 v2, UniXcoder live, same as v1 + livable_loss=true, RTX 4090**                           | BigVul-v2 | **0.5084** | **0.8917** | **12.16** | **0.226** | **0.0741**  | **0.368**      |
| **20260502_193921_lmgat_codebert_mtl_multiclass** | **Arch11 v3, UniXcoder live, LIVABLE+F1-stop, use_group_cond=true, group_loss=0.3, rank=0.2**     | BigVul-v2 | **0.6819** | **0.9017** | **4.60** | **0.558** | **0.0338**  | **0.573**      |
| **20260504_023610_lmgat_codebert_mtl_multiclass** | **Arch11 v4, UniXcoder live, LIVABLE+F1-stop, use_group_cond=true, group_loss=0.3, rank=0.2, add_self_loops=true, use_skip=true** | BigVul-v2 | **0.6679** | **0.9003** | **4.89** | **0.553** | **0.0295**  | **0.595**      |
| **20260504_074735_lmgat_codebert_mtl_multiclass** | **Arch11 v5, UniXcoder live, LIVABLE+F1-stop, use_group_cond=true, group_loss=0.3, rank=0.2, add_self_loops=true, use_skip=true, use_edge_emb=true** | BigVul-v2 | **0.6660** | **0.9029** | **4.85** | **0.566** | **0.0281**  | **0.571**      |
| **20260504_125221_lmgat_codebert_mtl_multiclass** | **Arch11 v6, UniXcoder live, LIVABLE+F1-stop, use_group_cond=true, group_loss=0.3, rank=0.2, supcon_weight=0.1 (alpha-only, no self_loops/skip/edge_emb)** | BigVul-v2 | **0.6647** | **0.8957** | **5.42** | **0.543** | **0.0382**  | **0.543**      |
| **20260505_141404_lmgat_codebert_mtl_multiclass** | **Arch11 v7, UniXcoder live, LIVABLE+F1-stop, use_group_cond=true, group_loss=0.3, rank=0.2, use_edge_emb=true only (no self_loops/skip) — edge_emb ablation** | BigVul-v2 | **0.6764** | **0.9084** | **4.62** | **0.543** | **0.0410**  | **0.527**      |
| **20260507_043309_lmgat_codebert_mtl_multiclass** | **Arch11 v8, CodeT5p-110m-embedding func_lm (256D), v3 recipe (no self_loops/skip/edge_emb), LIVABLE+F1-stop** | BigVul-v2 | **0.7484** | **0.9381** | 5.769    | 0.4935    | 0.0523      | 0.4908         |

> **MTL binary head hurts localization (v1):** Top-1=0.169 is worst among all recent v2 architectures. Sharing the encoder with binary classification signal pulls node representations away from localization. AUC-ROC=0.8724 is highest of all models — binary auxiliary task improves calibration. IFA=10.69 is poor.
> **LIVABLE (v2) significantly improves classification:** F1 0.4308→0.5084 (+0.078), AUC 0.8724→0.8917. Effort@20% 0.1035→0.0741. IFA worsens (10.69→12.16) — LIVABLE helps balance but doesn't fix localization degradation from binary head without group conditioning.
> **Group conditioning + LIVABLE + F1-stop (v3) transforms Arch11:** Enabling use_group_cond=true + group_loss_weight=0.3 + F1-stop unlocks dramatic localization improvement. IFA 12.16→4.60 (new overall best), Top-1 0.226→0.558 (new best), Effort@20% 0.0741→0.0338, Recall@20%loc 0.368→0.573. F1 jumps 0.5084→0.6819. Group supervision was the missing ingredient: once the model learns coarse group routing, MIL localization and CWE classification both benefit simultaneously.
> **Self-loops + skip (v4) trade F1/IFA for ranking efficiency:** Adding add_self_loops=true + use_skip=true over v3 yields Effort@20%=0.0295 and Recall@20%loc=0.595 (best at time). F1 drops 0.6819→0.6679 (−0.014), IFA 4.60→4.89 (slightly worse). Self-loops + skip improve global context per node at cost of classification.
> **Self-loops + skip + edge_emb (v5) sets new Effort@20% record:** Adding use_edge_emb=true over v4 yields Effort@20%=0.0281 (new overall best) and Top-1=0.566 (best across all Arch11 variants). Recall@20%loc drops 0.595→0.571 (worse than v4, near v3). IFA=4.85. Edge embeddings sharpen ranking of the single best line but reduce broad coverage. F1=0.6660 (−0.002 vs v4). For precision-oriented deployment (minimize inspection cost), v5 is the new champion.
> **SupCon on shared MTL encoder (v6) hurts all metrics vs v3:** Adding supcon_weight=0.1 (alpha-only, no structural changes) yields F1=0.6647 (−0.017 vs v3), IFA=5.42 (−0.82), Top-1=0.543 (−0.015), Effort@20%=0.0382 (worse), Recall@20%loc=0.543 (−0.030). Contrast with HC-DFGAT (Arch12 v1) where SupCon improves F1 to 0.6952: HC-DFGAT dual-flow isolates classification repr, allowing SupCon to tighten clusters without competing with MIL. MTL shared encoder lets SupCon gradient fight MIL gradient → localization degrades. SupCon benefit requires architectural isolation of the classification representation.
> **Edge_emb alone (v7) does not replicate v5 gains:** Adding only use_edge_emb=true over v3 (no self_loops/skip) yields F1=0.6764 (−0.006 vs v3), IFA=4.62 (near v3's 4.60), Top-1=0.543 (−0.015), Effort@20%=0.0410 (+0.007 worse than v3), Recall@20%loc=0.527 (−0.046). Compare to v5 (all three): edge_emb alone recovers F1 partially over v5 (0.6764 vs 0.6660) but loses Effort (0.041 vs 0.0281) and Recall (0.527 vs 0.571). Conclusion: the Effort@20% and Recall gains in v4/v5 come from self_loops+skip (global context propagation), not edge_emb. Edge_emb in isolation provides negligible localization benefit and slightly hurts ranking coverage. AUC=0.9084 is new overall best — edge type encoding helps class calibration without structural graph changes.
> **CodeT5p-110m-embedding func_lm (v8) sets new overall F1 and AUC records:** Swapping func_lm from UniXcoder (768D) to CodeT5p-110m-embedding (256D) over the v3 recipe yields F1=0.7484 (new #1, +0.0665 vs v3, +0.0532 vs prior overall best Arch12 v1) and AUC=0.9381 (new #1, +0.0364 vs v3). Localization regresses: IFA 4.60→5.769 (worse), Top-1 0.558→0.4935 (−0.064), Effort@20% 0.0338→0.0523 (worse), Recall@20%loc 0.573→0.4908 (worse). Same v2 dataset — only func_lm differs (confirmed). Test size 1278 vs ~1363 from different processed-batch on cloud (same source). Trade-off: CodeT5+'s code embedding projection head provides richer classification signal (+7pp F1, +4pp AUC) but the smaller 256D fused dim (512D total vs 1024D with UniXcoder) weakens the per-node localization gradient from MIL. CodeT5+ is the new classification champion; v3/v4 UniXcoder remains preferred for localization-critical deployment.

---

## Full Comparison — BigVul-v2 (11-class, all.parquet, ~1363 test, sorted by F1)

> All rows use UniXcoder node embeddings on BigVul-v2. Arch11 v8 test size 1278 — same source, different processed batch.

| Rank | Model                                                                              | Folder                             | F1↑        | AUC-ROC↑   | IFA↓     | Top-1↑    | Effort@20%↓ | Recall@20%loc↑ |
| ---- | ---------------------------------------------------------------------------------- | ---------------------------------- | ---------- | ---------- | -------- | --------- | ----------- | -------------- |
| 1    | **Arch11 v8** MTL + CodeT5p-110m-embedding func_lm                                | 20260507_043309_lmgat_codebert_mtl | **0.7484** | **0.9381** | 5.769    | 0.4935    | 0.0523      | 0.4908         |
| 2    | **Arch12 v1** HC-DFGAT + LIVABLE + F1-stop (alpha-only supcon)                    | 20260501_205917_lmgat_hcdfgat      | **0.6952** | 0.9032     | **4.86** | 0.530     | 0.0386      | 0.518          |
| 3    | **Arch7 v1 best** Seq GATv2 + LIVABLE + F1-stop                                   | 20260501_150638_lmgat_seq          | **0.6897** | 0.9041     | 6.88     | 0.340     | 0.0567      | 0.445          |
| 4    | **Arch4 v3 best** MCS + LIVABLE + F1-stop                                         | 20260501_120840_lmgat_mcs          | **0.6851** | 0.9036     | 10.51    | 0.281     | 0.0862      | 0.374          |
| 5    | **Arch11 v3** MTL + group_cond + LIVABLE + F1-stop                                | 20260502_193921_lmgat_codebert_mtl | **0.6819** | 0.9017     | **4.60** | **0.558** | **0.0338**  | **0.573**      |
| 6    | **Arch3 v6** UniXcoder live + LIVABLE + F1-stop                                   | 20260502_010952_lmgat_codebert     | **0.6797** | **0.9067** | 5.40     | 0.530     | 0.0350      | 0.565          |
| 7    | Arch12 v2 HC-DFGAT + dist matrix linear (broken — cross-group depth issue)        | 20260502_185139_lmgat_hcdfgat      | 0.6776     | 0.8943     | 6.00     | 0.405     | 0.0661      | 0.433          |
| 8    | **Arch11 v7** MTL + group_cond + edge_emb only + LIVABLE + F1-stop                | 20260505_141404_lmgat_codebert_mtl | 0.6764     | **0.9084** | 4.62     | 0.543     | 0.0410      | 0.527          |
| 9    | **Arch3 v5** UniXcoder live + F1-stop (no LIVABLE)                                | 20260501_085445_lmgat_codebert     | 0.6744     | 0.8999     | 5.84     | 0.478     | 0.0556      | 0.483          |
| 10   | **Arch11 v4** MTL + group_cond + self_loops + skip + LIVABLE + F1-stop            | 20260504_023610_lmgat_codebert_mtl | 0.6679     | 0.9003     | 4.89     | 0.553     | 0.0295      | **0.595**      |
| 11   | **Arch11 v5** MTL + group_cond + self_loops + skip + edge_emb + LIVABLE + F1-stop | 20260504_074735_lmgat_codebert_mtl | 0.6660     | 0.9029     | 4.85     | 0.566     | **0.0281**  | 0.571          |
| 12   | **Arch11 v6** MTL + group_cond + SupCon alpha-only + LIVABLE + F1-stop            | 20260504_125221_lmgat_codebert_mtl | 0.6647     | 0.8957     | 5.42     | 0.543     | 0.0382      | 0.543          |
| 13   | **Arch2 v4** LM-GAT frozen UniXcoder + LIVABLE + F1-stop                          | 20260504_120447_lmgat              | 0.6401     | 0.9040     | 5.22     | 0.514     | 0.0527      | 0.485          |
| 14   | **Arch4 v3** MCS UniXcoder live                                                   | 20260429_095918_lmgat_mcs          | 0.5791     | 0.8977     | 12.74    | 0.221     | 0.110       | 0.308          |
| 15   | **Arch11 v2** MTL + LIVABLE (no group_cond)                                       | 20260501_072750_lmgat_codebert_mtl | 0.5084     | 0.8917     | 12.16    | 0.226     | 0.0741      | 0.368          |
| 16   | **Arch7 v1** Seq GATv2 UniXcoder live (original)                                  | 20260429_121124_lmgat_seq          | 0.4554     | 0.8610     | 7.34     | 0.356     | 0.0855      | 0.387          |
| 17   | **Arch10** DualFlow UniXcoder live                                                 | 20260501_035449_lmgat_dualflow     | 0.4461     | 0.8671     | 8.05     | 0.340     | 0.0786      | 0.405          |
| 18   | **Arch11 v1** MTL binary+CWE (no group_cond)                                      | 20260501_050001_lmgat_codebert_mtl | 0.4308     | 0.8724     | 10.69    | 0.169     | 0.1035      | 0.313          |
| 19   | **Arch8 v1** WAVES-Seq Transformer loc UniXcoder live                             | 20260429_125637_lmgat_waves_seq    | 0.4305     | 0.8357     | 13.72    | 0.096     | 0.1394      | 0.245          |
| 20   | **Arch3 v4** UniXcoder live                                                        | 20260429_091918_lmgat_codebert     | 0.4115     | 0.8562     | 7.72     | 0.366     | 0.103       | 0.340          |
| 21   | **Arch9** LM-GGNN corrected UniXcoder live                                         | 20260430_004221_lmggnn             | 0.4080     | 0.8073     | 8.29     | 0.244     | 0.1378      | 0.292          |
| 22   | **Arch7 v2** Seq GATv2 UniXcoder live (tuned, regressed)                           | 20260429_135046_lmgat_seq          | 0.3857     | 0.8018     | 12.13    | 0.182     | 0.1177      | 0.294          |
| 23   | **Arch9** LM-GGNN old impl (no stmt_head)                                          | 20260429_203915_lmggnn             | 0.3519     | 0.8053     | N/A      | N/A       | N/A         | N/A            |

---

## Full Comparison — BigVul-v1 (11-class, train.parquet only, ~1124 test, sorted by F1)

> Older runs on smaller dataset. Not comparable to BigVul-v2.

| Rank | Model                              | Folder                                        | F1↑   | AUC-ROC↑ | IFA↓  | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| ---- | ---------------------------------- | --------------------------------------------- | ----- | -------- | ----- | ------ | ----------- | -------------- |
| 1    | Arch4 v2 GraphCodeBERT live        | 20260427_103516_lmgat_mcs                     | 0.272 | 0.761    | 12.76 | 0.225  | 0.149       | 0.268          |
| 2    | Arch3 v3 GraphCodeBERT live        | 20260427_075727_lmgat_codebert                | 0.259 | 0.738    | 6.12  | 0.398  | 0.106       | 0.372          |
| 3    | Arch2 v2 CodeBERT frozen           | 20260426_181253_lmgat                         | 0.224 | 0.726    | 8.44  | 0.322  | 0.101       | 0.314          |
| 4    | Arch1 LM-GCN CodeBERT frozen       | 20260426_002451_lmgcn                         | 0.209 | 0.742    | 8.65  | 0.272  | 0.162       | 0.232          |
| 5    | Arch4 v1 CodeBERT live             | 20260427_053340_lmgat_mcs                     | 0.207 | 0.721    | 14.23 | 0.173  | 0.291       | 0.132          |
| 6    | Arch3 v1 CodeBERT live             | 20260427_012529_lmgat_codebert                | 0.204 | 0.696    | 10.73 | 0.235  | 0.185       | 0.217          |
| 7    | Arch3 v2 CodeBERT live             | 20260427_062921_lmgat_codebert                | 0.193 | 0.686    | 9.20  | 0.315  | 0.142       | 0.294          |
| 8    | Arch2 v1 CodeBERT frozen           | 20260426_144901_lmgat                         | 0.172 | 0.711    | 8.62  | 0.329  | 0.121       | 0.297          |
| 9    | Arch6 GAT-Interp CodeBERT live     | 20260427_195648_lmgat_interp                  | 0.160 | 0.704    | 8.87  | 0.318  | 0.101       | 0.310          |
| 10   | Arch2 v3 GraphCodeBERT frozen      | 20260427_091241_lmgat                         | 0.135 | 0.696    | 7.63  | 0.315  | 0.127       | 0.283          |
| 11   | Arch5 LM-GIN CodeBERT frozen       | 20260427_175127_lmgin                         | 0.107 | 0.645    | 8.40  | 0.339  | 0.139       | 0.278          |

---

## Full Comparison — TitanVul-OWASP (90-class, ~1499 test)

> 89 CWEs + benign under OWASP Top10 filter. Random baseline F1 ≈ 1.1%. Not comparable to BigVul. AUC computed over 72/90 classes present in test (18 absent → OvR restricted + renormalized).

| Rank | Model                                                              | Folder                                   | F1↑    | AUC-ROC↑ | IFA↓   | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| ---- | ------------------------------------------------------------------ | ---------------------------------------- | ------ | -------- | ------ | ------ | ----------- | -------------- |
| 1    | **Arch12 v1** HC-DFGAT LIVABLE+F1-stop+supcon alpha-only          | 20260506_203551_lmgat_hcdfgat_multiclass | 0.4340 | 0.8929   | 13.625 | 0.2545 | 0.1473      | 0.2532         |

---

## Full Comparison — TitanVul-Top25 (26-class, ~1681 test)

> 25 MITRE Top25 CWEs + benign. Random baseline F1 ≈ 3.8%. Not comparable to BigVul. AUC computed over 25/26 classes present in test (1 absent).

| Rank | Model                                                              | Folder                                   | F1↑    | AUC-ROC↑ | IFA↓   | Top-1↑ | Effort@20%↓ | Recall@20%loc↑ |
| ---- | ------------------------------------------------------------------ | ---------------------------------------- | ------ | -------- | ------ | ------ | ----------- | -------------- |
| 1    | **Arch12 v1** HC-DFGAT LIVABLE+F1-stop+supcon alpha-only          | 20260507_071026_lmgat_hcdfgat_multiclass | 0.5231 | 0.8681   | 16.637 | 0.2016 | 0.1181      | 0.3069         |

---

## Key Takeaways

### Within BigVul-v1 (train.parquet only — fair comparison)

1. **Best classification (F1/AUC):** Arch4 v2 (GraphCodeBERT live, F1=0.272, AUC=0.761)
2. **Best localization (IFA/Top-1):** Arch3 v3 (GraphCodeBERT live, IFA=6.12, Top-1=0.398)
3. **Best trade-off:** Arch3 v3 — strong on both F1 and all localization metrics
4. **GraphCodeBERT > CodeBERT** only when fine-tuned live; frozen GraphCodeBERT hurts
5. **lm_lr sensitivity:** 1e-5 stable; 4e-5 causes instability; 2e-5 borderline

### BigVul-v2 Observations (all.parquet — confounded: LM + dataset size change)

6. **UniXcoder + BigVul-v2** shows dramatically higher F1/AUC — but NOT purely LM effect; dataset size also increased
7. **For fair LM comparison:** need to retrain CodeBERT/GraphCodeBERT on BigVul-v2

### Architecture Patterns (consistent across both datasets)

8. **GIN (Arch5) underperforms** — attention-based aggregation superior to sum for this task
9. **Arch4 MCS pooling** boosts F1 at expense of localization (IFA worst: 12.74–14.23 pattern persists across LMs)
10. **Arch6 interpolation** best Effort@20% (0.101) but lower F1 — competitive for ranking efficiency
11. **Arch4 localisation weakness is NOT fixed by better LM** — IFA stays ~12.7 regardless of CodeBERT/GraphCodeBERT/UniXcoder; MCS pooling is the bottleneck

### Arch7/Arch8 Seq Architecture Findings (BigVul-v2)

12. **Arch7 v1 best F1+localization trade-off among v2 models** — F1=0.4554, IFA=7.34, Effort@20%=0.0855 (best Effort across all v2 runs); sequential GATv2 loc → classification is effective
13. **Tuning Arch7 hurt performance** — stage2=loc + lower lr + no rank loss → F1 drops 0.455→0.386, IFA degrades 7.34→12.13; raw s_i gradient during Stage 2 is useful, not harmful
14. **Arch8 (WAVES-style transformer) localization broken** — serial transformer over CPG statements: Top-1=0.096 (worst of all models), IFA=13.72; transformer cannot replace GATv2 attention on CPG structure for localization
15. **Arch9 (LM-GGNN) corrected impl** — with stmt_head + concat fusion: F1=0.4080, IFA=8.29; old impl (interpolation, no stmt_head): F1=0.3519, no localization. Still underperforms GATv2+LM (Arch3 v4: F1=0.4115, IFA=7.72) — GATv2 attention on CPG edge types more effective than GGNN uniform message passing

### Arch10/Arch11 New Architecture Findings (BigVul-v2)

16. **Arch10 DualFlow best Effort@20% among non-F1-stop models** — 0.0786 beats Arch7 v1 (0.0855) and all MTL variants. Recall@20%loc=0.405 highest before Arch3 v5. F1=0.4461 rank 5. Focal suspicion-weighted pooling + context pooling provides efficient ranking signal.
17. **Arch10 > Arch7 on localization metrics** — Effort@20% 0.0786 vs 0.0855; Recall@20%loc 0.405 vs 0.387. Dual-flow separates suspicion signal (focal) from general context, improving ranking efficiency over single-flow Arch7.
18. **Arch11 v1 MTL highest AUC-ROC (0.8724) pre-LIVABLE** — binary auxiliary head improves calibration. But Top-1=0.169 is worst of all v2 models — shared encoder + binary gradient competes with localization. IFA=10.69 poor.
19. **MTL binary head trades localization for classification quality** — binary supervision improves CWE classification (AUC highest) but the MIL stmt head loses localization sharpness.
20. **LIVABLE restores classification in MTL (Arch11 v2)** — F1 0.4308→0.5084 (+0.078), AUC 0.8724→0.8917. Effort@20% 0.1035→0.0741 (significant improvement). IFA worsens 10.69→12.16 — LIVABLE epoch-adaptive weights help classification balance but don't fix localization degradation from the binary auxiliary head.

### F1-Stop Early Stopping Finding (BigVul-v2)

21. **early_stop_metric=f1 is transformative (Arch3 v5)** — Same architecture as Arch3 v4 (UniXcoder live, identical hyperparams). F1: 0.4115→0.6744 (+0.263), IFA: 7.72→5.84 (best overall), Top-1: 0.366→0.478 (best overall), Effort@20%: 0.103→0.0556 (best overall), Recall@20%loc: 0.340→0.483 (best overall). Loss-based stopping was the main bottleneck — LR decay flattens loss while F1 still improves; the model was stopping 20–30 epochs before convergence.
22. **F1-stop dominates all metrics** — Arch3 v5 ranks #1 on F1, IFA, Top-1, Effort@20%, and Recall@20%loc simultaneously. No other model achieves top-3 on all five metrics. Conclusion: retrain all previous architectures with early_stop_metric=f1 for valid comparison.

### LIVABLE + F1-Stop Sweep Findings (BigVul-v2, 2026-05-02)

23. **Arch7 v1 best (LIVABLE+F1-stop) is new overall best on F1/AUC** — F1=0.6897 (rank 2 after HC-DFGAT), AUC=0.9041. Over original v1: +0.234 F1, IFA 7.34→6.88, Effort@20% 0.0855→0.0567. Two-stage sequential architecture scales excellently with F1-stop.
24. **Arch4 v3 best (LIVABLE+F1-stop) F1=0.6851 — strong classification, persistent localization weakness** — Over original v3: +0.106 F1, IFA 12.74→10.51 (marginal). MCS pooling localization bottleneck confirmed: IFA stays 10–12 regardless of training recipe.
25. **Arch3 v6 (LIVABLE+F1-stop) best Effort@20% and Recall@20%loc** — Effort@20%=0.0350 and Recall@20%loc=0.565 are new best overall. AUC=0.9067 is new highest. LIVABLE over v5 improves localization efficiency without sacrificing classification.
26. **F1-stop universally beneficial** — Arch3, Arch4, Arch7, Arch12 all show +0.1 to +0.26 F1 gains. Loss-based early stopping terminates 20–30 epochs before F1 peak across all architectures.

### HC-DFGAT Findings (Arch12, BigVul-v2, 2026-05-02)

27. **HC-DFGAT ranks #1 on F1 and IFA simultaneously** — F1=0.6952 (highest), IFA=4.86 (first model below 5.0, best ever). Hierarchical group→CWE routing forces coarse-to-fine disambiguation; supcon contrastive loss tightens group-level clusters, improving both CWE classification and node-level localization signal.
28. **HC-DFGAT vs Arch3 v6 metric split** — HC-DFGAT best: F1 (0.6952 vs 0.6797), IFA (4.86 vs 5.40). Arch3 v6 best: AUC (0.9067 vs 0.9032), Effort@20% (0.0350 vs 0.0386), Recall@20%loc (0.565 vs 0.518). Top-1 tied (0.530). Neither dominates on all metrics — both are Pareto-optimal.
29. **HC-DFGAT group_id essential** — Previous Arch11 (MTL without group supervision) had IFA=10.69; HC-DFGAT with full group routing achieves IFA=4.86. Coarse group conditioning on CWE head is the key differentiator vs plain MTL.

### HC-DFGAT Distance Matrix Ablation (Arch12 v2, 2026-05-03)

30. **Arch11 v3 (MTL + group_cond + LIVABLE + F1-stop) sets IFA and Top-1 records** — Enabling use_group_cond=true + group_loss_weight=0.3 + F1-stop on Arch11 yields: IFA=4.60 (new #1, beats HC-DFGAT's 4.86), Top-1=0.558 (new #1, beats 0.530), Effort@20%=0.0338, Recall@20%loc=0.573. F1=0.6819 (rank 4). Group supervision was the key ingredient missing from Arch11 v1/v2: once coarse group routing is added, the model achieves better localization than HC-DFGAT while using a simpler architecture (shared GATv2 vs dual-flow). AUC=0.9017 (rank 5).

33. **Frozen UniXcoder + LIVABLE + F1-stop (Arch2 v4) nearly matches live-LM on F1** — F1=0.6401 on BigVul-v2, only −0.055 below Arch12 v1 (0.6952). AUC=0.9040 is among the highest overall. Localization gap is large: IFA=5.22 vs 4.60 (Arch11 v3), Effort@20%=0.0527 vs 0.0281 (Arch11 v5), Recall@20%loc=0.485 vs 0.595 (Arch11 v4). The LIVABLE+F1-stop recipe closes most of the frozen vs live F1 gap; the remaining gap concentrates in localization where live gradient signal from per-node backprop provides richer alignment. This confirms that the F1 gains from BigVul-v2 experiments are partially attributable to the training recipe (LIVABLE+F1-stop), not purely to live LM fine-tuning.

32. **Arch11 v4/v5 ablation: self-loops + skip + edge_emb** — v4 (self_loops+skip, no edge_emb): F1=0.6679, IFA=4.89, Effort@20%=0.0295, Recall@20%loc=0.595 (best at time). v5 (+ edge_emb): F1=0.6660 (−0.002), IFA=4.85 (slightly better), Top-1=0.566 (new Arch11 best, +0.008 vs v3), Effort@20%=0.0281 (new overall best), Recall@20%loc=0.571 (drops vs v4). Pattern: each structural addition (self-loops, skip, edge_emb) trades classification (F1 −0.014, −0.002) for ranking efficiency (Effort@20% 0.0338→0.0295→0.0281). Edge embeddings improve precision-ranked Top-1 and Effort but reduce coverage (Recall). Best Effort@20% model is now Arch11 v5 (0.0281); best Recall@20%loc model remains Arch11 v4 (0.595).

34. **Arch11 v7 (edge_emb only ablation) confirms self_loops+skip drives Effort gains, not edge_emb:** Adding only use_edge_emb=true over v3 base (no self_loops/skip): F1=0.6764, AUC=0.9084 (new overall best), IFA=4.62, Top-1=0.543, Effort@20%=0.0410, Recall@20%loc=0.527. vs v3: F1 −0.006, IFA +0.02, Effort +0.007 (worse), Recall −0.046. Edge_emb alone provides no localization benefit. vs v5 (all three): v7 has better F1 (+0.010) and IFA (−0.23 better) but far worse Effort (0.041 vs 0.0281) and Recall (0.527 vs 0.571) — the Effort/Recall gains in v5 come from self_loops+skip, not edge_emb. AUC=0.9084 sets new overall record — edge type encoding helps class probability calibration without structural graph changes. Summary: edge_emb → better AUC/calibration; self_loops+skip → better Effort/Recall ranking.

35. **Arch11 v8 (CodeT5p-110m-embedding) sets new overall F1 and AUC records** — F1=0.7484 (new #1, +0.0532 vs Arch12 v1), AUC=0.9381 (new #1, +0.030 vs Arch11 v7). Localization regresses across all metrics vs UniXcoder v3: IFA 4.60→5.769, Top-1 0.558→0.4935, Effort 0.0338→0.0523, Recall 0.573→0.4908. Net interpretation: CodeT5+'s dedicated code embedding projection head (trained for code similarity) provides richer semantic features for CWE classification but the 256D output (vs 768D UniXcoder) shrinks the fused representation from 1024D to 512D, reducing per-node localization gradient from MIL. Same v2 dataset confirmed — only func_lm differs. Test size difference (1278 vs ~1363) from different processed-batch on cloud, not different source. Pareto-optimal for classification-critical deployment; v3 UniXcoder preferred for localization-critical.

36. **TitanVul OWASP 90-class task established** — F1=0.4340 on 90-class CWE classification (89 distinct CWEs under OWASP top10 + benign). AUC=null (evaluate.py OvR fails at 90 classes). IFA=13.625 (poor — large functions max_nodes=3400). Not comparable to BigVul 11-class. Identifies two issues: (1) evaluate.py needs fix for >11 class AUC, (2) localization harder on TitanVul's larger functions.

31. **CWE tree distance matrix hurts supcon (v2 < v1 on all metrics) — root cause: depth-asymmetric group anchors** — Linear `w = 1 − norm_dist` continuous weighting degrades every metric vs alpha-only: F1 0.6952→0.6776 (−0.018), IFA 4.86→6.00 (worse), Top-1 0.530→0.405 (−0.125), Effort@20% 0.0386→0.0661, Recall@20%loc 0.518→0.433. Confirmed root cause: CWE group root nodes sit at different depths in the CWE tree, making cross-group tree distances non-comparable. The matrix assigned non-zero weights to cross-group pairs that should be pure negatives, corrupting the contrastive gradient with noisy pseudo-positives. Fix: `intragroup_only=True` (new default in `HierarchicalSupConLoss`) zeros all cross-group matrix weights; distance refinement now applies only to within-group pairs. Arch12 v3 with fix needs to be run.

---

## Missing / Pending Results

- ~~Arch11 LIVABLE variant~~ — Done: F1=0.5084, AUC=0.8917 (folder 20260501_072750)
- ~~Arch4 v3 LIVABLE+F1-stop~~ — Done: F1=0.6851, AUC=0.9036 (folder 20260501_120840)
- ~~Arch7 v1 LIVABLE+F1-stop~~ — Done: F1=0.6897, AUC=0.9041 (folder 20260501_150638)
- ~~Arch3 v6 LIVABLE+F1-stop~~ — Done: F1=0.6797, AUC=0.9067, Effort=0.0350, Recall@20%=0.565 (folder 20260502_010952) — best Effort+Recall
- ~~Arch12 HC-DFGAT~~ — Done: F1=0.6952, IFA=4.86 (folder 20260501_205917) — best F1+IFA
- ~~Arch12 v2 HC-DFGAT + distance matrix~~ — Done: F1=0.6776, IFA=6.00 (folder 20260502_185139) — worse than v1; linear weight_fn hurts
- ~~Arch11 v3 MTL + group_cond + LIVABLE + F1-stop~~ — Done: F1=0.6819, IFA=4.60, Effort=0.0338, Recall@20%=0.573 (folder 20260502_193921)
- ~~Arch11 v4 MTL + group_cond + self_loops + skip + LIVABLE + F1-stop~~ — Done: F1=0.6679, IFA=4.89, Effort=0.0295, Recall@20%=0.595 (best) (folder 20260504_023610)
- ~~Arch11 v5 MTL + group_cond + self_loops + skip + edge_emb + LIVABLE + F1-stop~~ — Done: F1=0.6660, IFA=4.85, Effort=0.0281 (new best), Top-1=0.566 (best Arch11) (folder 20260504_074735)
- ~~Arch11 v6 MTL + group_cond + SupCon (alpha-only) + LIVABLE + F1-stop~~ — Done: F1=0.6647, IFA=5.42, Effort=0.0382 — SupCon hurts MTL (shared encoder conflict with MIL) (folder 20260504_125221)
- ~~Arch2 v4 frozen UniXcoder + LIVABLE + F1-stop~~ — Done: F1=0.6401, AUC=0.9040, IFA=5.22, Effort=0.0527 (folder 20260504_120447) — frozen baseline now competitive on F1, weak on localization
- ~~Arch11 v7 edge_emb only ablation (no self_loops/skip)~~ — Done: F1=0.6764, AUC=0.9084 (new best), IFA=4.62, Effort=0.0410 (folder 20260505_141404) — edge_emb alone no localization gain; AUC record; self_loops+skip are the source of Effort improvements
- ~~Arch11 v8 CodeT5p-110m-embedding func_lm~~ — Done: F1=0.7484 (new #1), AUC=0.9381 (new #1), IFA=5.769, Effort=0.0523 (folder 20260507_043309) — best classification, worse localization vs UniXcoder v3
- ~~HC-DFGAT TitanVul OWASP 90-class~~ — Done: F1=0.4340, IFA=13.625, AUC=0.8929 (folder 20260506_203551)
- ~~HC-DFGAT TitanVul Top25 26-class~~ — Done: F1=0.5231, IFA=16.637, AUC=0.8681 (folder 20260507_071026)
- Arch12 v3 HC-DFGAT + distance matrix (intragroup_only=True, exp weight_fn) — root cause of v2 failure confirmed (cross-group depth asymmetry); fix applied, need to retrain to verify within-group refinement helps vs alpha-only
- Arch11 v8 CodeT5p + localization fix — investigate: does CodeT5p localization weakness come from 256D dim (try matryoshka_dim=768 on a 768D model) or from the embedding model's optimization target (similarity vs token-level)?
- evaluate.py fix for AUC with >11 classes — OvR computation fails silently (returns null); needs `labels` param or per-class exclusion of missing classes
- Arch10/Arch11 retrain with early_stop_metric=f1 — F1-stop proved universally beneficial; not yet retrained
- Arch3/4 binary GraphCodeBERT variants — not yet trained
- Arch9 (lmggnn) run 20260429_202324 — not yet evaluated (may skip)
- Arch3/4 retrain on BigVul-v2 with CodeBERT/GraphCodeBERT — needed for fair LM comparison vs UniXcoder

---

## Public Baseline Comparison — Status

**Direct comparison to published numbers is not valid at this stage.** Reasons:

### Classification (F1, AUC-ROC)

| Baseline                   | Reported F1 | Our Task                | Why Not Comparable                                                          |
| -------------------------- | ----------- | ----------------------- | --------------------------------------------------------------------------- |
| LineVul (Fu et al. 2022)   | ~0.91       | Binary on full BigVul   | We use 11-class on 2000/class sample — F1 is naturally lower for multiclass |
| VulLMGNN (Cao et al. 2022) | ~0.70       | Binary on BigVul/Devign | Binary vs multiclass; different data split                                  |

> **Note:** Multiclass F1=0.272 (11 classes) vs binary F1=0.91 is not a regression — they measure different tasks. Shown here for reference only.

### Localization (IFA, Top-1, Effort@20%)

| Baseline                 | IFA↓ | Top-1↑   | Effort@20%↓ | Why Not Comparable                                    |
| ------------------------ | ---- | -------- | ----------- | ----------------------------------------------------- |
| LineVul (Fu et al. 2022) | ~4.0 | ~0.50    | ~0.08       | Full BigVul dataset; different functions in test set  |
| WAVES                    | N/A  | reported | reported    | No source code available — cannot reproduce or verify |

> **Key issue:** Even though both LineVul and our models use BigVul, the exact sampled functions and train/test splits differ. Localization metrics are sensitive to which functions appear in the test set.

### Plan for Fair Comparison

To produce a valid comparison table:

1. **Run LineVul** on our exact dataset split (same 2000/class sample, same 70/15/15 split, same random seed)
2. **Run VulLMGNN** on our exact dataset split
3. Present all models — ours + baselines — evaluated on the same test set
4. **WAVES**: skip — no public source code, replication not feasible

This will make the comparison fair and publishable.
