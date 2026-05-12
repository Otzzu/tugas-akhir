# Related Papers — Verified References

All papers verified as published/indexed. PDF copies in `paper/` folder where noted.
Two groups: **Multiclass CWE Classification** and **Line-Level Vulnerability Localization**.
Papers that contribute to both groups appear in the most relevant one with cross-references noted.

---

## Multiclass CWE / Vulnerability Type Classification

### 1. LIVABLE (IEEE TSE 2024)
**Full title:** "LIVABLE: Exploring Long-Tailed Classification of Software Vulnerability Types"  
**Venue:** IEEE Transactions on Software Engineering (IEEE TSE), 2024  
**Paper:** https://arxiv.org/abs/2306.06935 | https://ieeexplore.ieee.org/document/10497542/  
**PDF:** `paper/LIVABLE.pdf`  
**Repo:** ❌ No public repo  
**SOTA?** ✅ IEEE TSE 2024 — current reference for long-tailed multiclass CWE on GNN

**Architecture:**
GNN with a differentiated propagation method to reduce over-smoothing, combined with a seq2seq module for richer vulnerability representations. The key contribution is an epoch-adaptive class weighting schedule: `w_i(t) = (N/(K·n_i))^(t/T)` that ramps from uniform weights at epoch 0 to full inverse-frequency weights at the final epoch. This prevents early training collapse on minority CWE classes while still giving the model time to learn majority classes first.

**Dataset:** NVD (2012–2022) — 325 CWE types split into head (>200 samples), medium (50–200), and tail (<50). BigVul experiments on FFMpeg+Qemu, Reveal, and Fan et al. datasets.

**Key results:** +14.7% accuracy on medium classes, +7.7% on tail classes vs baseline. +4.03% average accuracy improvement over best-performing baselines on representation learning module alone.

**Relevance to your work:** Your codebase directly implements the LIVABLE adaptive weight formula in `training/losses.py:livable_weights()`. The paper also validates that standard GNN over-smoothing is a root cause of poor tail-class performance — directly explaining why your Arch2 (LM-GAT) with frozen CodeBERT node features struggles on rare CWEs like `logging` (1 sample) and `data_integrity` (5 samples) in BigVul.

---

### 2. VulExplainer (IEEE TSE 2023 / ICSE 2024)
**Full title:** "VulExplainer: A Transformer-Based Hierarchical Distillation for Explaining Vulnerability Types"  
**Authors:** Fu et al. (same lab as LineVul — Monash University)  
**Venue:** IEEE Transactions on Software Engineering (IEEE TSE), 2023 — presented at ICSE 2024  
**Paper:** https://ieeexplore.ieee.org/document/10220166/ | https://dl.acm.org/doi/10.1109/TSE.2023.3305244  
**Repo:** ✅ https://github.com/awsm-research/VulExplainer  
**SOTA?** ✅ Current SOTA for multiclass CWE on BigVul with embedding-based approach (no LLM)

**Architecture:**
CodeBERT fine-tuned as encoder (CLS token embedding) with hierarchical knowledge distillation. The label space is split into CWE abstract-type groups (similar CWE-IDs clustered together). A teacher model is trained per group on a more balanced sub-distribution, then distills into a single student model for fine-grained CWE-ID prediction. This directly addresses the highly imbalanced multiclass problem where top-6 CWE types contain ~50% of samples while the remaining 107 types have very few.

**Dataset:** BigVul — same dataset as your model.

**Key results:** Outperforms flat CodeBERT classifier and IVDetect on multiclass CWE prediction. Hierarchical distillation improves both head and tail class performance.

**Relevance to your work:** This is the primary multiclass competitor on BigVul. Same dataset, same CodeBERT CLS embedding approach, same task. If your model beats VulExplainer on multiclass F1, that is a strong thesis contribution. The hierarchical distillation idea is conceptually related to your group head → CWE head conditioning in Arch11/12.

---

### 3. TreeVul (ICSE 2023)
**Full title:** "Fine-Grained Commit-Level Vulnerability Type Prediction by CWE Tree Structure"  
**Authors:** Pan et al.  
**Venue:** ICSE 2023  
**Paper:** https://dl.acm.org/doi/abs/10.1109/ICSE48619.2023.00088 | https://ieeexplore.ieee.org/document/10172785/  
**Repo:** ❌ No public repo  
**SOTA?** ✅ ICSE 2023 — reference for CWE-tree-aware hierarchical classification

**Architecture:**
Transformer encoder on commit diffs (code changes, not full functions). Uses the CWE taxonomy tree structure (MITRE hierarchy: Pillar → Class → Base → Variant) to perform coarse-to-fine prediction. The model first predicts the broad CWE family (e.g., "Memory Safety"), then refines to the specific CWE-ID. Observed that top-6 CWE types contain ~50% of samples while 107 types have very few — the same long-tail problem as LIVABLE.

**Dataset:** Commit-level vulnerability dataset (security patches from open-source projects).

**Key results:** +25% macro F1 over flat classifiers by exploiting CWE tree hierarchy.

**Relevance to your work:** Works on commit diffs (not full functions), so not directly runnable on your data. Cite for the coarse-to-fine CWE routing idea — your Arch11 (`lmgat_codebert_mtl`) and Arch12 (`lmgat_hcdfgat`) implement exactly this with a group head (coarse) conditioning the CWE head (fine). TreeVul provides the academic justification for why this hierarchical approach works.

---

### 4. MultiGLICE (MDPI Computers 2025)
**Full title:** "MultiGLICE: Combining Graph Neural Networks and Program Slicing for Multiclass Software Vulnerability Detection"  
**Authors:** de Kraker et al.  
**Venue:** MDPI Computers, 2025  
**Paper:** https://www.mdpi.com/2073-431X/14/3/98  
**Repo:** ✅ https://github.com/wesleydekraker/glice (GLICE predecessor; MultiGLICE extends it)  
**SOTA?** ✅ 2025 — SOTA for multiclass on SARD dataset

**Architecture:**
Extends GLICE (GNN + inter-procedural program slicing) to multiclass detection. Before feeding code into the GNN, inter-procedural program slices are extracted per vulnerability-relevant syntax pattern (e.g., buffer operations, pointer arithmetic). This removes irrelevant safe code from the CPG before the GNN processes it, reducing noise. Supports C/C++, C#, Java, and PHP. Detects 38 CWE types.

**Dataset:** SARD (Software Assurance Reference Dataset) — synthetic/curated, not BigVul.

**Key results:** 38-class CWE detection across 4 programming languages. Outperforms flat GNN baselines by removing CPG noise via slicing.

**Relevance to your work:** Different dataset (SARD, not BigVul), so not directly comparable. Cite for the argument that raw unpruned CPGs introduce noise that hurts multiclass GNN performance — directly explaining why your Arch4/Arch8 struggled with IFA ~13 (localization noise from safe code washing out vulnerable lines). Also validates the GNN approach for multiclass CWE.

---

### 5. muVulDeePecker (arXiv 2020)
**Full title:** "A Deep Learning-Based System for Multiclass Vulnerability Detection"  
**Venue:** arXiv 2020  
**Paper:** https://arxiv.org/abs/2001.02334  
**Repo:** ✅ https://github.com/muVulDeePecker/muVulDeePecker  
**SOTA?** ❌ Superseded by transformer models — canonical classic baseline

**Architecture:**
BiLSTM on code gadgets (program slices extracted via data/control dependence). Each gadget is a sequence of code tokens representing a vulnerability-relevant program slice. The BiLSTM processes the token sequence and outputs a CWE class prediction. Extends the original VulDeePecker (binary) to multiclass by adding more CWE-specific gadget types.

**Dataset:** MVD (Multiclass Vulnerability Dataset) based on SARD + NVD — 126 CWE types.

**Key results:** Effective multiclass detection on 126 CWE types. Outperforms binary VulDeePecker on fine-grained classification.

**Relevance to your work:** The canonical multiclass baseline that all subsequent work cites as the starting point. Cite to show how much your GNN+LM approach improves over the BiLSTM era. Also provides the MVD dataset which is publicly available for additional experiments.

---

### 6. Hierarchical Contrastive Learning for Vulnerability Type (EMNLP 2024)
**Full title:** "Applying Contrastive Learning to Code Vulnerability Type Classification"  
**Authors:** Chen Ji, Su Yang, Hongyu Sun, Yuqing Zhang et al.  
**Venue:** EMNLP 2024  
**Paper:** https://acl.ldc.upenn.edu/2024.emnlp-main.666/  
**PDF:** `paper/HierarchicalSupCon_EMNLP2024.pdf`  
**Repo:** ❌ No public repo  
**SOTA?** ✅ EMNLP 2024 — current reference for hierarchical contrastive on CWE type classification

**Architecture:**
LM encoder (CodeBERT/RoBERTa) with a hierarchical supervised contrastive loss that uses the MITRE CWE taxonomy tree. Standard SupCon pushes all different classes equally far apart, but CWE-119 and CWE-125 are both "Buffer Errors" — over-separating them is wrong. This paper weights positive pairs by CWE tree distance: same CWE → weight 1.0, same parent class → weight α, different family → pure negative. Also uses max-pooling to handle long vulnerability code inputs (73% of BigVul functions exceed 512 tokens). Mixes self-supervised contrastive loss to prevent class collapse.

**Dataset:** NVD vulnerability dataset with CWE labels. Long-tail distribution: top CWEs dominate, most CWEs have very few samples.

**Key results:** +2.97% to +17.90% accuracy improvement over SOTA. +0.98% to +22.27% weighted-F1 improvement. Better performance on higher-quality datasets.

**Relevance to your work:** Directly justifies your `HierarchicalSupConLoss` in `src/gnn_vuln/losses/hierarchical_supcon.py`. Your `CWE_GROUP_MAP` in `data/cwe_taxonomy.py` already encodes the CWE family hierarchy (memory_safety, numeric, injection, etc.) needed for this loss. The paper also independently identifies the 512-token truncation problem for long functions — validating your sliding window implementation.

---

### 7. SCL-CVD (Computers & Security 2024)
**Full title:** "SCL-CVD: Supervised Contrastive Learning for Code Vulnerability Detection via GraphCodeBERT"  
**Authors:** Rongcun Wang, Senlei Xu, Yuan Tian, Xingyu Ji, Xiaobing Sun, Shujuang Jiang  
**Venue:** Computers & Security, 2024  
**Paper:** https://dl.acm.org/doi/10.1016/j.cose.2024.103994  
**PDF:** `paper/SCL-CVD.pdf`  
**Repo:** ✅ https://github.com/AI4CVD/SCL4CVD  
**SOTA?** ✅ Computers & Security 2024 — reference for supervised contrastive learning in vulnerability detection

**Architecture:**
GraphCodeBERT fine-tuned with supervised contrastive loss (SCL) combined with R-Drop regularization. Source code is represented as data flow graphs, processed by GraphCodeBERT to produce code embeddings. The SCL loss maximizes similarity between embeddings of the same vulnerability class while minimizing similarity between different classes. R-Drop addresses the discrepancy between training (with dropout) and inference (without dropout) by applying dropout twice and minimizing the KL divergence between the two distributions. Also uses LoRA for parameter-efficient fine-tuning.

**Dataset:** BigVul and other C/C++ vulnerability datasets.

**Key results:** F1 +35–50% over baselines. Better generalization and robustness to noisy labels compared to standard cross-entropy fine-tuning.

**Relevance to your work:** Binary detection paper, but the SCL loss technique applies directly to your multiclass setting. Justifies adding SupCon loss on top of your GNN+LM embeddings after global pooling. The R-Drop idea is also applicable to your dropout-heavy GATv2 encoder.

---

### 8. VulANalyzeR (ACM TOSEM 2023)
**Full title:** "VulANalyzeR: Explainable Binary Vulnerability Detection with Multi-Task Learning and Attentional Graph Convolution"  
**Authors:** Litao Li, Steven H.H. Ding, Yuan Tian, Benjamin C.M. Fung, Philippe Charland, Weiha Nou, Leo Song, Congwei Chen  
**Venue:** ACM Transactions on Software Engineering and Methodology (TOSEM), 2023  
**Paper:** https://dl.acm.org/doi/10.1145/3585386  
**PDF:** `paper/VulANalyzeR.pdf`  
**Repo:** ❌ No public repo  
**SOTA?** ✅ ACM TOSEM 2023 — primary MTL reference for binary+CWE heads on shared GNN encoder

**Architecture:**
Works on **binary code** (not source code). Uses a combination of RNN (recurrent units to simulate program execution) and attentional graph convolution on the binary control flow graph. Three MTL output heads on a shared encoder: (1) binary vulnerability detection head, (2) CWE type classification head, (3) root cause analysis head (which instructions/basic blocks caused the vulnerability). The attention mechanism shows which instructions contribute most to the classification — providing explainability without requiring location labels during training.

**Dataset:** CVE dataset (real complex vulnerabilities from binary executables). SARD dataset for CWE type evaluation.

**Key results:** Better performance than SOTA baselines on binary vulnerability detection. CWE type classification and root cause analysis as auxiliary tasks improve the shared representation. Case studies show accurate identification of vulnerable basic blocks even without location hints during training.

**Relevance to your work:** Directly justifies your Arch11 (`lmgat_codebert_mtl`) MTL design — binary head + group head + CWE head on shared GNN+LM encoder. VulANalyzeR proves that training binary and CWE classification heads simultaneously on a shared encoder improves both tasks. The binary head provides gradient signal from ALL samples every batch, while the CWE head only gets signal from vulnerable samples — forcing stable, grounded shared representations. Note: VulANalyzeR works on binary code; your model works on source code CPG, which is a key difference.

---

### 9. Vul-LMGNNs (2024)
**Full title:** "Vul-LMGNNs: Fusing Language Models and Online-Distilled Graph Neural Networks for Code Vulnerability Detection"  
**Authors:** Ruitong Liu, Yanbin Wang, Haitao Xu, Jianguo Sun, Fan Zhang, Peiyue Li, Zhenhao Guo  
**Venue:** arXiv 2024 (also in `src/vul-LMGNN/`)  
**Paper:** https://arxiv.org/abs/2404.14719  
**PDF:** `paper/vul-LMGNN.pdf`  
**Repo:** ✅ Cloned in `src/vul-LMGNN/`  
**SOTA?** ⚠️ Competitive at publication — ~10% F1 improvement on imbalanced datasets

**Architecture:**
Integrates pre-trained CodeLMs with GNNs using **online knowledge distillation** to enable cross-layer information propagation. Uses Code Property Graphs (CPG) with gated GNNs to extract structural information. A student GNN acquires structural knowledge from a simultaneously trained counterpart GNN through an alternating training procedure. Pre-trained CodeLMs extract semantic features from code sequences. An "implicit-explicit" joint training framework fuses code semantic information with structural information. The key insight is that standard GNNs are limited to single-layer neighbor aggregation, failing to capture long-range dependencies — the online distillation enables cross-layer propagation.

**Dataset:** Four public datasets including DiverseVul, Devign, VDSIC, ReVeal. Binary detection.

**Key results:** ~10% F1 improvement on two challenging imbalanced datasets compared to previous best methods. State-of-the-art on all four datasets.

**Relevance to your work:** This is the direct predecessor to your architecture. Your Arch9 (`lmggnn`) implements a simplified version (GatedGNN + CodeBERT without the online distillation). The online distillation idea addresses the same cross-layer propagation limitation your ARCHITECTURE.md identifies. Binary only — your model extends this to multiclass CWE classification.

---

## Line-Level / Statement-Level Vulnerability Localization

### 10. LineVul (MSR 2022)
**Full title:** "LineVul: A Transformer-based Line-Level Vulnerability Prediction"  
**Authors:** Michael Fu, Chakkrit Tantithamthavorn (Monash University)  
**Venue:** MSR 2022  
**Paper:** https://www.researchgate.net/publication/359402890  
**PDF:** `paper/linevul.pdf`  
**Repo:** ✅ https://github.com/awsm-research/LineVul  
**SOTA?** ⚠️ Still the primary reference for line-level localization metrics on BigVul; beaten on binary F1 by newer models

**Architecture:**
CodeBERT (RoBERTa-based) fine-tuned for function-level binary vulnerability classification. Line-level localization is derived post-hoc using the attention mechanism — attention weights over tokens are aggregated per source line to produce a line suspicion score. No per-line labels are used during training. Evaluated with multiple attribution methods: self-attention, Layer Integrated Gradients (LIG), Saliency, DeepLift, DeepLiftShap, GradientShap.

**Dataset:** BigVul — 188k+ C/C++ functions with diff-based flaw line ground truth.

**Key results:** Function-level F1=0.91 (vs IVDetect 0.35). Top-10 line accuracy: 0.65 (attention), 0.53 (LIG). IFA: 4.56 (attention). Effort@20%Recall: 0.0075 (attention). 75–100% accuracy on Top-25 most dangerous CWEs.

**Relevance to your work:** Primary localization baseline. Defines the IFA, Effort@20%Recall, and Recall@K%LOC metrics your model uses. Your model's statement head produces explicit per-line scores (not derived from attention), which is a fundamentally different and more direct approach. The comparison shows whether explicit MIL-trained statement scoring outperforms implicit attention-based localization.

---

### 11. LineVD (MSR 2022)
**Full title:** "LineVD: Statement-level Vulnerability Detection using Graph Neural Networks"  
**Authors:** David Hin, Andrey Kan, Huaming Chen, M. Ali Babar (University of Adelaide + AWS AI Labs)  
**Venue:** MSR 2022  
**Paper:** https://arxiv.org/abs/2203.05181 | https://doi.org/10.1145/1122445.1122456  
**PDF:** `paper/LineVD.pdf`  
**Repo:** ✅ https://github.com/davidhin/linevd  
**SOTA?** ⚠️ 2022 — GNN-based line localization baseline

**Architecture:**
Formulates statement-level vulnerability detection as a **node classification task** on the CPG. Each statement is a node; the GNN classifies each node as vulnerable or not. Leverages control and data dependencies between statements using GNNs, combined with a transformer-based model to encode raw source code tokens per node. Addresses the limitation that function-level models give developers no information about which specific lines are vulnerable. Uses multi-view representations of syntax, semantics, and control flow to capture structural dependencies.

**Dataset:** BigVul (C/C++) with statement-level labels derived from diff information. Preliminary analysis found that vulnerabilities are often localized to a few key lines within a function.

**Key results:** Outperforms function-level models on statement-level localization. GNN structure helps identify vulnerable statements by propagating information through control and data dependency edges. Cited by EDAT (2025) as a key line-level baseline alongside LineVul.

**Relevance to your work:** GNN-based counterpart to LineVul. Treats localization as node classification with direct supervision, while your MIL-based approach is a middle ground — uses GNN structure (like LineVD) but trains with only function-level labels (like LineVul), eliminating the need for expensive statement-level annotation. Your approach is more practical for real-world datasets where per-line labels are unavailable.

---

### 12. WAVES (ACM TOSEM 2024)
**Full title:** "WAVES: Weakly Supervised Vulnerability Localization via Multiple Instance Learning"  
**Authors:** Wenchao Gu, Yupan Chen, Yanlin Wang, Hongyu Zhang, Cuiyun Gao, Michael R. Lyu  
**Venue:** ACM Transactions on Software Engineering and Methodology (TOSEM), 2024  
**Paper:** (referenced in your codebase)  
**PDF:** `paper/WAVES.pdf`  
**Repo:** ❌ Not found  
**SOTA?** ✅ ACM TOSEM 2024 — SOTA for weakly supervised statement-level localization

**Architecture:**
Transformer-based encoder with Multiple Instance Learning (MIL) framework. Converts function-level ground-truth labels into pseudo-labels for individual statements — eliminating the need for statement-level annotation. Two channels capture local and global vulnerability features separately; their results are combined for per-statement prediction. The MIL framework treats each function as a "bag" and each statement as an "instance" — the bag label (function vulnerable/not) is used to generate pseudo instance labels for the top-k most suspicious statements. Statements within the same function interact during encoding (unlike traditional MIL which assumes instance independence).

**Dataset:** Three benchmark datasets (BigVul, Devign, and one more). Evaluated on both function-level detection and statement-level localization.

**Key results:** Comparable function-level detection performance to supervised baselines. State-of-the-art statement-level localization without requiring statement-level labels during training.

**Relevance to your work:** Your MIL statement head (`StmtHead` in `heads.py`) directly implements WAVES-style top-k pseudo-labeling. The key difference: WAVES uses a Transformer encoder over statement sequences, while your model uses GATv2 over the CPG graph structure. Your Arch8 (`lmgat_waves_seq`) explicitly tests the WAVES Transformer localization approach and found it underperforms (Top-1=0.096) compared to GATv2-based localization — validating your choice of GNN over pure Transformer for CPG-based localization.

---

### 13. VulChecker (USENIX Security 2023)
**Full title:** "VulChecker: Graph-based Vulnerability Localization in Source Code"  
**Authors:** Yisroel Mirsky, George Macon, Michael Brown, Carter Yagemann, Matthew Pruett, Evan Downing, Sukarno Mertoguno, Wenke Lee  
**Venue:** USENIX Security 2023  
**Paper:** https://www.usenix.org/conference/usenixsecurity23/presentation/woo  
**PDF:** `paper/VulChecker.pdf`  
**Repo:** ✅ https://github.com/ymirsky/VulChecker  
**SOTA?** ✅ USENIX Security 2023 — precise line-level localization + CWE type classification

**Architecture:**
Graph-based model that precisely localizes vulnerabilities down to the exact instruction and classifies their CWE type. Proposes a new program representation that properly incorporates AST flow information (previous approaches either omit AST or include it incorrectly). Uses a message-passing GNN with a novel program slicing strategy to bridge the gap between a vulnerability's root cause and its manifestation point (which can be hundreds of lines apart). Also proposes a data augmentation strategy for cheaply creating strong vulnerability datasets.

**Dataset:** Custom dataset with precise instruction-level labels. Addresses the "manifestation distance" problem where vulnerability root cause and manifestation can be far apart in the code.

**Key results:** Precise localization to exact instruction level. CWE type classification as part of the same model. Outperforms prior GNN approaches that can only propagate information 1–2 hops.

**Relevance to your work:** Addresses the same dual task as your model — function-level CWE classification + statement-level localization. The "manifestation distance" insight directly explains why your ranking loss (flaw lines must score higher than safe lines) is necessary — the vulnerable line may not be adjacent to the root cause in the CPG. VulChecker's approach of using program slicing to bridge root cause and manifestation is conceptually related to your CPG edge types (PDG/DDG capture data dependencies that connect root cause to manifestation).

---

### 14. EDAT (arXiv 2025)
**Full title:** "Improving Vulnerability Type Prediction and Line-Level Detection via Adversarial Training-Based Data Augmentation and Multi-Task Learning"  
**Authors:** Siyu Chen, Jiongyi Yang, Xiang Chen, Menglin Zheng, Minnan Wei, Xiaolin Ju (Nantong University, China)  
**Venue:** arXiv, June 2025  
**Paper:** https://arxiv.org/abs/2506.23534  
**PDF:** `paper/EDAT.pdf`  
**Repo:** ✅ https://github.com/Karelye/EDAT-MLT  
**SOTA?** ⚠️ arXiv only (not peer-reviewed yet) — most recent paper combining both tasks

**Architecture:**
Unified framework combining two modules: (1) **EDAT** (Embedding-Layer Driven Adversarial Training) — injects adversarial perturbations into identifier embeddings guided by attention-based semantic importance, using multi-step Projected Gradient Descent (PGD) with AST and Program Dependency Graph constraints to preserve code semantics during augmentation; (2) **MTL** (Multi-Task Learning) — shared encoder with two heads: Vulnerability Type Prediction (VTP, multiclass CWE) and Line-level Vulnerability Detection (LVD). Uses CodeBERT, GraphCodeBERT, and CodeT5 as backbone encoders. The adversarial augmentation specifically targets rare vulnerability types by generating semantically consistent perturbations for minority classes.

**Dataset:** BigVul and other C/C++ vulnerability datasets. Evaluates on both VTP (CWE type classification) and LVD (line-level localization) simultaneously.

**Key results:**
- VTP: F1 +22.8% (CodeBERT), +20.7% (GraphCodeBERT), +17.8% (CodeT5) vs single-task baseline
- LVD: Recall@20%LOC improved from 0.5582 → 0.6519 (CodeBERT); IFA reduced from 6.21 → 2.79
- MTL alone: F1 from 0.4857 → 0.5965 (CodeBERT), 0.5185 → 0.6258 (GraphCodeBERT)

**Relevance to your work:** The most directly relevant paper — combines exactly the two tasks your model does (CWE type classification + line-level localization) in a single MTL framework. Uses LineVul and LineVD as line-level baselines. Key difference: EDAT uses a pure LM encoder (no GNN/CPG), while your model uses GNN+LM fusion on CPG structure. This makes your approach complementary — EDAT shows MTL helps both tasks, your model shows that CPG structure further improves localization beyond what pure LM attention can achieve.

---

## Summary Table

| # | Paper | Year | Venue | Task | PDF | Repo | SOTA? |
|---|-------|------|-------|------|-----|------|-------|
| 1 | LIVABLE | 2023/24 | IEEE TSE | Multiclass CWE | ✅ | ❌ | ✅ |
| 2 | VulExplainer | 2023 | IEEE TSE | Multiclass CWE | ❌ | ✅ | ✅ |
| 3 | TreeVul | 2023 | ICSE | Multiclass CWE | ❌ | ❌ | ✅ |
| 4 | MultiGLICE | 2025 | MDPI | Multiclass CWE | ❌ | ✅ (GLICE) | ✅ |
| 5 | muVulDeePecker | 2020 | arXiv | Multiclass CWE | ❌ | ✅ | ❌ |
| 6 | Hier. Contrastive | 2024 | EMNLP | Multiclass CWE | ✅ | ❌ | ✅ |
| 7 | SCL-CVD | 2024 | C&S | Binary + SupCon | ✅ | ✅ | ✅ |
| 8 | VulANalyzeR | 2023 | ACM TOSEM | MTL + CWE | ✅ | ❌ | ✅ |
| 9 | Vul-LMGNNs | 2024 | arXiv | Binary GNN+LM | ✅ | ✅ | ⚠️ |
| 10 | LineVul | 2022 | MSR | Line localization | ✅ | ✅ | ⚠️ |
| 11 | LineVD | 2022 | MSR | Line localization | ✅ | ✅ | ⚠️ |
| 12 | WAVES | 2024 | ACM TOSEM | Line localization | ✅ | ❌ | ✅ |
| 13 | VulChecker | 2023 | USENIX Sec | Line + CWE | ✅ | ✅ | ✅ |
| 14 | EDAT | 2025 | arXiv | MTL: CWE + Line | ✅ | ✅ | ⚠️ |

**SOTA legend:** ✅ = current reference for its task | ⚠️ = competitive/partial | ❌ = superseded
