# Multiclass CWE Classification — Improvement Techniques

Techniques for improving multiclass vulnerability/CWE classification on GNN+LM architectures.
Categorized by concern: data preparation → pipeline architecture → loss functions → dataset handling.

---

## Category 1: Data Preparation & Graph Engineering

### MultiGLICE — Inter-procedural Program Slicing
- **Paper:** "MultiGLICE: Combining Graph Neural Networks and Program Slicing for Multiclass Software Vulnerability Detection" (MDPI Computers 2025, de Kraker et al.)
- **Idea:** Extract inter-procedural program slices per vulnerability-relevant syntax pattern before training. Removes irrelevant safe code from CPG before GNN sees it. 38-class CWE detection on C/C++, C#, Java, PHP (SARD dataset).
- **Why:** Feeding entire function CPG into GNN introduces massive noise from safe code. Slicing prunes dead variables before training → cleaner signal, lower IFA.
- **Relevance:** Arch4/Arch8 struggled with localization (IFA ~13) partly because standard pooling washes out vulnerable lines with safe code. Validates the argument that raw unpruned CPGs are noisy for multiclass prediction.
- **Effort:** Very High — inter-procedural call edge extraction requires Joern pipeline changes; current CPG is per-function
- **Stackable:** No (preprocessing change, not loss/arch)

### TreeVul — AST-Respecting Hierarchical Aggregation
- **Paper:** "Fine-grained Commit-level Vulnerability Type Prediction by CWE Tree Structure" (ICSE 2023, Pan et al.)
- **Idea:** Bottom-up aggregation that specifically respects parent-child relationships of the AST, rather than treating all graph edges equally. Code is hierarchical; standard GNNs ignore that. +25% macro F1 over flat classifiers.
- **Why:** Standard GNNs treat CFG, AST, DDG edges with same weight → AST structure lost. TreeVul's hierarchical aggregation preserves syntactic hierarchy.
- **Relevance:** Validates GATv2 architecture choice — GATv2 dynamically weights different edge types (CFG vs AST vs DDG), which is structurally analogous to TreeVul's philosophy of respecting code syntax. Early GCN models failed partly because they ignored edge types.
- **Effort:** High — requires changing message-passing aggregation order; not a drop-in
- **Stackable:** No (architecture-level change)

---

## Category 2: Pipeline Architecture & Orchestration

### VulANalyzeR — Multi-Task Learning (Binary + Multiclass)
- **Paper:** VulANalyzeR (Multi-Task Learning for vulnerability analysis)
- **Idea:** Train binary head (safe/vulnerable) and multiclass CWE head simultaneously on shared encoder. Binary head gives gradient signal from ALL samples every batch; multiclass head only gets signal from vulnerable samples. Forces stable, grounded shared representations.
- **Loss:** `loss = cwe_loss + binary_loss_weight * binary_loss`
- **Relevance:** Directly justifies Arch7/Arch10 design where Stage 1 binary localization score (s_i) informs Stage 2 multiclass classification.
- **Effort:** Low — one extra `nn.Linear(hidden_dim, 2)` + loss term
- **Stackable:** Yes

### MulVul — Coarse-to-Fine Routing
- **Paper:** MulVul
- **Idea:** Don't force one generic model to memorize all CWEs. Use a triage node to predict the broad vulnerability family first, then route to a specialist head for that family. Coarse → Fine prediction pipeline.
- **Why:** One flat classifier memorizing 50 CWEs collapses on minority classes. Routing separates the "what family?" question from the "which exact CWE?" question.
- **Relevance:** Inspired the Dual-Flow Router in Arch10 — splitting graph into "Focal" path (exact vulnerability) and "Global" path (surrounding context) to route the right signal to the final MLP head.
- **Effort:** Medium — requires family-level routing layer between encoder and CWE head
- **Stackable:** Yes (can combine with MTL binary head)

### Multimodal Fusion (2025)
- **Paper:** Multimodal Fusion for Code Vulnerability (2025)
- **Idea:** Fuse flat text sequence embeddings (Transformer/LM) with structural graph embeddings (GNN) immediately before classification head. Neither modality alone captures full vulnerability semantics.
- **Why:** LM captures token-level semantics; GNN captures structural data/control flow. Union of both is strictly richer than either alone.
- **Relevance:** Direct mathematical justification for Stage 3 Tri-Modal Fusion in Arch10 — concatenating `focal_emb`, `context_emb`, and `lm_emb` (UniXcoder) before the final output head.
- **Effort:** Low-Medium — fusion layer after GNN pool + LM CLS, before classifier
- **Stackable:** Yes

### Combined Hierarchical MTL (Proposed — Novel Contribution)
- Merge VulANalyzeR + MulVul: **binary head + group head (coarse) + CWE head (fine, conditioned on group)**
- Cite VulANalyzeR for MTL justification, MulVul for coarse-to-fine routing
- One model, one forward pass, three outputs
- **Loss:** `loss = cwe_loss + group_loss_weight * group_loss + binary_loss_weight * binary_loss`

---

## Category 3: Loss Functions & Embedding Physics

### LIVABLE — Epoch-Adaptive Class Weights
- **Paper:** "LIVABLE: Exploring Long-Tailed Classification of Software Vulnerability Types" (IEEE 2023/2024, arXiv:2306.06935)
- **Idea:** Replace static `class_weights` tensor with epoch-adaptive schedule. Weights dynamically increase penalty for rare classes as training progresses based on epoch + per-class sample count. +7.7% tail-class F1, +25.4% head-class F1 vs baseline.
- **Why:** Fixed class weights apply same pressure throughout training. Early epochs: model hasn't learned majority classes yet, so upweighting minorities too hard. LIVABLE ramps minority pressure gradually.
- **Relevance:** Current arch uses fixed `compute_class_weight()`. BigVul has logging=1, data_integrity=5 — these get ~0 F1 with static weights.
- **Effort:** Low — replace weight computation in `train.py`
- **Stackable:** Yes

### SCL-CVD — Supervised Contrastive Loss
- **Paper:** "SCL-CVD: Supervised Contrastive Learning for Code Vulnerability Detection via GraphCodeBERT" (Computers & Security 2024)
- **Idea:** SupCon loss at graph-level embedding — same CWE label embeddings attract, different CWE embeddings repel. Standard Cross-Entropy is lazy, allows similar classes to overlap in embedding space. SupCon forces distinct "islands" per CWE class before classification even happens. Binary detection F1 +35-50% vs baselines.
- **Why:** CWE classes that are semantically similar (CWE-119, CWE-125, CWE-787 all memory errors) cluster together in embedding space → classifier confused. SupCon separates them explicitly.
- **Relevance:** Fixes Arch4 F1 ceiling. Creates well-separated class islands in embedding space before the classification head.
- **Effort:** Low — drop-in loss addition after GAT pooling; no structural change
- **Stackable:** Yes (complementary with hierarchical contrastive)

### Hierarchical Contrastive Loss
- **Paper:** "Applying Contrastive Learning to Code Vulnerability Type Classification" (EMNLP 2024, Ji et al.)
- **Idea:** Upgrade to standard SupCon. Uses CWE taxonomy tree (MITRE) to group sibling CWEs (e.g., CWE-119 and CWE-125 both under "Buffer Errors") into broader neighborhoods. Sibling CWEs pulled closer together, different-family CWEs pushed far apart. Prevents over-separation of semantically related classes.
- **Why:** Standard SupCon pushes ALL different classes equally far apart. But CWE-119 and CWE-125 ARE related — over-separating them is wrong. Hierarchical version uses taxonomy structure.
- **Relevance:** `CWE_GROUP_MAP` already encodes this taxonomy (memory_safety, numeric, etc.). Pair construction is free — just use GROUP_VOCAB groups as the "family" level. Ultimate state-of-the-art addition.
- **Effort:** Low — secondary contrastive loss term on GAT graph-level embedding
- **Stackable:** Yes (stack on top of SupCon)

### Embedding Mixup for Rare Classes
- **Paper:** "A Study on Mixup-Inspired Augmentation Methods for Software Vulnerability Detection" (arXiv 2025)
- **Idea:** Interpolate rare-class graph embeddings AFTER GAT pooling → synthetic training signal without new CPG generation. Raw code mixup = syntactically broken → invalid CPG. Embedding mixup = always valid vector. Five variants: Mixup, Manifold Mixup, CutMix, SaliencyMix, FMix.
- **Why:** Classes with 1-13 samples (logging, data_integrity, error_handling in BigVul) can't be learned from real data alone. Synthetic interpolated embeddings fill the gap.
- **Effort:** Low — applied between GAT embedding and classification head during training only
- **Stackable:** Yes
- **Limitation:** Requires ≥2 samples of class in batch. For logging=1, combine with LIVABLE.

---

## Category 4: Pretraining & Self-Supervision

### VulMAE — Graph Masked Autoencoder Pretraining
- **Paper:** "VulMAE: Graph Masked Autoencoders for Vulnerability Detection from Source and Binary Codes" (Springer LNCS 2024, Zamani et al.)
- **Idea:** Pretrain GAT encoder on full unlabeled CPG corpus using masked node reconstruction (GraphMAE style) — no labels needed. Forces encoder to learn code grammar. Then fine-tune for CWE classification. Weighted F1=0.936. Reduces labeled-data requirement for rare classes.
- **Why:** Rare CWE classes have too few labeled samples for encoder to learn useful representations from scratch. Pretraining on unlabeled data gives stronger initialization.
- **Relevance:** Explains WHY UniXcoder is fundamentally necessary — it uses similar masking techniques (MLM pretraining). Citing VulMAE proves understanding of why pretrained LMs compensate for limited labeled data in BigVul. Rare CWEs (logging=1) need pretrained initialization to have any chance of being learned.
- **Effort:** High — separate pretraining loop on full CPG corpus before classification fine-tuning
- **Stackable:** Yes (pretraining phase; fine-tuning phase uses all other techniques)

---

## Summary Table

| Technique | Paper | Category | Effort | Impact | Stackable |
|---|---|---|---|---|---|
| MultiGLICE Slicing | MDPI 2025 | Data Prep | Very High | High | No |
| TreeVul AST Aggregation | ICSE 2023 | Data Prep | High | Medium | No |
| VulANalyzeR MTL Binary | VulANalyzeR | Pipeline | Low | High | Yes |
| MulVul Coarse-to-Fine | MulVul | Pipeline | Medium | High | Yes |
| Multimodal Fusion | 2025 | Pipeline | Low-Med | High | Yes |
| LIVABLE Adaptive Weights | IEEE 2023 | Loss | Low | High | Yes |
| SupCon Loss | SCL-CVD 2024 | Loss | Low | Medium | Yes |
| Hierarchical Contrastive | EMNLP 2024 | Loss | Low | Medium | Yes |
| Embedding Mixup | arXiv 2025 | Loss | Low | Medium | Yes |
| VulMAE Pretraining | LNCS 2024 | Pretrain | High | High | Yes |

## Recommended Implementation Order

1. **LIVABLE adaptive weights** — replace static class_weights in `train.py`
2. **VulANalyzeR MTL binary head** — add binary head to `lmgat.py`
3. **MulVul coarse-to-fine** — add group head + conditioning; combine with binary = full hierarchical MTL
4. **SupCon loss** — stack on graph-level embedding after pooling
5. **Hierarchical contrastive** — stack on top of SupCon using GROUP_VOCAB pairs
6. **Embedding Mixup** — add for rare-class synthetic signal
7. **VulMAE pretraining** — if time allows; highest setup cost but strongest rare-class boost
