# Contrastive Loss Research — Comparison & Improvement Plan

Research notes comparing the user's `HierarchicalSupConLoss` implementation against two inspiration papers (Ji et al. EMNLP 2024, Wang et al. SCL-CVD 2024) and the original Khosla SupCon 2020.

---

## 1. The Three Approaches at a Glance

### 1.1 Khosla et al. 2020 — Original SupCon

[Paper](https://arxiv.org/abs/2004.11362)

The foundation. Binary positive/negative decision per pair:
```
L_i = -1/|P(i)| · Σ_{p ∈ P(i)} log[ exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]

P(i) = {j : y_j == y_i, j ≠ i}   ← binary: same class = positive, else negative
```
- All positives weighted equally (1/|P(i)|)
- No hierarchy, no distance, no weighting
- Known issue: causes **class collapse** (Islam et al. 2021)

### 1.2 Ji et al. EMNLP 2024 — Hierarchical Contrastive Learning for CWE Classification

[Paper](https://aclanthology.org/2024.emnlp-main.666/) · [Repo](https://github.com/ChenJi98/HierarchicalSupCon)

**Task:** Multiclass CWE classification (88 types) on Big-Vul and PrimeVul.

**Backbone:** CodeBERT / GraphCodeBERT / CodeGPT (LM only, no GNN).

**Key ideas:**
1. **Label expansion** — each CWE expanded to 5-level MITRE hierarchy tuple (Pillar → Class → Base → Variant → Specific). Example CWE-119: `{664, 118, 119, 119, 119}`.
2. **Sequential training** — 5 phases × 300 epochs each. Phase k uses level-k labels for SupCon. Weights carry over between phases (curriculum learning: coarse → fine).
3. **Geometric spread** — adds `µ · L_self` (SimCLR-style self-supervised loss) to prevent class collapse.
4. **Max-pooling for long input** — split source into 512-token chunks, run LM on each, max-pool the `[CLS]` vectors.

**Loss formula (each phase):**
```
L = (1 - λ - µ) · L_CE + λ · L_sup + µ · L_self

λ = 0.3, µ = 0.2, τ = 0.5
```
- `L_sup` = unmodified Khosla SupCon (positives = same label at current phase's level)
- `L_self` = SimCLR (each sample's augmented view = positive, all others = negative)
- `L_CE` = standard cross-entropy for classification

**Results (Big-Vul):**
| Method | Accuracy | Weighted-F1 |
|--------|----------|-------------|
| CodeBERT (CE only) | 63.19% | 43.07% |
| LIVABLE | 64.01% | 64.36% |
| Ji (CodeBERT) | **69.06%** | **65.34%** |

**Ablation gains:** hierarchical CL +3.7-4.07%, self-supervised loss +1.71% (Big-Vul) / +4.3% (PrimeVul), max-pooling +0.47%.

### 1.3 Wang et al. 2024 — SCL-CVD

[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167404824002475) · [Repo](https://github.com/AI4CVD/SCL4CVD)

**Task:** Binary vulnerability detection (vulnerable vs benign) on Devign, Reveal, Big-Vul, Combined.

**Backbone:** GraphCodeBERT fine-tuned with LoRA (rank-decomposition adapters, base weights frozen).

**Key ideas:**
1. **Plain Khosla SupCon** — no modification. Positives = same binary label.
2. **R-Drop** — each input passes through model twice with different dropout masks; KL divergence between outputs is penalized. Prevents train/inference gap and indirectly prevents class collapse.
3. **LoRA** — parameter-efficient fine-tuning. Only ~0.5% of params trained. 16-93% faster.

**Loss formula:**
```
L_Con = λ · L_SCL + (1-λ) · L_NLL          # SCL = Khosla SupCon, NLL = CE
L_KL  = ½[KL(P¹‖P²) + KL(P²‖P¹)]          # R-Drop
L     = α · L_KL + L_Con

λ = 0.3, τ = 0.7, α = 3, dropout = 0.03
```

**Results (avg across 4 datasets vs GraphCodeBERT):**
- Accuracy +0.88%, Recall +41.51%, F1 +19.28%
- Training time -16.67% to -93.03% (LoRA)

### 1.4 Our Implementation — `HierarchicalSupConLoss`

**Task:** Multiclass CWE classification (up to 26 classes) + binary + group (MTL) + line-level localization.

**Backbone:** LM (UniXcoder/CodeBERT) + GNN (GATv2 with 7 typed edges over CPG).

**Key ideas:**
1. **Continuous tree-distance weighting** — pre-computed CWE distance matrix gives real-valued weight `w_ij ∈ [0, 1]` per pair. Generalizes Khosla's binary positive/negative.
2. **Intra-group restriction** — cross-group pairs zeroed even if both CWEs are in matrix (groups sit at different tree depths, making cross-group distances non-comparable).
3. **Vulnerable-only anchors** — benign samples excluded from anchor set, only serve as negatives.
4. **Configurable weight function** — `linear: 1-d`, `exp: exp(-k·d)`, `power: (1-d)^p`.

**Loss formula:**
```
L_i = -1/Σw_ij · Σ_j w_ij · log[ exp(z_i·z_j / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]

w_ij = {
    1.0                     if same CWE
    weight_fn(d_ij)         if same group, both in matrix
    α (default 0.5)         if same group, fallback
    0.0                     if different group / benign / self
}
```

**Integration with model:**
```
total = CE(func) + group_w·CE(group) + binary_w·CE(binary)
      + mil_w·MIL(stmt) + rank_w·RankLoss(stmt)
      + supcon_w·HierarchicalSupCon(z, labels, group_ids, cwe_vocab_ids)
```

---

## 2. Side-by-Side Comparison Table

| Property | Khosla 2020 | Ji 2024 | Wang 2024 (SCL-CVD) | Ours |
|----------|-------------|---------|---------------------|------|
| Task | General | Multiclass CWE | Binary vuln | Multiclass + binary + group MTL |
| Modifies SupCon formula? | n/a (original) | No | No | **Yes** (continuous weights) |
| Hierarchy used? | No | Yes (5-level label expansion) | No | Yes (continuous distance matrix) |
| How hierarchy enters | n/a | Sequential training phases | n/a | Per-pair weight in single phase |
| Training schedule | Single | Sequential 5×300 epochs | Single | Single |
| Class-collapse safeguard | None | `µ·L_self` (SimCLR) | R-Drop | **None** |
| Long-input handling | n/a | Chunk + max-pool | Truncation | Sliding window (separate module) |
| Backbone | General | LM only | GraphCodeBERT + LoRA | LM + GNN |
| Anchors include benign? | Yes (all classes) | Yes | Yes | **No** (vulnerable only) |
| Parameter-efficient FT | n/a | No | Yes (LoRA) | No |
| Best τ reported | 0.07 (original) | 0.5 | 0.7 | Default 0.07 (not tuned) |
| Best λ (SupCon weight) | n/a | 0.3 | 0.3 | Configured per run |
| Joint training? | n/a | Joint within phase, sequential across levels | Joint | Fully joint (all levels + all heads) |

---

## 3. What's Different (Our Novelty)

### 3.1 Continuous per-pair weighting (unique to us)

Neither paper modifies Khosla's loss formula. We replace the binary positive/negative decision with a real-valued weight from the CWE tree distance matrix. This is a **generalization** of Khosla's SupCon:
- Setting all same-class weights to 1.0 and others to 0.0 recovers Khosla exactly
- Setting weights to binary per-level recovers Ji's per-phase behavior (conceptually)
- Our continuous version captures finer distinctions: CWE-119 vs CWE-125 (both buffer, close in tree) gets higher weight than CWE-119 vs CWE-352 (CSRF, far in tree)

**Thesis positioning:** "We extend Khosla's SupCon by introducing continuous per-pair weighting derived from the CWE refinement hierarchy, unifying the binary same-label selection of Wang 2024 and the discrete level-by-level scheduling of Ji 2024 into a single differentiable weight function."

### 3.2 Single-phase joint training (practical advantage)

Ji requires 5×300 = 1500 epochs. We train all hierarchy levels simultaneously in one phase. This is necessary because our model has MTL heads (binary + group + CWE) + localization (MIL + ranking) that all need joint optimization. Sequential phases would break the MTL training.

### 3.3 Vulnerable-only anchors (defensible deviation)

Both papers include benign/non-vulnerable as anchors. We exclude them. Rationale: "benign" is not a coherent class — it represents diverse non-vulnerable code patterns. Pulling all benign toward one cluster is actively harmful for multiclass CWE classification. Benign samples still serve as negatives (pushed away from vulnerable), which is the useful signal.

### 3.4 Intra-group restriction (novel safeguard)

We zero cross-group matrix weights because CWE groups are anchored at different tree depths, making inter-group distances non-comparable. This was discovered empirically (v2 with full matrix < v1 alpha-only on all metrics). Neither paper addresses this because Ji uses discrete levels (no cross-level comparison) and Wang has no hierarchy.

---

## 4. What's Missing (Must Improve)

### 4.1 ⚠️ Class-Collapse Safeguard — CRITICAL

**Problem:** Both papers explicitly add a mechanism to prevent class collapse. We have none.

**Why it matters:** Same-CWE pairs have weight 1.0 → gradient pushes all same-CWE samples toward identical embeddings. For rare CWEs (50-100 samples), this can cause full collapse → loss decreases but F1 plateaus or drops (the loss-F1 gap symptom).

**Evidence from papers:**
- Ji 2024 ablation: removing `L_self` drops accuracy by 1.71% (Big-Vul) and 4.3% (PrimeVul)
- Islam et al. 2021: pure SupCon transfers worse than self-supervised CL due to class collapse

**Recommended fix (ranked by effort):**

| Option | Effort | Mechanism | Expected gain |
|--------|--------|-----------|---------------|
| A. Cap same-CWE weight to 0.95 | 1 line | Soft margin prevents perfect collapse | Small but safe |
| B. R-Drop on classification head | ~20 lines | KL between two dropout passes | Moderate, proven in SCL-CVD |
| C. Add `µ·L_self` (SimCLR term) | ~30 lines + double forward | Direct anti-collapse pressure | Largest, proven in Ji 2024 |

**Recommendation:** Start with Option A (zero risk). If rare-CWE F1 is still poor after phase 1, add Option B (R-Drop). Save Option C for later — it doubles forward compute.

**Implementation sketch for Option B (R-Drop):**
```python
# In trainer.py _forward():
if cfg.model.rdrop_alpha > 0:
    logits_1 = model(batch)  # first forward (dropout mask 1)
    logits_2 = model(batch)  # second forward (dropout mask 2)
    
    p1 = F.softmax(logits_1, dim=-1)
    p2 = F.softmax(logits_2, dim=-1)
    kl_loss = 0.5 * (F.kl_div(p1.log(), p2, reduction='batchmean')
                   + F.kl_div(p2.log(), p1, reduction='batchmean'))
    
    total_loss += cfg.model.rdrop_alpha * kl_loss
```

### 4.2 ⚠️ SupCon Warm-up — IMPORTANT

**Problem:** From epoch 0, the model receives gradient pressure from the distance matrix on random-init embeddings. The matrix-distance signal is noise when embeddings are random.

**Why it matters:** Ji avoids this by training coarse levels first (easy, balanced) before fine levels. Our joint approach doesn't have this protection. Early-epoch SupCon gradients on rare CWEs are essentially random noise that can push embeddings in wrong directions.

**Recommended fix:**
```python
# In trainer.py, per-epoch:
if cfg.model.supcon_warmup_epochs > 0:
    supcon_w = cfg.model.supcon_weight * min(1.0, epoch / cfg.model.supcon_warmup_epochs)
else:
    supcon_w = cfg.model.supcon_weight
```

**Config addition:**
```yaml
model:
  supcon_warmup_epochs: 5  # ramp from 0 to full weight over 5 epochs
```

This gives CE 5 epochs to establish a sane embedding geometry before SupCon starts pulling.

### 4.3 ⚠️ Temperature Too Low — TUNE

**Problem:** Our default `τ = 0.07`. Both papers report optimal `τ = 0.5–0.7`.

**Why it matters:** Lower τ = sharper similarity distribution. Pairs that are slightly more similar get *much* more gradient; slightly-less-similar pairs get almost nothing. At τ=0.07, only the nearest neighbors in embedding space contribute meaningful gradient. This makes training unstable and sensitive to batch composition.

At τ=0.5–0.7, the distribution is smoother — more pairs contribute gradient, training is more stable, and the model can learn from moderate-similarity pairs (which is exactly what the distance matrix provides).

**Recommended fix:** Change default from 0.07 to 0.5. Test 0.3, 0.5, 0.7 in ablation.

```python
# In hierarchical_supcon.py __init__:
temperature: float = 0.5  # was 0.07
```

---

## 5. What to Remove

### 5.1 Don't stack SupCon with focal + LIVABLE + class weights

Per §2-2.5 of `LOSS_F1_GAP.md`, stacking multiple rebalancing mechanisms causes gradient instability on minority classes. SupCon adds a fourth competing signal. If SupCon is enabled:

**Remove or disable:**
- `focal_loss_gamma` → set to 0.0 (SupCon already handles hard samples via the contrastive denominator)
- `livable_loss` → set to false (SupCon's per-pair weighting already handles class imbalance via the distance matrix)
- Keep `use_class_weights: true` for CE only — this is the mildest rebalancing and doesn't conflict

**Rationale from papers:**
- Ji 2024 uses plain CE (no focal, no class weights) alongside SupCon → works fine
- Wang 2024 uses plain NLL (CE) alongside SupCon → works fine
- Neither paper stacks focal or adaptive weights with SupCon

**Recommended config when SupCon is active:**
```yaml
model:
  use_supcon: true
  supcon_weight: 0.3          # match Ji/Wang's λ
  
train:
  focal_loss_gamma: 0.0       # remove — conflicts with SupCon
  livable_loss: false          # remove — conflicts with SupCon
  use_class_weights: true      # keep — mild, doesn't conflict
```

---

## 6. What to Add

### 6.1 R-Drop (from SCL-CVD) — Priority: HIGH

**What:** Bidirectional KL divergence between two forward passes with different dropout masks.

**Why:** Prevents class collapse + improves fine-tuning stability. Proven effective in SCL-CVD. Simpler than Ji's `L_self` (no data augmentation needed, just two forward passes).

**Config:**
```yaml
model:
  rdrop_alpha: 1.0  # 0 = disabled, 1-3 = recommended range from SCL-CVD
```

**Cost:** ~1.5× forward compute (second pass is cheaper — no gradient needed for KL target).

### 6.2 SupCon warm-up (inspired by Ji's curriculum) — Priority: HIGH

**What:** Ramp `supcon_weight` from 0 to target over first N epochs.

**Why:** Approximates Ji's "coarse first" curriculum without the 5-phase complexity. Lets CE establish sane embeddings before SupCon starts pulling.

**Config:**
```yaml
model:
  supcon_warmup_epochs: 5  # 0 = disabled (current behavior)
```

**Cost:** Zero compute overhead. Just a multiplier change per epoch.

### 6.3 Higher temperature default — Priority: MEDIUM

**What:** Change default τ from 0.07 to 0.5.

**Why:** Both papers converge on τ=0.5–0.7 as optimal. Current 0.07 is too sharp for the continuous-weight formulation.

**Cost:** Zero. Config change only.

### 6.4 Same-CWE weight cap — Priority: LOW (try first if collapse suspected)

**What:** Change `weights[same_cwe] = 1.0` to `weights[same_cwe] = 0.95`.

**Why:** Prevents perfect collapse of rare CWEs. Cheapest possible anti-collapse mechanism.

**Cost:** Zero. One-line change.

---

## 7. Recommended Ablation Order

After phase 1 results are in, test SupCon improvements in this order:

| Step | Change | What it tests | Config |
|------|--------|---------------|--------|
| S1 | τ = 0.5 (was 0.07) | Temperature sensitivity | `temperature: 0.5` |
| S2 | supcon_warmup = 5 | Cold-start protection | `supcon_warmup_epochs: 5` |
| S3 | Drop focal when SupCon active | Loss stack simplification | `focal_loss_gamma: 0.0` |
| S4 | R-Drop α = 1.0 | Class-collapse prevention | `rdrop_alpha: 1.0` |
| S5 | Same-CWE cap = 0.95 | Direct anti-collapse | Code change |

Each step is independent. Run S1-S3 together as a single "cleaned-up SupCon" config, then add S4 and S5 individually to measure marginal gain.

**Expected outcome:** S1 + S2 + S3 together should close most of the loss-F1 gap when SupCon is active. S4 adds robustness on rare CWEs specifically.

---

## 8. Summary — Positioning for Thesis

### What we contribute (novel):
1. **Continuous per-pair weighting** from CWE tree distance — generalizes Khosla's binary SupCon
2. **Intra-group restriction** — prevents noisy cross-group distances from diluting signal
3. **Single-phase joint training** with MTL heads — practical for complex multi-objective models
4. **Vulnerable-only anchors** — correct for multiclass CWE where "benign" is not a coherent class

### What we adopt from papers:
- SupCon framework (Khosla 2020)
- Hierarchical CWE structure as prior knowledge (Ji 2024)
- Joint CE + SupCon training (both papers)

### What we should add (improvements):
- Class-collapse safeguard: R-Drop (from SCL-CVD) or `L_self` (from Ji)
- SupCon warm-up: approximation of Ji's curriculum
- Temperature tuning: τ=0.5 (both papers' optimal range)

### What we should remove when SupCon is active:
- Focal loss (conflicts — both papers use plain CE with SupCon)
- LIVABLE adaptive weights (conflicts — SupCon already handles imbalance via weighting)

### Thesis defense statement:
> "We extend supervised contrastive learning (Khosla 2020) for multiclass CWE vulnerability classification by introducing continuous per-pair weighting derived from the CWE refinement hierarchy. This unifies the binary same-label selection of SCL-CVD (Wang 2024) and the discrete level-by-level scheduling of Ji 2024 into a single differentiable weight function w_ij = f(tree_distance(i, j)), enabling joint optimization with multi-task classification and line-level localization heads in a single training phase."

---

## 9. L_self vs R-Drop — Which is Better for Class-Collapse Prevention?

Both mechanisms use "two forward passes" but compute fundamentally different losses. This section compares them based on literature evidence and practical considerations for our architecture.

### 9.1 Mechanism Comparison

| | L_self (SimCLR / Geometric Spread) | R-Drop (Regularized Dropout) |
|---|---|---|
| **Source** | Islam et al. ICCV 2021; used by Ji 2024 | Liang et al. NeurIPS 2021; used by Wang/SCL-CVD 2024 |
| **Input** | Same sample, two forward passes (different dropout) | Same sample, two forward passes (different dropout) |
| **Loss space** | Embedding space (z) | Output space (logits/softmax) |
| **Formula** | `L = -log[exp(z₁·z₂/τ) / Σ exp(z₁·z_k/τ)]` — SimCLR on the two views | `L = ½[KL(P¹‖P²) + KL(P²‖P¹)]` — bidirectional KL on softmax |
| **What it compares** | The two views against ALL other samples in batch (including same-class) | Only the two views of the SAME sample against each other |
| **Negatives needed?** | Yes — all other batch samples are negatives | No — self-comparison only |
| **Anti-collapse mechanism** | **Direct** — same-class samples are negatives → pushed apart | **Indirect** — regularizes weight space → smoother manifold, less prone to collapse |
| **Competing with SupCon?** | **Yes** — L_self pushes same-class apart, L_sup pulls them together | **No** — R-Drop is orthogonal to SupCon (doesn't affect inter-sample distances) |
| **Batch-size sensitive?** | Yes — needs enough negatives for meaningful contrastive signal | No — works per-sample |
| **Compute cost** | 2× forward + contrastive computation over batch | 2× forward + KL computation (cheaper) |

### 9.2 Literature Evidence

#### Islam et al. ICCV 2021 — "A Broad Study on the Transferability of Visual Representations with Contrastive Learning"

Key finding: combining supervised contrastive loss with self-supervised contrastive loss (what they call "geometric spread") consistently improves transfer learning performance. Their SupCon+SelfSupCon combination outperforms either alone on downstream tasks. The self-supervised term prevents class collapse by maintaining intra-class diversity, which is critical for transferability.

#### Chen et al. ICML 2022 — "Improving Transfer and Robustness of Supervised Contrastive Learning"

This paper (Stanford/Hazy Research) directly studies class collapse in SupCon and proposes that "spread alone is insufficient" — you also need to break permutation invariance within classes. They show that simply adding a self-supervised term creates spread but doesn't guarantee meaningful spread. Their solution is more complex (balanced batches + specific augmentation strategies), but the key insight is: **L_self helps but isn't a complete solution for class collapse.**

#### Liang et al. NeurIPS 2021 — "R-Drop: Regularized Dropout for Neural Networks"

R-Drop is validated on 5 tasks (18 datasets): NMT, summarization, language understanding, language modeling, image classification. It's "universally effective" — consistent gains across all tasks. However, R-Drop was NOT designed for or tested against class collapse specifically. Its benefit is training stability and generalization, with anti-collapse being a side effect.

#### MCL-VD (Springer 2025) — "Multi-modal contrastive learning with LoRA-enhanced GraphCodeBERT"

A very recent paper (2025) that extends SCL-CVD's approach with multi-modal contrastive learning for vulnerability detection. Achieves F1 improvements of 4.86-17.26% over baselines. Uses R-Drop + LoRA + multi-modal contrastive. Confirms R-Drop's effectiveness in the vulnerability detection domain specifically.

#### arXiv 2503.08203 (March 2025) — "A Theoretical Framework for Preventing Class Collapse in Supervised Contrastive Learning"

The most recent theoretical work on this topic. Provides guidelines for hyperparameter selection to mitigate class collapse risk. Key finding: the ratio between supervised and self-supervised loss weights is critical — too much supervised → collapse, too much self-supervised → poor class separation. They recommend careful tuning of this balance.

### 9.3 Which is Better for Our Case?

**Short answer: R-Drop is the safer choice. L_self is the stronger choice.**

| Criterion | Winner | Why |
|-----------|--------|-----|
| Anti-collapse strength | **L_self** | Directly pushes same-class apart; R-Drop only indirectly prevents collapse |
| Ease of implementation | **R-Drop** | No contrastive computation, no batch-size dependency, just KL on outputs |
| Risk of conflicting with SupCon | **R-Drop** | L_self actively fights SupCon (push apart vs pull together); R-Drop is orthogonal |
| Hyperparameter sensitivity | **R-Drop** | L_self needs careful µ tuning (too high → destroys clusters); R-Drop α is more forgiving |
| Proven on code/vuln tasks | **R-Drop** | SCL-CVD 2024 + MCL-VD 2025 both validate on vulnerability detection |
| Proven for transfer/generalization | **L_self** | Islam 2021 + Chen 2022 show clear transfer gains |
| Compute overhead | **R-Drop** | Both need 2× forward, but R-Drop's KL is cheaper than L_self's contrastive over batch |
| Works with small batches? | **R-Drop** | L_self needs enough negatives; R-Drop works per-sample |

### 9.4 Recommendation for Our Architecture

Given our constraints:
- Complex multi-objective loss (CE + group + binary + MIL + ranking + SupCon)
- Already have competing gradient signals
- Batch size limited by GPU memory (GNN + LM is memory-heavy)
- Need stability over maximum theoretical gain

**Primary recommendation: R-Drop (α = 1.0–3.0)**

Reasons:
1. **Orthogonal to SupCon** — doesn't create a 7th competing gradient signal. L_self would be the 7th loss term fighting against SupCon.
2. **Proven in vulnerability detection** — SCL-CVD and MCL-VD both validate it on the same domain.
3. **Batch-size independent** — our batches are small (16-32) due to GNN memory. L_self needs large batches for good negatives.
4. **Simpler tuning** — one hyperparameter (α) vs two (µ + temperature for L_self).

**Fallback if R-Drop insufficient: Add L_self with µ = 0.05–0.1**

If rare-CWE F1 is still poor after R-Drop, add a small L_self term. Keep µ low (0.05-0.1, not Ji's 0.2) because:
- Ji has only 3 loss terms; you have 6-7. The "budget" for competing signals is smaller.
- Ji trains 300 epochs per phase; you train ~50-100 total. Less time for the balance to settle.

### 9.5 Implementation Priority

```
Phase 1 (current): No changes — wait for ablation results
Phase 2 (if collapse detected):
  Step 1: R-Drop α=1.0 → test
  Step 2: If insufficient, R-Drop α=3.0 → test  
  Step 3: If still insufficient, add L_self µ=0.05 → test
  Step 4: If still insufficient, L_self µ=0.1 + R-Drop α=1.0 → test (both together)
```

### 9.6 How to Detect if Class Collapse is Happening

Before implementing any fix, verify the problem exists:

1. **Per-class embedding variance.** After training, compute variance of embeddings within each CWE class. If rare CWEs have variance ≈ 0 while common CWEs have variance > 0, collapse is happening.

2. **t-SNE / UMAP visualization.** Plot embeddings colored by CWE. Collapsed classes appear as single dots; healthy classes appear as clusters with spread.

3. **Per-class F1 correlation with class size.** If F1 drops sharply for classes with < 100 samples, collapse is a likely cause.

4. **Cosine similarity within class.** Compute mean pairwise cosine similarity within each CWE. If rare CWEs have cosine ≈ 1.0 (all identical), that's collapse.

---

## 10. References

| Paper | Year | Key contribution relevant to us |
|-------|------|--------------------------------|
| [Khosla et al. — Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | 2020 | Original SupCon formula |
| [Ji et al. — Hierarchical CL for CWE Classification (EMNLP)](https://aclanthology.org/2024.emnlp-main.666/) | 2024 | Sequential hierarchy + L_self for class collapse |
| [Wang et al. — SCL-CVD (Computers & Security)](https://www.sciencedirect.com/science/article/abs/pii/S0167404824002475) | 2024 | R-Drop + LoRA + plain SupCon for binary vuln |
| [Islam et al. — Broad Study on Transferability with CL (ICCV)](https://openaccess.thecvf.com/content/ICCV2021/html/Islam_A_Broad_Study_on_the_Transferability_of_Visual_Representations_With_ICCV_2021_paper.html) | 2021 | Identified class collapse; geometric spread via SupCon+SelfSupCon |
| [Chen et al. — Improving Transfer and Robustness of SupCon (ICML)](https://proceedings.mlr.press/v162/chen22d) | 2022 | Spread alone insufficient; need to break permutation invariance |
| [Liang et al. — R-Drop (NeurIPS)](https://arxiv.org/abs/2106.14448) | 2021 | Regularized dropout via KL divergence; universally effective on 18 datasets |
| [Hu et al. — LoRA](https://arxiv.org/abs/2106.09685) | 2022 | Low-rank adaptation for efficient fine-tuning |
| [MCL-VD — Multi-modal CL with LoRA for Vuln Detection (Springer)](https://link.springer.com/article/10.1007/s10515-025-00543-3) | 2025 | Extends SCL-CVD with multi-modal CL; F1 +4.86-17.26% |
| [arXiv 2503.08203 — Theoretical Framework for Preventing Class Collapse](https://arxiv.org/abs/2503.08203) | 2025 | Guidelines for SupCon+SelfSupCon weight ratio to prevent collapse |
| [Adapting SupCon to Binary Imbalanced Datasets (arXiv 2503.17024)](https://arxiv.org/abs/2503.17024) | 2025 | SupCon performance degrades with class imbalance; proposes Supervised Minority + Prototypes |
