# Repository Guidelines

## Project Overview

Research codebase for **GNN-based C/C++ vulnerability detection** (Tugas Akhir / bachelor's thesis). Given a C/C++ function, the system simultaneously:
1. **Classifies** the function as benign or assigns a CWE category (binary or 26-class multiclass with 25 CWEs + benign)
2. **Localizes** suspicious source lines via statement-level scoring (no per-line labels required — MIL-based)

Primary dataset: **MegaVul** (Top-25 CWEs, max 1600/class). Also supports BigVul, Devign, DiverseVul, TitanVul.

---

## Architecture & Data Flow

```
C/C++ Source
    ↓ Joern (CPG: AST + CFG + PDG)
    ↓ JSON export
    ↓ graph_builder_lm.py / data/cpg/
PyG Data object:
  x            [N, 773]   node feats: node_type(1) + frozen LM CLS(768) + dist(3) + danger_api(1)
  edge_index   [2, E]
  edge_attr    [E, 7]     one-hot: AST/CFG/CDG/DDG/PDG/CALL/REACHING_DEF
  node_line    [N]        source line index per node (-1 = special)
  flaw_line_mask [N]      GT flaw lines (zeros if unavailable)
  func_input_ids  [L]     full function tokens (add_func_tokens=true)
  func_token_lines [L]    per-token source line id
  cwe_id, group_id        taxonomy IDs for SupCon / MTL
    ↓ CodeBERTGraphDataset (lazy or inmemory)
    ↓ DataLoader (batched PyG graphs)
    ↓ LMGATCodeBERTVulnDetector (or variant)
        ├─ GATEncoder (GATv2Conv × 4 layers, 256D hidden)
        │     optional: block_style=gnn_plus, norm_type=graph, use_ffn, use_pe (RWSE), balanced_init
        ├─ Graph pooling: meanmax | mean | attention | dualflow
        ├─ Live LM branch (optional): UniXcoder/CodeBERT fine-tuned → CLS [B, 768]
        │     optional: sliding window, per-line, window attn pool
        ├─ concat(GNN_pool, LM_cls) → FuncHead → logit_func [B, C]
        ├─ CrossTask module (optional): cross_attention | self_attention | mmoe
        └─ StmtHead: scatter per node_line → max+mean per line → stmt_scores [S]
    ↓ (logit_func, stmt_scores) [+ z for SupCon] [+ MTL heads]
    ↓ Trainer._forward() → loss composition
Loss = CE/Focal(func) + mil_weight×MIL(stmt) + rank_weight×Ranking(stmt)
     + group_weight×CE(logit_group) + binary_weight×CE(logit_binary)  # MTL
     + supcon_weight×HierarchicalSupCon(z)                            # SupCon
     + EWC penalty                                                     # continual
    ↓ Evaluator → metrics_summary.json + plots + predictions.csv
```

---

## Key Directories

```
src/gnn_vuln/           Main package (PYTHONPATH=src required)
  config.py             DataConfig / ModelConfig / TrainConfig dataclasses; YAML loader
  train.py              TrainingSession: full run orchestration; CLI entry `uv run train`
  evaluate.py           Evaluator: inference → metrics → plots; CLI entry `uv run evaluate`
  metrics.py            LocalizationMetrics: IFA, Top-K, Effort@20%, Recall@K%LOC
  baselines.py          Published numbers (LineVul, WAVES, VulLMGNN) for comparison
  inference.py          Single-function API: load_model() + predict() → structured dict
  utils.py              set_seed, setup_logging, CheckpointManager, save/load_checkpoint

  models/
    registry.py         MODEL_REGISTRY + build_model() factory
    base.py             VulnDetectorBase: _build_lm_branch(), lm_parameters(), has_live_lm()
    encoders.py         GATEncoder, GCNEncoder, RGCNEncoder, GINEncoder, GGNNEncoder
                        _compute_rwse(), apply_balanced_init(), build_gnn_encoder()
    heads.py            StmtHead, MulticlassStmtHead, FuncHead, ThinFuncHead, MTLHeads
    cross_task.py       CrossTaskAttn, SelfAttnCrossTask, MMOECrossTask, build_cross_task()
    lmgat_codebert.py   Primary model: LMGATCodeBERTVulnDetector (Arch 3)
    lmgat_codebert_mtl.py  MTL variant: LMGATCodeBERTMTLVulnDetector (Arch 11)
    [10 other variants] lmgat, lmgcn, lmgin, lmggnn, lmrgcn, lmrgcn_codebert,
                        lmgat_mcs, lmgat_interp, lmgat_seq, lmgat_dualflow,
                        lmgat_waves_seq, lmgat_hcdfgat

  training/
    trainer.py          Trainer: _forward() dispatch, train_epoch(), evaluate(), localise()
    losses.py           focal_loss, epoch_adaptive_class_weights, livable_loss,
                        mil_loss, mil_loss_multiclass, ranking_loss
    optimizer.py        build_optimizer_and_scheduler(): LLRD, warmup, cosine/plateau
    pgd.py              EmbeddingPGD: EDAT FGSM-sign identifier perturbation
    ewc.py              EWCDR: Elastic Weight Consolidation Done Right
    sampler.py          SupConBalancedSampler: class-balanced batch sampling
    unfreezer.py        GradualUnfreezer: ULMFiT progressive LM unfreezing

  evaluation/
    localize.py         LocalizationExtractor: inference → per-line suspicion scores
    plots.py            ResultPlotter: ROC, confusion matrix, PR, recall@LOC, IFA dist.

  losses/
    hierarchical_supcon.py  HierarchicalSupConLoss: CWE-tree distance weighted SupCon

  data/
    dataset_lm.py       CodeBERTGraphDataset (PyG InMemoryDataset); lazy or inmemory
    graph_builder_lm.py Shim → data/cpg/ (backward compat)
    cpg/                CPG parsing + PyG assembly (builder.py, parser.py, features.py)
    node_embedder.py    LMNodeEmbedder: frozen LM CLS per CPG node; NON_LM_FEAT_DIM=75
    cwe_taxonomy.py     CWE_GROUP_MAP, GROUP_VOCAB, OWASP mappings, _expand_cwe_filter()
    joern_runner.py     Joern subprocess wrapper (requires Joern at C:/joern/joern-cli)
    preprocess.py       C/C++ identifier normalization via tree-sitter

configs/
  TEMPLATE.yaml         Full 666-line schema — reference for all config fields
  ablation/             Phase 1–12 + gnn_only experiment configs
  lmgat*/               Production architecture configs
  data/                 Dataset-only config overrides

scripts/               34 utility scripts (training pipelines, data processing, analysis)
paper/                 27 PDFs of referenced papers
src/GNNPlus/           Luo 2025 ICLR clone (GNN+ layers, RWSE PE)
src/EDAT/              EDAT 2025 clone (MMOE, PGD adversarial training)
src/SCLCVD/            SCL-CVD clone (SupConLoss reference)
src/vul-LMGNN/         vul-LMGNN clone (GatedGNN + CodeBERT predecessor)
src/LineVul/           LineVul clone (metric definitions, baseline numbers)
src/VulChecker/        VulChecker clone (ranking loss justification)
```

---

## Development Commands

```bash
# Environment — ALWAYS use uv, never bare python/pip
uv sync                          # install all deps including dev
uv run python <script>           # run any script
uv add <package>                 # add dependency

# Training
uv run train --config configs/ablation/phase8/H4_...yaml
uv run train --config configs/lmgat_codebert_mtl/multiclass_mtl.yaml --resume

# Evaluation
uv run evaluate \
    --checkpoint checkpoints/<run_id>/best_lmgcn.pt \
    --config checkpoints/<run_id>/config.yaml

# Dataset preprocessing (once per config, before training)
uv run python scripts/prepare_dataset.py --config configs/... --device cuda
uv run python scripts/precompute_rwse.py \
    --graphs-dir data/processed/<dataset>_graphs \
    --walk-length 32

# Analysis scripts
uv run python scripts/analyze_node_counts.py --raw-dir data/raw
uv run python scripts/analyze_func_lines.py
uv run python scripts/analyze_dataset.py

# Export for baseline comparison
uv run python scripts/export_linevul.py --config configs/... --out-dir data/baselines/linevul

# Cloud training (Linux GPU server)
bash scripts/setup_cloud.sh          # first-time server setup
bash scripts/train_cloud.sh --init --config configs/... --dataset <name>

# Local training with auto-download + eval (Windows PowerShell)
.\scripts\train_local.ps1 -Config @("configs/...") -Dataset @("<name>")
```

**Windows path note**: Use forward slashes in bash (`C:/joern/joern-cli`), backslashes fine in PowerShell.

---

## Code Conventions & Common Patterns

### Config-driven everything
All hyperparameters live in YAML. Never hardcode values in model code. Config keys map 1:1 to dataclass fields in `config.py`.

```python
# Load + override
cfg = Config.from_yaml("configs/ablation/phase8/H4.yaml")
# Access
cfg.model.hidden_dim       # ModelConfig field
cfg.train.lm_lr            # TrainConfig field
cfg.data.filter_top25_dangerous  # DataConfig field
```

### Model registry pattern
Adding a new architecture = one class + one dict entry. Do not modify other files.

```python
# models/registry.py
MODEL_REGISTRY: dict[str, type] = {
    "lmgat_codebert": LMGATCodeBERTVulnDetector,
    "my_new_arch": MyNewModel,   # ← add here
}
# models/my_new_arch.py
class MyNewModel(VulnDetectorBase):
    @classmethod
    def from_config(cls, cfg, in_channels, **kwargs): ...
    def forward(self, x, edge_index, batch, node_line, edge_attr, ...): ...
        # return (logit, stmt_scores)          # 2-tuple: standard
        # return (logit, stmt_scores, z)       # 3-tuple: + SupCon embeddings
        # return (logit_cwe, logit_group, logit_binary, stmt_scores)  # 4-tuple: MTL
        # return (logit_cwe, logit_group, logit_binary, stmt_scores, z)  # 5-tuple: MTL+SupCon
```

### Forward return tuple contract
`Trainer._forward()` dispatches on tuple length — all variants must return exactly 2, 3, 4, or 5 elements in the correct order. Do not add positional elements without updating `_forward()`.

### Two optimizer regimes
`build_optimizer_and_scheduler()` auto-detects:
- **Live LM** (`model.has_live_lm()`): AdamW with 2 param groups — LM at `lm_lr=2e-5` + linear warmup, GNN at `lr=1e-3`. LLRD optional (`lm_llrd_decay`).
- **Frozen embeddings**: Adam + ReduceLROnPlateau or cosine.

### Vectorized scatter-based statement scoring
`StmtHead._score_vectorized()` uses PyG `scatter` — avoid the Python-loop fallback `_score_loop`. Enable with `cfg.train.stmt_head_vectorized = true`. The vectorized path is 37–54× faster on GPU.

### Loss composition
Losses are additive; zero-weight terms are skipped. Add new loss terms by adding a weight field to `TrainConfig` and a corresponding branch in `Trainer._forward()`. Never bake losses into models.

### Lazy vs inmemory dataset
- `storage: lazy` — per-graph `.pt` files loaded on demand; ~8 GB RAM for MegaVul
- `storage: inmemory` — all graphs preloaded; ~60 GB RAM; faster training

Default: `lazy`. Use `inmemory` only on servers with enough RAM.

### Per-class cache + streaming merge
Dataset preprocessing saves one `.pt` per CWE class for crash recovery. Cache key encodes all config params — different configs never collide. Do not delete cache files mid-processing.

### LIVABLE epoch-adaptive weights
`epoch_adaptive_class_weights(counts, epoch, total_epochs)` ramps from uniform (epoch 0) to full inverse-frequency (final epoch): `w_i(t) = (N/(K·n_i))^(t/T)`. Always prefer this over static class weights for long-tail CWE distributions.

### EDAT adversarial training
`EmbeddingPGD.perturb_and_forward()` mutates `word_embeddings.weight` in-place. **Incompatible** with `torch.compile`, sliding window (`func_chunk_size > 0`), and DDP. Gate with assertion:
```python
if cfg.train.use_edat:
    assert not cfg.train.compile_model and cfg.model.func_chunk_size == 0
```

### Naming conventions
- Config IDs: `[Phase][Run]_description.yaml` — e.g., `H4_a1_l1_window_attn_stride1024.yaml`
- Checkpoints: `checkpoints/<timestamp>_<arch>/best_<arch>.pt` + `last_<arch>.pt`
- Results: `results/ablation/<phase>/<run_id>/metrics_summary.json`
- Models: `LM[GNN][Variant]VulnDetector` — e.g., `LMGATCodeBERTMTLVulnDetector`

---

## Important Files

| File | Purpose |
|------|---------|
| `src/gnn_vuln/config.py` | Single source of truth for all config fields and defaults |
| `src/gnn_vuln/models/lmgat_codebert.py` | Primary model — start here for architecture changes |
| `src/gnn_vuln/models/registry.py` | Add new architectures here |
| `src/gnn_vuln/training/trainer.py` | Forward dispatch and loss composition — touch carefully |
| `src/gnn_vuln/training/losses.py` | All loss functions |
| `src/gnn_vuln/data/dataset_lm.py` | Dataset class — all per-graph attributes documented here |
| `src/gnn_vuln/data/cwe_taxonomy.py` | CWE groupings, OWASP mappings, vocab |
| `configs/TEMPLATE.yaml` | Full config schema with comments for every field |
| `ABLATION_RESULTS.md` | Ground truth for all experiment results — never hand-edit metrics |
| `ARCHITECTURE.md` | Architecture evolution (Arch 1–12) with forward pass diagrams |
| `C:/Users/Otzzu/.claude/projects/c--Users-Otzzu-Documents-tugas-akhir/memory/MEMORY.md` | Session memory index — read at session start |

---

## Runtime / Tooling Preferences

- **Python**: 3.11+ (required)
- **Package manager**: `uv` exclusively — never `pip install` directly
- **CUDA**: 12.4 (PyTorch wheels from custom index in `pyproject.toml`)
- **PyTorch**: 2.2.0+, **PyG**: 2.5.0+, **Transformers**: 4.48–5.0
- **Linter/formatter**: Ruff (`line-length = 100`, `target-version = "py311"`)
- **Build backend**: Hatchling
- **Entry points**: `train` → `gnn_vuln.train:main`, `evaluate` → `gnn_vuln.evaluate:main`
- **Shell**: bash (Linux/cloud), PowerShell (Windows local)
- **Joern**: Required for CPG generation. Path: `C:/joern/joern-cli` (Windows), on PATH (Linux)
- **JDK**: 25 at `C:/Program Files/Java/jdk-25.0.3` — auto-detected by `joern_runner.py`
- **Flash attention**: Optional, incompatible with CodeT5+ (`use_flash_attention: false` for CodeT5+)
- **`torch.compile`**: Incompatible with EDAT PGD and in-place weight mutation
- **AMP**: `use_amp: true` recommended for VRAM savings; use `bf16` on Ampere+

### Key LM identifiers
| Key | HuggingFace path | Dim |
|-----|-----------------|-----|
| `microsoft/unixcoder-base` | UniXcoder | 768 |
| `microsoft/codebert-base` | CodeBERT | 768 |
| `Salesforce/codet5p-110m-embedding` | CodeT5+ proj | 256 |
| `jinaai/jina-embeddings-v2-base-code` | Jina v2 | 768 |
| `answerdotai/ModernBERT-base` | ModernBERT | 768 |

---

## Testing & QA

**No formal test suite** (`tests/` exists but is empty). Testing via:

```bash
# Vectorization correctness + speed (run after modifying StmtHead or losses)
uv run python scripts/test_stmthead_vectorized.py   # loop vs scatter; GPU optional
uv run python scripts/test_losses_vectorized.py     # mil_loss + ranking_loss; determinism check

# GNNPlus unit tests (positional encoding, edge index)
uv run python -m pytest src/GNNPlus/unittests/ -v
```

**Ablation metrics** — never hand-type results. Use the `/ablation-metrics <run_id>` slash command which reads `metrics_summary.json` + `training_summary.json` and appends formatted rows to `ABLATION_RESULTS.md`.

**Evaluation outputs** (auto-generated by `uv run evaluate`):
- `predictions.csv` — per-function predictions
- `localization_scores.csv` — per-statement suspicion scores
- `metrics_summary.json` — all metrics (F1, AUC, IFA, Top-K, Effort@20%, Recall@K%LOC)
- Plots: `roc_curve.png`, `confusion_matrix.png`, `pr_curve.png`, `recall_at_loc_curve.png`, `ifa_distribution.png`

**Key metrics** (higher = better except IFA and Effort):
- `f1_macro` — primary classification metric; used for early stopping
- `auc_roc_macro_ovr` — AUC one-vs-rest
- `top_1_accuracy` — localization: ≥1 flaw line in top-1 ranked statement
- `ifa_mean` — mean clean lines inspected before first flaw (lower = better)
- `effort_at_20pct_recall` — fraction of code to inspect for 20% flaw recall (lower = better)
