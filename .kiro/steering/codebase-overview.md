# Codebase Overview — GNN Vulnerability Detection (Tugas Akhir)

## What It Is

A thesis project for automated C/C++ vulnerability detection using Graph Neural Networks. It classifies functions as benign or vulnerable (binary), or by CWE type (multiclass up to 11 classes), and localizes suspicious lines within vulnerable functions.

---

## The Pipeline

```
C/C++ Source Code
    → Joern (CPG extraction: AST + CFG + PDG)
    → graph_builder_lm.py (CPG → PyG Data objects)
    → CodeBERTGraphDataset (cached .pt files)
    → GNN Model (encoder + function head + statement head)
    → Training / Evaluation
```

Each CPG node gets a **773D feature vector**: `[node_type(1)] + [CodeBERT CLS(768)] + [dist_features(3)] + [danger_api(1)]`. Edges are 7D one-hot (AST/CFG/CDG/DDG/PDG/CALL/REACHING_DEF).

---

## Architecture Evolution (12 models)

All models are registered in `src/gnn_vuln/models/registry.py` via `MODEL_REGISTRY`.

| Arch | Key | Name | Key Idea |
|------|-----|------|----------|
| 1 | `lmgcn` | LM-GCN | GCNConv + frozen CodeBERT nodes |
| 2 | `lmgat` ✅ current baseline | LM-GAT v2 | GATv2 + edge-type attention + ranking loss |
| 3 | `lmgat_codebert` | LM-GAT+CodeBERT | GATv2 + **live** CodeBERT (concat fusion) |
| 4 | `lmgat_mcs` | LM-GAT-MCS | Multiclass statement head (WAVES-style strict sync) |
| 5 | `lmgin` | LM-GIN | GINEConv (most expressive aggregation) |
| 6 | `lmgat_interp` | LM-GAT-Interp | λ-interpolation between GNN and LM logits (VulLMGNN) |
| 7 | `lmgat_seq` | LM-GAT-Seq | Sequential: Stage 1 localizes → Stage 2 classifies |
| 8 | `lmgat_waves_seq` | LM-GAT-WAVES-Seq | Transformer localization → GATv2+LM classification |
| 9 | `lmggnn` | LM-GGNN | GatedGraphConv + live LM |
| 10 | `lmgat_dualflow` | LM-GAT-DualFlow | Dual-flow pooling (focal + context) |
| 11 | `lmgat_codebert_mtl` | LM-GAT-MTL | MTL heads: binary + group + CWE |
| 12 | `lmgat_hcdfgat` 🎯 target | HC-DFGAT | Dual-flow + MTL + hierarchical contrastive loss |

Also: `lmrgcn`, `lmrgcn_codebert` — Relational GCN variants (one weight matrix per edge type).

---

## Source Layout

```
src/gnn_vuln/
  config.py                  # DataConfig, ModelConfig, TrainConfig dataclasses; YAML loading
  train.py                   # CLI entry: uv run train; TrainingSession orchestrates full run
  evaluate.py                # CLI entry: uv run evaluate; Evaluator runs inference + saves outputs
  metrics.py                 # LocalizationMetrics: IFA, Effort@20%, Recall@K%LOC
  utils.py                   # Seed, logging, checkpoint save/load, CheckpointManager
  inference.py               # Single-function inference API
  baselines.py               # Baseline model implementations

  data/
    dataset_lm.py            # CodeBERTGraphDataset (PyG InMemoryDataset); per-class cache; CWE filtering
    graph_builder_lm.py      # Shim → cpg/ subpackage; CPG JSON → PyG Data objects
    cpg/                     # CPG parsing, node feature construction, edge encoding
    node_embedder.py         # LMNodeEmbedder — frozen CodeBERT for per-node CLS embeddings
    joern_runner.py          # Subprocess wrapper for joern-parse / joern-export
    preprocess.py            # C/C++ identifier normalization (tree-sitter)
    cwe_taxonomy.py          # CWE_GROUP_MAP, GROUP_VOCAB, OWASP mappings, filter expansion

  models/
    registry.py              # MODEL_REGISTRY dict + build_model() factory
    base.py                  # VulnDetectorBase: _build_lm_branch(), _lm_embed(), from_config()
    encoders.py              # GATEncoder, GCNEncoder, RGCNEncoder, GGNNEncoder, GINEncoder
    heads.py                 # StmtHead, MulticlassStmtHead, FuncHead, SmallFuncHead, MTLHeads
    lmgat.py                 # Arch 2 (current baseline)
    lmgcn.py                 # Arch 1
    lmgat_codebert.py        # Arch 3
    lmgat_codebert_mtl.py    # Arch 11
    lmgat_mcs.py             # Arch 4
    lmgat_interp.py          # Arch 6
    lmgat_seq.py             # Arch 7
    lmgat_dualflow.py        # Arch 10
    lmgat_hcdfgat.py         # Arch 12 (target)
    lmgat_waves_seq.py       # Arch 8
    lmggnn.py                # Arch 9
    lmgin.py                 # Arch 5
    lmrgcn.py                # RGCN (frozen)
    lmrgcn_codebert.py       # RGCN + live LM

  training/
    trainer.py               # Trainer: forward dispatch, train_epoch(), evaluate(), localise()
    losses.py                # focal_loss, livable_weights, mil_loss, mil_loss_multiclass, ranking_loss
    optimizer.py             # build_optimizer_and_scheduler(): AdamW+warmup vs Adam+ReduceLROnPlateau
    ewc.py                   # EWCDR: Elastic Weight Consolidation Done Right (continual learning)

  evaluation/
    localize.py              # LocalizationExtractor: inference → per-line scores
    plots.py                 # ResultPlotter: ROC, confusion matrix, PR, Recall@LOC, IFA histogram

  losses/
    hierarchical_supcon.py   # HierarchicalSupConLoss: contrastive loss with CWE tree distance weighting

scripts/
  download_datasets.py       # Download BigVul/Devign/DiverseVul from HuggingFace
  prepare_dataset.py         # Single-dataset CPG generation pipeline
  prepare_all.ps1            # Batch: all datasets (PowerShell)

configs/
  lmgat/                     # Arch 2 configs (binary.yaml, multiclass.yaml)
  lmgat_codebert/            # Arch 3 configs
  lmgat_codebert_mtl/        # Arch 11 configs
  lmgat_dualflow/            # Arch 10 configs
  lmgat_hcdfgat/             # Arch 12 configs
  lmgcn/                     # Arch 1 configs
  lmgin/                     # Arch 5 configs
  lmgat_interp/              # Arch 6 configs
  lmgat_seq/                 # Arch 7 configs
  lmgat_waves_seq/           # Arch 8 configs
  lmggnn/                    # Arch 9 configs
  lmrgcn/                    # RGCN configs
  data/                      # Dataset-specific configs (bigvul, megavul, titanvul variants)
```

---

## Key Design Patterns

### Model Registry
`MODEL_REGISTRY` maps architecture name strings → classes. `build_model(cfg, in_channels)` is the single factory. Every model implements `from_config(cfg, in_channels)`. Adding a new architecture = one dict entry + one class.

### Unified Forward Dispatch
`Trainer._forward()` inspects the return tuple length to handle all architectures:
- 2-tuple `(logit, stmt_scores)` — standard
- 3-tuple `(logit, stmt_scores, z)` — SupCon
- 4-tuple `(logit_cwe, logit_group, logit_binary, stmt_scores)` — MTL
- 5-tuple `(logit_cwe, logit_group, logit_binary, stmt_scores, z)` — MTL + SupCon

### Dual-Head Output
Every model produces `(logit_func, stmt_scores)`. The statement head (`StmtHead` / `MulticlassStmtHead` in `heads.py`) groups CPG nodes by source line, max/mean-pools per line, and returns per-line suspicion scores. Localization is trained via **MIL** — top-k lines per function get pseudo-labels from the function's class label, requiring no per-line annotations.

### Loss Composition
```
total_loss = CE/Focal(func, class_weight)
           + mil_weight    * MIL(stmt_scores)
           + rank_weight   * RankingLoss(stmt_scores, flaw_line_mask)
           + group_weight  * CE(logit_group)       # MTL archs only
           + binary_weight * CE(logit_binary)      # MTL archs only
           + supcon_weight * HierarchicalSupCon(z) # HC-DFGAT only
```

### Optimizer Bifurcation
`build_optimizer_and_scheduler()` auto-detects LM-fine-tuning architectures:
- **LM archs** (`lmgat_codebert`, `lmgat_seq`, `lmgat_dualflow`, `lmgat_hcdfgat`, etc.): AdamW with two param groups (LM at `lm_lr=2e-5` + linear warmup, GNN at `lr=1e-3`)
- **Frozen-embedding archs** (`lmgat`, `lmgcn`, `lmgin`, etc.): plain Adam + ReduceLROnPlateau

### Per-Class Cache + Streaming Merge
Dataset preprocessing saves one `.pt` per CWE class for crash recovery. A two-pass streaming merge (scan sizes → pre-allocate → fill) assembles the final dataset without loading everything into RAM. The `.pt` filename encodes all relevant config parameters so different configs never collide.

### LIVABLE Epoch-Adaptive Weights
`livable_weights()` ramps class weights from uniform at epoch 0 to full inverse-frequency at the final epoch: `w_i(t) = (N/(K·n_i))^(t/T)`. Prevents early training collapse on minority CWE classes.

### EWC-DR (Continual Learning)
`EWCDR` implements Elastic Weight Consolidation Done Right with Logits Reversal. Configurable scope (all/lm/gnn) lets you protect only the LM branch when fine-tuning on a new dataset.

---

## Configuration System

YAML files under `configs/<arch>/`. Three sections map to dataclasses in `config.py`:

**`data:`** — `source` (bigvul/megavul/titanvul), `mode` (binary/multiclass/group), `max_nodes`, `top_cwe`, `cwe_list`, `cwe_groups`, `filter_owasp`, `filter_top25_dangerous`, `max_per_class`, `source_val`, `source_test`

**`model:`** — `architecture` (registry key), `pretrained_lm`, `func_lm`, `add_func_tokens`, `hidden_dim`, `num_layers`, `dropout`, `heads`, `edge_dim`, `num_classes`, `mil_weight`, `mil_k`, `rank_loss_weight`, `group_loss_weight`, `binary_loss_weight`, `use_supcon`, `supcon_weight`, `active_heads`

**`train:`** — `lr`, `lm_lr`, `warmup_ratio`, `grad_clip`, `weight_decay`, `epochs`, `batch_size`, `patience`, `early_stop_metric`, `use_class_weights`, `focal_loss_gamma`, `livable_loss`, `device`, `use_amp`, `compile_model`, `grad_accum_steps`

---

## How to Run

```bash
# Setup
uv sync
uv run pytest tests/ -v

# Data preparation
uv run python scripts/download_datasets.py
uv run python scripts/prepare_dataset.py \
    --input data/datasets/bigvul/train.parquet \
    --format bigvul --joern-cli C:/joern/joern-cli \
    --out-dir data/raw --top-cwe 10 --sample-per-class 2000 --workers 4

# Training
uv run train --config configs/lmgat/multiclass.yaml
uv run train --config configs/lmgat_codebert_mtl/multiclass_mtl.yaml
uv run train --config configs/lmgat/multiclass.yaml --resume

# Evaluation
uv run evaluate \
    --checkpoint checkpoints/<run_id>/best_lmgat.pt \
    --config checkpoints/<run_id>/config.yaml
```

---

## Datasets

| Dataset | Size | Classification | Localization GT | Notes |
|---------|------|---------------|-----------------|-------|
| **BigVul** | ~183K | CWE multiclass or binary | Diff-based flaw lines | Best for multiclass + localization |
| **Devign** | ~21K | Binary | `vul_lines` field | — |
| **DiverseVul** | ~264K | Binary | None | — |
| **MegaVul** | — | CWE multiclass | — | Referenced in configs |
| **TitanVul** | — | CWE multiclass | — | Referenced in configs |

---

## Evaluation Metrics

**Function-level:** F1 macro, AUC-ROC (OvR), accuracy, confidence

**Statement-level localization** (requires flaw-line GT in `.meta.json`):
- **Top-K Accuracy** — fraction of functions with ≥1 flaw line in top-K ranked statements
- **IFA** — mean clean lines inspected before the first flaw line (lower = better)
- **Effort@20%Recall** — fraction of all lines to inspect to catch 20% of flaw lines (lower = better)
- **Recall@K%LOC** — flaw recall when inspecting top K% of lines (1%, 5%, 20%)

---

## References

- **VulLMGNN**: Cao et al., ICSE 2023
- **WAVES**: Ni et al., 2023
- **LineVul**: Fu & Tantithamthavorn, MSR 2022
- **Devign**: Zhou et al., NeurIPS 2019
- **BigVul**: Fan et al., MSR 2020
- **DiverseVul**: Chen et al., RAID 2023
- **LIVABLE**: arXiv:2306.06935
