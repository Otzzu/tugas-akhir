# Complete Run Guide

End-to-end instructions from a clean checkout to trained model and evaluation results.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| [uv](https://docs.astral.sh/uv/) | any | Python/venv manager |
| Python | 3.11 | uv installs it automatically |
| [Joern](https://joern.io) | latest | CPG extraction; install at `C:/joern/joern-cli` |
| JDK | 11+ | JDK 25 at `C:/Program Files/Java/jdk-25.0.3` is auto-detected |
| GPU (optional) | CUDA 11.8+ | Set `train.device: cuda` in config for ~10× speedup |

Install uv (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

---

## Step 0 — Install dependencies

```bash
uv sync
```

Smoke check:
```bash
uv run python -c "import torch; from torch_geometric.data import Data; print('OK')"
```

---

## Step 1 — Download datasets

Downloads BigVul, Devign, and DiverseVul from HuggingFace into `data/datasets/`.

```bash
uv run python scripts/download_datasets.py
```

Expected output structure:
```
data/datasets/
  bigvul/train.parquet       # ~183K functions, with CWE labels + flaw lines
  devign/train.parquet       # ~21K functions, binary labels
  diversevul/train.parquet   # ~264K functions, binary labels
```

---

## Step 2 — Generate CPGs (Joern pipeline)

Joern converts raw C/C++ functions into Code Property Graphs (CPG) stored as JSON.  
This is the slowest step. Use `--workers 4` to parallelise.

### Binary mode (benign vs vulnerable)

```bash
# BigVul — binary, 2000 samples per class
uv run python scripts/prepare_dataset.py \
    --input data/datasets/bigvul/train.parquet \
    --format bigvul \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --sample-per-class 2000 \
    --binary \
    --workers 4

# Devign — binary (has flaw-line GT for localization metrics)
uv run python scripts/prepare_dataset.py \
    --input data/datasets/devign/train.parquet \
    --format devign \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --sample-per-class 2000 \
    --workers 4
```

### Multi-class mode (CWE classification)

`--top-cwe N` selects the N most frequent CWE types from the dataset as classes 1..N. The default is 10. You can use any value — the only constraint is that `model.num_classes` in the config must equal `N + 1` (the +1 is for the benign class 0).

```bash
# Top 10 CWEs + benign = 11 classes (matches configs/lmgcn/multiclass.yaml and configs/lmgat/multiclass.yaml)
uv run python scripts/prepare_dataset.py \
    --input data/datasets/bigvul/train.parquet \
    --format bigvul \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --top-cwe 10 \
    --sample-per-class 2000 \
    --workers 4

# Top 5 CWEs + benign = 6 classes (set model.num_classes: 6 in config)
uv run python scripts/prepare_dataset.py \
    --input data/datasets/bigvul/train.parquet \
    --format bigvul \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --top-cwe 5 \
    --sample-per-class 2000 \
    --workers 4
```

Samples whose CWE type falls outside the top-N are dropped automatically.

Multi-class mode produces two extra files:

- `data/raw/cwe_vocab.json` — `{"benign": 0, "CWE-119": 1, ...}` (presence of this file enables multi-class mode)
- `data/raw/vulnerable/<name>.meta.json` — per-CPG sidecar: `{"class_id": 2, "cwe": "CWE-20", "flaw_lines": [14, 17]}`

### Validation and test splits

Run the same script with `--out-dir data/raw_val` and `--out-dir data/raw_test` using held-out parquet files, or let `CodeBERTGraphDataset` handle the 70/15/15 split automatically from a single `data/raw/` directory.

### Identifier normalization (recommended for ablation study)

Add `--normalize` to rename user-defined variable and function names to generic tokens (`VAR_0`, `FUNC_0`, etc.) before Joern parses the code. This prevents the model from memorizing dataset-specific names like `vuln_buf` or `CVE_2019_helper`.

```bash
uv run python scripts/prepare_dataset.py \
    --input data/datasets/bigvul/train.parquet \
    --format bigvul \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --normalize \
    --workers 4
```

C keywords (`int`, `void`, `return`, ...), standard types (`size_t`, `uint32_t`, ...), and common stdlib functions (`malloc`, `memcpy`, `printf`, ...) are preserved. Uses tree-sitter for accurate C parsing.

Run without `--normalize` as baseline, with `--normalize` as ablation, then compare metrics to see whether identifier bias affects your model.

### Clear processed cache after regenerating CPGs

```bash
# Remove only the final dataset (preprocessing restarts from scratch)
rm data/processed/lm_dataset.pt

# Remove per-class checkpoint cache (see Step 3 — Preprocessing cache)
rm -rf data/processed/cls_cache/

# Remove everything
rm -rf data/processed/
```

The final `lm_dataset.pt` is stale whenever CPG files change. Training rebuilds it automatically.

---

## Step 3 — Train

### Binary detection

```bash
# LM-GCN
uv run train --config configs/lmgcn/binary.yaml

# LM-GAT (attention + ranking loss + class weights)
uv run train --config configs/lmgat/binary.yaml
```

### Multi-class CWE classification

```bash
# LM-GCN
uv run train --config configs/lmgcn/multiclass.yaml

# LM-GAT
uv run train --config configs/lmgat/multiclass.yaml
```

### Preprocessing cache (per-class checkpoints)

The first time you run train, it preprocesses all CPG files through CodeBERT before training
starts. This can take **1-3 hours** depending on GPU speed and dataset size.

To protect against crashes, each class is saved to a checkpoint immediately after it finishes.
The cache directory is mode-specific so binary and multiclass never interfere:

```
data/processed/
    lm_dataset_binary.pt          <- final binary cache (built once, reused forever)
    lm_dataset_multiclass.pt      <- final multiclass cache

    cls_cache_binary/             <- temporary, deleted when binary cache is complete
        benign.pt
        vulnerable.pt

    cls_cache_multiclass/         <- temporary, deleted when multiclass cache is complete
        benign.pt
        vuln_CWE_119.pt           <- one file per CWE class
        vuln_CWE_20.pt
        ...
```

**If the process crashes**, just rerun the same train command. It will skip already-finished
classes and resume from where it stopped:

```
Resuming - 3 unit(s) already cached: ['benign', 'vuln_CWE_119', 'vuln_CWE_20']
Remaining: ['vuln_CWE_399', 'vuln_CWE_125', ...]
```

Once all classes are done the final `.pt` is written and the `cls_cache_*` folder is deleted.

**To restart preprocessing from scratch** (e.g. after changing `max_nodes` or node features):

```bash
# Binary only
rm data/processed/lm_dataset_binary.pt

# Multiclass only
rm data/processed/lm_dataset_multiclass.pt

# Everything
rm -rf data/processed/
```

### Checkpoint and results folders

Every training run creates its own timestamped folder named `YYYYMMDD_HHMMSS_<mode>`:

```
checkpoints/
    20260426_002451_multiclass/     <- one folder per run, never overwritten
        best_lmgcn.pt               <- best weights (val_loss), saved when improved
        last_lmgcn.pt               <- full state (model + optimizer + scheduler + epoch)
        config.yaml                 <- copy of config used for this run

results/
    20260426_002451_multiclass/     <- same run_id as checkpoint folder
        predictions.csv
        localization_scores.csv
        metrics_summary.json
        roc_curve.png
        confusion_matrix.png
        pr_curve.png
        recall_at_loc_curve.png     <- only when flaw-line GT exists
        ifa_distribution.png        <- only when flaw-line GT exists
```

The epoch log shows:
```
  Train 001/100: 100%|████| 85/85 [01:58<00:00, loss=2.1234]
Epoch 001/100 | train=2.3756 | val=2.3960 | acc=0.3046 | conf=0.3340 | lr=1.00e-03 | patience=0/10 | 135s *
```

| Field | Meaning |
|-------|---------|
| `train` / `val` | loss values |
| `acc` | validation accuracy |
| `conf` | mean max-softmax confidence |
| `lr` | current learning rate (drops when scheduler fires) |
| `patience` | epochs without improvement / early-stop threshold |
| `*` | new best model saved this epoch |

### Resume training after interruption

```bash
uv run train --config configs/lmgat/binary.yaml --resume
```

Finds the most recent `*_binary/last_lmgcn.pt` and continues from that epoch.
Resume only matches runs of the same mode — a binary resume will never pick up a multiclass checkpoint.

---

## Step 4 — Evaluate

Use the exact checkpoint path printed at the end of training:

```bash
# Binary (replace run_id with your actual folder name)
uv run evaluate \
    --checkpoint checkpoints/20260426_XXXXXX_binary/best_lmgcn.pt \
    --config checkpoints/20260426_XXXXXX_binary/config.yaml

# Multi-class
uv run evaluate \
    --checkpoint checkpoints/20260426_XXXXXX_multiclass/best_lmgcn.pt \
    --config checkpoints/20260426_XXXXXX_multiclass/config.yaml
```

Results are saved to `results/<run_id>/` matching the checkpoint folder name.

### Console output

```
=================================================================
Function-Level Classification Report
=================================================================
              precision    recall  f1-score   support
           0       0.85      0.82      0.83       620
           1       0.83      0.86      0.84       630
    accuracy                           0.84      1250
   macro avg       0.84      0.84      0.84      1250

AUC-ROC (macro OvR) : 0.9123
F1 Score (macro)    : 0.8400
Accuracy            : 0.8400
-----------------------------------------------------------------
Confidence (mean)   : 0.8562
Confidence (correct): 0.8890
Confidence (wrong)  : 0.7201  (n=200)
=================================================================

=================================================================
Statement-Level Localization  (functions with flaw GT: 630)
=================================================================
  Top-10 Accuracy    : 0.6800  (LineVul reported 0.65)
  IFA (mean)         : 4.23    (LineVul reported 4.56; lower = better)
  Effort@20%Recall   : 0.1550  (lower = better)
  Recall@1%LOC       : 0.2100
  Recall@5%LOC       : 0.4800
  Recall@20%LOC      : 0.7200
=================================================================
```

### Output files in `results/`

| File | Description |
|------|-------------|
| `predictions.csv` | y_true, y_pred, confidence, correct, prob\_\<class\> per column |
| `localization_scores.csv` | func\_idx, y\_true, y\_pred, line\_number, score, is\_flaw\_line |
| `metrics_summary.json` | all scalar metrics + Recall@LOC curve arrays (NaN → null) |
| `roc_curve.png` | ROC curve — single for binary, OvR per class for multi-class |
| `confusion_matrix.png` | confusion matrix with all classes shown |
| `pr_curve.png` | Precision-Recall curve |
| `recall_at_loc_curve.png` | Recall@K%LOC curve *(only when flaw-line GT exists)* |
| `ifa_distribution.png` | IFA distribution histogram *(only when flaw-line GT exists)* |

Localization outputs require non-empty `flaw_lines` in `.meta.json` sidecars.

---

## Configuration reference

Configs are organised by model under `configs/<model>/`:

| Config | Mode | `num_classes` | Notes |
|--------|------|:---:|-------|
| `configs/lmgcn/binary.yaml` | binary | 2 | LM-GCN: benign vs vulnerable |
| `configs/lmgcn/multiclass.yaml` | multi-class | 11 | LM-GCN: 10 CWE types + benign |
| `configs/lmgat/binary.yaml` | binary | 2 | LM-GAT: attention + ranking loss |
| `configs/lmgat/multiclass.yaml` | multi-class | 11 | LM-GAT: attention + class weights |

Key fields (edit in the YAML, not in code):

| Field | Default | Description |
|-------|---------|-------------|
| `data.max_nodes` | 500 | graphs larger than this are skipped |
| `model.hidden_dim` | 256 | GCN hidden dimension |
| `model.num_layers` | 4 | GCNConv message-passing layers |
| `model.dropout` | 0.3 | dropout rate |
| `model.num_classes` | 2 | must match `len(cwe_vocab.json)` in multi-class mode |
| `model.mil_weight` | 0.5 | statement MIL loss weight λ (0 = disable) |
| `model.mil_k` | 3 | top-k statements used per function in MIL |
| `train.epochs` | 100 | max epochs |
| `train.batch_size` | 32 | reduce if OOM |
| `train.lr` | 0.001 | Adam learning rate |
| `train.patience` | 10 | early-stopping patience |
| `train.device` | cpu | `cpu` or `cuda` |

---

## Common errors

### `ValueError: Config model.num_classes=2 but the dataset has 11 classes`

Config says binary (`num_classes: 2`) but the processed cache is multiclass, or vice versa.
Make sure `data.mode` in the config matches what you want: `binary` or `multiclass`.

### `FileNotFoundError: data/processed/lm_dataset_binary.pt` (or `_multiclass.pt`)

First run for that mode — preprocessing will start automatically. If the file is missing but
the directory exists, delete the directory and re-run:
```bash
rm -rf data/processed/
```

### `RuntimeError: CUDA out of memory`

Reduce `train.batch_size` in the config (try 8 or 4).

### Joern CPG generation hangs or times out

- Verify Joern is installed: `C:/joern/joern-cli/joern-parse --help`
- Check JDK is ≥11: `java -version`
- Reduce `--workers` to 1 to debug a single file

### Preprocessing restarts from scratch instead of resuming

The per-class cache (`data/processed/cls_cache/`) must exist. If it was deleted or the process
was killed before saving the first class, there is nothing to resume from. This is expected —
just let it run again.

If you see wrong data (e.g. you changed `max_nodes` but old embeddings are being loaded), delete
the cache manually and restart:

```bash
rm -rf data/processed/
```

### `No localization data collected (node_line not in dataset)`

`node_line` is built by `graph_builder_lm.py` from Joern's `lineNumber` property.  
If all values are -1 the CPG lacks line info — ensure the source file was parsed, not a pre-built binary.
