# GNN Vulnerability Detection — Tugas Akhir

> Deteksi kerentanan kode sumber menggunakan Graph Neural Network (GNN) dengan CodeBERT node embeddings.  
> Final project untuk tugas akhir, dikelola dengan [uv](https://docs.astral.sh/uv/).

---

## Background

Proyek ini membangun model AI berbasis **Graph Neural Network (GNN)** untuk mendeteksi kerentanan (*vulnerability*) secara otomatis pada kode sumber C/C++, sekaligus melokalisasi pernyataan (*statement*) yang mencurigakan.

Pipeline utama:
```
Source Code → Joern (CPG Extraction) → graph_builder_lm → CodeBERTGraphDataset → LM-GCN → Vuln / Benign + Line Scores
```

Representasi graf yang digunakan adalah **Code Property Graph (CPG)**, yang menggabungkan:
- **AST** (Abstract Syntax Tree)
- **CFG** (Control Flow Graph)
- **PDG** (Program Dependence Graph)

Model yang digunakan:
- `lmgcn` — **LM-GCN**: CodeBERT node features + GCNConv encoder + dual output heads
  - *Function head*: classifies the whole function (binary or multi-class CWE)
  - *Statement head*: per-line suspiciousness score for MIL-based localisation (WAVES-style)

---

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed globally:
  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```
- Python 3.11 (uv will install it automatically)
- [Joern](https://joern.io/docs/) installed at `C:/joern/joern-cli` for CPG extraction
- JDK 11+ (JDK 25 at `C:/Program Files/Java/jdk-25.0.3` is auto-detected)

### 1. Install dependencies

```bash
uv sync
```

### 2. Install dev extras (Jupyter, pytest, linter)

```bash
uv sync --extra dev
```

### 3. Run tests (smoke check)

```bash
uv run pytest tests/ -v
```

### 4. Open Jupyter

```bash
uv run jupyter lab
```

---

## Project Structure

```
tugas-akhir/
├── src/gnn_vuln/                   # Main source package
│   ├── config.py                   # Centralised config dataclasses
│   ├── utils.py                    # Seed, logging, checkpoint helpers
│   ├── train.py                    # Training loop (entry point: `uv run train`)
│   ├── evaluate.py                 # Evaluation + metrics plots
│   ├── data/
│   │   ├── preprocess.py           # C/C++ identifier normalisation
│   │   ├── graph_builder_lm.py     # Joern JSON → PyG Data (CodeBERT features)
│   │   ├── dataset_lm.py           # CodeBERTGraphDataset (InMemoryDataset)
│   │   ├── node_embedder.py        # Frozen CodeBERT wrapper for node embeddings
│   │   └── joern_runner.py         # Subprocess wrapper for joern-parse/export
│   └── models/
│       ├── lmgcn.py                # LMGCNVulnDetector: GCNConv + CodeBERT + MIL head
│       └── lmgat.py                # LMGATVulnDetector: GATConv + CodeBERT + MIL head
├── configs/
│   ├── lmgcn/
│   │   ├── binary.yaml             # LM-GCN binary detection
│   │   └── multiclass.yaml         # LM-GCN 11-class CWE classification
│   └── lmgat/
│       ├── binary.yaml             # LM-GAT binary (attention + ranking loss)
│       └── multiclass.yaml         # LM-GAT multiclass (attention + class weights)
├── data/
│   ├── datasets/                   # Raw HuggingFace parquet files
│   │   ├── devign/train.parquet
│   │   ├── bigvul/train.parquet
│   │   └── diversevul/train.parquet
│   ├── raw/                        # Generated Joern CPG files — train split
│   │   ├── benign/                 # Benign function CPGs (.json + .meta.json)
│   │   ├── vulnerable/             # Vulnerable function CPGs (.json + .meta.json)
│   │   └── cwe_vocab.json          # (optional) CWE→class_id map; enables multi-class
│   ├── raw_val/                    # Validation CPGs (same layout as raw/)
│   ├── raw_test/                   # Test CPGs
│   └── processed/                  # Auto-generated PyG dataset cache (gitignored)
├── scripts/
│   ├── download_datasets.py        # Download Devign / BigVul / DiverseVul (HuggingFace)
│   ├── prepare_dataset.py          # Single-dataset CPG generation pipeline
│   └── prepare_all.ps1             # Batch: all 3 datasets (PowerShell)
├── notebooks/                      # EDA and experiment notebooks
├── tests/                          # Pytest unit tests
├── checkpoints/                    # Saved model weights (gitignored)
│   ├── best_lmgcn.pt               # Best val-loss model weights
│   └── last_lmgcn.pt               # Full resume state (model + optimizer + scheduler)
├── results/                        # Plots and metrics output (gitignored)
└── pyproject.toml                  # uv/hatchling project definition
```

---

## Data Preparation

### Step 1 — Download datasets

```bash
uv run python scripts/download_datasets.py
```

Downloads BigVul, Devign, and DiverseVul from HuggingFace into `data/datasets/`.

### Step 2 — Generate CPGs with Joern

Each dataset has different label/localization support:

| Dataset    | Classification | Localization GT | Notes |
|------------|---------------|-----------------|-------|
| **BigVul** | Multi-class (CWE) or binary | Diff-based flaw lines | Default dataset |
| **Devign** | Binary | `vul_lines` field | ~21K functions |
| **DiverseVul** | Binary only | None | ~264K functions |

Run for a single dataset:

```bash
# BigVul — multi-class (top 10 CWEs), balanced 2000/class, 4 workers
uv run python scripts/prepare_dataset.py \
    --input data/datasets/bigvul/train.parquet \
    --format bigvul \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --sample-per-class 2000 \
    --workers 4

# BigVul — binary mode (collapse all CWEs → label 1)
uv run python scripts/prepare_dataset.py \
    --input data/datasets/bigvul/train.parquet \
    --format bigvul \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --binary

# Devign — binary + localization
uv run python scripts/prepare_dataset.py \
    --input data/datasets/devign/train.parquet \
    --format devign \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw
```

Or run all datasets at once (PowerShell):

```powershell
.\scripts\prepare_all.ps1 -SamplePerClass 2000 -Workers 4
```

**Important:** After regenerating CPGs, delete the processed cache before training:

```bash
rm data/processed/lm_dataset.pt
```

### Sidecar files

Each CPG file `<name>.json` may have a sidecar `<name>.meta.json`:

```json
{"class_id": 2, "cwe": "CWE-20", "flaw_lines": [14, 17]}
```

- `class_id` — integer class label (0 = benign, 1..K = CWE category)
- `flaw_lines` — 1-indexed source lines known to be vulnerable (ground truth for MIL)

If `raw/cwe_vocab.json` exists the dataset runs in **multi-class mode**; otherwise it falls back to binary.

---

## Training

```bash
# LM-GCN
uv run train --config configs/lmgcn/binary.yaml
uv run train --config configs/lmgcn/multiclass.yaml

# LM-GAT (attention heads + ranking loss + class weights)
uv run train --config configs/lmgat/binary.yaml
uv run train --config configs/lmgat/multiclass.yaml

# Resume from last checkpoint
uv run train --config configs/lmgat/binary.yaml --resume
```

Two checkpoint files are maintained per run:

| File | Contents | When saved |
|------|----------|-----------|
| `checkpoints/best_lmgcn.pt` | Model weights only | When val loss improves |
| `checkpoints/last_lmgcn.pt` | Model + optimizer + scheduler + epoch + patience state | Every epoch (atomic write) |

Training logs epoch loss, accuracy, and **mean confidence** (max softmax probability):

```
Epoch 001/100 | train_loss=0.5231 | val_loss=0.4812 | val_acc=0.7634 | val_conf=0.8120
```

---

## Evaluation

```bash
# LM-GCN binary
uv run evaluate --checkpoint checkpoints/<run_id>/best_lmgcn.pt --config configs/lmgcn/binary.yaml

# LM-GAT multiclass
uv run evaluate --checkpoint checkpoints/<run_id>/best_lmgat.pt --config configs/lmgat/multiclass.yaml
```

Prints function-level classification report (precision, recall, F1, AUC-ROC) and statement-level localization metrics.

**Saved to `results/`:**

| File | Description |
|------|-------------|
| `predictions.csv` | y_true, y_pred, confidence, correct, prob per class |
| `localization_scores.csv` | per-(function, line): score + is_flaw_line flag |
| `metrics_summary.json` | all scalar metrics + curve arrays (NaN → null) |
| `roc_curve.png` | ROC curve (OvR per class for multi-class) |
| `confusion_matrix.png` | confusion matrix (all classes shown) |
| `pr_curve.png` | Precision-Recall curve |
| `recall_at_loc_curve.png` | Recall@K%LOC curve (only if flaw GT exists) |
| `ifa_distribution.png` | IFA histogram (only if flaw GT exists) |

**Statement-level localization metrics** (LineVul/WAVES definitions, requires flaw-line GT in `.meta.json`):
- **Top-10 Accuracy** — fraction of functions with ≥1 flaw line in top-10 ranked statements
- **IFA** — mean clean lines inspected before the first flaw line (lower = better)
- **Effort@20%Recall** — fraction of all lines to inspect to catch 20% of flaw lines (lower = better)
- **Recall@K%LOC** — flaw recall when inspecting top K% of lines (1%, 5%, 20%)

See [RUNNING.md](RUNNING.md) for the complete step-by-step guide from scratch.

---

## Configuration

Configs live under `configs/<model>/` — one folder per architecture:

| Key | Default | Description |
|-----|---------|-------------|
| `model.architecture` | `lmgcn` | `lmgcn` or `lmgat` |
| `model.hidden_dim` | `256` | Hidden layer width |
| `model.num_layers` | `4` | Number of message-passing layers |
| `model.dropout` | `0.3` | Dropout rate |
| `model.heads` | `4` | GAT attention heads (lmgat only) |
| `model.num_classes` | `2` | `2` = binary; `11` = 10-CWE multiclass |
| `model.mil_weight` | `0.5` | MIL statement loss weight (0 = disable) |
| `model.mil_k` | `3` | Top-k statements per function for MIL pseudo-labels |
| `model.rank_loss_weight` | `0.0` | Pairwise ranking loss weight (0 = disable) |
| `train.epochs` | `100` | Maximum training epochs |
| `train.lr` | `0.001` | Learning rate |
| `train.patience` | `10` | Early-stopping patience |
| `train.use_class_weights` | `true` | Inverse-frequency class weighting |
| `train.device` | `cpu` | `cpu` or `cuda` |

---

## Model Architecture

**LMGCNVulnDetector** (`lmgcn.py`):

```
CodeBERT CLS token (768D) + node_type (1D) → 769D node features
    ↓
GCNConv × num_layers  (BatchNorm + ReLU + Dropout)
    ↓
┌─ Function head: global_mean_pool → MLP → logit_func [B, num_classes]
└─ Statement head: group nodes by line → max/mean pool → score_line [n_lines_i]
```

The statement head uses **WAVES-style MIL** loss: top-k scored lines per function are pushed toward the function's binary label (benign=0, any vuln=1) at training time, providing implicit line-level supervision without requiring per-line labels.

Node features are built by freezing `microsoft/codebert-base` and extracting the `[CLS]` token for each node's code token sequence.

---

## Dataset Loading

```python
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset

ds = CodeBERTGraphDataset(root="data", embedder_device="cpu")
print(ds[0])
# Data(x=[N, 769], edge_index=[2, E], y=1, node_line=[N], flaw_line_mask=[N])

train_idx, val_idx, test_idx = ds.get_splits(seed=42)
```

Each `Data` object has:
- `x` — node features `[N, 769]`
- `edge_index` — CPG edges `[2, E]`
- `y` — function-level label (int)
- `node_line` — source line number per node `[N]` (-1 if unknown)
- `flaw_line_mask` — ground-truth line mask `[N]` (1 if node is on a flaw line)

---

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Auto-fix lint
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/
```

---

## References

- **VulLMGNN**: Cao et al., *Vulnerability Detection with Graph Simplification and Enhanced Graph Representation Learning*, ICSE 2023
- **WAVES**: Ni et al., *WAVES: Weakly Supervised Vulnerability Statement Localization*, 2023
- **LineVul**: Fu & Tantithamthavorn, *LineVul: A Transformer-based Line-Level Vulnerability Prediction*, MSR 2022
- **Devign**: Zhou et al., *Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks*, NeurIPS 2019
- **BigVul**: Fan et al., *A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries*, MSR 2020
- **DiverseVul**: Chen et al., *DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection*, RAID 2023
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- Joern: https://joern.io
- CodeBERT: https://huggingface.co/microsoft/codebert-base
