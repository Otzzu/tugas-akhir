# GNN Vulnerability Detection — Tugas Akhir

> Deteksi kerentanan kode sumber menggunakan Graph Neural Network (GNN).  
> Final project untuk tugas akhir, dikelola dengan [uv](https://docs.astral.sh/uv/).

---

## Background

Proyek ini membangun model AI berbasis **Graph Neural Network (GNN)** untuk mendeteksi kerentanan (*vulnerability*) secara otomatis pada kode sumber C/C++.

Pipeline utama:
```
Source Code → Preprocessing → Joern (CPG Extraction) → Graph Builder → GNN Model → Vuln / Benign
```

Representasi graf yang digunakan adalah **Code Property Graph (CPG)**, yang menggabungkan:
- **AST** (Abstract Syntax Tree)
- **CFG** (Control Flow Graph)  
- **PDG** (Program Dependence Graph)

Model GNN yang tersedia:
- `gcn` — Graph Convolutional Network (baseline)
- `gat` — Graph Attention Network

---

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed globally:
  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```
- Python 3.11 (uv will install it automatically)
- [Joern](https://joern.io/docs/) for CPG extraction (optional for initial setup)

### 1. Install dependencies

```powershell
cd c:\Users\Otzzu\Documents\tugas-akhir
uv sync
```

> After `uv sync`, install PyG sparse extensions (needed for some GNN ops):
> ```powershell
> uv run pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
> ```

### 2. Install dev extras (Jupyter, pytest, linter)

```powershell
uv sync --extra dev
```

### 3. Run tests (smoke check)

```powershell
uv run pytest tests/ -v
```

### 4. Open Jupyter

```powershell
uv run jupyter lab
```

---

## Project Structure

```
tugas-akhir/
├── src/gnn_vuln/             # Main source package
│   ├── config.py             # Centralised config dataclasses
│   ├── utils.py              # Seed, logging, checkpoint helpers
│   ├── train.py              # Training loop (entry point: `uv run train`)
│   ├── evaluate.py           # Evaluation + metrics plots
│   ├── data/
│   │   ├── preprocess.py     # Code normalisation utilities
│   │   ├── graph_builder.py  # Joern CPG → PyG Data
│   │   └── dataset.py        # VulnerabilityDataset (InMemoryDataset)
│   └── models/
│       ├── gcn.py            # GCN baseline
│       └── gat.py            # Graph Attention Network
├── configs/
│   └── default.yaml          # Hyperparameter config (edit here, not in code)
├── data/
│   ├── raw/
│   │   ├── vulnerable/       # Joern CPG exports for vulnerable functions (.json/.graphml)
│   │   └── benign/           # Joern CPG exports for benign functions
│   └── processed/            # Auto-generated PyG dataset cache
├── notebooks/                # EDA and experiment notebooks
├── scripts/                  # CLI helper scripts
├── tests/                    # Pytest unit tests
├── checkpoints/              # Saved model weights (gitignored)
├── results/                  # Plots and metrics output (gitignored)
└── pyproject.toml            # uv/hatchling project definition
```

---

## Data Preparation

### Step 1 — Collect C/C++ source files

Collect vulnerable functions from public datasets (e.g., [Devign](https://sites.google.com/view/devign), [BigVul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset), [NVD](https://nvd.nist.gov/)).

### Step 2 — Extract CPG with Joern

```bash
# Install Joern (requires JDK 11+)
# https://joern.io/docs/

joern-parse path/to/source_file.c
joern-export --repr cpg14 --out output_dir/
```

Place the exported `.json` or `.graphml` files into:
- `data/raw/vulnerable/` for vulnerable functions
- `data/raw/benign/` for clean functions

### Step 3 — Process dataset

The dataset is processed automatically on first use:
```python
from gnn_vuln.data.dataset import VulnerabilityDataset
ds = VulnerabilityDataset(root="data")
print(f"Total graphs: {len(ds)}")
```

---

## Training

```powershell
# Using the CLI entry point
uv run train --config configs/default.yaml

# Or directly
uv run python -m gnn_vuln.train --config configs/default.yaml
```

Edit `configs/default.yaml` to change model architecture, hyperparameters, or epochs.

---

## Evaluation

```powershell
uv run evaluate --checkpoint checkpoints/best_gcn.pt --config configs/default.yaml
```

Outputs:
- Classification report (precision, recall, F1)
- AUC-ROC score
- `results/roc_curve.png`
- `results/confusion_matrix.png`

---

## Configuration

All settings live in `configs/default.yaml`. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `model.architecture` | `gcn` | `gcn` or `gat` |
| `model.hidden_dim` | `256` | GNN hidden layer width |
| `model.num_layers` | `4` | Number of message-passing layers |
| `train.epochs` | `100` | Maximum training epochs |
| `train.lr` | `0.001` | Learning rate |
| `train.patience` | `10` | Early stopping patience |
| `train.device` | `cpu` | `cpu` or `cuda` |

---

## Development

```powershell
# Lint
uv run ruff check src/ tests/

# Auto-fix lint
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/
```

---

## References

- Devign: *Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks* (Zhou et al., NeurIPS 2019)
- Reveal: *Deep Learning based Vulnerability Detection: Are We There Yet?* (Chakraborty et al., TSE 2022)
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- Joern: https://joern.io
