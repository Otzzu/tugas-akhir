# LLM Context — GNN Vulnerability Detection (Tugas Akhir)

> This file is the authoritative reference for an AI assistant working on this project.
> Read this before touching any code.

---

## Project Goal

Build an AI model using **Graph Neural Networks (GNN)** to detect vulnerabilities in source code (C/C++ focus). This is a college final project (*tugas akhir*).

The classification task is **multi-class**: given a function's Code Property Graph (CPG), predict whether it is benign or belongs to a specific vulnerability type (e.g., CWE-119, CWE-120, CWE-476, etc.).

---

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Package manager | `uv` — all commands prefixed with `uv run` |
| Graph neural nets | PyTorch Geometric (`torch_geometric`) |
| Graphs | NetworkX (`networkx`) |
| ML utilities | scikit-learn, numpy, pandas |
| Visualisation | matplotlib, seaborn |
| Code parsing/CPG | Joern (external CLI tool, JVM-based) |
| Config | YAML via `pyyaml` + Python dataclasses |
| Logging | `loguru` |
| Testing | `pytest` |
| Linting | `ruff` |

**Python version**: 3.11 (pinned in `.python-version`)

---

## Repository Layout

```
tugas-akhir/
├── pyproject.toml              # uv/hatchling project definition + all deps
├── .python-version             # pins Python 3.11
├── .gitignore
├── README.md                   # user-facing setup guide
├── CONTEXT.md                  # this file — LLM reference
│
├── configs/
│   └── default.yaml            # ALL hyperparameters live here; edit this, not Python
│
├── src/gnn_vuln/               # main Python package (installed as editable via uv)
│   ├── __init__.py
│   ├── config.py               # Config, DataConfig, ModelConfig, TrainConfig dataclasses
│   ├── utils.py                # set_seed, setup_logging, save/load_checkpoint, get_device
│   ├── train.py                # training loop + CLI entry point (`uv run train`)
│   ├── evaluate.py             # metrics + plots entry point (`uv run evaluate`)
│   │
│   ├── data/
│   │   ├── preprocess.py       # strip_comments, normalize_identifiers, preprocess()
│   │   ├── graph_builder.py    # CPG → torch_geometric.data.Data
│   │   └── dataset.py          # VulnerabilityDataset (InMemoryDataset subclass)
│   │
│   └── models/
│       ├── gcn.py              # GCNVulnDetector
│       └── gat.py              # GATVulnDetector
│
├── data/
│   ├── raw/                    # INPUT: one subdir per class, each contains .json/.graphml CPG files
│   │   ├── benign/             # label 0 (alphabetically first)
│   │   └── vulnerable/         # label 1  (or cwe_119/, cwe_120/, etc. for multi-class)
│   └── processed/              # OUTPUT: auto-generated PyG dataset cache (.pt files)
│
├── notebooks/                  # Jupyter notebooks for EDA
├── checkpoints/                # saved model .pt files (gitignored)
├── results/                    # roc_curve.png, confusion_matrix.png (gitignored)
├── logs/                       # loguru log files (gitignored)
└── tests/
    ├── test_preprocess.py      # 5 tests
    ├── test_graph_builder.py   # 5 tests
    └── test_models.py          # 4 tests
```

---

## Data Pipeline (End-to-End)

```
1. RAW SOURCE CODE (C/C++ .c files)
         ↓
2. JOERN  (external CLI — not part of this repo)
   joern-parse file.c
   joern-export --repr cpg14 --out data/raw/<class_name>/
         ↓
3. data/raw/<class_name>/<function>.json   ← CPG as JSON or GraphML
         ↓
4. VulnerabilityDataset.process()          ← dataset.py
   - Scans all subdirs of data/raw/ (sorted alphabetically = label 0, 1, 2, ...)
   - Calls graph_builder.build_graph_from_json() on each file
         ↓
5. graph_builder.py
   - Parses JSON nodes/edges into NetworkX DiGraph
   - Extracts node features: keyword one-hot (C keywords) + structural features
   - Encodes edge types (AST/CFG/CDG/DDG/PDG/CALL) as one-hot edge_attr
   - Returns torch_geometric.data.Data(x, edge_index, edge_attr, y)
         ↓
6. data/processed/vulnerability_dataset.pt   ← cached; delete to re-process
         ↓
7. train.py  →  GNN model  →  evaluate.py
```

---

## Configuration Reference (`configs/default.yaml`)

```yaml
data:
  max_nodes: 500         # graphs larger than this are skipped
  node_feat_dim: 100     # (informational — actual dim = len(C_KEYWORDS) + 4 = 40)

model:
  architecture: gcn      # "gcn" or "gat"
  hidden_dim: 256        # GNN hidden layer width
  num_layers: 4          # number of message-passing layers
  dropout: 0.3
  heads: 4               # attention heads (GAT only)
  num_classes: 2         # MUST match number of class subdirs in data/raw/

train:
  seed: 42
  epochs: 100
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  patience: 10           # early stopping
  device: cpu            # or "cuda"
```

> **Critical**: `model.num_classes` must equal the number of subdirectories in `data/raw/`.

---

## Key Classes & Functions

### `src/gnn_vuln/config.py`
- `Config` — top-level config, `.from_yaml(path)` merges YAML over defaults
- `load_default_config()` — loads `configs/default.yaml`, falls back to defaults

### `src/gnn_vuln/data/graph_builder.py`
- `NODE_FEAT_DIM` — int, actual node feature dimensionality (= `len(C_KEYWORDS) + 4`)
- `build_graph_from_json(path, label, max_nodes)` → `Data | None`
- `build_graph_from_graphml(path, label, max_nodes)` → `Data | None`
- `build_graph_from_networkx(g, label, max_nodes)` → `Data | None`
- Node features: C keyword one-hot + `[has_literal, is_call, lineno_norm, out_degree_norm]`
- Edge features: one-hot over `EDGE_TYPES = [AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF]`

### `src/gnn_vuln/data/dataset.py` — `VulnerabilityDataset`
- Subclass of `torch_geometric.data.InMemoryDataset`
- `root="data"` → expects `data/raw/` with class subdirs
- Labels assigned by **alphabetical sort** of subdir names
- Stores `self.class_names: list[str]` and `self.num_classes: int`
- `get_splits(train_ratio, val_ratio, seed)` → `(train_idx, val_idx, test_idx)`
- Delete `data/processed/vulnerability_dataset.pt` to force re-processing

### `src/gnn_vuln/models/gcn.py` — `GCNVulnDetector`
- `GCNConv × num_layers → BatchNorm → ReLU → Dropout → global_mean_pool → MLP(hidden//2, num_classes)`
- Constructor: `(in_channels, hidden_dim=256, num_layers=4, dropout=0.3, num_classes=2)`

### `src/gnn_vuln/models/gat.py` — `GATVulnDetector`
- `GATConv(heads, concat=True) × (num_layers-1) → GATConv(heads=1, concat=False) → global_mean_pool → MLP`
- Constructor: `(in_channels, hidden_dim=64, num_layers=3, heads=4, dropout=0.3, num_classes=2)`

### `src/gnn_vuln/train.py`
- `build_model(cfg, in_channels)` → model from config
- `train_one_epoch(model, loader, optimizer, device)` → avg loss
- `evaluate(model, loader, device)` → `(avg_loss, accuracy)`
- `main()` — CLI: reads `--config`, trains with early stopping, saves best checkpoint

### `src/gnn_vuln/evaluate.py`
- `get_predictions(model, loader, device)` → `(y_true [N], y_pred [N], y_prob [N, num_classes])`
- `plot_roc_curve(...)` — binary: single curve; multi-class: per-class OvR curves
- `plot_confusion_matrix(...)` — dynamically sized, works for any N classes
- AUC: `roc_auc_score(multi_class='ovr', average='macro')` for N>2
- F1: `f1_score(average='macro')` for N>2, `average='binary'` for N=2

### `src/gnn_vuln/utils.py`
- `set_seed(seed)`, `setup_logging(log_dir)`, `get_device(requested)`
- `save_checkpoint(model, path, **extra)`, `load_checkpoint(model, path, device)`

---

## CLI Entry Points

```powershell
# Install all deps (run once)
uv sync
uv sync --extra dev        # also installs jupyter, pytest, ruff

# Train
uv run train --config configs/default.yaml

# Evaluate a saved checkpoint
uv run evaluate --checkpoint checkpoints/best_gcn.pt --config configs/default.yaml

# Tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Jupyter
uv run jupyter lab
```

---

## Multi-Class Setup Instructions

1. Create one folder per class under `data/raw/`, named semantically:
   ```
   data/raw/benign/        → label 0
   data/raw/cwe_119/       → label 1
   data/raw/cwe_120/       → label 2
   data/raw/cwe_476/       → label 3
   ```
2. Fill each folder with Joern-exported `.json` or `.graphml` CPG files
3. Update `configs/default.yaml`: set `model.num_classes: 4`
4. If `data/processed/vulnerability_dataset.pt` exists, delete it first
5. Run `uv run train --config configs/default.yaml`

---

## Important Design Decisions

| Decision | Rationale |
|----------|-----------|
| `src/` layout | Prevents accidental relative imports; aligns with packaging best practices |
| YAML config over argparse everywhere | Single source of truth for hyperparams; reproducible experiments |
| `InMemoryDataset` with cache | Joern extraction is slow; caching avoids redoing it every run |
| Alphabetical label assignment | Deterministic, no hidden state — label mapping always visible via `ds.class_names` |
| `global_mean_pool` for graph readout | Simple and effective; swap to `global_add_pool` or `global_max_pool` to experiment |
| CPU default | Works on any machine; change `device: cuda` in YAML for GPU |
| `loguru` over stdlib `logging` | Simpler API, coloured console output, automatic file rotation |

---

## Known Limitations / Future Work

- Node features are shallow (keyword one-hot). Replace with **CodeBERT / UniXcoder embeddings** for better representations.
- Only `GCN` and `GAT` implemented. Can add **GIN**, **GraphSAGE**, or **Devign's GGNN**.
- No heterogeneous graph support yet (all edge types mixed into `edge_attr`). Could use `HeteroData` for richer modelling.
- `tree-sitter` is installed but not yet wired up — can replace the regex-based `preprocess.py` with a proper AST parser.
- No hyperparameter search script yet. Consider adding Optuna integration.
