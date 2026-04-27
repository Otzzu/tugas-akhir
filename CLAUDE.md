# CLAUDE.md — Project Instructions

## Python Environment

This project uses **uv** for dependency management. A virtual environment is already created at `.venv/`.

**Always run Python using one of these forms:**

```bash
# Preferred — uv resolves the venv automatically
uv run python <script>
uv run python scripts/prepare_dataset.py ...
uv run train --config configs/lmgcn/binary.yaml

# Acceptable — activate the uv-managed venv first
.venv/Scripts/python <script>       # Windows
.venv/bin/python <script>           # Linux/macOS
```

**Never use bare `python` or `pip` directly** — they may point to the system Python and will miss all project dependencies.

To add a new dependency:
```bash
uv add <package>
```

## Windows Path Note

When passing Windows paths as CLI arguments from **bash**, use **forward slashes** to avoid the shell eating backslashes:

```bash
# Good
--joern-cli C:/joern/joern-cli

# Bad — bash turns \j into j
--joern-cli C:\joern\joern-cli
```

From **PowerShell**, backslashes work fine.

## Project Structure

```
src/gnn_vuln/          # Main Python package
  data/
    graph_builder.py       # Joern GraphML/JSON → PyG Data (keyword features)
    graph_builder_lm.py    # Joern JSON → PyG Data (CodeBERT node features)
    dataset_lm.py          # CodeBERTGraphDataset (InMemoryDataset)
    node_embedder.py       # Frozen CodeBERT wrapper for per-node embeddings
    joern_runner.py        # Subprocess wrapper around joern-parse / joern-export
    preprocess.py          # C/C++ identifier normalisation
  models/
    lmgcn.py               # LMGCNVulnDetector — GCNConv + CodeBERT node features
    lmgat.py               # LMGATVulnDetector — GATConv + CodeBERT node features
  train.py                 # Training entry point

scripts/
  prepare_dataset.py       # Single-dataset CPG generation (Joern pipeline)
  prepare_all.ps1          # Batch script — all 3 datasets (PowerShell)
  download_datasets.py     # Download Devign / BigVul / DiverseVul from HuggingFace

data/
  datasets/                # Raw HuggingFace parquet files
    devign/train.parquet
    bigvul/train.parquet
    diversevul/train.parquet
  raw/                     # Generated Joern CPG files (train)
    benign/
    vulnerable/
  raw_val/                 # Validation CPGs
  raw_test/                # Test CPGs

configs/
  lmgcn/
    binary.yaml            # LM-GCN binary detection
    multiclass.yaml        # LM-GCN 11-class CWE classification
  lmgat/
    binary.yaml            # LM-GAT binary (GAT + ranking loss + class weights)
    multiclass.yaml        # LM-GAT multiclass (GAT + class weights — fixes CWE collapse)
```

## Joern CPG Generation

Joern must be installed at `C:/joern/joern-cli` (or pass `--joern-cli`).  
JDK 25 is at `C:/Program Files/Java/jdk-25.0.3` — auto-detected by `joern_runner.py`.

Run CPG generation:
```bash
# Single dataset, balanced 2000/class, 4 parallel workers
uv run python scripts/prepare_dataset.py \
    --input data/datasets/devign/train.parquet \
    --format devign \
    --joern-cli C:/joern/joern-cli \
    --out-dir data/raw \
    --sample-per-class 2000 \
    --workers 4

# All datasets at once (PowerShell)
.\scripts\prepare_all.ps1 -SamplePerClass 2000 -Workers 4
```

## Training

```bash
# LM-GCN
uv run train --config configs/lmgcn/binary.yaml
uv run train --config configs/lmgcn/multiclass.yaml

# LM-GAT (improved: attention heads + ranking loss + class weights)
uv run train --config configs/lmgat/binary.yaml
uv run train --config configs/lmgat/multiclass.yaml
```
