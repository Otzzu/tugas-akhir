# CLAUDE.md — Project Instructions

## Code Comments

Keep comments minimal. Match the surrounding code's comment density. No verbose multi-line explanations of a one-line change — one short inline comment max when truly needed.

## Git Commit Style

Always keep the conventional-commit type prefix (`feat:`, `fix:`, `docs:`, `refactor:`, etc.) — the colon after the type is required and allowed. Keep the description short, one line. In the description use only comma and dot — no slashes, semicolons, parentheses, or extra colons. Never add a Co-Authored-By trailer. Commit only when the user asks.

Good: `feat: add N42 N43 rank weight, N44 supcon group`
Bad: `feat: N42/N43 rank ablation; N44 (supcon group) + code` (slashes, semicolon, parens)
Bad: `add N42 N43 rank weight` (missing type prefix)

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

## Archiving — always use the fastest method

Datasets/results are multi-GB. **Never** plain `gzip`/`tar -z`/`zip` for these. Always use `pigz` (parallel gzip, all cores) for compression; install it if missing.

```bash
# install guard (put once near the top of any script that archives)
command -v pigz >/dev/null || apt-get install -y -q pigz

# compress (multi-core; pigz uses all cores by default, no -p needed)
tar -cf - <dir> | pigz > out.tar.gz

# extract
tar -I pigz -xf out.tar.gz
```

Note: pigz parallelizes **compression** only — gz **decompression** is single-core by format, so extraction speed is I/O-bound (especially lazy datasets = many tiny per-graph files). That is expected, not a misconfiguration. In scripts use `COMP="$(command -v pigz || echo gzip)"` then `tar -I "$COMP"` so it degrades gracefully but prefers pigz.

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

## Cloud Training

When asked for commands to run training in the cloud, **always give both**:
1. The direct `PYTHONPATH=src python -m gnn_vuln.train` command (for reference)
2. The `train_cloud.sh` command (for actual cloud use)

`scripts/train_cloud.sh` handles: rclone setup, dataset download from Drive, train, evaluate, zip+upload results, optional .pt cleanup.

**Flags:**
- `--init` — fresh server, runs `setup_cloud.sh` + installs `rclone.conf`
- `--skip` — server already configured, skip all setup
- `--resume` — continue from `last_{arch}.pt`
- `--clean-every N` — delete dataset .pt after every Nth run (use N=total_runs to clean after last)
- `--backbone ID` — download a trained backbone checkpoint before the run loop (for cRT). `ID` = the model_id whose `<ID>_checkpoints.zip` is on Drive `checkpoints/`. Unzips to `checkpoints/<ID>/best_*.pt` so a cRT config loads it via `crt_init_checkpoint`. Repeatable.

Each `--config` must be paired with a `--dataset` (same position).

**Dataset name** = zip/tar filename on `gdrive-mesach:tugas-akhir/` WITHOUT extension.
Find from a recent `training_summary.json` → `dataset_pt` field, strip `_meta.pt` suffix.
Current megavul multiclass dataset: `lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42`
Storage mode: **lazy** (graphs stored as individual files, not a single .pt).
Note: configs use CRLF on Windows — the script's storage-marker grep may show a cosmetic warning (`No inmemory-marked tar found`) due to CRLF, but the fallback finds the correct `_lazy_` tar and downloads fine.

```bash
# Single run (server already set up)
./scripts/train_cloud.sh --skip \
  --config configs/ablation/gnn_only/N48_a1_l1_jknet.yaml \
  --dataset lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42

# Multiple runs, same dataset, clean after last
./scripts/train_cloud.sh --skip --clean-every 2 \
  --config configs/ablation/gnn_only/N48_a1_l1_jknet.yaml \
  --dataset lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42 \
  --config configs/ablation/gnn_only/N49_a1_l1_imtl_mid2.yaml \
  --dataset lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42

# Fresh server
./scripts/train_cloud.sh --init \
  --config <config.yaml> \
  --dataset <dataset_name>

# cRT run (N53) — needs N48 backbone checkpoint first
./scripts/train_cloud.sh --skip \
  --backbone 20260606_163818_lmgat_codebert_multiclass \
  --config configs/ablation/gnn_only/N53_a1_l1_crt_n48.yaml \
  --dataset lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42
```
