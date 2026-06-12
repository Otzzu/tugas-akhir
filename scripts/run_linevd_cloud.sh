#!/usr/bin/env bash
# run_linevd_cloud.sh — train + localize LineVD on OUR MegaVul split, upload to Drive.
# PILOT: LineVD = DGL + PyTorch-Lightning + Joern (2022 repo). Expect 1-2 pod iterations
# on the DGL<->CUDA version match and Joern setup. The data path is locked (our split +
# our flaw GT via linevd_prepare_megavul.py); what may need tuning is the env, not the data.
#
# Usage (pod, from project root):
#   bash scripts/run_linevd_cloud.sh
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
RUN_ID="linevd_megavul_ml1024_$(date +%Y%m%d_%H%M%S)"
WORK="$PWD"

echo "=== [1/6] clone LineVD (gitignored in our repo) ==="
[[ -d src/linevd ]] || git clone --depth 1 https://github.com/davidhin/linevd.git src/linevd

echo "=== [2/6] env (DGL must match pod torch+CUDA — adjust if it fails) ==="
python -m venv lvd_env && source lvd_env/bin/activate
pip install -q torch pandas fastparquet scikit-learn numpy tqdm networkx \
    pytorch-lightning torch_scatter transformers
# DGL: pick the wheel matching the pod's torch+CUDA (example cu121); see https://www.dgl.ai/pages/start.html
pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html || \
  echo "!! DGL install needs the right torch/CUDA index — fix per dgl.ai/start"

echo "=== [3/6] data + LineVD cache files (our split + flaw GT, no code edit) ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar -xzf "$DATA_TAR"
fi
cd src/linevd
PYTHONPATH=. python "$WORK/scripts/linevd_prepare_megavul.py" --in-dir "$WORK/megavul_ml1024/linevd"

echo "=== [4/6] build graphs (Joern) + codebert embeddings ==="
# Needs Joern installed + on PATH. getgraphs writes per-id CPGs that BigVulDatasetLineVD reads.
PYTHONPATH=. python sastvd/scripts/getgraphs.py
PYTHONPATH=. python sastvd/scripts/prepare.py

echo "=== [5/6] train + localize ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
PYTHONPATH=. python sastvd/scripts/train_best.py 2>&1 | tee "$OUT/train.log"
# eval/localization metrics are emitted by linevd's eval; copy storage outputs
cp -rf storage/processed "$OUT/" 2>/dev/null || true
cp -rf storage/outputs "$OUT/" 2>/dev/null || true
cd "$WORK"

echo "=== [6/6] upload weights + results ==="
tar -czf "${RUN_ID}_results.tar.gz" -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
WCKPT=$(find "$OUT" src/linevd/storage -name "*.ckpt" 2>/dev/null | head -1 || true)
if [[ -n "$WCKPT" ]]; then
  tar -czf "${RUN_ID}_weights.tar.gz" -C "$(dirname "$WCKPT")" "$(basename "$WCKPT")"
  rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
fi
echo "DONE: $RUN_ID -> results/baselines/${RUN_ID}_results.tar.gz (+weights if any)"
