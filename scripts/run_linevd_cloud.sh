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

echo "=== [1/6] LineVD present (vendored in-repo; clone only if missing) ==="
[[ -d src/linevd ]] || git clone --depth 1 https://github.com/davidhin/linevd.git src/linevd

echo "=== [2/6] deps (POD env — torch from setup_cloud; no venv) ==="
# torch / torch_scatter / transformers / pandas / sklearn / numpy / tqdm = project env.
pip install -q pytorch-lightning networkx fastparquet
# DGL's graphbolt imports torchdata.datapipes, REMOVED in torchdata>=0.10 -> pin older.
# --no-deps so it can't drag torch to an older version (datapipes is pure-python).
pip install -q --no-deps 'torchdata==0.9.0' || pip install -q --no-deps 'torchdata<0.10'
# DGL: auto-pick the wheel matching the pod torch + CUDA (not a hardcoded cu121).
TORCH_MM=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')")
TORCH_CU=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.',''))")
echo "    DGL target: torch-${TORCH_MM} / ${TORCH_CU}"
pip install -q dgl -f "https://data.dgl.ai/wheels/torch-${TORCH_MM}/${TORCH_CU}/repo.html" \
  || pip install -q dgl \
  || echo "!! DGL wheel for torch-${TORCH_MM}/${TORCH_CU} not found — see https://www.dgl.ai/pages/start.html"
python -c "import torch,dgl; print('torch',torch.__version__,'dgl',dgl.__version__,'| cuda op:',(torch.randn(2).cuda()+1).sum().item())"

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
