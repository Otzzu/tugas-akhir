#!/usr/bin/env bash
# run_losver_cloud.sh — train + eval LOSVER (ASE'25) on OUR MegaVul split, upload to Drive.
# CWE multiclass + line localization baseline. Same funcs/split/flaw-GT as our model
# (jsonl built from our flaw_lines via export_losver_jsonl.py). Vuln-only, 25 CWE classes.
#
# Usage (pod, from project root):
#   bash scripts/run_losver_cloud.sh
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
RUN_ID="losver_megavul_ml1024_$(date +%Y%m%d_%H%M%S)"
WORK="$PWD"
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"

echo "=== [1/7] LOSVER present (vendored; clone only if missing) ==="
[[ -d src/losver ]] || git clone --depth 1 https://github.com/waroad/losver.git src/losver

echo "=== [2/7] deps + transformers-compat patch (POD env, transformers 4.48; no venv) ==="
# Run in the pod env (correct torch). LOSVER pins transformers 4.19 only for AdamW/WEIGHTS_NAME
# (removed in newer transformers); we patch those to torch.optim instead of downgrading
# (4.19 + modern torch breaks on torch._six). torch/sklearn/pandas/numpy/tqdm = project env.
pip install -q tensorboardX
python scripts/losver_patch_compat.py --files \
  src/losver/classification/run_line_CWE.py \
  src/losver/classification/run_weighted_CWE.py \
  src/losver/classification/run_base_CWE.py
python -c "import torch; print('torch', torch.__version__, '| cuda op:', (torch.randn(2).cuda()+1).sum().item())"

echo "=== [3/7] UniXcoder model ==="
cd src/losver && python download_unixcoder.py && cd "$WORK"   # -> src/losver/unixcoder-nine

echo "=== [4/7] data + build LOSVER jsonl from our split (our flaw GT) ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar -xzf "$DATA_TAR"
fi
PYTHONPATH=src python scripts/export_losver_jsonl.py \
  --in-dir megavul_ml1024/linevd --out-dir megavul_ml1024/losver \
  --tokenizer microsoft/unixcoder-base-nine --token-limit 512 2>&1 | tee "$OUT/convert.log"
# grab the printed cwe_labels list for the label patch
LABELS=$(grep '^cwe_labels = ' "$OUT/convert.log" | sed 's/^cwe_labels = //')
# LOSVER inserts a fold index into the filename (CWE_train -> CWE_train1) and runs
# 5-fold CV. We have ONE split -> name it fold "1" and run only fold 1 (patched below).
for s in train val test; do
  cp "megavul_ml1024/losver/CWE_${s}_unix_512.jsonl" "src/losver/classification/CWE_${s}1_unix_512.jsonl"
done

echo "=== [5/7] patch CWE label set (BigVul 36 -> our 25) ==="
python scripts/losver_patch_labels.py \
  --files src/losver/classification/run_line_CWE.py \
          src/losver/classification/run_weighted_CWE.py \
          src/losver/classification/run_base_CWE.py \
  --labels "$LABELS"

echo "=== [6/7] train: localizer -> weighted classifier ==="
cd src/losver/classification
U=../unixcoder-nine
python run_line_CWE.py --output_dir="$OUT/localizer" --model_type roberta \
  --model_name_or_path=$U --tokenizer_name=$U \
  --train_data_file=CWE_train_unix_512.jsonl --eval_data_file=CWE_val_unix_512.jsonl --test_data_file=CWE_test_unix_512.jsonl \
  --block_size=512 --seed=123456 --do_train --do_test 2>&1 | tee "$OUT/localizer.log"
python run_weighted_CWE.py --output_dir="$OUT/classifier" --model_type roberta \
  --model_name_or_path=$U --tokenizer_name=$U \
  --localized_location="$OUT/localizer" \
  --block_size=512 --seed=123456 --do_train --do_test 2>&1 | tee "$OUT/classifier.log"
cd "$WORK"

echo "=== [7/7] upload weights + results ==="
tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
WBIN=$(find "$OUT" -name "*.bin" | head -1 || true)
if [[ -n "$WBIN" ]]; then
  tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_weights.tar.gz" -C "$OUT" "$(realpath --relative-to="$OUT" "$WBIN")"
  rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
fi
echo "DONE: $RUN_ID -> results/baselines/${RUN_ID}_results.tar.gz (+weights)"
