#!/usr/bin/env bash
# run_linevul_cloud.sh — train + localize LineVul on OUR MegaVul split, upload to Drive.
# Baseline for the localization comparison (same funcs / split / flaw GT as our model).
#
# Usage (pod):
#   bash scripts/run_linevul_cloud.sh
# Assumes: rclone configured (gdrive-mesach:), src/LineVul present (git), Joern NOT needed.
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
RUN_ID="linevul_megavul_ml1024_$(date +%Y%m%d_%H%M%S)"
WORK="$PWD"
OUT="$WORK/baseline_runs/$RUN_ID"
mkdir -p "$OUT"

echo "=== [1/5] deps (POD env — torch already correct from setup_cloud; no venv) ==="
# torch/transformers/pandas/sklearn/numpy/tqdm = project env. Only captum is extra.
pip install -q captum
python -c "import torch; print('torch', torch.__version__, '| cuda op:', (torch.randn(2).cuda()+1).sum().item())"

echo "=== [2/5] data ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress
  tar -I "$(command -v pigz || echo gzip)" -xf "$DATA_TAR"
fi
D="$WORK/megavul_ml1024/linevul"

echo "=== [3/5] train + classify ==="
cd src/LineVul/linevul
python linevul_main.py \
  --output_dir="$OUT" --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
  --do_train --do_test \
  --train_data_file="$D/train.csv" --eval_data_file="$D/val.csv" --test_data_file="$D/test.csv" \
  --epochs 10 --block_size 512 --train_batch_size 16 --eval_batch_size 16 \
  --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 42 \
  2>&1 | tee "$OUT/train.log"

echo "=== [4/5] localization (IFA / Top-K / Effort@20%R / Recall@K%LOC) ==="
python linevul_main.py \
  --output_dir="$OUT" --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
  --do_test --do_local_explanation --reasoning_method=attention --do_sorting_by_line_scores \
  --test_data_file="$D/test.csv" --block_size 512 --eval_batch_size 16 --seed 42 \
  --effort_at_top_k 0.2 --top_k_recall_by_lines 0.01 --top_k_recall_by_pred_prob 0.2 \
  2>&1 | tee "$OUT/localization.log"
cd "$WORK"

echo "=== [5/5] upload weights + results ==="
# weights = best model .bin under OUT/checkpoint-best-f1; results = logs + any csv
# results MUST exclude model artifacts (the .bin goes to checkpoints/baselines separately,
# bundling it here bloats the results tar to ~800MB — see repackage_baseline_drive.sh).
cp -f "$OUT"/*.log "$OUT"/ 2>/dev/null || true
tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_results.tar.gz" \
  --exclude='*.bin' --exclude='*.pt' --exclude='*.safetensors' --exclude='*.ckpt' --exclude='optimizer*' \
  -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
# weights separately (can be large)
WBIN=$(find "$OUT" -name "*.bin" | head -1 || true)
if [[ -n "$WBIN" ]]; then
  tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_weights.tar.gz" -C "$(dirname "$WBIN")" "$(basename "$WBIN")"
  rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
fi
echo "DONE: $RUN_ID  -> results/baselines/${RUN_ID}_results.tar.gz  (+weights)"
