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

# EVAL_ONLY=1 -> skip training: restore saved weights from checkpoints/baselines, run --do_test only,
# dump test predictions, and recompute MACRO-F1 via compute_baseline_metrics (LOSVER logs only weighted).
# Requires WEIGHTS_TAR=<run>_weights.tar.gz (name in checkpoints/baselines). [VERIFY on pod]
EVAL_ONLY="${EVAL_ONLY:-}"
WEIGHTS_TAR="${WEIGHTS_TAR:-}"
if [[ -n "$EVAL_ONLY" && -z "$WEIGHTS_TAR" ]]; then echo "ERR: EVAL_ONLY=1 needs WEIGHTS_TAR=<...>_weights.tar.gz"; exit 1; fi
TRAINFLAG="--do_train"; [[ -n "$EVAL_ONLY" ]] && TRAINFLAG=""
export LOSVER_PRED_CSV="$OUT/losver_cls_preds.csv"

echo "=== [1/7] LOSVER present (vendored; clone only if missing) ==="
[[ -d src/losver ]] || git clone --depth 1 https://github.com/waroad/losver.git src/losver
# reset patched files to pristine so re-running on a persistent pod re-applies seds cleanly (no double-patch)
git -C src/losver checkout -- classification/run_line_CWE.py classification/run_weighted_CWE.py classification/run_base_CWE.py 2>/dev/null || true

echo "=== [2/7] deps + transformers-compat patch (POD env, transformers 4.48; no venv) ==="
# Run in the pod env (correct torch). LOSVER pins transformers 4.19 only for AdamW/WEIGHTS_NAME
# (removed in newer transformers); we patch those to torch.optim instead of downgrading
# (4.19 + modern torch breaks on torch._six). torch/sklearn/pandas/numpy/tqdm = project env.
# some pod base images ship torch only (no transformers/sklearn) -> install if missing.
python -c "import transformers" 2>/dev/null || pip install -q transformers
python -c "import sklearn, pandas, numpy, tqdm" 2>/dev/null || pip install -q scikit-learn pandas numpy tqdm
pip install -q tensorboardX
python scripts/losver_patch_compat.py --files \
  src/losver/classification/run_line_CWE.py \
  src/losver/classification/run_weighted_CWE.py \
  src/losver/classification/run_base_CWE.py
python -c "import torch; print('torch', torch.__version__, '| cuda op:', (torch.randn(2).cuda()+1).sum().item())"

echo "=== [3/7] UniXcoder model ==="
( cd src/losver && python download_unixcoder.py )   # subshell -> parent CWD stays $WORK even on failure; -> src/losver/unixcoder-nine

echo "=== [4/7] data + build LOSVER jsonl from our split (our flaw GT) ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar -I "$(command -v pigz || echo gzip)" -xf "$DATA_TAR"
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

# EVAL_ONLY: inject a test-predictions dump (y_true,y_pred) into test() so compute_baseline_metrics
# can report MACRO-F1. Injected right after `preds = np.argmax(...)` (also hits unused evaluate(), harmless).
if [[ -n "$EVAL_ONLY" ]]; then
  sed -i '/preds = np.argmax(logits, axis=1)/a\    __import__("pandas").DataFrame({"y_true": list(labels), "y_pred": list(preds)}).to_csv(__import__("os").environ["LOSVER_PRED_CSV"], index=False)' \
    src/losver/classification/run_weighted_CWE.py
fi

# EVAL_ONLY: restore saved localizer+classifier weights (tar holds localizer1/ + classifier1/ with
# checkpoint-best-f1) into $OUT so --do_test loads them ($OUT/classifier -> code appends fold "1").
if [[ -n "$EVAL_ONLY" ]]; then
  echo "=== [6a/7] restore weights $WEIGHTS_TAR -> $OUT ==="
  rclone copy "$REMOTE/checkpoints/baselines/$WEIGHTS_TAR" "$OUT/" --progress
  tar -I "$(command -v pigz || echo gzip)" -xf "$OUT/$WEIGHTS_TAR" -C "$OUT" && rm -f "$OUT/$WEIGHTS_TAR"
fi

echo "=== [6/7] ${EVAL_ONLY:+eval-only }localizer -> weighted classifier ==="
cd src/losver/classification
U=../unixcoder-nine
python run_line_CWE.py --output_dir="$OUT/localizer" --model_type roberta \
  --model_name_or_path=$U --tokenizer_name=$U \
  --train_data_file=CWE_train_unix_512.jsonl --eval_data_file=CWE_val_unix_512.jsonl --test_data_file=CWE_test_unix_512.jsonl \
  --block_size=512 --seed=123456 $TRAINFLAG --do_test 2>&1 | tee "$OUT/localizer.log"
python run_weighted_CWE.py --output_dir="$OUT/classifier" --model_type roberta \
  --model_name_or_path=$U --tokenizer_name=$U \
  --localized_location="$OUT/localizer" \
  --block_size=512 --seed=123456 $TRAINFLAG --do_test 2>&1 | tee "$OUT/classifier.log"
cd "$WORK"

# EVAL_ONLY: recompute MACRO-F1 (+ weighted/micro + per-class report) from the dumped predictions.
if [[ -n "$EVAL_ONLY" ]]; then
  echo "=== [6b/7] recompute macro-F1 from dumped predictions ==="
  PYTHONPATH=src python scripts/compute_baseline_metrics.py --name LOSVER \
    --classification "$LOSVER_PRED_CSV" --out "$OUT/losver_recomputed_metrics.json" 2>&1 | tee "$OUT/recomputed_metrics.log"
fi

echo "=== [7/7] upload weights + results ==="
# results MUST exclude model artifacts. LOSVER has TWO models (localizer/ + classifier/),
# each a UniXcoder .bin (~480MB) — bundling both here bloats results to >1GB.
tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_results.tar.gz" \
  --exclude='*.bin' --exclude='*.pt' --exclude='*.safetensors' --exclude='*.ckpt' --exclude='optimizer*' \
  -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
# weights = BOTH models (localizer + classifier), bundled together -> checkpoints/baselines.
# EVAL_ONLY just restored these from Drive -> skip re-upload (identical).
mapfile -t WBINS < <(find "$OUT" -name "*.bin")
if [[ -z "$EVAL_ONLY" && ${#WBINS[@]} -gt 0 ]]; then
  REL=(); for b in "${WBINS[@]}"; do REL+=("$(realpath --relative-to="$OUT" "$b")"); done
  tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_weights.tar.gz" -C "$OUT" "${REL[@]}"
  rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
fi
echo "DONE: $RUN_ID -> results/baselines/${RUN_ID}_results.tar.gz (+weights)"
