#!/usr/bin/env bash
# run_vulexplainer_megavul.sh — VulExplainer (TSE'23, CNN-teacher -> GraphCodeBERT-student hierarchical
# distillation) VTP on OUR megavul split, comparable head-to-head. Model code UNTOUCHED (faithful);
# only data swapped to our vuln-only N-class split + cwe_label_map regenerated. Classification only.
#
# !!! v1 — may iterate. Paste errors. !!!
# Pipeline: clone VulExplainer -> download megavul split -> adapter (parquet -> big_vul csv + label_map)
#   -> build tree-sitter parser (.so) -> train CNN teacher -> soft-distill GraphCodeBERT student (train+
#   test in one) -> upload logs + student model.
# NOTE: VulExplainer parses C with the C# tree-sitter parser (DFG_csharp) — their design, kept as-is.
# Paper config shipped in the .sh (teacher 50ep/bs128/5e-3, student 50ep/bs8/2e-5/alpha0.7) — unchanged.
# Usage (pod, project root):  nohup bash scripts/run_vulexplainer_megavul.sh > vulexp.log 2>&1 &
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
WORK="$PWD"; VPROOT="$WORK/src/VulExplainer"
VARIANT="$VPROOT/VulExplainer/VulExplainer_GraphCodeBERT"
BIGVUL="$VPROOT/data/big_vul"; SPLIT="$WORK/megavul_ml1024/linevd"
RUN_ID="vulexplainer_megavul_$(date +%Y%m%d_%H%M%S)"

# EVAL_ONLY=1 -> skip teacher+student TRAIN: restore saved checkpoints, run student --do_test only,
# dump preds, recompute macro+weighted+acc. Needs WEIGHTS_TAR=<run>_weights.tar.gz. [VERIFY on pod]
EVAL_ONLY="${EVAL_ONLY:-}"; WEIGHTS_TAR="${WEIGHTS_TAR:-}"
if [[ -n "$EVAL_ONLY" && -z "$WEIGHTS_TAR" ]]; then echo "ERR: EVAL_ONLY=1 needs WEIGHTS_TAR=<...>_weights.tar.gz"; exit 1; fi

echo "=== [1/7] deps + VulExplainer repo ==="
VENV="/workspace/vulexp_env"
BASEPY=/venv/main/bin/python; [[ -x "$BASEPY" ]] || BASEPY="$(command -v python3 || command -v python)"
[[ -d "$VENV" ]] || "$BASEPY" -m venv "$VENV"
source "$VENV/bin/activate"
# torch 2.7 cu128 for Blackwell (sm_120); transformers <4.40 (teacher uses transformers.AdamW, removed
# later); tree-sitter <0.22 (Language.build_library + Language(path,name) API the parser code needs);
# numpy <2 (np.bool_ + transformers 4.35). Real GPU-op check catches "no kernel image".
python -c "import torch,transformers,sklearn,pandas,numpy,tree_sitter,pyarrow; v=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])); assert torch.cuda.is_available() and v>=(2,6); torch.zeros(2,device='cuda').sum().item()" 2>/dev/null || {
  pip install -q --force-reinstall torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -q "transformers==4.35.2" "tree-sitter>=0.20,<0.22" "numpy==1.26.4" scikit-learn pandas pyarrow tqdm
}
[[ -f "$VARIANT/student_graphcodebert_main.py" ]] || { rm -rf "$VPROOT"; git clone --depth 1 https://github.com/awsm-research/VulExplainer.git "$VPROOT"; }
[[ -f "$VARIANT/student_graphcodebert_main.py" ]] || { echo "ERR: variant not found at $VARIANT"; exit 1; }

echo "=== [2/7] data: megavul split ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar --no-same-owner -xzf "$DATA_TAR"
fi

echo "=== [3/7] adapter: megavul -> big_vul csv + cwe_label_map ==="
mkdir -p "$BIGVUL"
python scripts/vulexplainer_prepare_megavul.py --in-dir "$SPLIT" --out-dir "$BIGVUL"
NC=$(python -c "import pickle;print(len(pickle.load(open('$BIGVUL/cwe_label_map.pkl','rb'))))")
echo "  num_classes=$NC  (data at $BIGVUL)"

echo "=== [4/7] build tree-sitter parser (.so) — C parsed via C# grammar (their design) ==="
command -v gcc >/dev/null || (apt-get update -q && apt-get install -y -q build-essential)
cd "$VARIANT/parser"
rm -f my-languages.so   # shipped .so is author's platform — rebuild for this pod
# code ONLY uses the C# parser (parsers["c_sharp"]) -> build c-sharp ONLY. build.py's full 7-lang list
# breaks on newer tree-sitter-php (parser moved to php/src/parser.c, not src/parser.c).
# grammar must be ABI<=14 for tree-sitter python 0.21 (supports 13-14); latest main = ABI 15 -> crash.
# pin v0.21.0 (ABI 14). force re-clone so a stale ABI-15 clone from a prior run is replaced.
rm -rf tree-sitter-c-sharp
git clone --depth 1 --branch v0.21.0 https://github.com/tree-sitter/tree-sitter-c-sharp
python - <<'PY'
from tree_sitter import Language
Language.build_library("my-languages.so", ["tree-sitter-c-sharp"])
print("built my-languages.so (c_sharp)")
PY
[[ -f my-languages.so ]] || { echo "ERR: parser my-languages.so build failed"; exit 1; }
cd "$WORK"

# single-GPU pod: repo hardcodes cuda:1 in both mains -> use cuda:0
sed -i 's/cuda:1/cuda:0/g' "$VARIANT/teacher_main.py" "$VARIANT/student_graphcodebert_main.py"
# teacher + student dump preds to ./raw_prediction/ (CWD=$VARIANT) but never mkdir it -> crash at test
mkdir -p "$VARIANT/raw_prediction"
# dump student test predictions (y_true,y_pred) so compute_baseline_metrics can report macro+weighted
# F1 + accuracy (student test() logs only accuracy). Injected after `acc = accuracy_score(...)` in test().
sed -i '/    acc = accuracy_score(y_trues, y_preds)/a\    __import__("pandas").DataFrame({"y_true": y_trues, "y_pred": y_preds}).to_csv(__import__("os").environ["VULEXP_PRED_CSV"], index=False)' \
  "$VARIANT/student_graphcodebert_main.py"

OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
export VULEXP_PRED_CSV="$OUT/vulexp_cls_preds.csv"   # student test() dumps preds here (patched above)

if [[ -n "$EVAL_ONLY" ]]; then
  echo "=== [5/7] EVAL_ONLY: restore teacher+student weights $WEIGHTS_TAR ==="
  # weights tar = the checkpoint-best-acc/ dir (cnnteacher.bin + soft_distil_model_*.bin)
  rclone copy "$REMOTE/checkpoints/baselines/$WEIGHTS_TAR" "$VARIANT/saved_models/" --progress
  tar -I "$(command -v pigz || echo gzip)" -xf "$VARIANT/saved_models/$WEIGHTS_TAR" -C "$VARIANT/saved_models" && rm -f "$VARIANT/saved_models/$WEIGHTS_TAR"
  SBIN=$(find "$VARIANT/saved_models/checkpoint-best-acc" -name "soft_distil_model_*.bin" | head -1)
  [[ -n "$SBIN" ]] || { echo "ERR: no student soft_distil_model_*.bin in restored weights"; exit 1; }
  MNAME=$(basename "$SBIN")
  echo "  student checkpoint: $MNAME"
  echo "=== [6/7] student --do_test only (loads teacher + student, no training) ==="
  ( cd "$VARIANT" && python student_graphcodebert_main.py --alpha 0.7 --output_dir=./saved_models \
      --model_name="$MNAME" --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
      --train_data_file=../../data/big_vul/train.csv --eval_data_file=../../data/big_vul/val.csv --test_data_file=../../data/big_vul/test.csv \
      --do_test --block_size 512 --eval_batch_size 8 --seed 123456 ) 2>&1 | tee "$OUT/eval_soft_distil.log"
else
  echo "=== [5/7] train CNN teacher (50ep bs128 5e-3) ==="
  ( cd "$VARIANT" && bash train_teacher.sh ) ; cp -f "$VARIANT"/train_cnn_teacher.log "$OUT/" 2>/dev/null || true
  [[ -f "$VARIANT/saved_models/checkpoint-best-acc/cnnteacher.bin" ]] || { echo "ERR: teacher cnnteacher.bin not produced"; exit 1; }

  echo "=== [6/7] soft-distill GraphCodeBERT student (train+test, 50ep bs8 2e-5 alpha0.7) ==="
  ( cd "$VARIANT" && bash soft_distillation.sh ) ; cp -f "$VARIANT"/train_soft_distil*.log "$OUT/" 2>/dev/null || true
fi

# recompute macro + weighted + accuracy from the dumped predictions (student logs only accuracy)
if [[ -f "$VULEXP_PRED_CSV" ]]; then
  PYTHONPATH=src python scripts/compute_baseline_metrics.py --name VulExplainer \
    --classification "$VULEXP_PRED_CSV" --out "$OUT/vulexplainer_recomputed_metrics.json" 2>&1 | tee "$OUT/recomputed_metrics.log"
fi

echo "=== [7/7] upload: results -> results/baselines, weights -> checkpoints/baselines ==="
COMP="$(command -v pigz || echo gzip)"
# results = logs only (test metrics in train_soft_distil log) -> results/baselines/
tar -I "$COMP" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
# weights = teacher + student best checkpoints -> checkpoints/baselines/ (skip on EVAL_ONLY, just restored)
if [[ -z "$EVAL_ONLY" && -d "$VARIANT/saved_models/checkpoint-best-acc" ]]; then
  tar -I "$COMP" -cf "${RUN_ID}_weights.tar.gz" -C "$VARIANT/saved_models" checkpoint-best-acc && \
    rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress 2>/dev/null || true
fi
echo "DONE: $RUN_ID  results -> results/baselines, weights -> checkpoints/baselines  (metrics in $OUT/train_soft_distil*.log)"
