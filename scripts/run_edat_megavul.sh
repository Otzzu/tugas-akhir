#!/usr/bin/env bash
# run_edat_megavul.sh — EDAT (VTP + LVD, GraphCodeBERT + MTL + EDAT/PGD) on OUR megavul split.
# EDAT = preprint (IST'25, not peer-reviewed) + near-scoop to our work -> ALTERNATE baseline; cite the
# CODE's behavior, not the paper's PGD claims. Text-based (NO joern). Best config = graphcodebert +
# context (上下文) + use_pgd=True. Comparable on BOTH tasks (classification + line-level).
#
# Pipeline: megavul -> edat_prepare_megavul (VTP + LVD jsonl) -> patch Config (model=graphcodebert-base,
# data paths=ours, use_pgd=True) -> multi_task_train_alternate.py -> upload results.
# Usage (pod, project root):  bash scripts/run_edat_megavul.sh
set -uo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
WORK="$PWD"; ED="$WORK/src/EDAT"
VD="$ED/EDAT-MLT/多任务/graphcodebert-上下文"     # best variant: GraphCodeBERT + context
OUTJ="$WORK/megavul_edat"; RUN_ID="edat_megavul_$(date +%Y%m%d_%H%M%S)"

echo "=== [1/6] deps + EDAT repo ==="
VENV="/workspace/edat_env"
BASEPY=/venv/main/bin/python; [[ -x "$BASEPY" ]] || BASEPY="$(command -v python3 || command -v python)"
[[ -d "$VENV" ]] || "$BASEPY" -m venv "$VENV"
source "$VENV/bin/activate"
# torch must be >=2.6 (transformers blocks torch.load of graphcodebert-base's .bin below that,
# CVE-2025-32434) AND have kernels for the pod GPU. Blackwell (sm_120) needs cu128 -> torch 2.7.0 cu128
# (supports Blackwell + Ada/Hopper, runtime 12.8 <= driver 12.8). The check runs a real GPU op so it
# catches "no kernel image" (cuda.is_available() alone returns True even when kernels are missing).
python -c "import torch,transformers,sklearn,pandas,fastparquet,matplotlib,tree_sitter_c; v=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])); assert torch.cuda.is_available() and v>=(2,6); torch.zeros(2,device='cuda').sum().item()" 2>/dev/null || {
  pip install -q --force-reinstall torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -q transformers scikit-learn numpy tqdm pandas fastparquet matplotlib tree-sitter tree-sitter-c
}
[[ -f "$VD/multi_task_train_alternate.py" ]] || { rm -rf "$ED"; git clone --depth 1 https://github.com/Karelye/EDAT-MLT.git "$ED"; }
[[ -f "$VD/multi_task_train_alternate.py" ]] || { echo "ERR: EDAT variant not found at $VD (check clone/path)"; exit 1; }

echo "=== [2/6] data: megavul split ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar --no-same-owner -xzf "$DATA_TAR"
fi

echo "=== [3/6] adapter: megavul -> EDAT VTP + LVD jsonl ==="
python scripts/edat_prepare_megavul.py --in-dir "$WORK/megavul_ml1024/linevd" --out-dir "$OUTJ"

echo "=== [4/6] patch Config: model=graphcodebert-base, data=ours, use_pgd=True ==="
T="$VD/multi_task_train_alternate.py"
git -C "$ED" checkout -- "EDAT-MLT/多任务/graphcodebert-上下文/multi_task_train_alternate.py" 2>/dev/null || true
sed -i "s#pretrained_model_path = r\"[^\"]*\"#pretrained_model_path = r\"microsoft/graphcodebert-base\"#" "$T"
sed -i "s#classification_train_path = r\"[^\"]*\"#classification_train_path = r\"$OUTJ/train.jsonl\"#" "$T"
sed -i "s#classification_test_path = r\"[^\"]*\"#classification_test_path = r\"$OUTJ/test.jsonl\"#" "$T"
sed -i "s#classification_valid_path = r\"[^\"]*\"#classification_valid_path = r\"$OUTJ/val.jsonl\"#" "$T"
sed -i "s#line_level_train_path = r\"[^\"]*\"#line_level_train_path = r\"$OUTJ/train_line_level.jsonl\"#" "$T"
sed -i "s#line_level_valid_path = r\"[^\"]*\"#line_level_valid_path = r\"$OUTJ/val_line_level.jsonl\"#" "$T"
sed -i "s#line_level_test_path = r\"[^\"]*\"#line_level_test_path = r\"$OUTJ/test_line_level.jsonl\"#" "$T"
sed -i "s#use_pgd = False#use_pgd = True#" "$T"
sed -i "s/pgd_epsilon = 0.03/pgd_epsilon = 0.02/" "$T"   # paper ϵ=0.02 (code ships 0.03)
# shipped epoch_alternate_order = [classification, classification] -> VTP-only, LVD NEVER runs!
# Set true MTL alternation so both tasks train (epoch cls, lvd, cls, lvd... = 10 VTP + 10 LVD / 20 ep).
sed -i 's/epoch_alternate_order = \[.*\]/epoch_alternate_order = ["classification", "line_level"]/' "$T"
# PAPER best config: AdamW(✓ already), batch 32, ϵ0.02, lr 1e-5, epoch 20, PGD on, GraphCodeBERT+ctx.
# batch 32 OOMs a 16GB card (~24GB needed) + grad-accum not implemented -> default 16 here; set
# EDAT_BS=32 on a >=24GB GPU for the EXACT paper batch. code default 2 = impractically slow.
sed -i "s/batch_size = 2$/batch_size = ${EDAT_BS:-16}/" "$T"
[[ -n "${EDAT_EPOCHS:-}" ]] && sed -i "s/num_epochs = 20$/num_epochs = ${EDAT_EPOCHS}/" "$T"
# patience=3 was tuned for the shipped VTP-only [cls,cls] order; under true cls<->lvd alternation the
# lvd epochs worsen the tracked classification val-loss -> waste patience -> premature stop (~12 ep,
# undertrained cls). Run the paper's full 20 epochs: raise patience (EDAT_PATIENCE, default 20 = ~off).
sed -i "s/^\(\s*\)patience = 3$/\1patience = ${EDAT_PATIENCE:-20}/" "$T"
echo "  patched paths:"; grep -nE "pretrained_model_path|_path = r|use_pgd = |batch_size = |num_epochs = |patience = " "$T" | head

echo "=== [5/7] train (skip if best model already exists) — GraphCodeBERT + context + MTL + PGD ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
EXIST=$(find "$VD" -maxdepth 2 -name "best_multi_task_model.pt" 2>/dev/null | head -1)
if [[ -n "$EXIST" && -z "${EDAT_FORCE_TRAIN:-}" ]]; then
  echo "  trained model exists ($EXIST) -> skip train (set EDAT_FORCE_TRAIN=1 to retrain)"
else
  ( cd "$VD" && python multi_task_train_alternate.py 2>&1 | tee "$OUT/train.log" )
fi

echo "=== [6/7] TEST eval (their multi_task_evaluate.py: VTP acc/F1 + LVD IFA/Top-k) ==="
OUTDIR=$(find "$VD" -maxdepth 1 -type d -name "multi_task_alternate_output*" | head -1)
[[ -n "$OUTDIR" ]] || { echo "ERR: no train output dir (train must run first)"; exit 1; }
E="$VD/multi_task_evaluate.py"; BASE=$(basename "$OUTDIR")
git -C "$ED" checkout -- "EDAT-MLT/多任务/graphcodebert-上下文/multi_task_evaluate.py" "EDAT-MLT/多任务/graphcodebert-上下文/data.py" 2>/dev/null || true
# point eval at OUR trained model + data (expert_num 6 / expert_dim 768 already match training).
sed -i "s#output_dir = \"multi_task_alternate_output_classification\"#output_dir = \"$BASE\"#" "$E"
sed -i 's#"multi_task_model_epoch_3.pt"#"best_multi_task_model.pt"#' "$E"
sed -i "s#pretrained_model_path = r\"[^\"]*\"#pretrained_model_path = r\"microsoft/graphcodebert-base\"#" "$E"
sed -i "s#classification_train_path = r\"[^\"]*\"#classification_train_path = r\"$OUTJ/train.jsonl\"#" "$E"
sed -i "s#classification_test_path = r\"[^\"]*\"#classification_test_path = r\"$OUTJ/test.jsonl\"#" "$E"
sed -i "s#line_level_test_path = r\"[^\"]*\"#line_level_test_path = r\"$OUTJ/test_line_level.jsonl\"#" "$E"
# numpy>=2 compat: get_line_level_metrics does float() on a [N,1] array element -> ravel first (same values)
sed -i 's#\[float(val) for val in list(line_score)\]#[float(val) for val in np.asarray(line_score).ravel()]#' "$VD/data.py"
sed -i 's#\[float(val) for val in list(han_line_score)\]#[float(val) for val in np.asarray(han_line_score).ravel()]#' "$VD/data.py"
( cd "$VD" && python multi_task_evaluate.py 2>&1 | tee "$OUT/eval.log" )

echo "=== [7/7] upload: results -> results/baselines, weights -> checkpoints/baselines ==="
cd "$WORK"
COMP="$(command -v pigz || echo gzip)"
# results = logs + eval metrics + configs + plots (NO model) -> results/baselines/
cp -f "$OUTDIR"/*.json "$OUTDIR"/*.png "$OUT/" 2>/dev/null || true          # eval json + configs + plot
[[ -f "$OUT/train.log" ]] || cp -f "$(ls -t "$WORK"/baseline_runs/edat_megavul_*/train.log 2>/dev/null | head -1)" "$OUT/" 2>/dev/null || true
tar -I "$COMP" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
# weights = best model -> checkpoints/baselines/ (find anywhere under $VD; verbose)
WMODEL=$(find "$VD" -name "best_multi_task_model.pt" 2>/dev/null | head -1)
if [[ -n "$WMODEL" ]]; then
  echo "  weights: $WMODEL -> checkpoints/baselines"
  tar -I "$COMP" -cf "${RUN_ID}_weights.tar.gz" -C "$(dirname "$WMODEL")" best_multi_task_model.pt && \
    rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress 2>/dev/null || true
else
  echo "  WARN: best_multi_task_model.pt not found under $VD -> no weights uploaded"
fi
echo "DONE: $RUN_ID  results -> results/baselines, weights -> checkpoints/baselines  (metrics in $OUT/eval.log)"
