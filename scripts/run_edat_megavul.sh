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
python -c "import torch,transformers,sklearn,pandas,numpy,fastparquet,matplotlib,seaborn" 2>/dev/null || \
  pip install -q torch transformers scikit-learn numpy tqdm pandas fastparquet matplotlib seaborn
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
echo "  patched paths:"; grep -nE "pretrained_model_path|_path = r|use_pgd = " "$T" | head

echo "=== [5/6] train (GraphCodeBERT + context + MTL + PGD) ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
( cd "$VD" && python multi_task_train_alternate.py 2>&1 | tee "$OUT/train.log" )

echo "=== [6/6] upload results ==="
cd "$WORK"
COMP="$(command -v pigz || echo gzip)"
tar -I "$COMP" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . 2>/dev/null && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
# trained model / EDAT output dir (only if train produced one — else skip, no spurious rclone error)
MODELDIR=$(find "$VD" -maxdepth 1 -type d -name "*output*" | head -1)
if [[ -n "$MODELDIR" ]]; then
  tar -I "$COMP" -cf "${RUN_ID}_model.tar.gz" -C "$VD" "$(basename "$MODELDIR")" 2>/dev/null && \
    rclone copy "${RUN_ID}_model.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
fi
echo "DONE: $RUN_ID"
