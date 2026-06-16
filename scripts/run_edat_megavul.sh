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
echo "  patched paths:"; grep -nE "pretrained_model_path|_path = r|use_pgd = |batch_size = |num_epochs = " "$T" | head

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
