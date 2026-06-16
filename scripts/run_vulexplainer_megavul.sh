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
set -uo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
WORK="$PWD"; VPROOT="$WORK/src/VulExplainer"
VARIANT="$VPROOT/VulExplainer/VulExplainer_GraphCodeBERT"
BIGVUL="$VPROOT/data/big_vul"; SPLIT="$WORK/megavul_ml1024/linevd"
RUN_ID="vulexplainer_megavul_$(date +%Y%m%d_%H%M%S)"

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
for r in go javascript python php java ruby c-sharp; do
  [[ -d "tree-sitter-$r" ]] || git clone --depth 1 "https://github.com/tree-sitter/tree-sitter-$r"
done
python build.py
[[ -f my-languages.so ]] || { echo "ERR: parser my-languages.so build failed"; exit 1; }
cd "$WORK"

echo "=== [5/7] train CNN teacher (50ep bs128 5e-3) ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
( cd "$VARIANT" && bash train_teacher.sh ) ; cp -f "$VARIANT"/train_cnn_teacher.log "$OUT/" 2>/dev/null || true
[[ -f "$VARIANT/saved_models/checkpoint-best-acc/cnnteacher.bin" ]] || { echo "ERR: teacher cnnteacher.bin not produced"; exit 1; }

echo "=== [6/7] soft-distill GraphCodeBERT student (train+test, 50ep bs8 2e-5 alpha0.7) ==="
( cd "$VARIANT" && bash soft_distillation.sh ) ; cp -f "$VARIANT"/train_soft_distil*.log "$OUT/" 2>/dev/null || true

echo "=== [7/7] upload logs + student model ==="
COMP="$(command -v pigz || echo gzip)"
# trained student model dir (best checkpoint)
[[ -d "$VARIANT/saved_models" ]] && cp -rf "$VARIANT/saved_models/checkpoint-best-acc" "$OUT/student_ckpt" 2>/dev/null || true
tar -I "$COMP" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
echo "DONE: $RUN_ID  (test metrics in $OUT/train_soft_distil*.log)"
