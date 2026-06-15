#!/usr/bin/env bash
# run_vulpcl_cloud.sh — train VulPCL categorization on ITS OWN Linux-Kernel data (faithful repro).
# Uses the PREPROCESSED token pkls shipped in dataset.zip (vul_categorization/vul_data/linux/*) so
# NO ancient Neo4j-Gremlin joern is needed. Model = their CodeBert_Blstm (untouched). 11-class.
# STANDALONE reference number (their data is tokenized only -> NOT a head-to-head with our model).
#
# Patches (faithfulness = numerical/config only, model unchanged):
#   - linux_code_vocab.json MISSING -> derive n_vocab = max(FCDS id)+1 from the pkls.
#   - Config.num_classes 12 -> 11 (data is 11-class: labels 0..10).
#   - codebert_blstm target_names 0..11 -> 0..(NC-1).
#   - pkl path layout: codebert_blstm reads ./vul_data/<cwe>/<p>/ -> stage pkls there.
#
# dataset.zip: local data/VulPCL/dataset.zip, else rclone gdrive .../data/baselines/vulpcl_dataset.zip
# Usage (pod, project root):  bash scripts/run_vulpcl_cloud.sh
set -euo pipefail

REMOTE_BASE="gdrive-mesach:tugas-akhir/data/baselines"
WORK="$PWD"; VP="$WORK/src/VulPCL"; CAT="$VP/vul_categorization"
CWE="all"; PROJ="linux"
RUN_ID="vulpcl_linux_$(date +%Y%m%d_%H%M%S)"

# VulPCL repo is untracked in our git (not pulled) — ensure a complete clone (else module/ missing).
if [[ ! -f "$CAT/module/CodeBert_Blstm.py" ]]; then
  echo "  VulPCL incomplete/missing -> fresh clone"
  rm -rf "$VP"; git clone --depth 1 https://github.com/liucyy/VulPCL.git "$VP"
fi

echo "=== [1/6] venv + deps ==="
VENV="/workspace/vulpcl_env"
BASEPY=/venv/main/bin/python; [[ -x "$BASEPY" ]] || BASEPY="$(command -v python3 || command -v python)"
[[ -d "$VENV" ]] || "$BASEPY" -m venv "$VENV"
source "$VENV/bin/activate"
python -c "import torch,transformers,sklearn,pandas" 2>/dev/null || \
  pip install -q torch transformers scikit-learn numpy tqdm pandas

echo "=== [2/6] get categorization linux pkls (small ~130MB, not the full 1.9GB zip) ==="
PKL_DIR="$WORK/data/VulPCL/_cat_linux"; mkdir -p "$PKL_DIR"
SRC="dataset/vul_categorization/vul_data/linux"
if [[ ! -f "$PKL_DIR/train_set_token.pkl" ]]; then
  if [[ -f "$WORK/data/VulPCL/dataset.zip" ]]; then               # local full zip
    unzip -o "$WORK/data/VulPCL/dataset.zip" "$SRC/*" -d "$WORK/data/VulPCL/_ext" >/dev/null
    cp -f "$WORK/data/VulPCL/_ext/$SRC"/*.pkl "$PKL_DIR/"
  else                                                            # small tar from Drive
    rclone copy "$REMOTE_BASE/vulpcl_cat_linux.tar.gz" /tmp/ --progress
    tar --no-same-owner -xzf /tmp/vulpcl_cat_linux.tar.gz -C "$PKL_DIR"
  fi
fi

echo "=== [3/6] stage pkls at codebert_blstm's expected path ./vul_data/$CWE/$PROJ ==="
DEST="$CAT/vul_data/$CWE/$PROJ"; mkdir -p "$DEST"
cp -f "$PKL_DIR"/{train,val,test}_set_token.pkl "$DEST/"
mkdir -p "$CAT/save_dict/$PROJ/$CWE"   # codebert_blstm uses os.mkdir (no -p)

echo "=== [4/6] derive vocab (n_vocab = max FCDS id + 1) + class count ==="
read NV NC < <(python - "$DEST" "$CAT/${PROJ}_code_vocab.json" <<'PY'
import pickle, json, sys
base, vocab_out = sys.argv[1], sys.argv[2]
mx = 0; labs = set()
for s in ("train", "val", "test"):
    for it in pickle.load(open(f"{base}/{s}_set_token.pkl", "rb")):
        labs.add(it[4])
        if it[2]: mx = max(mx, max(it[2]))
nv = mx + 1; nc = len(labs)
json.dump({str(i): i for i in range(nv)}, open(vocab_out, "w"))   # len() == n_vocab is all codebert_blstm needs
print(nv, nc)
PY
)
echo "  n_vocab=$NV  num_classes=$NC"

echo "=== [5/6] patch model num_classes + target_names to $NC (faithful: config only) ==="
git -C "$VP" checkout -- vul_categorization/module/CodeBert_Blstm.py vul_categorization/codebert_blstm.py 2>/dev/null || true
sed -i "s/self.num_classes = 12/self.num_classes = $NC/" "$CAT/module/CodeBert_Blstm.py"
sed -i "s/target_names=\[[^]]*\]/target_names=[str(i) for i in range($NC)]/" "$CAT/codebert_blstm.py"
# single-GPU pod: repo hardcodes GPU 3 -> use 0.
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = '3'/os.environ['CUDA_VISIBLE_DEVICES'] = '0'/" "$CAT/codebert_blstm.py"
# forward returns the 768-d concat with the classifier (linear2) commented out -> wire it so
# cross_entropy gets [B,num_classes] logits, not features. (linear2 is defined; faithful intent.)
sed -i "s/^\(\s*\)return out$/\1return self.linear2(out)/" "$CAT/module/CodeBert_Blstm.py"
grep -q "self.linear2(out)" "$CAT/module/CodeBert_Blstm.py" || { echo "ERR: linear2 wire patch failed"; exit 1; }

echo "=== [6/6] train (their CodeBert_Blstm, untouched arch) ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
cd "$CAT"
python codebert_blstm.py --p "$PROJ" --cwe "$CWE" 2>&1 | tee "$OUT/train.log"
cd "$WORK"
tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "gdrive-mesach:tugas-akhir/results/baselines/" --progress 2>/dev/null || true
echo "DONE: $RUN_ID"
