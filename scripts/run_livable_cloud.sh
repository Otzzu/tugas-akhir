#!/usr/bin/env bash
# run_livable_cloud.sh — train LIVABLE (multiclass CWE, long-tail) on OUR MegaVul split.
# LIVABLE = DGL GGNN + seq2seq + adaptive long-tail re-weighting (src/LIVABLE).
# Pipeline: our .c funcs -> compiled old-Joern (zenodo 7607623) -> Devign-style builder
# ori_ourdevign+token.py -> GGNNinput JSON -> code/main_sta.py train.
#
# STATUS: SKELETON. Like run_linevd_cloud.sh this will need 1-2 pod iterations on the exact
# Joern parse command + builder csv-dir layout + main_sta.py CLI. Stages marked [VERIFY].
#
# Use a NON-Blackwell GPU pod (DGL caps torch<=2.4 = sm<=90; same wall as LineVD).
# Usage (pod, project root):  bash scripts/run_livable_cloud.sh
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"   # our exported split (func_before, vul, cwe_name, flaw_lines)
RUN_ID="livable_megavul_$(date +%Y%m%d_%H%M%S)"
WORK="$PWD"
LV="$WORK/src/LIVABLE"
PREP="$WORK/megavul_livable"

echo "=== [1/7] LIVABLE present (vendored) ==="
[[ -d "$LV" ]] || git clone --depth 1 https://github.com/LIVABLE01/LIVABLE.git "$LV"

echo "=== [2/7] venv: DGL(cuda) + torch 2.4.1 + deps ==="
# Reuse the LineVD recipe: cuda dgl (cu121 wheel), torch 2.4.1, plus LIVABLE's deps.
VENV="/workspace/livable_env"
[[ -d "$VENV" ]] || /venv/main/bin/python -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
if ! python -c "import torch,dgl" 2>/dev/null; then
  pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -q dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
  pip install -q --upgrade --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
fi
pip install -q gensim nltk graphviz jsonlines yacs pandas scikit-learn tqdm fastparquet
python -c "import nltk; nltk.download('punkt', quiet=True)" || true
python -c "import torch,dgl; print('torch',torch.__version__,'dgl',dgl.__version__,'cuda',torch.cuda.is_available())"

echo "=== [3/7] data: our split + LIVABLE adapter (.c files + jsonl + cwe_labels) ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar -xzf "$DATA_TAR"
fi
# --keep-benign: 26-class (benign + 25 CWE), == our model + LineVD/LineVul (only LOSVER is vuln-only).
python "$WORK/scripts/livable_prepare_megavul.py" --in-dir "$WORK/megavul_ml1024/linevd" --out-dir "$PREP" --keep-benign
NUM_CLASSES=$(python -c "import json;print(json.load(open('$PREP/cwe_labels.json'))['num_classes'])")
echo "  num_classes=$NUM_CLASSES"

echo "=== [4/7] compiled Joern (zenodo 7607623) + parse each function ==="
# zenodo joern.zip is double-nested: extracts to preprocessing/joern/{slicer.sh,process.py,joern/}
# where preprocessing/joern/joern is the joern-lang repo root (joern-parse at its joern/joern-parse).
# slicer.sh's `./joern/joern-parse` only resolves run FROM that repo root, so we copy slicer.sh +
# process.py in, chmod +x the binaries, and run there. joern-parse writes <file>/tmp/<file>/nodes.csv
# (keyed by filename -> no collision; the slicer `rm -rf parsed` is harmless). We collect per split.
command -v java >/dev/null || (apt-get update -q && apt-get install -y -q default-jre)
if [[ ! -f "$LV/preprocessing/joern/slicer.sh" ]]; then
  echo "  fetching compiled joern (zenodo 7607623, ~95MB) ..."
  wget -q "https://zenodo.org/api/records/7607623/files/joern.zip/content" -O /tmp/joern.zip
  unzip -q -o /tmp/joern.zip -d "$LV/preprocessing/" && rm -f /tmp/joern.zip
fi
JD="$LV/preprocessing/joern/joern"   # joern-lang repo root
cp -f "$LV/preprocessing/joern/slicer.sh" "$LV/preprocessing/joern/process.py" "$JD/"
find "$JD" -type f \( -name joern-parse -o -name "*.sh" -o -name joern \) -exec chmod +x {} \;
# Preprocess cache (joern_csv + ggnn_input + wv) — parsing 9116 funcs is the slow part; restore
# from Drive on a fresh pod to skip straight to train (like LineVD's joern graph cache).
PREP_CACHE="livable_megavul_preprocess.tar.gz"
PREP_RESTORED=""
if [[ ! -d "$PREP/ggnn_input" ]] && rclone ls "$REMOTE/data/baselines/$PREP_CACHE" >/dev/null 2>&1; then
  echo "  restoring LIVABLE preprocess cache from Drive (skips joern parse + builder) ..."
  rclone copy "$REMOTE/data/baselines/$PREP_CACHE" /tmp/ --progress
  mkdir -p "$PREP"; tar -xzf "/tmp/$PREP_CACHE" -C "$PREP" && rm -f "/tmp/$PREP_CACHE"
  PREP_RESTORED=1
fi
HAVE=$( (ls -d "$PREP"/joern_csv/test/*.c 2>/dev/null || true) | wc -l )
if [[ "$HAVE" -lt 100 ]]; then
  cd "$JD"
  for split in train val test; do
    rm -rf "$JD"/*.c 2>/dev/null || true            # clear prev split's parse dirs from cwd
    python process.py --file_path "$PREP/functions/$split" --start 0 --end 100000 || true
    mkdir -p "$PREP/joern_csv/$split"
    for d in "$JD"/*.c; do [[ -d "$d" ]] && mv "$d" "$PREP/joern_csv/$split/" 2>/dev/null || true; done
    echo "  $split: $(ls -d "$PREP"/joern_csv/$split/*.c 2>/dev/null | wc -l) parsed"
  done
fi

echo "=== [5/7] word2vec + GGNNinput builder ==="
if [[ ! -d "$PREP/ggnn_input" ]]; then
  cd "$LV/preprocessing"   # builder imports my_tokenizer from local utils.py
  # word2vec on our func tokens (128-d, matches LIVABLE node feature size).
  python word2vec_multi.py --data_paths "$PREP/train.jsonl" \
    --save_model_dir "$PREP/wv" --model_name wvmodel --embedding_size 128
  # builder's --csv/--json_files have no nargs -> patch to accept multiple paths
  sed -i "s/'--csv', help/'--csv', nargs='+', help/" "ori_ourdevign+token.py"
  sed -i "s/'--json_files', help/'--json_files', nargs='+', help/" "ori_ourdevign+token.py"
  # Builder: csv dirs (joern out) + our jsonls + wv -> GGNNinput JSON
  # (--csv/--json_files order is train,test,valid). Output: ggnn_input/diverse-{train,test,valid}-v0.json
  python "ori_ourdevign+token.py" \
    --csv "$PREP/joern_csv/train" "$PREP/joern_csv/test" "$PREP/joern_csv/val" \
    --json_files "$PREP/train.jsonl" "$PREP/test.jsonl" "$PREP/val.jsonl" \
    --wv "$PREP/wv/wvmodel" \
    --output_dir "$PREP/ggnn_input"
fi
# Cache the preprocess to Drive if we built it (skip when restored) so future pods skip the parse.
if [[ -z "$PREP_RESTORED" && -d "$PREP/ggnn_input" ]] && ! rclone ls "$REMOTE/data/baselines/$PREP_CACHE" >/dev/null 2>&1; then
  echo "  caching LIVABLE preprocess to Drive ..."
  COMP=(gzip); command -v pigz >/dev/null && COMP=(pigz -p "$(nproc)")
  tar -cf - -C "$PREP" joern_csv ggnn_input wv 2>/dev/null | "${COMP[@]}" > "/tmp/$PREP_CACHE"
  rclone copy "/tmp/$PREP_CACHE" "$REMOTE/data/baselines/" --progress && rm -f "/tmp/$PREP_CACHE"
fi

echo "=== [6/7] train (patch num_classes 31 -> $NUM_CLASSES, rename GGNNinput, run) ==="
# main_sta.py reads input_dir/multi-{train1,valid,test}-v0.json but builder writes
# diverse-{train,test,valid}-v0.json -> rename. And 31 is hardcoded in main_sta + both
# DevignModel MLPReadout output heads -> patch to our label space.
GG="$PREP/ggnn_input"
[[ -f "$GG/diverse-train-v0.json" ]] && mv -f "$GG/diverse-train-v0.json" "$GG/multi-train1-v0.json"
[[ -f "$GG/diverse-valid-v0.json" ]] && mv -f "$GG/diverse-valid-v0.json" "$GG/multi-valid-v0.json"
[[ -f "$GG/diverse-test-v0.json" ]]  && mv -f "$GG/diverse-test-v0.json"  "$GG/multi-test-v0.json"
rm -f "$GG"/multi_128_64batch_*.bin   # stale cached dataset binary (rebuild for our classes)
sed -i "s/from trainer_test import train/from trainer_sta import train/" "$LV/code/main_sta.py"
sed -i "s/self.graph.add_edge(/self.graph.add_edges(/" "$LV/code/data_loader/dataset.py"   # dgl 2.x rename
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = '1'/os.environ['CUDA_VISIBLE_DEVICES'] = '0'/" "$LV/code/main_sta.py"  # single-GPU pod
sed -i "s/^    num_classes = 31/    num_classes = $NUM_CLASSES/" "$LV/code/main_sta.py"
sed -i "s/MLPReadout(self.hidden_dim2, 31)/MLPReadout(self.hidden_dim2, $NUM_CLASSES)/" "$LV/code/modules/model.py"
sed -i "s/MLPReadout(2 \* self.seq_hid, 31)/MLPReadout(2 * self.seq_hid, $NUM_CLASSES)/" "$LV/code/modules/model.py"
# num_class_list is hardcoded 31-long -> the BalancedSoftmaxCE/dual-branch loss builds a
# 31-wide one-hot, mismatching our 26 outputs. Replace with OUR per-class train counts
# (also makes LIVABLE's long-tail re-weighting faithful to our distribution).
NCL=$(python - "$GG/multi-train1-v0.json" "$NUM_CLASSES" <<'PY'
import json, sys
from collections import Counter
data = json.load(open(sys.argv[1])); n = int(sys.argv[2]); c = Counter()
for e in data:
    c[int(e["targets"][0][0])] += 1
print("[" + ", ".join(str(c.get(i, 1)) for i in range(n)) + "]")
PY
)
echo "  num_class_list = $NCL"
sed -i "s/^    num_class_list = \[.*\]/    num_class_list = $NCL/" "$LV/code/main_sta.py"
cd "$LV/code"
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
python main_sta.py --input_dir "$GG" 2>&1 | tee "$OUT/train.log"

echo "=== [7/7] upload results ==="
cd "$WORK"
tar -czf "${RUN_ID}_results.tar.gz" -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
echo "DONE: $RUN_ID"
