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
# [VERIFY] LIVABLE ships a compiled Joern at https://zenodo.org/record/7607623 ; preprocessing
# starts from preprocessing/process.py (runs slicer.sh per .c). Confirm the download URL +
# that process.py emits nodes.csv/edges.csv under the dir layout the builder expects
# (args.csv/<file>/tmp/<file>/nodes.csv).
command -v java >/dev/null || (apt-get update -q && apt-get install -y -q default-jre)
if [[ ! -f "$LV/preprocessing/slicer.sh" ]]; then
  echo "  fetching compiled joern (zenodo 7607623, ~95MB) ..."
  wget -q "https://zenodo.org/api/records/7607623/files/joern.zip/content" -O /tmp/joern.zip
  unzip -q -o /tmp/joern.zip -d "$LV/preprocessing/" && rm -f /tmp/joern.zip
fi
cd "$LV/preprocessing"
echo "  preprocessing/ contents (confirm slicer.sh + joern layout):"; ls
for split in train valid test; do
  s=$split; [[ "$split" == valid ]] && s=val
  # process.py runs ./slicer.sh per .c -> parsed/<file>/ (nodes.csv, edges.csv)
  python process.py --file_path "$PREP/functions/$s" --start 0 --end 100000 || true
done

echo "=== [5/7] word2vec + GGNNinput builder ==="
# [VERIFY] word2vec_multi.py CLI to train the wv model on our tokens.
python word2vec_multi.py || true   # produces wv model -> set --wv below
# Builder: csv dirs (Joern out) + our jsonls + wv -> GGNNinput JSON
python "ori_ourdevign+token.py" \
  --csv "$PREP/joern_csv/train" "$PREP/joern_csv/test" "$PREP/joern_csv/val" \
  --json_files "$PREP/train.jsonl" "$PREP/test.jsonl" "$PREP/val.jsonl" \
  --wv "$LV/preprocessing/wv_model" \
  --output_dir "$PREP/ggnn_input" || true

echo "=== [6/7] train (set num_classes to our label space) ==="
cd "$LV/code"
# [VERIFY] main_sta.py reads GGNNinput JSON; set num_classes=$NUM_CLASSES (it hardcodes 31)
# + num_class_list from our per-class counts. Patch via sed or a small config before train.
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
python main_sta.py 2>&1 | tee "$OUT/train.log"

echo "=== [7/7] upload results ==="
cd "$WORK"
tar -czf "${RUN_ID}_results.tar.gz" -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
echo "DONE: $RUN_ID"
