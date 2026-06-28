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
# Usage (pod, project root):  bash scripts/run_livable_cloud.sh [--vuln-only]
#   --vuln-only: 25-class vuln-only (drop benign), faithful to LIVABLE paper (group w/ LOSVER).
#                Reuses the SAME joern/ggnn cache (label-independent) -> just filters+relabels.
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"   # our exported split (func_before, vul, cwe_name, flaw_lines)
TS="$(date +%Y%m%d_%H%M%S)"
VULN_ONLY=""; OPT="adamw"; LR="1e-3"   # README pt6 says RAdam 1e-4 but it FREEZES on our data
                                        # (loss flat, acc 0); AdamW 1e-3 = repo's shipped default + trains.
MODE="megavul"; TOPCWE=31              # --bigvul = faithful repro on LIVABLE's OWN Big-Vul (all_vul.csv),
                                        # top-31 CWE (incl Nonetype) == their hardcoded class_num=31.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vuln-only) VULN_ONLY=1;;
    --bigvul)    MODE="bigvul";;
    --top-cwe)   TOPCWE="$2"; shift;;
    --opt) OPT="$2"; shift;;
    --lr)  LR="$2"; shift;;
    *) echo "unknown arg: $1";;
  esac; shift
done
WORK="$PWD"
LV="$WORK/src/LIVABLE"
SEED="${SEED:-10}"   # multi-seed: patched into main_sta.py below; default 10 = LIVABLE original
if [[ "$MODE" == "bigvul" ]]; then
  PREP="$WORK/bigvul_livable"; PREP_CACHE="livable_bigvul_preprocess.tar.gz"
  RUN_ID="livable_bigvul_${TS}_top${TOPCWE}_${OPT}${LR}"
else
  PREP="$WORK/megavul_livable"; PREP_CACHE="livable_megavul_preprocess.tar.gz"
  RUN_ID="livable_megavul_s${SEED}_${TS}"
  [[ -n "$VULN_ONLY" ]] && RUN_ID="${RUN_ID}_vo"
  RUN_ID="${RUN_ID}_${OPT}${LR}"
fi

echo "=== [1/7] LIVABLE present (vendored) ==="
[[ -d "$LV" ]] || git clone --depth 1 https://github.com/LIVABLE01/LIVABLE.git "$LV"

echo "=== [2/7] venv: DGL(cuda) + torch 2.4.1 + deps ==="
# Reuse the LineVD recipe: cuda dgl (cu121 wheel), torch 2.4.1, plus LIVABLE's deps.
VENV="/workspace/livable_env"
# base python differs per pod image: prefer RunPod's /venv/main, else system python3.
BASEPY=/venv/main/bin/python
[[ -x "$BASEPY" ]] || BASEPY="$(command -v python3 || command -v python || true)"
[[ -n "$BASEPY" ]] || { echo "ERR: no python on this pod (install python3)"; exit 1; }
echo "  base python: $BASEPY ($($BASEPY --version 2>&1))"
# --system-site-packages only for RunPod's clean /venv/main (reuses its torch). A plain system
# python3 can have a broken system dist (version=None) that crashes pip's resolver -> isolate it
# (we install torch/dgl explicitly below regardless, so nothing is lost).
VENV_FLAGS="--system-site-packages"
[[ "$BASEPY" == /venv/main/bin/python ]] || VENV_FLAGS=""
[[ -d "$VENV" ]] || "$BASEPY" -m venv "$VENV" $VENV_FLAGS
source "$VENV/bin/activate"
if ! python -c "import torch,dgl" 2>/dev/null; then
  pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -q dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
  pip install -q --upgrade --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
fi
pip install -q gensim nltk graphviz jsonlines yacs pandas scikit-learn tqdm fastparquet
python -c "import nltk; nltk.download('punkt', quiet=True)" || true
python -c "import torch,dgl; print('torch',torch.__version__,'dgl',dgl.__version__,'cuda',torch.cuda.is_available())"

echo "=== [3/7] data + LIVABLE adapter (.c files + jsonl + cwe_labels) [mode=$MODE] ==="
if [[ "$MODE" == "bigvul" ]]; then
  # LIVABLE's OWN Big-Vul (all_vul.csv): rclone from our Drive, else gdown the public link.
  CSV="$WORK/data/LIVABLE/all_vul.csv"
  if [[ ! -f "$CSV" ]]; then
    mkdir -p "$WORK/data/LIVABLE"
    rclone copy "$REMOTE/data/baselines/all_vul.csv" "$WORK/data/LIVABLE/" --progress 2>/dev/null || true
    [[ -f "$CSV" ]] || { pip install -q gdown && gdown 1j8WUSNte6Tda2uXR8Ue3Hk9jRa-33LGL -O "$CSV"; }
  fi
  [[ -f "$CSV" ]] || { echo "ERR: all_vul.csv missing (upload to $REMOTE/data/baselines/ or check gdown)"; exit 1; }
  python "$WORK/scripts/livable_prepare_bigvul.py" --csv "$CSV" --out-dir "$PREP" --top-cwe "$TOPCWE"
else
  if [[ ! -d megavul_ml1024 ]]; then
    rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar --no-same-owner -xzf "$DATA_TAR"
  fi
  # Always --keep-benign here (26-class ggnn) so ONE joern/ggnn cache serves both modes; --vuln-only
  # filters benign out of the built ggnn downstream (label-independent cache, no re-parse).
  python "$WORK/scripts/livable_prepare_megavul.py" --in-dir "$WORK/megavul_ml1024/linevd" --out-dir "$PREP" --keep-benign
fi
NUM_CLASSES=$(python -c "import json;print(json.load(open('$PREP/cwe_labels.json'))['num_classes'])")
echo "  mode=$MODE num_classes=$NUM_CLASSES"

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
# Preprocess cache (joern_csv + ggnn_input + wv) — parsing the funcs is the slow part; restore
# from Drive on a fresh pod to skip straight to train. PREP_CACHE set per-mode up top.
PREP_RESTORED=""
if [[ ! -d "$PREP/ggnn_input" ]] && rclone ls "$REMOTE/data/baselines/$PREP_CACHE" >/dev/null 2>&1; then
  echo "  restoring LIVABLE preprocess cache from Drive (skips joern parse + builder) ..."
  rclone copy "$REMOTE/data/baselines/$PREP_CACHE" /tmp/ --progress
  mkdir -p "$PREP"; tar --no-same-owner -xzf "/tmp/$PREP_CACHE" -C "$PREP" && rm -f "/tmp/$PREP_CACHE"
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
# --vuln-only: drop benign (label 0) + remap CWE 1..25 -> 0..24, in place on the renamed ggnn.
# joern/ggnn cache is label-independent so this only filters graphs (no re-parse/rebuild). Cache
# holds pristine diverse-*.json (cached pre-rename) so it stays 26-class. Marker = idempotent.
if [[ -n "$VULN_ONLY" && "$MODE" == "megavul" ]]; then   # bigvul label 0 is a real CWE, never filter
  if [[ ! -f "$GG/.vo_done" ]]; then     # filter once (marker); decrement always (below)
    python - "$GG" <<'PY'
import json, os, sys
gg = sys.argv[1]
for fn in ("multi-train1-v0.json", "multi-valid-v0.json", "multi-test-v0.json"):
    p = os.path.join(gg, fn)
    if not os.path.exists(p): continue
    data = json.load(open(p)); out = []
    for e in data:
        lab = int(e["targets"][0][0])
        if lab == 0: continue          # benign -> drop
        e["targets"] = [[lab - 1]]     # CWE 1..25 -> 0..24
        out.append(e)
    json.dump(out, open(p, "w"))
    print(f"  {fn}: kept {len(out)}/{len(data)} (dropped benign)")
PY
    touch "$GG/.vo_done"
  fi
  NUM_CLASSES=$((NUM_CLASSES - 1))     # benign class removed (always, even if already filtered)
  echo "  vuln-only: NUM_CLASSES=$NUM_CLASSES"
fi
# main_sta pickles the built DataSet (DGL graphs) to a .bin and reloads it next run — keep
# it as a cache; only invalidate when the GGNNinput actually changed (else rebuild the 4389
# graphs every run, ~1-2 min). .bin is data-only (graphs+labels), independent of num_classes.
[[ -f "$GG/multi-train1-v0.json" ]] && find "$GG" -maxdepth 1 -name "multi_*batch*.bin" ! -newer "$GG/multi-train1-v0.json" -delete 2>/dev/null || true
# Reset patched source to pristine first, so the seds below (which match the ORIGINAL 31 / [917,...]
# / AdamW text) always apply — otherwise a same-pod re-run silently no-ops and leaves stale values.
git -C "$LV" checkout -- code/main_sta.py code/trainer_sta.py code/modules/model.py code/data_loader/dataset.py 2>/dev/null || true
sed -i "s/from trainer_test import train/from trainer_sta import train/" "$LV/code/main_sta.py"
sed -i -E "s/torch\.manual_seed\([0-9]+\)/torch.manual_seed($SEED)/; s/np\.random\.seed\([0-9]+\)/np.random.seed($SEED)/" "$LV/code/main_sta.py"   # multi-seed
sed -i "s/self.graph.add_edge(/self.graph.add_edges(/" "$LV/code/data_loader/dataset.py"   # dgl 2.x rename
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = '1'/os.environ['CUDA_VISIBLE_DEVICES'] = '0'/" "$LV/code/main_sta.py"  # single-GPU pod
sed -i "s/MLPReadout(self.hidden_dim2, 31)/MLPReadout(self.hidden_dim2, $NUM_CLASSES)/" "$LV/code/modules/model.py"
sed -i "s/MLPReadout(2 \* self.seq_hid, 31)/MLPReadout(2 * self.seq_hid, $NUM_CLASSES)/" "$LV/code/modules/model.py"
# num_classes=31 + the 917-long num_class_list are hardcoded in BOTH main_sta.py AND
# trainer_sta.py (3 loss classes: CrossEntropy, BalancedSoftmaxCE, ClassBalanceFocal — the
# head loss at line 189). Each builds a 31-wide one-hot, mismatching our 26 outputs. Patch
# ALL of them to our label space + OUR per-class train counts (faithful long-tail re-weighting).
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
for f in "$LV/code/main_sta.py" "$LV/code/trainer_sta.py"; do
  sed -i "s/num_class_list = \[917,.*\]/num_class_list = $NCL/g" "$f"
  sed -i "s/num_classes = 31\$/num_classes = $NUM_CLASSES/g" "$f"
done
# ClassBalanceFocal head loss: pow(base, gamma=0.5) has an INFINITE gradient when the
# sigmoid saturates (base->0) -> nan loss. Add an epsilon to the base to stabilize, keeping
# LIVABLE's dual-branch long-tail loss intact (faithful — only numerical, not a loss change).
sed -i 's/torch.pow((1-p)\*label + p \* (1-label), self.gamma)/torch.pow((1-p)*label + p * (1-label) + 1e-6, self.gamma)/' "$LV/code/trainer_sta.py"
# Dump final-test predictions (y_true,y_pred) so compute_baseline_metrics can report weighted-F1 +
# accuracy too (LIVABLE's evaluate_metrics logs only MACRO). Injected before the return in
# evaluate_metrics; the last call (final test, line ~287) leaves the CSV = test preds.
sed -i '/return np.mean(_loss).item()/i\        if __import__("os").environ.get("LIVABLE_PRED_CSV"): __import__("pandas").DataFrame({"y_true": all_targets, "y_pred": all_predictions}).to_csv(__import__("os").environ["LIVABLE_PRED_CSV"], index=False)' "$LV/code/trainer_sta.py"
# Optimizer (--opt/--lr): repo ships AdamW(lr=1e-3) active; README pt6 says RAdam lr=1e-4 but that
# FREEZES on our data (loss flat, acc 0). Default here = AdamW 1e-3 (trains). Patch the active line.
case "$OPT" in
  radam) OPTLINE="optim = torch.optim.RAdam(model.parameters(), lr=$LR, weight_decay=1e-6)";;
  adam)  OPTLINE="optim = torch.optim.Adam(model.parameters(), lr=$LR, weight_decay=1e-6)";;
  *)     OPTLINE="optim = AdamW(model.parameters(), lr=$LR, weight_decay=1e-6)";;
esac
echo "  optimizer: $OPTLINE"
sed -i "s|^\(\s*\)optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)|\1$OPTLINE  # patched|" "$LV/code/main_sta.py"
grep -qF "$OPTLINE" "$LV/code/main_sta.py" || { echo "ERR: optimizer patch failed"; exit 1; }
cd "$LV/code"
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
export LIVABLE_PRED_CSV="$OUT/livable_cls_preds.csv"   # final-test preds dumped by patched evaluate_metrics
python main_sta.py --input_dir "$GG" 2>&1 | tee "$OUT/train.log"

# recompute macro + weighted + accuracy from the dumped predictions (LIVABLE logs only macro)
cd "$WORK"
if [[ -f "$LIVABLE_PRED_CSV" ]]; then
  PYTHONPATH=src python scripts/compute_baseline_metrics.py --name LIVABLE \
    --classification "$LIVABLE_PRED_CSV" --out "$OUT/livable_recomputed_metrics.json" 2>&1 | tee "$OUT/recomputed_metrics.log"
fi

echo "=== [7/7] upload: results -> results/baselines, weights -> checkpoints/baselines ==="
cd "$WORK"
COMP="$(command -v pigz || echo gzip)"
# results = log + metrics (metrics printed in train.log) -> results/baselines/
tar -I "$COMP" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
# weights = trained GGNN+BiLSTM model (newest *-model.bin) -> checkpoints/baselines/
WBIN=$(find "$LV/code" -name "*-model.bin" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [[ -n "$WBIN" ]]; then
  tar -I "$COMP" -cf "${RUN_ID}_weights.tar.gz" -C "$(dirname "$WBIN")" "$(basename "$WBIN")"
  rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
fi
echo "DONE: $RUN_ID  results -> results/baselines, weights -> checkpoints/baselines"
