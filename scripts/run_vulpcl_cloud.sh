#!/usr/bin/env bash
# run_vulpcl_cloud.sh — train VulPCL categorization (multiclass CWE, CodeBERT+BLSTM)
# on OUR MegaVul split.  VulPCL = IST'24, src/VulPCL/vul_categorization.
# Pipeline: our .c funcs -> graph_generation.py (OLD Neo4j-Gremlin Joern) -> ast/cfg_dfg/
# ddg_cdg SVG -> adc_features_extracting (deepwalk) + cd_features_extracting (FCDS seq) +
# CodeBERT tokens -> codebert_blstm.py train.
#
# STATUS: SKELETON. The BIG hurdle is the ANCIENT Joern (joern.all.JoernSteps + Gremlin +
# py2neo + Neo4j) used by graph_generation.py — far older/harder than LineVD's Joern.
# Marked [HARD]. Model side (CodeBERT+BLSTM) is light (no DGL).
# Usage (pod, project root):  bash scripts/run_vulpcl_cloud.sh
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
RUN_ID="vulpcl_megavul_$(date +%Y%m%d_%H%M%S)"
WORK="$PWD"
VP="$WORK/src/VulPCL"
CAT="$VP/vul_categorization"
PREP="$WORK/megavul_vulpcl"
PROJ="megavul"

echo "=== [1/7] VulPCL present (vendored) ==="
[[ -d "$VP" ]] || git clone --depth 1 https://github.com/liucyy/VulPCL.git "$VP"

echo "=== [2/7] venv: torch + transformers(CodeBERT) + graph deps ==="
VENV="/workspace/vulpcl_env"
[[ -d "$VENV" ]] || /venv/main/bin/python -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
pip install -q torch transformers scikit-learn networkx tqdm pandas fastparquet graphviz "gensim<4.4"
# deepwalk for graph node embeddings (adc_features_extracting/deepwalk_embed)
pip install -q deepwalk || true
python -c "import torch,transformers; print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"

echo "=== [3/7] data: our split + VulPCL adapter (source .c + label file) ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar -xzf "$DATA_TAR"
fi
python "$WORK/scripts/vulpcl_prepare_megavul.py" --in-dir "$WORK/megavul_ml1024/linevd" \
  --out-dir "$PREP" --project "$PROJ"
NUM_CLASSES=$(python -c "import json;print(json.load(open('$PREP/cwe_labels.json'))['num_classes'])")
echo "  num_classes=$NUM_CLASSES"

echo "=== [4/7] [HARD] graph generation: OLD Neo4j-Gremlin Joern -> SVG (ast/cfg_dfg/ddg_cdg) ==="
# graph_generation.py uses `from joern.all import JoernSteps` (Yamaguchi joern, ~2014):
#   - needs Neo4j + the old joern python bindings + a Joern DB built from our .c files.
#   - this is the single biggest reproduction risk. Options to try on the pod:
#     (a) the old joern docker image, (b) build Neo4j + joern-lang from source.
# [HARD/VERIFY] confirm: how graph_generation.py ingests source dir + writes SVGs to
#   data/graph/<project>/{ast,cfg_dfg,ddg_cdg}/<file>@<func>.svg
cd "$VP/data_preprocessing"
echo "  [HARD] set up old Joern (Neo4j/Gremlin) then:"
echo "    for g in ast cfg_dfg ddg_cdg; do python graph_generation.py --g \$g; done"
# for g in ast cfg_dfg ddg_cdg; do python graph_generation.py --g "$g"; done   # <- enable after Joern works

echo "=== [5/7] features: deepwalk graph (adc) + FCDS sequence (cd) + CodeBERT tokens ==="
cd "$CAT"
cp "$PREP/${PROJ}_labels.txt" "$CAT/${PROJ}_labels.txt"
# [VERIFY] adc/cd read ./data/graph/<project>/*.svg ; --p <project> --cwe CWE-TOP (multiclass)
# python adc_features_extracting.py --p "$PROJ" --cwe CWE-TOP
# python cd_features_extracting.py  --p "$PROJ" --cwe CWE-TOP
# python vul_files_label.py --p "$PROJ" --cwe CWE-TOP

echo "=== [6/7] train (set target_names/num_classes = our label space) ==="
# [VERIFY] codebert.py target_names hardcodes ['0'..'11'] -> patch to range($NUM_CLASSES).
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
# python codebert_blstm.py --p "$PROJ" --cwe CWE-TOP 2>&1 | tee "$OUT/train.log"
echo "  [VERIFY] python codebert_blstm.py --p $PROJ --cwe CWE-TOP"

echo "=== [7/7] upload results ==="
cd "$WORK"
[[ -d "$OUT" ]] && tar -czf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress || true
echo "DONE (skeleton): $RUN_ID  — old-Joern setup is the blocker, see [HARD] in step 4"
