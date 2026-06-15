#!/usr/bin/env bash
# run_vulpcl_megavul_oldjoern.sh — ATTEMPT VulPCL categorization on OUR megavul via their EXACT
# pipeline (old Yamaguchi joern -> SVG -> adc/cd extractors -> CodeBert_Blstm). If it runs, this is
# a TRUE comparable head-to-head (same megavul, same split, same classes as our model).
#
# !!! BEST-EFFORT SCAFFOLD !!! The old joern stack (joern ~0.3.x + Neo4j 2.x + Gremlin + python-joern,
# circa 2014, Python 2) is UNDOCUMENTED in their repo and may not build in 2026. Stages marked [TRY]
# are the wall you iterate on the pod. The non-[TRY] stages (export, path-fix, extractors, train) are solid.
#
# Pipeline (their README + graph_generation.py):
#   our .c -> joern parse -> Neo4j index -> graph_generation.py (joern-plot-* -> .dot -> dot -Tsvg)
#     -> ./vul_data/graph/megavul/{ast,cfg_dfg,ddg_cdg}/*.svg
#   -> adc_features_extracting.py (CPAG+DeepWalk) + cd_features_extracting.py (FCDS) -> pkls
#   -> codebert_blstm.py --p megavul
# Usage (pod, project root):  bash scripts/run_vulpcl_megavul_oldjoern.sh
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
WORK="$PWD"; VP="$WORK/src/VulPCL"; CAT="$VP/vul_categorization"
PROJ="megavul"; CWE="all"
RUN_ID="vulpcl_megavul_$(date +%Y%m%d_%H%M%S)"

echo "=== [1/7] export our megavul split as .c + labels (26-class, matches our model + split) ==="
# vulpcl_prepare_megavul.py: --keep-benign => 26-class (benign + 25 CWE), == our model/LineVD/LineVul.
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar --no-same-owner -xzf "$DATA_TAR"
fi
PREP="$WORK/megavul_vulpcl"
python "$WORK/scripts/vulpcl_prepare_megavul.py" --in-dir "$WORK/megavul_ml1024/linevd" \
  --out-dir "$PREP" --project "$PROJ" --keep-benign
NUM_CLASSES=$(python -c "import json;print(json.load(open('$PREP/cwe_labels.json'))['num_classes'])")
echo "  num_classes=$NUM_CLASSES"
# Stage code + labels where the extractors expect them.
mkdir -p "$CAT/vul_data/code/$PROJ" "$CAT/vul_data/graph/$PROJ"/{ast,cfg_dfg,ddg_cdg}
cp -f "$PREP"/source/*/*.c "$CAT/vul_data/code/$PROJ/" 2>/dev/null || true
cp -f "$PREP/${PROJ}_labels.txt" "$CAT/${PROJ}_${CWE}_labels.txt"

echo "=== [2/7] [TRY] old joern stack (the WALL — iterate here) ==="
# The original Yamaguchi joern (provides: joern parse, joern-lookup, joern-plot-ast,
# joern-plot-proggraph) + Neo4j 2.x + Gremlin + python-joern (joern.all.JoernSteps, Python 2).
# Known references to try (NONE guaranteed to build in 2026):
#   - joern (old):        https://github.com/octopus-platform/joern        (build with ant; needs JDK 8)
#   - python-joern:       https://github.com/octopus-platform/python-joern (Python 2; `joern.all`)
#   - joern-tools:        https://github.com/octopus-platform/joern-tools  (joern-lookup / joern-plot-*)
#   - Neo4j 2.1.8 + gremlin-plugin (old joern queries via Gremlin)
#   - graphviz (`dot`)
# A docker image for this era is the most realistic path — search "octopus joern docker".
echo "  [TRY] install: JDK8, ant, Neo4j 2.1.8 + gremlin-plugin, python-joern (py2), joern-tools, graphviz"
echo "  [TRY] verify on PATH:  joern  joern-lookup  joern-plot-proggraph  joern-plot-ast  dot"
command -v joern-plot-proggraph >/dev/null || { echo "  [WALL] old joern tools not installed — set them up, then re-run from here."; exit 2; }

echo "=== [3/7] [TRY] build joern index from our .c + load into Neo4j ==="
# old joern: `joern <code_dir>` builds .joernIndex; then start Neo4j on it so JoernSteps can query.
echo "  [TRY] joern $CAT/vul_data/code/$PROJ   # -> .joernIndex"
echo "  [TRY] start Neo4j 2.1.8 pointing at the index (gremlin-plugin enabled), confirm JoernSteps().connectToDatabase() works"

echo "=== [4/7] graph_generation.py -> .dot -> .svg (fix hardcoded /home/liucy path) ==="
cd "$VP/data_preprocessing"
# graph_generation.py hardcodes the author's skip-path /home/liucy/big_vul/cfg_dfg/ -> point to ours.
sed -i "s#/home/liucy/big_vul/cfg_dfg/#$CAT/vul_data/graph/$PROJ/cfg_dfg/#g" graph_generation.py
# It writes .dot/.svg into the cwd; run per graph type (py2 env where joern.all imports).
for g in ast cfg_dfg ddg_cdg; do
  echo "  [TRY] python2 graph_generation.py --g $g    # -> *_${g}.svg"
done
echo "  [TRY] then move the .svg into $CAT/vul_data/graph/$PROJ/{ast,cfg_dfg,ddg_cdg}/"

echo "=== [5/7] features: CPAG (adc, DeepWalk) + FCDS (cd) ==="
cd "$CAT"
echo "  [VERIFY] python adc_features_extracting.py --p $PROJ --cwe $CWE"
echo "  [VERIFY] python cd_features_extracting.py  --p $PROJ --cwe $CWE --get_vocab yes"
echo "  [VERIFY] python vul_files_label.py         --p $PROJ --cwe $CWE"

echo "=== [6/7] patch model num_classes/target_names to $NUM_CLASSES, then train ==="
git -C "$VP" checkout -- vul_categorization/module/CodeBert_Blstm.py vul_categorization/codebert_blstm.py 2>/dev/null || true
sed -i "s/self.num_classes = 12/self.num_classes = $NUM_CLASSES/" "$CAT/module/CodeBert_Blstm.py"
sed -i "s/target_names=\[[^]]*\]/target_names=[str(i) for i in range($NUM_CLASSES)]/" "$CAT/codebert_blstm.py"
mkdir -p "$CAT/save_dict/$PROJ/$CWE"
echo "  [VERIFY] python codebert_blstm.py --p $PROJ --cwe $CWE"

echo "=== [7/7] DONE scaffold. The [TRY] stages (old joern + Neo4j) are the wall; iterate there. ==="
echo "RUN_ID=$RUN_ID  (once trained: tar baseline_runs/$RUN_ID + rclone to $REMOTE/results/baselines/)"
