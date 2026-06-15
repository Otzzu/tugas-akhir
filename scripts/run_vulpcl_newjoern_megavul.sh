#!/usr/bin/env bash
# run_vulpcl_newjoern_megavul.sh — VulPCL categorization on OUR megavul, comparable head-to-head.
# Model = their CodeBert_Blstm (faithful, untouched). Graph->features RE-DERIVED from modern joern
# 4.0.526 CPGs (their 2014 Neo4j/SVG pipeline is dead). Features approximate VulPCL's -> documented.
# Same megavul split (export_baseline_split) + 26 classes (benign + 25 CWE) as our model.
#
# !!! v1 — EXPECT to iterate on the pod (like LIVABLE). Paste errors. !!!
# Pipeline: megavul split -> joern CPGs -> vulpcl_newjoern_adapter (PTSC+FCDS+CPAG-DeepWalk) -> pkls
#           -> patch CodeBert_Blstm (num_classes, target_names, GPU0, wire linear2) -> train -> upload.
# Usage (pod, project root):  bash scripts/run_vulpcl_newjoern_megavul.sh
set -uo pipefail   # not -e: keep going, surface the failing stage

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
JOERN_VER="4.0.526"
WORK="$PWD"; VP="$WORK/src/VulPCL"; CAT="$VP/vul_categorization"
SPLIT="$WORK/megavul_ml1024/linevd"; CPG="$WORK/megavul_vulpcl_cpg"; PKL="$WORK/megavul_vulpcl_pkl"
PROJ="megavul"; CWE="all"; RUN_ID="vulpcl_newjoern_megavul_$(date +%Y%m%d_%H%M%S)"

echo "=== [1/8] deps + VulPCL repo ==="
VENV="/workspace/vulpcl_env"
BASEPY=/venv/main/bin/python; [[ -x "$BASEPY" ]] || BASEPY="$(command -v python3 || command -v python)"
[[ -d "$VENV" ]] || "$BASEPY" -m venv "$VENV"
source "$VENV/bin/activate"
python -c "import torch,transformers,sklearn,pandas,networkx,gensim,fastparquet" 2>/dev/null || \
  pip install -q torch transformers scikit-learn numpy tqdm pandas networkx gensim fastparquet
[[ -f "$CAT/module/CodeBert_Blstm.py" ]] || { rm -rf "$VP"; git clone --depth 1 https://github.com/liucyy/VulPCL.git "$VP"; }
# deepwalk_embed uses gensim 3.x Word2Vec API (size=, iter=); gensim<4 won't build on py3.12, so use
# 4.x + patch the two renamed kwargs (size->vector_size, iter->epochs). Idempotent on reruns.
DWE="$CAT/deepwalk_embed/deepwalk_embedding.py"
sed -i 's/kwargs\["size"\] = embed_size/kwargs["vector_size"] = embed_size/' "$DWE"
sed -i 's/kwargs\["iter"\] = iter/kwargs["epochs"] = iter/' "$DWE"
command -v pigz >/dev/null || (apt-get update -q && apt-get install -y -q pigz)

echo "=== [2/8] JDK21 + joern v${JOERN_VER} (matches local) ==="
apt-get update -q && (apt-get install -y -q openjdk-21-jdk-headless || apt-get install -y -q default-jdk)
JH=$(ls -d /usr/lib/jvm/java-21-openjdk* /usr/lib/jvm/*-21-* 2>/dev/null | head -1 || true)
[[ -n "${JH:-}" ]] && export JAVA_HOME="$JH" && export PATH="$JH/bin:$PATH"
JCLI="$WORK/joern-cli"
if [[ ! -x "$JCLI/joern-parse" ]]; then
  wget -q --show-progress "https://github.com/joernio/joern/releases/download/v${JOERN_VER}/joern-cli.zip" -O /tmp/joern-cli.zip
  unzip -q -o /tmp/joern-cli.zip -d "$WORK" && rm -f /tmp/joern-cli.zip
  chmod +x "$JCLI"/joern-parse "$JCLI"/joern-export "$JCLI"/*.sh 2>/dev/null || true
fi

echo "=== [3/8] data: megavul split ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar --no-same-owner -xzf "$DATA_TAR"
fi

echo "=== [4/8] combined parquet for CPG gen (all funcs, binary so none dropped) ==="
python - "$SPLIT" "$WORK/megavul_vulpcl_cpg_input.parquet" <<'PY'
import pandas as pd, sys
sp, out = sys.argv[1], sys.argv[2]
fr = [pd.read_parquet(f"{sp}/{s}.parquet") for s in ("train","val","test")]
df = pd.concat(fr, ignore_index=True)
o = pd.DataFrame({"func_before": df["func_before"],
                  "func_after": df["func_after"] if "func_after" in df else "",
                  "CWE ID": df["cwe_name"].fillna("").astype(str),
                  "target": df["vul"].astype(int),
                  "id": df["id"].astype(int)})
o.to_parquet(out, index=False); print("combined", len(o))
PY

echo "=== [5/8] joern CPGs (modern, per-func json; --binary keeps ALL funcs) ==="
WORKERS=$(( $(nproc) < 8 ? $(nproc) : 8 ))   # joern-4 JVMs ~1.5GB each -> cap (nproc=112 OOMs)
if [[ ! -d "$CPG" ]]; then
  PYTHONPATH=src python scripts/prepare_dataset.py --input "$WORK/megavul_vulpcl_cpg_input.parquet" \
    --format bigvul --binary --joern-cli "$JCLI" --out-dir "$CPG" --workers "$WORKERS"
fi

echo "=== [6/8] adapter: CPGs -> VulPCL pkls (PTSC + FCDS + CPAG DeepWalk) ==="
python scripts/vulpcl_newjoern_adapter.py --split-dir "$SPLIT" --cpg-dir "$CPG/bigvul" --out-dir "$PKL" --vulpcl "$VP"
NC=$(python -c "import json;print(json.load(open('$PKL/cwe_labels.json'))['num_classes'])")
echo "  num_classes=$NC"

echo "=== [7/8] stage pkls + vocab, patch model, train ==="
DEST="$CAT/vul_data/$CWE/$PROJ"; mkdir -p "$DEST" "$CAT/save_dict/$PROJ/$CWE"
cp -f "$PKL"/{train,val,test}_set_token.pkl "$DEST/"
cp -f "$PKL/fcds_code_vocab.json" "$CAT/${PROJ}_code_vocab.json"
git -C "$VP" checkout -- vul_categorization/module/CodeBert_Blstm.py vul_categorization/codebert_blstm.py 2>/dev/null || true
sed -i "s/self.num_classes = 12/self.num_classes = $NC/" "$CAT/module/CodeBert_Blstm.py"
sed -i "s/target_names=\[[^]]*\]/target_names=[str(i) for i in range($NC)]/" "$CAT/codebert_blstm.py"
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = '3'/os.environ['CUDA_VISIBLE_DEVICES'] = '0'/" "$CAT/codebert_blstm.py"
sed -i "s/^\(\s*\)return out$/\1return self.linear2(out)/" "$CAT/module/CodeBert_Blstm.py"
grep -q "self.linear2(out)" "$CAT/module/CodeBert_Blstm.py" || { echo "ERR: linear2 patch failed"; exit 1; }
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
( cd "$CAT" && python codebert_blstm.py --p "$PROJ" --cwe "$CWE" 2>&1 | tee "$OUT/train.log" )

echo "=== [8/8] upload pkls (preprocess) + results to Drive ==="
COMP=(gzip); command -v pigz >/dev/null && COMP=(pigz)
tar -cf - -C "$PKL" . | "${COMP[@]}" > "/tmp/${RUN_ID}_pkls.tar.gz"
rclone copy "/tmp/${RUN_ID}_pkls.tar.gz" "$REMOTE/data/baselines/" --progress 2>/dev/null || true
tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
echo "DONE: $RUN_ID"
