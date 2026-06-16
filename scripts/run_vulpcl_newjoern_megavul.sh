#!/usr/bin/env bash
# run_vulpcl_newjoern_megavul.sh — VulPCL categorization on OUR megavul, comparable head-to-head.
# Model = their CodeBert_Blstm (faithful, untouched). Graph->features RE-DERIVED from our EXISTING
# modern-joern CPGs in data/graphs/megavul.hdf5 (the SAME graphs our model uses) — NO joern re-parse.
# Same megavul split (export_baseline_split) + 26 classes (benign + 25 CWE) as our model.
#
# !!! v1 — may iterate. Paste errors. !!!
# Pipeline: fetch megavul.hdf5.zst -> decompress -> vulpcl_newjoern_adapter (PTSC+FCDS+CPAG-DeepWalk
#   read from hdf5) -> pkls -> patch CodeBert_Blstm (num_classes, target_names, GPU0, wire linear2)
#   -> train -> upload pkls + results.
# Usage (pod, project root):  nohup bash scripts/run_vulpcl_newjoern_megavul.sh > vp_newjoern.log 2>&1 &
set -uo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
WORK="$PWD"; VP="$WORK/src/VulPCL"; CAT="$VP/vul_categorization"
SPLIT="$WORK/megavul_ml1024/linevd"; HDF5="$WORK/data/graphs/megavul.hdf5"; PKL="$WORK/megavul_vulpcl_pkl"
PROJ="megavul"; CWE="all"; RUN_ID="vulpcl_newjoern_megavul_$(date +%Y%m%d_%H%M%S)"

echo "=== [1/6] deps + VulPCL repo ==="
VENV="/workspace/vulpcl_env"
BASEPY=/venv/main/bin/python; [[ -x "$BASEPY" ]] || BASEPY="$(command -v python3 || command -v python)"
[[ -d "$VENV" ]] || "$BASEPY" -m venv "$VENV"
source "$VENV/bin/activate"
python -c "import torch,transformers,sklearn,pandas,networkx,gensim,fastparquet,h5py" 2>/dev/null || \
  pip install -q torch transformers scikit-learn numpy tqdm pandas networkx gensim fastparquet h5py
[[ -f "$CAT/module/CodeBert_Blstm.py" ]] || { rm -rf "$VP"; git clone --depth 1 https://github.com/liucyy/VulPCL.git "$VP"; }
# deepwalk_embed uses gensim 3.x Word2Vec API (size=, iter=) -> patch to 4.x (vector_size, epochs). Idempotent.
DWE="$CAT/deepwalk_embed/deepwalk_embedding.py"
sed -i 's/kwargs\["size"\] = embed_size/kwargs["vector_size"] = embed_size/' "$DWE"
sed -i 's/kwargs\["iter"\] = iter/kwargs["epochs"] = iter/' "$DWE"

echo "=== [2/6] data: megavul.hdf5 (our CPGs, reused) + split parquets ==="
if [[ ! -f "$HDF5" ]]; then
  mkdir -p data/graphs
  rclone copy "$REMOTE/data/raw/megavul.hdf5.zst" /tmp/ --progress
  command -v zstd >/dev/null || (apt-get update -q && apt-get install -y -q zstd)
  zstd -d -f /tmp/megavul.hdf5.zst -o "$HDF5" && rm -f /tmp/megavul.hdf5.zst
fi
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar --no-same-owner -xzf "$DATA_TAR"
fi

echo "=== [3/6] adapter: hdf5 CPGs -> VulPCL pkls (PTSC + FCDS + CPAG DeepWalk) ==="
python scripts/vulpcl_newjoern_adapter.py --hdf5 "$HDF5" --split-dir "$SPLIT" --out-dir "$PKL" --vulpcl "$VP"
NC=$(python -c "import json;print(json.load(open('$PKL/cwe_labels.json'))['num_classes'])")
echo "  num_classes=$NC"

echo "=== [4/6] stage pkls + vocab, patch model ==="
DEST="$CAT/vul_data/$CWE/$PROJ"; mkdir -p "$DEST" "$CAT/save_dict/$PROJ/$CWE"
cp -f "$PKL"/{train,val,test}_set_token.pkl "$DEST/"
cp -f "$PKL/fcds_code_vocab.json" "$CAT/${PROJ}_code_vocab.json"
git -C "$VP" checkout -- vul_categorization/module/CodeBert_Blstm.py vul_categorization/codebert_blstm.py 2>/dev/null || true
sed -i "s/self.num_classes = 12/self.num_classes = $NC/" "$CAT/module/CodeBert_Blstm.py"
sed -i "s/target_names=\[[^]]*\]/target_names=[str(i) for i in range($NC)]/" "$CAT/codebert_blstm.py"
# paper config: Adam, batch 8, epoch 20, dropout 0.5 (code ships 16/30 -> match the paper).
sed -i "s/self.batch_size = 16/self.batch_size = 8/" "$CAT/module/CodeBert_Blstm.py"
sed -i "s/self.num_epochs = 30/self.num_epochs = 20/" "$CAT/module/CodeBert_Blstm.py"
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = '3'/os.environ['CUDA_VISIBLE_DEVICES'] = '0'/" "$CAT/codebert_blstm.py"
sed -i "s/^\(\s*\)return out$/\1return self.linear2(out)/" "$CAT/module/CodeBert_Blstm.py"
grep -q "self.linear2(out)" "$CAT/module/CodeBert_Blstm.py" || { echo "ERR: linear2 patch failed"; exit 1; }

echo "=== [5/6] train (their CodeBert_Blstm, untouched) ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
( cd "$CAT" && python codebert_blstm.py --p "$PROJ" --cwe "$CWE" 2>&1 | tee "$OUT/train.log" )

echo "=== [6/6] upload pkls (preprocess) + results to Drive ==="
COMP=(gzip); command -v pigz >/dev/null && COMP=(pigz)
tar -cf - -C "$PKL" . | "${COMP[@]}" > "/tmp/${RUN_ID}_pkls.tar.gz"
rclone copy "/tmp/${RUN_ID}_pkls.tar.gz" "$REMOTE/data/baselines/" --progress 2>/dev/null || true
tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
echo "DONE: $RUN_ID"
