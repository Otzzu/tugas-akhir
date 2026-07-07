#!/usr/bin/env bash
# run_vulpcl_localization_megavul.sh — VulPCL *localization* (LVD) on OUR megavul, reusing our hdf5 CPGs.
# Model = their vul_localization/CodeBert_Blstm (BINARY vuln classifier; line localization via CodeBERT
# attention -> per-line score -> top-k line accuracy). UNTOUCHED arch. Separate model from categorization
# (no weight share). Binary label (benign INCLUDED), same megavul split.
#
# !!! v1 — line-mapping (flaw line i -> their line_num i-1) is a [VERIFY] assumption; paste errors/metrics. !!!
# Pipeline: fetch hdf5 + split -> vulpcl_localization_adapter (token pkls + seq pkls + msg + vocab) ->
#   stage at hardcoded paths -> patch (GPU0, wire linear2, ckpt-load path) -> train+test+atten-localize -> upload.
# Usage (pod, project root):  nohup bash scripts/run_vulpcl_localization_megavul.sh > vp_loc.log 2>&1 &
set -uo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="$(rclone lsf "$REMOTE/data/baselines/" 2>/dev/null | grep -E '^megavul_ml1024_baselines_.*\.tar\.gz$' | sort | tail -1)"
DATA_TAR="${DATA_TAR:-megavul_ml1024_baselines_20260613.tar.gz}"   # newest bundle on Drive, fallback to legacy
WORK="$PWD"; VP="$WORK/src/VulPCL"; LOC="$VP/vul_localization"
SPLIT="$WORK/megavul_ml1024/linevd"; HDF5="$WORK/data/graphs/megavul.hdf5"; PKL="$WORK/megavul_vulpcl_loc_pkl"
PROJ="megavul"; RUN_ID="vulpcl_loc_megavul_$(date +%Y%m%d_%H%M%S)"

echo "=== [1/6] deps + VulPCL repo ==="
VENV="/workspace/vulpcl_env"
BASEPY=/venv/main/bin/python; [[ -x "$BASEPY" ]] || BASEPY="$(command -v python3 || command -v python)"
[[ -d "$VENV" ]] || "$BASEPY" -m venv "$VENV"
source "$VENV/bin/activate"
python -c "import torch,transformers,sklearn,pandas,networkx,gensim,fastparquet,h5py" 2>/dev/null || \
  pip install -q torch transformers scikit-learn numpy tqdm pandas networkx gensim fastparquet h5py
[[ -f "$LOC/module/CodeBert_Blstm.py" ]] || { rm -rf "$VP"; git clone --depth 1 https://github.com/liucyy/VulPCL.git "$VP"; }
# deepwalk uses gensim 3.x API (size=, iter=) -> patch to 4.x (vector_size, epochs). Idempotent.
DWE="$LOC/deepwalk_embed/deepwalk_embedding.py"
sed -i 's/kwargs\["size"\] = embed_size/kwargs["vector_size"] = embed_size/' "$DWE"
sed -i 's/kwargs\["iter"\] = iter/kwargs["epochs"] = iter/' "$DWE"
# hs=1 (hierarchical softmax) over our 680k-node vocab = ~1hr, gensim feeder-bound. Switch to negative
# sampling (gensim default negative=5) = 5-10x faster, near-equivalent embeddings, standard word2vec/DeepWalk.
sed -i 's/kwargs\["hs"\] = 1/kwargs["hs"] = 0/' "$DWE"

echo "=== [2/6] data: megavul.hdf5 (our CPGs) + split parquets ==="
if [[ ! -f "$HDF5" ]]; then
  mkdir -p data/graphs
  rclone copy "$REMOTE/data/raw/megavul.hdf5.zst" /tmp/ --progress
  command -v zstd >/dev/null || (apt-get update -q && apt-get install -y -q zstd)
  zstd -d -f /tmp/megavul.hdf5.zst -o "$HDF5" && rm -f /tmp/megavul.hdf5.zst
fi
[[ -d megavul_ml1024 ]] || { rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar --no-same-owner -xzf "$DATA_TAR"; }

echo "=== [3/6] adapter: hdf5 CPGs -> localization pkls (token + seq + msg + vocab) ==="
python scripts/vulpcl_localization_adapter.py --hdf5 "$HDF5" --split-dir "$SPLIT" --out-dir "$PKL" --vulpcl "$VP" \
  --num-walks "${VP_NUM_WALKS:-10}" \
  || { echo "ERR: adapter failed -> NOT training on stale pkls"; exit 1; }

echo "=== [4/6] stage pkls/seq/vocab/msg at the paths the code expects ==="
mkdir -p "$LOC/data/$PROJ" "$LOC/data/big_vul" "$LOC/save_dict/$PROJ"
cp -f "$PKL"/{train,val,test}_set_token.pkl "$LOC/data/$PROJ/"
cp -f "$PKL"/{train,val,test}_seq.pkl       "$LOC/data/big_vul/"      # seq path hardcoded to big_vul
cp -f "$PKL/megavul_code_vocab.json" "$LOC/${PROJ}_code_vocab.json"
cp -f "$PKL/big_vul_msg.txt"         "$LOC/big_vul_msg.txt"           # msg name hardcoded
# patches (faithful: config/paths only, arch untouched)
git -C "$VP" checkout -- vul_localization/module/CodeBert_Blstm.py vul_localization/codebert_blstm.py 2>/dev/null || true
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = '1'/os.environ['CUDA_VISIBLE_DEVICES'] = '0'/" "$LOC/codebert_blstm.py"
sed -i 's#./save_dict/big_vul/CodeBert_Blstm.ckpt#./save_dict/megavul/CodeBert_Blstm.ckpt#' "$LOC/codebert_blstm.py"
# forward returns the 768-d fused features + attention; wire linear2 so cross_entropy gets [B,2] logits.
sed -i 's/return out, atten_score/return self.linear2(out), atten_score/' "$LOC/module/CodeBert_Blstm.py"
grep -q "self.linear2(out), atten_score" "$LOC/module/CodeBert_Blstm.py" || { echo "ERR: linear2 wire patch failed"; exit 1; }

echo "=== [5/6] train + test + attention localization (their CodeBert_Blstm, untouched arch) ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
( cd "$LOC" && python codebert_blstm.py --p "$PROJ" 2>&1 | tee "$OUT/train.log" )

echo "=== [6/6] upload: pkls -> data/baselines, results -> results/baselines, weights -> checkpoints/baselines ==="
COMP="$(command -v pigz || echo gzip)"
tar -cf - -C "$PKL" . | "$COMP" > "/tmp/${RUN_ID}_pkls.tar.gz"
rclone copy "/tmp/${RUN_ID}_pkls.tar.gz" "$REMOTE/data/baselines/" --progress 2>/dev/null || true
tar -I "$COMP" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" . && \
  rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress 2>/dev/null || true
WCKPT="$LOC/save_dict/$PROJ/CodeBert_Blstm.ckpt"
if [[ -f "$WCKPT" ]]; then
  tar -I "$COMP" -cf "${RUN_ID}_weights.tar.gz" -C "$(dirname "$WCKPT")" CodeBert_Blstm.ckpt && \
    rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress 2>/dev/null || true
fi
echo "DONE: $RUN_ID  (line-loc metrics line_p/line_tp in $OUT/train.log)"
