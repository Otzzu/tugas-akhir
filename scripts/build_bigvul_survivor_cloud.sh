#!/usr/bin/env bash
# build_bigvul_survivor_cloud.sh — build OUR bigvul-survivor .pt in the cloud and upload to Drive.
# Full self-contained: all_vul.csv + split record -> adapter -> joern CPG -> .pt (lazy) -> upload.
# Mirrors build_upload_datasets.sh, but bigvul needs joern (megavul had pre-built graphs).
# split_file is committed in the repo (configs/ablation/bigvul/bigvul_survivor_split.json), so the
# train pod gets it via git — no need to ship it here.
#
# Usage (fresh pod, project root):  bash scripts/build_bigvul_survivor_cloud.sh
set -euo pipefail

REMOTE_PROC="gdrive-mesach:tugas-akhir/data/processed/bigvul_survivor"
REMOTE_BASE="gdrive-mesach:tugas-akhir/data/baselines"
WORK="$PWD"; PROC="data/processed"; RAW="data/raw/bigvul_survivor"; DS="data/datasets/bigvul_survivor"
NODE_LM="microsoft/unixcoder-base"; FUNC_LM="microsoft/unixcoder-base-nine"; ML=5120
JOERN_VER="4.0.526"   # MUST match local C:/joern/joern-cli (io.joern.joern-cli-4.0.526.jar) so CPGs are identical

echo "=== [1/6] deps ==="
pip install -q torch transformers torch_geometric pandas scikit-learn tqdm fastparquet networkx gdown
command -v pigz >/dev/null || (apt-get update -q && apt-get install -y -q pigz)
# joern 4.x needs JDK 21+ (local uses 25). Install 21, fall back to default-jdk.
apt-get update -q && (apt-get install -y -q openjdk-21-jdk-headless || apt-get install -y -q default-jdk)

echo "=== [2/6] data: all_vul.csv + split record ==="
mkdir -p data/LIVABLE bigvul_livable
[[ -f data/LIVABLE/all_vul.csv ]] || rclone copy "$REMOTE_BASE/all_vul.csv" data/LIVABLE/ 2>/dev/null \
  || gdown 1j8WUSNte6Tda2uXR8Ue3Hk9jRa-33LGL -O data/LIVABLE/all_vul.csv
if [[ ! -f bigvul_livable/used_train.json ]]; then
  rclone copy "$REMOTE_BASE/livable_bigvul_split_record.tar.gz" .
  tar -xzf livable_bigvul_split_record.tar.gz -C bigvul_livable
fi

echo "=== [3/6] adapter -> parquet + cwe_vocab + split_file ==="
PYTHONPATH=src python scripts/bigvul_survivor_to_parquet.py \
  --csv data/LIVABLE/all_vul.csv --record-dir bigvul_livable --out-dir "$DS" --top-cwe 31
mkdir -p "$RAW"; cp -f "$DS/cwe_vocab.json" "$RAW/cwe_vocab.json"

echo "=== [4/6] joern-cli (1.1+) + CPG (no --sample/--limit) ==="
JCLI="$WORK/joern-cli"
if [[ ! -x "$JCLI/joern-parse" ]]; then
  wget -q "https://github.com/joernio/joern/releases/download/v${JOERN_VER}/joern-cli.zip" -O /tmp/joern-cli.zip
  unzip -q -o /tmp/joern-cli.zip -d "$WORK" && rm -f /tmp/joern-cli.zip
  chmod +x "$JCLI"/joern-parse "$JCLI"/joern-export "$JCLI"/*.sh 2>/dev/null || true
fi
"$JCLI/joern-parse" --version 2>/dev/null | head -1 || true   # sanity: should report ${JOERN_VER}
PYTHONPATH=src python scripts/prepare_dataset.py --input "$DS/survivors.parquet" \
  --format bigvul --joern-cli "$JCLI" --out-dir "$RAW" --top-cwe 31 --workers "$(nproc)"

echo "=== [5/6] build .pt (lazy) — matches O1_bigvul_survivor.yaml ==="
BUILD_PY="
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from pathlib import Path
ds = CodeBERTGraphDataset(root='data', source='bigvul_survivor', mode='multiclass',
    pretrained_lm='${NODE_LM}', func_lm='${FUNC_LM}', add_func_tokens=True,
    func_max_length=${ML}, top_cwe=0, max_per_class=0, max_nodes=2500,
    storage='lazy', embedder_device='${1:-cuda}', use_flash_attention=False)
print(Path(ds.processed_paths[0]).name.replace('_meta.pt',''))
"
DSNAME=$(PYTHONPATH=src python -c "$BUILD_PY" | tail -1)
echo "  dataset = $DSNAME"

echo "=== [6/6] tar (pigz) + upload ==="
TS=$(date +%Y%m%d_%H%M%S); TAR="${PROC}/${DSNAME}_lazy_${TS}.tar.gz"
tar -I pigz -cf "$TAR" -C "$PROC" "${DSNAME}_meta.pt" "${DSNAME}_graphs/"
rclone copy "$TAR" "$REMOTE_PROC" --progress && rm -f "$TAR"
echo "DONE. dataset = $DSNAME"
echo "train: ./scripts/train_cloud.sh --skip --config configs/ablation/bigvul/O1_bigvul_survivor.yaml --dataset $DSNAME"
