#!/usr/bin/env bash
# build_titanvul_cloud.sh
# ~~~~~~~~~~~~~~~~~~~~~~~
# TitanVul -> CPG (Joern) -> .pt untuk DUA target:
#   ml1024  : arsitektur berbasis graph + sekuensial
#   ml5120  : arsitektur hibrida (jalur LM sliding window)
#
# Filter: fungsi rentan C/C++, CWE dalam ruang label MegaVul 26 kelas, dan
# DIDEDUPLIKASI terhadap MegaVul (hash teks fungsi setelah normalisasi whitespace),
# sehingga tidak ada fungsi yang pernah dilihat model saat pelatihan.
# 38.548 -> C/C++ 17.101 -> Top25 3.762 -> baru 2.466 fungsi.
#
# Node feature dibangun sekali (ml1024); ml5120 hanya re-tokenisasi.
#
# Run (cloud, Linux, rclone.conf sudah terpasang):
#   bash scripts/build_titanvul_cloud.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

WORK="$PWD"
DRIVE="gdrive-mesach:tugas-akhir"
JOERN_VER="4.0.526"
TITAN="data/datasets/titanvul/raw.parquet"
MEGAVUL="data/datasets/megavul/train.parquet"
DS="data/datasets/titanvul_cc"
RAW="data/raw/titanvul"
PROC="data/processed"
COMP="$(command -v pigz || echo gzip)"

echo "=== [1/6] dependensi ==="
pip install -q torch transformers torch_geometric pandas scikit-learn tqdm fastparquet networkx loguru
command -v pigz >/dev/null || (apt-get update -q && apt-get install -y -q pigz)
apt-get update -q && (apt-get install -y -q openjdk-21-jdk-headless || apt-get install -y -q default-jdk)
JH=$(ls -d /usr/lib/jvm/java-21-openjdk* /usr/lib/jvm/temurin-21* /usr/lib/jvm/*-21-* 2>/dev/null | head -1 || true)
[[ -n "${JH:-}" ]] && export JAVA_HOME="$JH" && export PATH="$JH/bin:$PATH"
echo "  java: $(java -version 2>&1 | head -1)"
# tiap worker joern = 1 JVM (~3 GB puncak), jadi batasnya RAM bukan core
RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
AUTO=$(( RAM_GB / 4 )); [[ $AUTO -gt $(nproc) ]] && AUTO=$(nproc); [[ $AUTO -lt 1 ]] && AUTO=1
WORKERS="${WORKERS:-$AUTO}"
echo "  cpu=$(nproc) ram=${RAM_GB}G -> workers=$WORKERS"

echo "=== [2/6] data mentah ==="
for f in "$TITAN" "$MEGAVUL"; do
  if [[ ! -f "$f" ]]; then
    mkdir -p "$(dirname "$f")"
    rclone copy "$DRIVE/$f" "$(dirname "$f")" --progress
  fi
done

echo "=== [3/6] adapter (filter C/C++, Top25, dedup vs MegaVul) ==="
# Vocab 26 kelas DITANAM di adapter, tidak dibaca dari file.
python scripts/titanvul_to_parquet.py --input "$TITAN" --megavul "$MEGAVUL" --out-dir "$DS"
NCLS=$(python -c "import json;print(len(json.load(open('$DS/cwe_vocab.json'))))")
[[ "$NCLS" == "26" ]] || { echo "ERR: vocab harus 26 kelas, dapat $NCLS"; exit 1; }

echo "=== [4/6] joern-cli v${JOERN_VER} + CPG ==="
JCLI="$WORK/joern-cli"
if [[ ! -x "$JCLI/joern-parse" ]]; then
  wget -q --show-progress "https://github.com/joernio/joern/releases/download/v${JOERN_VER}/joern-cli.zip" -O /tmp/joern-cli.zip
  unzip -q -o /tmp/joern-cli.zip -d "$WORK" && rm -f /tmp/joern-cli.zip
  chmod +x "$JCLI"/joern-parse "$JCLI"/joern-export "$JCLI"/*.sh 2>/dev/null || true
fi
printf 'int f(){return 0;}\n' > /tmp/t.c
"$JCLI/joern-parse" /tmp/t.c --output /tmp/t.bin >/dev/null 2>&1 \
  || { echo "ERR: joern-parse gagal. java=$(java -version 2>&1|head -1)"; exit 1; }
echo "  joern sanity OK"

OUT="$WORK/data/_tv"; rm -rf "$OUT"; mkdir -p "$OUT/api"
python -m gnn_vuln.data.prepare --input "$DS/train.parquet" --format api \
  --joern-cli "$JCLI" --out-dir "$OUT" --top-cwe 0 --workers "$WORKERS" \
  --cwe-vocab "$DS/cwe_vocab.json"
rm -rf "$RAW"; mkdir -p "$(dirname "$RAW")"; mv "$OUT/api" "$RAW"; rm -rf "$OUT"
cp -f "$DS/cwe_vocab.json" "$RAW/cwe_vocab.json"
echo "  CPG: $(ls "$RAW"/vulnerable/*.xml 2>/dev/null | wc -l) fungsi"

echo "=== [5/6] build .pt (ml1024 lalu ml5120) ==="
python -m gnn_vuln.data.build_pt --config configs/titanvul/titanvul_ml1024.yaml
python -m gnn_vuln.data.build_pt --config configs/titanvul/titanvul_ml5120.yaml

echo "=== [6/6] tar + upload ==="
pack_upload() {   # $1 = pola ml, $2 = nama tar
  local meta name
  meta="$(ls $PROC/lm_dataset_titanvul_*${1}*_meta.pt | head -1)"
  name="$(basename "$meta" _meta.pt)"
  echo "  pack $name -> $2"
  tar -C "$PROC" -cf - "${name}_meta.pt" "${name}_graphs" | $COMP > "$PROC/$2"
  rclone copy "$PROC/$2" "$DRIVE/" --progress
  echo "  uploaded $2"
}
pack_upload "ml1024" "lm_dataset_titanvul_multiclass_unixcoder-base-nine_ft_ml1024_lazy.tar.gz"
pack_upload "ml5120" "lm_dataset_titanvul_multiclass_unixcoder-base-nine_ft_ml5120_lazy.tar.gz"

echo
echo "SELESAI. Dataset di Drive $DRIVE/:"
echo "  lm_dataset_titanvul_multiclass_unixcoder-base-nine_ft_ml1024_lazy.tar.gz  (graph, sekuensial)"
echo "  lm_dataset_titanvul_multiclass_unixcoder-base-nine_ft_ml5120_lazy.tar.gz  (hibrida)"
