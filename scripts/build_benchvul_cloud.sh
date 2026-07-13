#!/usr/bin/env bash
# build_benchvul_cloud.sh
# ~~~~~~~~~~~~~~~~~~~~~~~
# BenchVul -> CPG (Joern) -> .pt untuk DUA target:
#   ml1024  : arsitektur berbasis graph + sekuensial
#   ml5120  : arsitektur hibrida (jalur LM sliding window)
#
# Hanya fungsi rentan C/C++ yang CWE-nya ada di ruang label MegaVul 26 kelas
# (Top 25 versi 2025). BenchVul memakai daftar Top 25 tahun lain, jadi CWE-190,
# CWE-269, CWE-400, dan CWE-798 dibuang -> ~156 fungsi tersisa dari 230 C/C++.
#
# Node feature dibangun sekali (ml1024). ml5120 hanya me-re-tokenisasi teks fungsi
# lewat fast-path build_pt, jadi tidak ada embedding ulang.
#
# Run (cloud, Linux, rclone.conf sudah terpasang):
#   bash scripts/build_benchvul_cloud.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

WORK="$PWD"
DRIVE="gdrive-mesach:tugas-akhir"
JOERN_VER="4.0.526"
PARQUET="data/datasets/benchvul/train.parquet"
DS="data/datasets/benchvul_cc"          # hasil adapter
RAW="data/raw/benchvul"                 # CPG per fungsi
PROC="data/processed"
COMP="$(command -v pigz || echo gzip)"

echo "=== [1/6] dependensi ==="
pip install -q torch transformers torch_geometric pandas scikit-learn tqdm fastparquet networkx loguru
command -v pigz >/dev/null || (apt-get update -q && apt-get install -y -q pigz)
# joern 4 butuh JDK 21+. Image dasar sering memakai java lama -> joern-parse gagal DIAM-DIAM.
apt-get update -q && (apt-get install -y -q openjdk-21-jdk-headless || apt-get install -y -q default-jdk)
JH=$(ls -d /usr/lib/jvm/java-21-openjdk* /usr/lib/jvm/temurin-21* /usr/lib/jvm/*-21-* 2>/dev/null | head -1 || true)
[[ -n "${JH:-}" ]] && export JAVA_HOME="$JH" && export PATH="$JH/bin:$PATH"
echo "  java: $(java -version 2>&1 | head -1)"
WORKERS=$(( $(nproc) < 8 ? $(nproc) : 8 ))   # tiap JVM joern ~1,5 GB

echo "=== [2/6] data mentah ==="
if [[ ! -f "$PARQUET" ]]; then
  mkdir -p "$(dirname "$PARQUET")"
  rclone copy "$DRIVE/data/datasets/benchvul/train.parquet" "$(dirname "$PARQUET")" --progress
fi
# cwe_vocab MegaVul = ruang label acuan (26 kelas). SELALU ambil dari Drive: salinan di
# repo masih versi lama 193 kelas dengan id yang berbeda, dan memakainya akan membuat
# seluruh label BenchVul salah.
mkdir -p data/raw/megavul
rclone copy "$DRIVE/data/raw/megavul/cwe_vocab.json" data/raw/megavul/ --progress
NCLS=$(python -c "import json;print(len(json.load(open('data/raw/megavul/cwe_vocab.json'))))")
[[ "$NCLS" == "26" ]] || { echo "ERR: cwe_vocab harus 26 kelas, dapat $NCLS"; exit 1; }

echo "=== [3/6] adapter -> parquet berbentuk MegaVul ==="
python scripts/benchvul_to_parquet.py \
  --input "$PARQUET" \
  --cwe-vocab data/raw/megavul/cwe_vocab.json \
  --out-dir "$DS"

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

# prepare menambahkan subdir '<format>' sendiri -> $OUT/api. --cwe-vocab memaksa id kelas
# MegaVul dipakai apa adanya, jadi label benchvul sepadan dengan model 26 kelas.
OUT="$WORK/data/_bv"; rm -rf "$OUT"; mkdir -p "$OUT/api"
python -m gnn_vuln.data.prepare --input "$DS/train.parquet" --format api \
  --joern-cli "$JCLI" --out-dir "$OUT" --top-cwe 0 --workers "$WORKERS" \
  --cwe-vocab "$DS/cwe_vocab.json"
rm -rf "$RAW"; mkdir -p "$(dirname "$RAW")"; mv "$OUT/api" "$RAW"; rm -rf "$OUT"
cp -f "$DS/cwe_vocab.json" "$RAW/cwe_vocab.json"   # dataset_lm butuh vocab di raw dir
echo "  CPG: $(ls "$RAW"/vulnerable/*.json 2>/dev/null | wc -l) fungsi"

echo "=== [5/6] build .pt (ml1024 lalu ml5120) ==="
python -m gnn_vuln.data.build_pt --config configs/benchvul/benchvul_ml1024.yaml
python -m gnn_vuln.data.build_pt --config configs/benchvul/benchvul_ml5120.yaml

echo "=== [6/6] tar + upload ==="
pack_upload() {   # $1 = pola ml, $2 = nama tar
  local meta name
  meta="$(ls $PROC/lm_dataset_benchvul_*${1}*_meta.pt | head -1)"
  name="$(basename "$meta" _meta.pt)"
  echo "  pack $name -> $2"
  tar -C "$PROC" -cf - "${name}_meta.pt" "${name}_graphs" | $COMP > "$PROC/$2"
  rclone copy "$PROC/$2" "$DRIVE/" --progress
  echo "  uploaded $2"
}
pack_upload "ml1024" "lm_dataset_benchvul_multiclass_unixcoder-base-nine_ft_ml1024_lazy.tar.gz"
pack_upload "ml5120" "lm_dataset_benchvul_multiclass_unixcoder-base-nine_ft_ml5120_lazy.tar.gz"

echo
echo "SELESAI. Dataset di Drive $DRIVE/:"
echo "  lm_dataset_benchvul_multiclass_unixcoder-base-nine_ft_ml1024_lazy.tar.gz  (graph, sekuensial)"
echo "  lm_dataset_benchvul_multiclass_unixcoder-base-nine_ft_ml5120_lazy.tar.gz  (hibrida)"
