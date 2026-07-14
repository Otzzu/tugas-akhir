#!/usr/bin/env bash
# convert_datasets_inmemory_cloud.sh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Ubah dataset lazy di Drive menjadi inmemory, satu per satu:
#   unduh tar -> ekstrak -> konversi -> tar ulang -> unggah -> BERSIHKAN
# Selalu bersih sebelum dataset berikutnya, jadi disk yang dipakai = dataset terbesar saja.
#
# Kompresi pigz (semua core). Dekompresi gz single-core menurut formatnya, jadi
# ekstraksi memang I/O-bound; itu wajar, bukan salah konfigurasi.
#
# inmemory memuat SELURUH graph di RAM. Script menolak dataset yang tidak muat
# (butuh ~1,3x ukuran folder graphs). Paksa dengan FORCE=1 kalau yakin.
#
# Run:
#   bash scripts/convert_datasets_inmemory_cloud.sh                 # semua
#   bash scripts/convert_datasets_inmemory_cloud.sh benchvul_ml1024 titanvul_ml1024
#   FORCE=1 bash scripts/convert_datasets_inmemory_cloud.sh megavul_ml5120
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

DRIVE="gdrive-mesach:tugas-akhir"
PROC="data/processed"
COMP="$(command -v pigz || echo gzip)"

# key | folder Drive | nama tar sumber (lazy)
DATASETS=(
  "megavul_ml1024|data/processed/megavul|lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_lazy_20260707_192819.tar.gz"
  "megavul_ml5120|data/processed/megavul|lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_lazy_20260707_190841.tar.gz"
  "benchvul_ml1024|.|lm_dataset_benchvul_multiclass_unixcoder-base-nine_ft_ml1024.tar.gz"
  "benchvul_ml5120|.|lm_dataset_benchvul_multiclass_unixcoder-base-nine_ft_ml5120.tar.gz"
  "titanvul_ml1024|.|lm_dataset_titanvul_multiclass_unixcoder-base-nine_ft_ml1024_lazy.tar.gz"
  "titanvul_ml5120|.|lm_dataset_titanvul_multiclass_unixcoder-base-nine_ft_ml5120_lazy.tar.gz"
)

WANT=("$@")
mkdir -p "$PROC"
command -v pigz >/dev/null || (apt-get update -q && apt-get install -y -q pigz) || true
python -c "import torch, tqdm" 2>/dev/null || pip install -q torch tqdm

wanted() {
  [[ ${#WANT[@]} -eq 0 ]] && return 0
  for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done
  return 1
}

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r KEY DIR TAR <<< "$entry"
  wanted "$KEY" || continue

  echo
  echo "==================== $KEY ===================="
  rm -rf "${PROC:?}"/lm_dataset_* 2>/dev/null || true

  SRC="$DRIVE/$TAR"; [[ "$DIR" != "." ]] && SRC="$DRIVE/$DIR/$TAR"
  echo "--- unduh $TAR"
  rclone copy "$SRC" "$PROC/" --progress

  echo "--- ekstrak"
  tar -I "$COMP" -xf "$PROC/$TAR" -C "$PROC"
  rm -f "$PROC/$TAR"

  META="$(ls "$PROC"/*_meta.pt 2>/dev/null | head -1 || true)"
  [[ -n "$META" ]] || { echo "ERR: $KEY bukan dataset lazy (tidak ada _meta.pt)"; exit 1; }
  NAME="$(basename "$META" _meta.pt)"

  NEED=$(( $(du -sm "$PROC/${NAME}_graphs" | cut -f1) * 13 / 10 ))
  FREE=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 ))
  echo "--- RAM: butuh ~${NEED} MB, tersedia ${FREE} MB"
  if [[ $NEED -gt $FREE && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP $KEY: RAM tidak cukup. Pakai pod lebih besar, atau FORCE=1 kalau yakin."
    rm -rf "${PROC:?}"/lm_dataset_*
    continue
  fi

  echo "--- konversi ke inmemory"
  python scripts/lazy_to_inmemory.py --meta "$META"

  OUT="${TAR/_lazy/}"; OUT="${OUT%.tar.gz}_inmemory.tar.gz"
  echo "--- pack $OUT"
  tar -C "$PROC" -cf - "${NAME}.pt" | $COMP > "$PROC/$OUT"

  DST="$DRIVE/"; [[ "$DIR" != "." ]] && DST="$DRIVE/$DIR/"
  rclone copy "$PROC/$OUT" "$DST" --progress
  echo "--- terunggah ke ${DST}${OUT}"

  rm -rf "${PROC:?}"/lm_dataset_*
done

echo
echo "SELESAI. Sisa di $PROC:"
ls "$PROC" 2>/dev/null || true
