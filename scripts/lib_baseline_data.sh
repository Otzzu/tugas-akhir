#!/usr/bin/env bash
# lib_baseline_data.sh — pemilihan bundel data baseline yang sadar seed.
#
# KENAPA ADA. Sebelum ini tiap run script memilih bundel begini.
#
#   DATA_TAR="$(rclone lsf ... | grep -E '^megavul_ml1024_baselines_.*\.tar\.gz$' | sort | tail -1)"
#   DATA_TAR="${DATA_TAR:-megavul_ml1024_baselines_20260613.tar.gz}"
#
# Tiga cacat sekaligus. Pertama, pola itu tidak memuat seed sehingga semua seed mendapat bundel
# yang sama, yaitu split seed 42. Kedua, baris pertama menimpa tanpa syarat sehingga
# `DATA_TAR=... bash run_x.sh` tidak berpengaruh dan baris kedua tidak pernah terpakai. Ketiga,
# direktori hasil ekstrak dijaga dengan `if [[ ! -d megavul_ml1024 ]]`, sehingga run seed
# berikutnya di pod yang sama diam-diam memakai data seed sebelumnya.
#
# Akibatnya LineVD, LineVul, LOSVER, dan LIVABLE dilatih tiga kali pada split seed 42, dan
# standard deviation yang dilaporkan hanya menangkap keacakan pelatihan.
#
# CARA PAKAI di run script, tepat sebelum langkah data.
#
#   source "$WORK/scripts/lib_baseline_data.sh"
#   DATA_PREFIX=megavul_vulnonly_baselines NEEDS_FLAW=1 baseline_data_fetch "$REMOTE" "$SEED"
#
# Setelah itu direktori `megavul_ml1024/` dijamin berisi split seed yang diminta, dan nama
# bundelnya ada di $DATA_TAR. Semua path downstream tidak berubah.
#
# Variabel yang dibaca.
#   DATA_TAR     bila diisi, dipakai apa adanya dan pencarian dilewati
#   DATA_PREFIX  awalan nama bundel, default megavul_ml1024_baselines
#   NEEDS_FLAW   set ke 1 pada baseline lokalisasi supaya flaw mask diperiksa
#   DATA_DIR     nama direktori hasil ekstrak, default megavul_ml1024

baseline_data_fetch() {
  local remote="$1" seed="$2"
  local prefix="${DATA_PREFIX:-megavul_ml1024_baselines}"
  local dir="${DATA_DIR:-megavul_ml1024}"

  if [[ -z "$seed" ]]; then
    echo "ERR: SEED kosong. Jalankan dengan SEED=42 (atau 1, 2), jangan mengandalkan default." >&2
    return 1
  fi

  # 1. tentukan bundel. env menang, kalau tidak ada cari yang namanya memuat seed PERSIS.
  if [[ -z "${DATA_TAR:-}" ]]; then
    DATA_TAR="$(rclone lsf "$remote/data/baselines/" 2>/dev/null \
      | grep -E "^${prefix}_s${seed}\.tar\.gz$" | head -1)"
  fi
  if [[ -z "$DATA_TAR" ]]; then
    echo "ERR: tidak ada bundel untuk seed $seed." >&2
    echo "     dicari : ${prefix}_s${seed}.tar.gz di $remote/data/baselines/" >&2
    echo "     tersedia:" >&2
    rclone lsf "$remote/data/baselines/" 2>/dev/null | grep -E '\.tar\.gz$' | sed 's/^/       /' >&2
    echo "     Bangun dulu dengan scripts/export_vulnonly_baselines.sh, atau set DATA_TAR manual." >&2
    echo "     JANGAN mundur ke bundel seed 42, itu sumber cacat P#6." >&2
    return 1
  fi
  export DATA_TAR

  # 2. ekstrak. Direktori lama dari seed lain dibuang, tidak dipakai ulang diam-diam.
  local mark="$dir/.seed"
  if [[ -d "$dir" ]]; then
    local have=""; [[ -f "$mark" ]] && have="$(cat "$mark")"
    if [[ "$have" != "$seed" ]]; then
      echo "  buang $dir bekas seed '${have:-tak bertanda}', ganti dengan seed $seed"
      rm -rf "$dir"
    else
      echo "  pakai $dir yang sudah ada, seed $seed cocok"
    fi
  fi
  if [[ ! -d "$dir" ]]; then
    echo "  unduh $DATA_TAR"
    rclone copy "$remote/data/baselines/$DATA_TAR" . --progress
    tar -I "$(command -v pigz || echo gzip)" -xf "$DATA_TAR" || tar --no-same-owner -xzf "$DATA_TAR"
    [[ -d "$dir" ]] || { echo "ERR: $DATA_TAR tidak berisi direktori $dir" >&2; return 1; }
    echo "$seed" > "$mark"
  fi

  # 3. periksa isinya, dan pada baseline lokalisasi periksa juga versi flaw mask.
  NEEDS_FLAW="${NEEDS_FLAW:-}" python - "$dir" "$seed" "${NEEDS_FLAW:-0}" <<'CHECK'
import sys
from pathlib import Path
d, seed, needs_flaw = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
try:
    import numpy as np, pandas as pd
except ImportError:
    print("  PERINGATAN: pandas tidak ada, pemeriksaan data dilewati")
    sys.exit(0)
te = pd.read_parquet(d / "linevd" / "test.parquet")
n_vul = int((te["vul"] == 1).sum()) if "vul" in te else -1
print(f"  seed {seed}: test {len(te)} fungsi, rentan {n_vul}")
if not needs_flaw:
    sys.exit(0)
fl = te["flaw_lines"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
w = fl[fl.apply(len) > 0]
if len(w) == 0:
    sys.exit("  GAGAL. Tidak ada satu pun baris penyebab pada test, baseline lokalisasi tidak bisa dinilai.")
pct = w.apply(lambda x: min(x) == 1).mean() * 100
print(f"  flaw mask: {len(w)} fungsi berflaw, {pct:.1f}% memuat baris 1")
if pct > 40:
    sys.exit("  GAGAL. Ini flaw mask versi LAMA yang menandai seluruh badan fungsi (bug METHOD-span). "
             "Bundel seperti megavul_ml1024_baselines_20260613, split_s1, dan split_s2 tidak boleh "
             "dipakai untuk lokalisasi. Ekspor ulang dari dataset yang sudah ditambal.")
CHECK
}
