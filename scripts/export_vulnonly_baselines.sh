#!/usr/bin/env bash
# export_vulnonly_baselines.sh — siapkan masukan LOSVER, VulExplainer, dan LIVABLE dari
# dataset VULN-ONLY, bukan dari saringan split 26 kelas.
#
# KENAPA. Pada Tabel IV.10 dan IV.11 arsitektur usulan dinilai pada test dataset vuln-only
# (913 fungsi, tetap tiap seed), sedangkan ketiga baseline itu dinilai pada fungsi rentan yang
# tersaring dari split 26 kelas (914, 910, 921). Keduanya 10% acak dari populasi yang sama,
# tetapi bukan himpunan yang sama, sehingga perbandingannya tidak berpasangan. Skrip ini
# menyamakannya dengan mengekspor dari dataset vuln-only yang sama persis dipakai model.
#
# LineVD dan LineVul TIDAK perlu diulang. Keduanya dibandingkan pada Tabel IV.12 terhadap
# model 26 kelas, dan sudah memakai ekspor dari split 26 kelas yang sama.
#
# JEBAKAN YANG DITAMBAL, dan hanya SATU kolom. export_baseline_split.py menetapkan
# vul = int(y > 0) dan target = int(y > 0). Pada dataset 26 kelas itu benar karena id 0 memang
# benign. Pada dataset vuln-only id 0 adalah CWE sungguhan, sehingga tanpa tambalan satu kelas
# penuh akan ditandai tidak rentan lalu dibuang oleh ketiga baseline yang menyaring vul == 1.
# Skrip ini memaksa vul dan target menjadi 1, dan mencetak berapa baris yang tertambal.
#
# Kemungkinan besar kolom inilah alasan pendekatan sekarang menyaring dari split 26 kelas,
# karena di sana kolom vul datang gratis dan sudah benar.
#
# PENOMORAN ULANG KELAS TIDAK JADI MASALAH. build_vuln_only_subset.py memetakan id 1..25
# menjadi 0..24 dan membuang benign dari class_names, tetapi ketiga prep script membaca kolom
# cwe_name berupa string lalu membangun sendiri indeks kelasnya dari urutan unik terurut.
# Kolom label numerik tidak pernah mereka pakai, jadi penomoran ulang lewat begitu saja.
#
# Jalankan dari root repo. Semua tahap CPU saja, tidak menyentuh GPU dan tidak melatih apa pun.
#
#   bash scripts/export_vulnonly_baselines.sh
#
# Ubah lewat env bila perlu:
#   PROCESSED_DIR=data/processed DS_BASE=<nama dataset tanpa _meta.pt> OUT_ROOT=data/baselines/megavul_vulnonly
set -euo pipefail

PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
DS_BASE="${DS_BASE:-lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42}"
DS="${DS_BASE}_vulnonly"
OUT_ROOT="${OUT_ROOT:-data/baselines/megavul_vulnonly}"
SEEDS="${SEEDS:-42 1 2}"
TOKENIZER="${TOKENIZER:-microsoft/unixcoder-base-nine}"

command -v pigz >/dev/null || COMP=gzip
COMP="$(command -v pigz || echo gzip)"

echo "dataset : $PROCESSED_DIR/${DS}_meta.pt"
echo "keluaran: $OUT_ROOT/s{seed}"
echo "seed    : $SEEDS"
echo

[[ -f "$PROCESSED_DIR/${DS}_meta.pt" ]] || {
  echo "TIDAK ADA $PROCESSED_DIR/${DS}_meta.pt"
  echo "Bangun dulu subsetnya:"
  echo "  PYTHONPATH=src python scripts/build_vuln_only_subset.py \\"
  echo "      --processed-dir $PROCESSED_DIR --ds-name $DS_BASE --suffix _vulnonly"
  exit 1
}

for SEED in $SEEDS; do
  # nama direktori dalam TETAP megavul_ml1024 supaya semua run script dan lib_baseline_data.sh
  # menemukannya tanpa perubahan path
  OUT="$OUT_ROOT/s$SEED/megavul_ml1024"
  echo "=============================== seed $SEED ==============================="

  # 1. ekspor split vuln-only ke format teks baseline
  PYTHONPATH=src python scripts/export_baseline_split.py \
      --processed-dir "$PROCESSED_DIR" --ds-name "$DS" --out-dir "$OUT" --seed "$SEED"

  # 2. tambal kolom biner. Pada dataset vuln-only SELURUH fungsi rentan, sedangkan
  #    export_baseline_split menurunkan vul dan target dari y > 0 sehingga kelas id 0 salah
  #    ditandai tidak rentan.
  python - "$OUT" <<'PATCH'
import sys, pandas as pd
from pathlib import Path
out = Path(sys.argv[1])
for s in ("train", "val", "test"):
    p = out / "linevd" / f"{s}.parquet"
    df = pd.read_parquet(p)
    n0 = int((df["vul"] == 0).sum())
    df["vul"] = 1
    df.to_parquet(p, index=False)

    c = out / "linevul" / f"{s}.csv"
    dc = pd.read_csv(c)
    dc["target"] = 1
    dc.to_csv(c, index=False)
    print(f"  tambal {s}: {n0} baris vul=0 -> 1, total {len(df)}", file=sys.stderr)
PATCH

  # 2b. PENJAGA. Pastikan flaw mask yang dipakai versi yang sudah ditambal, bukan versi lama
  #     yang menandai seluruh badan fungsi. Ciri versi lama, sebagian besar himpunan baris
  #     penyebab dimulai dari baris 1, yaitu baris tanda tangan fungsi. Pada ekspor yang benar
  #     angkanya sekitar 7 persen, pada ekspor lama sekitar 85 persen.
  python - "$OUT" <<'GUARD'
import sys, numpy as np, pandas as pd
from pathlib import Path
out = Path(sys.argv[1])
df = pd.concat([pd.read_parquet(out / "linevd" / f"{s}.parquet") for s in ("train","val","test")])
fl = df["flaw_lines"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
w = fl[fl.apply(len) > 0]
pct = w.apply(lambda x: min(x) == 1).mean() * 100
print(f"  penjaga flaw mask: {len(w)} fungsi berflaw, {pct:.1f}% memuat baris 1", file=sys.stderr)
if pct > 40:
    sys.exit("  GAGAL. Flaw mask versi LAMA yang menandai seluruh badan fungsi. "
             "Bangun ulang dataset dengan mask yang sudah ditambal sebelum melanjutkan.")
GUARD

  # 3. siapkan masukan tiap baseline dari ekspor yang sama
  python scripts/vulexplainer_prepare_megavul.py \
      --in-dir "$OUT/linevd" --out-dir "$OUT/vulexplainer"

  python scripts/export_losver_jsonl.py \
      --in-dir "$OUT/linevd" --out-dir "$OUT/losver" \
      --tokenizer "$TOKENIZER" --token-limit 512

  python scripts/livable_prepare_megavul.py \
      --in-dir "$OUT/linevd" --out-dir "$OUT/livable"

  echo "  selesai seed $SEED"
  echo
done

# 4. bundel dan unggah, satu tar per seed supaya run script tinggal menariknya
echo "=============================== bundel ==============================="
for SEED in $SEEDS; do
  TAR="megavul_vulnonly_baselines_s${SEED}.tar.gz"
  tar -cf - -C "$OUT_ROOT/s$SEED" megavul_ml1024 | "$COMP" > "$OUT_ROOT/$TAR"
  echo "  $TAR  $(du -h "$OUT_ROOT/$TAR" | cut -f1)"
done

echo
echo "Unggah ke Drive:"
echo "  rclone copy $OUT_ROOT gdrive-mesach:tugas-akhir/data/baselines/ --include '*_vulnonly_baselines_s*.tar.gz' --progress"
echo
echo "Lalu LATIH ULANG DARI NOL ketiga baseline, satu run per seed:"
echo "  SEED=42 DATA_PREFIX=megavul_vulnonly_baselines bash scripts/run_losver_cloud.sh"
echo "  SEED=1  DATA_PREFIX=megavul_vulnonly_baselines bash scripts/run_losver_cloud.sh"
echo "  SEED=2  DATA_PREFIX=megavul_vulnonly_baselines bash scripts/run_losver_cloud.sh"
echo "Sama untuk run_vulexplainer_megavul.sh dan run_livable_cloud.sh."
echo
echo "JANGAN sekadar menilai ulang checkpoint lama pada test yang baru. Checkpoint lama dilatih"
echo "pada train seed 42, dan test seed 1 atau 2 beririsan sekitar 80 persen dengan train itu,"
echo "sehingga skornya bocor dan melambung."
echo
echo "Yang diharapkan setelah ini, test ketiga baseline menjadi HIMPUNAN BAGIAN dari test"
echo "arsitektur usulan, bukan lagi ambilan acak terpisah. Angkanya tidak seluruhnya sama,"
echo "karena tiap baseline tetap memakai saringannya sendiri:"
echo "  VulExplainer  ambil semua rentan            -> 913, sama persis dengan usulan"
echo "  LOSVER        <=512 token dan berlabel      -> sekitar 477"
echo "  LIVABLE       preprocessing berhasil        -> sekitar 575"
echo
echo "Perubahan hasil diperkirakan kecil. Data latih beririsan sekitar 80 persen, karena dua"
echo "ambilan 80 persen dari kolam yang sama. Yang berubah banyak justru data uji, dari"
echo "beririsan 9 persen menjadi bersarang penuh. Macro F1 paling bergejolak karena peka kelas"
echo "tail, sebagian kelas hanya punya satu sampai tiga sampel uji."
