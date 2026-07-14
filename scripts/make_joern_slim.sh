#!/usr/bin/env bash
# make_joern_slim.sh
# ~~~~~~~~~~~~~~~~~~
# joern-cli tanpa frontend yang tidak dipakai: 1,9 GB -> 208 MB.
#
# Joern merilis SEMUA frontend dalam satu zip (C#, Ghidra, JS, Swift, Kotlin, Ruby, Go,
# Java, Python, PHP ...). Kita cuma memakai c2cpg (C/C++). Sisanya dibuang.
# Terbukti menghasilkan prediksi identik dengan joern penuh.
#
# Run (pod, rclone terpasang):
#   bash scripts/make_joern_slim.sh [JOERN_VERSION]
set -euo pipefail
cd "$(dirname "$0")/.."

VER="${1:-v4.0.526}"
DRIVE="gdrive-mesach:tugas-akhir/tools"
OUT="joern-cli-slim-${VER}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "=== [1/4] unduh joern-cli ${VER} (1,7 GB) ==="
curl -fL "https://github.com/joernio/joern/releases/download/${VER}/joern-cli.zip" \
     -o "$TMP/joern-cli.zip"
unzip -q "$TMP/joern-cli.zip" -d "$TMP/x"

echo "=== [2/4] buang frontend selain c2cpg ==="
BEFORE=$(du -sm "$TMP/x/joern-cli" | cut -f1)
find "$TMP/x/joern-cli/frontends" -mindepth 1 -maxdepth 1 ! -name c2cpg -exec rm -rf {} +
AFTER=$(du -sm "$TMP/x/joern-cli" | cut -f1)
echo "  ${BEFORE} MB -> ${AFTER} MB"

echo "=== [3/4] sanity: parse C ==="
chmod +x "$TMP/x/joern-cli"/joern* 2>/dev/null || true
printf 'int f(char*s){char b[8];strcpy(b,s);return b[0];}\n' > "$TMP/t.c"
"$TMP/x/joern-cli/joern-parse" "$TMP/t.c" --output "$TMP/t.bin" >/dev/null
echo "  joern-parse OK"

echo "=== [4/4] lisensi + tar + unggah ==="
# Kita mendistribusikan ulang, jadi lisensinya wajib ikut (Apache-2.0 pasal 4). Rilis resmi
# Joern tidak menyertakannya di root, jadi ambil dari repo.
curl -fsSL https://raw.githubusercontent.com/joernio/joern/master/LICENSE \
     -o "$TMP/x/joern-cli/LICENSE"
cat > "$TMP/x/joern-cli/NOTICE" <<EOF
Joern ${VER} — https://github.com/joernio/joern
Apache License 2.0 (lihat LICENSE).

Redistribusi yang DIPANGKAS: seluruh frontend selain c2cpg (C/C++) dibuang untuk
menekan ukuran image (1,9 GB -> $(du -sm "$TMP/x/joern-cli" | cut -f1) MB). Tidak ada
berkas yang diubah isinya. Untuk bahasa lain, pakai rilis resmi.
EOF
tar -C "$TMP/x" -czf "$TMP/$OUT" joern-cli
rclone copy "$TMP/$OUT" "$DRIVE/" --progress
echo
echo "SELESAI -> $DRIVE/$OUT ($(du -h "$TMP/$OUT" | cut -f1))"
echo "Bagikan publik di UI Drive, lalu build dengan:"
echo "  docker compose -f API/docker-compose.yml build --build-arg JOERN_SLIM_GDRIVE_ID=<file-id>"
