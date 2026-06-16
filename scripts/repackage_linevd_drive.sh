#!/usr/bin/env bash
# repackage_linevd_drive.sh — ONE-OFF cleanup of the bloated LineVD results zip on Drive.
# The old run bundled storage/processed (the GB graph cache) into the results zip (~14GB). This
# downloads it, splits the real results (outputs + log) from the processed cache, and re-uploads 2
# correctly-placed zips:
#   results (outputs + log, slim) -> results/baselines/  (OVERWRITES the bloated zip, same name)
#   processed graph cache         -> data/baselines/<RUN>_processed.tar.gz
#
# Run on a pod (needs ~30GB free + bandwidth for the 14GB download). Usage:
#   bash scripts/repackage_linevd_drive.sh [results_zip_name]
set -uo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
ZIP="${1:-linevd_megavul_ml1024_20260614_142914_results.tar.gz}"
RUN="${ZIP%_results.tar.gz}"
TMP="/tmp/linevd_repack"; EX="$TMP/ex"
COMP="$(command -v pigz || echo gzip)"
rm -rf "$TMP"; mkdir -p "$EX"

echo "=== [1/4] download $ZIP ==="
rclone copy "$REMOTE/results/baselines/$ZIP" "$TMP/" --progress
[[ -f "$TMP/$ZIP" ]] || { echo "ERR: download failed"; exit 1; }

echo "=== [2/4] extract + inspect ==="
tar -I "$COMP" -xf "$TMP/$ZIP" -C "$EX"
echo "  top-level contents:"; ls -la "$EX"
du -sh "$EX"/* 2>/dev/null || true

echo "=== [3/4] repackage: results (everything EXCEPT processed) + processed (separate) ==="
# slim results = all members except the processed graph cache
tar -I "$COMP" -cf "$TMP/${RUN}_results.tar.gz" -C "$EX" --exclude=./processed --exclude=processed .
if [[ -d "$EX/processed" ]]; then
  tar -I "$COMP" -cf "$TMP/${RUN}_processed.tar.gz" -C "$EX" processed
else
  echo "  WARN: no ./processed dir found inside the zip (already slim? inspect $EX)"
fi
echo "  new sizes:"; ls -lh "$TMP"/*.tar.gz

echo "=== [4/4] upload: slim results overwrites old (same name) + processed -> data/baselines ==="
rclone copyto "$TMP/${RUN}_results.tar.gz" "$REMOTE/results/baselines/$ZIP" --progress
[[ -f "$TMP/${RUN}_processed.tar.gz" ]] && \
  rclone copy "$TMP/${RUN}_processed.tar.gz" "$REMOTE/data/baselines/" --progress
echo "DONE: $ZIP slimmed in results/baselines/ ; processed -> data/baselines/${RUN}_processed.tar.gz"
echo "  (verify with: rclone lsl $REMOTE/results/baselines/$ZIP ; then rm -rf $TMP)"
