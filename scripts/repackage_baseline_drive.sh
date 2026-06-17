#!/usr/bin/env bash
# repackage_baseline_drive.sh — slim a bloated baseline results tar on Drive.
# run_linevul_cloud.sh / run_losver_cloud.sh (pre-fix) tarred the ENTIRE output dir into the
# results tar, so the model .bin (CodeBERT/UniXcoder ~480MB each; LOSVER has TWO) got bundled
# in results/baselines — 800MB+ per run. This downloads the results tar, splits the model
# artifacts out, re-uploads:
#   slim results (everything EXCEPT *.bin/*.pt/*.safetensors/*.ckpt/optimizer*) -> results/baselines/<RUN>_results_slim.tar.gz
#   model artifacts -> checkpoints/baselines/<RUN>_weights.tar.gz  (only if not already there)
#
# NON-DESTRUCTIVE: the slim tar is uploaded under a NEW name (_results_slim), the original
# bloated <RUN>_results.tar.gz is left UNTOUCHED on Drive — delete it yourself after verifying.
#
# Generic for linevul + losver (and any baseline that bundled a model in results). Run on a pod
# (needs free space + bandwidth for the bloated download). Usage:
#   bash scripts/repackage_baseline_drive.sh <results_tar_name>
#   bash scripts/repackage_baseline_drive.sh linevul_megavul_ml1024_20260613_161040_results.tar.gz
set -uo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
ZIP="${1:?usage: repackage_baseline_drive.sh <results_tar_name>}"
RUN="${ZIP%_results.tar.gz}"
TMP="/tmp/baseline_repack/$RUN"; EX="$TMP/ex"
COMP="$(command -v pigz || echo gzip)"
rm -rf "$TMP"; mkdir -p "$EX"

echo "=== [1/5] download $ZIP ==="
rclone copy "$REMOTE/results/baselines/$ZIP" "$TMP/" --progress
[[ -f "$TMP/$ZIP" ]] || { echo "ERR: download failed"; exit 1; }

echo "=== [2/5] extract + inspect ==="
tar -I "$COMP" -xf "$TMP/$ZIP" -C "$EX"
echo "  before size:"; du -sh "$EX"
echo "  model artifacts found:"; find "$EX" \( -name '*.bin' -o -name '*.pt' -o -name '*.safetensors' -o -name '*.ckpt' -o -name 'optimizer*' \) -exec du -h {} + 2>/dev/null || true

echo "=== [3/5] save model -> checkpoints/baselines (only if absent there) ==="
mapfile -t MODELS < <(find "$EX" \( -name '*.bin' -o -name '*.pt' -o -name '*.safetensors' -o -name '*.ckpt' \))
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "  no model artifacts inside — already slim, nothing to strip. Aborting (no re-upload)."; exit 0
fi
if rclone ls "$REMOTE/checkpoints/baselines/${RUN}_weights.tar.gz" >/dev/null 2>&1; then
  echo "  ${RUN}_weights.tar.gz already in checkpoints/baselines — skip weights upload"
else
  REL=(); for m in "${MODELS[@]}"; do REL+=("$(realpath --relative-to="$EX" "$m")"); done
  tar -I "$COMP" -cf "$TMP/${RUN}_weights.tar.gz" -C "$EX" "${REL[@]}"
  rclone copy "$TMP/${RUN}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
  echo "  uploaded ${RUN}_weights.tar.gz -> checkpoints/baselines"
fi

echo "=== [4/5] repackage slim results (exclude model artifacts) under NEW name ==="
SLIM="${RUN}_results_slim.tar.gz"
tar -I "$COMP" -cf "$TMP/$SLIM" \
  --exclude='*.bin' --exclude='*.pt' --exclude='*.safetensors' --exclude='*.ckpt' --exclude='optimizer*' \
  -C "$EX" .
echo "  after size:"; ls -lh "$TMP/$SLIM"

echo "=== [5/5] upload slim results (original left intact, NOT overwritten) ==="
rclone copy "$TMP/$SLIM" "$REMOTE/results/baselines/" --progress
echo "DONE (non-destructive):"
echo "  slim  -> results/baselines/$SLIM   (NEW)"
echo "  model -> checkpoints/baselines/${RUN}_weights.tar.gz (only if it was absent)"
echo "  ORIGINAL bloated $ZIP left UNTOUCHED — delete it yourself after verifying:"
echo "    rclone lsl $REMOTE/results/baselines/ | grep ${RUN}"
echo "    rclone delete $REMOTE/results/baselines/$ZIP   # <- you run this manually"
echo "  then: rm -rf $TMP"
