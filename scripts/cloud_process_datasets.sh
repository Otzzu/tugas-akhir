#!/usr/bin/env bash
# Download megavul + titanvul from Google Drive, extract, then build .pt caches.
# Run once per process_dataset config — pass CONFIG as argument.
#
# Usage (run twice, once per config):
#   bash scripts/cloud_process_datasets.sh configs/lmgat_hcdfgat/megavul_multiclass_top25.yaml
#   bash scripts/cloud_process_datasets.sh configs/lmgat_hcdfgat/titanvul_multiclass_owasp.yaml
#
# Or run both in sequence (no GPU needed for dataset build — embeddings need GPU):
#   bash scripts/cloud_process_datasets.sh \
#       configs/lmgat_hcdfgat/megavul_multiclass_top25.yaml \
#       configs/lmgat_hcdfgat/titanvul_multiclass_owasp.yaml

set -euo pipefail

REMOTE_RAW="gdrive-mesach:tugas-akhir/data/raw"
REMOTE_PT="gdrive-mesach:tugas-akhir/data/processed"
MEGAVUL_TAR="megavul_20260505_120148.tar.gz"
TITANVUL_TAR="titanvul_20260505_120148.tar.gz"
DATA_DIR="data/raw"
PT_DIR="data/processed"
TS=$(date +"%Y%m%d_%H%M%S")

# ── 1. Download ────────────────────────────────────────────────────────────────
download_if_missing() {
    local name="$1"
    if [[ -d "$DATA_DIR/$name" ]]; then
        echo "[skip] $name already extracted"
        return
    fi
    local tar="${name}_20260505_120148.tar.gz"
    if [[ ! -f "$tar" ]]; then
        echo "Downloading $tar ..."
        rclone copy "$REMOTE_RAW/$tar" . --progress
    fi
    echo "Extracting $tar ..."
    tar -xzf "$tar" -C "$DATA_DIR"
    rm -f "$tar"
    echo "$name ready."
}

mkdir -p "$DATA_DIR"
download_if_missing megavul
download_if_missing titanvul

# ── 2. Process ─────────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "No config specified. Extracted datasets only. Run:"
    echo "  PYTHONPATH=src python scripts/process_dataset.py --config <yaml> --device cuda"
    exit 0
fi

for CONFIG in "$@"; do
    echo ""
    echo "=== Processing: $CONFIG ==="

    # Snapshot .pt files before processing
    before=$(ls "$PT_DIR"/*.pt 2>/dev/null | sort || true)

    PYTHONPATH=src python scripts/process_dataset.py --config "$CONFIG" --device cuda

    # Find newly created .pt files
    after=$(ls "$PT_DIR"/*.pt 2>/dev/null | sort || true)
    new_pts=$(comm -13 <(echo "$before") <(echo "$after"))

    if [[ -z "$new_pts" ]]; then
        echo "[warn] No new .pt file detected — skipping upload"
        continue
    fi

    for pt_file in $new_pts; do
        stem=$(basename "$pt_file" .pt)
        archive="${stem}_${TS}.tar.gz"
        echo "Zipping $pt_file -> $archive ..."
        tar -czf "$archive" -C "$PT_DIR" "$(basename "$pt_file")"
        echo "Uploading $archive to $REMOTE_PT ..."
        rclone copy "$archive" "$REMOTE_PT" --progress
        rm -f "$archive"
        echo "Uploaded and cleaned: $archive"
    done
done

echo ""
echo "All done."
