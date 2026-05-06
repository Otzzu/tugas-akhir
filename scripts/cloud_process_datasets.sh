#!/usr/bin/env bash
# Download megavul + titanvul from Google Drive, extract, then build .pt caches.
#
# Usage:
#   bash scripts/cloud_process_datasets.sh [--keep-pt] <config> [<config> ...]
#
# Flags:
#   --delete-pt      After upload, also delete local .pt to free disk space (default: keep).
#   --force-rebuild  Skip patch fast-path; rebuild .pt from scratch (passed to process_dataset.py).
#
# Examples:
#   bash scripts/cloud_process_datasets.sh configs/data/megavul_multiclass_top25.yaml
#   bash scripts/cloud_process_datasets.sh --delete-pt configs/data/megavul_multiclass_top25.yaml
#   bash scripts/cloud_process_datasets.sh --force-rebuild --delete-pt configs/data/megavul_multiclass_top25.yaml

set -euo pipefail

REMOTE_RAW="gdrive-mesach:tugas-akhir/data/raw"
REMOTE_PT="gdrive-mesach:tugas-akhir/data/processed"
MEGAVUL_TAR="megavul_20260505_120148.tar.gz"
TITANVUL_TAR="titanvul_20260505_120148.tar.gz"
DATA_DIR="data/raw"
PT_DIR="data/processed"
TS=$(date +"%Y%m%d_%H%M%S")
DELETE_PT=false
FORCE_REBUILD=false

# Parse flags
CONFIGS=()
for arg in "$@"; do
    case "$arg" in
        --delete-pt)     DELETE_PT=true ;;
        --force-rebuild) FORCE_REBUILD=true ;;
        *)               CONFIGS+=("$arg") ;;
    esac
done
set -- "${CONFIGS[@]+"${CONFIGS[@]}"}"

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

# Only download datasets actually referenced in the given configs
needed_sources() {
    grep -h "source:" "$@" 2>/dev/null | awk '{print $2}' | sort -u
}

if [[ $# -gt 0 ]]; then
    for src in $(needed_sources "$@"); do
        download_if_missing "$src"
    done
else
    download_if_missing megavul
    download_if_missing titanvul
fi

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

    REBUILD_FLAG=""
    [[ "$FORCE_REBUILD" == "true" ]] && REBUILD_FLAG="--force-rebuild"
    PYTHONPATH=src python scripts/process_dataset.py --config "$CONFIG" --device cuda $REBUILD_FLAG

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
        if [[ "$DELETE_PT" == "true" ]]; then
            rm -f "$pt_file"
            echo "Uploaded and cleaned: $archive + $pt_file"
        else
            echo "Uploaded and cleaned: $archive (keeping $pt_file)"
        fi
    done
done

echo ""
echo "All done."
