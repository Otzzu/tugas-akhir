#!/usr/bin/env bash
# Download datasets from Google Drive, build .pt caches, upload each .pt.
#
# Usage:
#   bash scripts/cloud_process_datasets.sh [flags] <config> [<config> ...]
#
# Flags:
#   --delete-pt        Delete .pt after upload (default: keep).
#   --force-rebuild    Rebuild .pt from scratch.
#   --clear-after N    After processing the Nth config, delete ALL local .pt files
#                      before continuing. Useful when configs share a node LM:
#                      process group-1 (same node LM) → upload → clear → process group-2.
#
# Example — 4 configs, 2 per node LM, clear between groups:
#   bash scripts/cloud_process_datasets.sh --clear-after 2 \
#     configs/data/titanvul_top25.yaml \
#     configs/data/titanvul_top25_codet5p.yaml \
#     configs/data/titanvul_top25_codet5p_node.yaml \
#     configs/data/titanvul_top25_codet5p_full.yaml

set -euo pipefail

REMOTE_RAW="gdrive-mesach:tugas-akhir/data/raw"
REMOTE_PT="gdrive-mesach:tugas-akhir/data/processed"
DATA_DIR="data/raw"
PT_DIR="data/processed"
TS=$(date +"%Y%m%d_%H%M%S")
DELETE_PT=false
FORCE_REBUILD=false
CLEAR_AFTER=0   # 0 = disabled

# ── Parse flags ────────────────────────────────────────────────────────────────
CONFIGS=()
skip_next=false
args=("$@")
for i in "${!args[@]}"; do
    if $skip_next; then skip_next=false; continue; fi
    case "${args[$i]}" in
        --delete-pt)     DELETE_PT=true ;;
        --force-rebuild) FORCE_REBUILD=true ;;
        --clear-after)   CLEAR_AFTER="${args[$((i+1))]}"; skip_next=true ;;
        *)               CONFIGS+=("${args[$i]}") ;;
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
    # Find latest matching tar on remote (sorted chronologically by name)
    local tar
    tar=$(rclone ls "$REMOTE_RAW" 2>/dev/null | grep "${name}_" | grep "\.tar\.gz" | awk '{print $2}' | sort | tail -1)
    if [[ -z "$tar" ]]; then
        echo "[error] No tar found for '$name' in $REMOTE_RAW" >&2
        return 1
    fi
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

needed_sources() {
    grep -h "source:" "$@" 2>/dev/null | awk '{print $2}' | sort -u
}

if [[ $# -gt 0 ]]; then
    for src in $(needed_sources "$@"); do
        download_if_missing "$src"
    done
else
    download_if_missing bigvul
    download_if_missing megavul
    download_if_missing titanvul
fi

# ── 2. Process + upload ────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "No config specified. Extracted datasets only."
    exit 0
fi

REBUILD_FLAG=""
[[ "$FORCE_REBUILD" == "true" ]] && REBUILD_FLAG="--force-rebuild"

config_idx=0
for CONFIG in "$@"; do
    config_idx=$((config_idx + 1))
    echo ""
    echo "=== [$config_idx/$#] Processing: $CONFIG ==="

    # Extract dataset source from config for upload subdirectory
    dataset_src=$(grep "^  source:" "$CONFIG" 2>/dev/null | awk '{print $2}' | head -1)
    remote_dest="$REMOTE_PT/${dataset_src:-misc}"
    echo "Upload destination: $remote_dest"

    before=$(ls "$PT_DIR"/*.pt 2>/dev/null | sort || true)

    PYTHONPATH=src python scripts/process_dataset.py --config "$CONFIG" --device cuda $REBUILD_FLAG

    after=$(ls "$PT_DIR"/*.pt 2>/dev/null | sort || true)
    new_pts=$(comm -13 <(echo "$before") <(echo "$after"))

    if [[ -z "$new_pts" ]]; then
        echo "[warn] No new .pt file — skipping upload"
    else
        for pt_file in $new_pts; do
            stem=$(basename "$pt_file" .pt)
            archive="${stem}_${TS}.tar.gz"
            echo "Compressing $pt_file → $archive ..."
            tar -cf - -C "$PT_DIR" "$(basename "$pt_file")" \
                | pv -s "$(du -sb "$pt_file" | awk '{print $1}')" \
                | gzip > "$archive"
            echo "Uploading $archive → $remote_dest ..."
            rclone copy "$archive" "$remote_dest" --progress
            rm -f "$archive"
            if [[ "$DELETE_PT" == "true" ]]; then
                rm -f "$pt_file"
                echo "Deleted local $pt_file"
            fi
        done
    fi

    # --clear-after N: delete ALL local .pt after the Nth config to free disk space
    if [[ "$CLEAR_AFTER" -gt 0 && "$config_idx" -eq "$CLEAR_AFTER" ]]; then
        echo ""
        echo "=== --clear-after $CLEAR_AFTER reached — deleting all local .pt files ==="
        local_pts=$(ls "$PT_DIR"/*.pt 2>/dev/null || true)
        if [[ -n "$local_pts" ]]; then
            echo "$local_pts" | xargs rm -f
            echo "Cleared: $local_pts"
        else
            echo "(no .pt files to clear)"
        fi
    fi
done

echo ""
echo "All done."
