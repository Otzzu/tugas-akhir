#!/usr/bin/env bash
# Download datasets from Google Drive, build .pt caches, upload each .pt.
#
# Usage:
#   bash scripts/cloud_process_datasets.sh [flags] <config> [<config> ...]
#
# Flags:
#   --delete-pt        Delete .pt after upload (default: keep).
#   --force-rebuild    Rebuild .pt from scratch.
#   --clear-after N    Clear ALL local .pt after the Nth config.
#   --clear-every N    Clear ALL local .pt after every Nth config (repeating).
#                      Use to free space between node-LM groups.
#
# Example — 8 configs, 2 per node-LM per filter, clear every 2:
#   bash scripts/cloud_process_datasets.sh --clear-every 2 \
#     configs/data/bigvul_multiclass_top25_node-unixcoder_func-unixcoder.yaml \
#     configs/data/bigvul_multiclass_top25_node-unixcoder_func-codet5p.yaml \
#     configs/data/bigvul_multiclass_top25_node-codet5p_func-unixcoder.yaml \
#     configs/data/bigvul_multiclass_top25_node-codet5p_func-codet5p.yaml \
#     configs/data/bigvul_multiclass_top10_node-unixcoder_func-unixcoder.yaml \
#     configs/data/bigvul_multiclass_top10_node-unixcoder_func-codet5p.yaml \
#     configs/data/bigvul_multiclass_top10_node-codet5p_func-unixcoder.yaml \
#     configs/data/bigvul_multiclass_top10_node-codet5p_func-codet5p.yaml

set -euo pipefail

REMOTE_RAW="gdrive-mesach:tugas-akhir/data/raw"
REMOTE_PT="gdrive-mesach:tugas-akhir/data/processed"
DATA_DIR="data/raw"
PT_DIR="data/processed"
TS=$(date +"%Y%m%d_%H%M%S")
DELETE_PT=false
FORCE_REBUILD=false
CLEAR_AFTER=0   # clear once at index N (0 = disabled)
CLEAR_EVERY=0   # clear every N configs (0 = disabled)

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
        --clear-every)   CLEAR_EVERY="${args[$((i+1))]}"; skip_next=true ;;
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
    tar=$(rclone ls "$REMOTE_RAW" 2>/dev/null | grep "${name}_" | grep "\.tar\.gz" | awk '{print $2}' | sort | tail -1 || true)
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
            if command -v pv &>/dev/null; then
                tar -cf - -C "$PT_DIR" "$(basename "$pt_file")" \
                    | pv -s "$(du -sb "$pt_file" | awk '{print $1}')" \
                    | gzip > "$archive"
            else
                tar -czf "$archive" -C "$PT_DIR" "$(basename "$pt_file")"
            fi
            echo "Uploading $archive → $remote_dest ..."
            rclone copy "$archive" "$remote_dest" --progress
            rm -f "$archive"
            if [[ "$DELETE_PT" == "true" ]]; then
                rm -f "$pt_file"
                echo "Deleted local $pt_file"
            fi
        done
    fi

    # Clear .pt files: --clear-after N (once) or --clear-every N (repeating)
    should_clear=false
    [[ "$CLEAR_AFTER" -gt 0 && "$config_idx" -eq "$CLEAR_AFTER" ]] && should_clear=true
    [[ "$CLEAR_EVERY" -gt 0 && $(( config_idx % CLEAR_EVERY )) -eq 0 ]] && should_clear=true

    if $should_clear; then
        echo ""
        echo "=== Clearing local .pt files after config $config_idx ==="
        local_pts=$(ls "$PT_DIR"/*.pt 2>/dev/null || true)
        if [[ -n "$local_pts" ]]; then
            echo "$local_pts" | xargs rm -f
            echo "Cleared $(echo "$local_pts" | wc -l) .pt file(s)"
        else
            echo "(no .pt files to clear)"
        fi
    fi
done

echo ""
echo "All done."
