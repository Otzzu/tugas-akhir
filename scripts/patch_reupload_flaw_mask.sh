#!/usr/bin/env bash
# scripts/patch_reupload_flaw_mask.sh
#
# Download a built dataset tar from Drive, drop the METHOD-node flaw flags
# (scripts/patch_flaw_mask.py), re-tar, and re-upload with a fresh timestamp so
# train_cloud.sh's `sort | tail -1` picks the corrected version automatically.
#
# The METHOD node spans the whole function in MegaVul CPGs, so the old
# flaw_line_mask marked line 1 (signature) as a flaw in ~90% of vulnerable
# functions and inflated localization Top-1. This corrects already-built
# datasets without a full re-embed rebuild.
#
# Usage (on a pod with rclone + torch + this repo):
#   bash scripts/patch_reupload_flaw_mask.sh <dataset_name> [<dataset_name> ...]
#
# Pass dataset names WITHOUT the _lazy_/_inmemory_ marker, timestamp, or .tar.gz.
# Example:
#   bash scripts/patch_reupload_flaw_mask.sh \
#     lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42 \
#     lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42

set -euo pipefail

GDRIVE_REMOTE="gdrive-mesach:tugas-akhir"
PROC="data/processed"
PY="${PYTHON:-python}"
COMP="$(command -v pigz || echo gzip)"
mkdir -p "$PROC"

command -v rclone >/dev/null || { echo "rclone not in PATH"; exit 1; }

patch_one() {
    local ds="$1"
    local source ts subdir tar
    source=$(echo "$ds" | sed 's/lm_dataset_\([^_]*\)_.*/\1/')
    ts=$(date +%Y%m%d_%H%M%S)

    # Locate the tar on Drive: prefer storage-marked, fall back to legacy.
    # relearn + cil datasets both live under data/processed/relearn/ regardless of
    # their name prefix (megavul_cil), so search it too.
    tar=""
    for subdir in "${GDRIVE_REMOTE}/data/processed/${source}" \
                  "${GDRIVE_REMOTE}/data/processed/relearn" \
                  "${GDRIVE_REMOTE}/data/processed"; do
        tar=$(rclone lsf "$subdir" 2>/dev/null | grep -E "^${ds}_(lazy|inmemory)_.*\.tar\.gz$" | sort | tail -1 || true)
        if [[ -z "$tar" ]]; then
            tar=$(rclone lsf "$subdir" 2>/dev/null | grep -E "^${ds}(_[0-9]{8}_[0-9]{6})?\.tar\.gz$" | sort | tail -1 || true)
        fi
        [[ -n "$tar" ]] && break
    done
    [[ -z "$tar" ]] && { echo "!! NOT FOUND on Drive: $ds"; return 1; }

    echo "==> [$ds] downloading $subdir/$tar"
    rclone copy "$subdir/$tar" "$PROC" --progress
    tar -I "$COMP" -xf "$PROC/$tar" -C "$PROC"
    rm -f "$PROC/$tar"

    # Detect storage layout produced by the extract.
    local target marker
    if [[ -d "$PROC/${ds}_graphs" ]]; then
        target="$PROC/${ds}_graphs"; marker="lazy"
    elif [[ -f "$PROC/${ds}.pt" ]]; then
        target="$PROC/${ds}.pt"; marker="inmemory"
    else
        echo "!! extracted layout unknown for $ds (no ${ds}_graphs/ or ${ds}.pt)"; return 1
    fi

    echo "==> [$ds] patching ($marker) $target"
    $PY scripts/patch_flaw_mask.py --data "$target"

    local newtar="${ds}_${marker}_${ts}.tar.gz"
    echo "==> [$ds] re-taring -> $newtar"
    if [[ "$marker" == "lazy" ]]; then
        tar -I "$COMP" -cf "$PROC/$newtar" -C "$PROC" "${ds}_graphs" "${ds}_meta.pt"
    else
        tar -I "$COMP" -cf "$PROC/$newtar" -C "$PROC" "${ds}.pt"
    fi

    echo "==> [$ds] uploading -> $subdir/$newtar"
    rclone copy "$PROC/$newtar" "$subdir/" --progress
    echo "==> [$ds] DONE. New tar wins by newest timestamp; old tar left in place."
}

[[ $# -ge 1 ]] || { echo "usage: $0 <dataset_name> [<dataset_name> ...]"; exit 1; }
for ds in "$@"; do patch_one "$ds"; done
echo "All datasets patched + reuploaded."
