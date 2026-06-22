#!/usr/bin/env bash
# scripts/patch_dataset_to_api.sh
#
# Convert a cloud-format lazy dataset into the API materialize bundle format and re-upload
# to Drive, so it integrates with the API exactly like megavul_mini.
#
#   cloud format (current on Drive):  <name>_meta.pt + <name>_graphs/  at TOP LEVEL, no vocab
#   API format (what materialize_dataset expects):
#       cwe_vocab.json                       (top level)
#       processed/<name>_meta.pt             (under processed/)
#       processed/<name>_graphs/...          (subdir preserved)
#
# The API bundle is uploaded as <name>_api.tar.gz into a DEDICATED folder for API-format
# datasets on Drive: api_datasets/<source>/  (kept separate from the cloud-format datasets
# under data/processed/<source>/). Seed it into the API by copying that object to MinIO as
# <dataset_id>.tar.gz (materialize downloads s3://datasets/<dataset_id>.tar.gz, subdirs preserved).
#
# Run on a cloud/Linux pod with rclone configured (gdrive-mesach):
#   ./scripts/patch_dataset_to_api.sh \
#       --dataset lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42 \
#       --source  megavul \
#       [--vocab data/raw/megavul/cwe_vocab.json]    # default: data/raw/<source>/cwe_vocab.json

set -euo pipefail

GDRIVE_REMOTE="gdrive-mesach:tugas-akhir"
PROC="data/processed"
COMP="$(command -v pigz || echo gzip)"   # parallel gzip if present (CLAUDE.md), else gzip

DATASET="" ; SOURCE="" ; VOCAB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --source)  SOURCE="$2";  shift 2 ;;
        --vocab)   VOCAB="$2";   shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$DATASET" || -z "$SOURCE" ]] && { echo "usage: --dataset <name> --source <source> [--vocab <path>]" >&2; exit 1; }
[[ -z "$VOCAB" ]] && VOCAB="data/raw/${SOURCE}/cwe_vocab.json"

REMOTE_SUBDIR="${GDRIVE_REMOTE}/data/processed/${SOURCE}"   # cloud-format datasets (download source)
API_REMOTE="${GDRIVE_REMOTE}/api_datasets/${SOURCE}"        # API-format patched datasets (dedicated upload target)
mkdir -p "$PROC"

# ── 1. Locate + download the cloud lazy tar (prefer _lazy_ marker, else legacy name) ───
echo "[1/4] Locating cloud bundle for $DATASET under $REMOTE_SUBDIR ..."
remote_tar=$(rclone lsf "$REMOTE_SUBDIR" 2>/dev/null | grep "^${DATASET}_lazy_.*\.tar\.gz$" | sort | tail -1 || true)
[[ -z "$remote_tar" ]] && remote_tar=$(rclone lsf "$REMOTE_SUBDIR" 2>/dev/null | grep -E "^${DATASET}(_[0-9]{8}_[0-9]{6})?\.tar\.gz$" | sort | tail -1 || true)
[[ -z "$remote_tar" ]] && { echo "ERR: no cloud tar for $DATASET in $REMOTE_SUBDIR" >&2; exit 1; }
echo "  found: $remote_tar"
rclone copy "${REMOTE_SUBDIR}/${remote_tar}" "$PROC" --progress

# ── 2. Extract cloud tar (top-level meta+graphs) straight into staging/processed/ ─────
echo "[2/4] Repackaging into API format ..."
STAGE="$(mktemp -d)"
mkdir -p "${STAGE}/processed"
tar -I "$COMP" -xf "${PROC}/${remote_tar}" -C "${STAGE}/processed"
rm -f "${PROC}/${remote_tar}"

# ── 3. Add the vocab at top level (API bundle requires cwe_vocab.json) ────────────────
[[ ! -f "$VOCAB" ]] && { echo "ERR: vocab not found: $VOCAB (pass --vocab)" >&2; exit 1; }
cp "$VOCAB" "${STAGE}/cwe_vocab.json"
echo "  vocab: $VOCAB ($(python3 -c "import json;print(len(json.load(open('$VOCAB'))))" 2>/dev/null || echo '?') classes)"

# ── 4. Build the API bundle + upload to Drive ─────────────────────────────────────────
OUT="${DATASET}_api.tar.gz"
tar -C "$STAGE" -I "$COMP" -cf "${PROC}/${OUT}" cwe_vocab.json processed
rm -rf "$STAGE"
echo "  built ${PROC}/${OUT} ($(du -h "${PROC}/${OUT}" | cut -f1))"
echo "[4/4] Uploading -> ${API_REMOTE}/${OUT}"
rclone copy "${PROC}/${OUT}" "$API_REMOTE" --progress
rm -f "${PROC}/${OUT}"

echo "Done. API-format bundle -> ${API_REMOTE}/${OUT}"
echo "Seed it: copy that object to MinIO as s3://datasets/<dataset_id>.tar.gz (materialize extracts it, subdirs preserved)."
