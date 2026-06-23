#!/usr/bin/env bash
# scripts/patch_cil_to_api.sh
#
# Patch the CLASS-INCREMENTAL (CIL) task-B dataset (megavul_cil, 10 new CWE) into the API
# bundle format and re-upload to Drive, so the API /relearn (class-incremental) flow can pull
# it from MinIO like any other dataset.
#
# Unlike relearn (domain-IL), the cil .pt on Drive is a STANDALONE 11-class dataset
# (benign=0, 10 new CWE = 1..10). It must first be remapped to the unified 36-class space
# (task-A 26 classes 0..25, then the 10 new at 26..35) by scripts/patch_cil_labels.py before
# the API bundle is built. We therefore cannot delegate to patch_dataset_to_api.sh — that
# script never runs the in-place label remap. The bundle's cwe_vocab.json is generated from
# the PATCHED meta's 36 class_names, giving the correct 36-class map.
#
#   Drive source:  gdrive-mesach:tugas-akhir/data/processed/relearn/<DS>_lazy.tar.gz
#   patch step:    scripts/patch_cil_labels.py  (y(j) -> 25+j, class_names -> 36, idempotent)
#   Output bundle: gdrive-mesach:tugas-akhir/api_datasets/megavul_cil/<DS>_api.tar.gz
#
# Run on a cloud/Linux pod (from repo root) with rclone (gdrive-mesach) + torch available:
#   ./scripts/patch_cil_to_api.sh
#
# Seed it after: copy api_datasets/megavul_cil/<DS>_api.tar.gz to MinIO as
# s3://datasets/megavul_cil.tar.gz (materialize extracts it, subdirs preserved).

set -euo pipefail

GDRIVE_REMOTE="gdrive-mesach:tugas-akhir"
PROC="data/processed"
COMP="$(command -v pigz || echo gzip)"
PY="$(command -v python3 || command -v python)"

# DS is the inner dataset name (no _lazy marker) that patch_cil_labels.py expects under PROC.
DS="lm_dataset_megavul_cil_multiclass_unixcoder-base_ft_ml1024"
REMOTE_SUBDIR="${GDRIVE_REMOTE}/data/processed/relearn"     # cil tar lives under relearn/
API_REMOTE="${GDRIVE_REMOTE}/api_datasets/megavul_cil"      # dedicated API-format target
mkdir -p "$PROC"

# Reclaim disk on ANY exit: staging dir, downloaded tar, built bundle. Lazy extract is GBs.
STAGE="" ; OUT="" ; remote_tar=""
cleanup() {
    [[ -n "$STAGE" ]]      && rm -rf "$STAGE"
    [[ -n "$remote_tar" ]] && rm -f "${PROC}/${remote_tar}"
    [[ -n "$OUT" ]]        && rm -f "${PROC}/${OUT}"
    true
}
trap cleanup EXIT

# ── 1. Locate + download the cil tar (any storage marker after the DS name) ────────────
echo "[1/5] Locating cil bundle for $DS under $REMOTE_SUBDIR ..."
remote_tar=$(rclone lsf "$REMOTE_SUBDIR" 2>/dev/null | grep -E "^${DS}(_lazy)?(_[0-9]{8}_[0-9]{6})?\.tar\.gz$" | sort | tail -1 || true)
[[ -z "$remote_tar" ]] && { echo "ERR: no cil tar for $DS in $REMOTE_SUBDIR" >&2; exit 1; }
echo "  found: $remote_tar"
rclone copy "${REMOTE_SUBDIR}/${remote_tar}" "$PROC" --progress

# ── 2. Extract into PROC so the files land exactly where patch_cil_labels.py expects ───
echo "[2/5] Extracting -> ${PROC}/${DS}_{meta.pt,graphs/} ..."
tar -I "$COMP" -xf "${PROC}/${remote_tar}" -C "$PROC"
rm -f "${PROC}/${remote_tar}"
[[ -f "${PROC}/${DS}_meta.pt" ]] || { echo "ERR: ${PROC}/${DS}_meta.pt missing after extract" >&2; exit 1; }

# ── 3. Remap labels to the unified 36-class space (idempotent) ─────────────────────────
echo "[3/5] Remapping cil labels to 36-class (patch_cil_labels.py) ..."
"$PY" scripts/patch_cil_labels.py

# ── 4. cwe_vocab.json from the PATCHED 36-class meta + build the API bundle ─────────────
echo "[4/5] Building API bundle ..."
STAGE="$(mktemp -d -p "$PROC")"
mkdir -p "${STAGE}/processed"
"$PY" - "${PROC}/${DS}_meta.pt" "${STAGE}/cwe_vocab.json" <<'PY'
import sys, json, torch
meta, out = sys.argv[1], sys.argv[2]
m = torch.load(meta, weights_only=False, map_location="cpu")
cn = m.get("class_names")
if not cn:
    sys.exit("ERR: meta has no class_names")
if len(cn) != 36:
    sys.exit(f"ERR: expected 36 class_names after patch, got {len(cn)} — patch_cil_labels.py did not run")
json.dump({c: i for i, c in enumerate(cn)}, open(out, "w"), indent=2)
print(f"  vocab: generated from {meta.split('/')[-1]} ({len(cn)} classes)")
PY
# mv (not cp) the patched meta + graphs into the bundle — same filesystem, instant, no GB copy.
mv "${PROC}/${DS}_meta.pt"  "${STAGE}/processed/"
mv "${PROC}/${DS}_graphs"   "${STAGE}/processed/"

OUT="${DS}_api.tar.gz"
tar -C "$STAGE" -I "$COMP" -cf "${PROC}/${OUT}" cwe_vocab.json processed
rm -rf "$STAGE"; STAGE=""
echo "  built ${PROC}/${OUT} ($(du -h "${PROC}/${OUT}" | cut -f1))"

# ── 5. Upload to Drive ─────────────────────────────────────────────────────────────────
echo "[5/5] Uploading -> ${API_REMOTE}/${OUT}"
rclone copy "${PROC}/${OUT}" "$API_REMOTE" --progress
rm -f "${PROC}/${OUT}"

echo "Done. API-format CIL bundle (36-class) -> ${API_REMOTE}/${OUT}"
echo "Disk free on data dir: $(df -h "$PROC" | awk 'NR==2{print $4}')"
echo "Seed it: copy that object to MinIO as s3://datasets/megavul_cil.tar.gz."
