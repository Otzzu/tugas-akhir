#!/usr/bin/env bash
# build_hybrid_ml5120_datasets.sh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build the hybrid (O1) continual task-B datasets at ml5120 by RE-TOKENIZING the
# existing ml1024 nine bases — no node re-embedding (the patch fast-path in
# dataset_lm.py reuses node features from the ml1024 base and only re-tokenizes the
# func tokens at 5120). CPU-only + fast. Then tar + upload to Drive so the hybrid
# continual runs download a prebuilt ml5120 .pt instead of rebuilding on every pod.
#
# megavul ml5120 nine already exists on Drive (H10/O1 trained on it) — NOT rebuilt here.
# Only relearn + megavul_cil need building.
#
# Run (cloud, Linux, after rclone.conf is in place):
#   bash scripts/build_hybrid_ml5120_datasets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src RELEARN_NINE=1
COMP="$(command -v pigz || echo gzip)"
DRIVE="gdrive-mesach:tugas-akhir/data/processed/relearn"
PROC="data/processed"
mkdir -p "$PROC"

fetch() {   # $1 = tar name on Drive, $2 = glob to test if already extracted
  if ! ls $PROC/$2 >/dev/null 2>&1; then
    echo "== download $1"
    rclone copy "$DRIVE/$1" "$PROC" --progress
    tar -I "$COMP" -xf "$PROC/$1" -C "$PROC"
  else
    echo "== $2 already present, skip download"
  fi
}

# 1. ml1024 nine bases (the re-tokenize source) — relearn + cil.
# PATCHED tars (exact-line flaw_line_mask); the older untimestamped tars have the inflated mask.
fetch "lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_lazy_20260708_055002.tar.gz" \
      "lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml1024*_meta.pt"
fetch "lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml1024_lazy_20260709_163416.tar.gz" \
      "lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml1024*_meta.pt"

# 2. cil needs its cwe_vocab.json (dataset constructor guard) + labels patched to 26..35
mkdir -p data/raw/megavul_cil
cp configs/ablation/relearn/cil/megavul_cil_cwe_vocab.json data/raw/megavul_cil/cwe_vocab.json
python scripts/patch_cil_labels.py   # idempotent — no-op if the ml1024 cil base is already 36-class

# 3. build ml5120 by re-tokenizing (source relearn / megavul_cil; naive cfg = same data params)
python -m gnn_vuln.data.build_pt --config configs/ablation/relearn/O1_relearn_naive_nine.yaml
python -m gnn_vuln.data.build_pt --config configs/ablation/relearn/cil/O1_cil_naive_nine.yaml

# 4. tar the produced ml5120 .pt dirs (lazy = <name>_meta.pt + <name>_graphs/) + upload
pack_upload() {   # $1 = source prefix, $2 = output tar name
  local meta name
  meta="$(ls $PROC/${1}*ml5120*_meta.pt | head -1)"
  name="$(basename "$meta" _meta.pt)"
  echo "== packing $name -> $2"
  tar -C "$PROC" -cf - "${name}_meta.pt" "${name}_graphs" | $COMP > "$PROC/$2"
  rclone copy "$PROC/$2" "$DRIVE" --progress
  echo "uploaded $2 -> $DRIVE"
}
# timestamped names so the orchestrators' newest-tar pick sorts these above any older upload
TS="$(date +%Y%m%d_%H%M%S)"
pack_upload "lm_dataset_relearn_multiclass"     "lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_lazy_${TS}.tar.gz"
pack_upload "lm_dataset_megavul_cil_multiclass" "lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml5120_lazy_${TS}.tar.gz"

echo "DONE — relearn + cil ml5120 built + uploaded. Hybrid continual can now download them."
