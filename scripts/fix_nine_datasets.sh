#!/usr/bin/env bash
# fix_nine_datasets.sh — delete the two corrupt nine ml5120 datasets, rebuild them
# correctly (fixed patch regenerates func_token_lines), and upload.
#
# Corrupt because the OLD patch re-tokenized func_input_ids to a new length/tokenizer
# but never regenerated func_token_lines:
#   nine ml5120  (node+func nine)  — token_lines stuck at 1024 -> IndexError (length).
#   funcnine     (node base, func nine) — token_lines is base-tokenizer map (content).
#
# Scan-ambiguity guard: the funcnine patch prefix `unixcoder-base*` ALSO matches the
# `unixcoder-base-nine` bases. So build funcnine FIRST with only the node=base ml5120
# source present, THEN seed the nine ml1024 source and build nine ml5120 (its prefix
# `unixcoder-base-nine` is unambiguous).
#
# Usage (pod, project root, GPU pod):
#   bash scripts/fix_nine_datasets.sh
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
PROC_REMOTE="$REMOTE/data/processed/megavul"
PT_DIR="data/processed"
COMP="$(command -v pigz || echo gzip)"

# ── names ───────────────────────────────────────────────────────────────────────
CORRUPT_NINE5120_TAR="lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_lazy_20260613_203058.tar.gz"
CORRUPT_FUNCNINE_TAR="lm_dataset_megavul_multiclass_unixcoder-base_live_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_lazy_20260614_093911.tar.gz"
BASE5120_TAR="lm_dataset_megavul_multiclass_unixcoder-base_ft_ml5120_f40f2e964_s1600r42_lazy_20260527_075901.tar.gz"
NINE1024_TAR="lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_lazy_20260613_195029.tar.gz"

NINE5120="lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42"
FUNCNINE="lm_dataset_megavul_multiclass_unixcoder-base_live_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42"
BASE5120="lm_dataset_megavul_multiclass_unixcoder-base_ft_ml5120_f40f2e964_s1600r42"
NINE1024="lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42"

CFG_FUNCNINE="configs/data/megavul_multiclass_top25_node-unixcoder_func-unixcoder-nine_ml5120.yaml"
CFG_NINE5120="configs/data/megavul_multiclass_top25_node-unixcoder-nine_func-unixcoder-nine_ml5120.yaml"

mkdir -p "$PT_DIR"

rm_local() {  # remove a lazy dataset's local _meta.pt + _graphs/ by stem
  rm -rf "$PT_DIR/$1_meta.pt" "$PT_DIR/${1}_graphs"
}
seed() {      # download+extract a lazy base into PT_DIR if its _graphs dir is missing
  local stem="$1" tar="$2"
  if [[ -d "$PT_DIR/${stem}_graphs" ]]; then echo "[seed] $stem already present"; return; fi
  echo "[seed] $stem"
  rclone copy "$PROC_REMOTE/$tar" "$PT_DIR/" --progress
  tar -I "$COMP" -xf "$PT_DIR/$tar" -C "$PT_DIR" && rm -f "$PT_DIR/$tar"
}

echo "=== [1/5] delete corrupt datasets (Drive + local) ==="
rclone deletefile "$PROC_REMOTE/$CORRUPT_NINE5120_TAR" 2>/dev/null || echo "  (nine ml5120 tar already gone)"
rclone deletefile "$PROC_REMOTE/$CORRUPT_FUNCNINE_TAR" 2>/dev/null || echo "  (funcnine tar already gone)"
rm_local "$NINE5120"; rm_local "$FUNCNINE"
# also drop any nine base locally so the funcnine scan can't grab nine node embeddings
rm_local "$NINE1024"

echo "=== [2/5] seed node=base ml5120 source, build FUNCNINE (only base present) ==="
seed "$BASE5120" "$BASE5120_TAR"
# --delete-pt: drop the funcnine output locally right after its Drive upload.
bash scripts/cloud_process_datasets.sh --delete-pt "$CFG_FUNCNINE"
rm_local "$FUNCNINE"; rm_local "$BASE5120"   # free both before the nine build

echo "=== [3/5] seed nine ml1024 source, build NINE ml5120 ==="
seed "$NINE1024" "$NINE1024_TAR"
bash scripts/cloud_process_datasets.sh --delete-pt "$CFG_NINE5120"
rm_local "$NINE5120"; rm_local "$NINE1024"

echo "=== [4/5] free build scratch (raw CPGs not needed for training) ==="
rm -rf data/raw/megavul 2>/dev/null || true

echo "=== [5/5] verify new tars on Drive ==="
rclone lsf "$PROC_REMOTE/" | grep -E "(${NINE5120}|${FUNCNINE})_lazy_" || echo "  WARN: new tars not found — check upload logs above"
df -h /workspace 2>/dev/null | tail -1
echo "DONE. Rebuilt + uploaded: $FUNCNINE  and  $NINE5120 (local scratch cleared)"
