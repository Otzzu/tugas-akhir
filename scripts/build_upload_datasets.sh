#!/usr/bin/env bash
# scripts/build_upload_datasets.sh
#
# Build Phase 8/10 datasets, upload to Google Drive, clean up.
# Order: H4 (ml5120 unixcoder) → J2 (jina func_lm) → J3 (ModernBERT func_lm)
# H4 kept local during J2/J3 build (patching reuses node embeddings — much faster).
#
# Prerequisites: run scripts/setup_cloud.sh first (needs torch, transformers, rclone).
# Usage:
#   bash scripts/build_upload_datasets.sh
#   bash scripts/build_upload_datasets.sh --skip-upload   # build only, no rclone

set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir/data/processed/megavul"
PROC="data/processed"
SKIP_UPLOAD=false

for arg in "$@"; do
    [[ "$arg" == "--skip-upload" ]] && SKIP_UPLOAD=true
done

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; exit 1; }

# ── Preflight checks ──────────────────────────────────────────────────────────
info "Checking environment..."
python -c "import torch, transformers, torch_geometric" 2>/dev/null \
    || error "torch/transformers/torch_geometric not importable. Run scripts/setup_cloud.sh first."

if ! $SKIP_UPLOAD; then
    command -v rclone &>/dev/null \
        || error "rclone not in PATH. Run scripts/setup_cloud.sh first."
    [[ -f ~/.config/rclone/rclone.conf ]] \
        || error "rclone.conf not found. Run scripts/train_cloud.sh --init first to install it."
    rclone lsd "gdrive-mesach:" &>/dev/null \
        || error "Cannot reach gdrive-mesach. Check rclone credentials."
fi

command -v pigz &>/dev/null || { info "pigz not found — falling back to gzip"; PIGZ=false; }
PIGZ=${PIGZ:-true}

success "Environment OK"

# ── Helpers ───────────────────────────────────────────────────────────────────
package_upload_clean() {
    local ds="$1"
    local keep="${2:-false}"
    local ts; ts=$(date +%Y%m%d_%H%M%S)
    local tarfile="${PROC}/${ds}_lazy_${ts}.tar.gz"

    info "Packaging: $ds"
    if $PIGZ; then
        tar -I pigz -cf "$tarfile" -C "$PROC" "${ds}_meta.pt" "${ds}_graphs/"
    else
        tar -czf "$tarfile" -C "$PROC" "${ds}_meta.pt" "${ds}_graphs/"
    fi

    if ! $SKIP_UPLOAD; then
        info "Uploading: $(basename "$tarfile")"
        rclone copy "$tarfile" "$REMOTE" --progress
        success "Uploaded: $ds"
    else
        info "--skip-upload: keeping $tarfile locally"
    fi
    rm -f "$tarfile"

    if [[ "$keep" != "true" ]]; then
        info "Cleaning local: $ds"
        rm -f "${PROC}/${ds}_meta.pt"
        rm -rf "${PROC}/${ds}_graphs"
        success "Cleaned: $ds"
    else
        info "Keeping local (needed for patching): $ds"
    fi
}

build_dataset() {
    local label="$1"
    local pretrained_lm="$2"
    local func_lm="$3"

    info "Building $label dataset..."
    PYTHONPATH=src python -c "
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
CodeBERTGraphDataset(
    root='data',
    source='megavul',
    mode='multiclass',
    pretrained_lm='${pretrained_lm}',
    func_lm='${func_lm}',
    add_func_tokens=True,
    func_max_length=5120,
    filter_top25_dangerous=True,
    max_per_class=1600,
    resample_seed=42,
    max_nodes=2500,
    storage='lazy',
    embedder_device='cuda',
    use_flash_attention=True,
)
print('Build complete: ${label}')
"
    success "Built: $label"
}

# ── Dataset names ─────────────────────────────────────────────────────────────
H4_DS="lm_dataset_megavul_multiclass_unixcoder-base_ft_ml5120_f40f2e964_s1600r42"
J2_DS="lm_dataset_megavul_multiclass_unixcoder-base_live_jina-embeddings-v2-base-code_ft_ml5120_f40f2e964_s1600r42"
J3_DS="lm_dataset_megavul_multiclass_unixcoder-base_live_ModernBERT-base_ft_ml5120_f40f2e964_s1600r42"

mkdir -p "$PROC"

# ── 1. H4 — unixcoder ml5120 (keep local; J2/J3 patch from it) ───────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  [1/3] H4 — unixcoder-base ml5120${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
build_dataset "H4" "microsoft/unixcoder-base" "microsoft/unixcoder-base"
package_upload_clean "$H4_DS" true   # keep=true → J2/J3 patch from this

# ── 2. J2 — jina-embeddings-v2-base-code func_lm ─────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  [2/3] J2 — jina-embeddings-v2-base-code ml5120${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
build_dataset "J2" "microsoft/unixcoder-base" "jinaai/jina-embeddings-v2-base-code"
package_upload_clean "$J2_DS"

# ── 3. J3 — ModernBERT-base func_lm ──────────────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  [3/3] J3 — ModernBERT-base ml5120${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
build_dataset "J3" "microsoft/unixcoder-base" "answerdotai/ModernBERT-base"
package_upload_clean "$J3_DS"

# ── 4. Clean H4 (J2/J3 done, no longer needed) ───────────────────────────────
echo ""
info "Cleaning H4 (patching source, no longer needed)"
rm -f "${PROC}/${H4_DS}_meta.pt"
rm -rf "${PROC}/${H4_DS}_graphs"
success "H4 cleaned"

echo ""
success "All 3 datasets built, uploaded, and cleaned."
echo ""
echo "Training command (no patching needed):"
echo "  ./scripts/train_cloud.sh --init \\"
echo "    --config configs/ablation/phase9/I2_line_encoder.yaml \\"
echo "    --dataset lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42 \\"
echo "    --config configs/ablation/phase9/I3_line_encoder_live.yaml \\"
echo "    --dataset lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42 \\"
echo "    --config configs/ablation/phase8/H4_unixcoder_sliding_chunk1024_stride1024_winattn.yaml \\"
echo "    --dataset ${H4_DS} \\"
echo "    --config configs/ablation/phase10/J2_jina_embeddings_v2_code.yaml \\"
echo "    --dataset ${J2_DS} \\"
echo "    --config configs/ablation/phase10/J3_modernbert_base.yaml \\"
echo "    --dataset ${J3_DS}"
