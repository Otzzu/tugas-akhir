#!/usr/bin/env bash
# scripts/build_upload_datasets.sh
#
# Build Phase 8/10 datasets, upload to Google Drive, clean up.
# Order: download patch base (ml1024) → H4 → J2 → J3
# All 3 patch from the same ml1024 base (same node embedder; func_lm / func_max_length recomputed).
# Each dataset: build → pigz tar → upload → clean before starting the next.
#
# Prerequisites: run scripts/setup_cloud.sh first (needs torch, transformers, rclone).
# Usage:
#   bash scripts/build_upload_datasets.sh
#   bash scripts/build_upload_datasets.sh --skip-upload   # build only, no rclone (need patch base locally)

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

# Auto-detect flash_attn
USE_FA=$(python -c "import flash_attn; print('True')" 2>/dev/null || echo "False")
if [[ "$USE_FA" == "True" ]]; then
    info "flash_attn available — use_flash_attention=True"
else
    info "flash_attn not available — use_flash_attention=False"
fi

success "Environment OK"

# ── Dataset names ─────────────────────────────────────────────────────────────
BASE_DS="lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42"
H4_DS="lm_dataset_megavul_multiclass_unixcoder-base_ft_ml5120_f40f2e964_s1600r42"
J2_DS="lm_dataset_megavul_multiclass_unixcoder-base_live_jina-embeddings-v2-base-code_ft_ml5120_f40f2e964_s1600r42"
J3_DS="lm_dataset_megavul_multiclass_unixcoder-base_live_ModernBERT-base_ft_ml5120_f40f2e964_s1600r42"

mkdir -p "$PROC"

# ── Helpers ───────────────────────────────────────────────────────────────────

# Download a lazy dataset from gdrive megavul folder and extract it.
# Skipped if already present locally or if --skip-upload.
download_base_dataset() {
    local ds="$1"

    if $SKIP_UPLOAD; then
        info "--skip-upload: skipping base dataset download. Ensure $ds is present locally."
        return 0
    fi

    local local_meta="${PROC}/${ds}_meta.pt"
    local local_graphs="${PROC}/${ds}_graphs"

    if [[ -f "$local_meta" ]] || [[ -d "$local_graphs" ]]; then
        success "Base dataset already present: $ds"
        return 0
    fi

    info "Base dataset not found locally — searching Drive..."

    # Prefer _lazy_ marked tar; fall back to any tar for this dataset
    local remote_tar
    remote_tar=$(rclone lsf "$REMOTE" 2>/dev/null \
        | grep "^${ds}.*_lazy_.*\.tar\.gz$" | sort | tail -1 || true)
    if [[ -z "$remote_tar" ]]; then
        remote_tar=$(rclone lsf "$REMOTE" 2>/dev/null \
            | grep "^${ds}.*\.tar\.gz$" | sort | tail -1 || true)
        [[ -n "$remote_tar" ]] && info "No _lazy_-marked tar found — using: $remote_tar"
    fi

    if [[ -z "$remote_tar" ]]; then
        error "Base dataset not found on Drive: $ds  (path: $REMOTE)"
    fi

    info "Downloading: ${REMOTE}/${remote_tar}"
    local local_tar="${PROC}/${remote_tar}"
    rclone copy "${REMOTE}/${remote_tar}" "$PROC" --progress

    info "Extracting: $remote_tar"
    if $PIGZ; then
        tar -I pigz -xf "$local_tar" -C "$PROC"
    else
        tar -xzf "$local_tar" -C "$PROC"
    fi
    rm -f "$local_tar"
    success "Base dataset ready: $ds"
}

# Build → pigz tar → upload → clean (upload and clean skipped if --skip-upload).
build_upload_clean() {
    local label="$1"
    local ds="$2"
    local pretrained_lm="$3"
    local func_lm="$4"

    # ── Build ──────────────────────────────────────────────────────────────
    info "Building $label... (flash_attn=${USE_FA})"
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
    use_flash_attention=${USE_FA},
)
print('Build complete: ${label}')
"
    success "Built: $label"

    # ── Package ────────────────────────────────────────────────────────────
    local ts; ts=$(date +%Y%m%d_%H%M%S)
    local tarfile="${PROC}/${ds}_lazy_${ts}.tar.gz"
    info "Packaging: $label → $(basename "$tarfile")"
    if $PIGZ; then
        tar -I pigz -cf "$tarfile" -C "$PROC" "${ds}_meta.pt" "${ds}_graphs/"
    else
        tar -czf "$tarfile" -C "$PROC" "${ds}_meta.pt" "${ds}_graphs/"
    fi
    success "Packaged: $label"

    # ── Upload ─────────────────────────────────────────────────────────────
    if ! $SKIP_UPLOAD; then
        info "Uploading: $(basename "$tarfile")"
        rclone copy "$tarfile" "$REMOTE" --progress
        success "Uploaded: $label"
    else
        info "--skip-upload: keeping $tarfile locally"
    fi
    rm -f "$tarfile"

    # ── Clean ──────────────────────────────────────────────────────────────
    info "Cleaning local: $label"
    rm -f "${PROC}/${ds}_meta.pt"
    rm -rf "${PROC}/${ds}_graphs"
    success "Cleaned: $label"
}

# ── 0. Download patch base (ml1024 unixcoder) ─────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  [0/3] Download patch base (ml1024)${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
download_base_dataset "$BASE_DS"

# ── 1. H4 — unixcoder ml5120 ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  [1/3] H4 — unixcoder-base ml5120${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
build_upload_clean "H4" "$H4_DS" "microsoft/unixcoder-base" "microsoft/unixcoder-base"

# ── 2. J2 — jina-embeddings-v2-base-code func_lm ─────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  [2/3] J2 — jina-embeddings-v2-base-code ml5120${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
build_upload_clean "J2" "$J2_DS" "microsoft/unixcoder-base" "jinaai/jina-embeddings-v2-base-code"

# ── 3. J3 — ModernBERT-base func_lm ──────────────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  [3/3] J3 — ModernBERT-base ml5120${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
build_upload_clean "J3" "$J3_DS" "microsoft/unixcoder-base" "answerdotai/ModernBERT-base"

# ── 4. Clean base (all 3 done) ────────────────────────────────────────────────
echo ""
info "Cleaning base patch dataset (ml1024)"
rm -f "${PROC}/${BASE_DS}_meta.pt"
rm -rf "${PROC}/${BASE_DS}_graphs"
success "Base cleaned"

echo ""
success "All 3 datasets built, uploaded, and cleaned."
echo ""
echo "Training command:"
echo "  ./scripts/train_cloud.sh --init \\"
echo "    --config configs/ablation/phase9/I2_line_encoder.yaml \\"
echo "    --dataset ${BASE_DS} \\"
echo "    --config configs/ablation/phase9/I3_line_encoder_live.yaml \\"
echo "    --dataset ${BASE_DS} \\"
echo "    --config configs/ablation/phase8/H4_unixcoder_sliding_chunk1024_stride1024_winattn.yaml \\"
echo "    --dataset ${BASE_DS} \\"
echo "    --config configs/ablation/phase10/J2_jina_embeddings_v2_code.yaml \\"
echo "    --dataset ${H4_DS} \\"
echo "    --config configs/ablation/phase10/J3_modernbert_base.yaml \\"
echo "    --dataset ${H4_DS}"
