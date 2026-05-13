#!/usr/bin/env bash
# scripts/train_cloud.sh
#
# Flexible cloud training pipeline: setup → download datasets → train → evaluate → upload.
# Each --config must be paired with a --dataset (same position order).
# Datasets are downloaded once and cached; safe to repeat same dataset across configs.
# Upload runs in background — next training starts immediately after evaluate.
#
# Flags:
#   --init          Force run setup_cloud.sh + reinstall rclone.conf (fresh server)
#   --skip          Skip setup_cloud.sh + rclone setup entirely (already configured)
#   --clean-every N Delete dataset .pt cache after every Nth run (0 = never, default: 0)
#                   Use N=total_runs to clean only after last run (safe for shared datasets).
#                   Clean happens inside background upload after rclone finishes.
#   (default: auto-detect — runs setup only if uv/torch missing or rclone.conf absent)
#
# Usage:
#   ./scripts/train_cloud.sh --init \          # fresh server, everything needs setup
#     --config configs/lmgat_codebert/multiclass_mtl_livable_f1stop.yaml \
#     --dataset lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10
#
#   ./scripts/train_cloud.sh --skip --clean-every 4 \   # clean after every 4th run
#     --config configs/ablation/phase1/A1_lmgat.yaml \
#     --dataset lm_dataset_megavul_... \
#     ...
#
# Dataset name = zip filename WITHOUT .zip on gdrive-mesach:tugas-akhir/
# e.g. lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10
#      → gdrive-mesach:tugas-akhir/lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.zip
#      → extracted to data/processed/lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10/

set -euo pipefail

GDRIVE_REMOTE="gdrive-mesach:tugas-akhir"
PROCESSED_DIR="data/processed"
CHECKPOINTS_DIR="checkpoints"
RESULTS_DIR="results"

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }

# ─── Argument parsing ─────────────────────────────────────────────────────────
CONFIG_LIST=()
DATASET_LIST=()
FLAG_INIT=false    # --init: force run setup_cloud.sh + rclone setup regardless of state
FLAG_SKIP=false    # --skip: skip setup_cloud.sh + rclone setup entirely
CLEAN_EVERY=0      # --clean-every N: delete dataset .pt after every Nth run (0 = never)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG_LIST+=("$2");  shift 2 ;;
        --dataset)       DATASET_LIST+=("$2"); shift 2 ;;
        --init)          FLAG_INIT=true;       shift ;;
        --skip)          FLAG_SKIP=true;       shift ;;
        --clean-every)   CLEAN_EVERY="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) error "Unknown argument: $1"; exit 1 ;;
    esac
done

if $FLAG_INIT && $FLAG_SKIP; then
    error "--init and --skip are mutually exclusive"
    exit 1
fi

if [[ ${#CONFIG_LIST[@]} -eq 0 ]]; then
    error "No configs provided. Use --config <yaml> --dataset <name>"
    exit 1
fi

if [[ ${#CONFIG_LIST[@]} -ne ${#DATASET_LIST[@]} ]]; then
    error "--config count (${#CONFIG_LIST[@]}) != --dataset count (${#DATASET_LIST[@]}). Must be paired."
    exit 1
fi

info "Configs to run: ${#CONFIG_LIST[@]}"
for i in "${!CONFIG_LIST[@]}"; do
    echo "  [$((i+1))] ${CONFIG_LIST[$i]}  |  dataset: ${DATASET_LIST[$i]}"
done
echo ""

# ─── 1. rclone check ─────────────────────────────────────────────────────────
check_rclone() {
    if $FLAG_SKIP; then
        warn "rclone setup skipped (--skip)"
        return 0
    fi

    info "Checking rclone..."

    if ! command -v rclone &>/dev/null; then
        error "rclone not in PATH. Install rclone first."
        exit 1
    fi

    # --init: force re-install rclone.conf even if it exists
    if $FLAG_INIT || [[ ! -f ~/.config/rclone/rclone.conf ]]; then
        if $FLAG_INIT; then
            info "Forcing rclone setup (--init)..."
        else
            warn "rclone.conf not found. Setting up from rclone.zip..."
        fi
        if [[ ! -f rclone.zip ]]; then
            error "rclone.zip not found in project root. Cannot set up rclone."
            exit 1
        fi
        mkdir -p ~/.config/rclone
        unzip -j -o rclone.zip -d ~/.config/rclone/
        success "rclone.conf installed"
    else
        success "rclone.conf exists"
    fi

    if ! rclone listremotes 2>/dev/null | grep -q "gdrive-mesach:"; then
        error "Remote 'gdrive-mesach' not found in rclone config. Check rclone.zip contents."
        exit 1
    fi

    # Quick connectivity test
    if ! rclone lsd "${GDRIVE_REMOTE}/" &>/dev/null; then
        error "Cannot reach ${GDRIVE_REMOTE}. Check credentials / network."
        exit 1
    fi

    success "rclone connected to ${GDRIVE_REMOTE}"
}

# ─── 2. Python / env check ───────────────────────────────────────────────────
check_setup() {
    if $FLAG_SKIP; then
        warn "Python env setup skipped (--skip)"
        return 0
    fi

    info "Checking Python environment..."

    local need_setup=false

    if $FLAG_INIT; then
        info "Forcing setup_cloud.sh (--init)..."
        need_setup=true
    else
        if ! command -v uv &>/dev/null; then
            warn "uv not found"
            need_setup=true
        elif ! python -c "import torch, torch_geometric" &>/dev/null 2>&1; then
            warn "torch / torch_geometric not importable"
            need_setup=true
        fi
    fi

    if $need_setup; then
        if [[ ! -f scripts/setup_cloud.sh ]]; then
            error "scripts/setup_cloud.sh not found"
            exit 1
        fi
        chmod +x scripts/setup_cloud.sh
        ./scripts/setup_cloud.sh
    else
        success "Python environment ready"
    fi

    # Report CUDA
    local cuda
    cuda=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
    if [[ "$cuda" == "True" ]]; then
        success "CUDA available"
    else
        warn "CUDA not available — training will be slow"
    fi
}

# ─── 3. Dataset download (cached) ────────────────────────────────────────────
DOWNLOADED_DATASETS=()

download_dataset() {
    local dataset="$1"
    local config="${2:-}"
    # Remote names include timestamp (_YYYYMMDD_HHMMSS); local .pt files do not.
    # Strip timestamp for local existence check.
    local local_name
    local_name=$(echo "$dataset" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')
    local local_dir="${PROCESSED_DIR}/${local_name}"

    # Detect storage mode from config (default: inmemory)
    local storage="inmemory"
    if [[ -n "$config" && -f "$config" ]]; then
        local cfg_storage
        cfg_storage=$(grep -E '^\s+storage:\s+' "$config" | awk '{print $2}' | head -1 || true)
        [[ -n "$cfg_storage" ]] && storage="$cfg_storage"
    fi

    # Already downloaded this session
    for d in "${DOWNLOADED_DATASETS[@]:-}"; do
        [[ "$d" == "$dataset" ]] && return 0
    done

    if [[ -d "$local_dir" ]] || compgen -G "${PROCESSED_DIR}/${local_name}*.pt" > /dev/null 2>&1; then
        success "Dataset already exists: $local_name"
        DOWNLOADED_DATASETS+=("$dataset")
        return 0
    fi
    info "Not found locally (checked: ${local_dir}/ and ${PROCESSED_DIR}/${local_name}*.pt)"

    info "Downloading dataset: $dataset (storage=$storage)"
    mkdir -p "$PROCESSED_DIR"

    # Try 1: .zip at gdrive root (legacy)
    local remote_zip="${GDRIVE_REMOTE}/${dataset}.zip"
    if rclone lsf "$remote_zip" &>/dev/null 2>&1; then
        local local_zip="${PROCESSED_DIR}/${dataset}.zip"
        rclone copy "$remote_zip" "$PROCESSED_DIR" --progress
        unzip -o "$local_zip" -d "$PROCESSED_DIR"
        rm -f "$local_zip"
        success "Dataset ready: $dataset"
        DOWNLOADED_DATASETS+=("$dataset")
        return 0
    fi

    # Try 2: .tar.gz in data/processed/<source>/ subdir
    # Prefer tar with matching storage marker (_lazy_ / _inmemory_); fall back to legacy (no marker).
    local remote_proc="${GDRIVE_REMOTE}/data/processed"
    local source
    source=$(echo "$dataset" | sed 's/lm_dataset_\([^_]*\)_.*/\1/')
    local remote_tar remote_subdir
    for remote_subdir in "${remote_proc}/${source}" "${remote_proc}"; do
        # Preferred: tar with storage marker
        remote_tar=$(rclone lsf "$remote_subdir" 2>/dev/null \
            | grep "^${dataset}.*_${storage}_.*\.tar\.gz$" | sort | tail -1 || true)
        # Fallback: any tar for this dataset (legacy, no marker)
        if [[ -z "$remote_tar" ]]; then
            remote_tar=$(rclone lsf "$remote_subdir" 2>/dev/null \
                | grep "^${dataset}.*\.tar\.gz$" | sort | tail -1 || true)
            [[ -n "$remote_tar" ]] && warn "No ${storage}-marked tar found — using legacy: $remote_tar"
        fi
        if [[ -n "$remote_tar" ]]; then
            local local_tar="${PROCESSED_DIR}/${remote_tar}"
            info "Found: ${remote_subdir}/${remote_tar}"
            rclone copy "${remote_subdir}/${remote_tar}" "$PROCESSED_DIR" --progress
            tar -xzf "$local_tar" -C "$PROCESSED_DIR"
            rm -f "$local_tar"
            success "Dataset ready: $dataset"
            DOWNLOADED_DATASETS+=("$dataset")
            return 0
        fi
    done

    error "Dataset not found on gdrive: $dataset (tried zip at root, tar.gz in ${remote_proc}/${source}/ and ${remote_proc}/)"
    exit 1
}

# ─── 4. Train ────────────────────────────────────────────────────────────────
run_train() {
    local config="$1"
    info "Training: $config"
    PYTHONPATH=src python -m gnn_vuln.train --config "$config"
    success "Training done: $config"
}

# ─── 5. Evaluate ─────────────────────────────────────────────────────────────
run_evaluate() {
    local model_dir="$1"
    local model_id="$2"

    # Find best checkpoint (best_*.pt)
    local ckpt
    ckpt=$(find "$model_dir" -maxdepth 1 -name "best_*.pt" | head -1)
    if [[ -z "$ckpt" ]]; then
        warn "No best_*.pt found in $model_dir — skipping evaluate"
        return 0
    fi

    local config_yaml="${model_dir}/config.yaml"
    if [[ ! -f "$config_yaml" ]]; then
        warn "config.yaml not found in $model_dir — skipping evaluate"
        return 0
    fi

    info "Evaluating: $model_id"
    PYTHONPATH=src python -m gnn_vuln.evaluate \
        --checkpoint "$ckpt" \
        --config "$config_yaml"
    success "Evaluate done: $model_id"
}

# ─── 6. Zip and upload ───────────────────────────────────────────────────────
# Takes: model_id dataset do_clean log_file
# Runs upload then optionally cleans — called in background.
# All output redirected to log_file by caller; errors logged with step context.
upload_run() {
    local model_id="$1"
    local dataset="$2"
    local do_clean="$3"
    local ckpt_zip="${model_id}_checkpoints.zip"
    local res_zip="${model_id}_results.zip"

    set -euo pipefail  # ensure errors propagate inside background subshell

    local upload_ok=false  # guard: local files only removed if ALL uploads succeed

    info "[upload:$model_id] Zipping checkpoints..."
    zip -r "$ckpt_zip" "${CHECKPOINTS_DIR}/${model_id}" \
        || { error "[upload:$model_id] zip checkpoints FAILED"; exit 1; }

    info "[upload:$model_id] Uploading checkpoints -> ${GDRIVE_REMOTE}/checkpoints/"
    rclone copy "$ckpt_zip" "${GDRIVE_REMOTE}/checkpoints/" --progress \
        || { error "[upload:$model_id] rclone checkpoints FAILED"; exit 1; }
    rm -f "$ckpt_zip"

    if [[ -d "${RESULTS_DIR}/${model_id}" ]]; then
        info "[upload:$model_id] Zipping results..."
        zip -r "$res_zip" "${RESULTS_DIR}/${model_id}" \
            || { error "[upload:$model_id] zip results FAILED"; exit 1; }
        info "[upload:$model_id] Uploading results -> ${GDRIVE_REMOTE}/results/"
        rclone copy "$res_zip" "${GDRIVE_REMOTE}/results/" --progress \
            || { error "[upload:$model_id] rclone results FAILED"; exit 1; }
        rm -f "$res_zip"
    else
        warn "[upload:$model_id] No results dir — skipping results upload"
    fi

    upload_ok=true
    success "[upload:$model_id] Upload done"

    # Clean local checkpoints and results ONLY after all uploads confirmed success
    if [[ "$upload_ok" == "true" ]]; then
        info "[upload:$model_id] Cleaning local checkpoints..."
        rm -rf "${CHECKPOINTS_DIR:?}/${model_id}" \
            || warn "[upload:$model_id] Failed to remove checkpoints dir"
        if [[ -d "${RESULTS_DIR}/${model_id}" ]]; then
            info "[upload:$model_id] Cleaning local results..."
            rm -rf "${RESULTS_DIR:?}/${model_id}" \
                || warn "[upload:$model_id] Failed to remove results dir"
        fi
    else
        warn "[upload:$model_id] Upload failed — local checkpoints and results preserved for recovery"
    fi

    # Clean dataset .pt cache after upload completes (not before)
    if [[ "$do_clean" == "true" ]]; then
        local local_name
        local_name=$(echo "$dataset" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')
        info "[upload:$model_id] Cleaning dataset files for: $local_name"
        local pts
        pts=$(compgen -G "${PROCESSED_DIR}/${local_name}*.pt" 2>/dev/null || true)
        if [[ -n "$pts" ]]; then
            echo "$pts" | xargs rm -f
            info "[upload:$model_id] Deleted .pt files: $pts"
        else
            warn "[upload:$model_id] No .pt files found to clean for $local_name"
        fi
        # lazy storage: also remove _graphs/ directories
        local graphs_dir="${PROCESSED_DIR}/${local_name}_graphs"
        if [[ -d "$graphs_dir" ]]; then
            rm -rf "$graphs_dir"
            info "[upload:$model_id] Deleted graphs dir: $graphs_dir"
        fi
    fi

    success "[upload:$model_id] All post-train cleanup done"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
check_rclone
check_setup

UPLOAD_PIDS=()
UPLOAD_LOGS=()
UPLOAD_IDS=()
RUN_COUNT=0

for i in "${!CONFIG_LIST[@]}"; do
    CONFIG="${CONFIG_LIST[$i]}"
    DATASET="${DATASET_LIST[$i]}"
    N=$((i + 1))
    TOTAL=${#CONFIG_LIST[@]}

    echo ""
    echo -e "${CYAN}════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Run $N/$TOTAL: $(basename $CONFIG)${NC}"
    echo -e "${CYAN}════════════════════════════════════════${NC}"

    download_dataset "$DATASET" "$CONFIG"

    # Snapshot latest checkpoint dir before training
    BEFORE=$(ls -dt ${CHECKPOINTS_DIR}/* 2>/dev/null | head -1 || echo "")

    run_train "$CONFIG"

    # Find the new checkpoint dir (newest after training)
    MODEL_DIR=$(ls -dt ${CHECKPOINTS_DIR}/* 2>/dev/null | head -1)
    if [[ -z "$MODEL_DIR" || "$MODEL_DIR" == "$BEFORE" ]]; then
        error "Could not detect new checkpoint dir after training"
        exit 1
    fi
    MODEL_ID=$(basename "$MODEL_DIR")
    success "New model: $MODEL_ID"

    run_evaluate "$MODEL_DIR" "$MODEL_ID"

    # Decide whether to clean .pt on this run
    RUN_COUNT=$((RUN_COUNT + 1))
    DO_CLEAN="false"
    if [[ "$CLEAN_EVERY" -gt 0 && $((RUN_COUNT % CLEAN_EVERY)) -eq 0 ]]; then
        DO_CLEAN="true"
        info "Run $RUN_COUNT: will clean dataset .pt after upload (--clean-every $CLEAN_EVERY)"
    fi

    # Upload in background — next training starts immediately
    UPLOAD_LOG="/tmp/upload_${MODEL_ID}.log"
    upload_run "$MODEL_ID" "$DATASET" "$DO_CLEAN" >"$UPLOAD_LOG" 2>&1 &
    UPLOAD_PIDS+=($!)
    UPLOAD_LOGS+=("$UPLOAD_LOG")
    UPLOAD_IDS+=("$MODEL_ID")
    info "Upload started in background (PID ${UPLOAD_PIDS[-1]}, log: $UPLOAD_LOG): $MODEL_ID"

    echo -e "${GREEN}  DONE $N/$TOTAL: $MODEL_ID (upload running in background)${NC}"
done

# Wait for all background uploads before exit
if [[ ${#UPLOAD_PIDS[@]} -gt 0 ]]; then
    echo ""
    info "Waiting for ${#UPLOAD_PIDS[@]} background upload(s) to finish..."
    ALL_UPLOADS_OK=true
    for i in "${!UPLOAD_PIDS[@]}"; do
        pid="${UPLOAD_PIDS[$i]}"
        log="${UPLOAD_LOGS[$i]}"
        mid="${UPLOAD_IDS[$i]}"
        if wait "$pid"; then
            success "Upload OK: $mid"
            rm -f "$log"
        else
            error "Upload FAILED: $mid (PID $pid)"
            error "---- upload log: $log ----"
            cat "$log" >&2
            error "---- end log ----"
            ALL_UPLOADS_OK=false
        fi
    done
    if ! $ALL_UPLOADS_OK; then
        error "One or more uploads failed — check logs above"
        exit 1
    fi
fi

echo ""
success "All $TOTAL run(s) complete."
