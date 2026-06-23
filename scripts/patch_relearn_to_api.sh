#!/usr/bin/env bash
# scripts/patch_relearn_to_api.sh
#
# Patch the RELEARN (domain-incremental task-B = BigVul + TitanVul) dataset into the API
# bundle format and re-upload to Drive, so the API /relearn flow can pull it from MinIO like
# any other dataset. Thin wrapper around patch_dataset_to_api.sh with relearn baked in.
#
# Source tar on Drive:  gdrive-mesach:tugas-akhir/data/processed/relearn/<DATASET>.tar.gz
# Output bundle:        gdrive-mesach:tugas-akhir/api_datasets/relearn/<DATASET>_api.tar.gz
#
# VOCAB — do NOT pass --vocab here. The relearn .pt on Drive is already vocab-aligned: it was
# built with task-A's canonical CWE vocab (run_relearn_experiment.py:_align_relearn_vocab), so
# its .pt class_names are the SAME 26-class order as the main megavul dataset. Letting
# patch_dataset_to_api.sh generate cwe_vocab.json from the .pt's own class_names yields the
# correct 26-class canonical map whose ids line up with the 26-class model head.
# (configs/ablation/relearn/taskA_cwe_vocab.json is the 193-class alignment INPUT, not the
# final 26-class API vocab — passing it as --vocab would re-introduce the 193-class bug.)
#
# Run on a cloud/Linux pod with rclone (gdrive-mesach) configured:
#   ./scripts/patch_relearn_to_api.sh
#   ./scripts/patch_relearn_to_api.sh --dataset <other_relearn_name>
#
# Seed it after: copy api_datasets/relearn/<DATASET>_api.tar.gz to MinIO as
# s3://datasets/relearn.tar.gz (materialize extracts it, subdirs preserved).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="lm_dataset_relearn_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1 (only --dataset <name> supported)" >&2; exit 1 ;;
    esac
done

exec "${HERE}/patch_dataset_to_api.sh" --dataset "$DATASET" --source relearn
