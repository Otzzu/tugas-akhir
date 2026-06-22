#!/usr/bin/env pwsh
# scripts/upload_cil_dataset.ps1
# Zip the prebuilt megavul_cil lazy dataset (the _graphs dir + _meta.pt) and upload
# to gdrive at the exact path/name run_relearn_cil_experiment.py --setup downloads from
# (CIL_PT_DIR = data/processed/megavul_cil, CIL_PT_ARCHIVE = ..._lazy.tar.gz).
# Contents are tarred at top level so `tar -xf -C data/processed` on the pod lands them
# back at data/processed/<...>_graphs/ + <...>_meta.pt.
# Run from project root.

$ErrorActionPreference = "Stop"

$BASE    = "lm_dataset_megavul_cil_multiclass_unixcoder-base_ft_ml1024"
$GRAPHS  = "${BASE}_graphs"          # lazy per-graph dir
$META    = "${BASE}_meta.pt"         # meta (slices, class_names)
$ARCHIVE = "${BASE}_lazy.tar.gz"     # MUST match CIL_PT_ARCHIVE in run_relearn_cil_experiment.py
$PROC    = "data/processed"
$REMOTE  = "gdrive-mesach:tugas-akhir/data/processed/relearn"  # both continual task-B datasets live here

# ── 0. Already on Drive? then nothing to do (don't re-upload) ───────────────
$onDrive = rclone lsf "$REMOTE/$ARCHIVE" 2>$null
if ($onDrive) {
    Write-Host "Already on Drive: $REMOTE/$ARCHIVE -- nothing to do (uniform with relearn layout)."
    exit 0
}

# ── 0b. Verify local sources present (only needed if we must upload) ────────
if (-not (Test-Path "$PROC/$GRAPHS")) { throw "Missing $PROC/$GRAPHS" }
if (-not (Test-Path "$PROC/$META"))   { throw "Missing $PROC/$META" }

# ── 1. Tar + gzip (Windows bsdtar; pigz is not available on Windows) ─────────
Write-Host "[1/2] Compressing $GRAPHS + $META -> $ARCHIVE ..."
tar -czf "$PROC/$ARCHIVE" -C $PROC $GRAPHS $META
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
Write-Host ("  archive size: {0:N1} MB" -f ((Get-Item "$PROC/$ARCHIVE").Length / 1MB))

# ── 2. Upload to Drive ──────────────────────────────────────────────────────
Write-Host "`n[2/2] Uploading -> $REMOTE"
rclone copy "$PROC/$ARCHIVE" $REMOTE --progress
if ($LASTEXITCODE -ne 0) { throw "rclone failed" }

Remove-Item "$PROC/$ARCHIVE"
Write-Host "`nDone. $ARCHIVE -> $REMOTE/"
Write-Host "run_relearn_cil_experiment.py --setup will now find it."
