#!/usr/bin/env pwsh
# scripts/upload_checkpoints.ps1
# Compress (tar.gz), upload to GDrive, remove archive for missing checkpoint folders.
#
# Usage:
#   .\scripts\upload_checkpoints.ps1
#   .\scripts\upload_checkpoints.ps1 -DryRun

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$CHECKPOINTS_DIR = "checkpoints"
$REMOTE = "gdrive-mesach:tugas-akhir/checkpoints"

$MISSING = @(
    "20260429_091918_lmgat_codebert_multiclass",
    "20260429_095918_lmgat_mcs_multiclass",
    "20260429_121124_lmgat_seq_multiclass",
    "20260429_125637_lmgat_waves_seq_multiclass",
    "20260429_135046_lmgat_seq_multiclass",
    "20260429_203915_lmggnn_multiclass",
    "20260430_004221_lmggnn_multiclass",
    "20260501_035449_lmgat_dualflow_multiclass",
    "20260501_050001_lmgat_codebert_mtl_multiclass",
    "20260501_072750_lmgat_codebert_mtl_multiclass",
    "20260501_085445_lmgat_codebert_multiclass",
    "20260501_120840_lmgat_mcs_multiclass",
    "20260501_150638_lmgat_seq_multiclass"
)

$total = $MISSING.Count
$i = 0

foreach ($id in $MISSING) {
    $i++
    $src = Join-Path $CHECKPOINTS_DIR $id
    $archive = "${id}.tar.gz"

    Write-Host ""
    Write-Host "[$i/$total] $id" -ForegroundColor Cyan

    if (-not (Test-Path $src)) {
        Write-Host "  SKIP: local folder not found -> $src" -ForegroundColor Yellow
        continue
    }

    # Compress
    Write-Host "  Compressing -> $archive"
    if (-not $DryRun) {
        tar -czf $archive -C $CHECKPOINTS_DIR $id
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: tar failed (exit $LASTEXITCODE)" -ForegroundColor Red
            continue
        }
        $sizeMB = [math]::Round((Get-Item $archive).Length / 1MB, 1)
        Write-Host "  Archive size: ${sizeMB} MB"
    }

    # Upload
    Write-Host "  Uploading -> $REMOTE/$archive"
    if (-not $DryRun) {
        rclone copy $archive $REMOTE --progress
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: rclone upload failed (exit $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "  Keeping archive: $archive" -ForegroundColor Yellow
            continue
        }
    }

    # Remove archive
    Write-Host "  Removing local archive"
    if (-not $DryRun) {
        Remove-Item $archive -Force
    }

    Write-Host "  Done" -ForegroundColor Green
}

Write-Host ""
Write-Host "All done. [$total folders processed]" -ForegroundColor Green
