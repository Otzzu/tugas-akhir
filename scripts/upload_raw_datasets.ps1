#!/usr/bin/env pwsh
# Upload each raw dataset as a tar.gz to Google Drive, one at a time.
# Uses tar (not Compress-Archive — avoids 2 GB limit).

param(
    [string]$Remote   = "gdrive-mesach:tugas-akhir/data/raw",
    [string]$DataRoot = "$PSScriptRoot\..\data\raw",
    [string[]]$Datasets = @("bigvul", "megavul", "titanvul")
)

$DataRoot = Resolve-Path $DataRoot
$ts       = Get-Date -Format "yyyyMMdd_HHmmss"
$datasets = $Datasets

foreach ($ds in $datasets) {
    $srcDir  = Join-Path $DataRoot $ds
    $archive = Join-Path $DataRoot "${ds}_${ts}.tar.gz"

    if (-not (Test-Path $srcDir)) {
        Write-Warning "[$ds] Source dir not found: $srcDir — skipping"
        continue
    }

    Write-Host "`n=== [$ds] ===" -ForegroundColor Cyan
    Write-Host "Compressing $srcDir → $archive"
    $tarStart = Get-Date
    & tar -czf $archive -C $DataRoot $ds
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[$ds] tar failed (exit $LASTEXITCODE) — skipping upload"
        continue
    }
    $tarSec = [int]((Get-Date) - $tarStart).TotalSeconds
    $sizeMB = [math]::Round((Get-Item $archive).Length / 1MB, 1)
    Write-Host "Compressed in ${tarSec}s  size=${sizeMB} MB"

    Write-Host "Uploading to $Remote ..."
    $upStart = Get-Date
    & rclone copy $archive $Remote --progress
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[$ds] rclone upload failed (exit $LASTEXITCODE) — archive kept at $archive"
        continue
    }
    $upSec = [int]((Get-Date) - $upStart).TotalSeconds
    Write-Host "Uploaded in ${upSec}s"

    Write-Host "Removing local archive..."
    Remove-Item $archive -Force
    Write-Host "[$ds] Done."
}

Write-Host "`nAll datasets processed." -ForegroundColor Green
