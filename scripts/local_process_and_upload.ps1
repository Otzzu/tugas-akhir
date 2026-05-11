#!/usr/bin/env pwsh
# Process .pt datasets locally and upload to Google Drive.
# Mirrors cloud_process_datasets.sh but skips download (data already local).
#
# Usage:
#   .\scripts\local_process_and_upload.ps1 [flags] <config> [<config> ...]
#
# Flags:
#   -ForceRebuild   Rebuild .pt from scratch
#   -DeletePt       Delete local .pt after upload
#   -Device <str>   cuda or cpu (default: cuda)
#
# Example:
#   .\scripts\local_process_and_upload.ps1 `
#     configs/data/titanvul_multiclass_top25_node-codet5p_func-unixcoder.yaml `
#     configs/data/titanvul_multiclass_top25_node-codet5p_func-codet5p.yaml

param(
    [Parameter(Position = 0, ValueFromRemainingArguments)]
    [string[]]$Configs,
    [switch]$ForceRebuild,
    [switch]$DeletePt,
    [Parameter()][string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$REMOTE_PT = "gdrive-mesach:tugas-akhir/data/processed"
$PT_DIR    = "data/processed"
$TS        = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path $PT_DIR | Out-Null

if (-not $Configs) {
    Write-Error "No configs specified. Pass config paths as arguments."
    exit 1
}

$configIdx = 0
foreach ($config in $Configs) {
    $configIdx++
    Write-Host ""
    Write-Host "=== [$configIdx/$($Configs.Count)] Processing: $config ==="

    # Extract dataset source from config for upload subdirectory
    $datasetSrc = (Select-String -Path $config -Pattern "^\s+source:\s+(\S+)" |
        Select-Object -First 1).Matches.Groups[1].Value
    if (-not $datasetSrc) { $datasetSrc = "misc" }
    $remoteDest = "$REMOTE_PT/$datasetSrc"
    Write-Host "Upload destination: $remoteDest"

    # Snapshot .pt files before processing
    $before = @(Get-ChildItem "$PT_DIR/*.pt" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName | Sort-Object)

    # Run process_dataset.py
    $uvArgs = @("run", "python", "scripts/process_dataset.py", "--config", $config, "--device", $Device)
    if ($ForceRebuild) { $uvArgs += "--force-rebuild" }
    & uv @uvArgs

    # Detect new .pt files
    $after = @(Get-ChildItem "$PT_DIR/*.pt" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName | Sort-Object)
    $newPts = $after | Where-Object { $_ -notin $before }

    if (-not $newPts) {
        Write-Host "[warn] No new .pt file — skipping upload"
    } else {
        foreach ($ptFile in $newPts) {
            $stem    = [System.IO.Path]::GetFileNameWithoutExtension($ptFile)
            $archive = "${stem}_${TS}.tar.gz"
            Write-Host "Compressing $ptFile → $archive ..."
            tar -czf $archive -C $PT_DIR (Split-Path $ptFile -Leaf)
            Write-Host "Uploading $archive → $remoteDest ..."
            rclone copy $archive $remoteDest --progress
            Remove-Item $archive -Force
            if ($DeletePt) {
                Remove-Item $ptFile -Force
                Write-Host "Deleted local $ptFile"
            }
        }
    }
}

Write-Host ""
Write-Host "All done."
