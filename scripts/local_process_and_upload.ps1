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

    # Extract dataset source and storage mode from config
    $datasetSrc = (Select-String -Path $config -Pattern "^\s+source:\s+(\S+)" |
        Select-Object -First 1).Matches.Groups[1].Value
    if (-not $datasetSrc) { $datasetSrc = "misc" }
    $remoteDest = "$REMOTE_PT/$datasetSrc"
    Write-Host "Upload destination: $remoteDest"

    $storageLine = (Select-String -Path $config -Pattern "^\s+storage:\s+(\S+)" |
        Select-Object -First 1).Matches.Groups[1].Value
    $storage = if ($storageLine) { $storageLine } else { "inmemory" }
    Write-Host "Storage mode: $storage"

    # Snapshot .pt files before processing
    $before = @(Get-ChildItem "$PT_DIR/*.pt" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName | Sort-Object)

    # Run process_dataset.py
    $uvArgs = @("run", "python", "scripts/process_dataset.py", "--config", $config, "--device", $Device)
    if ($ForceRebuild) { $uvArgs += "--force-rebuild" }
    & uv @uvArgs

    # Detect new .pt files (_meta.pt for lazy, plain .pt for inmemory)
    $after = @(Get-ChildItem "$PT_DIR/*.pt" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName | Sort-Object)
    $newPts = $after | Where-Object { $_ -notin $before }

    if (-not $newPts) {
        Write-Host "[warn] No new .pt file — skipping upload"
    } else {
        foreach ($ptFile in $newPts) {
            $leaf     = Split-Path $ptFile -Leaf
            $stem     = [System.IO.Path]::GetFileNameWithoutExtension($ptFile)
            $baseStem = $stem -replace '_meta$', ''   # strip _meta suffix for lazy format
            $archive  = "${baseStem}_${storage}_${TS}.tar.gz"

            # Build list of items to tar
            $tarItems = @($leaf)
            if ($storage -eq "lazy") {
                $graphsDir = "${baseStem}_graphs"
                $graphsDirPath = Join-Path $PT_DIR $graphsDir
                if (Test-Path $graphsDirPath) {
                    $tarItems += $graphsDir
                } else {
                    Write-Warning "lazy storage but _graphs/ dir not found: $graphsDirPath"
                }
            }

            Write-Host "Compressing ($storage) → $archive ..."
            tar -czf $archive -C $PT_DIR @tarItems
            Write-Host "Uploading $archive → $remoteDest ..."
            rclone copy $archive $remoteDest --progress
            Remove-Item $archive -Force

            if ($DeletePt) {
                Remove-Item $ptFile -Force
                if ($storage -eq "lazy") {
                    $graphsDirPath = Join-Path $PT_DIR "${baseStem}_graphs"
                    if (Test-Path $graphsDirPath) {
                        Remove-Item $graphsDirPath -Recurse -Force
                    }
                }
                Write-Host "Deleted local $ptFile"
            }
        }
    }
}

Write-Host ""
Write-Host "All done."
