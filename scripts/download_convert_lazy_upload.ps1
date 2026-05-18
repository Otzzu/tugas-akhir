#!/usr/bin/env pwsh
# Download an existing processed .pt archive from Google Drive, convert it to
# the target storage/func config via process_dataset.py's patch fast-path
# (re-tokenize func tokens, reuse node embeddings — no rebuild from raw),
# re-archive, and upload back.
#
# Use case: cloud has inmemory .pt at ml1024; phase 6 wants lazy .pt (CodeT5+
# at ml512). The patch path matches on node-LM + filter + sample params only,
# so any func_lm / func_max_length base is reusable.
#
# Disk-safe: each config is fully cleaned up (base + converted .pt + _graphs/
# + archives) before the next one starts — datasets are large, only one is
# ever on disk at a time. Pass -KeepPt to disable cleanup.
#
# Usage:
#   .\scripts\download_convert_lazy_upload.ps1 [flags] `
#       -Configs  cfgA.yaml,cfgB.yaml `
#       -Archives archiveA.tar.gz,archiveB.tar.gz
#
# Configs[i] is built using Archives[i] as the patch base. The config's
# `storage:` and `func_max_length:` fields drive the conversion.
#
# Flags:
#   -SkipDownload    Base archive already extracted in data/processed/
#   -KeepPt          Do NOT clean up local .pt after upload (default: clean)
#   -Device <str>    cuda or cpu (default: cuda)
#   -Compress <str>  Archive compression for the upload (default: none):
#                      none — plain tar, no compression. Fastest, no install.
#                             .pt = float tensors → gzip barely shrinks them,
#                             so this is the recommended default.
#                      pigz — parallel gzip (needs pigz on PATH). Multi-core.
#                      gzip — single-threaded gzip. Slow on large datasets.
#
# Example (phase 6 — F2/F3/F4 inmemory ml1024 → lazy):
#   .\scripts\download_convert_lazy_upload.ps1 `
#     -Configs  configs/ablation/phase6/F2_node-codet5p_func-unixcoder.yaml, `
#               configs/ablation/phase6/F3_node-unixcoder_func-codet5p.yaml, `
#               configs/ablation/phase6/F4_node-codet5p_func-codet5p.yaml `
#     -Archives lm_dataset_megavul_multiclass_codet5p-110m-embedding_live_unixcoder-base_ft_ml1024_f40f2e964_s1600r42_20260512_041852.tar.gz, `
#               lm_dataset_megavul_multiclass_unixcoder-base_live_codet5p-110m-embedding_ft_ml1024_f40f2e964_s1600r42_20260512_012605.tar.gz, `
#               lm_dataset_megavul_multiclass_codet5p-110m-embedding_ft_ml1024_f40f2e964_s1600r42_20260512_041852.tar.gz

param(
    [Parameter(Mandatory)][string[]]$Configs,
    [Parameter(Mandatory)][string[]]$Archives,
    [switch]$SkipDownload,
    [switch]$KeepPt,
    [Parameter()][string]$Device = "cuda",
    [Parameter()][ValidateSet("none", "pigz", "gzip")][string]$Compress = "none"
)

$ErrorActionPreference = "Stop"

$REMOTE_PT = "gdrive-mesach:tugas-akhir/data/processed"
$PT_DIR    = "data/processed"
$TS        = Get-Date -Format "yyyyMMdd_HHmmss"

if ($Configs.Count -ne $Archives.Count) {
    Write-Error "Configs ($($Configs.Count)) and Archives ($($Archives.Count)) count mismatch — must be parallel."
    exit 1
}

New-Item -ItemType Directory -Force -Path $PT_DIR | Out-Null

# Snapshot of .pt files + *_graphs/ dirs in $PT_DIR — used to detect and clean
# everything an iteration creates.
function Get-PtState {
    $files = @(Get-ChildItem "$PT_DIR/*.pt" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName)
    $dirs  = @(Get-ChildItem $PT_DIR -Directory -Filter "*_graphs" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName)
    return @($files + $dirs)
}

for ($i = 0; $i -lt $Configs.Count; $i++) {
    $config  = $Configs[$i]
    $archive = $Archives[$i]
    Write-Host ""
    Write-Host "=== [$($i + 1)/$($Configs.Count)] $config ==="
    Write-Host "Base archive: $archive"

    # ── Pre-iteration snapshot — anything created after this gets cleaned ──
    $preState = Get-PtState

    # Dataset source (subdir on the remote) + storage mode from config
    $datasetSrc = (Select-String -Path $config -Pattern "^\s+source:\s+(\S+)" |
        Select-Object -First 1).Matches.Groups[1].Value
    if (-not $datasetSrc) { $datasetSrc = "misc" }
    $remoteDir = "$REMOTE_PT/$datasetSrc"

    $storageLine = (Select-String -Path $config -Pattern "^\s+storage:\s+(\S+)" |
        Select-Object -First 1).Matches.Groups[1].Value
    $storage = if ($storageLine) { $storageLine } else { "inmemory" }
    Write-Host "Target storage: $storage   |   remote: $remoteDir"

    # ── 1. Download + extract the patch base ──────────────────────────────
    if (-not $SkipDownload) {
        Write-Host "Downloading base $archive ..."
        rclone copy "$remoteDir/$archive" $PT_DIR --progress
        $archivePath = Join-Path $PT_DIR $archive
        if (-not (Test-Path $archivePath)) {
            Write-Error "Download failed — $archive not found in $PT_DIR"
            exit 1
        }
        Write-Host "Extracting base ..."
        tar -xf $archivePath -C $PT_DIR      # -xf auto-detects gz / plain / zst
        Remove-Item $archivePath -Force      # clean: drop archive once extracted
    } else {
        Write-Host "[skip-download] expecting base already extracted in $PT_DIR"
    }

    # ── 2. Snapshot .pt BEFORE conversion (base is now on disk) ────────────
    $beforeConvert = @(Get-ChildItem "$PT_DIR/*.pt" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName | Sort-Object)

    # ── 3. Convert via patch fast-path (process_dataset.py) ────────────────
    # storage:/func_max_length: from the config drive the output. The patch
    # path matches the extracted base on node-LM + filter + sample params,
    # re-tokenizes func tokens, and writes the target storage format.
    Write-Host "Converting via process_dataset.py (patch fast-path) ..."
    & uv run python scripts/process_dataset.py --config $config --device $Device

    # ── 4. Detect the new .pt (plain .pt inmemory / _meta.pt lazy) ─────────
    $afterConvert = @(Get-ChildItem "$PT_DIR/*.pt" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName | Sort-Object)
    $newPts = $afterConvert | Where-Object { $_ -notin $beforeConvert }

    if (-not $newPts) {
        Write-Host "[warn] No new .pt produced — skipping upload"
    } else {
        foreach ($ptFile in $newPts) {
            $leaf     = Split-Path $ptFile -Leaf
            $stem     = [System.IO.Path]::GetFileNameWithoutExtension($ptFile)
            $baseStem = $stem -replace '_meta$', ''   # strip _meta for lazy format
            $ext        = if ($Compress -eq "none") { "tar" } else { "tar.gz" }
            $outArchive = "${baseStem}_${storage}_${TS}.${ext}"

            # ── 5. Re-archive (lazy = _meta.pt + _graphs/) ─────────────────
            $tarItems = @($leaf)
            if ($storage -eq "lazy") {
                $graphsDir = "${baseStem}_graphs"
                if (Test-Path (Join-Path $PT_DIR $graphsDir)) {
                    $tarItems += $graphsDir
                } else {
                    Write-Warning "lazy storage but _graphs/ dir missing: $graphsDir"
                }
            }
            # Archive. .pt files are float tensors → gzip barely shrinks them;
            # default 'none' (plain tar) skips compression CPU entirely.
            Write-Host "Archiving ($storage, compress=$Compress) -> $outArchive ..."
            switch ($Compress) {
                "pigz" {
                    if (-not (Get-Command pigz -ErrorAction SilentlyContinue)) {
                        Write-Error "Compress=pigz but pigz not on PATH. Install it, or use -Compress none."
                        exit 1
                    }
                    tar -C $PT_DIR --use-compress-program "pigz -1" -cf $outArchive @tarItems
                }
                "gzip" { tar -C $PT_DIR -czf $outArchive @tarItems }
                default { tar -C $PT_DIR -cf $outArchive @tarItems }   # none
            }

            # ── 6. Upload, then drop the archive ───────────────────────────
            Write-Host "Uploading $outArchive -> $remoteDir ..."
            rclone copy $outArchive $remoteDir --progress
            Remove-Item $outArchive -Force
        }
    }

    # ── 7. Clean up — remove everything this iteration created ─────────────
    # (base .pt, base _graphs/, converted .pt, converted _graphs/) so the
    # next config starts with a clean, empty data/processed.
    if ($KeepPt) {
        Write-Host "[keep-pt] leaving local .pt in place"
    } else {
        $postState = Get-PtState
        $created   = $postState | Where-Object { $_ -notin $preState }
        foreach ($path in $created) {
            if (Test-Path $path -PathType Container) {
                Remove-Item $path -Recurse -Force
            } elseif (Test-Path $path) {
                Remove-Item $path -Force
            }
            Write-Host "Cleaned: $path"
        }
    }
}

Write-Host ""
Write-Host "All done."
