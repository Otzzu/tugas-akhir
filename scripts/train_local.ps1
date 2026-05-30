# scripts/train_local.ps1
#
# Local training pipeline (PowerShell) mirroring train_cloud.sh logic:
#   download dataset (rclone) -> train -> evaluate.
# Differences from cloud:
#   - No setup_cloud.sh (env assumed ready via uv)
#   - No upload step (results stay local)
#   - Uses `uv run python` instead of system python
#
# Each -Config must be paired with a -Dataset at the same array index.
#
# Flags:
#   -Resume           Pass --resume to train.py (continues from last_{arch}.pt)
#   -CleanCheckpoints Delete entire run folder under checkpoints/ after each run. Frees GBs over many runs.
#   -CleanResults     Delete entire run folder under results/ after each run.
#   -CleanDataset     Delete dataset .pt + _graphs/ after final run. Skip if reusing same dataset.
#
# Usage:
#   .\scripts\train_local.ps1 -CleanCheckpoints -CleanDataset `
#     -Config @(
#       'configs/ablation/gnn_only/N1_a1_l1.yaml',
#       'configs/ablation/gnn_only/N2_a1_l1_meanmax.yaml',
#       'configs/ablation/gnn_only/N3_a1_l1_cnn.yaml',
#       'configs/ablation/gnn_only/N4_a1_l1_meanmax_residual.yaml',
#       'configs/ablation/gnn_only/N5_a1_l1_gnn_plus.yaml'
#     ) `
#     -Dataset @(
#       'lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42',
#       'lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42',
#       'lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42',
#       'lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42',
#       'lm_dataset_megavul_multiclass_unixcoder-base_ft_ml1024_f40f2e964_s1600r42'
#     )

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string[]]$Config,

    [Parameter(Mandatory=$true)]
    [string[]]$Dataset,

    [switch]$Resume,

    # Delete entire checkpoint dir after each run (best + last + config). Saves GBs over many runs.
    [switch]$CleanCheckpoints,

    # Delete results dir after each run. Use only if logged metrics not needed locally.
    [switch]$CleanResults,

    # Delete dataset .pt + _graphs/ after final run. Not needed if same dataset reused across runs.
    [switch]$CleanDataset
)

$ErrorActionPreference = 'Stop'

$GDRIVE_REMOTE  = 'gdrive-mesach:tugas-akhir'
$PROCESSED_DIR  = 'data/processed'
$CHECKPOINTS_DIR = 'checkpoints'
$RESULTS_DIR    = 'results'

# --- Colour helpers ----------------------------------------------------------
function Info($msg)    { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Success($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Warn($msg)    { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Err($msg)     { Write-Host "[ERR]  $msg" -ForegroundColor Red }

# --- Validate ----------------------------------------------------------------
if ($Config.Count -eq 0) {
    Err 'No configs provided. Use -Config <yaml> -Dataset <name>'
    exit 1
}
if ($Config.Count -ne $Dataset.Count) {
    Err "-Config count ($($Config.Count)) != -Dataset count ($($Dataset.Count)). Must be paired."
    exit 1
}

Info "Configs to run: $($Config.Count)"
for ($i = 0; $i -lt $Config.Count; $i++) {
    Write-Host "  [$($i+1)] $($Config[$i])  |  dataset: $($Dataset[$i])"
}
Write-Host ''

# --- 1. rclone check ---------------------------------------------------------
function Test-Rclone {
    Info 'Checking rclone...'
    if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
        Err 'rclone not in PATH. Install rclone first.'
        exit 1
    }
    $rcloneConf = Join-Path $env:USERPROFILE '.config\rclone\rclone.conf'
    if (-not (Test-Path $rcloneConf)) {
        Warn "rclone.conf not found at $rcloneConf"
        if (Test-Path 'rclone.zip') {
            Info 'Installing rclone.conf from rclone.zip'
            $confDir = Join-Path $env:USERPROFILE '.config\rclone'
            if (-not (Test-Path $confDir)) { New-Item -ItemType Directory -Force -Path $confDir | Out-Null }
            Expand-Archive -Path 'rclone.zip' -DestinationPath $confDir -Force
        } else {
            Err 'rclone.zip not found in project root. Cannot set up rclone.'
            exit 1
        }
    }
    $remotes = rclone listremotes 2>$null
    if (-not ($remotes -match 'gdrive-mesach:')) {
        Err "Remote 'gdrive-mesach' not found in rclone config."
        exit 1
    }
    Success 'rclone connected'
}

# --- 2. Python env check -----------------------------------------------------
function Test-Env {
    Info 'Checking Python env (uv)...'
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Err 'uv not in PATH. Install uv first (https://docs.astral.sh/uv).'
        exit 1
    }
    $cuda = (uv run python -c "import torch; print(torch.cuda.is_available())" 2>$null) -join ''
    if ($cuda.Trim() -eq 'True') {
        $gpuName = (uv run python -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null) -join ''
        Success "CUDA available - $($gpuName.Trim())"
    } else {
        Warn 'CUDA not available - training will be slow'
    }
}

# --- 3. Dataset download (cached) --------------------------------------------
$script:DownloadedDatasets = @()

function Get-Dataset {
    param([string]$DatasetName, [string]$ConfigPath = '')

    # Strip trailing _YYYYMMDD_HHMMSS if present
    $localName = $DatasetName -replace '_\d{8}_\d{6}$', ''
    $localDir  = Join-Path $PROCESSED_DIR $localName

    # Detect storage mode from config (default: inmemory)
    $storage = 'inmemory'
    if ($ConfigPath -and (Test-Path $ConfigPath)) {
        $storageLine = Select-String -Path $ConfigPath -Pattern '^\s+storage:\s+(\S+)' | Select-Object -First 1
        if ($storageLine) { $storage = $storageLine.Matches[0].Groups[1].Value }
    }

    if ($script:DownloadedDatasets -contains $DatasetName) {
        if ((Test-Path $localDir) -or (Get-ChildItem $PROCESSED_DIR -Filter "$localName*.pt" -ErrorAction SilentlyContinue)) {
            return
        }
    }

    if ((Test-Path $localDir) -or (Get-ChildItem $PROCESSED_DIR -Filter "$localName*.pt" -ErrorAction SilentlyContinue)) {
        Success "Dataset already exists: $localName"
        $script:DownloadedDatasets += $DatasetName
        return
    }
    Info "Not found locally (checked: $localDir/ and $PROCESSED_DIR/$localName*.pt)"

    Info "Downloading dataset: $DatasetName (storage=$storage)"
    if (-not (Test-Path $PROCESSED_DIR)) { New-Item -ItemType Directory -Force -Path $PROCESSED_DIR | Out-Null }

    # Try 1: .zip at gdrive root (legacy)
    $remoteZip = "$GDRIVE_REMOTE/$DatasetName.zip"
    $zipExists = rclone lsf $remoteZip 2>$null
    if ($zipExists) {
        $localZip = Join-Path $PROCESSED_DIR "$DatasetName.zip"
        rclone copy $remoteZip $PROCESSED_DIR --progress
        Expand-Archive -Path $localZip -DestinationPath $PROCESSED_DIR -Force
        Remove-Item $localZip -Force
        Success "Dataset ready: $DatasetName"
        $script:DownloadedDatasets += $DatasetName
        return
    }

    # Try 2: .tar.gz in data/processed/<source>/ subdir
    $remoteProc = "$GDRIVE_REMOTE/data/processed"
    $source = ($DatasetName -replace 'lm_dataset_([^_]+)_.*', '$1')
    foreach ($remoteSubdir in @("$remoteProc/$source", $remoteProc)) {
        $listing = rclone lsf $remoteSubdir 2>$null
        if (-not $listing) { continue }
        $remoteTar = $listing | Where-Object { $_ -match "^$([Regex]::Escape($DatasetName)).*_${storage}_.*\.tar\.gz$" } | Sort-Object | Select-Object -Last 1
        if (-not $remoteTar) {
            $remoteTar = $listing | Where-Object { $_ -match "^$([Regex]::Escape($DatasetName)).*\.tar\.gz$" } | Sort-Object | Select-Object -Last 1
            if ($remoteTar) { Warn "No $storage-marked tar found - using legacy: $remoteTar" }
        }
        if ($remoteTar) {
            $localTar = Join-Path $PROCESSED_DIR $remoteTar
            Info "Found: $remoteSubdir/$remoteTar"
            rclone copy "$remoteSubdir/$remoteTar" $PROCESSED_DIR --progress
            tar -xzf $localTar -C $PROCESSED_DIR
            Remove-Item $localTar -Force
            Success "Dataset ready: $DatasetName"
            $script:DownloadedDatasets += $DatasetName
            return
        }
    }

    Err "Dataset not found on gdrive: $DatasetName"
    exit 1
}

# --- 4. Train ----------------------------------------------------------------
function Invoke-Train {
    param([string]$ConfigPath)
    $extraArgs = @()
    if ($Resume) { $extraArgs += '--resume' }
    Info "Training: $ConfigPath$(if ($Resume) {' (--resume)'})"
    # Clear Linux-only allocator hint if leaked from prior session
    Remove-Item env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    $env:PYTHONPATH = 'src'
    uv run python -m gnn_vuln.train --config $ConfigPath @extraArgs
    if ($LASTEXITCODE -ne 0) { Err "Training failed: $ConfigPath"; exit 1 }
    Success "Training done: $ConfigPath"
}

# --- 5. Evaluate -------------------------------------------------------------
function Invoke-Evaluate {
    param([string]$ModelDir, [string]$ModelId)
    $ckpt = Get-ChildItem -Path $ModelDir -Filter 'best_*.pt' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $ckpt) {
        Warn "No best_*.pt found in $ModelDir - skipping evaluate"
        return
    }
    $configYaml = Join-Path $ModelDir 'config.yaml'
    if (-not (Test-Path $configYaml)) {
        Warn "config.yaml not found in $ModelDir - skipping evaluate"
        return
    }
    Info "Evaluating: $ModelId"
    $env:PYTHONPATH = 'src'
    uv run python -m gnn_vuln.evaluate --checkpoint $ckpt.FullName --config $configYaml
    if ($LASTEXITCODE -ne 0) { Err "Evaluate failed: $ModelId"; exit 1 }
    Success "Evaluate done: $ModelId"
}

# --- Main --------------------------------------------------------------------
Test-Rclone
Test-Env

for ($i = 0; $i -lt $Config.Count; $i++) {
    $cfg = $Config[$i]
    $ds  = $Dataset[$i]
    $N = $i + 1
    $TOTAL = $Config.Count

    Write-Host ''
    Write-Host '================================================' -ForegroundColor Cyan
    Write-Host "  Run $N/$TOTAL`: $(Split-Path $cfg -Leaf)" -ForegroundColor Cyan
    Write-Host '================================================' -ForegroundColor Cyan

    Get-Dataset -DatasetName $ds -ConfigPath $cfg

    $before = Get-ChildItem -Path $CHECKPOINTS_DIR -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    Invoke-Train -ConfigPath $cfg

    $modelDir = Get-ChildItem -Path $CHECKPOINTS_DIR -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $modelDir) {
        Err 'Could not detect checkpoint dir after training'
        exit 1
    }
    if ($before -and $modelDir.FullName -eq $before.FullName -and -not $Resume) {
        Err 'Could not detect new checkpoint dir (dir unchanged - if resuming, add -Resume)'
        exit 1
    }
    $modelId = $modelDir.Name
    Success "New model: $modelId"

    Invoke-Evaluate -ModelDir $modelDir.FullName -ModelId $modelId

    if ($CleanCheckpoints) {
        $sizeMB = [Math]::Round((Get-ChildItem -Path $modelDir.FullName -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
        Remove-Item -Recurse -Force $modelDir.FullName
        Info "Cleaned checkpoint dir $modelId ($sizeMB MB freed)"
    }

    if ($CleanResults) {
        $resultDir = Join-Path $RESULTS_DIR $modelId
        if (Test-Path $resultDir) {
            $sizeMB = [Math]::Round((Get-ChildItem -Path $resultDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
            Remove-Item -Recurse -Force $resultDir
            Info "Cleaned results dir $modelId ($sizeMB MB freed)"
        }
    }

    Write-Host "  DONE $N/$TOTAL`: $modelId" -ForegroundColor Green
}

if ($CleanDataset) {
    foreach ($ds in ($Dataset | Select-Object -Unique)) {
        $localName = $ds -replace '_\d{8}_\d{6}$', ''
        $pts = Get-ChildItem -Path $PROCESSED_DIR -Filter "$localName*.pt" -File -ErrorAction SilentlyContinue
        if ($pts) {
            $totalGB = [Math]::Round(($pts | Measure-Object Length -Sum).Sum / 1GB, 2)
            $pts | Remove-Item -Force
            Info "Cleaned dataset .pt for $localName ($totalGB GB freed)"
        }
        $graphsDir = Join-Path $PROCESSED_DIR "${localName}_graphs"
        if (Test-Path $graphsDir) {
            Remove-Item -Recurse -Force $graphsDir
            Info "Cleaned $graphsDir"
        }
    }
}

Write-Host ''
Success "All $($Config.Count) run(s) complete. Results stay local under $RESULTS_DIR/ and $CHECKPOINTS_DIR/."
