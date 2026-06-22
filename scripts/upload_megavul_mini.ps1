#!/usr/bin/env pwsh
# scripts/upload_megavul_mini.ps1
# Bundle the megavul_mini inmemory dataset (.pt + cwe_vocab.json) and upload to gdrive
# for sharing. Friend extracts at the repo root to restore:
#   data/processed/lm_dataset_megavul_mini_..._ml1024.pt
#   data/raw/megavul_mini/cwe_vocab.json
# Run from project root.

$ErrorActionPreference = "Stop"

$PT      = "data/processed/lm_dataset_megavul_mini_multiclass_unixcoder-base_ft_ml1024.pt"
$VOCAB   = "data/raw/megavul_mini/cwe_vocab.json"
$ARCHIVE = "megavul_mini.tar.gz"
$REMOTE  = "gdrive-mesach:tugas-akhir/data/processed/megavul_mini"

if (-not (Test-Path $PT))    { throw "Missing $PT" }
if (-not (Test-Path $VOCAB)) { throw "Missing $VOCAB" }

Write-Host "[1/2] Compressing -> $ARCHIVE ..."
tar -czf $ARCHIVE $PT $VOCAB
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
Write-Host ("  size: {0:N1} MB" -f ((Get-Item $ARCHIVE).Length / 1MB))

Write-Host "`n[2/2] Uploading -> $REMOTE"
rclone copy $ARCHIVE $REMOTE --progress
if ($LASTEXITCODE -ne 0) { throw "rclone failed" }

Remove-Item $ARCHIVE
Write-Host "`nDone. $ARCHIVE -> $REMOTE/"
Write-Host "Friend restore: rclone copy $REMOTE/$ARCHIVE . ; tar -xf $ARCHIVE   (run at repo root)"
