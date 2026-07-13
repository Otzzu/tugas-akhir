# build_titanvul_local.ps1
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Versi lokal (Windows) dari build_titanvul_cloud.sh.
#
# Datanya 2.466 fungsi — jauh lebih berat daripada BenchVul. Joern memakan waktu
# paling lama, embedding node butuh GPU agar wajar. Kalau lambat, pakai pod.
#
# Jalankan:
#   .\scripts\build_titanvul_local.ps1
#   .\scripts\build_titanvul_local.ps1 -Upload -Workers 6
param(
    [string]$JoernCli = "C:/joern/joern-cli",
    [int]$Workers = 4,
    [switch]$Upload
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$TITAN   = "data/datasets/titanvul/raw.parquet"
$MEGAVUL = "data/datasets/megavul/train.parquet"
$DS      = "data/datasets/titanvul_cc"
$RAW     = "data/raw/titanvul"
$TMP     = "data/_tv"
$PROC    = "data/processed"

if (-not (Test-Path $TITAN))   { throw "tidak ada $TITAN" }
if (-not (Test-Path $MEGAVUL)) { throw "tidak ada $MEGAVUL (dipakai untuk dedup)" }

Write-Host "`n=== [1/4] adapter (C/C++, Top25, dedup vs MegaVul) ===" -ForegroundColor Cyan
uv run python scripts/titanvul_to_parquet.py --input $TITAN --megavul $MEGAVUL --out-dir $DS

Write-Host "`n=== [2/4] CPG (Joern) — bagian paling lama ===" -ForegroundColor Cyan
if (Test-Path $TMP) { Remove-Item $TMP -Recurse -Force }
uv run python -m gnn_vuln.data.prepare `
    --input "$DS/train.parquet" --format api `
    --joern-cli $JoernCli --out-dir $TMP `
    --top-cwe 0 --workers $Workers `
    --cwe-vocab "$DS/cwe_vocab.json"

if (Test-Path $RAW) { Remove-Item $RAW -Recurse -Force }
Move-Item "$TMP/api" $RAW
Remove-Item $TMP -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item "$DS/cwe_vocab.json" "$RAW/cwe_vocab.json" -Force
$nCpg = (Get-ChildItem "$RAW/vulnerable" -Filter "func_*.json" -ErrorAction SilentlyContinue |
         Where-Object { $_.Name -notlike "*.meta.json" }).Count
Write-Host "  CPG terbentuk: $nCpg fungsi"

Write-Host "`n=== [3/4] build .pt (ml1024 lalu ml5120) ===" -ForegroundColor Cyan
uv run python -m gnn_vuln.data.build_pt --config configs/titanvul/titanvul_ml1024.yaml
uv run python -m gnn_vuln.data.build_pt --config configs/titanvul/titanvul_ml5120.yaml

Write-Host "`n=== [4/4] hasil ===" -ForegroundColor Cyan
Get-ChildItem $PROC -Filter "lm_dataset_titanvul_*_meta.pt" | ForEach-Object {
    $name = $_.BaseName -replace "_meta$", ""
    $nGraph = (Get-ChildItem "$PROC/${name}_graphs" -Filter "*.pt" -ErrorAction SilentlyContinue).Count
    Write-Host "  $name  ($nGraph graph)"
}

if ($Upload) {
    Write-Host "`n=== unggah ke Drive ===" -ForegroundColor Cyan
    Get-ChildItem $PROC -Filter "lm_dataset_titanvul_*_meta.pt" | ForEach-Object {
        $name = $_.BaseName -replace "_meta$", ""
        $tar  = "$name.tar.gz"
        tar -C $PROC -czf "$PROC/$tar" "${name}_meta.pt" "${name}_graphs"
        rclone copy "$PROC/$tar" "gdrive-mesach:tugas-akhir/" --progress
        Write-Host "  uploaded $tar"
    }
}
