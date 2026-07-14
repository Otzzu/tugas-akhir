# build_benchvul_local.ps1
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Versi lokal (Windows) dari build_benchvul_cloud.sh. Datanya kecil (156 fungsi),
# jadi tidak perlu pod.
#
# Butuh: Joern di C:\joern\joern-cli (atau lewat -JoernCli) dan JDK yang dikenali
# joern_runner. Embedding node memakai GPU bila ada.
#
# Jalankan:
#   .\scripts\build_benchvul_local.ps1
#   .\scripts\build_benchvul_local.ps1 -Upload        # sekalian unggah ke Drive
param(
    [string]$JoernCli = "C:/joern/joern-cli",
    [int]$Workers = 4,
    [switch]$Upload
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$PARQUET = "data/datasets/benchvul/train.parquet"
$DS      = "data/datasets/benchvul_cc"
$RAW     = "data/raw/benchvul"
$TMP     = "data/_bv"
$PROC    = "data/processed"

if (-not (Test-Path $PARQUET)) { throw "tidak ada $PARQUET" }
if (-not (Test-Path "$JoernCli/joern-parse.bat") -and -not (Test-Path "$JoernCli/joern-parse")) {
    throw "joern-parse tidak ditemukan di $JoernCli"
}

Write-Host "`n=== [1/4] adapter -> parquet berbentuk MegaVul ===" -ForegroundColor Cyan
uv run python scripts/benchvul_to_parquet.py --input $PARQUET --out-dir $DS

Write-Host "`n=== [2/4] CPG (Joern) ===" -ForegroundColor Cyan
# prepare membuat subdir '<format>' sendiri -> $TMP/api
if (Test-Path $TMP) { Remove-Item $TMP -Recurse -Force }
uv run python -m gnn_vuln.data.prepare `
    --input "$DS/train.parquet" --format api `
    --joern-cli $JoernCli --out-dir $TMP `
    --top-cwe 0 --workers $Workers `
    --cwe-vocab "$DS/cwe_vocab.json"

if (Test-Path $RAW) { Remove-Item $RAW -Recurse -Force }
Move-Item "$TMP/api" $RAW
Remove-Item $TMP -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item "$DS/cwe_vocab.json" "$RAW/cwe_vocab.json" -Force   # dataset_lm butuh vocab di raw dir
$nCpg = (Get-ChildItem "$RAW/vulnerable" -Filter "func_*.xml" -ErrorAction SilentlyContinue).Count
Write-Host "  CPG terbentuk: $nCpg fungsi"

Write-Host "`n=== [3/4] build .pt (ml1024 lalu ml5120) — GPU ===" -ForegroundColor Cyan
uv run python -m gnn_vuln.data.build_pt --config configs/benchvul/benchvul_ml1024.yaml --device cuda
uv run python -m gnn_vuln.data.build_pt --config configs/benchvul/benchvul_ml5120.yaml --device cuda

Write-Host "`n=== [4/4] hasil ===" -ForegroundColor Cyan
Get-ChildItem $PROC -Filter "lm_dataset_benchvul_*_meta.pt" | ForEach-Object {
    $name = $_.BaseName -replace "_meta$", ""
    $nGraph = (Get-ChildItem "$PROC/${name}_graphs" -Filter "*.pt" -ErrorAction SilentlyContinue).Count
    Write-Host "  $name  ($nGraph graph)"
}

if ($Upload) {
    Write-Host "`n=== unggah ke Drive ===" -ForegroundColor Cyan
    Get-ChildItem $PROC -Filter "lm_dataset_benchvul_*_meta.pt" | ForEach-Object {
        $name = $_.BaseName -replace "_meta$", ""
        $tar  = "$name.tar.gz"
        tar -C $PROC -czf "$PROC/$tar" "${name}_meta.pt" "${name}_graphs"
        rclone copy "$PROC/$tar" "gdrive-mesach:tugas-akhir/" --progress
        Write-Host "  uploaded $tar"
    }
}
