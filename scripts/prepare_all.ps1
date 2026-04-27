# prepare_all.ps1
# ~~~~~~~~~~~~~~~
# Runs Joern CPG generation for Devign, BigVul, and DiverseVul.
#
# Dataset behaviour
# -----------------
# BigVul    → multi-class by default (top --TopCwe CWE categories).
#             cwe_vocab.json is saved to data/raw/ on the train run and
#             reused for val/test so class indices stay consistent.
#             Binary mode: add -BigVulBinary to the command.
# Devign    → binary + flaw lines (vul_lines column).
# DiverseVul→ binary only, no localization ground truth.
#
# All datasets write into the same benign/ and vulnerable/ dirs using
# --idx-offset to guarantee unique file names.
#
# Timing estimate (4 workers, ~6s/function):
#   Devign     5000/class => ~250 min (~4 h)
#   BigVul     5000/class => ~250 min (~4 h)  (per-CWE class if multi-class)
#   DiverseVul 5000/class => ~250 min (~4 h)
#   TOTAL (sequential) => ~24 h overnight
#
# Reduce --SamplePerClass to 1000 for a quick smoke-test (~1 h total).
#
# Usage:
#   cd C:\Users\Otzzu\Documents\tugas-akhir
#   .\scripts\prepare_all.ps1
#   .\scripts\prepare_all.ps1 -JoernCli C:/joern/joern-cli -Workers 4 -SamplePerClass 2000
#   .\scripts\prepare_all.ps1 -BigVulBinary   # binary mode (no CWE classes)

param(
    [string]$JoernCli       = "C:/joern/joern-cli",
    [int]   $Workers        = 4,
    [int]   $SamplePerClass = 5000,
    [int]   $TopCwe         = 10,
    [switch]$BigVulBinary
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

function Run-Prepare {
    param(
        [string]$Input,
        [string]$Format,
        [string]$OutDir,
        [int]   $IdxOffset,
        [int]   $Sample,
        [string]$Label      = "Processing",
        [string]$CweVocab   = "",    # path to existing vocab (val/test reuse)
        [int]   $TopCwe_    = 0,     # >0 enables multi-class for bigvul
        [switch]$Binary_              # forces binary for bigvul
    )

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "  Input:  $Input" -ForegroundColor Cyan
    Write-Host "  Output: $OutDir  (offset=$IdxOffset, sample=$Sample/class)" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan

    $args_ = @(
        "scripts/prepare_dataset.py",
        "--input",      $Input,
        "--format",     $Format,
        "--out-dir",    $OutDir,
        "--joern-cli",  $JoernCli,
        "--workers",    $Workers,
        "--idx-offset", $IdxOffset,
        "--resume"
    )
    if ($Sample -gt 0) {
        $args_ += @("--sample-per-class", $Sample)
    }
    if ($TopCwe_ -gt 0) {
        $args_ += @("--top-cwe", $TopCwe_)
    }
    if ($Binary_) {
        $args_ += "--binary"
    }
    if ($CweVocab -ne "") {
        $args_ += @("--cwe-vocab", $CweVocab)
    }

    uv run python @args_
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "prepare_dataset.py exited with code $LASTEXITCODE for $Label"
    }
}

$out     = "$Root\data\raw"
$outVal  = "$Root\data\raw_val"
$outTest = "$Root\data\raw_test"

# Vocab is built during BigVul train and reused by val/test
$bigvulVocab = "$out\cwe_vocab.json"

# ---------------------------------------------------------------------------
# Devign  (offset 0 – 49 999)
# Binary + flaw lines from vul_lines column.
# ---------------------------------------------------------------------------
Run-Prepare `
    -Input     "$Root\data\datasets\devign\train.parquet" `
    -Format    "devign" `
    -OutDir    $out `
    -IdxOffset 0 `
    -Sample    $SamplePerClass `
    -Label     "Devign — train (binary + flaw lines)"

Run-Prepare `
    -Input     "$Root\data\datasets\devign\validation.parquet" `
    -Format    "devign" `
    -OutDir    $outVal `
    -IdxOffset 50000 `
    -Sample    0 `
    -Label     "Devign — validation (all)"

Run-Prepare `
    -Input     "$Root\data\datasets\devign\test.parquet" `
    -Format    "devign" `
    -OutDir    $outTest `
    -IdxOffset 55000 `
    -Sample    0 `
    -Label     "Devign — test (all)"

# ---------------------------------------------------------------------------
# BigVul  (offset 100 000 – 199 999)
# Multi-class by default (top --TopCwe CWEs become class indices 1..K).
# Train run saves cwe_vocab.json; val/test reuse it via --cwe-vocab.
# Pass -BigVulBinary for binary mode.
# Flaw lines are extracted from func_before/func_after diff.
# ---------------------------------------------------------------------------
$bvArgs = @{
    Format    = "bigvul"
    TopCwe_   = $TopCwe
    Binary_   = $BigVulBinary
}

Run-Prepare `
    -Input     "$Root\data\datasets\bigvul\train.parquet" `
    -OutDir    $out `
    -IdxOffset 100000 `
    -Sample    $SamplePerClass `
    -Label     "BigVul — train (multi-class top-$TopCwe CWEs + flaw lines)" `
    @bvArgs

Run-Prepare `
    -Input     "$Root\data\datasets\bigvul\validation.parquet" `
    -OutDir    $outVal `
    -IdxOffset 150000 `
    -Sample    1000 `
    -CweVocab  $bigvulVocab `
    -Label     "BigVul — validation (1000/class, reuse vocab)" `
    @bvArgs

Run-Prepare `
    -Input     "$Root\data\datasets\bigvul\test.parquet" `
    -OutDir    $outTest `
    -IdxOffset 155000 `
    -Sample    1000 `
    -CweVocab  $bigvulVocab `
    -Label     "BigVul — test (1000/class, reuse vocab)" `
    @bvArgs

# ---------------------------------------------------------------------------
# DiverseVul  (offset 200 000 – 299 999)
# Binary only. No localization ground truth (flaw_line_mask will be all-zero).
# ---------------------------------------------------------------------------
Run-Prepare `
    -Input     "$Root\data\datasets\diversevul\train.parquet" `
    -Format    "diversevul" `
    -OutDir    $out `
    -IdxOffset 200000 `
    -Sample    $SamplePerClass `
    -Label     "DiverseVul — train (binary, no localization)"

Run-Prepare `
    -Input     "$Root\data\datasets\diversevul\validation.parquet" `
    -Format    "diversevul" `
    -OutDir    $outVal `
    -IdxOffset 250000 `
    -Sample    1000 `
    -Label     "DiverseVul — validation (1000/class)"

Run-Prepare `
    -Input     "$Root\data\datasets\diversevul\test.parquet" `
    -Format    "diversevul" `
    -OutDir    $outTest `
    -IdxOffset 255000 `
    -Sample    1000 `
    -Label     "DiverseVul — test (1000/class)"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  All datasets done!" -ForegroundColor Green
Write-Host "  Train CPGs : $out\benign\  +  $out\vulnerable\" -ForegroundColor Green
Write-Host "  Val   CPGs : $outVal\" -ForegroundColor Green
Write-Host "  Test  CPGs : $outTest\" -ForegroundColor Green
if (-not $BigVulBinary) {
    Write-Host "  CWE vocab  : $bigvulVocab" -ForegroundColor Green
}
Write-Host "`n  Next step:" -ForegroundColor Green
Write-Host "    uv run train --config configs/lmgcn/binary.yaml" -ForegroundColor Yellow
Write-Host "    uv run train --config configs/lmgat/binary.yaml" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Green
