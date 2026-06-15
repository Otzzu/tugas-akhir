# prepare_bigvul_survivor.ps1 — build OUR .pt for LIVABLE's Big-Vul survivors (fair head-to-head).
#   1) fetch the split-record bundle (used_*.json + cwe_labels.json) if missing
#   2) adapter: survivors -> survivors.parquet + cwe_vocab.json + split_file.json
#   3) pre-seed cwe_vocab.json into the CPG out-dir so prepare_dataset reuses LIVABLE's 31-class labels
#   4) joern CPG generation (NO --sample-per-class / --limit -> parquet position == parquet_id)
#   5) print the train command (the .pt builds on first train run; split_file pins LIVABLE's split)
#
# Run from project root:  .\scripts\prepare_bigvul_survivor.ps1
param(
  [string]$Csv       = "data/LIVABLE/all_vul.csv",
  [string]$RecordDir = "bigvul_livable",                 # gets used_*.json + cwe_labels.json
  [string]$DataDir   = "data/datasets/bigvul_survivor",  # adapter outputs (parquet, vocab, split_file)
  [string]$RawOut    = "data/raw/bigvul_survivor",        # joern CPG output
  [int]   $TopCwe    = 31,
  [int]   $Workers   = 4,
  [string]$JoernCli  = "C:/joern/joern-cli",
  [string]$Remote    = "gdrive-mesach:tugas-akhir/data/baselines/livable_bigvul_split_record.tar.gz"
)
$ErrorActionPreference = "Stop"

# 1) split-record bundle (used_*.json + cwe_labels.json) -------------------------------------
if (-not (Test-Path "$RecordDir/used_train.json")) {
  Write-Host "[1/5] fetching split-record bundle -> $RecordDir"
  New-Item -ItemType Directory -Force -Path $RecordDir | Out-Null
  rclone copy $Remote .
  tar -xzf "livable_bigvul_split_record.tar.gz" -C $RecordDir
} else { Write-Host "[1/5] split record present ($RecordDir)" }

# 2) adapter: survivors -> parquet + cwe_vocab + split_file -----------------------------------
Write-Host "[2/5] adapter -> $DataDir"
uv run python scripts/bigvul_survivor_to_parquet.py --csv $Csv --record-dir $RecordDir --out-dir $DataDir --top-cwe $TopCwe

# 3+4) joern CPG. prepare_dataset auto-appends a 'bigvul' subdir to --out-dir; pre-seed cwe_vocab
#       THERE so load_bigvul reuses LIVABLE's 31-class map (else it builds benign+top31 = 32),
#       then flatten that subdir into $RawOut (where dataset_lm source=bigvul_survivor reads).
$OutTmp = "data/_bvs"
Write-Host "[3/5] pre-seed cwe_vocab -> $OutTmp/bigvul"
if (Test-Path $OutTmp) { Remove-Item -Recurse -Force $OutTmp }
New-Item -ItemType Directory -Force -Path "$OutTmp/bigvul" | Out-Null
Copy-Item "$DataDir/cwe_vocab.json" "$OutTmp/bigvul/cwe_vocab.json" -Force

Write-Host "[4/5] joern CPG -> $RawOut"
uv run python scripts/prepare_dataset.py --input "$DataDir/survivors.parquet" --format bigvul --joern-cli $JoernCli --out-dir $OutTmp --top-cwe $TopCwe --workers $Workers
if (Test-Path $RawOut) { Remove-Item -Recurse -Force $RawOut }
$rawParent = Split-Path $RawOut
if ($rawParent) { New-Item -ItemType Directory -Force -Path $rawParent | Out-Null }
Move-Item "$OutTmp/bigvul" $RawOut
Remove-Item -Recurse -Force $OutTmp

# 5) train ------------------------------------------------------------------------------------
Write-Host "[5/5] CPGs ready in $RawOut."
Write-Host "Train (builds .pt on first run; split_file pins LIVABLE's exact split):"
Write-Host "  uv run train --config configs/ablation/bigvul/O1_bigvul_survivor.yaml"
