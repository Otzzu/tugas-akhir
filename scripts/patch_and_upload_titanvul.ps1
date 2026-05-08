#!/usr/bin/env pwsh
# scripts/patch_and_upload_titanvul.ps1
# Patch titanvul .pt files (reindex y labels) then tar.gz + upload to gdrive.
# Run from project root.

$ErrorActionPreference = "Stop"

$PT_FILES = @(
    "data/processed/lm_dataset_titanvul_multiclass_unixcoder-base_ft_f01075040_s1500r42.pt",
    "data/processed/lm_dataset_titanvul_multiclass_unixcoder-base_ft_f40f2e964_s1500r42.pt"
)
$VOCAB    = "data/raw/titanvul/cwe_vocab.json"
$REMOTE   = "gdrive-mesach:tugas-akhir/data/processed"
$TS       = Get-Date -Format "yyyyMMdd_HHmmss"

# ── 1. Patch ──────────────────────────────────────────────────────────────────
Write-Host "[1/3] Patching .pt files..."
uv run python scripts/patch_pt_reindex_labels.py --pt @PT_FILES --vocab $VOCAB
if ($LASTEXITCODE -ne 0) { throw "Patch failed" }

# ── 2. Verify ─────────────────────────────────────────────────────────────────
Write-Host "`n[2/3] Verifying y labels..."
$verify_script = @'
import sys, torch
for pt in sys.argv[1:]:
    d = torch.load(pt, weights_only=False)
    y = d[0].y
    u = sorted(y.unique().tolist())
    expected = list(range(len(u)))
    ok = (u == expected)
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {pt.split('/')[-1]}: {len(u)} classes, y={u[:8]}...")
    if not ok:
        sys.exit(1)
'@
$verify_script | uv run python - @PT_FILES
if ($LASTEXITCODE -ne 0) { throw "Verify failed" }

# ── 3. Tar + upload ───────────────────────────────────────────────────────────
Write-Host "`n[3/3] Tar + upload..."
foreach ($pt in $PT_FILES) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($pt)
    $tar  = "data/processed/${stem}_${TS}.tar.gz"

    Write-Host "  Compressing $stem..."
    tar -czf $tar -C data/processed "${stem}.pt"
    if ($LASTEXITCODE -ne 0) { throw "tar failed for $stem" }

    Write-Host "  Uploading $tar -> $REMOTE"
    rclone copy $tar $REMOTE --progress
    if ($LASTEXITCODE -ne 0) { throw "rclone failed for $tar" }

    Remove-Item $tar
    Write-Host "  Done: ${stem}_${TS}.tar.gz"
}

Write-Host "`nAll done."
