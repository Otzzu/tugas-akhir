# Localization Fix — Re-run Plan

Flaw-line-mask fix (exact-line): the METHOD node's whole-function span marked line 1
(signature) as a flaw in ~90% of vulnerable funcs, inflating localization Top-1. Fixed
so a node is a flaw only if its own source line is a patch line. Ranking loss was trained
on the bad mask, so the affected models must be **retrained**. Classification is
unaffected (mask never enters classification loss).

## What must re-run vs excluded

| Re-run | Skip (mask irrelevant) |
|---|---|
| 3 archs × (26-class + vuln-only) × 3 seeds = 18 runs | VulExplainer, LIVABLE, VulPCL, EDAT (classification only) |
| LOSVER, LineVD, LineVul (localization baselines) | seq + hybrid continual (not in bab-4) |
| graph continual (domain + CIL) — optional, for provenance | — |

Continual in bab-4 is **graph-only** ([hasil_evaluasi.md:87](docs/bab-4/hasil_evaluasi.md)). The encoder is shared,
so the retrained N48 backbone shifts (within seed noise) — re-running graph continual keeps the provenance
chain clean. It's light (ml1024, ~30 short runs). Optional; skip it and the old continual numbers stay valid.

## Prep (once, before pods train)

1. **Patch + reupload each dataset** (drops the METHOD flag = exact-line; pure `.pt`, no rebuild, no Joern):
   ```bash
   bash scripts/patch_reupload_flaw_mask.sh \
     lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42 \
     lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42 \
     lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_vulnonly \
     lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_vulnonly
   ```
   (Each pod can patch only the datasets it needs — see per-pod notes. Patch each dataset on exactly one pod.)

2. **Re-export the baseline bundle** from the patched ml1024 base (the old `megavul_ml1024_baselines_20260613.tar.gz` on Drive has the STALE flaw GT — it must be regenerated). Runs on the pod that holds the patched ml1024 base (Pod C):
   ```bash
   BASE=lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42
   PYTHONPATH=src python scripts/export_baseline_split.py \
     --processed-dir data/processed --ds-name "$BASE" --out-dir data/baselines/megavul_ml1024
   NEWTAR=megavul_ml1024_baselines_$(date +%Y%m%d).tar.gz
   tar -I "$(command -v pigz || echo gzip)" -cf "$NEWTAR" -C data/baselines megavul_ml1024
   rclone copy "$NEWTAR" gdrive-mesach:tugas-akhir/data/baselines/ --progress
   ```
   Then set `DATA_TAR="$NEWTAR"` in `run_losver_cloud.sh`, `run_linevd_cloud.sh`, `run_linevul_cloud.sh`.

## Pod distribution (balanced ~7.5–10.5 h each)

Real per-run times: hybrid O1 ~3.5 h (ml5120, LM-heavy), graph N48 ~1 h, seq S1 ~0.5 h.
Hybrid is the bottleneck → split across two pods so no pod runs much longer than the others.

| Pod | GPU | Work | Runs | Est. time | Est. cost |
|-----|-----|------|------|-----------|-----------|
| **A** | RTX 5090 | hybrid O1 — 26-class, seeds 42/1/2 (ml5120) | 3 | ~10.5 h | ~$5.25 |
| **B** | RTX 5090 | hybrid O1 — vuln-only, seeds 42/1/2 (ml5120) | 3 | ~10.5 h | ~$5.25 |
| **C** | RTX 5090 | graph N48 + seq S1 — 26-class + vuln-only (ml1024) + baseline bundle re-export | 12 | ~9 h | ~$4.50 |
| **D** | RTX 4090 | LOSVER + LineVD + LineVul (baseline bundle) | 3 | ~7.5 h | ~$3.00 |

Rate: RTX 5090 $0.50/h, RTX 4090 $0.40/h.
**Total ≈ $18 compute + ~$2 setup/download ≈ $20** (range $16–32; hybrid varies 3–9 h/run).
Optional graph continual adds ~10 h on ml1024 (~30 short runs) ≈ **+$5**, on whichever pod frees up after the N48 26-class backbones exist (e.g. Pod D after baselines, or Pod C after its main runs).

Data efficiency: graph + seq share the same ml1024 downloads on Pod C; A/B each hold ml5120.
LineVD is 4090-locked (old Joern/DGL); its Joern graph cache on Drive is reused (no Joern re-run).

## Per-pod commands

**Pod A — hybrid 26-class (patch ml5120 base first)**
```bash
bash scripts/patch_reupload_flaw_mask.sh lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42
ML5120=lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42
./scripts/train_cloud.sh --skip \
  --config configs/ablation/phase13/O1_unixcoder_nine.yaml    --dataset $ML5120 \
  --config configs/ablation/phase13/O1_unixcoder_nine_s1.yaml --dataset $ML5120 \
  --config configs/ablation/phase13/O1_unixcoder_nine_s2.yaml --dataset $ML5120
```

**Pod B — hybrid vuln-only (patch ml5120 vulnonly first)**
```bash
bash scripts/patch_reupload_flaw_mask.sh lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_vulnonly
VO5120=lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml5120_f40f2e964_s1600r42_vulnonly
./scripts/train_cloud.sh --skip \
  --config configs/ablation/vulnonly/O1_nine_vulnonly.yaml    --dataset $VO5120 \
  --config configs/ablation/vulnonly/O1_nine_vulnonly_s1.yaml --dataset $VO5120 \
  --config configs/ablation/vulnonly/O1_nine_vulnonly_s2.yaml --dataset $VO5120
```

**Pod C — graph + seq (patch ml1024 base + vulnonly, then re-export bundle, then train)**
```bash
bash scripts/patch_reupload_flaw_mask.sh \
  lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42 \
  lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42_vulnonly \
  lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42 \
  lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42
# (last two = graph continual datasets; drop them if you skip the optional continual re-run)
# --- run the baseline bundle re-export from Prep step 2 here (Pod C holds patched ml1024) ---
ML1024=lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42
VO1024=${ML1024}_vulnonly
./scripts/train_cloud.sh --skip \
  --config configs/ablation/gnn_only/N48_nine.yaml            --dataset $ML1024 \
  --config configs/ablation/gnn_only/N48_nine_s1.yaml         --dataset $ML1024 \
  --config configs/ablation/gnn_only/N48_nine_s2.yaml         --dataset $ML1024 \
  --config configs/ablation/seqgnn/S1_nine.yaml              --dataset $ML1024 \
  --config configs/ablation/seqgnn/S1_nine_s1.yaml           --dataset $ML1024 \
  --config configs/ablation/seqgnn/S1_nine_s2.yaml           --dataset $ML1024 \
  --config configs/ablation/vulnonly/N48_nine_vulnonly.yaml    --dataset $VO1024 \
  --config configs/ablation/vulnonly/N48_nine_vulnonly_s1.yaml --dataset $VO1024 \
  --config configs/ablation/vulnonly/N48_nine_vulnonly_s2.yaml --dataset $VO1024 \
  --config configs/ablation/vulnonly/S1_nine_vulnonly.yaml     --dataset $VO1024 \
  --config configs/ablation/vulnonly/S1_nine_vulnonly_s1.yaml  --dataset $VO1024 \
  --config configs/ablation/vulnonly/S1_nine_vulnonly_s2.yaml  --dataset $VO1024
```

**Pod D — localization baselines (after Pod C uploads the new bundle)**
```bash
# set DATA_TAR to the new megavul_ml1024_baselines_<date>.tar.gz in each script first
bash scripts/run_losver_cloud.sh    # vuln-only, uses patched flaw GT from the bundle
bash scripts/run_linevd_cloud.sh    # 26-class, Joern graph cache reused
bash scripts/run_linevul_cloud.sh   # 26-class
```

**Pod C follow-on — graph continual (optional, after N48 26-class backbones are retrained)**

Depends on the retrained **N48 nine 26-class** checkpoints (from Pod C's `N48_nine{,_s1,_s2}` runs) being
zipped to Drive `checkpoints/`. Continual inits from those, so the run-ids must be updated first.

```bash
# 1. patch the relearn + cil ml1024 datasets (skip if already covered by Pod C's patch line):
bash scripts/patch_reupload_flaw_mask.sh \
  lm_dataset_relearn_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42 \
  lm_dataset_megavul_cil_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42
# 2. Update the per-seed backbone run-ids to the retrained N48 runs (read them from Pod C's
#    training_summary.json / train_cloud upload log), in BOTH orchestrators:
#      scripts/run_relearn_experiment.py     line ~33  "ckpt": {42: "<new>", 1: "<new>", 2: "<new>"}
#      scripts/run_relearn_cil_experiment.py (same ckpt dict)
#    and make sure <run_id>_lmgat_codebert_multiclass_checkpoints.zip for each is on Drive checkpoints/.
# 3. Run domain-incremental + class-incremental, nine graph, 3 seeds:
for S in 42 1 2; do
  RELEARN_NINE=1 PYTHONPATH=src python scripts/run_relearn_experiment.py     --setup --seed $S
  RELEARN_NINE=1 PYTHONPATH=src python scripts/run_relearn_cil_experiment.py --setup --seed $S
done
```

`--setup` downloads the patched relearn/cil datasets + the (updated) backbone from Drive. Outputs the
`RELEARN_RESULTS_nine.md` / `RELEARN_CIL_RESULTS_nine.md` used for Tabel IV.15/IV.16.

## Notes / dependencies

- **Baseline data:** the bundle + LineVD Joern cache live on Drive (`data/baselines/`), not local. The bundle must be **re-exported** (stale GT); the Joern cache is reused as-is.
- **Pod D waits** only for Pod C's bundle upload (~15 min of CPU work near the start of Pod C).
- **Patch each dataset on exactly one pod** (per-pod first line above) so no dataset is re-tarred twice.
- **Output change:** localization now returns the statement text per suspicious line (multi-line span), not just the line number — surfaces automatically in the eval/inference output, no extra step.
- **Classification tables** (IV.6–IV.9) and **continual tables** (IV.15/16) stay valid — the mask never touches them.
