# Baseline Results (from Drive `results/baselines/`)

Metrics extracted from each tool's **native logs** (heterogeneous formats, not our `metrics_summary.json`).
Source archives downloaded from `gdrive-mesach:tugas-akhir/results/baselines/` and extracted to
`results/baselines/<run>/`. VulPCL skipped (data invalid, per user). Numbers copied verbatim from
the logs — see per-row notes and the **Gaps / caveats** section before using in the report.

## Multi-class CWE Classification

| Baseline     | Run                                            | Label space    | Acc   | Macro-F1  | Weighted-F1 | Notes                                                                       |
| ------------ | ---------------------------------------------- | -------------- | ----- | --------- | ----------- | --------------------------------------------------------------------------- |
| LOSVER       | `losver_megavul_ml1024_20260617_141736` (re-eval) | vuln-only 25   | 0.620 | **0.580** | 0.623       | recomputed via compute_baseline_metrics (macro-P 0.607, macro-R 0.623); 23 of 25 CWE have test support |
| LIVABLE      | `livable_megavul_20260617_152415_vo_adamw1e-3` (re-eval) | vuln-only 25   | 0.268 | **0.047** | 0.113       | COLLAPSED to majority class (CWE idx 18, recall 1.0); only 9 of 25 CWE in test n=448; domain shift, run-to-run unstable (prior run was 0.174) |
| VulExplainer | `vulexplainer_megavul_20260617_143234` (re-eval, model_08) | vuln-only      | 0.595 | **0.576** | 0.592       | recomputed (macro-P 0.556, macro-R 0.620, n=914); loaded model_08 (acc 0.595) — the α0.7 model_07 headline (acc 0.646) not in the weights tar |
| EDAT (VTP)   | `edat_megavul_20260616_221911`                 | 26 (benign+25) | 0.411 | 0.170     | 0.381       | **REFERENCE ONLY — dropped from bab-4** (preprint + 26-cls incl benign ≠ vuln-only). macro P 0.16 / R 0.19 |

**Soundness / off-data references (not on our MegaVul test, do not put in the comparison table):**

| Baseline             | Run                                              | Label space       | Acc   | Macro-F1 | Notes                                                            |
| -------------------- | ------------------------------------------------ | ----------------- | ----- | -------- | ---------------------------------------------------------------- |
| LIVABLE (Big-Vul)    | `livable_bigvul_20260615_124939_top31_adamw1e-3` | 31 vuln (Big-Vul) | 0.642 | 0.517    | faithful repro — confirms repo SOUND; megavul gap = domain shift |
| LIVABLE (megavul 26) | `livable_megavul_20260615_082009`                | 26 (benign+25)    | 0.104 | 0.012    | COLLAPSED (benign-majority); off-paper, not faithful — exclude   |

## Statement / Line-Level Localization

| Baseline   | Run                                     | IFA ↓ | Top-1 ↑ | Tops-5 ↑ | Top-10 ↑ | R@1%LOC | R@20%LOC | Effort@20%R ↓ | line-F1 | Notes                                                                                                                          |
| ---------- | --------------------------------------- | ----- | ------- | -------- | -------- | ------- | -------- | ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| LineVul    | `linevul_..._20260617_171401` (re-eval) | 6.144 | 0.220   | 0.591    | 0.780    | 0.036   | 0.372    | 0.085         | —       | our metric (R@5% 0.135); 450 evaluable funcs (447 N/A: 210 >512 tok, 237 wrong-pred); attention localization; binary func-detect F1=0.930 |
| LOSVER     | `losver_..._20260617_160636` (re-eval)  | 0.183 | 0.975   | 0.988    | 0.997    | 0.115   | 0.723    | 0.017         | 0.942   | recomputed via our metric (R@5% 0.494); 678 vuln funcs; line-F1 0.942 = LOSVER native per-line BCE                             |
| EDAT (LVD) | `edat_..._20260616_221911`              | 3.0   | —       | 0.20     | 0.20     | —       | 0.195    | 0.203         | 0.399   | **REFERENCE ONLY — dropped from bab-4.** GLOBAL line ranking (pooled ~58k lines), NOT per-function → not head-to-head; per-func recompute was degenerate. ROC-AUC 0.765 |
| LineVD     | `linevd_megavul_ml1024_20260617_173924` (re-eval) | 0.656 | 0.808   | 0.967    | 0.985    | 0.087   | 0.383    | 0.049         | —       | our metric (R@5% 0.201); 599 vuln funcs (of 1025); GNN per-statement node classification |

## Gaps / caveats (resolve before using in bab-4)

1. **LineVD** — archive contains only the raytune training log (validation metrics). Test classification + statement-localization eval was not captured (`storage/outputs` empty in tar). **Needs a re-eval run** to get test stmt-F1 / IFA / localization.
2. ~~VulExplainer F1 missing~~ **RESOLVED** — re-eval recomputed macro-F1 0.576 / weighted 0.592 / acc 0.595 (model_08). Note: the α0.7 headline (model_07, acc 0.646) weights are NOT in the tar, so only model_08 reproducible.
3. ~~LOSVER macro missing~~ **RESOLVED** — re-eval recomputed macro-F1 0.580 (acc 0.620, weighted 0.623). Localizer still uses a different metric set (not a report localization baseline).
4. **EDAT line-level** — metrics computed by GLOBAL ranking over all pooled test lines, not per-function → IFA/Top-k/R@LOC not directly comparable to our per-function localization. Report with methodology note.
5. **Label-space mismatch** — LOSVER / LIVABLE-vo / VulExplainer are **vuln-only 25-class**; EDAT VTP is **26-class with benign**. For a fair multi-class table, evaluate everything (incl. our model) vuln-only 25-class (per the fairness plan).
6. **VulPCL** — skipped (invalid data). Re-run pending.
7. **Differing test sizes** — recompute n differs per adapter: LOSVER n=678 vs VulExplainer n=914 (both nominally vuln-only test). Their adapters filter the test set differently (e.g. token/length limits), so macro-F1 is over slightly different functions + class supports. Note the per-baseline n; ideally align the vuln-only test set across baselines.
