# Baseline Results (from Drive `results/baselines/`)

Metrics extracted from each tool's **native logs** (heterogeneous formats, not our `metrics_summary.json`).
Source archives downloaded from `gdrive-mesach:tugas-akhir/results/baselines/` and extracted to
`results/baselines/<run>/`. VulPCL skipped (data invalid, per user). Numbers copied verbatim from
the logs — see per-row notes and the **Gaps / caveats** section before using in the report.

## Multi-class CWE Classification

| Baseline     | Run                                            | Label space    | Acc   | Macro-F1  | Weighted-F1 | Notes                                                                       |
| ------------ | ---------------------------------------------- | -------------- | ----- | --------- | ----------- | --------------------------------------------------------------------------- |
| LOSVER       | `losver_megavul_ml1024_20260613_163425`        | vuln-only 25   | 0.620 | —         | **0.623**   | log F1 is **weighted** (acc=recall=0.620, prec-w 0.651); macro NOT reported |
| LIVABLE      | `livable_megavul_20260615_094205_vo`           | vuln-only 25   | 0.208 | **0.174** | —           | macro (P 0.244 / R 0.168); domain shift vs Big-Vul                          |
| VulExplainer | `vulexplainer_megavul_20260616_112007` (α=0.7) | vuln-only      | 0.646 | —         | —           | only `test_accuracy` logged; F1 NOT in log                                  |
| EDAT (VTP)   | `edat_megavul_20260616_221911`                 | 26 (benign+25) | 0.411 | 0.170     | 0.381       | full re-run; macro P 0.16 / R 0.19; includes benign so label space differs  |

**Soundness / off-data references (not on our MegaVul test, do not put in the comparison table):**

| Baseline             | Run                                              | Label space       | Acc   | Macro-F1 | Notes                                                            |
| -------------------- | ------------------------------------------------ | ----------------- | ----- | -------- | ---------------------------------------------------------------- |
| LIVABLE (Big-Vul)    | `livable_bigvul_20260615_124939_top31_adamw1e-3` | 31 vuln (Big-Vul) | 0.642 | 0.517    | faithful repro — confirms repo SOUND; megavul gap = domain shift |
| LIVABLE (megavul 26) | `livable_megavul_20260615_082009`                | 26 (benign+25)    | 0.104 | 0.012    | COLLAPSED (benign-majority); off-paper, not faithful — exclude   |

## Statement / Line-Level Localization

| Baseline   | Run                                     | IFA ↓ | Top-1 ↑ | Tops-5 ↑ | Top-10 ↑ | R@1%LOC | R@20%LOC | Effort@20%R ↓ | line-F1 | Notes                                                                                                                          |
| ---------- | --------------------------------------- | ----- | ------- | -------- | -------- | ------- | -------- | ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| LineVul    | `linevul_..._20260613_161040`           | 6.14  | —       | —        | 0.640    | 0.040   | —        | 0.088         | —       | attention localization; recall@LOC-AUC 0.573; 450 funcs. Binary func-detect F1=0.930 (acc 0.874)                               |
| LOSVER     | `losver_..._20260613_163425`            | —     | ~0.98   | 0.988    | —        | —       | —        | —             | 0.942   | Top-k stmt acc + F1 only; **no IFA / R@LOC / Effort**; one eval block shows 0.999 (anomalous, ignored)                         |
| EDAT (LVD) | `edat_..._20260616_221911`              | 3.0   | —       | 0.20     | 0.20     | —       | 0.195    | 0.203         | 0.399   | **GLOBAL line ranking (pooled ~58k lines), NOT per-function** → not head-to-head; ROC-AUC 0.765, line P 0.604 / R 0.298        |
| LineVD     | `linevd_megavul_ml1024_20260614_142914` | —     | —       | —        | —        | —       | —        | —             | —       | **no test/localization metrics captured** — archive has only raytune train log (val AUROC 0.703, val acc 0.950, val MCC 0.332) |

## Gaps / caveats (resolve before using in bab-4)

1. **LineVD** — archive contains only the raytune training log (validation metrics). Test classification + statement-localization eval was not captured (`storage/outputs` empty in tar). **Needs a re-eval run** to get test stmt-F1 / IFA / localization.
2. **VulExplainer** — only `test_accuracy=0.646` in the log; macro/weighted-F1 not printed. **Re-run eval with F1** or compute from saved predictions.
3. **LOSVER** — classifier reports **weighted** F1 (0.623), not macro. Recompute macro-F1 from predictions for fair comparison vs our macro. Localizer uses a different metric set (Top-k acc + F1), no IFA/R@LOC/Effort.
4. **EDAT line-level** — metrics computed by GLOBAL ranking over all pooled test lines, not per-function → IFA/Top-k/R@LOC not directly comparable to our per-function localization. Report with methodology note.
5. **Label-space mismatch** — LOSVER / LIVABLE-vo / VulExplainer are **vuln-only 25-class**; EDAT VTP is **26-class with benign**. For a fair multi-class table, evaluate everything (incl. our model) vuln-only 25-class (per the fairness plan).
6. **VulPCL** — skipped (invalid data). Re-run pending.
