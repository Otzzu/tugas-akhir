# Dataset Analysis — Complete CWE and Group Distribution

Generated from raw parquet files. Group mapping via `CWE_GROUP_MAP` in `dataset_lm.py`.

**Fixed Group IDs:**
0=benign · 1=memory_safety · 2=numeric · 3=resource_management · 4=input_validation · 5=injection · 6=broken_access_control · 7=authentication_failures · 8=cryptographic_failures · 9=concurrency · 10=code_quality · 11=security_misconfiguration · 12=software_or_data_integrity_failures · 13=logging_and_alerting_failures · 14=mishandling_exceptional_conditions · 15=deprecated · -1=UNKNOWN (not in CWE_GROUP_MAP)

---

## Summary

| Dataset | Total | Benign | Vulnerable | Has CWE | Has Flaw Lines | Notes |
|---|---|---|---|---|---|---|
| BigVul | 217,007 | 206,112 | 10,895 | Yes | Yes (diff) | Primary training dataset |
| DiverseVul | 330,492 | 311,547 | 18,945 | Yes (multi-label) | No | Binary only |
| MegaVul | 55,868 | 27,934 | 27,934 | Yes | Yes (diff) | Balanced 1:1 |
| Devign | 27,318 | 14,858 | 12,460 | No | Yes (vul_lines) | Binary only |
| Merged (BigVul+MegaVul) | 176,674 | 154,205 | 22,469 | Yes | Yes (diff) | Combined |
| TitanVul | 77,096 | 38,548 | 38,548 | Yes | Yes (diff) | Balanced 1:1, Unfiltered |
| BenchVul | 2,100 | 1,050 | 1,050 | Yes | Yes (diff) | **Benchmark for Top 25 Most Dangerous CWEs** |
| ReposVul | 6,897 CVEs → ~115,670 funcs | ~113,952 | ~1,718 | Yes (236 CWEs) | Yes (patch diff) | Multi-granularity, untangled patches |

---

## Language Distribution (sampled from raw_func / func_before)


---

| Dataset | c | cpp | cs | go | java | js | lua | objective-c | php | py | rb | rust | scala | swift | typescript | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BigVul | 99% | 1% | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| MegaVul | 94% | 4% | — | — | 2% | — | — | — | — | — | — | — | — | — | — | — |
| TitanVul | 37% | 7% | 1% | 0% | 22% | 5% | 0% | 0% | 2% | 5% | 1% | 0% | 0% | 0% | 0% | 20% |
| BigVul (raw CPGs) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

> Detected via `joern_runner.detect_language()` on sampled `raw_func` / `func_before` fields.
> Go/PHP fall back to C extension (Joern native frontends produce sparser CPGs).

---

## 1. BigVul (`data/datasets/bigvul/all.parquet`)

Total: **217,007** | Benign: **206,112** | Vulnerable: **10,895**

### Group Distribution

| Group ID | Group | Count | OWASP Top 10 |
|---|---|---|---|
| 0 | benign | 206,112 |  |
| 1 | memory_safety | 3,618 | A10 |
| -1 | UNKNOWN | 2,135 |  |
| 6 | broken_access_control | 1,341 | A01, A06 |
| 4 | input_validation | 1,152 | A05, A06 |
| 3 | resource_management | 916 |  |
| 2 | numeric | 703 | A05, A10 |
| 15 | deprecated | 271 | A06 |
| 9 | concurrency | 266 | A06 |
| 8 | cryptographic_failures | 130 | A04, A06 |
| 10 | code_quality | 121 | A08 |
| 5 | injection | 101 | A05 |
| 11 | security_misconfiguration | 75 | A01, A02, A08 |
| 7 | authentication_failures | 36 | A07 |
| 14 | mishandling_exceptional_conditions | 24 | A10 |
| 12 | software_or_data_integrity_failures | 5 | A06, A08 |
| 13 | logging_and_alerting_failures | 1 | A09 |

> **Unique Groups**: 17

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group | Top 25 | OWASP Top 10 |
|---|---|---|---|---|---|
| CWE-119 | 2,139 | 1 | memory_safety |  |  |
| *(empty/unknown)* | 2,135 | -1 | UNKNOWN |  |  |
| CWE-20 | 1,136 | 4 | input_validation | ✅ | ✅ |
| CWE-399 | 742 | 3 | resource_management |  |  |
| CWE-125 | 630 | 1 | memory_safety | ✅ |  |
| CWE-200 | 508 | 6 | broken_access_control | ✅ | ✅ |
| CWE-264 | 493 | 6 | broken_access_control |  |  |
| CWE-189 | 341 | 2 | numeric |  |  |
| CWE-416 | 323 | 1 | memory_safety | ✅ |  |
| CWE-190 | 311 | 2 | numeric |  |  |
| CWE-362 | 266 | 9 | concurrency |  | ✅ |
| CWE-476 | 225 | 1 | memory_safety | ✅ | ✅ |
| CWE-787 | 191 | 1 | memory_safety | ✅ |  |
| CWE-284 | 169 | 6 | broken_access_control | ✅ | ✅ |
| CWE-254 | 125 | 15 | deprecated |  |  |
| CWE-310 | 88 | 8 | cryptographic_failures |  |  |
| CWE-415 | 78 | 1 | memory_safety |  |  |
| CWE-732 | 65 | 11 | security_misconfiguration |  | ✅ |
| CWE-404 | 64 | 3 | resource_management |  |  |
| CWE-19 | 60 | 15 | deprecated |  |  |
| CWE-59 | 55 | 6 | broken_access_control |  | ✅ |
| CWE-17 | 53 | 15 | deprecated |  |  |
| CWE-79 | 53 | 5 | injection | ✅ | ✅ |
| CWE-772 | 52 | 3 | resource_management |  |  |
| CWE-400 | 43 | 3 | resource_management |  |  |
| CWE-22 | 40 | 6 | broken_access_control | ✅ | ✅ |
| CWE-835 | 37 | 10 | code_quality |  |  |
| CWE-18 | 32 | 15 | deprecated |  |  |
| CWE-269 | 31 | 6 | broken_access_control |  | ✅ |
| CWE-369 | 31 | 2 | numeric |  | ✅ |
| CWE-704 | 25 | 10 | code_quality |  |  |
| CWE-285 | 24 | 6 | broken_access_control |  | ✅ |
| CWE-287 | 22 | 7 | authentication_failures |  | ✅ |
| CWE-134 | 19 | 1 | memory_safety |  |  |
| CWE-617 | 18 | 10 | code_quality |  |  |
| CWE-311 | 18 | 8 | cryptographic_failures |  | ✅ |
| CWE-358 | 17 | 10 | code_quality |  |  |
| CWE-77 | 13 | 5 | injection | ✅ | ✅ |
| CWE-78 | 13 | 5 | injection | ✅ | ✅ |
| CWE-682 | 12 | 2 | numeric |  |  |
| CWE-834 | 11 | 10 | code_quality |  |  |
| CWE-674 | 10 | 10 | code_quality |  |  |
| CWE-754 | 10 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-94 | 10 | 5 | injection | ✅ | ✅ |
| CWE-388 | 10 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-120 | 9 | 1 | memory_safety | ✅ |  |
| CWE-281 | 9 | 6 | broken_access_control |  | ✅ |
| CWE-770 | 9 | 3 | resource_management | ✅ |  |
| CWE-354 | 9 | 4 | input_validation |  |  |
| CWE-347 | 8 | 8 | cryptographic_failures |  | ✅ |
| CWE-320 | 8 | 8 | cryptographic_failures |  |  |
| CWE-611 | 7 | 11 | security_misconfiguration |  | ✅ |
| CWE-255 | 6 | 7 | authentication_failures |  |  |
| CWE-89 | 5 | 5 | injection | ✅ | ✅ |
| CWE-74 | 5 | 5 | injection |  | ✅ |
| CWE-862 | 5 | 6 | broken_access_control | ✅ | ✅ |
| CWE-665 | 5 | 3 | resource_management |  |  |
| CWE-346 | 5 | 7 | authentication_failures |  | ✅ |
| CWE-129 | 4 | 2 | numeric |  | ✅ |
| CWE-191 | 4 | 2 | numeric |  |  |
| CWE-436 | 4 | 4 | input_validation |  | ✅ |
| CWE-601 | 3 | 6 | broken_access_control |  | ✅ |
| CWE-522 | 3 | 8 | cryptographic_failures |  | ✅ |
| CWE-426 | 3 | 10 | code_quality |  | ✅ |
| CWE-172 | 3 | 4 | input_validation |  |  |
| CWE-327 | 3 | 8 | cryptographic_failures |  | ✅ |
| CWE-494 | 3 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-295 | 3 | 7 | authentication_failures |  | ✅ |
| CWE-345 | 2 | 11 | security_misconfiguration |  | ✅ |
| CWE-755 | 2 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-824 | 2 | 1 | memory_safety |  |  |
| CWE-918 | 2 | 6 | broken_access_control | ✅ | ✅ |
| CWE-763 | 2 | 1 | memory_safety |  |  |
| CWE-90 | 2 | 5 | injection |  | ✅ |
| CWE-330 | 2 | 8 | cryptographic_failures |  | ✅ |
| CWE-502 | 1 | 12 | software_or_data_integrity_failures | ✅ | ✅ |
| CWE-1021 | 1 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-693 | 1 | 15 | deprecated |  | ✅ |
| CWE-532 | 1 | 13 | logging_and_alerting_failures |  | ✅ |
| CWE-668 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-664 | 1 | 3 | resource_management |  |  |
| CWE-352 | 1 | 6 | broken_access_control | ✅ | ✅ |
| CWE-252 | 1 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-16 | 1 | 11 | security_misconfiguration |  |  |
| CWE-209 | 1 | 14 | mishandling_exceptional_conditions |  | ✅ |

> **Unique CWEs**: 84
> **Unique Groups**: 15

### Top 25 Most Dangerous CWEs

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-20 | 1,136 | 4 | input_validation |
| CWE-125 | 630 | 1 | memory_safety |
| CWE-200 | 508 | 6 | broken_access_control |
| CWE-416 | 323 | 1 | memory_safety |
| CWE-476 | 225 | 1 | memory_safety |
| CWE-787 | 191 | 1 | memory_safety |
| CWE-284 | 169 | 6 | broken_access_control |
| CWE-79 | 53 | 5 | injection |
| CWE-22 | 40 | 6 | broken_access_control |
| CWE-77 | 13 | 5 | injection |
| CWE-78 | 13 | 5 | injection |
| CWE-94 | 10 | 5 | injection |
| CWE-120 | 9 | 1 | memory_safety |
| CWE-770 | 9 | 3 | resource_management |
| CWE-89 | 5 | 5 | injection |
| CWE-862 | 5 | 6 | broken_access_control |
| CWE-918 | 2 | 6 | broken_access_control |
| CWE-502 | 1 | 12 | software_or_data_integrity_failures |
| CWE-352 | 1 | 6 | broken_access_control |
| **Total** | **3,343** | | |

> **Unique CWEs**: 19
> **Unique Groups**: 6

#### Top 25 Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 1 | memory_safety | 1,378 |
| 4 | input_validation | 1,136 |
| 6 | broken_access_control | 725 |
| 5 | injection | 94 |
| 3 | resource_management | 9 |
| 12 | software_or_data_integrity_failures | 1 |
| **Total** | **3,343** |

### OWASP Top 10 (2025)

| CWE | Count | Group | OWASP |
|---|---|---|---|
| CWE-20 | 1,136 | input_validation | A05:2025 - Injection |
| CWE-200 | 508 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-362 | 266 | concurrency | A06:2025 - Insecure Design |
| CWE-476 | 225 | memory_safety | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-284 | 169 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-732 | 65 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-59 | 55 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-79 | 53 | injection | A05:2025 - Injection |
| CWE-22 | 40 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-269 | 31 | broken_access_control | A06:2025 - Insecure Design |
| CWE-369 | 31 | numeric | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-285 | 24 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-287 | 22 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-311 | 18 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-77 | 13 | injection | A05:2025 - Injection |
| CWE-78 | 13 | injection | A05:2025 - Injection |
| CWE-754 | 10 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-94 | 10 | injection | A05:2025 - Injection |
| CWE-281 | 9 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-347 | 8 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-611 | 7 | security_misconfiguration | A02:2025 - Security Misconfiguration |
| CWE-89 | 5 | injection | A05:2025 - Injection |
| CWE-74 | 5 | injection | A05:2025 - Injection |
| CWE-862 | 5 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-346 | 5 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-129 | 4 | numeric | A05:2025 - Injection |
| CWE-436 | 4 | input_validation | A06:2025 - Insecure Design |
| CWE-601 | 3 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-522 | 3 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-426 | 3 | code_quality | A08:2025 - Software or Data Integrity Failures |
| CWE-327 | 3 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-494 | 3 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-295 | 3 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-345 | 2 | security_misconfiguration | A08:2025 - Software or Data Integrity Failures |
| CWE-755 | 2 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-918 | 2 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-90 | 2 | injection | A05:2025 - Injection |
| CWE-330 | 2 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-502 | 1 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-1021 | 1 | software_or_data_integrity_failures | A06:2025 - Insecure Design |
| CWE-693 | 1 | deprecated | A06:2025 - Insecure Design |
| CWE-532 | 1 | logging_and_alerting_failures | A09:2025 - Logging & Alerting Failures |
| CWE-668 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-352 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-252 | 1 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-209 | 1 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| **Total** | **2,777** | | |

> **Unique CWEs**: 46
> **Unique Groups**: 9

#### OWASP Top 10 Group Distribution

| OWASP Category | Count |
|---|---|
| A05:2025 - Injection | 1,241 |
| A01:2025 - Broken Access Control | 882 |
| A06:2025 - Insecure Design | 324 |
| A10:2025 - Mishandling of Exceptional Conditions | 270 |
| A07:2025 - Authentication Failures | 30 |
| A04:2025 - Cryptographic Failures | 13 |
| A08:2025 - Software or Data Integrity Failures | 9 |
| A02:2025 - Security Misconfiguration | 7 |
| A09:2025 - Logging & Alerting Failures | 1 |
| **Total** | **2,777** |

### CPG Files vs all.parquet (data/raw/bigvul/)

| Group ID | Group | CPG Files | all.parquet | Coverage |
|---|---|---|---|---|
| 0 | benign | N/A | 206,112 | subsampled |

> All vulnerable CPG files = 100% coverage of all.parquet vulnerable.
> Benign subsampled to ~4,000.

---

## 2. DiverseVul (`data/datasets/diversevul/all.parquet`)

Total: **330,492** | Benign: **311,547** | Vulnerable: **18,945**

> CWE column is multi-label array (e.g. `['CWE-787', 'CWE-119']`).
> Group assignment uses primary (first) CWE only.
> No flaw line ground truth available.

### Group Distribution

| Group ID | Group | Count | OWASP Top 10 |
|---|---|---|---|
| 0 | benign | 311,547 |  |
| 1 | memory_safety | 7,065 | A10 |
| -1 | UNKNOWN | 2,836 |  |
| 6 | broken_access_control | 1,816 | A01, A06, A07 |
| 4 | input_validation | 1,400 | A05, A06 |
| 3 | resource_management | 1,383 | A02 |
| 2 | numeric | 1,187 | A05, A10 |
| 14 | mishandling_exceptional_conditions | 869 | A10 |
| 8 | cryptographic_failures | 547 | A04, A06 |
| 9 | concurrency | 450 | A06 |
| 5 | injection | 395 | A05, A06 |
| 10 | code_quality | 393 | A08 |
| 7 | authentication_failures | 304 | A07, A08 |
| 15 | deprecated | 179 | A06 |
| 11 | security_misconfiguration | 96 | A01, A02, A08 |
| 12 | software_or_data_integrity_failures | 18 | A06, A08 |
| 13 | logging_and_alerting_failures | 7 | A09 |

> **Unique Groups**: 17

### CWE Distribution (all vulnerable, primary CWE)

| CWE | Count | Group ID | Group | Top 25 | OWASP Top 10 |
|---|---|---|---|---|---|
| *(empty/unknown)* | 2,836 | -1 | UNKNOWN |  |  |
| CWE-125 | 1,635 | 1 | memory_safety | ✅ |  |
| CWE-119 | 1,433 | 1 | memory_safety |  |  |
| CWE-787 | 1,379 | 1 | memory_safety | ✅ |  |
| CWE-20 | 1,315 | 4 | input_validation | ✅ | ✅ |
| CWE-416 | 999 | 1 | memory_safety | ✅ |  |
| CWE-476 | 915 | 1 | memory_safety | ✅ | ✅ |
| CWE-703 | 735 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-200 | 717 | 6 | broken_access_control | ✅ | ✅ |
| CWE-190 | 674 | 2 | numeric |  |  |
| CWE-399 | 462 | 3 | resource_management |  |  |
| CWE-362 | 398 | 9 | concurrency |  | ✅ |
| CWE-400 | 395 | 3 | resource_management |  |  |
| CWE-310 | 343 | 8 | cryptographic_failures |  |  |
| CWE-284 | 286 | 6 | broken_access_control | ✅ | ✅ |
| CWE-120 | 285 | 1 | memory_safety | ✅ |  |
| CWE-189 | 273 | 2 | numeric |  |  |
| CWE-415 | 247 | 1 | memory_safety |  |  |
| CWE-264 | 217 | 6 | broken_access_control |  |  |
| CWE-401 | 193 | 3 | resource_management |  |  |
| CWE-59 | 164 | 6 | broken_access_control |  | ✅ |
| CWE-617 | 162 | 10 | code_quality |  |  |
| CWE-369 | 158 | 2 | numeric |  | ✅ |
| CWE-22 | 140 | 6 | broken_access_control | ✅ | ✅ |
| CWE-269 | 111 | 6 | broken_access_control |  | ✅ |
| CWE-835 | 111 | 10 | code_quality |  |  |
| CWE-287 | 105 | 7 | authentication_failures |  | ✅ |
| CWE-295 | 101 | 7 | authentication_failures |  | ✅ |
| CWE-772 | 100 | 3 | resource_management |  |  |
| CWE-770 | 89 | 3 | resource_management | ✅ |  |
| CWE-94 | 84 | 5 | injection | ✅ | ✅ |
| CWE-78 | 76 | 5 | injection | ✅ | ✅ |
| CWE-122 | 68 | 1 | memory_safety | ✅ |  |
| CWE-19 | 66 | 15 | deprecated |  |  |
| CWE-444 | 60 | 5 | injection |  | ✅ |
| CWE-241 | 57 | 4 | input_validation |  |  |
| CWE-674 | 56 | 10 | code_quality |  |  |
| CWE-89 | 54 | 5 | injection | ✅ | ✅ |
| CWE-17 | 46 | 15 | deprecated |  |  |
| CWE-755 | 46 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-191 | 45 | 2 | numeric |  |  |
| CWE-459 | 43 | 3 | resource_management |  |  |
| CWE-862 | 42 | 6 | broken_access_control | ✅ | ✅ |
| CWE-613 | 40 | 7 | authentication_failures |  | ✅ |
| CWE-330 | 39 | 8 | cryptographic_failures |  | ✅ |
| CWE-732 | 38 | 11 | security_misconfiguration |  | ✅ |
| CWE-327 | 38 | 8 | cryptographic_failures |  | ✅ |
| CWE-134 | 38 | 1 | memory_safety |  |  |
| CWE-319 | 35 | 8 | cryptographic_failures |  | ✅ |
| CWE-601 | 35 | 6 | broken_access_control |  | ✅ |
| CWE-18 | 34 | 15 | deprecated |  |  |
| CWE-665 | 32 | 3 | resource_management |  |  |
| CWE-77 | 31 | 5 | injection | ✅ | ✅ |
| CWE-254 | 31 | 15 | deprecated |  |  |
| CWE-754 | 30 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-704 | 28 | 10 | code_quality |  |  |
| CWE-79 | 27 | 5 | injection | ✅ | ✅ |
| CWE-74 | 25 | 5 | injection |  | ✅ |
| CWE-326 | 25 | 8 | cryptographic_failures |  | ✅ |
| CWE-347 | 25 | 8 | cryptographic_failures |  | ✅ |
| CWE-203 | 24 | 6 | broken_access_control |  |  |
| CWE-252 | 24 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-290 | 23 | 7 | authentication_failures |  | ✅ |
| CWE-662 | 22 | 9 | concurrency |  |  |
| CWE-354 | 21 | 4 | input_validation |  |  |
| CWE-667 | 21 | 9 | concurrency |  |  |
| CWE-908 | 21 | 3 | resource_management |  |  |
| CWE-345 | 20 | 11 | security_misconfiguration |  | ✅ |
| CWE-358 | 20 | 10 | code_quality |  |  |
| CWE-276 | 19 | 11 | security_misconfiguration |  | ✅ |
| CWE-406 | 19 | 3 | resource_management |  |  |
| CWE-209 | 18 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-863 | 18 | 6 | broken_access_control | ✅ | ✅ |
| CWE-763 | 17 | 1 | memory_safety |  |  |
| CWE-434 | 15 | 5 | injection | ✅ | ✅ |
| CWE-843 | 14 | 1 | memory_safety |  |  |
| CWE-611 | 14 | 11 | security_misconfiguration |  | ✅ |
| CWE-288 | 13 | 7 | authentication_failures |  | ✅ |
| CWE-323 | 13 | 8 | cryptographic_failures |  | ✅ |
| CWE-668 | 12 | 6 | broken_access_control |  | ✅ |
| CWE-824 | 12 | 1 | memory_safety |  |  |
| CWE-522 | 11 | 8 | cryptographic_failures |  | ✅ |
| CWE-193 | 11 | 2 | numeric |  |  |
| CWE-131 | 11 | 2 | numeric |  |  |
| CWE-552 | 10 | 6 | broken_access_control |  | ✅ |
| CWE-913 | 10 | 12 | software_or_data_integrity_failures |  |  |
| CWE-909 | 10 | 3 | resource_management |  |  |
| CWE-388 | 10 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-116 | 9 | 5 | injection |  | ✅ |
| CWE-834 | 9 | 10 | code_quality |  |  |
| CWE-367 | 9 | 9 | concurrency |  |  |
| CWE-320 | 9 | 8 | cryptographic_failures |  |  |
| CWE-404 | 8 | 3 | resource_management |  |  |
| CWE-126 | 7 | 1 | memory_safety |  |  |
| CWE-121 | 7 | 1 | memory_safety | ✅ |  |
| CWE-212 | 7 | 6 | broken_access_control |  |  |
| CWE-532 | 7 | 13 | logging_and_alerting_failures |  | ✅ |
| CWE-294 | 7 | 7 | authentication_failures |  | ✅ |
| CWE-682 | 6 | 2 | numeric |  |  |
| CWE-297 | 6 | 6 | broken_access_control |  | ✅ |
| CWE-918 | 6 | 6 | broken_access_control | ✅ | ✅ |
| CWE-273 | 6 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-707 | 6 | 4 | input_validation |  |  |
| CWE-281 | 6 | 6 | broken_access_control |  | ✅ |
| CWE-1187 | 6 | 3 | resource_management |  |  |
| CWE-346 | 6 | 7 | authentication_failures |  | ✅ |
| CWE-16 | 5 | 11 | security_misconfiguration |  |  |
| CWE-681 | 5 | 2 | numeric |  |  |
| CWE-502 | 4 | 12 | software_or_data_integrity_failures | ✅ | ✅ |
| CWE-1021 | 4 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-823 | 4 | 1 | memory_safety |  |  |
| CWE-61 | 4 | 6 | broken_access_control |  | ✅ |
| CWE-303 | 4 | 7 | authentication_failures |  | ✅ |
| CWE-88 | 4 | 5 | injection |  | ✅ |
| CWE-91 | 4 | 5 | injection |  | ✅ |
| CWE-311 | 4 | 8 | cryptographic_failures |  | ✅ |
| CWE-331 | 4 | 8 | cryptographic_failures |  | ✅ |
| CWE-352 | 4 | 6 | broken_access_control | ✅ | ✅ |
| CWE-129 | 3 | 2 | numeric |  | ✅ |
| CWE-426 | 3 | 10 | code_quality |  | ✅ |
| CWE-672 | 3 | 1 | memory_safety |  |  |
| CWE-670 | 3 | 10 | code_quality |  |  |
| CWE-113 | 3 | 5 | injection |  | ✅ |
| CWE-417 | 2 | 3 | resource_management |  |  |
| CWE-776 | 2 | 3 | resource_management |  | ✅ |
| CWE-924 | 2 | 6 | broken_access_control |  |  |
| CWE-307 | 2 | 7 | authentication_failures |  | ✅ |
| CWE-693 | 2 | 15 | deprecated |  | ✅ |
| CWE-266 | 2 | 6 | broken_access_control |  | ✅ |
| CWE-93 | 2 | 5 | injection |  | ✅ |
| CWE-285 | 2 | 6 | broken_access_control |  | ✅ |
| CWE-697 | 1 | 2 | numeric |  |  |
| CWE-565 | 1 | 7 | authentication_failures |  | ✅ |
| CWE-943 | 1 | 5 | injection |  |  |
| CWE-255 | 1 | 7 | authentication_failures |  |  |
| CWE-349 | 1 | 8 | cryptographic_failures |  |  |
| CWE-457 | 1 | 3 | resource_management |  |  |
| CWE-805 | 1 | 1 | memory_safety |  |  |
| CWE-436 | 1 | 4 | input_validation |  | ✅ |
| CWE-798 | 1 | 7 | authentication_failures |  | ✅ |
| CWE-786 | 1 | 1 | memory_safety |  |  |
| CWE-428 | 1 | 10 | code_quality |  |  |
| CWE-706 | 1 | 6 | broken_access_control |  |  |

> **Unique CWEs**: 142
> **Unique Groups**: 15

### Top 25 Most Dangerous CWEs

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-125 | 1,635 | 1 | memory_safety |
| CWE-787 | 1,379 | 1 | memory_safety |
| CWE-20 | 1,315 | 4 | input_validation |
| CWE-416 | 999 | 1 | memory_safety |
| CWE-476 | 915 | 1 | memory_safety |
| CWE-200 | 717 | 6 | broken_access_control |
| CWE-284 | 286 | 6 | broken_access_control |
| CWE-120 | 285 | 1 | memory_safety |
| CWE-22 | 140 | 6 | broken_access_control |
| CWE-770 | 89 | 3 | resource_management |
| CWE-94 | 84 | 5 | injection |
| CWE-78 | 76 | 5 | injection |
| CWE-122 | 68 | 1 | memory_safety |
| CWE-89 | 54 | 5 | injection |
| CWE-862 | 42 | 6 | broken_access_control |
| CWE-77 | 31 | 5 | injection |
| CWE-79 | 27 | 5 | injection |
| CWE-863 | 18 | 6 | broken_access_control |
| CWE-434 | 15 | 5 | injection |
| CWE-121 | 7 | 1 | memory_safety |
| CWE-918 | 6 | 6 | broken_access_control |
| CWE-502 | 4 | 12 | software_or_data_integrity_failures |
| CWE-352 | 4 | 6 | broken_access_control |
| **Total** | **8,196** | | |

> **Unique CWEs**: 23
> **Unique Groups**: 6

#### Top 25 Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 1 | memory_safety | 5,288 |
| 4 | input_validation | 1,315 |
| 6 | broken_access_control | 1,213 |
| 5 | injection | 287 |
| 3 | resource_management | 89 |
| 12 | software_or_data_integrity_failures | 4 |
| **Total** | **8,196** |

### OWASP Top 10 (2025)

| CWE | Count | Group | OWASP |
|---|---|---|---|
| CWE-20 | 1,315 | input_validation | A05:2025 - Injection |
| CWE-476 | 915 | memory_safety | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-703 | 735 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-200 | 717 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-362 | 398 | concurrency | A06:2025 - Insecure Design |
| CWE-284 | 286 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-59 | 164 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-369 | 158 | numeric | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-22 | 140 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-269 | 111 | broken_access_control | A06:2025 - Insecure Design |
| CWE-287 | 105 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-295 | 101 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-94 | 84 | injection | A05:2025 - Injection |
| CWE-78 | 76 | injection | A05:2025 - Injection |
| CWE-444 | 60 | injection | A06:2025 - Insecure Design |
| CWE-89 | 54 | injection | A05:2025 - Injection |
| CWE-755 | 46 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-862 | 42 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-613 | 40 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-330 | 39 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-732 | 38 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-327 | 38 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-319 | 35 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-601 | 35 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-77 | 31 | injection | A05:2025 - Injection |
| CWE-754 | 30 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-79 | 27 | injection | A05:2025 - Injection |
| CWE-74 | 25 | injection | A05:2025 - Injection |
| CWE-326 | 25 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-347 | 25 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-252 | 24 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-290 | 23 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-345 | 20 | security_misconfiguration | A08:2025 - Software or Data Integrity Failures |
| CWE-276 | 19 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-209 | 18 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-863 | 18 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-434 | 15 | injection | A06:2025 - Insecure Design |
| CWE-611 | 14 | security_misconfiguration | A02:2025 - Security Misconfiguration |
| CWE-288 | 13 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-323 | 13 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-668 | 12 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-522 | 11 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-552 | 10 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-116 | 9 | injection | A05:2025 - Injection |
| CWE-532 | 7 | logging_and_alerting_failures | A09:2025 - Logging & Alerting Failures |
| CWE-294 | 7 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-297 | 6 | broken_access_control | A07:2025 - Authentication Failures |
| CWE-918 | 6 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-281 | 6 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-346 | 6 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-502 | 4 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-1021 | 4 | software_or_data_integrity_failures | A06:2025 - Insecure Design |
| CWE-61 | 4 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-303 | 4 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-88 | 4 | injection | A05:2025 - Injection |
| CWE-91 | 4 | injection | A05:2025 - Injection |
| CWE-311 | 4 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-331 | 4 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-352 | 4 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-129 | 3 | numeric | A05:2025 - Injection |
| CWE-426 | 3 | code_quality | A08:2025 - Software or Data Integrity Failures |
| CWE-113 | 3 | injection | A05:2025 - Injection |
| CWE-776 | 2 | resource_management | A02:2025 - Security Misconfiguration |
| CWE-307 | 2 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-693 | 2 | deprecated | A06:2025 - Insecure Design |
| CWE-266 | 2 | broken_access_control | A06:2025 - Insecure Design |
| CWE-93 | 2 | injection | A05:2025 - Injection |
| CWE-285 | 2 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-565 | 1 | authentication_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-436 | 1 | input_validation | A06:2025 - Insecure Design |
| CWE-798 | 1 | authentication_failures | A07:2025 - Authentication Failures |
| **Total** | **6,212** | | |

> **Unique CWEs**: 71
> **Unique Groups**: 9

#### OWASP Top 10 Group Distribution

| OWASP Category | Count |
|---|---|
| A10:2025 - Mishandling of Exceptional Conditions | 1,926 |
| A05:2025 - Injection | 1,637 |
| A01:2025 - Broken Access Control | 1,503 |
| A06:2025 - Insecure Design | 608 |
| A07:2025 - Authentication Failures | 308 |
| A04:2025 - Cryptographic Failures | 179 |
| A08:2025 - Software or Data Integrity Failures | 28 |
| A02:2025 - Security Misconfiguration | 16 |
| A09:2025 - Logging & Alerting Failures | 7 |
| **Total** | **6,212** |

---

## 3. MegaVul (`data/datasets/megavul/train.parquet`)

Total: **395,822** | Benign: **375,414** | Vulnerable: **20,408**

### Group Distribution

| Group ID | Group | Count | OWASP Top 10 |
|---|---|---|---|
| 0 | benign | 375,414 |  |
| 1 | memory_safety | 7,659 | A10 |
| 6 | broken_access_control | 2,249 | A01, A05, A06, A07 |
| 3 | resource_management | 1,883 | A02 |
| -1 | UNKNOWN | 1,768 | A01, A02, A06 |
| 2 | numeric | 1,398 | A05, A10 |
| 4 | input_validation | 1,317 | A05, A06 |
| 5 | injection | 1,001 | A05, A06, A08 |
| 10 | code_quality | 814 | A06, A08 |
| 9 | concurrency | 686 | A06 |
| 7 | authentication_failures | 401 | A07 |
| 11 | security_misconfiguration | 361 | A01, A02, A08 |
| 8 | cryptographic_failures | 300 | A04, A06 |
| 15 | deprecated | 283 | A06 |
| 14 | mishandling_exceptional_conditions | 144 | A10 |
| 12 | software_or_data_integrity_failures | 132 | A06, A08 |
| 13 | logging_and_alerting_failures | 12 | A09 |

> **Unique Groups**: 17

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group | Top 25 | OWASP Top 10 |
|---|---|---|---|---|---|
| *(empty/unknown)* | 1,719 | -1 | UNKNOWN |  |  |
| CWE-119 | 1,621 | 1 | memory_safety |  |  |
| CWE-787 | 1,553 | 1 | memory_safety | ✅ |  |
| CWE-125 | 1,330 | 1 | memory_safety | ✅ |  |
| CWE-476 | 1,265 | 1 | memory_safety | ✅ | ✅ |
| CWE-20 | 1,187 | 4 | input_validation | ✅ | ✅ |
| CWE-416 | 975 | 1 | memory_safety | ✅ |  |
| CWE-190 | 723 | 2 | numeric |  |  |
| CWE-200 | 603 | 6 | broken_access_control | ✅ | ✅ |
| CWE-362 | 560 | 9 | concurrency |  | ✅ |
| CWE-264 | 544 | 6 | broken_access_control |  |  |
| CWE-399 | 471 | 3 | resource_management |  |  |
| CWE-120 | 424 | 1 | memory_safety | ✅ |  |
| CWE-400 | 420 | 3 | resource_management |  |  |
| CWE-79 | 336 | 5 | injection | ✅ | ✅ |
| CWE-22 | 323 | 6 | broken_access_control | ✅ | ✅ |
| CWE-401 | 288 | 3 | resource_management |  |  |
| CWE-189 | 261 | 2 | numeric |  |  |
| CWE-617 | 256 | 10 | code_quality |  |  |
| CWE-835 | 247 | 10 | code_quality |  |  |
| CWE-611 | 229 | 11 | security_misconfiguration |  | ✅ |
| CWE-287 | 214 | 7 | authentication_failures |  | ✅ |
| CWE-89 | 209 | 5 | injection | ✅ | ✅ |
| CWE-415 | 195 | 1 | memory_safety |  |  |
| CWE-772 | 178 | 3 | resource_management |  |  |
| CWE-369 | 172 | 2 | numeric |  | ✅ |
| CWE-254 | 142 | 15 | deprecated |  |  |
| CWE-770 | 138 | 3 | resource_management | ✅ |  |
| CWE-674 | 130 | 10 | code_quality |  |  |
| CWE-74 | 130 | 5 | injection |  | ✅ |
| CWE-502 | 112 | 12 | software_or_data_integrity_failures | ✅ | ✅ |
| CWE-122 | 111 | 1 | memory_safety | ✅ |  |
| CWE-284 | 107 | 6 | broken_access_control | ✅ | ✅ |
| CWE-908 | 99 | 3 | resource_management |  |  |
| CWE-863 | 98 | 6 | broken_access_control | ✅ | ✅ |
| CWE-668 | 97 | 6 | broken_access_control |  | ✅ |
| CWE-310 | 96 | 8 | cryptographic_failures |  |  |
| CWE-834 | 95 | 10 | code_quality |  |  |
| CWE-78 | 94 | 5 | injection | ✅ | ✅ |
| CWE-269 | 84 | 6 | broken_access_control |  | ✅ |
| CWE-295 | 83 | 7 | authentication_failures |  | ✅ |
| CWE-59 | 79 | 6 | broken_access_control |  | ✅ |
| CWE-862 | 77 | 6 | broken_access_control | ✅ | ✅ |
| CWE-667 | 72 | 9 | concurrency |  |  |
| CWE-909 | 72 | 3 | resource_management |  |  |
| CWE-404 | 68 | 3 | resource_management |  |  |
| CWE-17 | 67 | 15 | deprecated |  |  |
| CWE-732 | 63 | 11 | security_misconfiguration |  | ✅ |
| CWE-19 | 63 | 15 | deprecated |  |  |
| CWE-459 | 59 | 3 | resource_management |  |  |
| CWE-843 | 56 | 1 | memory_safety |  |  |
| CWE-193 | 56 | 2 | numeric |  |  |
| CWE-681 | 56 | 2 | numeric |  |  |
| CWE-94 | 56 | 5 | injection | ✅ | ✅ |
| CWE-354 | 50 | 4 | input_validation |  |  |
| CWE-665 | 48 | 3 | resource_management |  |  |
| CWE-918 | 47 | 6 | broken_access_control | ✅ | ✅ |
| CWE-345 | 45 | 11 | security_misconfiguration |  | ✅ |
| CWE-434 | 44 | 5 | injection | ✅ | ✅ |
| CWE-352 | 44 | 6 | broken_access_control | ✅ | ✅ |
| CWE-755 | 42 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-252 | 41 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-203 | 39 | 6 | broken_access_control |  |  |
| CWE-754 | 38 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-77 | 38 | 5 | injection | ✅ | ✅ |
| CWE-704 | 37 | 10 | code_quality |  |  |
| CWE-327 | 37 | 8 | cryptographic_failures |  | ✅ |
| CWE-444 | 36 | 5 | injection |  | ✅ |
| CWE-330 | 33 | 8 | cryptographic_failures |  | ✅ |
| CWE-129 | 32 | 2 | numeric |  | ✅ |
| CWE-367 | 32 | 9 | concurrency |  |  |
| CWE-1284 | 31 | 4 | input_validation |  |  |
| CWE-134 | 30 | 1 | memory_safety |  |  |
| CWE-763 | 28 | 1 | memory_safety |  |  |
| CWE-682 | 27 | 2 | numeric |  |  |
| CWE-824 | 26 | 1 | memory_safety |  |  |
| CWE-601 | 25 | 6 | broken_access_control |  | ✅ |
| CWE-326 | 25 | 8 | cryptographic_failures |  | ✅ |
| CWE-116 | 25 | 5 | injection |  | ✅ |
| CWE-384 | 25 | 7 | authentication_failures |  | ✅ |
| CWE-552 | 23 | 6 | broken_access_control |  | ✅ |
| CWE-697 | 23 | 2 | numeric |  |  |
| CWE-121 | 22 | 1 | memory_safety | ✅ |  |
| CWE-191 | 22 | 2 | numeric |  |  |
| CWE-436 | 22 | 4 | input_validation |  | ✅ |
| CWE-131 | 21 | 2 | numeric |  |  |
| CWE-306 | 21 | 7 | authentication_failures | ✅ | ✅ |
| CWE-346 | 21 | 7 | authentication_failures |  | ✅ |
| CWE-347 | 20 | 8 | cryptographic_failures |  | ✅ |
| CWE-776 | 19 | 3 | resource_management |  | ✅ |
| CWE-312 | 19 | 8 | cryptographic_failures |  | ✅ |
| CWE-276 | 17 | 11 | security_misconfiguration |  | ✅ |
| CWE-662 | 16 | 9 | concurrency |  |  |
| CWE-290 | 15 | 7 | authentication_failures |  | ✅ |
| CWE-212 | 13 | 6 | broken_access_control |  |  |
| CWE-917 | 13 | 5 | injection |  | ✅ |
| CWE-440 | 12 | 10 | code_quality |  |  |
| CWE-522 | 12 | 8 | cryptographic_failures |  | ✅ |
| CWE-532 | 11 | 13 | logging_and_alerting_failures |  | ✅ |
| CWE-338 | 11 | 8 | cryptographic_failures |  | ✅ |
| CWE-426 | 10 | 10 | code_quality |  | ✅ |
| CWE-672 | 10 | 1 | memory_safety |  |  |
| CWE-197 | 10 | -1 | UNKNOWN |  |  |
| CWE-388 | 9 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-319 | 9 | 8 | cryptographic_failures |  | ✅ |
| CWE-1021 | 9 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-241 | 9 | 4 | input_validation |  |  |
| CWE-613 | 9 | 7 | authentication_failures |  | ✅ |
| CWE-428 | 8 | 10 | code_quality |  |  |
| CWE-670 | 8 | 10 | code_quality |  |  |
| CWE-913 | 8 | 12 | software_or_data_integrity_failures |  |  |
| CWE-378 | 8 | 6 | broken_access_control |  |  |
| CWE-260 | 8 | -1 | UNKNOWN |  | ✅ |
| CWE-911 | 7 | 3 | resource_management |  |  |
| CWE-285 | 7 | 6 | broken_access_control |  | ✅ |
| CWE-320 | 7 | 8 | cryptographic_failures |  |  |
| CWE-18 | 7 | 15 | deprecated |  |  |
| CWE-361 | 6 | 9 | concurrency |  |  |
| CWE-331 | 6 | 8 | cryptographic_failures |  | ✅ |
| CWE-172 | 6 | 4 | input_validation |  |  |
| CWE-385 | 6 | 8 | cryptographic_failures |  |  |
| CWE-407 | 6 | 3 | resource_management |  |  |
| CWE-470 | 6 | 4 | input_validation |  | ✅ |
| CWE-281 | 6 | 6 | broken_access_control |  | ✅ |
| CWE-126 | 5 | 1 | memory_safety |  |  |
| CWE-823 | 5 | 1 | memory_safety |  |  |
| CWE-248 | 5 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-255 | 5 | 7 | authentication_failures |  |  |
| CWE-337 | 5 | 8 | cryptographic_failures |  | ✅ |
| CWE-90 | 5 | 5 | injection |  | ✅ |
| CWE-1187 | 5 | 3 | resource_management |  |  |
| CWE-639 | 5 | 6 | broken_access_control | ✅ | ✅ |
| CWE-409 | 5 | -1 | UNKNOWN |  |  |
| CWE-91 | 5 | 5 | injection |  | ✅ |
| CWE-916 | 5 | 8 | cryptographic_failures |  | ✅ |
| CWE-693 | 4 | 15 | deprecated |  | ✅ |
| CWE-273 | 4 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-427 | 4 | 10 | code_quality |  | ✅ |
| CWE-16 | 4 | 11 | security_misconfiguration |  |  |
| CWE-1077 | 4 | 2 | numeric |  |  |
| CWE-626 | 4 | 10 | code_quality |  |  |
| CWE-88 | 4 | 5 | injection |  | ✅ |
| CWE-209 | 4 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-610 | 4 | 6 | broken_access_control |  | ✅ |
| CWE-798 | 4 | 7 | authentication_failures |  | ✅ |
| CWE-1188 | 3 | 11 | security_misconfiguration |  |  |
| CWE-494 | 3 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-1333 | 3 | 3 | resource_management |  |  |
| CWE-118 | 3 | 1 | memory_safety |  |  |
| CWE-184 | 3 | 4 | input_validation |  |  |
| CWE-229 | 3 | -1 | UNKNOWN |  |  |
| CWE-307 | 3 | 7 | authentication_failures |  | ✅ |
| CWE-325 | 3 | 8 | cryptographic_failures |  | ✅ |
| CWE-706 | 3 | 6 | broken_access_control |  |  |
| CWE-707 | 3 | 4 | input_validation |  |  |
| CWE-648 | 3 | -1 | UNKNOWN |  |  |
| CWE-130 | 3 | -1 | UNKNOWN |  |  |
| CWE-377 | 3 | 6 | broken_access_control |  | ✅ |
| CWE-113 | 2 | 5 | injection |  | ✅ |
| CWE-762 | 2 | -1 | UNKNOWN |  |  |
| CWE-208 | 2 | -1 | UNKNOWN |  |  |
| CWE-924 | 2 | 6 | broken_access_control |  |  |
| CWE-324 | 2 | 8 | cryptographic_failures |  | ✅ |
| CWE-358 | 2 | 10 | code_quality |  |  |
| CWE-335 | 2 | 8 | cryptographic_failures |  | ✅ |
| CWE-838 | 2 | -1 | UNKNOWN |  |  |
| CWE-214 | 2 | 6 | broken_access_control |  |  |
| CWE-749 | 2 | -1 | UNKNOWN |  | ✅ |
| CWE-379 | 2 | -1 | UNKNOWN |  | ✅ |
| CWE-176 | 2 | -1 | UNKNOWN |  |  |
| CWE-829 | 2 | 5 | injection |  | ✅ |
| CWE-185 | 1 | 2 | numeric |  |  |
| CWE-1049 | 1 | 3 | resource_management |  |  |
| CWE-628 | 1 | 10 | code_quality |  | ✅ |
| CWE-202 | 1 | 6 | broken_access_control |  |  |
| CWE-460 | 1 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-73 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-93 | 1 | 5 | injection |  | ✅ |
| CWE-117 | 1 | 13 | logging_and_alerting_failures |  | ✅ |
| CWE-788 | 1 | -1 | UNKNOWN |  |  |
| CWE-1050 | 1 | 3 | resource_management |  |  |
| CWE-282 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-323 | 1 | 8 | cryptographic_failures |  | ✅ |
| CWE-943 | 1 | 5 | injection |  |  |
| CWE-300 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-26 | 1 | -1 | UNKNOWN |  |  |
| CWE-349 | 1 | 8 | cryptographic_failures |  |  |
| CWE-501 | 1 | -1 | UNKNOWN |  | ✅ |
| CWE-612 | 1 | -1 | UNKNOWN |  |  |
| CWE-359 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-538 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-840 | 1 | -1 | UNKNOWN |  |  |
| CWE-521 | 1 | 7 | authentication_failures |  | ✅ |

> **Unique CWEs**: 192
> **Unique Groups**: 16

### Top 25 Most Dangerous CWEs

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-787 | 1,553 | 1 | memory_safety |
| CWE-125 | 1,330 | 1 | memory_safety |
| CWE-476 | 1,265 | 1 | memory_safety |
| CWE-20 | 1,187 | 4 | input_validation |
| CWE-416 | 975 | 1 | memory_safety |
| CWE-200 | 603 | 6 | broken_access_control |
| CWE-120 | 424 | 1 | memory_safety |
| CWE-79 | 336 | 5 | injection |
| CWE-22 | 323 | 6 | broken_access_control |
| CWE-89 | 209 | 5 | injection |
| CWE-770 | 138 | 3 | resource_management |
| CWE-502 | 112 | 12 | software_or_data_integrity_failures |
| CWE-122 | 111 | 1 | memory_safety |
| CWE-284 | 107 | 6 | broken_access_control |
| CWE-863 | 98 | 6 | broken_access_control |
| CWE-78 | 94 | 5 | injection |
| CWE-862 | 77 | 6 | broken_access_control |
| CWE-94 | 56 | 5 | injection |
| CWE-918 | 47 | 6 | broken_access_control |
| CWE-434 | 44 | 5 | injection |
| CWE-352 | 44 | 6 | broken_access_control |
| CWE-77 | 38 | 5 | injection |
| CWE-121 | 22 | 1 | memory_safety |
| CWE-306 | 21 | 7 | authentication_failures |
| CWE-639 | 5 | 6 | broken_access_control |
| **Total** | **9,219** | | |

> **Unique CWEs**: 25
> **Unique Groups**: 7

#### Top 25 Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 1 | memory_safety | 5,680 |
| 6 | broken_access_control | 1,304 |
| 4 | input_validation | 1,187 |
| 5 | injection | 777 |
| 3 | resource_management | 138 |
| 12 | software_or_data_integrity_failures | 112 |
| 7 | authentication_failures | 21 |
| **Total** | **9,219** |

### OWASP Top 10 (2025)

| CWE | Count | Group | OWASP |
|---|---|---|---|
| CWE-476 | 1,265 | memory_safety | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-20 | 1,187 | input_validation | A05:2025 - Injection |
| CWE-200 | 603 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-362 | 560 | concurrency | A06:2025 - Insecure Design |
| CWE-79 | 336 | injection | A05:2025 - Injection |
| CWE-22 | 323 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-611 | 229 | security_misconfiguration | A02:2025 - Security Misconfiguration |
| CWE-287 | 214 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-89 | 209 | injection | A05:2025 - Injection |
| CWE-369 | 172 | numeric | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-74 | 130 | injection | A05:2025 - Injection |
| CWE-502 | 112 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-284 | 107 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-863 | 98 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-668 | 97 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-78 | 94 | injection | A05:2025 - Injection |
| CWE-269 | 84 | broken_access_control | A06:2025 - Insecure Design |
| CWE-295 | 83 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-59 | 79 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-862 | 77 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-732 | 63 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-94 | 56 | injection | A05:2025 - Injection |
| CWE-918 | 47 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-345 | 45 | security_misconfiguration | A08:2025 - Software or Data Integrity Failures |
| CWE-434 | 44 | injection | A06:2025 - Insecure Design |
| CWE-352 | 44 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-755 | 42 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-252 | 41 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-754 | 38 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-77 | 38 | injection | A05:2025 - Injection |
| CWE-327 | 37 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-444 | 36 | injection | A06:2025 - Insecure Design |
| CWE-330 | 33 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-129 | 32 | numeric | A05:2025 - Injection |
| CWE-601 | 25 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-326 | 25 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-116 | 25 | injection | A05:2025 - Injection |
| CWE-384 | 25 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-552 | 23 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-436 | 22 | input_validation | A06:2025 - Insecure Design |
| CWE-306 | 21 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-346 | 21 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-347 | 20 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-776 | 19 | resource_management | A02:2025 - Security Misconfiguration |
| CWE-312 | 19 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-276 | 17 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-290 | 15 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-917 | 13 | injection | A05:2025 - Injection |
| CWE-522 | 12 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-532 | 11 | logging_and_alerting_failures | A09:2025 - Logging & Alerting Failures |
| CWE-338 | 11 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-426 | 10 | code_quality | A08:2025 - Software or Data Integrity Failures |
| CWE-319 | 9 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-1021 | 9 | software_or_data_integrity_failures | A06:2025 - Insecure Design |
| CWE-613 | 9 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-260 | 8 | UNKNOWN | A02:2025 - Security Misconfiguration |
| CWE-285 | 7 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-331 | 6 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-470 | 6 | input_validation | A05:2025 - Injection |
| CWE-281 | 6 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-248 | 5 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-337 | 5 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-90 | 5 | injection | A05:2025 - Injection |
| CWE-639 | 5 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-91 | 5 | injection | A05:2025 - Injection |
| CWE-916 | 5 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-693 | 4 | deprecated | A06:2025 - Insecure Design |
| CWE-427 | 4 | code_quality | A08:2025 - Software or Data Integrity Failures |
| CWE-88 | 4 | injection | A05:2025 - Injection |
| CWE-209 | 4 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-610 | 4 | broken_access_control | A05:2025 - Injection |
| CWE-798 | 4 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-494 | 3 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-307 | 3 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-325 | 3 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-377 | 3 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-113 | 2 | injection | A05:2025 - Injection |
| CWE-324 | 2 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-335 | 2 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-749 | 2 | UNKNOWN | A01:2025 - Broken Access Control |
| CWE-379 | 2 | UNKNOWN | A01:2025 - Broken Access Control |
| CWE-829 | 2 | injection | A08:2025 - Software or Data Integrity Failures |
| CWE-628 | 1 | code_quality | A06:2025 - Insecure Design |
| CWE-460 | 1 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-73 | 1 | broken_access_control | A06:2025 - Insecure Design |
| CWE-93 | 1 | injection | A05:2025 - Injection |
| CWE-117 | 1 | logging_and_alerting_failures | A09:2025 - Logging & Alerting Failures |
| CWE-282 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-323 | 1 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-300 | 1 | broken_access_control | A07:2025 - Authentication Failures |
| CWE-501 | 1 | UNKNOWN | A06:2025 - Insecure Design |
| CWE-359 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-538 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-521 | 1 | authentication_failures | A07:2025 - Authentication Failures |
| **Total** | **7,139** | | |

> **Unique CWEs**: 94
> **Unique Groups**: 9

#### OWASP Top 10 Group Distribution

| OWASP Category | Count |
|---|---|
| A05:2025 - Injection | 2,147 |
| A01:2025 - Broken Access Control | 1,631 |
| A10:2025 - Mishandling of Exceptional Conditions | 1,568 |
| A06:2025 - Insecure Design | 793 |
| A07:2025 - Authentication Failures | 397 |
| A02:2025 - Security Misconfiguration | 256 |
| A08:2025 - Software or Data Integrity Failures | 176 |
| A04:2025 - Cryptographic Failures | 159 |
| A09:2025 - Logging & Alerting Failures | 12 |
| **Total** | **7,139** |

---

## 6. TitanVul (`data/datasets/titanvul/train.parquet`)

Total: **77,096** | Benign: **38,548** | Vulnerable: **38,548**

> Aggregated from 7 public vulnerability datasets (BigVul, D2A, CVEfixes, Devign, ReVeal, DiverseVul, MegaVul),
> deduplicated and validated with a multi-agent LLM framework.
> Contains 38,548 multilingual pairs (unfiltered).
> Balanced 1:1 (func_after = benign). Has `func_before` + `func_after` for diff-based flaw lines.

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 38,548 |
| -1 | UNKNOWN | 20,546 |
| 1 | memory_safety | 6,218 |
| 6 | broken_access_control | 2,500 |
| 5 | injection | 2,252 |
| 4 | input_validation | 1,696 |
| 3 | resource_management | 1,279 |
| 2 | numeric | 1,163 |
| 8 | cryptographic_failures | 543 |
| 14 | mishandling_exceptional_conditions | 519 |
| 7 | authentication_failures | 512 |
| 10 | code_quality | 445 |
| 9 | concurrency | 412 |
| 15 | deprecated | 164 |
| 11 | security_misconfiguration | 160 |
| 12 | software_or_data_integrity_failures | 128 |
| 13 | logging_and_alerting_failures | 11 |

> **Unique Groups**: 17

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group | Top 25 | OWASP Top 10 |
|---|---|---|---|---|---|
| *(empty/unknown)* | 20,546 | -1 | UNKNOWN |  |  |
| CWE-20 | 1,488 | 4 | input_validation | ✅ | ✅ |
| CWE-119 | 1,483 | 1 | memory_safety |  |  |
| CWE-125 | 1,395 | 1 | memory_safety | ✅ |  |
| CWE-787 | 1,266 | 1 | memory_safety | ✅ |  |
| CWE-79 | 871 | 5 | injection | ✅ | ✅ |
| CWE-416 | 723 | 1 | memory_safety | ✅ |  |
| CWE-200 | 705 | 6 | broken_access_control | ✅ | ✅ |
| CWE-476 | 692 | 1 | memory_safety | ✅ | ✅ |
| CWE-89 | 560 | 5 | injection | ✅ | ✅ |
| CWE-190 | 534 | 2 | numeric |  |  |
| CWE-399 | 440 | 3 | resource_management |  |  |
| CWE-264 | 424 | 6 | broken_access_control |  |  |
| CWE-703 | 397 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-189 | 372 | 2 | numeric |  |  |
| CWE-362 | 368 | 9 | concurrency |  | ✅ |
| CWE-120 | 317 | 1 | memory_safety | ✅ |  |
| CWE-94 | 306 | 5 | injection | ✅ | ✅ |
| CWE-22 | 302 | 6 | broken_access_control | ✅ | ✅ |
| CWE-310 | 252 | 8 | cryptographic_failures |  |  |
| CWE-400 | 244 | 3 | resource_management |  |  |
| CWE-78 | 238 | 5 | injection | ✅ | ✅ |
| CWE-401 | 187 | 3 | resource_management |  |  |
| CWE-415 | 168 | 1 | memory_safety |  |  |
| CWE-284 | 154 | 6 | broken_access_control | ✅ | ✅ |
| CWE-369 | 147 | 2 | numeric |  | ✅ |
| CWE-1321 | 141 | 4 | input_validation |  |  |
| CWE-617 | 138 | 10 | code_quality |  |  |
| CWE-352 | 136 | 6 | broken_access_control | ✅ | ✅ |
| CWE-835 | 133 | 10 | code_quality |  |  |
| CWE-918 | 128 | 6 | broken_access_control | ✅ | ✅ |
| CWE-772 | 120 | 3 | resource_management |  |  |
| CWE-863 | 119 | 6 | broken_access_control | ✅ | ✅ |
| CWE-287 | 113 | 7 | authentication_failures |  | ✅ |
| CWE-59 | 112 | 6 | broken_access_control |  | ✅ |
| CWE-269 | 111 | 6 | broken_access_control |  | ✅ |
| CWE-384 | 103 | 7 | authentication_failures |  | ✅ |
| CWE-502 | 98 | 12 | software_or_data_integrity_failures | ✅ | ✅ |
| CWE-295 | 88 | 7 | authentication_failures |  | ✅ |
| CWE-613 | 87 | 7 | authentication_failures |  | ✅ |
| CWE-601 | 81 | 6 | broken_access_control |  | ✅ |
| CWE-770 | 74 | 3 | resource_management | ✅ |  |
| CWE-674 | 71 | 10 | code_quality |  |  |
| CWE-74 | 71 | 5 | injection |  | ✅ |
| CWE-327 | 63 | 8 | cryptographic_failures |  | ✅ |
| CWE-254 | 62 | 15 | deprecated |  |  |
| CWE-19 | 57 | 15 | deprecated |  |  |
| CWE-347 | 56 | 8 | cryptographic_failures |  | ✅ |
| CWE-732 | 52 | 11 | security_misconfiguration |  | ✅ |
| CWE-862 | 51 | 6 | broken_access_control | ✅ | ✅ |
| CWE-1333 | 46 | 3 | resource_management |  |  |
| CWE-134 | 45 | 1 | memory_safety |  |  |
| CWE-77 | 45 | 5 | injection | ✅ | ✅ |
| CWE-122 | 42 | 1 | memory_safety | ✅ |  |
| CWE-704 | 41 | 10 | code_quality |  |  |
| CWE-459 | 40 | 3 | resource_management |  |  |
| CWE-17 | 37 | 15 | deprecated |  |  |
| CWE-755 | 36 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-763 | 36 | 1 | memory_safety |  |  |
| CWE-345 | 36 | 11 | security_misconfiguration |  | ✅ |
| CWE-276 | 35 | 11 | security_misconfiguration |  | ✅ |
| CWE-434 | 35 | 5 | injection | ✅ | ✅ |
| CWE-330 | 35 | 8 | cryptographic_failures |  | ✅ |
| CWE-444 | 35 | 5 | injection |  | ✅ |
| CWE-908 | 33 | 3 | resource_management |  |  |
| CWE-319 | 32 | 8 | cryptographic_failures |  | ✅ |
| CWE-668 | 31 | 6 | broken_access_control |  | ✅ |
| CWE-611 | 31 | 11 | security_misconfiguration |  | ✅ |
| CWE-326 | 31 | 8 | cryptographic_failures |  | ✅ |
| CWE-191 | 30 | 2 | numeric |  |  |
| CWE-404 | 29 | 3 | resource_management |  |  |
| CWE-252 | 29 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-285 | 29 | 6 | broken_access_control |  | ✅ |
| CWE-665 | 28 | 3 | resource_management |  |  |
| CWE-354 | 28 | 4 | input_validation |  |  |
| CWE-639 | 27 | 6 | broken_access_control | ✅ | ✅ |
| CWE-754 | 27 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-88 | 25 | 5 | injection |  | ✅ |
| CWE-640 | 24 | 7 | authentication_failures |  | ✅ |
| CWE-203 | 23 | 6 | broken_access_control |  |  |
| CWE-843 | 23 | 1 | memory_safety |  |  |
| CWE-681 | 22 | 2 | numeric |  |  |
| CWE-320 | 21 | 8 | cryptographic_failures |  |  |
| CWE-290 | 21 | 7 | authentication_failures |  | ✅ |
| CWE-834 | 20 | 10 | code_quality |  |  |
| CWE-116 | 20 | 5 | injection |  | ✅ |
| CWE-346 | 19 | 7 | authentication_failures |  | ✅ |
| CWE-667 | 17 | 9 | concurrency |  |  |
| CWE-377 | 15 | 6 | broken_access_control |  | ✅ |
| CWE-212 | 15 | 6 | broken_access_control |  |  |
| CWE-1103 | 15 | 10 | code_quality |  |  |
| CWE-193 | 14 | 2 | numeric |  |  |
| CWE-209 | 14 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-662 | 14 | 9 | concurrency |  |  |
| CWE-824 | 13 | 1 | memory_safety |  |  |
| CWE-682 | 13 | 2 | numeric |  |  |
| CWE-913 | 13 | 12 | software_or_data_integrity_failures |  |  |
| CWE-406 | 13 | 3 | resource_management |  |  |
| CWE-388 | 13 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-129 | 13 | 2 | numeric |  | ✅ |
| CWE-367 | 12 | 9 | concurrency |  |  |
| CWE-1021 | 11 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-909 | 11 | 3 | resource_management |  |  |
| CWE-532 | 11 | 13 | logging_and_alerting_failures |  | ✅ |
| CWE-798 | 10 | 7 | authentication_failures |  | ✅ |
| CWE-91 | 10 | 5 | injection |  | ✅ |
| CWE-1236 | 10 | 5 | injection |  |  |
| CWE-331 | 10 | 8 | cryptographic_failures |  | ✅ |
| CWE-241 | 9 | 4 | input_validation |  |  |
| CWE-521 | 8 | 7 | authentication_failures |  | ✅ |
| CWE-358 | 8 | 10 | code_quality |  |  |
| CWE-255 | 8 | 7 | authentication_failures |  |  |
| CWE-307 | 8 | 7 | authentication_failures |  | ✅ |
| CWE-131 | 8 | 2 | numeric |  |  |
| CWE-697 | 8 | 2 | numeric |  |  |
| CWE-281 | 8 | 6 | broken_access_control |  | ✅ |
| CWE-18 | 8 | 15 | deprecated |  |  |
| CWE-917 | 7 | 5 | injection |  | ✅ |
| CWE-323 | 7 | 8 | cryptographic_failures |  | ✅ |
| CWE-470 | 7 | 4 | input_validation |  | ✅ |
| CWE-1284 | 7 | 4 | input_validation |  |  |
| CWE-916 | 7 | 8 | cryptographic_failures |  | ✅ |
| CWE-311 | 7 | 8 | cryptographic_failures |  | ✅ |
| CWE-672 | 7 | 1 | memory_safety |  |  |
| CWE-522 | 7 | 8 | cryptographic_failures |  | ✅ |
| CWE-288 | 6 | 7 | authentication_failures |  | ✅ |
| CWE-294 | 6 | 7 | authentication_failures |  | ✅ |
| CWE-494 | 6 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-338 | 6 | 8 | cryptographic_failures |  | ✅ |
| CWE-93 | 6 | 5 | injection |  | ✅ |
| CWE-16 | 5 | 11 | security_misconfiguration |  |  |
| CWE-426 | 5 | 10 | code_quality |  | ✅ |
| CWE-427 | 5 | 10 | code_quality |  | ✅ |
| CWE-306 | 5 | 7 | authentication_failures | ✅ | ✅ |
| CWE-670 | 5 | 10 | code_quality |  |  |
| CWE-915 | 5 | 4 | input_validation |  | ✅ |
| CWE-21 | 5 | 6 | broken_access_control |  |  |
| CWE-121 | 4 | 1 | memory_safety | ✅ |  |
| CWE-436 | 4 | 4 | input_validation |  | ✅ |
| CWE-150 | 4 | 5 | injection |  |  |
| CWE-776 | 3 | 3 | resource_management |  | ✅ |
| CWE-273 | 3 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-552 | 3 | 6 | broken_access_control |  | ✅ |
| CWE-823 | 3 | 1 | memory_safety |  |  |
| CWE-359 | 3 | 6 | broken_access_control |  | ✅ |
| CWE-829 | 3 | 5 | injection |  | ✅ |
| CWE-312 | 3 | 8 | cryptographic_failures |  | ✅ |
| CWE-407 | 3 | 3 | resource_management |  |  |
| CWE-303 | 3 | 7 | authentication_failures |  | ✅ |
| CWE-90 | 3 | 5 | injection |  | ✅ |
| CWE-565 | 3 | 7 | authentication_failures |  | ✅ |
| CWE-61 | 3 | 6 | broken_access_control |  | ✅ |
| CWE-1187 | 3 | 3 | resource_management |  |  |
| CWE-774 | 2 | 3 | resource_management |  |  |
| CWE-73 | 2 | 6 | broken_access_control |  | ✅ |
| CWE-417 | 2 | 3 | resource_management |  |  |
| CWE-676 | 2 | 10 | code_quality |  | ✅ |
| CWE-335 | 2 | 8 | cryptographic_failures |  | ✅ |
| CWE-275 | 2 | 6 | broken_access_control |  |  |
| CWE-178 | 2 | 4 | input_validation |  |  |
| CWE-924 | 2 | 6 | broken_access_control |  |  |
| CWE-185 | 2 | 2 | numeric |  |  |
| CWE-706 | 2 | 6 | broken_access_control |  |  |
| CWE-184 | 2 | 4 | input_validation |  |  |
| CWE-297 | 2 | 6 | broken_access_control |  | ✅ |
| CWE-172 | 2 | 4 | input_validation |  |  |
| CWE-361 | 1 | 9 | concurrency |  |  |
| CWE-922 | 1 | 8 | cryptographic_failures |  | ✅ |
| CWE-707 | 1 | 4 | input_validation |  |  |
| CWE-266 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-378 | 1 | 6 | broken_access_control |  |  |
| CWE-214 | 1 | 6 | broken_access_control |  |  |
| CWE-943 | 1 | 5 | injection |  |  |
| CWE-80 | 1 | 5 | injection |  | ✅ |
| CWE-538 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-113 | 1 | 5 | injection |  | ✅ |
| CWE-684 | 1 | 10 | code_quality |  |  |
| CWE-425 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-1188 | 1 | 11 | security_misconfiguration |  |  |
| CWE-321 | 1 | 8 | cryptographic_failures |  | ✅ |
| CWE-664 | 1 | 3 | resource_management |  |  |
| CWE-526 | 1 | 8 | cryptographic_failures |  | ✅ |
| CWE-349 | 1 | 8 | cryptographic_failures |  |  |
| CWE-126 | 1 | 1 | memory_safety |  |  |
| CWE-428 | 1 | 10 | code_quality |  |  |

> **Unique CWEs**: 184
> **Unique Groups**: 15

### Top 25 Most Dangerous CWEs

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-20 | 1,488 | 4 | input_validation |
| CWE-125 | 1,395 | 1 | memory_safety |
| CWE-787 | 1,266 | 1 | memory_safety |
| CWE-79 | 871 | 5 | injection |
| CWE-416 | 723 | 1 | memory_safety |
| CWE-200 | 705 | 6 | broken_access_control |
| CWE-476 | 692 | 1 | memory_safety |
| CWE-89 | 560 | 5 | injection |
| CWE-120 | 317 | 1 | memory_safety |
| CWE-94 | 306 | 5 | injection |
| CWE-22 | 302 | 6 | broken_access_control |
| CWE-78 | 238 | 5 | injection |
| CWE-284 | 154 | 6 | broken_access_control |
| CWE-352 | 136 | 6 | broken_access_control |
| CWE-918 | 128 | 6 | broken_access_control |
| CWE-863 | 119 | 6 | broken_access_control |
| CWE-502 | 98 | 12 | software_or_data_integrity_failures |
| CWE-770 | 74 | 3 | resource_management |
| CWE-862 | 51 | 6 | broken_access_control |
| CWE-77 | 45 | 5 | injection |
| CWE-122 | 42 | 1 | memory_safety |
| CWE-434 | 35 | 5 | injection |
| CWE-639 | 27 | 6 | broken_access_control |
| CWE-306 | 5 | 7 | authentication_failures |
| CWE-121 | 4 | 1 | memory_safety |
| **Total** | **9,781** | | |

> **Unique CWEs**: 25
> **Unique Groups**: 7

#### Top 25 Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 1 | memory_safety | 4,439 |
| 5 | injection | 2,055 |
| 6 | broken_access_control | 1,622 |
| 4 | input_validation | 1,488 |
| 12 | software_or_data_integrity_failures | 98 |
| 3 | resource_management | 74 |
| 7 | authentication_failures | 5 |
| **Total** | **9,781** |

### OWASP Top 10 (2025)

| CWE | Count | Group | OWASP |
|---|---|---|---|
| CWE-20 | 1,488 | input_validation | A05:2025 - Injection |
| CWE-79 | 871 | injection | A05:2025 - Injection |
| CWE-200 | 705 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-476 | 692 | memory_safety | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-89 | 560 | injection | A05:2025 - Injection |
| CWE-703 | 397 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-362 | 368 | concurrency | A06:2025 - Insecure Design |
| CWE-94 | 306 | injection | A05:2025 - Injection |
| CWE-22 | 302 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-78 | 238 | injection | A05:2025 - Injection |
| CWE-284 | 154 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-369 | 147 | numeric | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-352 | 136 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-918 | 128 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-863 | 119 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-287 | 113 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-59 | 112 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-269 | 111 | broken_access_control | A06:2025 - Insecure Design |
| CWE-384 | 103 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-502 | 98 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-295 | 88 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-613 | 87 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-601 | 81 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-74 | 71 | injection | A05:2025 - Injection |
| CWE-327 | 63 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-347 | 56 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-732 | 52 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-862 | 51 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-77 | 45 | injection | A05:2025 - Injection |
| CWE-755 | 36 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-345 | 36 | security_misconfiguration | A08:2025 - Software or Data Integrity Failures |
| CWE-276 | 35 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-434 | 35 | injection | A06:2025 - Insecure Design |
| CWE-330 | 35 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-444 | 35 | injection | A06:2025 - Insecure Design |
| CWE-319 | 32 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-668 | 31 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-611 | 31 | security_misconfiguration | A02:2025 - Security Misconfiguration |
| CWE-326 | 31 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-252 | 29 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-285 | 29 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-639 | 27 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-754 | 27 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-88 | 25 | injection | A05:2025 - Injection |
| CWE-640 | 24 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-290 | 21 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-116 | 20 | injection | A05:2025 - Injection |
| CWE-346 | 19 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-377 | 15 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-209 | 14 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-129 | 13 | numeric | A05:2025 - Injection |
| CWE-1021 | 11 | software_or_data_integrity_failures | A06:2025 - Insecure Design |
| CWE-532 | 11 | logging_and_alerting_failures | A09:2025 - Logging & Alerting Failures |
| CWE-798 | 10 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-91 | 10 | injection | A05:2025 - Injection |
| CWE-331 | 10 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-521 | 8 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-307 | 8 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-281 | 8 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-917 | 7 | injection | A05:2025 - Injection |
| CWE-323 | 7 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-470 | 7 | input_validation | A05:2025 - Injection |
| CWE-916 | 7 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-311 | 7 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-522 | 7 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-288 | 6 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-294 | 6 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-494 | 6 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-338 | 6 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-93 | 6 | injection | A05:2025 - Injection |
| CWE-426 | 5 | code_quality | A08:2025 - Software or Data Integrity Failures |
| CWE-427 | 5 | code_quality | A08:2025 - Software or Data Integrity Failures |
| CWE-306 | 5 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-915 | 5 | input_validation | A08:2025 - Software or Data Integrity Failures |
| CWE-436 | 4 | input_validation | A06:2025 - Insecure Design |
| CWE-776 | 3 | resource_management | A02:2025 - Security Misconfiguration |
| CWE-552 | 3 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-359 | 3 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-829 | 3 | injection | A08:2025 - Software or Data Integrity Failures |
| CWE-312 | 3 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-303 | 3 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-90 | 3 | injection | A05:2025 - Injection |
| CWE-565 | 3 | authentication_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-61 | 3 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-73 | 2 | broken_access_control | A06:2025 - Insecure Design |
| CWE-676 | 2 | code_quality | A06:2025 - Insecure Design |
| CWE-335 | 2 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-297 | 2 | broken_access_control | A07:2025 - Authentication Failures |
| CWE-922 | 1 | cryptographic_failures | A01:2025 - Broken Access Control |
| CWE-266 | 1 | broken_access_control | A06:2025 - Insecure Design |
| CWE-80 | 1 | injection | A05:2025 - Injection |
| CWE-538 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-113 | 1 | injection | A05:2025 - Injection |
| CWE-425 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-321 | 1 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-526 | 1 | cryptographic_failures | A02:2025 - Security Misconfiguration |
| **Total** | **8,557** | | |

> **Unique CWEs**: 96
> **Unique Groups**: 9

#### OWASP Top 10 Group Distribution

| OWASP Category | Count |
|---|---|
| A05:2025 - Injection | 3,672 |
| A01:2025 - Broken Access Control | 1,997 |
| A10:2025 - Mishandling of Exceptional Conditions | 1,342 |
| A06:2025 - Insecure Design | 586 |
| A07:2025 - Authentication Failures | 503 |
| A04:2025 - Cryptographic Failures | 250 |
| A08:2025 - Software or Data Integrity Failures | 161 |
| A02:2025 - Security Misconfiguration | 35 |
| A09:2025 - Logging & Alerting Failures | 11 |
| **Total** | **8,557** |

---

## 7. BenchVul (`data/datasets/benchvul/train.parquet`)

# Benchmark for Top 25 Most Dangerous CWEs

Total: **2,100** | Benign: **1,050** | Vulnerable: **1,050**

> Manually verified benchmark designed for **evaluating** vulnerability detection models.
> Covers a refined set of the Top 25 Most Dangerous CWEs (MITRE 2024).
> 50 vulnerable + 50 fixed samples per CWE (before C/C++ filter).
> Labels achieve 92% correctness rate per manual review.
> Contains 1,050 multilingual pairs (unfiltered).
> **Intended for evaluation/testing only — not suitable for training.**

### Group Distribution

| Group ID | Group | Count | OWASP Top 10 |
|---|---|---|---|
| 0 | benign | 1,050 |  |
| 6 | broken_access_control | 350 | A01, A06 |
| 5 | injection | 250 | A05, A06 |
| 1 | memory_safety | 200 | A10 |
| 7 | authentication_failures | 100 | A07 |
| 12 | software_or_data_integrity_failures | 50 | A08 |
| 2 | numeric | 50 |  |
| 3 | resource_management | 50 |  |

> **Unique Groups**: 8

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group | Top 25 | OWASP Top 10 |
|---|---|---|---|---|---|
| CWE-79 | 50 | 5 | injection | ✅ | ✅ |
| CWE-89 | 50 | 5 | injection | ✅ | ✅ |
| CWE-22 | 50 | 6 | broken_access_control | ✅ | ✅ |
| CWE-78 | 50 | 5 | injection | ✅ | ✅ |
| CWE-502 | 50 | 12 | software_or_data_integrity_failures | ✅ | ✅ |
| CWE-94 | 50 | 5 | injection | ✅ | ✅ |
| CWE-863 | 50 | 6 | broken_access_control | ✅ | ✅ |
| CWE-352 | 50 | 6 | broken_access_control | ✅ | ✅ |
| CWE-787 | 50 | 1 | memory_safety | ✅ |  |
| CWE-306 | 50 | 7 | authentication_failures | ✅ | ✅ |
| CWE-416 | 50 | 1 | memory_safety | ✅ |  |
| CWE-190 | 50 | 2 | numeric |  |  |
| CWE-476 | 50 | 1 | memory_safety | ✅ | ✅ |
| CWE-269 | 50 | 6 | broken_access_control |  | ✅ |
| CWE-798 | 50 | 7 | authentication_failures |  | ✅ |
| CWE-400 | 50 | 3 | resource_management |  |  |
| CWE-125 | 50 | 1 | memory_safety | ✅ |  |
| CWE-862 | 50 | 6 | broken_access_control | ✅ | ✅ |
| CWE-434 | 50 | 5 | injection | ✅ | ✅ |
| CWE-918 | 50 | 6 | broken_access_control | ✅ | ✅ |
| CWE-200 | 50 | 6 | broken_access_control | ✅ | ✅ |

> **Unique CWEs**: 21
> **Unique Groups**: 7

### Top 25 Most Dangerous CWEs

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-79 | 50 | 5 | injection |
| CWE-89 | 50 | 5 | injection |
| CWE-22 | 50 | 6 | broken_access_control |
| CWE-78 | 50 | 5 | injection |
| CWE-502 | 50 | 12 | software_or_data_integrity_failures |
| CWE-94 | 50 | 5 | injection |
| CWE-863 | 50 | 6 | broken_access_control |
| CWE-352 | 50 | 6 | broken_access_control |
| CWE-787 | 50 | 1 | memory_safety |
| CWE-306 | 50 | 7 | authentication_failures |
| CWE-416 | 50 | 1 | memory_safety |
| CWE-476 | 50 | 1 | memory_safety |
| CWE-125 | 50 | 1 | memory_safety |
| CWE-862 | 50 | 6 | broken_access_control |
| CWE-434 | 50 | 5 | injection |
| CWE-918 | 50 | 6 | broken_access_control |
| CWE-200 | 50 | 6 | broken_access_control |
| **Total** | **850** | | |

> **Unique CWEs**: 17
> **Unique Groups**: 5

#### Top 25 Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 6 | broken_access_control | 300 |
| 5 | injection | 250 |
| 1 | memory_safety | 200 |
| 12 | software_or_data_integrity_failures | 50 |
| 7 | authentication_failures | 50 |
| **Total** | **850** |

### OWASP Top 10 (2025)

| CWE | Count | Group | OWASP |
|---|---|---|---|
| CWE-79 | 50 | injection | A05:2025 - Injection |
| CWE-89 | 50 | injection | A05:2025 - Injection |
| CWE-22 | 50 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-78 | 50 | injection | A05:2025 - Injection |
| CWE-502 | 50 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-94 | 50 | injection | A05:2025 - Injection |
| CWE-863 | 50 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-352 | 50 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-306 | 50 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-476 | 50 | memory_safety | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-269 | 50 | broken_access_control | A06:2025 - Insecure Design |
| CWE-798 | 50 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-862 | 50 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-434 | 50 | injection | A06:2025 - Insecure Design |
| CWE-918 | 50 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-200 | 50 | broken_access_control | A01:2025 - Broken Access Control |
| **Total** | **800** | | |

> **Unique CWEs**: 16
> **Unique Groups**: 6

#### OWASP Top 10 Group Distribution

| OWASP Category | Count |
|---|---|
| A01:2025 - Broken Access Control | 300 |
| A05:2025 - Injection | 200 |
| A07:2025 - Authentication Failures | 100 |
| A06:2025 - Insecure Design | 100 |
| A08:2025 - Software or Data Integrity Failures | 50 |
| A10:2025 - Mishandling of Exceptional Conditions | 50 |
| **Total** | **800** |

---

## 8. Devign (`data/datasets/devign/{train,validation,test}.parquet`)

Total: **27,318** | Benign: **14,858** | Vulnerable: **12,460**

> No CWE column. Binary only (`target` column). Cannot be used for multiclass/group mode.
> Flaw lines available via `vul_lines` column (Devign-format dict).

---

## 9. Merged (`data/datasets/merged/train.parquet`)

Total: **176,674** | Benign: **154,205** | Vulnerable: **22,469**

> Combined BigVul + MegaVul. Has `CWE ID`, `func_before`, `func_after`.

### Group Distribution

| Group ID | Group | Count | OWASP Top 10 |
|---|---|---|---|
| 0 | benign | 154,205 |  |
| 1 | memory_safety | 8,962 | A10 |
| -1 | UNKNOWN | 2,759 |  |
| 6 | broken_access_control | 2,285 | A01, A05, A06, A07 |
| 3 | resource_management | 2,100 | A02 |
| 4 | input_validation | 1,778 | A05, A06 |
| 2 | numeric | 1,606 | A05, A10 |
| 9 | concurrency | 824 | A06 |
| 10 | code_quality | 703 | A06, A08 |
| 15 | deprecated | 339 | A06 |
| 8 | cryptographic_failures | 313 | A04, A06 |
| 5 | injection | 310 | A05, A06 |
| 7 | authentication_failures | 173 | A07 |
| 11 | security_misconfiguration | 149 | A01, A02, A08 |
| 14 | mishandling_exceptional_conditions | 134 | A10 |
| 12 | software_or_data_integrity_failures | 27 | A06, A08 |
| 13 | logging_and_alerting_failures | 7 | A09 |

> **Unique Groups**: 17

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group | Top 25 | OWASP Top 10 |
|---|---|---|---|---|---|
| CWE-119 | 3,000 | 1 | memory_safety |  |  |
| *(empty/unknown)* | 2,746 | -1 | UNKNOWN |  |  |
| CWE-20 | 1,688 | 4 | input_validation | ✅ | ✅ |
| CWE-125 | 1,556 | 1 | memory_safety | ✅ |  |
| CWE-787 | 1,261 | 1 | memory_safety | ✅ |  |
| CWE-476 | 1,189 | 1 | memory_safety | ✅ | ✅ |
| CWE-416 | 1,102 | 1 | memory_safety | ✅ |  |
| CWE-399 | 923 | 3 | resource_management |  |  |
| CWE-200 | 840 | 6 | broken_access_control | ✅ | ✅ |
| CWE-190 | 819 | 2 | numeric |  |  |
| CWE-362 | 708 | 9 | concurrency |  | ✅ |
| CWE-264 | 685 | 6 | broken_access_control |  |  |
| CWE-189 | 460 | 2 | numeric |  |  |
| CWE-120 | 380 | 1 | memory_safety | ✅ |  |
| CWE-400 | 285 | 3 | resource_management |  |  |
| CWE-401 | 260 | 3 | resource_management |  |  |
| CWE-835 | 244 | 10 | code_quality |  |  |
| CWE-415 | 226 | 1 | memory_safety |  |  |
| CWE-772 | 193 | 3 | resource_management |  |  |
| CWE-284 | 186 | 6 | broken_access_control | ✅ | ✅ |
| CWE-617 | 161 | 10 | code_quality |  |  |
| CWE-310 | 143 | 8 | cryptographic_failures |  |  |
| CWE-254 | 135 | 15 | deprecated |  |  |
| CWE-369 | 130 | 2 | numeric |  | ✅ |
| CWE-22 | 126 | 6 | broken_access_control | ✅ | ✅ |
| CWE-59 | 121 | 6 | broken_access_control |  | ✅ |
| CWE-122 | 110 | 1 | memory_safety | ✅ |  |
| CWE-674 | 105 | 10 | code_quality |  |  |
| CWE-404 | 104 | 3 | resource_management |  |  |
| CWE-269 | 102 | 6 | broken_access_control |  | ✅ |
| CWE-834 | 100 | 10 | code_quality |  |  |
| CWE-287 | 99 | 7 | authentication_failures |  | ✅ |
| CWE-732 | 91 | 11 | security_misconfiguration |  | ✅ |
| CWE-17 | 85 | 15 | deprecated |  |  |
| CWE-19 | 84 | 15 | deprecated |  |  |
| CWE-908 | 78 | 3 | resource_management |  |  |
| CWE-770 | 71 | 3 | resource_management | ✅ |  |
| CWE-667 | 68 | 9 | concurrency |  |  |
| CWE-79 | 66 | 5 | injection | ✅ | ✅ |
| CWE-78 | 63 | 5 | injection | ✅ | ✅ |
| CWE-909 | 55 | 3 | resource_management |  |  |
| CWE-193 | 54 | 2 | numeric |  |  |
| CWE-295 | 52 | 7 | authentication_failures |  | ✅ |
| CWE-74 | 50 | 5 | injection |  | ✅ |
| CWE-704 | 48 | 10 | code_quality |  |  |
| CWE-459 | 48 | 3 | resource_management |  |  |
| CWE-665 | 43 | 3 | resource_management |  |  |
| CWE-754 | 42 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-863 | 42 | 6 | broken_access_control | ✅ | ✅ |
| CWE-89 | 40 | 5 | injection | ✅ | ✅ |
| CWE-252 | 40 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-354 | 37 | 4 | input_validation |  |  |
| CWE-682 | 35 | 2 | numeric |  |  |
| CWE-862 | 34 | 6 | broken_access_control | ✅ | ✅ |
| CWE-668 | 34 | 6 | broken_access_control |  | ✅ |
| CWE-330 | 33 | 8 | cryptographic_failures |  | ✅ |
| CWE-843 | 33 | 1 | memory_safety |  |  |
| CWE-367 | 32 | 9 | concurrency |  |  |
| CWE-681 | 32 | 2 | numeric |  |  |
| CWE-77 | 31 | 5 | injection | ✅ | ✅ |
| CWE-18 | 31 | 15 | deprecated |  |  |
| CWE-327 | 31 | 8 | cryptographic_failures |  | ✅ |
| CWE-203 | 30 | 6 | broken_access_control |  |  |
| CWE-755 | 29 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-134 | 28 | 1 | memory_safety |  |  |
| CWE-285 | 25 | 6 | broken_access_control |  | ✅ |
| CWE-129 | 25 | 2 | numeric |  | ✅ |
| CWE-121 | 25 | 1 | memory_safety | ✅ |  |
| CWE-611 | 22 | 11 | security_misconfiguration |  | ✅ |
| CWE-191 | 22 | 2 | numeric |  |  |
| CWE-436 | 22 | 4 | input_validation |  | ✅ |
| CWE-347 | 21 | 8 | cryptographic_failures |  | ✅ |
| CWE-1284 | 21 | 4 | input_validation |  |  |
| CWE-326 | 20 | 8 | cryptographic_failures |  | ✅ |
| CWE-345 | 19 | 11 | security_misconfiguration |  | ✅ |
| CWE-824 | 19 | 1 | memory_safety |  |  |
| CWE-502 | 18 | 12 | software_or_data_integrity_failures | ✅ | ✅ |
| CWE-763 | 18 | 1 | memory_safety |  |  |
| CWE-776 | 18 | 3 | resource_management |  | ✅ |
| CWE-444 | 18 | 5 | injection |  | ✅ |
| CWE-311 | 16 | 8 | cryptographic_failures |  | ✅ |
| CWE-662 | 16 | 9 | concurrency |  |  |
| CWE-358 | 15 | 10 | code_quality |  |  |
| CWE-601 | 15 | 6 | broken_access_control |  | ✅ |
| CWE-94 | 15 | 5 | injection | ✅ | ✅ |
| CWE-388 | 14 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-697 | 13 | 2 | numeric |  |  |
| CWE-426 | 11 | 10 | code_quality |  | ✅ |
| CWE-276 | 11 | 11 | security_misconfiguration |  | ✅ |
| CWE-212 | 11 | 6 | broken_access_control |  |  |
| CWE-116 | 11 | 5 | injection |  | ✅ |
| CWE-131 | 10 | 2 | numeric |  |  |
| CWE-346 | 9 | 7 | authentication_failures |  | ✅ |
| CWE-255 | 8 | 7 | authentication_failures |  |  |
| CWE-320 | 8 | 8 | cryptographic_failures |  |  |
| CWE-552 | 8 | 6 | broken_access_control |  | ✅ |
| CWE-331 | 8 | 8 | cryptographic_failures |  | ✅ |
| CWE-319 | 7 | 8 | cryptographic_failures |  | ✅ |
| CWE-440 | 7 | 10 | code_quality |  |  |
| CWE-911 | 7 | 3 | resource_management |  |  |
| CWE-385 | 7 | 8 | cryptographic_failures |  |  |
| CWE-281 | 6 | 6 | broken_access_control |  | ✅ |
| CWE-522 | 6 | 8 | cryptographic_failures |  | ✅ |
| CWE-532 | 6 | 13 | logging_and_alerting_failures |  | ✅ |
| CWE-494 | 6 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-90 | 6 | 5 | injection |  | ✅ |
| CWE-428 | 6 | 10 | code_quality |  |  |
| CWE-407 | 6 | 3 | resource_management |  |  |
| CWE-918 | 5 | 6 | broken_access_control | ✅ | ✅ |
| CWE-823 | 5 | 1 | memory_safety |  |  |
| CWE-337 | 5 | 8 | cryptographic_failures |  | ✅ |
| CWE-606 | 5 | -1 | UNKNOWN |  |  |
| CWE-693 | 4 | 15 | deprecated |  | ✅ |
| CWE-126 | 4 | 1 | memory_safety |  |  |
| CWE-273 | 4 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-1077 | 4 | 2 | numeric |  |  |
| CWE-670 | 4 | 10 | code_quality |  |  |
| CWE-241 | 4 | 4 | input_validation |  |  |
| CWE-229 | 4 | -1 | UNKNOWN |  |  |
| CWE-706 | 4 | 6 | broken_access_control |  |  |
| CWE-1021 | 3 | 12 | software_or_data_integrity_failures |  | ✅ |
| CWE-172 | 3 | 4 | input_validation |  |  |
| CWE-16 | 3 | 11 | security_misconfiguration |  |  |
| CWE-1188 | 3 | 11 | security_misconfiguration |  |  |
| CWE-672 | 3 | 1 | memory_safety |  |  |
| CWE-118 | 3 | 1 | memory_safety |  |  |
| CWE-434 | 3 | 5 | injection | ✅ | ✅ |
| CWE-707 | 3 | 4 | input_validation |  |  |
| CWE-325 | 3 | 8 | cryptographic_failures |  | ✅ |
| CWE-209 | 2 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-113 | 2 | 5 | injection |  | ✅ |
| CWE-762 | 2 | -1 | UNKNOWN |  |  |
| CWE-1333 | 2 | 3 | resource_management |  |  |
| CWE-306 | 2 | 7 | authentication_failures | ✅ | ✅ |
| CWE-924 | 2 | 6 | broken_access_control |  |  |
| CWE-88 | 2 | 5 | injection |  | ✅ |
| CWE-639 | 2 | 6 | broken_access_control | ✅ | ✅ |
| CWE-307 | 2 | 7 | authentication_failures |  | ✅ |
| CWE-838 | 2 | -1 | UNKNOWN |  |  |
| CWE-1325 | 2 | 3 | resource_management |  |  |
| CWE-757 | 2 | 8 | cryptographic_failures |  | ✅ |
| CWE-99 | 2 | 5 | injection |  | ✅ |
| CWE-664 | 1 | 3 | resource_management |  |  |
| CWE-352 | 1 | 6 | broken_access_control | ✅ | ✅ |
| CWE-185 | 1 | 2 | numeric |  |  |
| CWE-1049 | 1 | 3 | resource_management |  |  |
| CWE-628 | 1 | 10 | code_quality |  | ✅ |
| CWE-202 | 1 | 6 | broken_access_control |  |  |
| CWE-248 | 1 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-626 | 1 | 10 | code_quality |  |  |
| CWE-460 | 1 | 14 | mishandling_exceptional_conditions |  | ✅ |
| CWE-290 | 1 | 7 | authentication_failures |  | ✅ |
| CWE-73 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-324 | 1 | 8 | cryptographic_failures |  | ✅ |
| CWE-93 | 1 | 5 | injection |  | ✅ |
| CWE-117 | 1 | 13 | logging_and_alerting_failures |  | ✅ |
| CWE-1050 | 1 | 3 | resource_management |  |  |
| CWE-282 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-610 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-338 | 1 | 8 | cryptographic_failures |  | ✅ |
| CWE-300 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-349 | 1 | 8 | cryptographic_failures |  |  |
| CWE-457 | 1 | 3 | resource_management |  |  |
| CWE-392 | 1 | 14 | mishandling_exceptional_conditions |  |  |
| CWE-114 | 1 | 6 | broken_access_control |  | ✅ |
| CWE-680 | 1 | 2 | numeric |  |  |
| CWE-789 | 1 | 3 | resource_management |  |  |

> **Unique CWEs**: 166
> **Unique Groups**: 16

### Top 25 Most Dangerous CWEs

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-20 | 1,688 | 4 | input_validation |
| CWE-125 | 1,556 | 1 | memory_safety |
| CWE-787 | 1,261 | 1 | memory_safety |
| CWE-476 | 1,189 | 1 | memory_safety |
| CWE-416 | 1,102 | 1 | memory_safety |
| CWE-200 | 840 | 6 | broken_access_control |
| CWE-120 | 380 | 1 | memory_safety |
| CWE-284 | 186 | 6 | broken_access_control |
| CWE-22 | 126 | 6 | broken_access_control |
| CWE-122 | 110 | 1 | memory_safety |
| CWE-770 | 71 | 3 | resource_management |
| CWE-79 | 66 | 5 | injection |
| CWE-78 | 63 | 5 | injection |
| CWE-863 | 42 | 6 | broken_access_control |
| CWE-89 | 40 | 5 | injection |
| CWE-862 | 34 | 6 | broken_access_control |
| CWE-77 | 31 | 5 | injection |
| CWE-121 | 25 | 1 | memory_safety |
| CWE-502 | 18 | 12 | software_or_data_integrity_failures |
| CWE-94 | 15 | 5 | injection |
| CWE-918 | 5 | 6 | broken_access_control |
| CWE-434 | 3 | 5 | injection |
| CWE-306 | 2 | 7 | authentication_failures |
| CWE-639 | 2 | 6 | broken_access_control |
| CWE-352 | 1 | 6 | broken_access_control |
| **Total** | **8,856** | | |

> **Unique CWEs**: 25
> **Unique Groups**: 7

#### Top 25 Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 1 | memory_safety | 5,623 |
| 4 | input_validation | 1,688 |
| 6 | broken_access_control | 1,236 |
| 5 | injection | 218 |
| 3 | resource_management | 71 |
| 12 | software_or_data_integrity_failures | 18 |
| 7 | authentication_failures | 2 |
| **Total** | **8,856** |

### OWASP Top 10 (2025)

| CWE | Count | Group | OWASP |
|---|---|---|---|
| CWE-20 | 1,688 | input_validation | A05:2025 - Injection |
| CWE-476 | 1,189 | memory_safety | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-200 | 840 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-362 | 708 | concurrency | A06:2025 - Insecure Design |
| CWE-284 | 186 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-369 | 130 | numeric | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-22 | 126 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-59 | 121 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-269 | 102 | broken_access_control | A06:2025 - Insecure Design |
| CWE-287 | 99 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-732 | 91 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-79 | 66 | injection | A05:2025 - Injection |
| CWE-78 | 63 | injection | A05:2025 - Injection |
| CWE-295 | 52 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-74 | 50 | injection | A05:2025 - Injection |
| CWE-754 | 42 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-863 | 42 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-89 | 40 | injection | A05:2025 - Injection |
| CWE-252 | 40 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-862 | 34 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-668 | 34 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-330 | 33 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-77 | 31 | injection | A05:2025 - Injection |
| CWE-327 | 31 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-755 | 29 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-285 | 25 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-129 | 25 | numeric | A05:2025 - Injection |
| CWE-611 | 22 | security_misconfiguration | A02:2025 - Security Misconfiguration |
| CWE-436 | 22 | input_validation | A06:2025 - Insecure Design |
| CWE-347 | 21 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-326 | 20 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-345 | 19 | security_misconfiguration | A08:2025 - Software or Data Integrity Failures |
| CWE-502 | 18 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-776 | 18 | resource_management | A02:2025 - Security Misconfiguration |
| CWE-444 | 18 | injection | A06:2025 - Insecure Design |
| CWE-311 | 16 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-601 | 15 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-94 | 15 | injection | A05:2025 - Injection |
| CWE-426 | 11 | code_quality | A08:2025 - Software or Data Integrity Failures |
| CWE-276 | 11 | security_misconfiguration | A01:2025 - Broken Access Control |
| CWE-116 | 11 | injection | A05:2025 - Injection |
| CWE-346 | 9 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-552 | 8 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-331 | 8 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-319 | 7 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-281 | 6 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-522 | 6 | cryptographic_failures | A06:2025 - Insecure Design |
| CWE-532 | 6 | logging_and_alerting_failures | A09:2025 - Logging & Alerting Failures |
| CWE-494 | 6 | software_or_data_integrity_failures | A08:2025 - Software or Data Integrity Failures |
| CWE-90 | 6 | injection | A05:2025 - Injection |
| CWE-918 | 5 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-337 | 5 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-693 | 4 | deprecated | A06:2025 - Insecure Design |
| CWE-1021 | 3 | software_or_data_integrity_failures | A06:2025 - Insecure Design |
| CWE-434 | 3 | injection | A06:2025 - Insecure Design |
| CWE-325 | 3 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-209 | 2 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-113 | 2 | injection | A05:2025 - Injection |
| CWE-306 | 2 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-88 | 2 | injection | A05:2025 - Injection |
| CWE-639 | 2 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-307 | 2 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-757 | 2 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-99 | 2 | injection | A05:2025 - Injection |
| CWE-352 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-628 | 1 | code_quality | A06:2025 - Insecure Design |
| CWE-248 | 1 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-460 | 1 | mishandling_exceptional_conditions | A10:2025 - Mishandling of Exceptional Conditions |
| CWE-290 | 1 | authentication_failures | A07:2025 - Authentication Failures |
| CWE-73 | 1 | broken_access_control | A06:2025 - Insecure Design |
| CWE-324 | 1 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-93 | 1 | injection | A05:2025 - Injection |
| CWE-117 | 1 | logging_and_alerting_failures | A09:2025 - Logging & Alerting Failures |
| CWE-282 | 1 | broken_access_control | A01:2025 - Broken Access Control |
| CWE-610 | 1 | broken_access_control | A05:2025 - Injection |
| CWE-338 | 1 | cryptographic_failures | A04:2025 - Cryptographic Failures |
| CWE-300 | 1 | broken_access_control | A07:2025 - Authentication Failures |
| CWE-114 | 1 | broken_access_control | A05:2025 - Injection |
| **Total** | **6,269** | | |

> **Unique CWEs**: 78
> **Unique Groups**: 9

#### OWASP Top 10 Group Distribution

| OWASP Category | Count |
|---|---|
| A05:2025 - Injection | 2,004 |
| A01:2025 - Broken Access Control | 1,548 |
| A10:2025 - Mishandling of Exceptional Conditions | 1,434 |
| A06:2025 - Insecure Design | 884 |
| A07:2025 - Authentication Failures | 166 |
| A04:2025 - Cryptographic Failures | 132 |
| A08:2025 - Software or Data Integrity Failures | 54 |
| A02:2025 - Security Misconfiguration | 40 |
| A09:2025 - Logging & Alerting Failures | 7 |
| **Total** | **6,269** |

---

## Cross-Dataset Group Coverage (vulnerable only)

| Group ID | Group | BigVul | DiverseVul | MegaVul | TitanVul | BenchVul | Merged |
|---|---|---|---|---|---|---|---|
| 1 | memory_safety | 3618 | 7065 | 7659 | 6218 | 200 | 8962 |
| -1 | UNKNOWN | 2135 | 2836 | 1768 | 20546 | 0 | 2759 |
| 6 | broken_access_control | 1341 | 1816 | 2249 | 2500 | 350 | 2285 |
| 3 | resource_management | 916 | 1383 | 1883 | 1279 | 50 | 2100 |
| 4 | input_validation | 1152 | 1400 | 1317 | 1696 | 0 | 1778 |
| 2 | numeric | 703 | 1187 | 1398 | 1163 | 50 | 1606 |
| 5 | injection | 101 | 395 | 1001 | 2252 | 250 | 310 |
| 9 | concurrency | 266 | 450 | 686 | 412 | 0 | 824 |
| 10 | code_quality | 121 | 393 | 814 | 445 | 0 | 703 |
| 8 | cryptographic_failures | 130 | 547 | 300 | 543 | 0 | 313 |
| 14 | mishandling_exceptional_conditions | 24 | 869 | 144 | 519 | 0 | 134 |
| 7 | authentication_failures | 36 | 304 | 401 | 512 | 100 | 173 |
| 15 | deprecated | 271 | 179 | 283 | 164 | 0 | 339 |
| 11 | security_misconfiguration | 75 | 96 | 361 | 160 | 0 | 149 |
| 12 | software_or_data_integrity_failures | 5 | 18 | 132 | 128 | 50 | 27 |
| 13 | logging_and_alerting_failures | 1 | 7 | 12 | 11 | 0 | 7 |
