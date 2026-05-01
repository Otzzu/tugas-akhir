# Dataset Analysis — Complete CWE and Group Distribution

Generated from raw parquet files. Group mapping via `CWE_GROUP_MAP` in `dataset_lm.py`.

**Fixed Group IDs:**
0=benign · 1=memory_safety · 2=numeric · 3=resource_management · 4=input_validation · 5=injection · 6=access_control · 7=authentication · 8=cryptography · 9=concurrency · 10=code_quality · 11=configuration · 12=data_integrity · 13=logging · 14=error_handling · 15=deprecated · -1=UNKNOWN (not in CWE_GROUP_MAP)

---

## Summary

| Dataset | Total | Benign | Vulnerable | Has CWE | Has Flaw Lines | Notes |
|---|---|---|---|---|---|---|
| BigVul | 217,007 | 206,112 | 10,895 | Yes | Yes (diff) | Primary training dataset |
| DiverseVul | 330,492 | 311,547 | 18,945 | Yes (multi-label) | No | Binary only |
| MegaVul | 55,868 | 27,934 | 27,934 | Yes | Yes (diff) | Balanced 1:1 |
| Devign | 27,318 | 14,858 | 12,460 | No | Yes (vul_lines) | Binary only |
| Merged (BigVul+MegaVul) | 176,674 | 154,205 | 22,469 | Yes | Yes (diff) | Combined |
| TitanVul | 15,300\* | 7,650 | 7,650 | Yes | Yes (diff) | Balanced 1:1, C/C++ filtered |
| BenchVul | 460\* | 230 | 230 | Yes | Yes (diff) | **Benchmark for Top 25 Most Dangerous CWEs** |

\*After C/C++ language filter (from 38,548 and 1,050 original pairs respectively).

---

## 1. BigVul (`data/datasets/bigvul/all.parquet`)

Total: **217,007** | Benign: **206,112** | Vulnerable: **10,895**

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 206,112 |
| 1 | memory_safety | 3,618 |
| -1 | UNKNOWN | 2,135 |
| 6 | access_control | 1,341 |
| 4 | input_validation | 1,153 |
| 3 | resource_management | 916 |
| 2 | numeric | 703 |
| 15 | deprecated | 271 |
| 9 | concurrency | 266 |
| 10 | code_quality | 131 |
| 8 | cryptography | 130 |
| 5 | injection | 101 |
| 11 | configuration | 75 |
| 7 | authentication | 36 |
| 14 | error_handling | 13 |
| 12 | data_integrity | 5 |
| 13 | logging | 1 |

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-119 | 2,139 | 1 | memory_safety |
| *(empty/unknown)* | 2,135 | -1 | UNKNOWN |
| CWE-20 | 1,136 | 4 | input_validation |
| CWE-399 | 742 | 3 | resource_management |
| CWE-125 | 630 | 1 | memory_safety |
| CWE-200 | 508 | 6 | access_control |
| CWE-264 | 493 | 6 | access_control |
| CWE-189 | 341 | 2 | numeric |
| CWE-416 | 323 | 1 | memory_safety |
| CWE-190 | 311 | 2 | numeric |
| CWE-362 | 266 | 9 | concurrency |
| CWE-476 | 225 | 1 | memory_safety |
| CWE-787 | 191 | 1 | memory_safety |
| CWE-284 | 169 | 6 | access_control |
| CWE-254 | 125 | 15 | deprecated |
| CWE-310 | 88 | 8 | cryptography |
| CWE-415 | 78 | 1 | memory_safety |
| CWE-732 | 65 | 11 | configuration |
| CWE-404 | 64 | 3 | resource_management |
| CWE-19 | 60 | 15 | deprecated |
| CWE-59 | 55 | 6 | access_control |
| CWE-17 | 53 | 15 | deprecated |
| CWE-79 | 53 | 5 | injection |
| CWE-772 | 52 | 3 | resource_management |
| CWE-400 | 43 | 3 | resource_management |
| CWE-22 | 40 | 6 | access_control |
| CWE-835 | 37 | 10 | code_quality |
| CWE-18 | 32 | 15 | deprecated |
| CWE-269 | 31 | 6 | access_control |
| CWE-369 | 31 | 2 | numeric |
| CWE-704 | 25 | 10 | code_quality |
| CWE-285 | 24 | 6 | access_control |
| CWE-287 | 22 | 7 | authentication |
| CWE-134 | 19 | 1 | memory_safety |
| CWE-617 | 18 | 10 | code_quality |
| CWE-311 | 18 | 8 | cryptography |
| CWE-358 | 17 | 10 | code_quality |
| CWE-77 | 13 | 5 | injection |
| CWE-78 | 13 | 5 | injection |
| CWE-682 | 12 | 2 | numeric |
| CWE-834 | 11 | 10 | code_quality |
| CWE-674 | 10 | 10 | code_quality |
| CWE-754 | 10 | 14 | error_handling |
| CWE-94 | 10 | 5 | injection |
| CWE-388 | 10 | 10 | code_quality |
| CWE-120 | 9 | 1 | memory_safety |
| CWE-281 | 9 | 6 | access_control |
| CWE-770 | 9 | 3 | resource_management |
| CWE-354 | 9 | 4 | input_validation |
| CWE-347 | 8 | 8 | cryptography |
| CWE-320 | 8 | 8 | cryptography |
| CWE-611 | 7 | 11 | configuration |
| CWE-255 | 6 | 7 | authentication |
| CWE-89 | 5 | 5 | injection |
| CWE-74 | 5 | 5 | injection |
| CWE-862 | 5 | 6 | access_control |
| CWE-665 | 5 | 3 | resource_management |
| CWE-346 | 5 | 7 | authentication |
| CWE-129 | 4 | 2 | numeric |
| CWE-191 | 4 | 2 | numeric |
| CWE-436 | 4 | 4 | input_validation |
| CWE-601 | 3 | 6 | access_control |
| CWE-522 | 3 | 8 | cryptography |
| CWE-426 | 3 | 10 | code_quality |
| CWE-172 | 3 | 4 | input_validation |
| CWE-327 | 3 | 8 | cryptography |
| CWE-494 | 3 | 12 | data_integrity |
| CWE-295 | 3 | 7 | authentication |
| CWE-345 | 2 | 11 | configuration |
| CWE-755 | 2 | 14 | error_handling |
| CWE-824 | 2 | 1 | memory_safety |
| CWE-918 | 2 | 6 | access_control |
| CWE-763 | 2 | 1 | memory_safety |
| CWE-90 | 2 | 5 | injection |
| CWE-330 | 2 | 8 | cryptography |
| CWE-502 | 1 | 12 | data_integrity |
| CWE-1021 | 1 | 12 | data_integrity |
| CWE-693 | 1 | 15 | deprecated |
| CWE-532 | 1 | 13 | logging |
| CWE-668 | 1 | 6 | access_control |
| CWE-664 | 1 | 3 | resource_management |
| CWE-352 | 1 | 6 | access_control |
| CWE-252 | 1 | 4 | input_validation |
| CWE-16 | 1 | 11 | configuration |
| CWE-209 | 1 | 14 | error_handling |

### CPG Files vs all.parquet (data/raw/bigvul/)

| Group ID | Group | CPG Files | all.parquet | Coverage |
|---|---|---|---|---|
| 0 | benign | 4,000 | 206,112 | subsampled |
| -1 | UNKNOWN | 8,760 | 2,135 | 410% |
| -1 | UNKNOWN | 8,760 | 2,135 | filtered |

> All vulnerable CPG files = 100% coverage of all.parquet vulnerable.
> Benign subsampled to ~4,000.

---

## 2. DiverseVul (`data/datasets/diversevul/all.parquet`)

Total: **330,492** | Benign: **311,547** | Vulnerable: **18,945**

> CWE column is multi-label array (e.g. `['CWE-787', 'CWE-119']`).
> Group assignment uses primary (first) CWE only.
> No flaw line ground truth available.

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 311,547 |
| 1 | memory_safety | 7,065 |
| -1 | UNKNOWN | 2,836 |
| 6 | access_control | 1,839 |
| 4 | input_validation | 1,424 |
| 3 | resource_management | 1,383 |
| 2 | numeric | 1,187 |
| 14 | error_handling | 835 |
| 8 | cryptography | 547 |
| 9 | concurrency | 450 |
| 10 | code_quality | 403 |
| 5 | injection | 395 |
| 7 | authentication | 281 |
| 15 | deprecated | 179 |
| 11 | configuration | 96 |
| 12 | data_integrity | 18 |
| 13 | logging | 7 |

### CWE Distribution (all vulnerable, primary CWE)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| *(empty/unknown)* | 2,836 | -1 | UNKNOWN |
| CWE-125 | 1,635 | 1 | memory_safety |
| CWE-119 | 1,433 | 1 | memory_safety |
| CWE-787 | 1,379 | 1 | memory_safety |
| CWE-20 | 1,315 | 4 | input_validation |
| CWE-416 | 999 | 1 | memory_safety |
| CWE-476 | 915 | 1 | memory_safety |
| CWE-703 | 735 | 14 | error_handling |
| CWE-200 | 717 | 6 | access_control |
| CWE-190 | 674 | 2 | numeric |
| CWE-399 | 462 | 3 | resource_management |
| CWE-362 | 398 | 9 | concurrency |
| CWE-400 | 395 | 3 | resource_management |
| CWE-310 | 343 | 8 | cryptography |
| CWE-284 | 286 | 6 | access_control |
| CWE-120 | 285 | 1 | memory_safety |
| CWE-189 | 273 | 2 | numeric |
| CWE-415 | 247 | 1 | memory_safety |
| CWE-264 | 217 | 6 | access_control |
| CWE-401 | 193 | 3 | resource_management |
| CWE-59 | 164 | 6 | access_control |
| CWE-617 | 162 | 10 | code_quality |
| CWE-369 | 158 | 2 | numeric |
| CWE-22 | 140 | 6 | access_control |
| CWE-269 | 111 | 6 | access_control |
| CWE-835 | 111 | 10 | code_quality |
| CWE-287 | 105 | 7 | authentication |
| CWE-295 | 101 | 7 | authentication |
| CWE-772 | 100 | 3 | resource_management |
| CWE-770 | 89 | 3 | resource_management |
| CWE-94 | 84 | 5 | injection |
| CWE-78 | 76 | 5 | injection |
| CWE-122 | 68 | 1 | memory_safety |
| CWE-19 | 66 | 15 | deprecated |
| CWE-444 | 60 | 5 | injection |
| CWE-241 | 57 | 4 | input_validation |
| CWE-674 | 56 | 10 | code_quality |
| CWE-89 | 54 | 5 | injection |
| CWE-17 | 46 | 15 | deprecated |
| CWE-755 | 46 | 14 | error_handling |
| CWE-191 | 45 | 2 | numeric |
| CWE-459 | 43 | 3 | resource_management |
| CWE-862 | 42 | 6 | access_control |
| CWE-613 | 40 | 7 | authentication |
| CWE-330 | 39 | 8 | cryptography |
| CWE-732 | 38 | 11 | configuration |
| CWE-327 | 38 | 8 | cryptography |
| CWE-134 | 38 | 1 | memory_safety |
| CWE-319 | 35 | 8 | cryptography |
| CWE-601 | 35 | 6 | access_control |
| CWE-18 | 34 | 15 | deprecated |
| CWE-665 | 32 | 3 | resource_management |
| CWE-77 | 31 | 5 | injection |
| CWE-254 | 31 | 15 | deprecated |
| CWE-754 | 30 | 14 | error_handling |
| CWE-704 | 28 | 10 | code_quality |
| CWE-79 | 27 | 5 | injection |
| CWE-74 | 25 | 5 | injection |
| CWE-326 | 25 | 8 | cryptography |
| CWE-347 | 25 | 8 | cryptography |
| CWE-203 | 24 | 6 | access_control |
| CWE-252 | 24 | 4 | input_validation |
| CWE-290 | 23 | 6 | access_control |
| CWE-662 | 22 | 9 | concurrency |
| CWE-354 | 21 | 4 | input_validation |
| CWE-667 | 21 | 9 | concurrency |
| CWE-908 | 21 | 3 | resource_management |
| CWE-345 | 20 | 11 | configuration |
| CWE-358 | 20 | 10 | code_quality |
| CWE-276 | 19 | 11 | configuration |
| CWE-406 | 19 | 3 | resource_management |
| CWE-209 | 18 | 14 | error_handling |
| CWE-863 | 18 | 6 | access_control |
| CWE-763 | 17 | 1 | memory_safety |
| CWE-434 | 15 | 5 | injection |
| CWE-843 | 14 | 1 | memory_safety |
| CWE-611 | 14 | 11 | configuration |
| CWE-288 | 13 | 7 | authentication |
| CWE-323 | 13 | 8 | cryptography |
| CWE-668 | 12 | 6 | access_control |
| CWE-824 | 12 | 1 | memory_safety |
| CWE-522 | 11 | 8 | cryptography |
| CWE-193 | 11 | 2 | numeric |
| CWE-131 | 11 | 2 | numeric |
| CWE-552 | 10 | 6 | access_control |
| CWE-913 | 10 | 12 | data_integrity |
| CWE-909 | 10 | 3 | resource_management |
| CWE-388 | 10 | 10 | code_quality |
| CWE-116 | 9 | 5 | injection |
| CWE-834 | 9 | 10 | code_quality |
| CWE-367 | 9 | 9 | concurrency |
| CWE-320 | 9 | 8 | cryptography |
| CWE-404 | 8 | 3 | resource_management |
| CWE-126 | 7 | 1 | memory_safety |
| CWE-121 | 7 | 1 | memory_safety |
| CWE-212 | 7 | 6 | access_control |
| CWE-532 | 7 | 13 | logging |
| CWE-294 | 7 | 7 | authentication |
| CWE-682 | 6 | 2 | numeric |
| CWE-297 | 6 | 6 | access_control |
| CWE-918 | 6 | 6 | access_control |
| CWE-273 | 6 | 14 | error_handling |
| CWE-707 | 6 | 4 | input_validation |
| CWE-281 | 6 | 6 | access_control |
| CWE-1187 | 6 | 3 | resource_management |
| CWE-346 | 6 | 7 | authentication |
| CWE-16 | 5 | 11 | configuration |
| CWE-681 | 5 | 2 | numeric |
| CWE-502 | 4 | 12 | data_integrity |
| CWE-1021 | 4 | 12 | data_integrity |
| CWE-823 | 4 | 1 | memory_safety |
| CWE-61 | 4 | 6 | access_control |
| CWE-303 | 4 | 7 | authentication |
| CWE-88 | 4 | 5 | injection |
| CWE-91 | 4 | 5 | injection |
| CWE-311 | 4 | 8 | cryptography |
| CWE-331 | 4 | 8 | cryptography |
| CWE-352 | 4 | 6 | access_control |
| CWE-129 | 3 | 2 | numeric |
| CWE-426 | 3 | 10 | code_quality |
| CWE-672 | 3 | 1 | memory_safety |
| CWE-670 | 3 | 10 | code_quality |
| CWE-113 | 3 | 5 | injection |
| CWE-417 | 2 | 3 | resource_management |
| CWE-776 | 2 | 3 | resource_management |
| CWE-924 | 2 | 6 | access_control |
| CWE-307 | 2 | 7 | authentication |
| CWE-693 | 2 | 15 | deprecated |
| CWE-266 | 2 | 6 | access_control |
| CWE-93 | 2 | 5 | injection |
| CWE-285 | 2 | 6 | access_control |
| CWE-697 | 1 | 2 | numeric |
| CWE-565 | 1 | 7 | authentication |
| CWE-943 | 1 | 5 | injection |
| CWE-255 | 1 | 7 | authentication |
| CWE-349 | 1 | 8 | cryptography |
| CWE-457 | 1 | 3 | resource_management |
| CWE-805 | 1 | 1 | memory_safety |
| CWE-436 | 1 | 4 | input_validation |
| CWE-798 | 1 | 7 | authentication |
| CWE-786 | 1 | 1 | memory_safety |
| CWE-428 | 1 | 10 | code_quality |
| CWE-706 | 1 | 6 | access_control |

---

## 3. MegaVul (`data/datasets/megavul/train.parquet`)

Total: **55,868** | Benign: **27,934** | Vulnerable: **27,934**

> Perfectly balanced (1:1 ratio). Has `func_before` + `func_after` for diff-based flaw lines.

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 27,934 |
| 1 | memory_safety | 12,214 |
| 3 | resource_management | 2,784 |
| 6 | access_control | 2,424 |
| -1 | UNKNOWN | 2,143 |
| 4 | input_validation | 2,020 |
| 2 | numeric | 2,012 |
| 10 | code_quality | 1,214 |
| 9 | concurrency | 1,195 |
| 7 | authentication | 462 |
| 5 | injection | 437 |
| 8 | cryptography | 386 |
| 15 | deprecated | 258 |
| 11 | configuration | 192 |
| 14 | error_handling | 132 |
| 12 | data_integrity | 50 |
| 13 | logging | 11 |

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-119 | 2,710 | 1 | memory_safety |
| CWE-787 | 2,226 | 1 | memory_safety |
| *(empty/unknown)* | 2,125 | -1 | UNKNOWN |
| CWE-125 | 2,072 | 1 | memory_safety |
| CWE-476 | 2,026 | 1 | memory_safety |
| CWE-20 | 1,778 | 4 | input_validation |
| CWE-416 | 1,663 | 1 | memory_safety |
| CWE-190 | 1,065 | 2 | numeric |
| CWE-362 | 968 | 9 | concurrency |
| CWE-200 | 889 | 6 | access_control |
| CWE-120 | 737 | 1 | memory_safety |
| CWE-399 | 702 | 3 | resource_management |
| CWE-264 | 583 | 6 | access_control |
| CWE-400 | 519 | 3 | resource_management |
| CWE-401 | 517 | 3 | resource_management |
| CWE-835 | 424 | 10 | code_quality |
| CWE-189 | 399 | 2 | numeric |
| CWE-287 | 327 | 7 | authentication |
| CWE-415 | 315 | 1 | memory_safety |
| CWE-772 | 307 | 3 | resource_management |
| CWE-617 | 297 | 10 | code_quality |
| CWE-122 | 207 | 1 | memory_safety |
| CWE-22 | 193 | 6 | access_control |
| CWE-369 | 188 | 2 | numeric |
| CWE-674 | 181 | 10 | code_quality |
| CWE-834 | 175 | 10 | code_quality |
| CWE-908 | 161 | 3 | resource_management |
| CWE-269 | 151 | 6 | access_control |
| CWE-59 | 148 | 6 | access_control |
| CWE-310 | 141 | 8 | cryptography |
| CWE-770 | 138 | 3 | resource_management |
| CWE-667 | 135 | 9 | concurrency |
| CWE-404 | 119 | 3 | resource_management |
| CWE-295 | 117 | 7 | authentication |
| CWE-193 | 107 | 2 | numeric |
| CWE-909 | 104 | 3 | resource_management |
| CWE-78 | 102 | 5 | injection |
| CWE-284 | 96 | 6 | access_control |
| CWE-17 | 92 | 15 | deprecated |
| CWE-863 | 87 | 6 | access_control |
| CWE-732 | 84 | 11 | configuration |
| CWE-665 | 79 | 3 | resource_management |
| CWE-19 | 78 | 15 | deprecated |
| CWE-74 | 78 | 5 | injection |
| CWE-252 | 77 | 4 | input_validation |
| CWE-79 | 71 | 5 | injection |
| CWE-254 | 68 | 15 | deprecated |
| CWE-459 | 66 | 3 | resource_management |
| CWE-89 | 65 | 5 | injection |
| CWE-367 | 64 | 9 | concurrency |
| CWE-681 | 64 | 2 | numeric |
| CWE-330 | 63 | 8 | cryptography |
| CWE-862 | 63 | 6 | access_control |
| CWE-843 | 62 | 1 | memory_safety |
| CWE-668 | 62 | 6 | access_control |
| CWE-754 | 62 | 14 | error_handling |
| CWE-354 | 60 | 4 | input_validation |
| CWE-704 | 59 | 10 | code_quality |
| CWE-327 | 59 | 8 | cryptography |
| CWE-755 | 56 | 14 | error_handling |
| CWE-134 | 53 | 1 | memory_safety |
| CWE-682 | 50 | 2 | numeric |
| CWE-203 | 49 | 6 | access_control |
| CWE-1284 | 46 | 4 | input_validation |
| CWE-121 | 44 | 1 | memory_safety |
| CWE-129 | 42 | 2 | numeric |
| CWE-191 | 41 | 2 | numeric |
| CWE-77 | 41 | 5 | injection |
| CWE-326 | 40 | 8 | cryptography |
| CWE-502 | 40 | 12 | data_integrity |
| CWE-611 | 38 | 11 | configuration |
| CWE-776 | 36 | 3 | resource_management |
| CWE-436 | 36 | 4 | input_validation |
| CWE-763 | 35 | 1 | memory_safety |
| CWE-345 | 32 | 11 | configuration |
| CWE-824 | 32 | 1 | memory_safety |
| CWE-276 | 28 | 11 | configuration |
| CWE-662 | 28 | 9 | concurrency |
| CWE-601 | 26 | 6 | access_control |
| CWE-697 | 26 | 2 | numeric |
| CWE-212 | 24 | 6 | access_control |
| CWE-444 | 22 | 5 | injection |
| CWE-347 | 21 | 8 | cryptography |
| CWE-131 | 20 | 2 | numeric |
| CWE-552 | 18 | 6 | access_control |
| CWE-116 | 18 | 5 | injection |
| CWE-388 | 16 | 10 | code_quality |
| CWE-426 | 16 | 10 | code_quality |
| CWE-440 | 16 | 10 | code_quality |
| CWE-94 | 16 | 5 | injection |
| CWE-911 | 14 | 3 | resource_management |
| CWE-18 | 14 | 15 | deprecated |
| CWE-319 | 13 | 8 | cryptography |
| CWE-428 | 12 | 10 | code_quality |
| CWE-331 | 12 | 8 | cryptography |
| CWE-346 | 12 | 7 | authentication |
| CWE-385 | 12 | 8 | cryptography |
| CWE-407 | 12 | 3 | resource_management |
| CWE-126 | 10 | 1 | memory_safety |
| CWE-823 | 10 | 1 | memory_safety |
| CWE-670 | 10 | 10 | code_quality |
| CWE-337 | 10 | 8 | cryptography |
| CWE-532 | 9 | 13 | logging |
| CWE-918 | 8 | 6 | access_control |
| CWE-273 | 8 | 14 | error_handling |
| CWE-1077 | 8 | 2 | numeric |
| CWE-90 | 8 | 5 | injection |
| CWE-241 | 8 | 4 | input_validation |
| CWE-1188 | 6 | 11 | configuration |
| CWE-494 | 6 | 12 | data_integrity |
| CWE-672 | 6 | 1 | memory_safety |
| CWE-1333 | 6 | 3 | resource_management |
| CWE-118 | 6 | 1 | memory_safety |
| CWE-434 | 6 | 5 | injection |
| CWE-693 | 6 | 15 | deprecated |
| CWE-172 | 6 | 4 | input_validation |
| CWE-184 | 6 | 4 | input_validation |
| CWE-229 | 6 | -1 | UNKNOWN |
| CWE-522 | 5 | 8 | cryptography |
| CWE-113 | 4 | 5 | injection |
| CWE-16 | 4 | 11 | configuration |
| CWE-255 | 4 | 7 | authentication |
| CWE-762 | 4 | -1 | UNKNOWN |
| CWE-306 | 4 | -1 | UNKNOWN |
| CWE-924 | 4 | 6 | access_control |
| CWE-324 | 4 | 8 | cryptography |
| CWE-1021 | 4 | 12 | data_integrity |
| CWE-285 | 4 | 6 | access_control |
| CWE-358 | 4 | 10 | code_quality |
| CWE-88 | 4 | 5 | injection |
| CWE-639 | 4 | 6 | access_control |
| CWE-320 | 3 | 8 | cryptography |
| CWE-706 | 3 | 6 | access_control |
| CWE-707 | 3 | 4 | input_validation |
| CWE-185 | 2 | 2 | numeric |
| CWE-1049 | 2 | 3 | resource_management |
| CWE-628 | 2 | 10 | code_quality |
| CWE-202 | 2 | 6 | access_control |
| CWE-248 | 2 | 14 | error_handling |
| CWE-626 | 2 | 10 | code_quality |
| CWE-460 | 2 | 14 | error_handling |
| CWE-290 | 2 | 6 | access_control |
| CWE-73 | 2 | 6 | access_control |
| CWE-93 | 2 | 5 | injection |
| CWE-117 | 2 | 13 | logging |
| CWE-209 | 2 | 14 | error_handling |
| CWE-1050 | 2 | 3 | resource_management |
| CWE-282 | 2 | 6 | access_control |
| CWE-610 | 2 | 6 | access_control |
| CWE-338 | 2 | 8 | cryptography |
| CWE-300 | 2 | 6 | access_control |
| CWE-307 | 2 | 7 | authentication |
| CWE-838 | 2 | -1 | UNKNOWN |
| CWE-325 | 2 | -1 | UNKNOWN |
| CWE-349 | 1 | 8 | cryptography |

---

## 6. TitanVul (`data/datasets/titanvul/train.parquet`)

Total: **15,300** | Benign: **7,650** | Vulnerable: **7,650**

> Aggregated from 7 public vulnerability datasets (BigVul, D2A, CVEfixes, Devign, ReVeal, DiverseVul, MegaVul),
> deduplicated and validated with a multi-agent LLM framework.
> Original 38,548 multilingual pairs filtered to **C/C++ only** (via `extension` field).
> Balanced 1:1 (func_after = benign). Has `func_before` + `func_after` for diff-based flaw lines.

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 7,650 |
| 1 | memory_safety | 2,593 |
| -1 | UNKNOWN | 1,353 |
| 6 | access_control | 750 |
| 3 | resource_management | 553 |
| 4 | input_validation | 540 |
| 2 | numeric | 488 |
| 8 | cryptography | 299 |
| 14 | error_handling | 275 |
| 9 | concurrency | 191 |
| 5 | injection | 186 |
| 10 | code_quality | 183 |
| 7 | authentication | 138 |
| 11 | configuration | 51 |
| 15 | deprecated | 36 |
| 12 | data_integrity | 12 |
| 13 | logging | 2 |

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| *(empty/unknown)* | 1,351 | -1 | UNKNOWN |
| CWE-787 | 684 | 1 | memory_safety |
| CWE-119 | 526 | 1 | memory_safety |
| CWE-20 | 503 | 4 | input_validation |
| CWE-125 | 434 | 1 | memory_safety |
| CWE-416 | 422 | 1 | memory_safety |
| CWE-200 | 325 | 6 | access_control |
| CWE-476 | 302 | 1 | memory_safety |
| CWE-703 | 245 | 14 | error_handling |
| CWE-190 | 235 | 2 | numeric |
| CWE-399 | 186 | 3 | resource_management |
| CWE-310 | 166 | 8 | cryptography |
| CWE-189 | 164 | 2 | numeric |
| CWE-362 | 162 | 9 | concurrency |
| CWE-264 | 128 | 6 | access_control |
| CWE-400 | 128 | 3 | resource_management |
| CWE-120 | 100 | 1 | memory_safety |
| CWE-401 | 90 | 3 | resource_management |
| CWE-415 | 79 | 1 | memory_safety |
| CWE-284 | 77 | 6 | access_control |
| CWE-835 | 59 | 10 | code_quality |
| CWE-369 | 57 | 2 | numeric |
| CWE-94 | 54 | 5 | injection |
| CWE-617 | 53 | 10 | code_quality |
| CWE-59 | 50 | 6 | access_control |
| CWE-269 | 49 | 6 | access_control |
| CWE-295 | 48 | 7 | authentication |
| CWE-287 | 44 | 7 | authentication |
| CWE-22 | 42 | 6 | access_control |
| CWE-770 | 40 | 3 | resource_management |
| CWE-459 | 35 | 3 | resource_management |
| CWE-89 | 35 | 5 | injection |
| CWE-613 | 34 | 7 | authentication |
| CWE-674 | 33 | 10 | code_quality |
| CWE-772 | 32 | 3 | resource_management |
| CWE-319 | 31 | 8 | cryptography |
| CWE-78 | 29 | 5 | injection |
| CWE-732 | 25 | 11 | configuration |
| CWE-327 | 23 | 8 | cryptography |
| CWE-330 | 22 | 8 | cryptography |
| CWE-191 | 21 | 2 | numeric |
| CWE-444 | 21 | 5 | injection |
| CWE-122 | 20 | 1 | memory_safety |
| CWE-77 | 20 | 5 | injection |
| CWE-704 | 19 | 10 | code_quality |
| CWE-290 | 17 | 6 | access_control |
| CWE-254 | 16 | 15 | deprecated |
| CWE-134 | 16 | 1 | memory_safety |
| CWE-862 | 15 | 6 | access_control |
| CWE-347 | 14 | 8 | cryptography |
| CWE-326 | 14 | 8 | cryptography |
| CWE-209 | 13 | 14 | error_handling |
| CWE-79 | 13 | 5 | injection |
| CWE-320 | 13 | 8 | cryptography |
| CWE-345 | 13 | 11 | configuration |
| CWE-406 | 13 | 3 | resource_management |
| CWE-252 | 13 | 4 | input_validation |
| CWE-354 | 13 | 4 | input_validation |
| CWE-908 | 12 | 3 | resource_management |
| CWE-203 | 12 | 6 | access_control |
| CWE-662 | 12 | 9 | concurrency |
| CWE-367 | 9 | 9 | concurrency |
| CWE-755 | 9 | 14 | error_handling |
| CWE-17 | 9 | 15 | deprecated |
| CWE-241 | 9 | 4 | input_validation |
| CWE-834 | 9 | 10 | code_quality |
| CWE-668 | 8 | 6 | access_control |
| CWE-667 | 8 | 9 | concurrency |
| CWE-754 | 8 | 14 | error_handling |
| CWE-276 | 8 | 11 | configuration |
| CWE-323 | 7 | 8 | cryptography |
| CWE-19 | 7 | 15 | deprecated |
| CWE-863 | 7 | 6 | access_control |
| CWE-665 | 7 | 3 | resource_management |
| CWE-74 | 6 | 5 | injection |
| CWE-288 | 5 | 7 | authentication |
| CWE-388 | 5 | 10 | code_quality |
| CWE-913 | 5 | 12 | data_integrity |
| CWE-311 | 5 | 8 | cryptography |
| CWE-601 | 4 | 6 | access_control |
| CWE-1021 | 4 | 12 | data_integrity |
| CWE-909 | 4 | 3 | resource_management |
| CWE-281 | 4 | 6 | access_control |
| CWE-18 | 4 | 15 | deprecated |
| CWE-502 | 3 | 12 | data_integrity |
| CWE-352 | 3 | 6 | access_control |
| CWE-434 | 3 | 5 | injection |
| CWE-16 | 3 | 11 | configuration |
| CWE-358 | 3 | 10 | code_quality |
| CWE-121 | 3 | 1 | memory_safety |
| CWE-763 | 3 | 1 | memory_safety |
| CWE-193 | 3 | 2 | numeric |
| CWE-285 | 3 | 6 | access_control |
| CWE-303 | 3 | 7 | authentication |
| CWE-611 | 2 | 11 | configuration |
| CWE-426 | 2 | 10 | code_quality |
| CWE-294 | 2 | 7 | authentication |
| CWE-417 | 2 | 3 | resource_management |
| CWE-131 | 2 | 2 | numeric |
| CWE-212 | 2 | 6 | access_control |
| CWE-824 | 2 | 1 | memory_safety |
| CWE-522 | 2 | 8 | cryptography |
| CWE-681 | 2 | 2 | numeric |
| CWE-532 | 2 | 13 | logging |
| CWE-297 | 2 | 6 | access_control |
| CWE-776 | 2 | 3 | resource_management |
| CWE-682 | 2 | 2 | numeric |
| CWE-404 | 1 | 3 | resource_management |
| CWE-116 | 1 | 5 | injection |
| CWE-129 | 1 | 2 | numeric |
| CWE-823 | 1 | 1 | memory_safety |
| CWE-697 | 1 | 2 | numeric |
| CWE-707 | 1 | 4 | input_validation |
| CWE-266 | 1 | 6 | access_control |
| CWE-331 | 1 | 8 | cryptography |
| CWE-93 | 1 | 5 | injection |
| NVD-CWE-Other | 1 | -1 | UNKNOWN |
| CWE-91 | 1 | 5 | injection |
| CWE-346 | 1 | 7 | authentication |
| CWE-552 | 1 | 6 | access_control |
| CWE-113 | 1 | 5 | injection |
| CWE-1187 | 1 | 3 | resource_management |
| CWE-172 | 1 | 4 | input_validation |
| CWE-349 | 1 | 8 | cryptography |
| CWE-126 | 1 | 1 | memory_safety |
| CWE-307 | 1 | 7 | authentication |
| CWE-275 | 1 | -1 | UNKNOWN |
| CWE-88 | 1 | 5 | injection |

---

## 7. BenchVul (`data/datasets/benchvul/train.parquet`)

# Benchmark for Top 25 Most Dangerous CWEs

Total: **460** | Benign: **230** | Vulnerable: **230**

> Manually verified benchmark designed for **evaluating** vulnerability detection models.
> Covers a refined set of the Top 25 Most Dangerous CWEs (MITRE 2024).
> 50 vulnerable + 50 fixed samples per CWE (before C/C++ filter).
> Labels achieve 92% correctness rate per manual review.
> Original 1,050 multilingual pairs filtered to **C/C++ only** (via `programming_language` field).
> **Intended for evaluation/testing only — not suitable for training.**

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 230 |
| 1 | memory_safety | 73 |
| 6 | access_control | 54 |
| 5 | injection | 26 |
| 3 | resource_management | 24 |
| 2 | numeric | 23 |
| 7 | authentication | 19 |
| -1 | UNKNOWN | 7 |
| 12 | data_integrity | 4 |

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-400 | 24 | 3 | resource_management |
| CWE-190 | 23 | 2 | numeric |
| CWE-416 | 21 | 1 | memory_safety |
| CWE-476 | 19 | 1 | memory_safety |
| CWE-798 | 19 | 7 | authentication |
| CWE-125 | 17 | 1 | memory_safety |
| CWE-787 | 16 | 1 | memory_safety |
| CWE-434 | 16 | 5 | injection |
| CWE-352 | 10 | 6 | access_control |
| CWE-862 | 10 | 6 | access_control |
| CWE-200 | 10 | 6 | access_control |
| CWE-94 | 8 | 5 | injection |
| CWE-269 | 8 | 6 | access_control |
| CWE-306 | 7 | -1 | UNKNOWN |
| CWE-863 | 6 | 6 | access_control |
| CWE-918 | 6 | 6 | access_control |
| CWE-22 | 4 | 6 | access_control |
| CWE-502 | 4 | 12 | data_integrity |
| CWE-79 | 2 | 5 | injection |

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

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 154,205 |
| 1 | memory_safety | 8,962 |
| -1 | UNKNOWN | 2,764 |
| 6 | access_control | 2,286 |
| 3 | resource_management | 2,100 |
| 4 | input_validation | 1,818 |
| 2 | numeric | 1,606 |
| 9 | concurrency | 824 |
| 10 | code_quality | 717 |
| 15 | deprecated | 339 |
| 5 | injection | 310 |
| 8 | cryptography | 310 |
| 7 | authentication | 170 |
| 11 | configuration | 149 |
| 14 | error_handling | 80 |
| 12 | data_integrity | 27 |
| 13 | logging | 7 |

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-119 | 3,000 | 1 | memory_safety |
| *(empty/unknown)* | 2,746 | -1 | UNKNOWN |
| CWE-20 | 1,688 | 4 | input_validation |
| CWE-125 | 1,556 | 1 | memory_safety |
| CWE-787 | 1,261 | 1 | memory_safety |
| CWE-476 | 1,189 | 1 | memory_safety |
| CWE-416 | 1,102 | 1 | memory_safety |
| CWE-399 | 923 | 3 | resource_management |
| CWE-200 | 840 | 6 | access_control |
| CWE-190 | 819 | 2 | numeric |
| CWE-362 | 708 | 9 | concurrency |
| CWE-264 | 685 | 6 | access_control |
| CWE-189 | 460 | 2 | numeric |
| CWE-120 | 380 | 1 | memory_safety |
| CWE-400 | 285 | 3 | resource_management |
| CWE-401 | 260 | 3 | resource_management |
| CWE-835 | 244 | 10 | code_quality |
| CWE-415 | 226 | 1 | memory_safety |
| CWE-772 | 193 | 3 | resource_management |
| CWE-284 | 186 | 6 | access_control |
| CWE-617 | 161 | 10 | code_quality |
| CWE-310 | 143 | 8 | cryptography |
| CWE-254 | 135 | 15 | deprecated |
| CWE-369 | 130 | 2 | numeric |
| CWE-22 | 126 | 6 | access_control |
| CWE-59 | 121 | 6 | access_control |
| CWE-122 | 110 | 1 | memory_safety |
| CWE-674 | 105 | 10 | code_quality |
| CWE-404 | 104 | 3 | resource_management |
| CWE-269 | 102 | 6 | access_control |
| CWE-834 | 100 | 10 | code_quality |
| CWE-287 | 99 | 7 | authentication |
| CWE-732 | 91 | 11 | configuration |
| CWE-17 | 85 | 15 | deprecated |
| CWE-19 | 84 | 15 | deprecated |
| CWE-908 | 78 | 3 | resource_management |
| CWE-770 | 71 | 3 | resource_management |
| CWE-667 | 68 | 9 | concurrency |
| CWE-79 | 66 | 5 | injection |
| CWE-78 | 63 | 5 | injection |
| CWE-909 | 55 | 3 | resource_management |
| CWE-193 | 54 | 2 | numeric |
| CWE-295 | 52 | 7 | authentication |
| CWE-74 | 50 | 5 | injection |
| CWE-704 | 48 | 10 | code_quality |
| CWE-459 | 48 | 3 | resource_management |
| CWE-665 | 43 | 3 | resource_management |
| CWE-754 | 42 | 14 | error_handling |
| CWE-863 | 42 | 6 | access_control |
| CWE-89 | 40 | 5 | injection |
| CWE-252 | 40 | 4 | input_validation |
| CWE-354 | 37 | 4 | input_validation |
| CWE-682 | 35 | 2 | numeric |
| CWE-862 | 34 | 6 | access_control |
| CWE-668 | 34 | 6 | access_control |
| CWE-330 | 33 | 8 | cryptography |
| CWE-843 | 33 | 1 | memory_safety |
| CWE-367 | 32 | 9 | concurrency |
| CWE-681 | 32 | 2 | numeric |
| CWE-77 | 31 | 5 | injection |
| CWE-18 | 31 | 15 | deprecated |
| CWE-327 | 31 | 8 | cryptography |
| CWE-203 | 30 | 6 | access_control |
| CWE-755 | 29 | 14 | error_handling |
| CWE-134 | 28 | 1 | memory_safety |
| CWE-285 | 25 | 6 | access_control |
| CWE-129 | 25 | 2 | numeric |
| CWE-121 | 25 | 1 | memory_safety |
| CWE-611 | 22 | 11 | configuration |
| CWE-191 | 22 | 2 | numeric |
| CWE-436 | 22 | 4 | input_validation |
| CWE-347 | 21 | 8 | cryptography |
| CWE-1284 | 21 | 4 | input_validation |
| CWE-326 | 20 | 8 | cryptography |
| CWE-345 | 19 | 11 | configuration |
| CWE-824 | 19 | 1 | memory_safety |
| CWE-502 | 18 | 12 | data_integrity |
| CWE-763 | 18 | 1 | memory_safety |
| CWE-776 | 18 | 3 | resource_management |
| CWE-444 | 18 | 5 | injection |
| CWE-311 | 16 | 8 | cryptography |
| CWE-662 | 16 | 9 | concurrency |
| CWE-358 | 15 | 10 | code_quality |
| CWE-601 | 15 | 6 | access_control |
| CWE-94 | 15 | 5 | injection |
| CWE-388 | 14 | 10 | code_quality |
| CWE-697 | 13 | 2 | numeric |
| CWE-426 | 11 | 10 | code_quality |
| CWE-276 | 11 | 11 | configuration |
| CWE-212 | 11 | 6 | access_control |
| CWE-116 | 11 | 5 | injection |
| CWE-131 | 10 | 2 | numeric |
| CWE-346 | 9 | 7 | authentication |
| CWE-255 | 8 | 7 | authentication |
| CWE-320 | 8 | 8 | cryptography |
| CWE-552 | 8 | 6 | access_control |
| CWE-331 | 8 | 8 | cryptography |
| CWE-319 | 7 | 8 | cryptography |
| CWE-440 | 7 | 10 | code_quality |
| CWE-911 | 7 | 3 | resource_management |
| CWE-385 | 7 | 8 | cryptography |
| CWE-281 | 6 | 6 | access_control |
| CWE-522 | 6 | 8 | cryptography |
| CWE-532 | 6 | 13 | logging |
| CWE-494 | 6 | 12 | data_integrity |
| CWE-90 | 6 | 5 | injection |
| CWE-428 | 6 | 10 | code_quality |
| CWE-407 | 6 | 3 | resource_management |
| CWE-918 | 5 | 6 | access_control |
| CWE-823 | 5 | 1 | memory_safety |
| CWE-337 | 5 | 8 | cryptography |
| CWE-606 | 5 | -1 | UNKNOWN |
| CWE-693 | 4 | 15 | deprecated |
| CWE-126 | 4 | 1 | memory_safety |
| CWE-273 | 4 | 14 | error_handling |
| CWE-1077 | 4 | 2 | numeric |
| CWE-670 | 4 | 10 | code_quality |
| CWE-241 | 4 | 4 | input_validation |
| CWE-229 | 4 | -1 | UNKNOWN |
| CWE-706 | 4 | 6 | access_control |
| CWE-1021 | 3 | 12 | data_integrity |
| CWE-172 | 3 | 4 | input_validation |
| CWE-16 | 3 | 11 | configuration |
| CWE-1188 | 3 | 11 | configuration |
| CWE-672 | 3 | 1 | memory_safety |
| CWE-118 | 3 | 1 | memory_safety |
| CWE-434 | 3 | 5 | injection |
| CWE-707 | 3 | 4 | input_validation |
| CWE-325 | 3 | -1 | UNKNOWN |
| CWE-209 | 2 | 14 | error_handling |
| CWE-113 | 2 | 5 | injection |
| CWE-762 | 2 | -1 | UNKNOWN |
| CWE-1333 | 2 | 3 | resource_management |
| CWE-306 | 2 | -1 | UNKNOWN |
| CWE-924 | 2 | 6 | access_control |
| CWE-88 | 2 | 5 | injection |
| CWE-639 | 2 | 6 | access_control |
| CWE-307 | 2 | 7 | authentication |
| CWE-838 | 2 | -1 | UNKNOWN |
| CWE-1325 | 2 | 3 | resource_management |
| CWE-757 | 2 | 8 | cryptography |
| CWE-99 | 2 | 5 | injection |
| CWE-664 | 1 | 3 | resource_management |
| CWE-352 | 1 | 6 | access_control |
| CWE-185 | 1 | 2 | numeric |
| CWE-1049 | 1 | 3 | resource_management |
| CWE-628 | 1 | 10 | code_quality |
| CWE-202 | 1 | 6 | access_control |
| CWE-248 | 1 | 14 | error_handling |
| CWE-626 | 1 | 10 | code_quality |
| CWE-460 | 1 | 14 | error_handling |
| CWE-290 | 1 | 6 | access_control |
| CWE-73 | 1 | 6 | access_control |
| CWE-324 | 1 | 8 | cryptography |
| CWE-93 | 1 | 5 | injection |
| CWE-117 | 1 | 13 | logging |
| CWE-1050 | 1 | 3 | resource_management |
| CWE-282 | 1 | 6 | access_control |
| CWE-610 | 1 | 6 | access_control |
| CWE-338 | 1 | 8 | cryptography |
| CWE-300 | 1 | 6 | access_control |
| CWE-349 | 1 | 8 | cryptography |
| CWE-457 | 1 | 3 | resource_management |
| CWE-392 | 1 | 14 | error_handling |
| CWE-114 | 1 | 6 | access_control |
| CWE-680 | 1 | 2 | numeric |
| CWE-789 | 1 | 3 | resource_management |

---

## Cross-Dataset Group Coverage (vulnerable only)

| Group ID | Group | BigVul | DiverseVul | MegaVul | TitanVul | BenchVul | Merged |
|---|---|---|---|---|---|---|---|
| 1 | memory_safety | 3618 | 7065 | 12214 | 2593 | 73 | 8962 |
| -1 | UNKNOWN | 2135 | 2836 | 2143 | 1353 | 7 | 2764 |
| 6 | access_control | 1341 | 1839 | 2424 | 750 | 54 | 2286 |
| 3 | resource_management | 916 | 1383 | 2784 | 553 | 24 | 2100 |
| 4 | input_validation | 1153 | 1424 | 2020 | 540 | 0 | 1818 |
| 2 | numeric | 703 | 1187 | 2012 | 488 | 23 | 1606 |
| 9 | concurrency | 266 | 450 | 1195 | 191 | 0 | 824 |
| 10 | code_quality | 131 | 403 | 1214 | 183 | 0 | 717 |
| 8 | cryptography | 130 | 547 | 386 | 299 | 0 | 310 |
| 5 | injection | 101 | 395 | 437 | 186 | 26 | 310 |
| 14 | error_handling | 13 | 835 | 132 | 275 | 0 | 80 |
| 7 | authentication | 36 | 281 | 462 | 138 | 19 | 170 |
| 15 | deprecated | 271 | 179 | 258 | 36 | 0 | 339 |
| 11 | configuration | 75 | 96 | 192 | 51 | 0 | 149 |
| 12 | data_integrity | 5 | 18 | 50 | 12 | 4 | 27 |
| 13 | logging | 1 | 7 | 11 | 2 | 0 | 7 |
