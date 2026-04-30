# Dataset Analysis — Complete CWE and Group Distribution

Generated from raw parquet files. Group mapping via `CWE_GROUP_MAP` in `dataset_lm.py`.

**Fixed Group IDs:**
0=benign · 1=memory_safety · 2=numeric · 3=resource_management · 4=input_validation ·
5=injection · 6=access_control · 7=authentication · 8=cryptography · 9=concurrency ·
10=code_quality · 11=configuration · 12=data_integrity · 13=logging · 14=error_handling ·
15=deprecated · -1=UNKNOWN (not in CWE_GROUP_MAP)

---

## Summary

| Dataset | Total | Benign | Vulnerable | Has CWE | Has Flaw Lines |
|---|---|---|---|---|---|
| BigVul | 217,007 | 206,112 | 10,895 | Yes | Yes (diff) |
| DiverseVul | 330,492 | 311,547 | 18,945 | Yes (multi-label) | No |
| MegaVul | 55,868 | 27,934 | 27,934 | Yes | Yes (diff) |
| Devign | 27,318 | 14,858 | 12,460 | No | Yes (vul_lines) |
| Merged (BigVul+MegaVul) | 176,674 | 154,205 | 22,469 | Yes | Yes (diff) |

---

## 1. BigVul (`data/datasets/bigvul/all.parquet`)

Total: **217,007** | Benign: **206,112** | Vulnerable: **10,895**

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 206,112 |
| 1 | memory_safety | 3,618 |
| -1 | UNKNOWN | 2,140 |
| 6 | access_control | 1,341 |
| 4 | input_validation | 1,153 |
| 3 | resource_management | 911 |
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
| CWE-665 | 5 | -1 | UNKNOWN |
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
| 0 | benign | 4000 | 206,112 | 1% (subsampled) |
| 1 | memory_safety | 3,618 | 3,618 | 100% |
| 6 | access_control | 1,341 | 1,341 | 100% |
| 4 | input_validation | 1,153 | 1,153 | 100% |
| 3 | resource_management | 911 | 911 | 100% |
| 2 | numeric | 703 | 703 | 100% |
| 15 | deprecated | 271 | 271 | 100% |
| 9 | concurrency | 266 | 266 | 100% |
| 10 | code_quality | 131 | 131 | 100% |
| 8 | cryptography | 130 | 130 | 100% |
| 5 | injection | 101 | 101 | 100% |
| 11 | configuration | 75 | 75 | 100% |
| 7 | authentication | 36 | 36 | 100% |
| 14 | error_handling | 13 | 13 | 100% |
| 12 | data_integrity | 5 | 5 | 100% |
| 13 | logging | 1 | 1 | 100% |
| -1 | UNKNOWN | 5 | 2,140 | filtered |

> All vulnerable CPG files = 100% coverage of all.parquet vulnerable.
> Benign subsampled to ~2,200 (planning to extend to ~4,000).

---

## 2. DiverseVul (`data/datasets/diversevul/all.parquet`)

Total: **330,492** | Benign: **311,547** | Vulnerable: **18,945**

> CWE column is multi-label list (e.g. `['CWE-787', 'CWE-119']`).
> Group assignment uses primary (first) CWE only.
> No flaw line ground truth available.

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 311,547 |
| 1 | memory_safety | 6,970 |
| -1 | UNKNOWN | 4,492 |
| 6 | access_control | 1,766 |
| 4 | input_validation | 1,361 |
| 2 | numeric | 1,159 |
| 3 | resource_management | 1,054 |
| 8 | cryptography | 469 |
| 10 | code_quality | 399 |
| 9 | concurrency | 398 |
| 5 | injection | 299 |
| 7 | authentication | 213 |
| 15 | deprecated | 179 |
| 14 | error_handling | 94 |
| 11 | configuration | 77 |
| 12 | data_integrity | 8 |
| 13 | logging | 7 |

### CWE Distribution (all vulnerable, primary CWE)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| *(empty)* | 2,836 | -1 | UNKNOWN |
| CWE-125 | 1,635 | 1 | memory_safety |
| CWE-119 | 1,433 | 1 | memory_safety |
| CWE-787 | 1,379 | 1 | memory_safety |
| CWE-20 | 1,315 | 4 | input_validation |
| CWE-416 | 999 | 1 | memory_safety |
| CWE-476 | 915 | 1 | memory_safety |
| CWE-703 | 735 | -1 | UNKNOWN |
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
| CWE-401 | 193 | -1 | UNKNOWN |
| CWE-59 | 164 | 6 | access_control |
| CWE-617 | 162 | 10 | code_quality |
| CWE-369 | 158 | 2 | numeric |
| CWE-22 | 140 | 6 | access_control |
| CWE-835 | 111 | 10 | code_quality |
| CWE-269 | 111 | 6 | access_control |
| CWE-287 | 105 | 7 | authentication |
| CWE-295 | 101 | 7 | authentication |
| CWE-772 | 100 | 3 | resource_management |
| CWE-770 | 89 | 3 | resource_management |
| CWE-94 | 84 | 5 | injection |
| CWE-78 | 76 | 5 | injection |
| CWE-122 | 68 | -1 | UNKNOWN |
| CWE-19 | 66 | 15 | deprecated |
| CWE-444 | 60 | -1 | UNKNOWN |
| CWE-241 | 57 | -1 | UNKNOWN |
| CWE-674 | 56 | 10 | code_quality |
| CWE-89 | 54 | 5 | injection |
| CWE-17 | 46 | 15 | deprecated |
| CWE-755 | 46 | 14 | error_handling |
| CWE-191 | 45 | 2 | numeric |
| CWE-459 | 43 | -1 | UNKNOWN |
| CWE-862 | 42 | 6 | access_control |
| CWE-613 | 40 | -1 | UNKNOWN |
| CWE-330 | 39 | 8 | cryptography |
| CWE-134 | 38 | 1 | memory_safety |
| CWE-327 | 38 | 8 | cryptography |
| CWE-732 | 38 | 11 | configuration |
| CWE-319 | 35 | -1 | UNKNOWN |
| CWE-601 | 35 | 6 | access_control |
| CWE-18 | 34 | 15 | deprecated |
| CWE-665 | 32 | -1 | UNKNOWN |
| CWE-77 | 31 | 5 | injection |
| CWE-254 | 31 | 15 | deprecated |
| CWE-754 | 30 | 14 | error_handling |
| CWE-704 | 28 | 10 | code_quality |
| CWE-79 | 27 | 5 | injection |
| CWE-74 | 25 | 5 | injection |
| CWE-347 | 25 | 8 | cryptography |
| CWE-326 | 25 | -1 | UNKNOWN |
| CWE-203 | 24 | -1 | UNKNOWN |
| CWE-252 | 24 | 4 | input_validation |
| CWE-290 | 23 | 6 | access_control |
| CWE-662 | 22 | -1 | UNKNOWN |
| CWE-667 | 21 | -1 | UNKNOWN |
| CWE-354 | 21 | 4 | input_validation |
| CWE-908 | 21 | -1 | UNKNOWN |
| CWE-358 | 20 | 10 | code_quality |
| CWE-345 | 20 | 11 | configuration |
| CWE-406 | 19 | -1 | UNKNOWN |
| CWE-276 | 19 | -1 | UNKNOWN |
| CWE-863 | 18 | -1 | UNKNOWN |
| CWE-209 | 18 | 14 | error_handling |
| CWE-763 | 17 | 1 | memory_safety |
| CWE-434 | 15 | -1 | UNKNOWN |
| CWE-611 | 14 | 11 | configuration |
| CWE-843 | 14 | -1 | UNKNOWN |
| CWE-323 | 13 | -1 | UNKNOWN |
| CWE-288 | 13 | -1 | UNKNOWN |
| CWE-668 | 12 | 6 | access_control |
| CWE-824 | 12 | 1 | memory_safety |
| CWE-131 | 11 | -1 | UNKNOWN |
| CWE-522 | 11 | 8 | cryptography |
| CWE-193 | 11 | -1 | UNKNOWN |
| CWE-388 | 10 | 10 | code_quality |
| CWE-909 | 10 | 1 | memory_safety |
| CWE-913 | 10 | -1 | UNKNOWN |
| CWE-552 | 10 | -1 | UNKNOWN |
| CWE-834 | 9 | 10 | code_quality |
| CWE-116 | 9 | -1 | UNKNOWN |
| CWE-367 | 9 | -1 | UNKNOWN |
| CWE-320 | 9 | 8 | cryptography |
| CWE-404 | 8 | 3 | resource_management |
| CWE-212 | 7 | -1 | UNKNOWN |
| CWE-121 | 7 | -1 | UNKNOWN |
| CWE-532 | 7 | 13 | logging |
| CWE-294 | 7 | -1 | UNKNOWN |
| CWE-126 | 7 | -1 | UNKNOWN |
| CWE-273 | 6 | -1 | UNKNOWN |
| CWE-918 | 6 | 6 | access_control |
| CWE-707 | 6 | -1 | UNKNOWN |
| CWE-682 | 6 | 2 | numeric |
| CWE-281 | 6 | 6 | access_control |
| CWE-1187 | 6 | -1 | UNKNOWN |
| CWE-346 | 6 | 7 | authentication |
| CWE-297 | 6 | -1 | UNKNOWN |
| CWE-16 | 5 | 11 | configuration |
| CWE-681 | 5 | -1 | UNKNOWN |
| CWE-1021 | 4 | 12 | data_integrity |
| CWE-823 | 4 | -1 | UNKNOWN |
| CWE-61 | 4 | -1 | UNKNOWN |
| CWE-303 | 4 | -1 | UNKNOWN |
| CWE-88 | 4 | -1 | UNKNOWN |
| CWE-91 | 4 | -1 | UNKNOWN |
| CWE-502 | 4 | 12 | data_integrity |
| CWE-311 | 4 | 8 | cryptography |
| CWE-331 | 4 | -1 | UNKNOWN |
| CWE-352 | 4 | 6 | access_control |
| CWE-426 | 3 | 10 | code_quality |
| CWE-672 | 3 | -1 | UNKNOWN |
| CWE-129 | 3 | 2 | numeric |
| CWE-670 | 3 | -1 | UNKNOWN |
| CWE-113 | 3 | -1 | UNKNOWN |
| CWE-307 | 2 | -1 | UNKNOWN |
| CWE-693 | 2 | 15 | deprecated |
| CWE-266 | 2 | -1 | UNKNOWN |
| CWE-93 | 2 | 5 | injection |
| CWE-776 | 2 | -1 | UNKNOWN |
| CWE-285 | 2 | 6 | access_control |
| CWE-924 | 2 | -1 | UNKNOWN |
| CWE-417 | 2 | -1 | UNKNOWN |
| CWE-565 | 1 | -1 | UNKNOWN |
| CWE-943 | 1 | -1 | UNKNOWN |
| CWE-255 | 1 | 7 | authentication |
| CWE-349 | 1 | -1 | UNKNOWN |
| CWE-457 | 1 | -1 | UNKNOWN |
| CWE-805 | 1 | -1 | UNKNOWN |
| CWE-436 | 1 | 4 | input_validation |
| CWE-798 | 1 | -1 | UNKNOWN |
| CWE-786 | 1 | -1 | UNKNOWN |
| CWE-428 | 1 | -1 | UNKNOWN |
| CWE-706 | 1 | -1 | UNKNOWN |
| CWE-697 | 1 | -1 | UNKNOWN |

---

## 3. MegaVul (`data/datasets/megavul/train.parquet`)

Total: **55,868** | Benign: **27,934** | Vulnerable: **27,934**

> Perfectly balanced (1:1 ratio). Has `func_before` + `func_after` for diff-based flaw lines.

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 27,934 |
| 1 | memory_safety | 11,973 |
| -1 | UNKNOWN | 4,336 |
| 6 | access_control | 2,228 |
| 4 | input_validation | 1,957 |
| 2 | numeric | 1,785 |
| 3 | resource_management | 1,785 |
| 10 | code_quality | 1,172 |
| 9 | concurrency | 968 |
| 7 | authentication | 460 |
| 5 | injection | 383 |
| 8 | cryptography | 292 |
| 15 | deprecated | 258 |
| 11 | configuration | 158 |
| 14 | error_handling | 120 |
| 12 | data_integrity | 50 |
| 13 | logging | 9 |

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-119 | 2,710 | 1 | memory_safety |
| CWE-787 | 2,226 | 1 | memory_safety |
| CWE-Other | 2,125 | -1 | UNKNOWN |
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
| CWE-401 | 517 | -1 | UNKNOWN |
| CWE-835 | 424 | 10 | code_quality |
| CWE-189 | 399 | 2 | numeric |
| CWE-287 | 327 | 7 | authentication |
| CWE-415 | 315 | 1 | memory_safety |
| CWE-772 | 307 | 3 | resource_management |
| CWE-617 | 297 | 10 | code_quality |
| CWE-122 | 207 | -1 | UNKNOWN |
| CWE-22 | 193 | 6 | access_control |
| CWE-369 | 188 | 2 | numeric |
| CWE-674 | 181 | 10 | code_quality |
| CWE-834 | 175 | 10 | code_quality |
| CWE-908 | 161 | -1 | UNKNOWN |
| CWE-269 | 151 | 6 | access_control |
| CWE-59 | 148 | 6 | access_control |
| CWE-310 | 141 | 8 | cryptography |
| CWE-770 | 138 | 3 | resource_management |
| CWE-667 | 135 | -1 | UNKNOWN |
| CWE-404 | 119 | 3 | resource_management |
| CWE-295 | 117 | 7 | authentication |
| CWE-193 | 107 | -1 | UNKNOWN |
| CWE-909 | 104 | 1 | memory_safety |
| CWE-78 | 102 | 5 | injection |
| CWE-284 | 96 | 6 | access_control |
| CWE-17 | 92 | 15 | deprecated |
| CWE-863 | 87 | -1 | UNKNOWN |
| CWE-732 | 84 | 11 | configuration |
| CWE-665 | 79 | -1 | UNKNOWN |
| CWE-19 | 78 | 15 | deprecated |
| CWE-74 | 78 | 5 | injection |
| CWE-252 | 77 | 4 | input_validation |
| CWE-79 | 71 | 5 | injection |
| CWE-254 | 68 | 15 | deprecated |
| CWE-459 | 66 | -1 | UNKNOWN |
| CWE-89 | 65 | 5 | injection |
| CWE-367 | 64 | -1 | UNKNOWN |
| CWE-681 | 64 | -1 | UNKNOWN |
| CWE-330 | 63 | 8 | cryptography |
| CWE-862 | 63 | 6 | access_control |
| CWE-843 | 62 | -1 | UNKNOWN |
| CWE-668 | 62 | 6 | access_control |
| CWE-754 | 62 | 14 | error_handling |
| CWE-354 | 60 | 4 | input_validation |
| CWE-704 | 59 | 10 | code_quality |
| CWE-327 | 59 | 8 | cryptography |
| CWE-755 | 56 | 14 | error_handling |
| CWE-134 | 53 | 1 | memory_safety |
| CWE-682 | 50 | 2 | numeric |
| CWE-203 | 49 | -1 | UNKNOWN |
| CWE-1284 | 46 | -1 | UNKNOWN |
| CWE-121 | 44 | -1 | UNKNOWN |
| CWE-129 | 42 | 2 | numeric |
| CWE-191 | 41 | 2 | numeric |
| CWE-77 | 41 | 5 | injection |
| CWE-326 | 40 | -1 | UNKNOWN |
| CWE-502 | 40 | 12 | data_integrity |
| CWE-611 | 38 | 11 | configuration |
| CWE-776 | 36 | -1 | UNKNOWN |
| CWE-436 | 36 | 4 | input_validation |
| CWE-763 | 35 | 1 | memory_safety |
| CWE-345 | 32 | 11 | configuration |
| CWE-824 | 32 | 1 | memory_safety |
| CWE-276 | 28 | -1 | UNKNOWN |
| CWE-662 | 28 | -1 | UNKNOWN |
| CWE-601 | 26 | 6 | access_control |
| CWE-697 | 26 | -1 | UNKNOWN |
| CWE-212 | 24 | -1 | UNKNOWN |
| CWE-444 | 22 | -1 | UNKNOWN |
| CWE-347 | 21 | 8 | cryptography |
| CWE-131 | 20 | -1 | UNKNOWN |
| CWE-552 | 18 | -1 | UNKNOWN |
| CWE-116 | 18 | -1 | UNKNOWN |
| CWE-388 | 16 | 10 | code_quality |
| CWE-426 | 16 | 10 | code_quality |
| CWE-440 | 16 | -1 | UNKNOWN |
| CWE-94 | 16 | 5 | injection |
| CWE-911 | 14 | -1 | UNKNOWN |
| CWE-18 | 14 | 15 | deprecated |
| CWE-319 | 13 | -1 | UNKNOWN |
| CWE-428 | 12 | -1 | UNKNOWN |
| CWE-331 | 12 | -1 | UNKNOWN |
| CWE-346 | 12 | 7 | authentication |
| CWE-385 | 12 | -1 | UNKNOWN |
| CWE-407 | 12 | -1 | UNKNOWN |
| CWE-126 | 10 | -1 | UNKNOWN |
| CWE-823 | 10 | -1 | UNKNOWN |
| CWE-670 | 10 | -1 | UNKNOWN |
| CWE-337 | 10 | -1 | UNKNOWN |
| CWE-532 | 9 | 13 | logging |
| CWE-918 | 8 | 6 | access_control |
| CWE-273 | 8 | -1 | UNKNOWN |
| CWE-1077 | 8 | -1 | UNKNOWN |
| CWE-90 | 8 | 5 | injection |
| CWE-241 | 8 | -1 | UNKNOWN |
| CWE-1188 | 6 | -1 | UNKNOWN |
| CWE-494 | 6 | 12 | data_integrity |
| CWE-672 | 6 | -1 | UNKNOWN |
| CWE-1333 | 6 | -1 | UNKNOWN |
| CWE-118 | 6 | -1 | UNKNOWN |
| CWE-434 | 6 | -1 | UNKNOWN |
| CWE-693 | 6 | 15 | deprecated |
| CWE-172 | 6 | 4 | input_validation |
| CWE-184 | 6 | -1 | UNKNOWN |
| CWE-229 | 6 | -1 | UNKNOWN |
| CWE-522 | 5 | 8 | cryptography |
| CWE-113 | 4 | -1 | UNKNOWN |
| CWE-16 | 4 | 11 | configuration |
| CWE-255 | 4 | 7 | authentication |
| CWE-762 | 4 | -1 | UNKNOWN |
| CWE-306 | 4 | -1 | UNKNOWN |
| CWE-924 | 4 | -1 | UNKNOWN |
| CWE-324 | 4 | -1 | UNKNOWN |
| CWE-1021 | 4 | 12 | data_integrity |
| CWE-285 | 4 | 6 | access_control |
| CWE-358 | 4 | 10 | code_quality |
| CWE-88 | 4 | -1 | UNKNOWN |
| CWE-639 | 4 | -1 | UNKNOWN |
| CWE-320 | 3 | 8 | cryptography |
| CWE-706 | 3 | -1 | UNKNOWN |
| CWE-707 | 3 | -1 | UNKNOWN |
| CWE-185 | 2 | -1 | UNKNOWN |
| CWE-1049 | 2 | -1 | UNKNOWN |
| CWE-628 | 2 | -1 | UNKNOWN |
| CWE-202 | 2 | -1 | UNKNOWN |
| CWE-248 | 2 | -1 | UNKNOWN |
| CWE-626 | 2 | -1 | UNKNOWN |
| CWE-460 | 2 | -1 | UNKNOWN |
| CWE-290 | 2 | 6 | access_control |
| CWE-73 | 2 | -1 | UNKNOWN |
| CWE-93 | 2 | 5 | injection |
| CWE-117 | 2 | -1 | UNKNOWN |
| CWE-209 | 2 | 14 | error_handling |
| CWE-1050 | 2 | -1 | UNKNOWN |
| CWE-282 | 2 | -1 | UNKNOWN |
| CWE-610 | 2 | -1 | UNKNOWN |
| CWE-338 | 2 | -1 | UNKNOWN |
| CWE-300 | 2 | -1 | UNKNOWN |
| CWE-307 | 2 | -1 | UNKNOWN |
| CWE-838 | 2 | -1 | UNKNOWN |
| CWE-325 | 2 | -1 | UNKNOWN |
| CWE-349 | 1 | -1 | UNKNOWN |

---

## 4. Devign (`data/datasets/devign/{train,validation,test}.parquet`)

Total: **27,318** | Benign: **14,858** | Vulnerable: **12,460**

> No CWE column. Binary only (`target` column). Cannot be used for multiclass/group mode.
> Flaw lines available via `vul_lines` column (Devign-format dict).

---

## 5. Merged (`data/datasets/merged/train.parquet`)

Total: **176,674** | Benign: **154,205** | Vulnerable: **22,469**

> Combined BigVul + MegaVul. Has `CWE ID`, `func_before`, `func_after`.

### Group Distribution

| Group ID | Group | Count |
|---|---|---|
| 0 | benign | 154,205 |
| 1 | memory_safety | 8,834 |
| -1 | UNKNOWN | 3,908 |
| 6 | access_control | 2,185 |
| 4 | input_validation | 1,790 |
| 3 | resource_management | 1,577 |
| 2 | numeric | 1,491 |
| 9 | concurrency | 708 |
| 10 | code_quality | 698 |
| 15 | deprecated | 339 |
| 5 | injection | 272 |
| 8 | cryptography | 258 |
| 7 | authentication | 168 |
| 11 | configuration | 135 |
| 14 | error_handling | 73 |
| 12 | data_integrity | 27 |
| 13 | logging | 6 |

### CWE Distribution (all vulnerable)

| CWE | Count | Group ID | Group |
|---|---|---|---|
| CWE-119 | 3,000 | 1 | memory_safety |
| CWE-unknown | 2,746 | -1 | UNKNOWN |
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
| CWE-401 | 260 | -1 | UNKNOWN |
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
| CWE-122 | 110 | -1 | UNKNOWN |
| CWE-674 | 105 | 10 | code_quality |
| CWE-404 | 104 | 3 | resource_management |
| CWE-269 | 102 | 6 | access_control |
| CWE-834 | 100 | 10 | code_quality |
| CWE-287 | 99 | 7 | authentication |
| CWE-732 | 91 | 11 | configuration |
| CWE-17 | 85 | 15 | deprecated |
| CWE-19 | 84 | 15 | deprecated |
| CWE-908 | 78 | -1 | UNKNOWN |
| CWE-770 | 71 | 3 | resource_management |
| CWE-667 | 68 | -1 | UNKNOWN |
| CWE-79 | 66 | 5 | injection |
| CWE-78 | 63 | 5 | injection |
| CWE-909 | 55 | 1 | memory_safety |
| CWE-193 | 54 | -1 | UNKNOWN |
| CWE-295 | 52 | 7 | authentication |
| CWE-74 | 50 | 5 | injection |
| CWE-704 | 48 | 10 | code_quality |
| CWE-459 | 48 | -1 | UNKNOWN |
| CWE-665 | 43 | -1 | UNKNOWN |
| CWE-754 | 42 | 14 | error_handling |
| CWE-863 | 42 | -1 | UNKNOWN |
| CWE-89 | 40 | 5 | injection |
| CWE-252 | 40 | 4 | input_validation |
| CWE-354 | 37 | 4 | input_validation |
| CWE-682 | 35 | 2 | numeric |
| CWE-862 | 34 | 6 | access_control |
| CWE-668 | 34 | 6 | access_control |
| CWE-330 | 33 | 8 | cryptography |
| CWE-843 | 33 | -1 | UNKNOWN |
| CWE-367 | 32 | -1 | UNKNOWN |
| CWE-681 | 32 | -1 | UNKNOWN |
| CWE-77 | 31 | 5 | injection |
| CWE-18 | 31 | 15 | deprecated |
| CWE-327 | 31 | 8 | cryptography |
| CWE-203 | 30 | -1 | UNKNOWN |
| CWE-755 | 29 | 14 | error_handling |
| CWE-134 | 28 | 1 | memory_safety |
| CWE-285 | 25 | 6 | access_control |
| CWE-129 | 25 | 2 | numeric |
| CWE-121 | 25 | -1 | UNKNOWN |
| CWE-611 | 22 | 11 | configuration |
| CWE-191 | 22 | 2 | numeric |
| CWE-436 | 22 | 4 | input_validation |
| CWE-347 | 21 | 8 | cryptography |
| CWE-1284 | 21 | -1 | UNKNOWN |
| CWE-326 | 20 | -1 | UNKNOWN |
| CWE-345 | 19 | 11 | configuration |
| CWE-824 | 19 | 1 | memory_safety |
| CWE-502 | 18 | 12 | data_integrity |
| CWE-763 | 18 | 1 | memory_safety |
| CWE-776 | 18 | -1 | UNKNOWN |
| CWE-444 | 18 | -1 | UNKNOWN |
| CWE-311 | 16 | 8 | cryptography |
| CWE-662 | 16 | -1 | UNKNOWN |
| CWE-358 | 15 | 10 | code_quality |
| CWE-601 | 15 | 6 | access_control |
| CWE-94 | 15 | 5 | injection |
| CWE-388 | 14 | 10 | code_quality |
| CWE-697 | 13 | -1 | UNKNOWN |
| CWE-426 | 11 | 10 | code_quality |
| CWE-276 | 11 | -1 | UNKNOWN |
| CWE-212 | 11 | -1 | UNKNOWN |
| CWE-116 | 11 | -1 | UNKNOWN |
| CWE-131 | 10 | -1 | UNKNOWN |
| CWE-346 | 9 | 7 | authentication |
| CWE-255 | 8 | 7 | authentication |
| CWE-320 | 8 | 8 | cryptography |
| CWE-552 | 8 | -1 | UNKNOWN |
| CWE-331 | 8 | -1 | UNKNOWN |
| CWE-319 | 7 | -1 | UNKNOWN |
| CWE-440 | 7 | -1 | UNKNOWN |
| CWE-911 | 7 | -1 | UNKNOWN |
| CWE-385 | 7 | -1 | UNKNOWN |
| CWE-281 | 6 | 6 | access_control |
| CWE-522 | 6 | 8 | cryptography |
| CWE-532 | 6 | 13 | logging |
| CWE-494 | 6 | 12 | data_integrity |
| CWE-90 | 6 | 5 | injection |
| CWE-428 | 6 | -1 | UNKNOWN |
| CWE-407 | 6 | -1 | UNKNOWN |
| CWE-918 | 5 | 6 | access_control |
| CWE-823 | 5 | -1 | UNKNOWN |
| CWE-337 | 5 | -1 | UNKNOWN |
| CWE-606 | 5 | -1 | UNKNOWN |
| CWE-693 | 4 | 15 | deprecated |
| CWE-126 | 4 | -1 | UNKNOWN |
| CWE-273 | 4 | -1 | UNKNOWN |
| CWE-1077 | 4 | -1 | UNKNOWN |
| CWE-670 | 4 | -1 | UNKNOWN |
| CWE-241 | 4 | -1 | UNKNOWN |
| CWE-229 | 4 | -1 | UNKNOWN |
| CWE-706 | 4 | -1 | UNKNOWN |
| CWE-1021 | 3 | 12 | data_integrity |
| CWE-172 | 3 | 4 | input_validation |
| CWE-16 | 3 | 11 | configuration |
| CWE-1188 | 3 | -1 | UNKNOWN |
| CWE-672 | 3 | -1 | UNKNOWN |
| CWE-118 | 3 | -1 | UNKNOWN |
| CWE-434 | 3 | -1 | UNKNOWN |
| CWE-707 | 3 | -1 | UNKNOWN |
| CWE-325 | 3 | -1 | UNKNOWN |
| CWE-209 | 2 | 14 | error_handling |
| CWE-113 | 2 | -1 | UNKNOWN |
| CWE-762 | 2 | -1 | UNKNOWN |
| CWE-1333 | 2 | -1 | UNKNOWN |
| CWE-306 | 2 | -1 | UNKNOWN |
| CWE-924 | 2 | -1 | UNKNOWN |
| CWE-88 | 2 | -1 | UNKNOWN |
| CWE-639 | 2 | -1 | UNKNOWN |
| CWE-307 | 2 | -1 | UNKNOWN |
| CWE-838 | 2 | -1 | UNKNOWN |
| CWE-1325 | 2 | -1 | UNKNOWN |
| CWE-757 | 2 | -1 | UNKNOWN |
| CWE-99 | 2 | -1 | UNKNOWN |
| CWE-664 | 1 | 3 | resource_management |
| CWE-352 | 1 | 6 | access_control |
| CWE-185 | 1 | -1 | UNKNOWN |
| CWE-1049 | 1 | -1 | UNKNOWN |
| CWE-628 | 1 | -1 | UNKNOWN |
| CWE-202 | 1 | -1 | UNKNOWN |
| CWE-248 | 1 | -1 | UNKNOWN |
| CWE-626 | 1 | -1 | UNKNOWN |
| CWE-460 | 1 | -1 | UNKNOWN |
| CWE-290 | 1 | 6 | access_control |
| CWE-73 | 1 | -1 | UNKNOWN |
| CWE-324 | 1 | -1 | UNKNOWN |
| CWE-93 | 1 | 5 | injection |
| CWE-117 | 1 | -1 | UNKNOWN |
| CWE-1050 | 1 | -1 | UNKNOWN |
| CWE-282 | 1 | -1 | UNKNOWN |
| CWE-610 | 1 | -1 | UNKNOWN |
| CWE-338 | 1 | -1 | UNKNOWN |
| CWE-300 | 1 | -1 | UNKNOWN |
| CWE-349 | 1 | -1 | UNKNOWN |
| CWE-457 | 1 | -1 | UNKNOWN |
| CWE-392 | 1 | -1 | UNKNOWN |
| CWE-114 | 1 | -1 | UNKNOWN |
| CWE-680 | 1 | -1 | UNKNOWN |
| CWE-789 | 1 | -1 | UNKNOWN |

---

## Cross-Dataset Group Coverage (vulnerable only)

| Group ID | Group | BigVul | DiverseVul | MegaVul | Merged |
|---|---|---|---|---|---|
| 1 | memory_safety | 3,618 | 6,970 | 11,973 | 8,834 |
| 6 | access_control | 1,341 | 1,766 | 2,228 | 2,185 |
| 4 | input_validation | 1,153 | 1,361 | 1,957 | 1,790 |
| 3 | resource_management | 911 | 1,054 | 1,785 | 1,577 |
| 2 | numeric | 703 | 1,159 | 1,785 | 1,491 |
| 9 | concurrency | 266 | 398 | 968 | 708 |
| 10 | code_quality | 131 | 399 | 1,172 | 698 |
| 8 | cryptography | 130 | 469 | 292 | 258 |
| 5 | injection | 101 | 299 | 383 | 272 |
| 7 | authentication | 36 | 213 | 460 | 168 |
| 15 | deprecated | 271 | 179 | 258 | 339 |
| 11 | configuration | 75 | 77 | 158 | 135 |
| 14 | error_handling | 13 | 94 | 120 | 73 |
| 12 | data_integrity | 5 | 8 | 50 | 27 |
| 13 | logging | 1 | 7 | 9 | 6 |
| -1 | UNKNOWN | 2,140 | 4,492 | 4,336 | 3,908 |
