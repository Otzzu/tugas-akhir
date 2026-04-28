# CWE Distribution in BigVul Dataset — Grouped by Category

Source: `data/datasets/bigvul/train.parquet` — 150,908 rows total  
References: OWASP Top 10:2025 (full CWE list), CWE Top 25 Most Dangerous (2025)

Legend:
- `[OWASP: A0X]` = mapped in OWASP Top 10:2025 under that category
- `[Top25 #N]` = appears in CWE Top 25 Most Dangerous Software Weaknesses 2025

---

## 1. Memory Safety — Buffer / Pointer

> C/C++ systems-level memory errors. No OWASP equivalent.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-119 | 21,293 | Buffer Bounds (parent) | — | — |
| CWE-416 | 7,887 | Use After Free | — | #7 |
| CWE-125 | 6,220 | Out-of-bounds Read | — | #8 |
| CWE-476 | 3,631 | NULL Pointer Dereference | — | #13 |
| CWE-787 | 2,302 | Out-of-bounds Write | — | #5 |
| CWE-415 | 812 | Double Free | — | — |
| CWE-134 | 536 | Use of Externally-Controlled Format String | — | — |
| CWE-120 | 84 | Buffer Copy Without Checking Size | — | #11 |
| CWE-909 | 41 | Missing Initialization of Resource | — | — |
| CWE-763 | 8 | Release of Invalid Pointer or Reference | — | — |
| CWE-824 | 7 | Access of Uninitialized Pointer | — | — |
| **Total** | **42,821** | | | |

---

## 2. Numeric / Arithmetic Errors

> Integer math errors — overflow, underflow, divide-by-zero, bad indexing.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-189 | 5,552 | Numeric Errors (deprecated parent) | — | — |
| CWE-190 | 2,922 | Integer Overflow or Wraparound | — | — |
| CWE-369 | 584 | Divide by Zero | — | — |
| CWE-682 | 55 | Incorrect Calculation | — | — |
| CWE-191 | 55 | Integer Underflow | — | — |
| CWE-129 | 52 | Improper Validation of Array Index | — | — |
| **Total** | **9,220** | | | |

---

## 3. Resource Management

> Leaks, exhaustion, improper release of system resources.  
> Note: CWE-400 and CWE-404 mapped to A05 (Injection) in OWASP 2025.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-399 | 11,769 | Resource Management Errors (deprecated parent) | — | — |
| CWE-400 | 1,013 | Uncontrolled Resource Consumption | A05 (Injection) | — |
| CWE-772 | 971 | Missing Release of Resource After Effective Lifetime | — | — |
| CWE-404 | 733 | Improper Resource Shutdown or Release | A05 (Injection) | — |
| CWE-770 | 49 | Allocation of Resources Without Limits or Throttling | — | #25 |
| CWE-664 | 13 | Improper Control of Resource Lifetime | — | — |
| CWE-769 | 3 | File Descriptor Exhaustion | — | — |
| **Total** | **14,551** | | | |

---

## 4. Input Validation

> Improper validation at system boundaries. Parent of injection and many others.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-20 | 16,392 | Improper Input Validation (deprecated parent) | — | #18 |
| CWE-436 | 63 | Interpretation Conflict | — | — |
| CWE-172 | 18 | Encoding Error | — | — |
| CWE-354 | 16 | Improper Validation of Integrity Check Value | — | — |
| CWE-252 | 1 | Unchecked Return Value | — | — |
| **Total** | **16,490** | | | |

---

## 5. Injection

> Untrusted input interpreted as command or query.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-79 | 698 | Cross-site Scripting (XSS) | A05 | #1 |
| CWE-78 | 224 | OS Command Injection | A05 | #9 |
| CWE-77 | 182 | Command Injection | A05 | #23 |
| CWE-89 | 177 | SQL Injection | A05 | #2 |
| CWE-94 | 146 | Code Injection | A05 | #10 |
| CWE-93 | 82 | CRLF Injection | A05 | — |
| CWE-74 | 24 | Injection (parent) | A05 | — |
| CWE-90 | 17 | LDAP Injection | A05 | — |
| **Total** | **1,550** | | | |

---

## 6. Access Control / Privileges

> Unauthorized actions, missing permission checks, path traversal.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-264 | 9,894 | Permissions, Privileges, Access Controls (deprecated parent) | A01 | — |
| CWE-200 | 6,959 | Exposure of Sensitive Information to Unauthorized Actor | A01 | #20 |
| CWE-284 | 1,916 | Improper Access Control | A01 | #19 |
| CWE-269 | 747 | Improper Privilege Management | A06 (Insecure Design) | — |
| CWE-59 | 869 | Improper Link Resolution (Symlink Attack) | A01 | — |
| CWE-22 | 559 | Path Traversal | A01 | #6 |
| CWE-285 | 359 | Improper Authorization | A01 | — |
| CWE-601 | 65 | URL Redirection to Untrusted Site (Open Redirect) | A10 | — |
| CWE-862 | 42 | Missing Authorization | A01 | #4 |
| CWE-281 | 39 | Improper Preservation of Permissions | — | — |
| CWE-352 | 27 | Cross-Site Request Forgery (CSRF) | A01 | #3 |
| CWE-918 | 29 | Server-Side Request Forgery (SSRF) | A01 | #22 |
| CWE-290 | 55 | Authentication Bypass by Spoofing | A07 | — |
| CWE-668 | 21 | Exposure of Resource to Wrong Sphere | — | — |
| CWE-706 | 10 | Use of Incorrectly-Resolved Name or Reference | — | — |
| **Total** | **21,591** | | | |

---

## 7. Authentication

> Identity verification failures, credential handling.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-287 | 475 | Improper Authentication | A07 | — |
| CWE-255 | 60 | Credentials Management (deprecated parent) | A07 | — |
| CWE-346 | 56 | Origin Validation Error | A07, A03 | — |
| CWE-295 | 53 | Improper Certificate Validation | A04, A07 | — |
| **Total** | **644** | | | |

---

## 8. Cryptography

> Broken crypto, missing encryption, weak key management.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-310 | 1,173 | Cryptographic Issues (deprecated parent) | A04 | — |
| CWE-311 | 222 | Missing Encryption of Sensitive Data | A04, A06 | — |
| CWE-320 | 109 | Key Management Errors | — | — |
| CWE-330 | 59 | Use of Insufficiently Random Values | A04 | — |
| CWE-347 | 28 | Improper Verification of Cryptographic Signature | A04 | — |
| CWE-522 | 5 | Insufficiently Protected Credentials | A06, A07 | — |
| CWE-327 | 4 | Use of Broken or Risky Cryptographic Algorithm | A04 | — |
| **Total** | **1,600** | | | |

---

## 9. Concurrency

> Shared state issues under concurrent execution.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-362 | 4,829 | Race Condition (Concurrent Execution) | — | — |
| CWE-361 | 22 | Time and State (deprecated parent) | — | — |
| **Total** | **4,851** | | | |

---

## 10. Code Quality / Control Flow

> Logic errors, assertion failures, type confusion, infinite loops.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-704 | 721 | Incorrect Type Conversion or Cast | — | — |
| CWE-617 | 838 | Reachable Assertion | — | — |
| CWE-835 | 693 | Infinite Loop | — | — |
| CWE-834 | 261 | Excessive Iteration | — | — |
| CWE-388 | 234 | Error Handling (deprecated) | — | — |
| CWE-674 | 55 | Uncontrolled Recursion | — | — |
| CWE-358 | 47 | Improperly Implemented Security Check | — | — |
| CWE-426 | 23 | Untrusted Search Path | A08 (Data Integrity) | — |
| **Total** | **2,872** | | | |

---

## 11. Configuration / Supply Chain

> Insecure settings, untrusted components, permission assignment.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-732 | 1,169 | Incorrect Permission Assignment for Critical Resource | A01, A02 | — |
| CWE-611 | 370 | Improper Restriction of XML External Entity (XXE) | A02, A05 | — |
| CWE-16 | 48 | Configuration | A02 | — |
| CWE-346 | 56 | Origin Validation Error | A03, A07 | — |
| CWE-345 | 2 | Insufficient Verification of Data Authenticity | A03, A08 | — |
| **Total** | **1,645** | | | |

---

## 12. Data / Software Integrity

> Untrusted data treated as valid, deserialization, download integrity.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-494 | 36 | Download of Code Without Integrity Check | A08 | — |
| CWE-502 | 33 | Deserialization of Untrusted Data | A08 | #15 |
| CWE-1021 | 54 | Improper Restriction of Rendered UI Layers (Clickjacking) | A06 | — |
| **Total** | **123** | | | |

---

## 13. Logging / Monitoring

> Failure to log, or leaking sensitive info into logs.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-532 | 67 | Insertion of Sensitive Information into Log File | A09 | — |
| **Total** | **67** | | | |

---

## 14. Error / Exception Handling

> Software fails to handle unexpected or exceptional conditions.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-754 | 311 | Improper Check for Unusual or Exceptional Conditions | A10 | — |
| CWE-755 | 17 | Improper Handling of Exceptional Conditions | A10 | — |
| CWE-209 | 1 | Error Messages Containing Sensitive Information | A02, A06, A10 | — |
| **Total** | **329** | | | |

---

## 15. Deprecated / General (Unmappable)

> Broad or obsolete CWE categories with no specific modern mapping.

| CWE | Count | Name | OWASP 2025 | CWE Top 25 |
|---|---|---|---|---|
| CWE-254 | 2,420 | Security Features (deprecated parent) | — | — |
| CWE-19 | 563 | Data Handling (deprecated) | — | — |
| CWE-17 | 560 | Code (deprecated) | — | — |
| CWE-693 | 27 | Protection Mechanism Failure (deprecated) | — | — |
| CWE-18 | 37 | Source Code (deprecated) | — | — |
| **Total** | **3,607** | | | |

---

## Summary Table

| Category | Samples | OWASP Match | Top 25 CWEs in Category |
|---|---|---|---|
| Memory Safety | 42,821 | None | CWE-416(#7), CWE-125(#8), CWE-476(#13), CWE-787(#5), CWE-120(#11) |
| Input Validation | 16,490 | None | CWE-20(#18) |
| Resource Management | 14,551 | Partial — CWE-400, CWE-404 → A05 | CWE-770(#25) |
| Access Control / Privileges | 21,591 | Strong — CWE-264, 200, 284, 285, 22, 59, 862, 352, 918 → A01; CWE-269 → A06 | CWE-200(#20), CWE-284(#19), CWE-22(#6), CWE-862(#4), CWE-352(#3), CWE-918(#22) |
| Numeric / Arithmetic | 9,220 | None | — |
| Concurrency | 4,851 | None | — |
| Injection | 1,550 | Full — all → A05 | CWE-79(#1), CWE-89(#2), CWE-78(#9), CWE-94(#10), CWE-77(#23) |
| Cryptography | 1,600 | Strong — CWE-310, 311, 330, 347, 327, 295, 522 → A04/A06/A07 | — |
| Code Quality / Control Flow | 2,872 | Partial — CWE-426 → A08 | — |
| Configuration / Supply Chain | 1,645 | Full — all → A01/A02/A03 | — |
| Authentication | 644 | Full — all → A04/A07 | — |
| Data / Software Integrity | 123 | Full — all → A06/A08 | CWE-502(#15) |
| Error / Exception Handling | 329 | Full — all → A02/A06/A10 | — |
| Logging / Monitoring | 67 | Full — CWE-532 → A09 | — |
| Deprecated / General | 3,607 | None | — |

---

## Coverage Statistics

| Metric | Value |
|---|---|
| Total BigVul train samples | 150,908 |
| Unique CWEs | 59 |
| CWEs with OWASP 2025 mapping | 26 |
| CWEs in CWE Top 25 | 12 |
| Samples with OWASP mapping (est.) | ~42,000 (27.8%) |
| Samples without OWASP mapping | ~108,000 (71.6%) |

> **Key insight:** With the full OWASP CWE list (not just headline CWEs), dataset OWASP coverage increases from 8.7% → ~27.8%. Still, ~72% of BigVul is C/C++ memory safety / numeric / concurrency / deprecated — no OWASP equivalent. Dataset is better suited for systems-level vulnerability research than web application security.
