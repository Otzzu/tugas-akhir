# Ablation Study Plan

**Data:** MegaVul Top25, no CWE drop (25 classes), node_lm=unixcoder, func_lm=unixcoder
**Config base:** `configs/data/megavul_multiclass_top25_node-unixcoder_func-unixcoder.yaml`
**Strategy:** each phase builds on best model from previous phase

---

## Phase 1: Encoder + Localization Encoder

| ID | Model | localization_encoder | Notes |
|----|-------|---------------------|-------|
| A1 | lmgat | — | no live func LM |
| A2 | lmgat_codebert | gnn | live LM, GNN loc |
| A3 | lmgat_codebert | lm | live LM, LM loc |
| A4 | lmgat_codebert | both | live LM, both loc |

→ Best of A1-A4 continues to Phase 2

---

## Phase 2: Bidirectional Cross-Task

| ID | Method | Notes |
|----|--------|-------|
| B1 | none | baseline from Phase 1 best |
| B2 | direct conditioning | simplest, extend group→CWE pattern |
| B3 | FiLM | learned scale+shift per graph |
| B4 | cross-task attention | GNN nodes as K/V (works any localization_encoder) |

**Blocker:** B2/B3/B4 not yet implemented.

→ Best of B1-B4 continues to Phase 3

---

## Phase 3: MTL Head

| ID | active_heads | use_group_cond | Notes |
|----|-------------|----------------|-------|
| C1 | [cwe] | — | VulANalyzeR-like (CWE only) |
| C2 | [binary, cwe] | false | VulANalyzeR exact |
| C3 | [binary, group, cwe] | false | our MTL, no conditioning |
| C4 | [binary, group, cwe] | true | our full MTL with group→CWE cond |

→ Best of C1-C4 continues to Phase 4

---

## Phase 4: LM Choice

| ID | node_lm | func_lm |
|----|---------|---------|
| D1 | unixcoder | unixcoder | base |
| D2 | codebert | unixcoder | |
| D3 | unixcoder | codebert | |
| D4 | codebert | codebert | |

→ Best of D1-D4 continues to Phase 5

---

## Phase 5: Sliding Context for func_lm (UniXcoder max=1024)

| ID | func_chunk_size | func_chunk_stride | Notes |
|----|-----------------|-------------------|-------|
| E1 | 0 (disabled) | — | truncate at 1024 |
| E2 | 512 | 256 | 50% overlap, short chunks |
| E3 | 768 | 512 | 50% overlap, large chunks |
| E4 | 512 | 128 | aggressive overlap |

→ Best of E1-E4 continues to Phase 6

---

## Phase 6: Contrastive Learning

| ID | contrastive | Notes |
|----|-------------|-------|
| F1 | without | baseline |
| F2 | with | SupCon — same-CWE pulled together, diff-CWE pushed apart |

→ Final best non-sequential model

---

## Phase 7+: Sequential Model Fallback

> Only if Phase 1-6 results are insufficient.

Use `lmgat_seq` (Arch7) — two-stage sequential:
- Stage 1: GNN → per-node suspicion `s_i`
- Stage 2: GNN(cat(x, s_i)) + live LM → classification

Ablation dimensions TBD based on Phase 1-6 findings.

**Gradient note:** `lmgat_seq` currently detaches `s_i` (train.py Stage 1 = MIL only).
To enable bidirectional: remove `.detach()` → classification loss also trains Stage 1.

---

## Status

| Phase | Status |
|-------|--------|
| Phase 1 | Ready to run |
| Phase 2 | Blocked — cross-task methods not implemented |
| Phase 3 | Ready (config changes only) |
| Phase 4 | Ready (config changes only) |
| Phase 5 | Ready (config changes only) |
| Phase 6 | Blocked — contrastive loss not implemented |
| Phase 7+ | Fallback, not started |
