# Training Speedup Plan

Consolidated list of all performance optimizations for the training pipeline.
Ordered by impact × safety. Tick items as they land.

---

## Done

- [x] **Remove `.item()` syncs from train hot path** (commit `3caaa37`, 2026-05-13)
  - Accumulate loss on GPU, sync once at epoch end
  - Throttle tqdm refresh to ~100 updates/epoch
  - Expected: 10-20% faster epochs on live-LM archs (A2/A3/A4)

- [x] **Vectorize `StmtHead.score()`** (2026-05-13)
  - Scatter-based implementation covers gnn / lm / both modes
  - Validated: `torch.allclose(old, new, rtol=1e-4)` — max abs error < 5e-7
  - Measured speedup on A100:
    - gnn mode: 105ms → 1.94ms (**54×**)
    - lm mode: 159ms → 3.71ms (**43×**)
    - both mode: 171ms → 4.69ms (**37×**)
  - Gated by config flag (`StmtHead._vectorized`)

- [x] **Vectorize MIL + ranking losses** (2026-05-13)
  - `mil_loss`: removed per-graph `.item()`, uses `(labels > 0).float()` once
  - `mil_loss_multiclass`: uses `label.long().expand()` instead of `.item()`
  - `ranking_loss`: `scatter_reduce_` replaces per-line Python loop
  - Measured speedup on A100:
    - mil_loss: 9.56ms → 4.68ms (**2×**)
    - ranking_loss: 53.22ms → 14.98ms (**3.5×**)
  - All pass numerical correctness tests

---

## Pending (Priority Order)

### 1. Dynamic padding per batch (live-LM configs A2/A3/A4) ⭐⭐⭐
**Target:** `src/gnn_vuln/train.py` — `_strip_collate_fn` or new collate function

**Problem:** `func_max_length: 1024` in all phase1 configs. Every function is padded to 1024 tokens at dataset processing time, so every UniXcoder forward processes **1024 tokens per sample regardless of actual function length**. If median function is ~300 tokens, ~70% of LM compute is wasted on padding.

**Approach:** Custom collate_fn that pads each batch to the longest sequence in that batch:
```python
def _dynamic_pad_collate(batch):
    pyg_batch = Batch.from_data_list(batch)
    if hasattr(pyg_batch, "func_attention_mask"):
        max_len = int(pyg_batch.func_attention_mask.sum(dim=1).max().item())
        max_len = max(max_len, 1)
        pyg_batch.func_input_ids      = pyg_batch.func_input_ids[:, :max_len]
        pyg_batch.func_attention_mask = pyg_batch.func_attention_mask[:, :max_len]
        if hasattr(pyg_batch, "func_token_lines"):
            pyg_batch.func_token_lines = pyg_batch.func_token_lines[:, :max_len]
    return pyg_batch
```

**Verify first:** Check `FUNC_LINE_ANALYSIS.md` for actual token length distribution. Confirm tokens are right-padded (UniXcoder default).

**Expected speedup:** 2-4× on UniXcoder forward if median function length << 1024. **Biggest single live-LM win available without model changes.**

**Risk:** Low. Attention masks already handle variable-length inputs correctly.

**Effort:** ~30 min

---

### 2. Flash Attention 2 on the live LM branch ⭐⭐⭐
**Target:** `src/gnn_vuln/models/base.py` — `_build_lm_branch()`

**Problem:** Current `_build_lm_branch()` loads the live LM with default eager attention. Flash Attention 2 is only wired up for the **frozen node embedder during dataset preprocessing** (`data/node_embedder.py`). Your configs have `use_flash_attention: true` for A2-A4 but it has no effect on training — only on preprocessing.

**Approach:** Port the pattern from `node_embedder.py` to `base.py`:
```python
load_kwargs = {"trust_remote_code": True, "config": _cfg}
if use_flash_attention and torch.cuda.is_available():
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        load_kwargs["torch_dtype"] = torch.bfloat16
    except ImportError:
        logger.warning("flash-attn not installed — using standard attention")

try:
    self.codebert = AutoModel.from_pretrained(_func_lm, **load_kwargs)
except (ValueError, NotImplementedError):
    # Fallback if model doesn't support FA2
    load_kwargs.pop("attn_implementation", None)
    load_kwargs.pop("torch_dtype", None)
    self.codebert = AutoModel.from_pretrained(_func_lm, **load_kwargs)
```

Add `use_flash_attention` parameter to `_build_lm_branch` and thread through from config.

**Expected speedup:** 1.5-3× on LM forward. Biggest win on Ampere/Hopper GPUs (A100/H100/L40/RTX 4090).

**Risk:** None — falls back to standard attention on older GPUs or when flash-attn not installed. UniXcoder (RoBERTa-based) supports FA2 in recent transformers versions.

**Effort:** ~20 min

---

### 3. Make gradient checkpointing configurable ⭐⭐
**Target:** `src/gnn_vuln/models/base.py` — `_build_lm_branch()`

**Problem:** `gradient_checkpointing_enable()` is called unconditionally when loading live LM. Trades ~30% compute for ~40% VRAM savings. For GPUs with sufficient VRAM (A100/L40/H100), this is pure overhead.

**Approach:** Add `use_grad_checkpoint: bool = True` parameter (default preserves current behavior), thread through from config:
```python
if use_grad_checkpoint and hasattr(self.codebert, "gradient_checkpointing_enable"):
    self.codebert.gradient_checkpointing_enable()
```

In configs where VRAM allows, set `model.use_grad_checkpoint: false`.

**Expected speedup:** ~30% faster LM forward+backward when VRAM allows disabling it.

**Risk:** None — defaults preserve current behavior.

**Effort:** ~15 min

---

### 4. Vectorize class weight setup at startup ⭐⭐
**Target:** `src/gnn_vuln/train.py` — `_setup_class_weights()`

**Problem:**
```python
train_labels = torch.tensor(
    [int(dataset[i].y.item()) for i in train_idx], dtype=torch.long
)
```
For `max_per_class=1600` × 26 classes = ~42k samples, this calls `dataset[i]` 42k times at startup → 30-60s stall before training begins.

**Approach:** Read `y` directly from the underlying `.pt` tensor in one slice:
```python
train_labels = dataset._data.y[torch.tensor(train_idx)].long()
```

**Expected saving:** 30-60s per run, especially noticeable in short debug runs.

**Effort:** ~10 min

---

### 5. Reduce `evaluate()` syncs ⭐⭐
**Target:** `src/gnn_vuln/training/trainer.py` — `evaluate()`

**Problem:** 4 GPU↔CPU syncs per batch:
```python
total_loss += loss.item() * batch.num_graphs             # sync
total_conf += probs.max(dim=-1).values.sum().item()      # sync
all_preds.extend(preds.cpu().tolist())                   # sync + transfer
all_labels.extend(batch.y.cpu().tolist())                # sync + transfer
```

**Approach:** Accumulate `preds`, `labels` as list of GPU tensors, single `torch.cat` + `.cpu()` at end. Accumulate `loss_sum` and `conf_sum` on GPU, one sync at end.

**Expected speedup:** 5-10% faster validation.

**Effort:** ~15 min

---

### 6. `torch.compile` on LM branch only ⭐⭐
**Target:** `src/gnn_vuln/models/base.py` — after `_build_lm_branch()`

**Problem:** Full-model compile (`compile_model: true`) often fails with PyG dynamic shapes. But compiling **only the LM branch** usually works because it has fixed input shape (batch × seq_len × hidden).

**Approach:**
```python
if getattr(cfg.train, "compile_lm", False) and torch.cuda.is_available():
    self.codebert = torch.compile(self.codebert, mode="reduce-overhead", dynamic=True)
```

`dynamic=True` handles variable seq_len (needed if item 1 is enabled).

**Expected speedup:** 15-25% on LM forward after warmup.

**Risk:** Medium. First batch is slow (compilation). Recompiles if input shape changes.

**Effort:** ~15 min

**Dependency:** Test after item 1 (dynamic padding) lands since they interact.

---

### 7. `torch.inference_mode()` in evaluate ⭐
**Target:** `src/gnn_vuln/training/trainer.py` — `evaluate()` and `localise()`

**Problem:** `@torch.no_grad()` disables gradient tracking but still increments tensor version counters. `@torch.inference_mode()` is 5-10% faster.

**Approach:** Replace both decorators.

**Risk:** None for standard eval flow.

**Effort:** ~5 min, stack with item 5.

---

### 8. Freeze bottom layers of live LM (quality tradeoff) ⭐⭐
**Target:** `src/gnn_vuln/models/base.py` — `_build_lm_branch()`

**Problem:** Full 12-layer UniXcoder fine-tuning is expensive. Bottom layers capture general language features that rarely need to adapt.

**Approach:** Add `freeze_lm_layers: int` config. Freeze embeddings + first N transformer layers:
```python
if freeze_lm_layers > 0:
    for p in self.codebert.embeddings.parameters():
        p.requires_grad = False
    for layer in self.codebert.encoder.layer[:freeze_lm_layers]:
        for p in layer.parameters():
            p.requires_grad = False
```

Common choices: freeze 6 of 12 layers. Also reduces optimizer memory.

**Expected speedup:** ~2× faster backward on LM if freezing 6 of 12 layers.

**Risk:** **Quality tradeoff.** May drop F1 slightly. Requires validation experiment vs full fine-tuning.

**Effort:** ~20 min code + 1 epoch experiment per freeze depth

---

### 9. Benchmark `num_workers` (diagnostic) ⭐-⭐⭐
**Target:** `configs/ablation/phase1/*.yaml` — `num_workers` setting (currently 12)

**Problem:** 12 workers × `persistent_workers=True` takes ~12× dataset state in RAM. On cloud GPUs with 16GB RAM, context-switching and memory pressure can make 12 workers slower than 6-8.

**Approach:** Run 1 epoch with `num_workers ∈ {4, 8, 12, 16}` on same GPU. Pick fastest.

**Monitor:**
```bash
htop         # watch RAM usage per worker + CPU util
nvidia-smi   # watch GPU util — should stay > 90%
```

**Expected:** Could be 5-15% either direction depending on GPU/storage speed.

**Effort:** ~30 min benchmark

---

### 10. Test `torch.compile` full model on A1 ⭐⭐
**Target:** `configs/ablation/phase1/A1_lmgat.yaml` — set `compile_model: true`

**Problem:** Your `train.py` already attempts full-model compile when enabled. A1 (no live LM, fixed architecture) is the best candidate to test. PyG dynamic shapes may cause graph breaks.

**Approach:** Flip the flag on A1, run 1-2 epochs, check for errors.

**Expected:** 20-30% forward speedup if compilation succeeds.

**Risk:** Low — falls back to eager mode on failure via existing try/except.

**Effort:** 10 min test run.

---

### 11. Profile with `torch.profiler` (guidance for next wave)
**Target:** One-off profiling run after items 1-5 land.

**Approach:**
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
) as prof:
    for step, batch in enumerate(train_loader):
        ...
        prof.step()
```

Inspect via TensorBoard. Identify next biggest bottleneck based on actual timing data.

**Effort:** ~30 min

---

## Skipped (considered)

- **Custom CUDA prefetcher** — User tried before, got stuck. Built-in `num_workers=12, prefetch_factor=4` already achieves most of the benefit with zero risk.
- **MulticlassStmtHead vectorization** — Not used in current experiments.
- **BatchNorm kernel fusion** — `torch.compile` handles this automatically if it works.
- **Edge attr argmax optimization in `lmgat_codebert_mtl.py`** — Only relevant if `use_edge_emb=true` is used; not in A1-A4.
- **Full GPU-side ranking_loss** — Would require padding scores across graphs with different stmt counts; current 3.5× speedup is good enough.

---

## Quick Reference — Impact Estimate for Live-LM Configs (A2/A3/A4)

| Component | Status | Saving |
|-----------|--------|--------|
| Loss.item() syncs (train) | ✅ done | 10-20% epoch time |
| StmtHead double loop | ✅ done | 15-25% epoch time |
| MIL/ranking loops | ✅ done | 5-10% epoch time |
| **Dynamic padding (item 1)** | planned | **~2-4× LM forward** |
| **Flash Attention 2 on live LM (item 2)** | planned | **1.5-3× LM forward** |
| Gradient checkpoint off (item 3) | planned | ~30% LM forward+backward |
| Class weight startup (item 4) | planned | 30-60s per run |
| Evaluate syncs (item 5) | planned | 5-10% eval time |
| torch.compile LM (item 6) | planned | 15-25% LM forward |

**If items 1-6 all land: expected 3-6× faster training on live-LM configs.**
Items 1 and 2 alone (dynamic padding + FA2) are the dominant wins for A2/A3/A4.

Items 9-11 are diagnostic/opportunistic — could add another 10-30% depending on results.
