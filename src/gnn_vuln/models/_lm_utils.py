"""Shared helpers for live LM branches across model architectures."""

from __future__ import annotations

import torch
from loguru import logger


def _is_codet5p_embedding(model) -> bool:
    """True for Salesforce/codet5p-*-embedding — returns raw pooled tensor, not ModelOutput."""
    return getattr(model.config, "model_type", "") == "codet5p_embedding"


def _is_t5_like(model) -> bool:
    """True for T5-family models (enc-dec or enc-only), False for BERT-family."""
    if _is_codet5p_embedding(model):
        return False
    cfg = model.config
    return (
        getattr(cfg, "is_encoder_decoder", False)
        or "t5" in getattr(cfg, "model_type", "").lower()
    )


def _is_decoder_only(model) -> bool:
    """True for decoder-only (GPT/Qwen/LLaMA-family) models."""
    cfg = model.config
    return (
        not getattr(cfg, "is_encoder_decoder", False)
        and getattr(cfg, "model_type", "") in {
            "qwen2", "gpt2", "llama", "mistral", "gemma", "phi", "falcon", "bloom",
        }
    )


def lm_hidden_dim(model, matryoshka_dim: int | None = None) -> int:
    """Return hidden size of the LM; returns matryoshka_dim if set."""
    if matryoshka_dim is not None:
        return matryoshka_dim
    if _is_codet5p_embedding(model):
        # Internal T5 hidden_size (768) != projected output (256). Probe to get real dim.
        device = next(model.parameters()).device
        dummy = torch.zeros(1, 2, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=dummy)
        return out.shape[-1]
    cfg = model.config
    if _is_t5_like(model):
        return getattr(cfg, "d_model", cfg.hidden_size)
    return cfg.hidden_size


def lm_pool(
    model,
    is_enc_dec: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    matryoshka_dim: int | None = None,
) -> torch.Tensor:
    """
    Extract fixed-size LM representation.
    T5 enc-dec / enc-only: mean-pool over encoder output.
    Decoder-only (Qwen2 etc.): last non-padding token.
    BERT-family: CLS token (position 0).
    Truncates to matryoshka_dim if set.
    """
    if _is_codet5p_embedding(model):
        # codet5p-110m-embedding uses T5 attention internally — relative-position
        # bias overflows in fp16/bf16 under AMP → NaN. Force float32.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            emb = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = emb.float()
    elif _is_t5_like(model):
        enc = model.encoder if is_enc_dec else model
        # T5 relative-position bias overflows in fp16/bf16 → NaN loss under AMP.
        # Force float32 for just the encoder forward regardless of outer autocast.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            out = enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hs = out.last_hidden_state.float()  # [B, seq, d_model]
        mask = (
            attention_mask.unsqueeze(-1).float()
            if attention_mask is not None
            else torch.ones(*input_ids.shape, 1, device=input_ids.device)
        )
        emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif _is_decoder_only(model):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state  # [B, seq, hidden]
        if attention_mask is not None:
            left_pad = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_pad:
                emb = hs[:, -1]
            else:
                last_idx = attention_mask.sum(dim=1) - 1  # [B]
                emb = hs[torch.arange(hs.size(0), device=hs.device), last_idx]
        else:
            emb = hs[:, -1]
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state[:, 0]

    if matryoshka_dim is not None:
        emb = emb[:, :matryoshka_dim]
    return emb


def lm_pool_windowed(
    model,
    is_enc_dec: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    chunk_size: int = 512,
    stride: int = 256,
    matryoshka_dim: int | None = None,
) -> torch.Tensor:
    """
    Sliding-window LM encoding for sequences longer than chunk_size.

    The sequence is split into overlapping windows of `chunk_size` tokens,
    stepping by `stride` tokens each time (overlap = chunk_size - stride).
    Each window is encoded independently with lm_pool(), then all window
    embeddings are mean-pooled into a single [B, hidden] vector.

    Falls back to a single lm_pool() call when the sequence fits in one window,
    so there is no overhead for short functions.

    Parameters
    ----------
    chunk_size : int
        Tokens per window. Must be ≤ model's trained max length (e.g. 512).
    stride : int
        Step between window starts. stride < chunk_size → overlapping windows.
        stride == chunk_size → non-overlapping (faster, less context sharing).
        Recommended: stride = chunk_size // 2 for 50% overlap.
    """
    B, L = input_ids.shape

    # Fast path — sequence fits in a single window
    if L <= chunk_size:
        return lm_pool(model, is_enc_dec, input_ids, attention_mask, matryoshka_dim)

    # Clamp stride to [1, chunk_size] to avoid infinite loops or no-ops
    stride = max(1, min(stride, chunk_size))

    chunk_embs: list[torch.Tensor] = []   # [B, hidden] per window
    valid_counts: list[torch.Tensor] = [] # [B] float — 1.0 if sample has real tokens in window

    start = 0
    while start < L:
        end = min(start + chunk_size, L)
        ids_chunk  = input_ids[:, start:end]
        mask_chunk = attention_mask[:, start:end] if attention_mask is not None else None

        # Per-sample: which samples have at least one real token in this window?
        if mask_chunk is not None:
            per_sample_valid = (mask_chunk.sum(dim=1) > 0).float()  # [B]
        else:
            per_sample_valid = torch.ones(B, device=input_ids.device)

        # Skip window entirely only when NO sample has real tokens
        if per_sample_valid.sum() == 0:
            if end == L:
                break
            start += stride
            continue

        emb = lm_pool(model, is_enc_dec, ids_chunk, mask_chunk, matryoshka_dim)  # [B, hidden]
        # Zero out embedding for samples that have no real tokens in this window
        emb = emb * per_sample_valid.unsqueeze(-1)
        chunk_embs.append(emb)
        valid_counts.append(per_sample_valid)

        if end == L:
            break
        start += stride

    # Fall back to first-window single pass if every window was all-padding (degenerate)
    if not chunk_embs:
        ids_fb = input_ids[:, :chunk_size]
        mask_fb = attention_mask[:, :chunk_size] if attention_mask is not None else None
        return lm_pool(model, is_enc_dec, ids_fb, mask_fb, matryoshka_dim)

    # Per-sample weighted mean: divide by number of valid windows per sample
    embs  = torch.stack(chunk_embs, dim=1)        # [B, n_windows, hidden]
    counts = torch.stack(valid_counts, dim=1)      # [B, n_windows]
    sum_embs = embs.sum(dim=1)                     # [B, hidden]
    count    = counts.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    return sum_embs / count
