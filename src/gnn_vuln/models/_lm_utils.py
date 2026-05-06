"""Shared helpers for live LM branches across model architectures."""

from __future__ import annotations

import torch


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
        emb = model(input_ids=input_ids, attention_mask=attention_mask)  # [B, 256] tensor
    elif _is_t5_like(model):
        enc = model.encoder if is_enc_dec else model
        out = enc(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state  # [B, seq, d_model]
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
