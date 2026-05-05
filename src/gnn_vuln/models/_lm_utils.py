"""Shared helpers for live LM branches across model architectures."""

from __future__ import annotations

import torch


def lm_hidden_dim(model) -> int:
    """Return hidden size of the LM (d_model for T5, hidden_size for BERT)."""
    cfg = model.config
    if getattr(cfg, "is_encoder_decoder", False):
        return cfg.d_model
    return cfg.hidden_size


def lm_pool(
    model,
    is_enc_dec: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Extract fixed-size LM representation.
    BERT-family: CLS token (position 0).
    T5-family:   masked mean-pool over encoder output (no [CLS]).
    """
    if is_enc_dec:
        out = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state  # [B, seq, d_model]
        mask = (
            attention_mask.unsqueeze(-1).float()
            if attention_mask is not None
            else torch.ones(*input_ids.shape, 1, device=input_ids.device)
        )
        return (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]
