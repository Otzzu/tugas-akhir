"""Abstract base class for all vulnerability detectors."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from gnn_vuln.models._lm_utils import lm_hidden_dim, lm_pool, lm_pool_windowed


class VulnDetectorBase(nn.Module):
    """
    Shared logic for all vulnerability detectors:
      - Live LM branch setup (_build_lm_branch)
      - LM embedding helper (_lm_embed)
      - Optimizer param grouping (lm_parameters / has_live_lm)
      - Config-driven construction classmethod (from_config)

    Subclasses implement forward(). Statement scoring is handled by
    StmtHead / MulticlassStmtHead in heads.py.
    """

    # ── LM branch (optional) ─────────────────────────────────────────────────
    # Populated by _build_lm_branch(); absent on frozen-embedding models.
    codebert: nn.Module
    _lm_dim: int
    _is_enc_dec: bool
    _matryoshka_dim: int | None
    _func_chunk_size: int   # 0 = disabled (single forward pass)
    _func_chunk_stride: int

    # ── LM branch helpers ─────────────────────────────────────────────────────

    def _build_lm_branch(
        self,
        pretrained_lm: str,
        func_lm: str,
        matryoshka_dim: int | None = None,
        func_chunk_size: int = 0,
        func_chunk_stride: int = 0,
    ) -> None:
        """
        Load a live LM and store as self.codebert.
        Call once from subclass __init__ when a live LM is needed.

        Parameters
        ----------
        func_chunk_size : int
            Sliding-window chunk size in tokens. 0 = disabled (single forward pass).
            Should match the model's trained max length (e.g. 512 for UniXcoder).
        func_chunk_stride : int
            Step between windows. 0 = defaults to chunk_size // 2 (50% overlap).
            Only used when func_chunk_size > 0.
        """
        _func_lm = func_lm if func_lm else pretrained_lm
        _cfg = AutoConfig.from_pretrained(_func_lm, trust_remote_code=True)
        if not hasattr(_cfg, "is_decoder"):
            _cfg.is_decoder = False
        self.codebert = AutoModel.from_pretrained(
            _func_lm, config=_cfg, trust_remote_code=True
        )
        if hasattr(self.codebert, "gradient_checkpointing_enable"):
            self.codebert.gradient_checkpointing_enable()
        self._lm_dim = lm_hidden_dim(self.codebert, matryoshka_dim)
        self._is_enc_dec = getattr(
            self.codebert.config, "is_encoder_decoder", False
        )
        self._matryoshka_dim = matryoshka_dim
        self._func_chunk_size = func_chunk_size
        # Default stride to chunk_size // 2 (50% overlap) when not explicitly set
        self._func_chunk_stride = func_chunk_stride if func_chunk_stride > 0 else max(1, func_chunk_size // 2)

    def _lm_embed(
        self,
        func_input_ids: torch.Tensor | None,
        func_attention_mask: torch.Tensor | None,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return LM embedding [B, lm_dim], or zeros when no input provided.

        When func_chunk_size > 0, uses sliding-window encoding so sequences
        longer than chunk_size are split into overlapping windows and
        mean-pooled, keeping each window within the model's trained length.
        """
        if func_input_ids is None:
            return torch.zeros(B, self._lm_dim, device=device)

        if self._func_chunk_size > 0:
            return lm_pool_windowed(
                self.codebert,
                self._is_enc_dec,
                func_input_ids,
                func_attention_mask,
                chunk_size=self._func_chunk_size,
                stride=self._func_chunk_stride,
                matryoshka_dim=self._matryoshka_dim,
            )

        return lm_pool(
            self.codebert,
            self._is_enc_dec,
            func_input_ids,
            func_attention_mask,
            matryoshka_dim=self._matryoshka_dim,
        )

    # ── Optimizer helpers ─────────────────────────────────────────────────────

    def lm_parameters(self) -> list[nn.Parameter]:
        """Return LM parameters for a separate optimizer param group.
        Returns empty list for frozen-embedding models (no live LM)."""
        if hasattr(self, "codebert"):
            return list(self.codebert.parameters())
        return []

    def has_live_lm(self) -> bool:
        """True when the model has a live fine-tunable LM branch."""
        return hasattr(self, "codebert")

    # ── Config constructor ────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg, in_channels: int, **kwargs):
        """
        Build model from a Config object.
        Must be implemented by every subclass.
        """
        raise NotImplementedError(
            f"{cls.__name__}.from_config() not implemented."
        )

    # ── Forward (abstract) ────────────────────────────────────────────────────

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        func_input_ids: torch.Tensor | None = None,
        func_attention_mask: torch.Tensor | None = None,
    ):
        ...
