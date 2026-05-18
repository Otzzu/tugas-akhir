"""Abstract base class for all vulnerability detectors."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from gnn_vuln.models._lm_utils import (
    lm_hidden_dim, lm_pool, lm_pool_windowed, lm_full_windowed, lm_per_line_embed,
    lm_per_line_raw, lm_full_codet5p, lm_full_codet5p_raw, _is_codet5p_embedding,
)


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
        use_flash_attention: bool = False,
        compile_lm: bool = False,
        use_grad_checkpoint: bool = True,
        lm_per_line: bool = False,
        codet5p_raw_encoder: bool = False,
        codet5p_normalize_per_token: bool = False,
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
        use_flash_attention : bool
            Load the LM with flash_attention_2 if available. Requires flash-attn package.
        """
        _func_lm = func_lm if func_lm else pretrained_lm
        _cfg = AutoConfig.from_pretrained(_func_lm, trust_remote_code=True)
        if not hasattr(_cfg, "is_decoder"):
            _cfg.is_decoder = False
        load_kwargs: dict = {"config": _cfg, "trust_remote_code": True}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
                load_kwargs["torch_dtype"] = torch.bfloat16
            except ImportError:
                pass  # flash-attn not installed — fall back silently
        self.codebert = AutoModel.from_pretrained(_func_lm, **load_kwargs)
        if use_grad_checkpoint and hasattr(self.codebert, "gradient_checkpointing_enable"):
            self.codebert.config.use_cache = False
            self.codebert.gradient_checkpointing_enable()
        if compile_lm:
            try:
                self.codebert = torch.compile(self.codebert, mode="reduce-overhead", dynamic=False)
            except Exception:
                pass  # unsupported platform or torch version — skip silently
        self._codet5p_raw = codet5p_raw_encoder and _is_codet5p_embedding(self.codebert)
        self._codet5p_norm_per_token = codet5p_normalize_per_token
        if self._codet5p_raw:
            d = getattr(self.codebert.config, "d_model", 768)
            self._lm_dim = min(d, matryoshka_dim) if matryoshka_dim else d
        else:
            self._lm_dim = lm_hidden_dim(self.codebert, matryoshka_dim)
        self._is_enc_dec = getattr(
            self.codebert.config, "is_encoder_decoder", False
        )
        self._matryoshka_dim = matryoshka_dim
        self._func_chunk_size = func_chunk_size
        # Default stride to chunk_size // 2 (50% overlap) when not explicitly set
        self._func_chunk_stride = func_chunk_stride if func_chunk_stride > 0 else max(1, func_chunk_size // 2)
        self._lm_per_line = lm_per_line

    def _lm_embed_full(
        self,
        func_input_ids: torch.Tensor | None,
        func_attention_mask: torch.Tensor | None,
        B: int,
        device: torch.device,
        func_token_lines: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (cls_emb [B, lm_dim], last_hidden_state [B, L, lm_dim] or None).
        last_hidden_state is None for non-BERT LMs that don't expose it.

        When func_chunk_size > 0, uses sliding-window encoding via
        lm_full_windowed — per-token hidden states are aligned to original
        input positions, overlap regions averaged. Localization (mode=lm|both)
        gets full per-line LM features for functions longer than chunk_size.

        When lm_per_line is set (and func_token_lines given): function tokens
        are regrouped per source line, each line forwarded independently → the
        returned hidden carries each line's [CLS] at all its token positions
        (EDAT-style line isolation). Classification cls_emb still comes from a
        function-level forward.
        """
        if func_input_ids is None:
            return torch.zeros(B, self._lm_dim, device=device), None
        # CodeT5+ raw: skip proj/L2-norm, use <s> token hidden (d_model dim).
        if self._codet5p_raw:
            return lm_full_codet5p_raw(
                self.codebert, func_input_ids, func_attention_mask, self._matryoshka_dim,
            )
        # CodeT5+ embedding model — public forward is pooled-only. Pull per-token
        # states from the internal T5 encoder so localization=lm|both works.
        if _is_codet5p_embedding(self.codebert):
            return lm_full_codet5p(
                self.codebert, func_input_ids, func_attention_mask, self._matryoshka_dim,
                normalize_per_token=self._codet5p_norm_per_token,
            )
        # Per-line embedding — EDAT-style line isolation (reuses func tokens)
        if self._lm_per_line and func_token_lines is not None:
            try:
                cls_emb = self._lm_embed(func_input_ids, func_attention_mask, B, device)
                synth_hidden = lm_per_line_embed(
                    self.codebert, func_input_ids, func_token_lines,
                )
                if self._matryoshka_dim is not None:
                    synth_hidden = synth_hidden[:, :, :self._matryoshka_dim]
                return cls_emb, synth_hidden
            except (AttributeError, TypeError):
                return self._lm_embed(func_input_ids, func_attention_mask, B, device), None
        # Sliding-window full forward — per-token hidden aligned to input positions
        if self._func_chunk_size > 0:
            try:
                return lm_full_windowed(
                    self.codebert, self._is_enc_dec,
                    func_input_ids, func_attention_mask,
                    chunk_size=self._func_chunk_size,
                    stride=self._func_chunk_stride,
                    matryoshka_dim=self._matryoshka_dim,
                )
            except (AttributeError, TypeError):
                # Non-BERT LM (e.g. CodeT5+) — fall back to pooled CLS-only
                return self._lm_embed(func_input_ids, func_attention_mask, B, device), None
        try:
            out = self.codebert(
                input_ids=func_input_ids,
                attention_mask=func_attention_mask,
            )
            hidden = out.last_hidden_state  # [B, L, hidden]
            cls = hidden[:, 0]              # [B, hidden] — CLS token
            if self._matryoshka_dim is not None:
                cls    = cls[:, :self._matryoshka_dim]
                hidden = hidden[:, :, :self._matryoshka_dim]
            return cls, hidden
        except (AttributeError, TypeError):
            # CodeT5+ or other models that don't return last_hidden_state in standard form
            return self._lm_embed(func_input_ids, func_attention_mask, B, device), None

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

        if self._codet5p_raw:
            cls, _ = lm_full_codet5p_raw(
                self.codebert, func_input_ids, func_attention_mask, self._matryoshka_dim,
            )
            return cls

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

    def _lm_embed_per_line_raw(
        self,
        func_input_ids: torch.Tensor,
        func_token_lines: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Per-line LM forward → (line_cls [n, lm_dim], uniq_sid [n], B, L).

        Used by live_lm=line: each source line forwarded through the LM
        independently → per-line [CLS]. No whole-function forward. The caller
        (a line-level transformer) recovers cross-line context.
        """
        return lm_per_line_raw(self.codebert, func_input_ids, func_token_lines)

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
        func_token_lines: torch.Tensor | None = None,
    ):
        ...
