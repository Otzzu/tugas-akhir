from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

# CPG node type vocabulary — all Joern/MegaVul node labels + UNKNOWN fallback
NODE_TYPES = [
    "ANNOTATION",
    "BINDING",
    "BLOCK",
    "CALL",
    "CLOSURE_BINDING",
    "COMMENT",
    "CONTROL_STRUCTURE",
    "DEPENDENCY",
    "FIELD_IDENTIFIER",
    "FILE",
    "IDENTIFIER",
    "IMPORT",
    "JUMP_TARGET",
    "LITERAL",
    "LOCAL",
    "MEMBER",
    "META_DATA",
    "METHOD",
    "METHOD_PARAMETER_IN",
    "METHOD_PARAMETER_OUT",
    "METHOD_REF",
    "METHOD_RETURN",
    "MODIFIER",
    "NAMESPACE",
    "NAMESPACE_BLOCK",
    "RETURN",
    "TAG",
    "TYPE",
    "TYPE_DECL",
    "TYPE_REF",
    "UNKNOWN",  # fallback for unseen labels
]
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

# Non-LM feature dimensions (fixed regardless of which LM is used):
#   node_type(22 one-hot) + dist(3) + dangerous_api(1)
#   + is_external(1) + ctrl_struct_type(12 one-hot) + has_type(1) + type_feats(3)
#   + eval_strategy(4 one-hot) + arg_idx(1) + dispatch_type(3 one-hot)
#   + is_variadic(1) + span_normalized(1)
NON_LM_FEAT_DIM = 31 + 3 + 1 + 1 + 14 + 1 + 3 + 4 + 1 + 4 + 1 + 1  # = 65

# Default LM embedding dim (CodeBERT / UniXcoder / GraphCodeBERT)
CODEBERT_DIM = 768

# Default total node feature dim (CodeBERT). Changes with different LMs.
NODE_FEAT_DIM = NON_LM_FEAT_DIM + CODEBERT_DIM  # = 833


def compute_node_feat_dim(lm_dim: int) -> int:
    """Total node feature dimension for a given LM embedding size."""
    return NON_LM_FEAT_DIM + lm_dim


class LMNodeEmbedder:
    """
    Embeds CPG node code strings using a frozen LM CLS token.

    Supports any HuggingFace encoder (CodeBERT, UniXcoder, CodeT5p-embedding,
    Jina with matryoshka_dim, etc.). The actual embedding dimension is resolved
    at init time and exposed as self.lm_dim.

    Each node produces a (NON_LM_FEAT_DIM + lm_dim,) vector:
        [node_type_idx(1)] + [LM CLS(lm_dim)] + [dist(3)] + [danger(1)] + [extra(9)]

    Weights are never updated — call only during dataset preprocessing.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: str = "cpu",
        max_length: int = 128,
        matryoshka_dim: int | None = None,
        use_flash_attention: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Flash Attention 2 — requires flash-attn package + Ampere+ GPU + half precision
        attn_impl = None
        if use_flash_attention and device != "cpu":
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                import logging
                logging.getLogger(__name__).warning(
                    "use_flash_attention=True but flash-attn not installed. "
                    "Falling back to standard attention. Install with: pip install flash-attn"
                )

        load_kwargs: dict = {"trust_remote_code": True}
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.max_length = max_length
        self._use_amp = self.device.type == "cuda"
        self._flash = attn_impl == "flash_attention_2"
        self.matryoshka_dim = matryoshka_dim

        # Resolve actual LM output dimension
        cfg = self.model.config
        full_dim = getattr(cfg, "hidden_size", getattr(cfg, "d_model", CODEBERT_DIM))
        self.lm_dim = matryoshka_dim if matryoshka_dim and matryoshka_dim < full_dim else full_dim
        self.node_feat_dim = compute_node_feat_dim(self.lm_dim)

    @torch.no_grad()
    def embed_batch(self, codes: list[str]) -> torch.Tensor:
        """
        Returns (N, lm_dim) float32 CLS embeddings for a list of code strings.
        Empty / whitespace strings are treated as a single space.
        """
        safe_codes = [c if c.strip() else " " for c in codes]
        inputs = self.tokenizer(
            safe_codes,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        # Flash Attention runs in bfloat16 natively — no separate AMP needed
        amp_enabled = self._use_amp and not self._flash
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0, :].float().cpu()  # (N, full_dim)
        if self.matryoshka_dim and self.matryoshka_dim < cls.shape[1]:
            cls = cls[:, :self.matryoshka_dim]
        return cls  # (N, lm_dim)

    @torch.no_grad()
    def embed(self, code: str) -> torch.Tensor:
        """Returns (lm_dim,) CLS embedding for a single code string."""
        return self.embed_batch([code])[0]
