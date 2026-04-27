from __future__ import annotations

import torch
from transformers import RobertaModel, RobertaTokenizer

# CPG node type vocabulary (Joern labels)
NODE_TYPES = [
    "METHOD",
    "METHOD_PARAMETER_IN",
    "METHOD_PARAMETER_OUT",
    "BLOCK",
    "CALL",
    "IDENTIFIER",
    "LITERAL",
    "RETURN",
    "CONTROL_STRUCTURE",
    "FIELD_IDENTIFIER",
    "METHOD_RETURN",
    "LOCAL",
    "UNKNOWN",
]
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

CODEBERT_DIM = 768
NODE_FEAT_DIM = 1 + CODEBERT_DIM + 3 + 1  # node_type + CodeBERT CLS + distances + dangerous_api = 773


class CodeBERTNodeEmbedder:
    """
    Embeds CPG node code strings using CodeBERT's CLS token (frozen weights).

    Each node produces a (773,) vector:
        [node_type_idx (1)] + [CodeBERT CLS (768)] + [dist_entry, dist_exit, dist_call (3)] + [dangerous_api (1)]

    GPU optimisations applied during embed_batch:
    - max_length=128  (CPG nodes are short statements; 512 wastes GPU on padding)
    - AMP float16     (halves VRAM, enables larger batches on RTX cards)
    - non_blocking transfer (overlaps CPU->GPU copy with GPU compute)

    Weights are never updated — call this only during dataset preprocessing,
    not inside the training loop.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: str = "cpu",
        max_length: int = 128,
    ):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.max_length = max_length
        self._use_amp = self.device.type == "cuda"

    @torch.no_grad()
    def embed_batch(self, codes: list[str]) -> torch.Tensor:
        """
        Returns (N, 768) float32 CLS embeddings for a list of code strings.
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
        # non_blocking overlaps CPU->GPU transfer with GPU compute
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :].float().cpu()  # (N, 768) float32

    @torch.no_grad()
    def embed(self, code: str) -> torch.Tensor:
        """Returns (768,) CLS embedding for a single code string."""
        return self.embed_batch([code])[0]
