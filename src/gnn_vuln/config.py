"""
Central configuration for the gnn_vuln project.

Edit configs under configs/<model>/ to override defaults without changing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    splits_dir: Path = DATA_DIR / "splits"
    # Graph building
    max_nodes: int = 500        # drop graphs larger than this
    edge_types: list[str] = field(
        default_factory=lambda: ["AST", "CFG", "PDG", "CDG", "DDG"]
    )
    # Dataset mode: "binary" (benign/vuln) or "multiclass" (per-CWE).
    # Controls which processed cache file is used so both can coexist.
    mode: str = "binary"
    # Source subdirectory under data/raw/ — isolates datasets and appears in
    # the processed .pt filename so bigvul and merged never collide.
    source: str = "bigvul"
    # Optional separate val/test source dirs (e.g. bigvul_val, bigvul_test).
    # When both are set, official splits are used instead of internal 70/15/15.
    # Leave empty for datasets without separate val/test parquets.
    source_val: str = ""
    source_test: str = ""
    # Filter vocab to top-K CWE classes at .pt build time (0 = use all in vocab).
    # Raw data can be generated with --top-cwe 999; this narrows it at processed stage.
    top_cwe: int = 0
    # Explicit CWE whitelist, e.g. ["CWE-119", "CWE-787"]. Unioned with cwe_groups.
    cwe_list: list | None = None
    # Group whitelist, e.g. ["memory_safety", "injection"]. Expanded via CWE_GROUP_MAP.
    cwe_groups: list | None = None
    # Automatically filter to OWASP Top 10 (2025) CWEs (unioned with cwe_list/cwe_groups)
    filter_owasp: bool = False
    # Automatically filter to MITRE Top 25 CWEs (unioned with cwe_list/cwe_groups)
    filter_top25_dangerous: bool = False
    # Max graphs per class/CWE/group bucket during .pt build (0 = no limit).
    max_per_class: int = 0
    # Random seed for max_per_class sampling. Change for a different sample.
    resample_seed: int = 42


@dataclass
class ModelConfig:
    architecture: str = "lmgat_codebert"  # lmgat_codebert | lmgat_codebert_mtl | ...
    # GNN encoder backbone (lmgat_codebert). Replaces the old standalone
    # lmgcn / lmgin / lmrgcn / lmrgcn_codebert / lmggnn architectures.
    #   gat  — GATv2Conv, edge-feature attention (uses heads, edge_dim)
    #   gcn  — GCNConv, edge-agnostic
    #   gin  — GINEConv, per-layer edge projection (uses edge_dim)
    #   rgcn — RGCNConv, one weight per edge type (uses num_relations, num_bases)
    #   ggnn — GatedGraphConv, edge-agnostic
    gnn_model: str = "gat"
    pretrained_lm: str = "microsoft/codebert-base"  # HuggingFace model ID for node embeddings (frozen)
    func_lm: str = ""               # live LM for function branch; if empty falls back to pretrained_lm
    add_func_tokens: bool = False   # tokenize full function text → stored in Data for live LM
    func_lm_source: str = "raw"    # source for func_lm text: "raw" | "normalized" (cpg reconstruct kept for compat only)
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    # Graph-level pooling for the function classification representation.
    #   mean      — global mean pool over nodes (default)
    #   meanmax   — 0.8*max + 0.6*mean (parameter-free, peak + context)
    #   attention — gated attention pool: per-node score → softmax → weighted sum
    #   dualflow  — focal (per-node suspicion-weighted pool) + context (mean);
    #               single-encoder suspicion head, no two-stage GNN
    # Support: lmgat_codebert
    graph_pool: str = "mean"
    heads: int = 4              # number of GAT attention heads (gnn_model=gat)
    edge_dim: int = 7          # edge feature dimension (gnn_model=gat injects into GATv2 attention; gin projects it)
    num_relations: int = 7     # RGCN: number of edge-type relations (gnn_model=rgcn)
    num_bases: int | None = None  # RGCN: basis-decomposition count, None = no decomposition (gnn_model=rgcn)
    num_classes: int = 2        # 2 = binary; set higher for multi-class
    # Statement-level MIL head
    mil_weight: float = 0.5     # λ: weight of stmt MIL loss vs function loss
    mil_k: int = 3              # top-k statements used for pseudo-label assignment
    rank_loss_weight: float = 0.0  # pairwise ranking loss weight (0 = disabled)
    # Sliding-window encoding for long functions (live LM branch only).
    # func_chunk_size: tokens per window; should match the model's trained max length
    #   (512 for UniXcoder/CodeBERT, 512 for codet5p-110m-embedding, 1024 for codet5p-220m).
    # func_chunk_stride: step between windows (< chunk_size → overlapping windows).
    #   0 = disabled (single forward pass, truncates at func_max_length as before).
    #   Recommended: chunk_size // 2 for 50% overlap, chunk_size for non-overlapping.
    func_chunk_size: int = 0    # 0 = disabled
    func_chunk_stride: int = 0  # 0 = defaults to chunk_size // 2 when chunking is enabled
    # Max token length stored per function in the .pt cache.
    # When func_chunk_size > 0, set this to func_chunk_size * N_chunks you want to cover.
    # E.g. func_chunk_size=512, func_max_length=2048 → up to 4 windows per function.
    func_max_length: int = 512  # default matches model trained length
    # Live LM branch mode (lmgat_codebert).
    #   none          — no live LM. GNN-only, fused = h_graph (no LM concat).
    #                   localization_encoder MUST be "gnn" (lm/both need LM).
    #                   Replaces the old standalone "lmgat" architecture.
    #   func          — function-level forward (single [CLS] over full function,
    #                   sliding window via func_chunk_size if set). Default.
    #   func_and_line — func-level [CLS] for classification + per-line LM forward
    #                   for localization. EDAT-style line isolation: each source
    #                   line forwarded through LM independently → per-line [CLS]
    #                   used as synthetic hidden for localization. Pair with
    #                   mmoe_loc_transformer to recover cross-line context.
    live_lm: str = "func"
    # ── Bidirectional cross-task (Phase 2, lmgat_codebert) ────────────────────
    # Makes localization (stmt_head) and classification (func_head) inform each
    # other.
    #   none            — independent heads (Phase 1 baseline)
    #   cross_attention — Q from one task, K/V from the other task's encoder
    #                     units (decoder-style cross-attention)
    #   self_attention  — EDAT-style: self-attention over a task's own encoder
    #                     units, query biased by the other task's signal
    #   mmoe            — Multi-gate Mixture-of-Experts (Ma et al. 2018): shared
    #                     expert pool + per-task gates (EDAT's released code)
    cross_task_method: str = "none"
    # MMOE only: replace the single Linear task projections with a per-task MLP
    # encoder (Linear→LN→ReLU→Dropout→Linear) — EDAT's TaskSpecificEncoder, light
    # variant. Gives each task a private adapter before the shared experts.
    # General encoder: cls + loc both use MLP when true; loc can be overridden
    # by mmoe_loc_transformer (line-level transformer for localization).
    mmoe_task_encoder: bool = False
    # MMOE only: override the localization task encoder with a transformer over
    # per-statement features (EDAT line_level_encoder pattern). Statements
    # within each graph self-attend → cross-statement context recovered at line
    # granularity. Useful when per-line LM embedding is used (lines isolated by
    # per-line CodeBERT need a downstream encoder to recover cross-line
    # context). Cls path always uses the general encoder.
    mmoe_loc_transformer: bool = False
    # Cross-task fusion mode (ablation):
    #   true  — gated residual side-branch: fused_mod = fused + γ·cross
    #           (γ zero-init, baseline-safe); func_head stays the fat MLP.
    #   false — in-path replace: fused_mod = cross (no residual, no gate);
    #           func_head simplified to LayerNorm+Linear (EDAT-style thin head).
    cross_task_residual: bool = True
    # ── Statement localization "both" mode ────────────────────────────────────
    # Only used when localization_encoder="both". Controls how GNN + LM features combine.
    #   concat   — torch.cat([gnn, lm]) (legacy, LM dim dominates GNN by 3:1 on UniXcoder)
    #   weighted — (1-α)*gnn + α*lm_proj, α fixed by stmt_lm_alpha
    #   gated    — per-statement learnable gate σ(W·[gnn;lm_proj]), no manual α
    stmt_both_mode: str = "concat"
    stmt_lm_alpha: float = 0.5   # only for stmt_both_mode="weighted"


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    lm_lr: float = 2e-5         # CodeBERT learning rate for lmgat_ft / lmgat_mc
    warmup_ratio: float = 0.0   # fraction of total steps for linear warmup (0 = disabled)
    grad_clip: float = 0.0      # gradient clipping max norm (0 = disabled)
    weight_decay: float = 1e-4
    patience: int = 10          # early stopping patience
    early_stop_metric: str = "f1"  # "f1" (macro, maximize) or "loss" (minimize)
    checkpoint_dir: Path = CHECKPOINT_DIR
    results_dir: Path = RESULTS_DIR
    log_dir: Path = LOG_DIR
    device: str = "cpu"         # set to "cuda" if GPU available
    use_class_weights: bool = True  # inverse-frequency weighting for imbalanced classes
    focal_loss_gamma: float = 0.0  # focal loss gamma; 0 = standard CE, 2.0 recommended for imbalanced
    # Epoch-adaptive inverse-frequency class weights (NOT the real LIVABLE paper loss).
    # Ramps class weights from uniform → inverse-frequency over training epochs.
    # Use this as a simple baseline rebalancing strategy.
    epoch_adaptive_weights: bool = False
    # Real LIVABLE two-branch loss (arXiv:2306.06935, Eq. 11-12).
    # L = T * focal + (1-T) * label_smooth_CE, T = 1 - (epoch/max_epoch)^2
    # focal_gamma and label_smoothing are reused for the two branches.
    # Mutually exclusive with epoch_adaptive_weights — use one or the other.
    livable_loss: bool = False
    # Label smoothing for cross-entropy loss (0.0 = disabled, 0.1 recommended).
    # Prevents overconfidence by replacing hard one-hot targets with soft targets.
    # Helps reduce the loss-F1 gap on imbalanced multiclass (see LOSS_F1_GAP.md §3.2).
    label_smoothing: float = 0.0
    # LR scheduler type: "plateau" (ReduceLROnPlateau, default) or "cosine" (CosineAnnealingLR).
    # "cosine" smoothly decays LR to 0 over all epochs — reduces overfitting in later epochs.
    # "plateau" reduces LR when val_loss stops improving (legacy behavior).
    lr_scheduler: str = "plateau"
    # Bit-exact determinism across runs (CUDA atomics, FlashAttention-2 backward, cuBLAS).
    # Enable only for replication studies — costs 20-40% training speed.
    # For ablation statistical comparisons, prefer multi-seed runs with this flag off.
    deterministic: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file, merging with defaults."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        if "data" in raw:
            for k, v in raw["data"].items():
                setattr(cfg.data, k, v)
        if "model" in raw:
            for k, v in raw["model"].items():
                setattr(cfg.model, k, v)
        if "train" in raw:
            for k, v in raw["train"].items():
                setattr(cfg.train, k, v)
        return cfg


def load_default_config() -> Config:
    default_yaml = PROJECT_ROOT / "configs" / "lmgcn" / "binary.yaml"
    if default_yaml.exists():
        return Config.from_yaml(default_yaml)
    return Config()
