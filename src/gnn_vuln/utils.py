"""Utility helpers shared across the project."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    """Configure loguru to write to console and a rotating file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "run_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(model: torch.nn.Module, path: str | Path, **extra) -> None:
    """Save model weights + optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state_dict": model.state_dict(), **extra}
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> dict:
    """Load model weights from a checkpoint file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Checkpoint loaded ← {path}")
    return {k: v for k, v in state.items() if k != "model_state_dict"}


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def get_device(requested: str = "cpu") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if requested == "cuda":
            logger.warning("CUDA requested but not available — falling back to CPU.")
        else:
            logger.info("Using CPU.")
    return device
