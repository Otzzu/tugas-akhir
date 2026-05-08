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
    """Save model weights + optional metadata (no optimizer — for best-model checkpoints)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **extra}, path)
    logger.info(f"Checkpoint saved → {path}")


def save_resume_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
    **extra,
) -> None:
    """
    Save full training state so that training can be resumed exactly.
    Writes to a temp file first then renames, so a crash mid-write never
    corrupts the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            **extra,
        },
        tmp,
    )
    tmp.replace(path)
    logger.debug(f"Resume checkpoint saved → {path}")


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> dict:
    """Load model weights from a best-model checkpoint. Returns metadata dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Checkpoint loaded ← {path}")
    return {k: v for k, v in state.items() if k != "model_state_dict"}


def load_resume_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cpu",
) -> dict:
    """
    Restore full training state from a resume checkpoint.
    Returns metadata dict with epoch, best_val_loss, patience_counter, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    meta = {k: v for k, v in state.items()
            if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")}
    best_display = meta.get("best_val_f1", meta.get("best_val_loss", float("nan")))
    best_key = "best_val_f1" if "best_val_f1" in meta else "best_val_loss"
    logger.info(
        f"Resumed from {path}  "
        f"(epoch {meta['epoch']}, {best_key}={best_display:.4f})"
    )
    return meta


# ---------------------------------------------------------------------------
# Checkpoint manager (OOP wrapper)
# ---------------------------------------------------------------------------


class CheckpointManager:
    """
    Manages best + resume checkpoints for a single training run.

    Usage
    -----
        cm = CheckpointManager(run_dir, arch="lmgat_codebert")
        cm.save_best(model, epoch=10, val_f1=0.72)
        cm.save_last(model, optimizer, scheduler, epoch=10, ...)
        meta = cm.load_resume(model, optimizer, scheduler, device="cuda")
    """

    def __init__(self, run_dir: str | Path, arch: str) -> None:
        self.run_dir = Path(run_dir)
        self.arch    = arch
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> Path:
        return self.run_dir / f"best_{self.arch}.pt"

    @property
    def last_path(self) -> Path:
        return self.run_dir / f"last_{self.arch}.pt"

    def save_best(self, model: torch.nn.Module, **meta) -> None:
        save_checkpoint(model, self.best_path, **meta)

    def save_last(self, model, optimizer, scheduler, epoch: int,
                  best_val_loss: float, patience_counter: int, **extra) -> None:
        save_resume_checkpoint(self.last_path, model, optimizer, scheduler,
                               epoch=epoch, best_val_loss=best_val_loss,
                               patience_counter=patience_counter, **extra)

    def load_model(self, model: torch.nn.Module, device: str = "cpu") -> dict:
        return load_checkpoint(model, self.best_path, device)

    def load_resume(self, model, optimizer, scheduler, device: str = "cpu") -> dict:
        return load_resume_checkpoint(self.last_path, model, optimizer, scheduler, device)

    def has_resume(self) -> bool:
        return self.last_path.exists()


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
