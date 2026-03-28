import logging
import pathlib
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ..config import RootConfig

logger = logging.getLogger("checkpoint")


class Checkpoint:
    @staticmethod
    def save(
        filename: Path,
        *,
        config: RootConfig,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        best_val_loss: float = float("inf"),
    ):
        # Store the file in the provided directory
        logger.info(f"Saving checkpoint to {filename} ...")
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(config),
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_fn": loss_fn,
                "best_val_loss": best_val_loss,
            },
            filename,
        )

    @staticmethod
    def load(
        filename: Path, *, device: torch.device | str, config: RootConfig | None, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
    ) -> tuple[int, nn.Module, int, float]:
        logger.info(f"Loading checkpoint from {filename} ...")
        # Checkpoints may contain PosixPath or WindowsPath objects pickled on a different OS/Python
        # version. Remap both to the generic Path so pickle can deserialize them cross-platform.
        # Python 3.12 also removed pathlib.Path._flavour; patch it in if missing so older
        # checkpoints that reference it during unpickling don't raise AttributeError.
        _orig_posix = pathlib.PosixPath
        _orig_windows = pathlib.WindowsPath
        pathlib.PosixPath = Path  # type: ignore[misc]
        pathlib.WindowsPath = Path  # type: ignore[misc]
        if not hasattr(pathlib.PurePosixPath, "_flavour"):
            pathlib.PurePosixPath._flavour = object()  # type: ignore[attr-defined]
            pathlib.PureWindowsPath._flavour = object()  # type: ignore[attr-defined]
        try:
            checkpoint = torch.load(filename, map_location=device, weights_only=False)
        finally:
            pathlib.PosixPath = _orig_posix  # type: ignore[misc]
            pathlib.WindowsPath = _orig_windows  # type: ignore[misc]

        # Validate the checkpoint config
        if config is not None and checkpoint["config"] != asdict(config):
            logger.warning("Loading checkpoint from a different configuration!")

        # Load all weights and data
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        # self.scheduler.load_state_dict(checkpoint["scheduler"])

        # Load global_step (if it was stored)
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        return checkpoint["epoch"], checkpoint["loss_fn"], global_step, best_val_loss
