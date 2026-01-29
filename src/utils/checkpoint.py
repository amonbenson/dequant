import logging
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import asdict
from typing import Optional
from ..config import RootConfig

logger = logging.getLogger("checkpoint")


class Checkpoint:
    @staticmethod
    def save(filename: Path, *, config: RootConfig, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module):
        # Store the file in the provided directory
        logger.info(f"Saving checkpoint to {filename} ...")
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(config),
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "scheduler": self.scheduler.state_dict(),
                "loss_fn": loss_fn,
            },
            filename,
        )

    @staticmethod
    def load(filename: Path, *, device: torch.device | str, config: RootConfig, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> tuple[int, nn.Module]:
        logger.info(f"Loading checkpoint from {filename} ...")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)

        # Validate the checkpoint config
        if checkpoint["config"] != asdict(config):
            logger.warning("Loading checkpoint from a different configuration!")

        # Load all weights and data
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        # self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint["epoch"], checkpoint["loss_fn"]
