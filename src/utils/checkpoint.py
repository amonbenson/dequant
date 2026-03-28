import logging
import pickle
import types
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ..config import RootConfig

logger = logging.getLogger("checkpoint")


def _path_as_str(*args) -> str:
    """Deserialise a pickled Path as a plain string, accepting both the parts-tuple
    and single-string pickle formats used by different Python versions."""
    return str(Path(*args))


class _PathCompatUnpickler(pickle.Unpickler):
    """Unpickler that deserialises any pathlib.* type as a plain string,
    ensuring cross-platform and cross-version compatibility."""

    def find_class(self, module, name):
        if module == "pathlib":
            return _path_as_str
        return super().find_class(module, name)


# Pickle-module shim for torch.load that substitutes the custom Unpickler above.
_compat_pickle = types.ModuleType(pickle.__name__)
_compat_pickle.__dict__.update(vars(pickle))
_compat_pickle.Unpickler = _PathCompatUnpickler  # type: ignore[attr-defined]


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
        checkpoint = torch.load(filename, map_location=device, weights_only=False, pickle_module=_compat_pickle)

        # Compare only the model sub-config: it defines the architecture and is the only
        # part relevant to checkpoint compatibility.  data/train config fields are
        # run-specific (paths, hyper-params) and intentionally ignored here.
        if config is not None and checkpoint["config"].get("model") != asdict(config.model):
            logger.warning("Loading checkpoint from a different configuration!")

        # Load all weights and data
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        # Load global_step (if it was stored)
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        return checkpoint["epoch"], checkpoint["loss_fn"], global_step, best_val_loss
