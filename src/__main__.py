import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional, Union

import tyro

from . import cli
from .config import RootConfig, update_config

logger = logging.getLogger("main")


@dataclass
class PreprocessCommand:
    """Download datasets and convert them to HOV representation."""


@dataclass
class TrainCommand:
    """Train the model."""


@dataclass
class PlayCommand:
    """Play back a midi file or a dataset sample."""

    input: Annotated[Path, tyro.conf.Positional]
    sample: Annotated[Optional[int], tyro.conf.arg(prefix_name=False)] = 0


@dataclass
class QuantizeCommand:
    """Use the HOV representation to quantize a midi file."""

    input: Annotated[Path, tyro.conf.Positional]
    output: Annotated[Path, tyro.conf.Positional]


@dataclass
class DequantizeCommand:
    """Use a trained model to dequantize a midi file"""

    input: Annotated[Path, tyro.conf.Positional]
    output: Annotated[Path, tyro.conf.Positional]
    checkpoint: Annotated[Path, tyro.conf.Positional]


@dataclass
class AppCommand:
    pass


@dataclass
class Args:
    config: RootConfig
    command: Union[
        Annotated[PreprocessCommand, tyro.conf.subcommand("preprocess", prefix_name=False)],
        Annotated[TrainCommand, tyro.conf.subcommand("train", prefix_name=False)],
        Annotated[PlayCommand, tyro.conf.subcommand("play", prefix_name=False)],
        Annotated[QuantizeCommand, tyro.conf.subcommand("quantize", prefix_name=False)],
        Annotated[DequantizeCommand, tyro.conf.subcommand("dequantize", prefix_name=False)],
        Annotated[AppCommand, tyro.conf.subcommand("app", prefix_name=False)],
    ]


def main():
    args = tyro.cli(Args)

    # Apply the command line configuration
    update_config(args.config)

    # Run the selected action
    match args.command:
        case PreprocessCommand():
            cli.run_preprocess()
        case TrainCommand():
            cli.run_train()
        case PlayCommand():
            cli.run_play(
                args.command.input,
                args.command.sample,
            )
        case QuantizeCommand():
            cli.run_quantize(
                args.command.input,
                args.command.output,
            )
        case DequantizeCommand():
            cli.run_dequantize(
                args.command.input,
                args.command.output,
                args.command.checkpoint,
            )
        case AppCommand():
            cli.run_app()
        case _:
            logger.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
