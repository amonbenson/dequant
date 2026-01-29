import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union
import tyro
from .config import RootConfig, update_config, CONFIG
from .preprocess import preprocess
from .train import train
from .hov import HOVConverter, HOVConverterConfig

logger = logging.getLogger("main")

try:
    import fluidsynth
except ImportError as e:
    if e.msg == "Couldn't find the FluidSynth library.":
        logger.warning("Fluidsynth library was not found. You will not be able to play midi files.")


@dataclass
class PreprocessCommand:
    pass


@dataclass
class TrainCommand:
    pass


@dataclass
class PlayCommand:
    filename: Union[Annotated[Path, tyro.conf.Positional]]


@dataclass
class Args:
    config: RootConfig
    command: Union[
        Annotated[PreprocessCommand, tyro.conf.subcommand("preprocess", prefix_name="")],
        Annotated[TrainCommand, tyro.conf.subcommand("train", prefix_name="")],
        Annotated[PlayCommand, tyro.conf.subcommand("play", prefix_name="")],
    ]


def main():
    args = tyro.cli(Args)

    # Apply the command line configuration
    update_config(args.config)

    # Run the selected action
    match args.command:
        case PreprocessCommand():
            preprocess()
        case TrainCommand():
            if CONFIG.train.sample_stride % CONFIG.model.drums.steps_per_beat == 0:
                logger.warning(
                    f"The parameter data.sample_stride ({CONFIG.data.sample_stride}) is equally divisible by model.drums.steps_per_beat ({CONFIG.model.drums.steps_per_beat}). "
                    + "This will result in poor model performance, because the model will never receive sequences starting at any other beat than 0."
                )

            if CONFIG.train.auto_preprocess:
                preprocess()

            train()
        case PlayCommand():
            converter = HOVConverter(
                HOVConverterConfig(
                    steps_per_beat=CONFIG.model.drums.steps_per_beat,
                    categories=CONFIG.model.drums.categories,
                )
            )
            converter.play(args.command.filename)
        case _:
            logger.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
