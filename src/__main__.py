import logging
from dataclasses import dataclass
from typing import Annotated, Union
import tyro
from .config import RootConfig, update_config, CONFIG
from .preprocess import preprocess
from .train import train

logger = logging.getLogger("main")


@dataclass
class TrainCommand:
    pass


@dataclass
class PreprocessCommand:
    pass


@dataclass
class Args:
    config: RootConfig
    command: Union[
        Annotated[PreprocessCommand, tyro.conf.subcommand("preprocess", prefix_name="")],
        Annotated[TrainCommand, tyro.conf.subcommand("train", prefix_name="")],
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
        case _:
            logger.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
