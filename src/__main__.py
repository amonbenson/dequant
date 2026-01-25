import logging
from simple_parsing import ArgumentParser
from .config import RootConfig, update_config, CONFIG
from .preprocess import preprocess
from .train import train

logger = logging.getLogger("main")


def main():
    parser = ArgumentParser()

    # Define available commands
    command = parser.add_subparsers(dest="command")
    command.add_parser("preprocess")
    command.add_parser("train")

    parser.add_arguments(RootConfig, dest="config")
    args = parser.parse_args()

    # Apply the command line configuration
    update_config(args.config)

    # Run the selected action
    match args.command:
        case "preprocess":
            preprocess()
        case "train":
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
