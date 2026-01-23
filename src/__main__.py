import logging
from simple_parsing import ArgumentParser
from .config import MainConfig, update_config
from .preprocess import preprocess
from .train import train

logger = logging.getLogger("main")


def main():
    parser = ArgumentParser()

    # Define available commands
    command = parser.add_subparsers(dest="command")
    command.add_parser("preprocess")
    command.add_parser("train")

    parser.add_arguments(MainConfig, dest="config")
    args = parser.parse_args()

    # Apply the command line configuration
    update_config(args.config)

    # Run the selected action
    match args.command:
        case "preprocess":
            preprocess()
        case "train":
            train()
        case _:
            logger.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
