import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union
import tyro
from pretty_midi import PrettyMIDI
import torch
from .config import RootConfig, update_config, CONFIG
from .preprocess import preprocess
from .train import train
from .hov import HOVConverter, HOVConverterConfig
from .predict import Predictor

logger = logging.getLogger("main")

try:
    import fluidsynth
except ImportError as e:
    if e.msg == "Couldn't find the FluidSynth library.":
        logger.warning("Fluidsynth library was not found. You will not be able to play midi files.")


@dataclass
class PreprocessCommand:
    """Download datasets and convert them to HOV representation."""


@dataclass
class TrainCommand:
    "Train the model."


@dataclass
class PlayCommand:
    """Play back a midi file or a dataset sample."""

    input: Annotated[Path, tyro.conf.Positional]


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
class Args:
    config: RootConfig
    command: Union[
        Annotated[PreprocessCommand, tyro.conf.subcommand("preprocess", prefix_name="")],
        Annotated[TrainCommand, tyro.conf.subcommand("train", prefix_name="")],
        Annotated[PlayCommand, tyro.conf.subcommand("play", prefix_name="")],
        Annotated[QuantizeCommand, tyro.conf.subcommand("quantize", prefix_name="")],
        Annotated[DequantizeCommand, tyro.conf.subcommand("dequantize", prefix_name="")],
    ]


def main():
    args = tyro.cli(Args)

    # Apply the command line configuration
    update_config(args.config)

    # Create a shared midi converter used by multiple commands
    converter = HOVConverter(
        HOVConverterConfig(
            steps_per_beat=CONFIG.model.drums.steps_per_beat,
            categories=CONFIG.model.drums.categories,
        )
    )

    # Run the selected action
    match args.command.__class__.__name__:
        case "PreprocessCommand":
            preprocess()
        case "TrainCommand":
            if CONFIG.train.sample_stride % CONFIG.model.drums.steps_per_beat == 0:
                logger.warning(
                    f"The parameter data.sample_stride ({CONFIG.data.sample_stride}) is equally divisible by model.drums.steps_per_beat ({CONFIG.model.drums.steps_per_beat}). "
                    + "This will result in poor model performance, because the model will never receive sequences starting at any other beat than 0."
                )

            if CONFIG.train.auto_preprocess:
                preprocess()

            train()
        case "PlayCommand":
            # Play a midi file
            converter.play(args.command.input)
        case "QuantizeCommand":
            # Load the input midi file
            midi = PrettyMIDI(args.command.input)
            tempo_bpm = converter.extract_tempo(midi)

            # Load the midi file
            hov = converter.midi_to_hov(midi, tempo_bpm=tempo_bpm)

            hov[..., 1] = 0  # Remove offset data
            hov[..., 2] = hov[..., 0]  # Set velocity to 100% for each hit

            # Store as a midi file
            output_midi = converter.hov_to_midi(hov, tempo_bpm=tempo_bpm)
            output_midi.write(args.command.output)
        case "DequantizeCommand":
            # Load the input midi file
            midi = PrettyMIDI(args.command.input)
            tempo_bpm = converter.extract_tempo(midi)

            # Load the midi file
            hov = converter.midi_to_hov(midi, tempo_bpm=tempo_bpm)

            # Predict full HOV matrix from only the hits (index 0)
            predictor = Predictor(args.command.checkpoint)
            hov = predictor.predict_sequence(torch.from_numpy(hov[..., 0])).numpy()

            # Store as a midi file
            output_midi = converter.hov_to_midi(hov, tempo_bpm=tempo_bpm)
            output_midi.write(args.command.output)
        case _:
            logger.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
