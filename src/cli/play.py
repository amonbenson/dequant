from pathlib import Path
from pretty_midi import PrettyMIDI
import numpy as np
import simpleaudio as sa
from ..data.datasets.hov_dataset import HOVDataset, HOVDatasetConfig
from .common import logger, create_converter


try:
    import fluidsynth  # noqa: F401
except ImportError as e:
    if e.msg == "Couldn't find the FluidSynth library.":
        logger.warning("Fluidsynth library was not found. You will not be able to play midi files.")


def run_play(input: Path, sample: int = 0):
    converter = create_converter()

    # Read input data
    filename = Path(input)
    if not filename.exists():
        raise FileNotFoundError(f"Midi/HOV file {filename} does not exist.")

    if filename.suffix in (".mid", ".midi"):
        # Load a midi file
        data = PrettyMIDI(filename)
    else:
        # Load a dataset directory and play the first sequence element
        dataset = HOVDataset(HOVDatasetConfig(dir=filename))
        data = dataset[sample]

    # Convert hov to midi object if we got a matrix
    if not isinstance(data, PrettyMIDI):
        hov = np.array(data)
        midi = converter.hov_to_midi(hov)
    else:
        midi = data

    # Play the file
    audio = midi.fluidsynth(fs=44100)
    audio = (audio * 32767).astype(np.int16)  # convert to 16-bit PCM

    player = sa.play_buffer(audio, num_channels=1, bytes_per_sample=2, sample_rate=44100)
    player.wait_done()
