from pathlib import Path
from pretty_midi import PrettyMIDI
import torch
from ..inference.predictor import Predictor
from .common import create_converter


def run_dequantize(input: Path, output: Path, checkpoint: Path):
    converter = create_converter()

    # Load the input midi file
    midi = PrettyMIDI(input)
    tempo_bpm = converter.extract_tempo(midi)

    # Load the midi file
    hov = converter.midi_to_hov(midi, tempo_bpm=tempo_bpm)

    # Predict full HOV matrix from only the hits (index 0)
    predictor = Predictor(checkpoint)
    hov = predictor.predict_sequence(torch.from_numpy(hov[..., 0])).numpy()

    # Store as a midi file
    output_midi = converter.hov_to_midi(hov, tempo_bpm=tempo_bpm)
    output_midi.write(output)
