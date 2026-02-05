from pathlib import Path

from pretty_midi import PrettyMIDI

from .common import create_converter


def run_quantize(input: Path, output: Path):
    converter = create_converter()

    # Load the input midi file
    midi = PrettyMIDI(input)
    tempo_bpm = converter.extract_tempo(midi)

    # Load the midi file
    hov, _ = converter.midi_to_hov(midi, tempo_bpm=round(tempo_bpm))

    hov[..., 1] = 0  # Remove offset data
    hov[..., 2] = hov[..., 0]  # Set velocity to 100% for each hit

    # Store as a midi file
    output_midi = converter.hov_to_midi(hov, tempo_bpm=round(tempo_bpm))
    output_midi.write(output)
