import pytest
import numpy as np
from dataclasses import dataclass
from pretty_midi import PrettyMIDI, Instrument, Note
from src.preprocess.midi_to_hov import read_midi, MidiConfig, DrumCategory


REDUCED_DRUM_CATEGORIES = [
    DrumCategory("Kick", (35, 36)),
    DrumCategory("Snare", (37,)),
    DrumCategory("Hi-Hat", (42,)),
]


@dataclass
class DrumHit:
    time: float
    drum: str
    velocity: float = 127


def create_sequence(hits: list[DrumHit], bpm: int = 120):
    # create a midi sequence with a single drum track
    sequence = PrettyMIDI()
    track = Instrument(program=0, is_drum=True)

    # append all drum hits
    track.notes.extend([Note(h.velocity, h.drum, h.time, h.time + 0.1) for h in hits])

    sequence.instruments.append(track)
    return sequence


def test_single_kick():
    sequence = create_sequence([DrumHit(0.0, 36)])
    hov = read_midi(sequence, 120, MidiConfig(categories=REDUCED_DRUM_CATEGORIES))

    # pattern should get extended to 16 steps, with 3 instruments each
    assert hov.shape == (3, 16, 3)

    # check the single hit
    assert hov[0, 0, 0] == pytest.approx(1.0)
    assert hov[1, 0, 0] == pytest.approx(0.0)
    assert hov[2, 0, 0] == pytest.approx(1.0)

    # make sure no other bits are set
    hov[:, 0, 0] = 0.0
    assert np.all(hov == 0)


def test_pattern_length():
    seq1 = create_sequence([DrumHit(0 / 8, 36)])
    seq2 = create_sequence([DrumHit(15 / 8, 36)])
    seq3 = create_sequence([DrumHit(16 / 8, 36)])
    seq4 = create_sequence([DrumHit(17 / 8, 36)])

    hov1 = read_midi(seq1, 120, MidiConfig(categories=REDUCED_DRUM_CATEGORIES))
    hov2 = read_midi(seq2, 120, MidiConfig(categories=REDUCED_DRUM_CATEGORIES))
    hov3 = read_midi(seq3, 120, MidiConfig(categories=REDUCED_DRUM_CATEGORIES))
    hov4 = read_midi(seq4, 120, MidiConfig(categories=REDUCED_DRUM_CATEGORIES))

    assert hov1.shape[1] == 16
    assert hov2.shape[1] == 16
    assert hov3.shape[1] == 32
    assert hov4.shape[1] == 32


def test_hits():
    sequence = create_sequence([
        DrumHit(0 / 8, 35),  # Kick
        DrumHit(0 / 8, 36),  # Kick
        DrumHit(3 / 8, 37),  # Snare
        DrumHit(4 / 8, 37),  # Snare
        DrumHit(13 / 8, 42),  # Hi-Hat
        DrumHit(15 / 8, 42),  # Hi-Hat
    ])
    hov = read_midi(sequence, 120, MidiConfig(categories=REDUCED_DRUM_CATEGORIES))

    # Should still be 16 steps
    assert hov.shape == (3, 16, 3)

    # Check locations of hits for each instrument
    assert np.flatnonzero(hov[0, :, 0]).tolist() == [0]  # Kick
    assert np.flatnonzero(hov[0, :, 1]).tolist() == [3, 4]  # Snare
    assert np.flatnonzero(hov[0, :, 2]).tolist() == [13, 15]  # Hi-Hat


def test_velocities():
    sequence = create_sequence([
        DrumHit(0 / 8, 35, 64),  # Kick
        DrumHit(4 / 8, 37, 100),  # Snare
        DrumHit(5 / 8, 42, 1),  # Hi-Hat
        DrumHit(6 / 8, 42, 20),  # Hi-Hat
        # Velocity 0, should be ignored:
        DrumHit(7 / 8, 42, 0),
        # Two simultaneous kicks, the louder one should win:
        DrumHit(10 / 8, 36, 100),  # Kick
        DrumHit(10 / 8, 35, 64),  # Kick
    ])
    hov = read_midi(sequence, 120, MidiConfig(categories=REDUCED_DRUM_CATEGORIES))

    # Check velocities for each instrument
    # fmt: off
    assert hov[2, :, 0].tolist() == pytest.approx([64/127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100/127, 0, 0, 0, 0, 0])
    assert hov[2, :, 1].tolist() == pytest.approx([0, 0, 0, 0, 100/127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert hov[2, :, 2].tolist() == pytest.approx([0, 0, 0, 0, 0, 1/127, 20/127, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # fmt: on

    # Make sure that velocity 0 results in no hit
    assert hov[0, 7, 2] == 0
