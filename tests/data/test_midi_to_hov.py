from dataclasses import dataclass

import numpy as np
import pytest

#from IPython.lib.pretty import MAX_SEQ_LENGTH
from pretty_midi import Instrument, Note, PrettyMIDI

from src.data import DrumCategory, HOVConverter, HOVConverterConfig

REDUCED_DRUM_CATEGORIES = [
    DrumCategory("Kick", (35, 36)),
    DrumCategory("Snare", (37,)),
    DrumCategory("Hi-Hat", (42,)),
]

MAX_SEQ_LENGTH = 256


@dataclass
class DrumHit:
    time: float
    drum: int
    velocity: float = 127


def create_midi(hits: list[DrumHit], bpm: int = 120) -> PrettyMIDI:
    # create a midi sequence with a single drum track
    sequence = PrettyMIDI()
    track = Instrument(program=0, is_drum=True)

    # append all drum hits
    track.notes.extend([Note(h.velocity, h.drum, h.time, h.time + 0.1) for h in hits])

    sequence.instruments.append(track)
    return sequence


def create_hov(midi: PrettyMIDI, tempo_bpm, config: HOVConverterConfig):
    converter = HOVConverter(config)
    hov, pos = converter.midi_to_hov(midi, tempo_bpm)
    return hov, pos


def test_single_kick():
    sequence = create_midi([DrumHit(0.0, 36)])
    hov, pos = create_hov(sequence, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))

    # a pattern should get extended to 16 steps, with 3 instruments each
    assert hov.shape == (16, 3, 3)

    # check the single hit
    assert hov[0, 0, 0] == pytest.approx(1.0)
    assert hov[0, 0, 1] == pytest.approx(0.0)
    assert hov[0, 0, 2] == pytest.approx(1.0)

    # make sure no other bits are set
    hov[0, 0, :] = 0.0
    assert np.all(hov == 0)
    # alignment with time dimension
    assert pos.shape[0] == hov.shape[0]


def test_pattern_length():
    seq1 = create_midi([DrumHit(0 / 8, 36)])
    seq2 = create_midi([DrumHit(14 / 8, 36)])
    seq3 = create_midi([DrumHit(15 / 8, 36)])
    seq4 = create_midi([DrumHit(16 / 8, 36)])

    hov1, pos1 = create_hov(seq1, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))
    hov2, pos2 = create_hov(seq2, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))
    hov3, pos3 = create_hov(seq3, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))
    hov4, pos4 = create_hov(seq4, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))

    assert len(hov1) == 16
    assert len(hov2) == 16
    assert len(hov3) == 32
    assert len(hov4) == 32


def test_hits():
    sequence = create_midi([
        DrumHit(0 / 8, 35),  # Kick
        DrumHit(0 / 8, 36),  # Kick
        DrumHit(3 / 8, 37),  # Snare
        DrumHit(4 / 8, 37),  # Snare
        DrumHit(12 / 8, 42),  # Hi-Hat
        DrumHit(14 / 8, 42),  # Hi-Hat
    ])
    hov, pos = create_hov(sequence, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))

    # Should still be 16 steps
    assert hov.shape == (16, 3, 3)

    # Check locations of hits for each instrument
    assert np.flatnonzero(hov[:, 0, 0]).tolist() == [0]  # Kick
    assert np.flatnonzero(hov[:, 1, 0]).tolist() == [3, 4]  # Snare
    assert np.flatnonzero(hov[:, 2, 0]).tolist() == [12, 14]  # Hi-Hat


def test_offsets():
    sequence = create_midi([
        DrumHit(2 / 8, 36),
        DrumHit(4.2 / 8, 36),
        DrumHit(6.4 / 8, 36),
        DrumHit(8.6 / 8, 36),
        DrumHit(10.8 / 8, 36),
        DrumHit(13 / 8, 36),
    ])
    hov, pos = create_hov(sequence, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))

    # Check hit positions and corresponding offsets
    # fmt: off
    assert hov[:, 0, 0].tolist() == pytest.approx([0, 0, 1, 0, 1,   0, 1,   0, 0,  1,   0,  1,   0, 1, 0, 0])
    assert hov[:, 0, 1].tolist() == pytest.approx([0, 0, 0, 0, 0.2, 0, 0.4, 0, 0, -0.4, 0, -0.2, 0, 0, 0, 0])
    # fmt: on


def test_velocities():
    sequence = create_midi([
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
    hov, pos = create_hov(sequence, 120, HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES, max_seq_len=MAX_SEQ_LENGTH))

    # Check velocities for each instrument
    # fmt: off
    assert hov[:, 0, 2].tolist() == pytest.approx([64/127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100/127, 0, 0, 0, 0, 0])
    assert hov[:, 1, 2].tolist() == pytest.approx([0, 0, 0, 0, 100/127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert hov[:, 2, 2].tolist() == pytest.approx([0, 0, 0, 0, 0, 1/127, 20/127, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # fmt: on

    # Make sure that velocity 0 results in no hit
    assert hov[7, 2, 0] == 0
