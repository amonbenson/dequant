from tempfile import TemporaryFile
import numpy as np
import mido
from src.data import DrumCategory, HOVConverter, HOVConverterConfig


REDUCED_DRUM_CATEGORIES = [
    DrumCategory("Kick", (35, 36)),
    DrumCategory("Snare", (37,)),
    DrumCategory("Hi-Hat", (42,)),
]

# 480 pulses per quarter note (set as PrettyMIDI.resolution in hov_to_midi)
# and 4 steps per beat (default for HOVConverterConfig) result in 120 pulses per step
TICKS_PER_STEP = 480 / 4


def convert(hov: np.ndarray) -> mido.MidiFile:
    config = HOVConverterConfig(categories=REDUCED_DRUM_CATEGORIES)
    converter = HOVConverter(config)
    pmidi = converter.hov_to_midi(hov, 120)

    # To better analyze the low-level format, we will write it out
    # to a temporary file pointer and then re-open it using mido
    with TemporaryFile() as f:
        pmidi.write(f)
        f.seek(0)
        midi = mido.MidiFile(file=f)

        return midi


def use_absolute_time(track: mido.MidiTrack):
    t = 0
    for message in track:
        t += message.time
        message.time = t


def test_metadata():
    # hov = np.array(
    #     [
    #         [[1.0, 0, 1.0], [0, 0, 0], [0, 0, 0]],  # Step 0
    #         [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Step 1
    #     ],
    #     dtype=np.float32,
    # )
    hov = np.zeros((2, 3, 3))
    midi = convert(hov)

    assert midi.ticks_per_beat == 480

    # We should have one metadata and one drum track
    assert len(midi.tracks) == 2

    # Check if all required meta events are present
    meta_track: mido.MidiTrack = midi.tracks[0]
    assert mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0) in meta_track
    assert mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0) in meta_track
    assert mido.MetaMessage("key_signature", key="C", time=0) in meta_track
    assert mido.MetaMessage("end_of_track", time=1) in meta_track

    # Check the drum track
    drum_track: mido.MidiTrack = midi.tracks[1]
    assert mido.MetaMessage("track_name", name="Drums", time=0) in drum_track
    assert mido.Message("program_change", channel=9, program=0, time=0) in drum_track
    assert mido.MetaMessage("end_of_track", time=1) in drum_track


def test_single_kick():
    hov = np.array([
        [[1.0, 0, 64 / 127], [0, 0, 0], [0, 0, 0]],  # Step 0
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Step 1
    ]).astype(np.float32)

    midi = convert(hov)
    drum_track: mido.MidiTrack = midi.tracks[1]
    use_absolute_time(drum_track)

    # Note: PrettyMIDI uses note_on events with velocity 0 as note_off, which is fine by the MIDI standard
    assert mido.Message("note_on", channel=9, note=35, velocity=64, time=0) in drum_track
    assert mido.Message("note_on", channel=9, note=35, velocity=0, time=TICKS_PER_STEP) in drum_track
    assert mido.MetaMessage("end_of_track", time=TICKS_PER_STEP + 1) in drum_track


def test_hits_and_velocities():
    hov = np.array([
        [[1, 0, 127 / 127], [0, 0, 0], [0, 0, 0]],  # Kick
        [[0, 0, 0], [0, 0, 0], [1, 0, 64 / 127]],  # Hi-Hat
        [[1, 0, 127 / 127], [1, 0, 100 / 127], [0, 0, 0]],  # Kick + Snare
        [[0, 0, 0], [0, 0, 0], [1, 0, 1 / 127]],  # Hi-Hat
    ]).astype(np.float32)

    midi = convert(hov)
    drum_track: mido.MidiTrack = midi.tracks[1]
    use_absolute_time(drum_track)

    # Check if all note ons exist with the correct pitch and velocity
    assert mido.Message("note_on", channel=9, note=35, velocity=127, time=0 * TICKS_PER_STEP) in drum_track
    assert mido.Message("note_on", channel=9, note=42, velocity=64, time=1 * TICKS_PER_STEP) in drum_track
    assert mido.Message("note_on", channel=9, note=35, velocity=127, time=2 * TICKS_PER_STEP) in drum_track
    assert mido.Message("note_on", channel=9, note=37, velocity=100, time=2 * TICKS_PER_STEP) in drum_track
    assert mido.Message("note_on", channel=9, note=42, velocity=1, time=3 * TICKS_PER_STEP) in drum_track


def test_offsets():
    hov = np.array([
        [[1, 0, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 0.25, 1], [0, 0, 0], [0, 0, 0]],
        [[1, 0.49, 1], [0, 0, 0], [0, 0, 0]],
        [[1, -0.49, 1], [0, 0, 0], [0, 0, 0]],
        [[1, -0.25, 1], [0, 0, 0], [0, 0, 0]],
    ]).astype(np.float32)

    midi = convert(hov)
    drum_track: mido.MidiTrack = midi.tracks[1]
    use_absolute_time(drum_track)

    # Check if all the offsets get applied correctly
    assert mido.Message("note_on", channel=9, note=35, velocity=127, time=round(0 * TICKS_PER_STEP)) in drum_track
    assert mido.Message("note_on", channel=9, note=35, velocity=127, time=round(1.25 * TICKS_PER_STEP)) in drum_track
    assert mido.Message("note_on", channel=9, note=35, velocity=127, time=round(2.49 * TICKS_PER_STEP), skip_checks=True) in drum_track
    assert mido.Message("note_on", channel=9, note=35, velocity=127, time=round(2.51 * TICKS_PER_STEP), skip_checks=True) in drum_track
    assert mido.Message("note_on", channel=9, note=35, velocity=127, time=round(3.75 * TICKS_PER_STEP), skip_checks=True) in drum_track
