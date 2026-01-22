"""
Preprocessing module for MIDI drum files
Optimized with parallel processing and vectorized operations
"""

import numpy as np
from pathlib import Path
import pretty_midi
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os

logger = logging.getLogger("preprocess")


@dataclass
class DrumCategory:
    label: str
    pitches: tuple[int]


# See https://musescore.org/sites/musescore.org/files/General%20MIDI%20Standard%20Percussion%20Set%20Key%20Map.pdf
DEFAULT_DRUM_CATEGORIES = [
    DrumCategory("Kick", (35, 36)),
    DrumCategory("Snare", (37, 38, 40)),
    DrumCategory("Floor Tom", (41, 43)),
    DrumCategory("Low Tom", (45, 47)),
    DrumCategory("High Tom", (48, 50)),
    DrumCategory("Closed Hi-Hat", (42, 44)),
    DrumCategory("Open Hi-Hat", (46,)),
    DrumCategory("Crash", (49, 52, 55, 57)),
    DrumCategory("Ride", (51, 53, 59)),
]


@dataclass
class MidiConfig:
    categories: list[DrumCategory] = field(
        default_factory=lambda: DEFAULT_DRUM_CATEGORIES
    )
    resolution: int = 4  # steps per beat (4 steps per beat == 16 steps per bar)
    _category_lookup: np.ndarray = field(init=False)

    def __post_init__(self):
        # initialize the category lookup table
        self._category_lookup = -np.ones(128, dtype=np.int8)
        for i, cat in enumerate(self.categories):
            for pitch in cat.pitches:
                if pitch < 0 or pitch > 127:
                    raise ValueError(
                        f"Category {cat.label}: pitch {pitch} is out of range."
                    )
                if self._category_lookup[pitch] != -1:
                    raise ValueError(
                        f"Category: {cat.label}: pitch {pitch} was already mapped to another category. Category pitches must be unique!"
                    )
                self._category_lookup[pitch] = i


def read_midi(
    midi_obj: Path | pretty_midi.PrettyMIDI,
    tempo_bpm: int,
    config: MidiConfig = MidiConfig(),
):
    """
    Read MIDI file and extract all needed data

    Args:
        midi_path: Path to MIDI file
        tempo_bpm: Tempo in beats per minute

    Returns:
        Stacked matrices (3, n_instruments, n_timesteps) containing onset, offset, and velocity grids
    """
    # Check if a midi object was passed directly or we should read it from a file
    if isinstance(midi_obj, pretty_midi.PrettyMIDI):
        midi_data = midi_obj
    else:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_obj)
        except Exception as e:
            raise ValueError(f"Failed to parse MIDI file {midi_obj}: {e}")

    # Get drum track (should be only drum track)
    drum_track = None
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            drum_track = instrument
            break

    if drum_track is None:
        raise ValueError(f"No drum track found in {midi_obj}")

    # iterate over notes
    onsets = []
    pitches = []
    velocities = []
    for note in drum_track.notes:
        if note.velocity > 0:
            onsets.append(note.start)
            pitches.append(note.pitch)
            velocities.append(note.velocity)

    # sort by onset time
    sort_idx = np.argsort(onsets)
    onsets = np.array(onsets)[sort_idx]
    pitches = np.array(pitches)[sort_idx]
    velocities = np.array(velocities)[sort_idx] / 127.0

    # Cache repeated calculations (optimization #6)
    bps = tempo_bpm / 60.0  # beats per second
    steps_per_bar = (
        config.resolution * 4
    )  # ONLY FOR 4/4 TIME SIGNATURE! need to change this if we allow other signatures
    steps_ps = bps * config.resolution  # steps per second
    step = 1.0 / steps_ps

    # get vector / grid length for binary Matrix
    last_onset = onsets[-1]  # in sec
    n_grid_onsets = int(last_onset * steps_ps) + 1

    # extend n_grid_onsets to a full number of bars (15 -> 16, 16 -> 16, 17 -> 32, ...)
    if n_grid_onsets % steps_per_bar != 0:
        n_grid_onsets += steps_per_bar - n_grid_onsets % steps_per_bar

    # snap to grid
    nearest_idc = np.rint(onsets * steps_ps).astype(
        np.int32
    )  # get snapped onsets as grid indices
    nearest = nearest_idc * step  # snapped values

    offsets = onsets - nearest  # timing derivations ('feel')
    offset_relative = offsets * steps_ps  # relative to grid, not in seconds

    # create instrument grouped array --> "round" different drum parts (ride bell, ride edge --> ride)
    pitch_rows = config._category_lookup[pitches]

    # Handle duplicates: if multiple hits quantize to same (instrument, timestep),
    # keep only the one with highest velocity
    if len(pitch_rows) > 0:
        # Create composite key for each note: (instrument_row, timestep)
        composite_keys = np.column_stack((pitch_rows, nearest_idc))

        # Find unique positions and indices
        unique_positions, inverse_indices = np.unique(
            composite_keys, axis=0, return_inverse=True
        )

        # For each unique position, find the note with maximum velocity
        selected_indices = []
        for i in range(len(unique_positions)):
            # Get all indices that map to this unique position
            duplicate_mask = inverse_indices == i
            duplicate_indices = np.where(duplicate_mask)[0]

            # Select the one with highest velocity
            max_vel_idx = duplicate_indices[np.argmax(velocities[duplicate_indices])]
            selected_indices.append(max_vel_idx)

        selected_indices = np.array(selected_indices)

        # Filter arrays to keep only selected notes (loudest for each position)
        pitch_rows = pitch_rows[selected_indices]
        nearest_idc = nearest_idc[selected_indices]
        offset_relative = offset_relative[selected_indices]
        velocities = velocities[selected_indices]

    # Pre-allocate grids (optimization #4)
    # Note: onsets also use float32, because torch uses float32 internally
    onset_grid = np.zeros((n_grid_onsets, len(config.categories)), dtype=np.float32)
    offset_grid = np.zeros_like(onset_grid)
    vel_grid = np.zeros_like(onset_grid)

    # Single assignment operation for each grid
    onset_grid[nearest_idc, pitch_rows] = 1
    offset_grid[nearest_idc, pitch_rows] = offset_relative
    vel_grid[nearest_idc, pitch_rows] = velocities

    # Stack matrices (fix for np.concat bug)
    matrices = np.stack([onset_grid, offset_grid, vel_grid], axis=0)

    return matrices


def _process_midi_file(filepath, bpm, config=MidiConfig()):
    """
    Process a single MIDI file - helper for parallel processing
    This function must be at module level to be picklable
    """
    return read_midi(Path(filepath), bpm, config)


FileInfos = list[tuple[os.PathLike, int]]


def extract_matrices(
    file_infos: FileInfos,
    config: MidiConfig = MidiConfig(),
    n_workers=None,
):
    """
    Extract matrices from MIDI files with parallel processing

    Args:
        df: DataFrame with MIDI file information (must have 'midi_filename' and 'bpm' columns)
        ds_root: Root directory of the dataset
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        List of matrices extracted from MIDI files
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    start = time.perf_counter()

    # Convert DataFrame to list of tuples (picklable format)
    # file_infos = [
    #     (str(Path(ds_root) / row.midi_filename), row.bpm)
    #     for row in df.itertuples(index=False)
    # ]

    data = []

    # Use ProcessPoolExecutor for better Jupyter compatibility
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_process_midi_file, filepath, bpm, config): i
            for i, (filepath, bpm) in enumerate(file_infos)
        }

        # Process results as they complete with progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            try:
                result = future.result()
                data.append((idx, result))
            except Exception as e:
                logger.error(f"Error processing file at index {idx}: {e}")
                data.append((idx, None))

    # Sort by original index to maintain order
    data.sort(key=lambda x: x[0])
    data = [item[1] for item in data]

    # Pack data into a numpy array
    data_np = np.empty(len(data), dtype=object)
    for i, item in enumerate(data):
        data_np[i] = item

    elapsed = time.perf_counter() - start
    logger.info(f"Processed {len(file_infos)} MIDI files in {elapsed:.6f} seconds")

    return data_np
