"""
Preprocessing module for MIDI drum files
Optimized with parallel processing and vectorized operations
"""

import numpy as np
from pathlib import Path
import pretty_midi
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os

logger = logging.getLogger("preprocess")

# Constants
N_INSTRUMENTS = 9


def pitch_to_category(pitches):
    """
    Maps the Midi pitch values 0 - 127 to the Drum category according to the Paper's Appendix B
    Optimized with NumPy vectorization for better performance
    """

    # Pre-allocate output array
    categories = np.zeros(len(pitches), dtype=np.int32)

    # Define mapping as two arrays for vectorized operation
    pitch_map = np.array([
        36,
        38,
        40,
        37,
        48,
        50,
        45,
        47,
        43,
        58,
        46,
        26,
        42,
        22,
        44,
        49,
        55,
        57,
        52,
        51,
        59,
        53,
        54,
        39,
        56,
    ])
    category_map = np.array([
        36,
        38,
        38,
        38,
        50,
        50,
        47,
        47,
        43,
        43,
        46,
        46,
        42,
        42,
        42,
        49,
        49,
        49,
        49,
        51,
        51,
        51,
        42,
        38,
        51,
    ])

    # Vectorized lookup
    pitches_array = np.array(pitches)
    for pitch_val, cat_val in zip(pitch_map, category_map):
        categories[pitches_array == pitch_val] = cat_val

    # Check for unmapped pitches
    unmapped_mask = (categories == 0) & (pitches_array != 0)
    if np.any(unmapped_mask):
        unmapped_pitches = pitches_array[unmapped_mask]
        for pitch in np.unique(unmapped_pitches):
            logger.warning(f"{pitch} not in mapping")

    # Validate number of instruments
    n_instruments = len(np.unique(category_map))
    if n_instruments != N_INSTRUMENTS:
        raise ValueError(
            "Global constant N_INSTRUMENTS does not fit unique values in mapping dict"
        )

    return categories


def read_midi(midi_path: Path, tempo_bpm):
    """
    Read MIDI file and extract all needed data

    Args:
        midi_path: Path to MIDI file
        tempo_bpm: Tempo in beats per minute

    Returns:
        Stacked matrices (3, n_instruments, n_timesteps) containing onset, offset, and velocity grids
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        raise ValueError(f"Failed to parse MIDI file {midi_path}: {e}")

    # Get drum track (should be only drum track)
    drum_track = None
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            drum_track = instrument
            break

    if drum_track is None:
        raise ValueError(f"No drum track found in {midi_path}")

    # iterate over notes
    onsets = []
    pitches = []
    velocities = []
    for note in drum_track.notes:
        onsets.append(note.start)
        pitches.append(note.pitch)
        velocities.append(note.velocity)

    # sort by onset time
    sort_idx = np.argsort(onsets)
    onsets = np.array(onsets)[sort_idx]
    pitches = np.array(pitches)[sort_idx]
    velocities = np.array(velocities)[sort_idx] / 126.0

    # Cache repeated calculations (optimization #6)
    bps = tempo_bpm / 60.0  # beats per second
    sixteenth_ps = bps * 4  # sixteenth notes per second
    step = 1.0 / sixteenth_ps

    # get vector / grid length for binary Matrix
    last_onset = onsets[-1]  # in sec
    n_grid_onsets = int(last_onset * sixteenth_ps) + 2  # TODO should only be +1; bug?

    # snap to grid
    nearest_idc = np.rint(onsets * sixteenth_ps).astype(
        np.int32
    )  # get snapped onsets as grid indices
    nearest = nearest_idc * step  # snapped values

    offsets = np.abs(onsets - nearest)  # timing derivations ('feel')
    offset_relative = offsets * sixteenth_ps  # relative to grid, not in seconds

    # create instrument grouped array --> "round" different drum parts (ride bell, ride edge --> ride)
    grouped_pitches = pitch_to_category(pitches)
    pitch_categories = np.array([36, 38, 50, 47, 43, 46, 42, 49, 51])
    pitch_to_row = {p: i for i, p in enumerate(pitch_categories)}
    pitch_rows = np.array([pitch_to_row[p] for p in grouped_pitches])

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
    onset_grid = np.zeros((len(pitch_categories), n_grid_onsets), dtype=np.uint8)
    offset_grid = np.zeros_like(onset_grid, dtype=np.float32)
    vel_grid = np.zeros_like(onset_grid, dtype=np.float32)

    # Single assignment operation for each grid
    onset_grid[pitch_rows, nearest_idc] = 1
    offset_grid[pitch_rows, nearest_idc] = offset_relative
    vel_grid[pitch_rows, nearest_idc] = velocities

    # Stack matrices (fix for np.concat bug)
    matrices = np.stack([onset_grid, offset_grid, vel_grid], axis=0)

    return matrices


def _process_midi_file(filepath, bpm):
    """
    Process a single MIDI file - helper for parallel processing
    This function must be at module level to be picklable
    """
    return read_midi(Path(filepath), bpm)


FileInfos = list[tuple[os.PathLike, int]]


def extract_matrices(file_infos: FileInfos, n_workers=None):
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
            executor.submit(_process_midi_file, filepath, bpm): i
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

    elapsed = time.perf_counter() - start
    logger.info(f"Processed {len(file_infos)} MIDI files in {elapsed:.6f} seconds")

    return data
