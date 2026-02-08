"""Preprocessing module for MIDI drum files
Optimized with parallel processing and vectorized operations
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from pretty_midi import Instrument, KeySignature, Note, PrettyMIDI, TimeSignature
from tqdm import tqdm

from ..drum_category import DEFAULT_DRUM_CATEGORIES, DrumCategory

logger = logging.getLogger("hov_converter")

FileInfos = list[tuple[Path, int]]


@dataclass
class HOVConverterConfig:
    steps_per_beat: int = 4
    max_seq_len: int = 256
    categories: list[DrumCategory] = field(default_factory=lambda: DEFAULT_DRUM_CATEGORIES)
    _category_lookup: np.ndarray = field(init=False)
    _category_reverse_lookup: np.ndarray = field(init=False)

    # bar_period: int = 8
    def __post_init__(self):
        self._category_lookup = DrumCategory.generate_forward_lookup(self.categories)
        self._category_reverse_lookup = DrumCategory.generate_reverse_lookup(self.categories)


class HOVConverter:
    def __init__(self, config: HOVConverterConfig):
        self.config = config

    def _as_pretty_midi(self, midi: Path | PrettyMIDI) -> PrettyMIDI:
        # Check if a midi object was passed directly or we should read it from a file
        if isinstance(midi, PrettyMIDI):
            return midi
        else:
            try:
                return PrettyMIDI(midi)
            except Exception as e:
                raise ValueError(f"Failed to parse MIDI file {midi}: {e}")

    def extract_tempo(self, midi: Path | PrettyMIDI) -> float:
        midi = self._as_pretty_midi(midi)
        _, tempi = midi.get_tempo_changes()

        if len(tempi) == 0:
            logger.warning("Midi contains no tempo data.")
            return 120.0

        if len(tempi) > 1:
            logger.warning("Midi file contains tempo changes. Using only the first value")

        return float(tempi[0])

    def positional_encoding(self, bar_idx: np.ndarray, pos_in_bar: np.ndarray, total_bars: int):
        bar_phase = 2 * np.pi * bar_idx / total_bars  # one full cycle over the fixed model horizon (max bars)
        beat_phase = 2 * np.pi * pos_in_bar / (self.config.steps_per_beat * 4)

        beat_sin = np.sin(beat_phase).astype(np.float32)
        beat_cos = np.cos(beat_phase).astype(np.float32)
        bar_sin = np.sin(bar_phase).astype(np.float32)
        bar_cos = np.cos(bar_phase).astype(np.float32)

        return np.stack([beat_sin, beat_cos, bar_sin, bar_cos], axis=-1)  # shape: (T, 4)

    def midi_to_hov(self, midi: Path | PrettyMIDI, tempo_bpm: Optional[int] = None):
        midi_data = self._as_pretty_midi(midi)

        # If no tempo was provided, we can still try to extract it from the midi file
        if tempo_bpm is None:
            tempo_bpm = round(self.extract_tempo(midi_data))

        # Get drum track (should be only drum track)
        drum_track: Optional[Instrument] = None
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                drum_track = instrument
                break

        if not drum_track:
            raise ValueError(f"No drum track found in {midi}")

        # iterate over notes
        onsets = []
        pitches = []
        velocities = []
        note: Note
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
        steps_per_bar = self.config.steps_per_beat * 4  # ONLY FOR 4/4 TIME SIGNATURE! need to change this if we allow other signatures
        steps_ps = bps * self.config.steps_per_beat  # steps per second
        step = 1.0 / steps_ps  # duration of one grid step in seconds

        # get vector / grid length for binary Matrix
        # We add +2 to the last onset, because it might get shifted one extra step if its
        # timing deviation is large enough.
        last_onset = onsets[-1]  # in sec
        n_grid_onsets = int(last_onset * steps_ps) + 2

        # extend n_grid_onsets to a full number of bars (15 -> 16, 16 -> 16, 17 -> 32, ...)
        if n_grid_onsets % steps_per_bar != 0:
            n_grid_onsets += steps_per_bar - n_grid_onsets % steps_per_bar

        # Limit positional encoding length to the model’s maximum sequence length
        # so that timing representations are consistent across clips of different durations
        T = min(n_grid_onsets, self.config.max_seq_len)
        step_idx = np.arange(T)
        pos_in_bar = step_idx % steps_per_bar  # 0..15 repeating
        bar_idx = step_idx // steps_per_bar

        #total_bars - max_bars
        total_bars = max(1,  self.config.max_seq_len // steps_per_bar)  #guardrail against division by zero
        pos_enc = self.positional_encoding(bar_idx, pos_in_bar, total_bars)

        # snap to grid
        nearest_idc = np.rint(onsets * steps_ps).astype(np.int32)  # get snapped onsets as grid indices
        nearest = nearest_idc * step  # snapped values

        offsets = onsets - nearest  # timing derivations ('feel')
        offset_relative = offsets * steps_ps  # relative to grid, not in seconds

        # create instrument grouped array --> "round" different drum parts (ride bell, ride edge --> ride)
        pitch_rows = self.config._category_lookup[pitches]

        # Handle duplicates: if multiple hits quantize to same (instrument, timestep),
        # keep only the one with highest velocity
        if len(pitch_rows) > 0:
            # Create composite key for each note: (instrument_row, timestep)
            composite_keys = np.column_stack((pitch_rows, nearest_idc))

            # Find unique positions and indices
            unique_positions, inverse_indices = np.unique(composite_keys, axis=0, return_inverse=True)

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
        onset_grid = np.zeros((n_grid_onsets, len(self.config.categories)), dtype=np.float32)
        offset_grid = np.zeros_like(onset_grid)
        vel_grid = np.zeros_like(onset_grid)

        # Single assignment operation for each grid
        onset_grid[nearest_idc, pitch_rows] = 1
        offset_grid[nearest_idc, pitch_rows] = offset_relative
        vel_grid[nearest_idc, pitch_rows] = velocities

        # Stack matrices (fix for np.concat bug)
        matrices = np.stack([onset_grid, offset_grid, vel_grid], axis=-1)

        return matrices, pos_enc

    def midi_to_hov_batch(self, file_infos: FileInfos, n_workers: int = 0):
        """Extract matrices from MIDI files with parallel processing

        Args:
            df: DataFrame with MIDI file information (must have 'midi_filename' and 'bpm' columns)
            ds_root: Root directory of the dataset
            n_workers: Number of parallel workers (default: CPU count)

        Returns:
            List of matrices extracted from MIDI files

        """
        if n_workers <= 0:
            n_workers = os.cpu_count() or 1

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
            futures = {executor.submit(HOVConverter.midi_to_hov, self, filepath, bpm): i for i, (filepath, bpm) in enumerate(file_infos)}

            # Process results as they complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx = futures[future]
                try:
                    result = future.result()
                    data.append((idx, result))
                except Exception as e:
                    logger.error(f"Error processing file at index {idx}: {e}")
                    traceback.print_exc()
                    data.append((idx, (None, None)))  # as now HOVConverter returns tuple of (hov, pe)

        # Sort by original index to maintain order
        data.sort(key=lambda x: x[0])
        data = [item[1] for item in data]

        # Pack data into a numpy array
        data_list = np.empty(len(data), dtype=list)
        for i, item in enumerate(data):
            data_list[i] = item

        elapsed = time.perf_counter() - start
        logger.info(f"Processed {len(file_infos)} MIDI files in {elapsed:.6f} seconds")

        return data_list

    def hov_to_midi(self, hov: np.ndarray, tempo_bpm: int = 120) -> PrettyMIDI:
        assert len(hov.shape) == 3, "HOV should have shape (seq_len, num_instruments, 3)"
        assert hov.shape[2] == 3, f"HOV dimension should have been 3 but was {hov.shape[2]}"

        # Setup metadata
        midi = PrettyMIDI(resolution=480, initial_tempo=tempo_bpm)
        midi.time_signature_changes.append(TimeSignature(4, 4, 0))
        midi.key_signature_changes.append(KeySignature(0, 0))

        # Create a drum track
        drum_track = Instrument(program=0, is_drum=True, name="Drums")

        # Iterate through each hit (e.g. where the hit matrix is set to 1)
        steps_per_second = self.config.steps_per_beat * tempo_bpm / 60
        for step, category in np.argwhere(hov[..., 0]):
            # Calculate the time from the step number and offset
            offset = hov[step, category, 1]
            step_with_offset = max(0, step + offset)  # Prevent negative time on the first step
            start = step_with_offset / steps_per_second
            end = start + 1 / steps_per_second  # make duration equal one step

            # Get the note pitch and velocity
            pitch = int(self.config._category_reverse_lookup[category])
            velocity = int(hov[step, category, 2] * 127)

            # Append the note
            drum_track.notes.append(Note(pitch=pitch, velocity=velocity, start=start, end=end))

        # Add the drum track
        midi.instruments.append(drum_track)

        return midi
