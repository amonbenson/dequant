import logging
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("hov_dataset")

_FILE_CACHE_SIZE = 3  # number of decompressed npz files to keep in memory at once


@dataclass
class HOVDatasetConfig:
    dir: Path
    seq_len: int = 128
    sample_stride: int = 1
    filter_empty: bool = False
    max_samples: Optional[int] = None

    def __post_init__(self):
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be > 0")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be > 0")


class HOVDataset(Dataset):
    def __init__(self, config: HOVDatasetConfig, *, data: Optional[np.ndarray] = None, pos_enc=None):
        self.config = config

        if self.config.filter_empty:
            logger.warning("filter_empty is enabled but is currently not implemented")

        if data is not None:
            if pos_enc is None:
                raise ValueError("pos_enc must be provided when data is provided")
            self._inline_data = torch.from_numpy(data).float()
            self._inline_pos = torch.from_numpy(pos_enc).float()
            self._inline = True
            self._num_instruments = data.shape[1]
            self._num_sequences = (len(data) - config.seq_len) // config.sample_stride + 1
        else:
            self._inline = False
            self._build_index()

        n = self._num_sequences if self._inline else self._total_seqs
        if self.config.max_samples is not None and self.config.max_samples < n:
            logger.info(f"Dataset initialized with {n} sequences (limited to {self.config.max_samples})")
        else:
            logger.info(f"Dataset initialized with {n} sequences")

    def _build_index(self):
        """Scan all npz files one at a time to build a sequence index. Data is not retained."""
        npz_files = sorted(Path(self.config.dir).glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {self.config.dir}. Did you run preprocess?")

        self._npz_files = npz_files  # kept for later loading

        # Per-segment index (one segment = one track with ≥1 usable sequence)
        self._seg_seq_starts: list[int] = []  # cumulative global seq index
        self._seg_file_idx: list[int] = []  # index into self._npz_files
        self._seg_track_idx: list[int] = []  # index within that file's object array

        # File-level LRU cache: file_idx -> list of (data_t, pos_t) per track
        self._file_cache: OrderedDict[int, list[tuple[torch.Tensor, torch.Tensor] | None]] = OrderedDict()

        total_seqs = 0
        num_instruments: int = 0
        t_start = time.time()

        for file_idx, npz_path in enumerate(npz_files):
            t0 = time.time()
            n_tracks = 0
            with np.load(npz_path, allow_pickle=True) as f:
                pos_key = "pos_enc" if "pos_enc" in f.files else "pos_en"
                for track_idx, (data_arr, pos_arr) in enumerate(zip(f["data"], f[pos_key])):
                    if data_arr is None or pos_arr is None:
                        continue
                    n = data_arr.shape[0]
                    if num_instruments == 0:
                        num_instruments = data_arr.shape[1]
                    n_seqs = max(0, (n - self.config.seq_len) // self.config.sample_stride + 1)
                    if n_seqs == 0:
                        continue
                    self._seg_seq_starts.append(total_seqs)
                    self._seg_file_idx.append(file_idx)
                    self._seg_track_idx.append(track_idx)
                    total_seqs += n_seqs
                    n_tracks += 1
            logger.info(f"Indexed {npz_path.name}: {n_tracks} tracks in {time.time() - t0:.1f}s")

        if total_seqs == 0:
            raise RuntimeError(f"No usable sequences found in {self.config.dir} (seq_len={self.config.seq_len})")

        self._total_seqs = total_seqs
        self._num_instruments = num_instruments
        logger.info(f"Index complete: {len(npz_files)} files, {len(self._seg_seq_starts)} tracks, {total_seqs} sequences in {time.time() - t_start:.1f}s")

    def _load_file(self, file_idx: int) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """Load all tracks from a single npz file into the file-level LRU cache."""
        if file_idx in self._file_cache:
            self._file_cache.move_to_end(file_idx)
            return self._file_cache[file_idx]

        npz_path = self._npz_files[file_idx]
        logger.debug(f"Cache miss - loading {npz_path.name}")
        tracks: list[tuple[torch.Tensor, torch.Tensor] | None] = []

        with np.load(npz_path, allow_pickle=True) as f:
            pos_key = "pos_enc" if "pos_enc" in f.files else "pos_en"
            for data_arr, pos_arr in zip(f["data"], f[pos_key]):
                if data_arr is None or pos_arr is None:
                    tracks.append(None)
                    continue
                data_t = torch.from_numpy(data_arr.astype(np.float32, copy=False))
                pos_t = torch.from_numpy(pos_arr.astype(np.float32, copy=False))
                tracks.append((data_t, pos_t))

        self._file_cache[file_idx] = tracks
        if len(self._file_cache) > _FILE_CACHE_SIZE:
            self._file_cache.popitem(last=False)  # evict least-recently-used file

        return tracks

    def __len__(self) -> int:
        n = self._num_sequences if self._inline else self._total_seqs
        if self.config.max_samples is not None:
            return min(n, self.config.max_samples)
        return n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._inline:
            start = index * self.config.sample_stride
            return (
                self._inline_data[start : start + self.config.seq_len],
                self._inline_pos[start : start + self.config.seq_len],
            )

        seg_idx = bisect_right(self._seg_seq_starts, index) - 1
        local_seq = index - self._seg_seq_starts[seg_idx]

        file_idx = self._seg_file_idx[seg_idx]
        track_idx = self._seg_track_idx[seg_idx]

        track = self._load_file(file_idx)[track_idx]
        assert track is not None
        data, pos = track

        start = local_seq * self.config.sample_stride
        return data[start : start + self.config.seq_len], pos[start : start + self.config.seq_len]


class HOVEncoderDecoderDataset(HOVDataset):
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target, pos = super().__getitem__(index)

        encoder_input = target[..., 0]

        decoder_input = torch.cat(
            [
                self.start_token().unsqueeze(0),
                target[:-1, :, 1:3],
            ],
            dim=0,
        )

        decoder_target = target[:, :, 1:3]

        return encoder_input, decoder_input, decoder_target, pos

    def start_token(self) -> torch.Tensor:
        return torch.zeros(self._num_instruments, 2)
