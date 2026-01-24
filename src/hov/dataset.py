import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("dataset")


@dataclass
class HOVDatasetConfig:
    dir: Path
    seq_len: int = 128
    step_size: Optional[int] = None
    filter_empty: bool = True

    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.seq_len

        if not (0 < self.step_size <= self.seq_len):
            raise ValueError(f"Invalid step_size {self.step_size}")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")


class HOVDataset(Dataset):
    def __init__(self, config: HOVDatasetConfig, *, data: Optional[np.ndarray] = None):
        self.config = config

        if data is not None:
            # Use provided data
            self._data = torch.from_numpy(data)
        else:
            # Load data from the configured directory
            npz_files = sorted(Path(self.config.dir).glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(
                    f"No .npz files found in {self.config.dir}. Did you run preprocess?"
                )

            # Load each npz file
            arrays = []
            for filename in npz_files:
                with np.load(filename, allow_pickle=True) as f:
                    arrays.append(f["data"])

            # Concatenate twice: 1. stich each file together, 2. stich each sequence within each file together
            concatenated = np.concatenate(arrays, axis=0)
            concatenated = np.concatenate(concatenated, axis=0)
            self._data = torch.from_numpy(concatenated).to(torch.float32)

        # Validate the data shape (N, instruments, hov=3)
        assert len(self._data.shape) == 3
        assert self._data.shape[0] >= self.config.seq_len, (
            f"Not enough data to generate sequences of length {self.config.seq_len}"
        )
        assert self._data.shape[2] == 3, "Expected HOV dimension to be of size 3"

        # Unfold the data into separate chunks
        unfolded = self._data.unfold(0, self.config.seq_len, self.config.step_size)
        self._chunks = unfolded.movedim(-1, 1)  # move the chunk dimension to the start

        # Remove completely empty chunks
        if self.config.filter_empty:
            non_empty = self._chunks.any(dim=(1, 2, 3))
            self._chunks = self._chunks[non_empty]

            n_empty = len(self) - non_empty.sum()
            if n_empty > 0:
                logger.info(
                    f"Removed {n_empty}/{len(self)} training sequences because they are completely empty."
                )

        logger.info(f"Loaded {len(self)} sequences.")

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._chunks[index]

    @property
    def raw_data(self) -> torch.Tensor:
        return self._data
