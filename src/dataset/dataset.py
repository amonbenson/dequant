import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger("dataset")


@dataclass
class HOVDatasetConfig:
    hov_dir: os.PathLike
    seq_len: int = 128
    overlap: int = 64

    def __post_init__(self):
        if not (0 <= self.overlap < self.seq_len):
            raise ValueError(f"Invalid overlap {self.overlap}")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")


class HOVDataset(Dataset):
    def __init__(self, config: HOVDatasetConfig):
        self.config = config
        self._stride = config.seq_len - config.overlap

        npz_files = sorted(Path(self.config.hov_dir).glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No .npz files found in {self.config.hov_dir}. Did you run preprocess?"
            )

        arrays = []
        for filename in npz_files:
            with np.load(filename) as f:
                arrays.append(f)

        concatenated = np.concatenate(arrays, axis=0)
        self._data = torch.from_numpy(concatenated).to(torch.float32)

    def __len__(self):
        return (len(self._data) - self.config.seq_len) // self._stride + 1

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if not 0 <= index <= len(self):
            raise IndexError(f"Index {index} out of range")

        start = index * self._stride
        subsequence = self._data[start : start + self.config.seq_len]

        return subsequence

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def shape(self) -> torch.Size:
        return self._data.shape
