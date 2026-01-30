import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("hov_dataset")


@dataclass
class HOVDatasetConfig:
    dir: Path
    seq_len: int = 128
    sample_stride: int = 1
    filter_empty: bool = False

    def __post_init__(self):
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be > 0")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")


class HOVDataset(Dataset):
    def __init__(self, config: HOVDatasetConfig, *, data: Optional[np.ndarray] = None):
        self.config = config

        # Warn if filtering is enabled (not implemented for on-the-fly generation)
        if self.config.filter_empty:
            logger.warning("filter_empty is enabled but is currently not implemented")

        if data is not None:
            # Use provided data
            self._data = torch.from_numpy(data)
        else:
            # Load data from the configured directory
            npz_files = sorted(Path(self.config.dir).glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No .npz files found in {self.config.dir}. Did you run preprocess?")

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
        assert self._data.shape[0] >= self.config.seq_len, f"Not enough data to generate sequences of length {self.config.seq_len}"
        assert self._data.shape[2] == 3, "Expected HOV dimension to be of size 3"

        # Calculate the number of sequences without actually generating them
        self._num_sequences = (len(self._data) - self.config.seq_len) // self.config.sample_stride + 1
        logger.info(f"Dataset initialized with {self._num_sequences} sequences from raw data of length {len(self._data)}")

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        # Calculate the start position for this chunk
        start_idx = index * self.config.sample_stride
        end_idx = start_idx + self.config.seq_len

        # Extract and return the chunk on-the-fly
        return self._data[start_idx:end_idx]

    @property
    def raw_data(self) -> torch.Tensor:
        return self._data


class HOVEncoderDecoderDataset(HOVDataset):
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the full target sequence that should be generated
        target = super().__getitem__(index)

        # Encoder input: Use the full sequence, but include only the hits
        # (this is what the encoder uses as its "baseline" for generating the next timestep)
        encoder_input = target[..., 0]

        # Decoder input: Shift the sequence to exclude the next step and include the start token.
        # This is what has already been generated
        decoder_input = torch.cat(
            [
                self.start_token().unsqueeze(0),
                target[:-1, :, 1:3],
            ],
            dim=0,
        )

        # Decoder target: Use the full sequence, but include only the offset and velocity info.
        # This is what should be generated
        decoder_target = target[:, :, 1:3]

        return (encoder_input, decoder_input, decoder_target)

    def start_token(self) -> torch.Tensor:
        num_instruments = self._data.shape[1]
        return torch.zeros((num_instruments, 2))
