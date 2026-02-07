import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("hov_dataset")


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

        # Warn if filtering is enabled (not implemented for on-the-fly generation)
        if self.config.filter_empty:
            logger.warning("filter_empty is enabled but is currently not implemented")

        if data is not None:
            if pos_enc is None:
                raise ValueError("pos_enc must be provided when data is provided")
            # Use provided data
            self._data = torch.from_numpy(data)
            self._pos_enc = torch.from_numpy(pos_enc).float()
        else:
            # Load data from the configured directory
            npz_files = sorted(Path(self.config.dir).glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No .npz files found in {self.config.dir}. Did you run preprocess?")

            # Load all sub-arrays, compute total length, pre-allocate, and copy in one pass per file
            all_data_chunks = []
            all_pos_chunks = []
            total_len = 0
            for filename in npz_files:
                with np.load(filename, allow_pickle=True) as f:
                    pos_key = "pos_enc" if "pos_enc" in f.files else "pos_en"
                    for data_arr, pos_arr in zip(f["data"], f[pos_key]):
                        all_data_chunks.append(data_arr)
                        all_pos_chunks.append(pos_arr)
                        total_len += data_arr.shape[0]

            # Pre-allocate final tensors and fill from chunks
            num_instruments = all_data_chunks[0].shape[1]
            self._data = torch.empty(total_len, num_instruments, 3, dtype=torch.float32)
            self._pos_enc = torch.empty(total_len, 4, dtype=torch.float32)

            offset = 0
            for data_arr, pos_arr in zip(all_data_chunks, all_pos_chunks):
                n = data_arr.shape[0]
                self._data[offset:offset + n] = torch.from_numpy(data_arr.astype(np.float32, copy=False))
                self._pos_enc[offset:offset + n] = torch.from_numpy(pos_arr.astype(np.float32, copy=False))
                offset += n
            del all_data_chunks, all_pos_chunks

        # Validate the data shape (N, instruments, hov=3)
        assert len(self._data.shape) == 3
        assert self._data.shape[0] >= self.config.seq_len, f"Not enough data to generate sequences of length {self.config.seq_len}"
        assert self._data.shape[2] == 3, "Expected HOV dimension to be of size 3"
        assert self._data.shape[0] == self._pos_enc.shape[0]

        # Calculate the number of sequences without actually generating them
        self._num_sequences = (len(self._data) - self.config.seq_len) // self.config.sample_stride + 1

        # Log dataset size with max_samples info if applicable
        if self.config.max_samples is not None and self.config.max_samples < self._num_sequences:
            logger.info(f"Dataset initialized with {self._num_sequences} sequences (limited to {self.config.max_samples}) from raw data of length {len(self._data)}")
        else:
            logger.info(f"Dataset initialized with {self._num_sequences} sequences from raw data of length {len(self._data)}")

    def __len__(self) -> int:
        if self.config.max_samples is not None:
            return min(self._num_sequences, self.config.max_samples)
        return self._num_sequences

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate the start position for this chunk
        start_idx = index * self.config.sample_stride
        end_idx = start_idx + self.config.seq_len

        # Extract and return the chunk on-the-fly
        return self._data[start_idx:end_idx], self._pos_enc[start_idx:end_idx]

    @property
    def raw_data(self) -> torch.Tensor:
        return self._data

    @property
    def pos_enc(self) -> torch.Tensor:
        return self._pos_enc


class HOVEncoderDecoderDataset(HOVDataset):
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the full target sequence that should be generated
        target, pos = super().__getitem__(index)

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

        return encoder_input, decoder_input, decoder_target, pos

    def start_token(self) -> torch.Tensor:
        num_instruments = self._data.shape[1]
        return torch.zeros((num_instruments, 2))
