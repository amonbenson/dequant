import torch
from torch.utils.data import Dataset
import numpy as np


class DrumDataset(Dataset):
    """
    Simple PyTorch Dataset for drum HVO (Hits, Velocities, Offsets) sequences.

    Loads pre-processed .npz files and transforms them for the transformer encoder.

    Input data format from .npz:
        - Shape: (3, 9, n_timesteps)
        - Axis 0: [hits, offsets, velocities]
        - Axis 1: 9 drum instruments
        - Axis 2: variable number of timesteps (16th notes)

    Output format:
        - Shape: (n_timesteps, 27)
        - 27 features: [9 hits, 9 velocities, 9 offsets]
    """

    def __init__(self, npz_path):
        """
        Args:
            npz_path (str): Path to .npz file containing preprocessed drum data
        """
        # Load data from .npz file
        data = np.load(npz_path, allow_pickle=True)
        self.sequences = data['data']  # Object array of variable-length sequences

        print(f"Loaded {len(self.sequences)} sequences from {npz_path}")

    def _transform_sequence(self, seq):
        """
        Transform from (3, 9, n_timesteps) to (n_timesteps, 27).

        Input axes: [hits, offsets, velocities] x 9_instruments x n_timesteps
        Output: n_timesteps x [h0,h1,...h8, v0,v1,...v8, o0,o1,...o8]

        Note: The model expects the order as [hits, velocities, offsets] based on
        the DrumLoss class splitting logic.
        """
        # seq shape: (3, 9, n_timesteps)
        hits = seq[0]       # (9, n_timesteps)
        offsets = seq[1]    # (9, n_timesteps)
        velocities = seq[2] # (9, n_timesteps)

        # Transpose to (n_timesteps, 9)
        hits = hits.T       # (n_timesteps, 9)
        velocities = velocities.T
        offsets = offsets.T

        # Concatenate along feature dimension: [hits, velocities, offsets]
        # This matches the DrumLoss expectation: [:,:,:9], [:,:,9:18], [:,:,18:]
        transformed = np.concatenate([hits, velocities, offsets], axis=1)  # (n_timesteps, 27)

        return transformed

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single sequence.

        Returns:
            sequence: Tensor of shape (n_timesteps, 27)
        """
        seq = self.sequences[idx]  # (3, 9, n_timesteps)

        # Transform to (n_timesteps, 27)
        transformed = self._transform_sequence(seq)

        # Convert to tensor
        sequence = torch.from_numpy(transformed).float()

        return sequence
