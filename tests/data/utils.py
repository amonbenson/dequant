import numpy as np
from src.data.datasets.hov_dataset import HOVDataset, HOVDatasetConfig


def create_dummy_hov(
    num_steps: int,
    num_instruments: int,
    *,
    seed: int = 24601,
    hit_probability: float = 0.9,
) -> np.ndarray:
    shape = (num_steps, num_instruments)

    # Instantiate a new random number generator
    rng = np.random.default_rng(seed)

    # Generate random hits, offsets, and velocities
    hits = (rng.random(shape) > hit_probability).astype(np.float32)
    offsets = np.clip(rng.standard_normal(shape), -0.5, 0.5).astype(np.float32) * hits
    velocities = rng.random(shape).astype(np.float32) * hits

    # Stack them into an HOV matrix
    return np.stack([hits, offsets, velocities], axis=-1)


def create_dummy_dataset(
    num_steps: int,
    num_instruments: int,
    seq_len: int,
    sample_stride: int = 1,
    filter_empty: bool = True,
) -> HOVDataset:
    data = create_dummy_hov(
        num_steps=num_steps,
        num_instruments=num_instruments,
    )
    config = HOVDatasetConfig(
        dir="dummy",
        seq_len=seq_len,
        sample_stride=sample_stride,
        filter_empty=filter_empty,
    )
    return HOVDataset(config, data=data)
