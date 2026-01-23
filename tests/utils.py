import numpy as np
from src.hov.dataset import HOVDataset, HOVDatasetConfig


def create_dummy_hov(
    n_steps: int,
    n_instruments: int,
    *,
    seed: int = 24601,
    hit_probability: float = 0.9,
) -> np.ndarray:
    shape = (n_steps, n_instruments)

    # Instantiate a new random number generator
    rng = np.random.default_rng(seed)

    # Generate random hits, offsets, and velocities
    hits = (rng.random(shape) > hit_probability).astype(np.float32)
    offsets = np.clip(rng.standard_normal(shape), -0.5, 0.5).astype(np.float32) * hits
    velocities = rng.random(shape).astype(np.float32) * hits

    # Stack them into an HOV matrix
    return np.stack([hits, offsets, velocities], axis=-1)


def create_dummy_dataset(
    n_steps: int,
    n_instruments: int,
    seq_len: int,
    step_size: int = None,
    trim: bool = True,
) -> HOVDataset:
    data = create_dummy_hov(
        n_steps=n_steps,
        n_instruments=n_instruments,
    )
    config = HOVDatasetConfig(
        hov_dir="dummy",
        seq_len=seq_len,
        step_size=step_size,
        trim=trim,
    )
    return HOVDataset(config, data=data)
