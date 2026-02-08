from pathlib import Path

import sys

import os

import subprocess

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
    pos_enc = create_dummy_pos_enc(num_steps=num_steps)

    config = HOVDatasetConfig(
        dir=Path("dummy"),
        seq_len=seq_len,
        sample_stride=sample_stride,
        filter_empty=filter_empty,
    )
    return HOVDataset(config, data=data, pos_enc=pos_enc)


# for pipeline tests
def create_dummy_pos_enc(num_steps: int, *, seed: int = 24601) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    return rng.uniform(low=-1.0, high=1.0, size=(num_steps, 4)).astype(np.float32)


def compute_pos_enc(n_steps, steps_per_bar):
    step_idx = np.arange(n_steps)
    pos_in_bar = step_idx % steps_per_bar
    bar_idx = step_idx // steps_per_bar

    total_bars = max(1, n_steps // steps_per_bar)
    beat_phase = 2 * np.pi * pos_in_bar / steps_per_bar
    bar_phase = 2 * np.pi * bar_idx / total_bars

    return np.stack(
        [
            np.sin(beat_phase),
            np.cos(beat_phase),
            np.sin(bar_phase),
            np.cos(bar_phase),
        ],
        axis=-1,
    ).astype(np.float32)


def run_cli(cmd, cwd, timeout = 180):

    if timeout == 0:
        timeout = None

    PYTHON = sys.executable
    REPO_ROOT = Path(__file__).resolve().parents[2]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    return subprocess.run(
        [
            PYTHON,
            "-m",
            "src",
            *cmd,
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )