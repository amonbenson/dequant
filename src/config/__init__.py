from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from ..hov.converter import DrumCategory, DEFAULT_DRUM_CATEGORIES


@dataclass
class EGMDSourceConfig:
    """Configuration for the E-GMD dataset source."""

    enabled: bool = True
    midi_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip"
    metadata_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.csv"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dir: Path = Path(".data/dataset")
    cache_dir: Path = Path(".data/tmp")

    # Preprocessing
    num_workers: Optional[int] = None

    # Data sources
    egmd: EGMDSourceConfig = field(default_factory=EGMDSourceConfig)


@dataclass
class DrumsConfig:
    """Configuration for drum instrument mapping."""

    num_instruments: int = 9
    steps_per_beat: int = 4
    categories: list[DrumCategory] = field(
        init=False,
        default_factory=lambda: DEFAULT_DRUM_CATEGORIES,
    )


@dataclass
class TransformerConfig:
    """Configuration for the transformer architecture."""

    d_model: int = 128


@dataclass
class ModelConfig:
    """Configuration for the Dequant model."""

    seq_len: int = 128
    drums: DrumsConfig = field(default_factory=DrumsConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class TrainConfig:
    """Configuration for training."""

    # Hardware
    device: Optional[str] = None  # None = auto-detect

    # Optimization
    learning_rate: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 32

    # Data loading
    auto_preprocess: bool = True
    sample_stride: Optional[int] = 777
    sample_shuffle: bool = True

    # Checkpointing
    checkpoint_dir: Path = Path(".data/checkpoints")
    save_every_n_epochs: int = 1

    # Early stopping
    # patience: int = 10


@dataclass
class RootConfig:
    """Root configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = RootConfig()


def update_config(new_config: RootConfig):
    global CONFIG
    CONFIG = new_config
