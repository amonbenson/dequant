from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional, Literal
import tyro
from ..hov.converter import DrumCategory, DEFAULT_DRUM_CATEGORIES


@dataclass
class EGMDSourceConfig:
    """Configuration for the E-GMD dataset source."""

    enabled: bool = True  # use this dataset
    midi_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip"
    metadata_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.csv"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dir: Path = Path(".data/dataset")  # target directory where the dataset is extracted to
    cache_dir: Path = Path(".data/tmp")  # temporary directory for downloads

    # Preprocessing
    num_workers: int = 0  # parallel processes. 0 = choose automatically

    # Data sources
    egmd: EGMDSourceConfig = field(default_factory=EGMDSourceConfig)  # egmd configuration


@dataclass
class DrumsConfig:
    """Configuration for drum instrument mapping."""

    num_instruments: int = 9  # total number of different instrument categories
    steps_per_beat: int = 4  # resolution of each beat
    categories: list[DrumCategory] = field(
        init=False,
        default_factory=lambda: DEFAULT_DRUM_CATEGORIES,
    )  # defines how midi notes are mapped to different drum categories

    def __post_init__(self):
        assert self.num_instruments == len(self.categories), f"number of categories ({len(self.categories)}) should match the defined num_instruments ({self.num_instruments})"


@dataclass
class TransformerConfig:
    """Configuration for the transformer architecture."""

    d_model: int = 128  # internal model depth


@dataclass
class ModelConfig:
    """Configuration for the Dequant model."""

    max_seq_len: int = 128  # maximum number of sequences
    drums: DrumsConfig = field(default_factory=DrumsConfig)  # drum-specific configuration
    transformer: TransformerConfig = field(default_factory=TransformerConfig)  # transformer-specific configuration


@dataclass
class TrainConfig:
    """Configuration for training."""

    device: Optional[str] = None  # "cpu", "cuda", "mps"

    learning_rate: float = 1e-4  # optimizer learning rate
    num_epochs: int = 100  # maximum number of epochs to train for
    batch_size: int = 32  # number of samples per batch

    auto_preprocess: bool = True  # always run preprocess before training
    sample_stride: int = 777  # offset in which sample sequences are taken from the dataset
    sample_shuffle: bool = True  # whether samples should be ordered randomly

    checkpoint_dir: Path = Path(".data/checkpoints")  # where to store checkpoints
    save_every_n_epochs: int = 1  # how often to store checkpoints

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

    # Do not replace the global config directly, because this would make modules which already
    # imported the name use the wrong configuration object.
    for f in fields(RootConfig):
        setattr(CONFIG, f.name, getattr(new_config, f.name))
