from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from .data.converters.hov_converter import DEFAULT_DRUM_CATEGORIES, DrumCategory


@dataclass
class EGMDSourceConfig:
    """Configuration for the E-GMD dataset source."""

    enabled: bool = True  # use this dataset
    midi_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip"
    metadata_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.csv"


@dataclass
class LMDSourceConfig:
    """Configuration for the Lakh MIDI Dataset source."""

    enabled: bool = True  # use this dataset
    url: str = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
    train_split: float = 0.8  # fraction of files used for training
    val_split: float = 0.1  # fraction used for validation (remainder goes to test)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dir: Path = Path(".data/dataset")  # target directory where the dataset is extracted to
    cache_dir: Path = Path(".data/tmp")  # temporary directory for downloads

    # Preprocessing
    num_workers: int = 0  # parallel processes. 0 = choose automatically

    # Data sources
    egmd: EGMDSourceConfig = field(default_factory=EGMDSourceConfig)  # egmd configuration
    lmd: LMDSourceConfig = field(default_factory=LMDSourceConfig)  # lmd configuration


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
    n_heads: int = 2  # number of attention heads
    n_layers: int = 5  # number of encoder/decoder layers
    dropout: float = 0.0  # dropout for all layers
    use_activation: bool = True  # enable the output activation functions (tanh for offsets and sigmoid for velocities)


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
    run_name: Optional[str] = None  # name for this run, used in TensorBoard log dir (e.g. "runs/<timestamp>_<run_name>")

    learning_rate: float = 5e-5  # optimizer learning rate
    weight_decay: float = 1e-4  # AdamW weight decay
    lr_scheduler: str = "cosine"  # "none" or "cosine"
    lr_warmup_epochs: int = 3  # linear warmup epochs before scheduler kicks in
    num_epochs: int = 100  # # maximum number of epochs to train for
    batch_size: int = 512  # number of samples per batch

    auto_preprocess: bool = False  # always run preprocess before training
    sample_stride: int = 128  # offset in which sample sequences are taken from the dataset
    sample_shuffle: bool = False  # whether samples should be ordered randomly

    max_train_samples: Optional[int] = None  # limit number of training sequences. None = use all
    max_val_samples: Optional[int] = None  # limit number of validation sequences. None = use all
    max_test_samples: Optional[int] = None  # limit number of test sequences. None = use all

    resume: bool = True  # resume training from a previously saved checkpoint
    resume_from: Optional[Path] = None  # path to the checkpoint to resume from. If not provided, use the latest
    checkpoint_dir: Path = Path(".data/checkpoints")  # where to store checkpoints
    save_every_n_epochs: int = 1  # how often to store checkpoints
    early_stopping_patience: int = 5  # stop after N epochs without improvement. 0 = disabled


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
