from pathlib import Path
from dataclasses import dataclass, field, asdict
from ..hov.converter import DrumCategory, DEFAULT_DRUM_CATEGORIES


@dataclass
class ModelConfig:
    steps: int = 128
    instruments: int = 7
    resolution: int = 16
    categories: list[DrumCategory] = field(
        init=False,
        default_factory=lambda: DEFAULT_DRUM_CATEGORIES,
    )


@dataclass
class EGMDPreprocessorConfig:
    enabled: bool = True
    midi_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip"
    meta_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.csv"


@dataclass
class PreprocessConfig:
    egmd: EGMDPreprocessorConfig = field(default_factory=EGMDPreprocessorConfig)


@dataclass
class TrainConfig:
    pass


@dataclass
class MainConfig:
    tmp_dir: Path = Path(".data/tmp")
    dataset_dir: Path = Path(".data/dataset")
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = MainConfig()


def update_config(new_config: MainConfig):
    global CONFIG
    CONFIG = new_config
