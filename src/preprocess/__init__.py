import pyrallis
import json
from dataclasses import dataclass, field
from .egmd import preprocess as preprocess_egmd, EGMDConfig
from .midi_to_hov import DrumCategory, DEFAULT_DRUM_CATEGORIES


@dataclass
class PreprocessConfig:
    tmp_dir: str = ".data/tmp"
    hov_dir: str = ".data/hov"
    drum_categories: list[DrumCategory] = field(
        default_factory=lambda: DEFAULT_DRUM_CATEGORIES
    )
    egmd: EGMDConfig = field(default_factory=EGMDConfig)

    def __post_init__(self):
        # convert drum map from json string to proper object
        if isinstance(self.drum_categories, str):
            self.drum_categories = [
                DrumCategory(c["label"], c["pitches"])
                for c in json.loads(self.drum_categories)
            ]


@pyrallis.wrap()
def preprocess(config: PreprocessConfig):
    if config.egmd is not None:
        preprocess_egmd(config.tmp_dir, config.hov_dir, config.egmd)
