import pyrallis
import json
from dataclasses import dataclass, field
from .egmd import preprocess as preprocess_egmd, EGMDConfig
from .midi_to_hov import DrumCategory as DrumCategory, MidiConfig


@dataclass
class PreprocessConfig:
    tmp_dir: str = ".data/tmp"
    hov_dir: str = ".data/hov"
    midi: MidiConfig = field(default_factory=MidiConfig)
    egmd: EGMDConfig = field(default_factory=EGMDConfig)


@pyrallis.wrap()
def preprocess(config: PreprocessConfig):
    if config.egmd is not None:
        preprocess_egmd(
            tmp_dir=config.tmp_dir,
            hov_dir=config.hov_dir,
            midi_config=config.midi,
            config=config.egmd,
        )
