import pyrallis
from dataclasses import dataclass, field, asdict
from .egmd import preprocess as preprocess_egmd, EGMDConfig


@dataclass
class PreprocessConfig:
    tmp_dir: str = ".data/tmp"
    hov_dir: str = ".data/hov"
    egmd: EGMDConfig = field(default_factory=EGMDConfig)


@pyrallis.wrap()
def preprocess(config: PreprocessConfig):
    if config.egmd is not None:
        preprocess_egmd(config.tmp_dir, config.hov_dir, config.egmd)
