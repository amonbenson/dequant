from ..config import CONFIG
from .preprocess import run_preprocess
from ..training import Trainer
from .common import logger


def run_train():
    if CONFIG.train.sample_stride % CONFIG.model.drums.steps_per_beat == 0:
        logger.warning(
            f"The parameter data.sample_stride ({CONFIG.data.sample_stride}) is equally divisible by model.drums.steps_per_beat ({CONFIG.model.drums.steps_per_beat}). "
            + "This will result in poor model performance, because the model will never receive sequences starting at any other beat than 0."
        )

    if CONFIG.train.auto_preprocess:
        run_preprocess()

    def train():
        trainer = Trainer()
        trainer.train()
