from ..config import CONFIG
from ..training import Trainer
from .common import logger
from .preprocess import run_preprocess


def run_train():
    if CONFIG.train.sample_stride % CONFIG.model.drums.steps_per_beat == 0:
        logger.warning(
            f"The parameter data.sample_stride ({CONFIG.train.sample_stride}) is equally divisible by model.drums.steps_per_beat ({CONFIG.model.drums.steps_per_beat}). "
            + "This might result in poor model performance, because the model will never receive sequences starting at any other beat than 0."
        )

    if CONFIG.train.auto_preprocess:
        run_preprocess()

    trainer = Trainer()
    trainer.train()
