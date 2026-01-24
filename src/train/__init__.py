import logging
from pathlib import Path
from ..hov.dataset import HOVDataset, HOVDatasetConfig
from ..config import CONFIG

logger = logging.getLogger("train")


def load_dataset(dir: Path):
    logger.info(f"Loading dataset '{dir}' ...")
    return HOVDataset(
        HOVDatasetConfig(
            dir=dir,
            seq_len=CONFIG.model.seq_len,
            step_size=CONFIG.dataset.step_size,
            filter_empty=True,
        )
    )


def train():
    train_set, test_set, validation_set = [
        load_dataset(CONFIG.dataset.dir / split_name)
        for split_name in ("train", "test", "validation")
    ]
