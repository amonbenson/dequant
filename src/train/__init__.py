from ..hov.dataset import HOVDataset, HOVDatasetConfig
from ..config import CONFIG


def train():
    train_set = HOVDataset(
        HOVDatasetConfig(
            hov_dir=CONFIG.dataset_dir / "train",
            seq_len=128,
        )
    )
