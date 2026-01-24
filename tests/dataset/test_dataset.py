import numpy as np
from src.hov.dataset import HOVDataset, HOVDatasetConfig
from tests.utils import create_dummy_hov, create_dummy_dataset


def test_dumm_hov_generator():
    # Make sure hov generator is deterministic
    d1 = create_dummy_hov(64, 9)
    d2 = create_dummy_hov(64, 9)
    assert np.all(d1 == d2)


def test_dataset_default_step_size():
    dataset = create_dummy_dataset(
        n_steps=32,
        n_instruments=9,
        seq_len=16,
        filter_empty=False,
    )

    assert len(dataset) == 2
    assert dataset[0].tolist() == dataset.raw_data[0:16].tolist()
    assert dataset[1].tolist() == dataset.raw_data[16:32].tolist()


def test_dataset_aligned_step_size():
    dataset = create_dummy_dataset(
        n_steps=32,
        n_instruments=9,
        seq_len=16,
        step_size=8,
        filter_empty=False,
    )

    assert len(dataset) == 3
    assert dataset[0].tolist() == dataset.raw_data[0:16].tolist()
    assert dataset[1].tolist() == dataset.raw_data[8:24].tolist()
    assert dataset[2].tolist() == dataset.raw_data[16:32].tolist()


def test_dataset_misaligned_step_size():
    dataset = create_dummy_dataset(
        n_steps=31,
        n_instruments=9,
        seq_len=16,
        step_size=8,
        filter_empty=False,
    )
    assert len(dataset) == 2
    assert dataset[0].tolist() == dataset.raw_data[0:16].tolist()
    assert dataset[1].tolist() == dataset.raw_data[8:24].tolist()

    dataset = create_dummy_dataset(
        n_steps=33,
        n_instruments=9,
        seq_len=16,
        step_size=8,
        filter_empty=False,
    )
    assert len(dataset) == 3
    assert dataset[0].numpy().tolist() == dataset.raw_data[0:16].tolist()
    assert dataset[1].numpy().tolist() == dataset.raw_data[8:24].tolist()
    assert dataset[2].numpy().tolist() == dataset.raw_data[16:32].tolist()


def test_trim():
    raw_data = np.concatenate(
        [
            np.zeros((8, 9, 3)),
            np.ones((6, 9, 3)),
            np.zeros((4, 9, 3)),
            np.ones((6, 9, 3)),
            np.zeros((8, 9, 3)),
        ],
        axis=0,
    )
    dataset = HOVDataset(
        HOVDatasetConfig(
            dir="dummy",
            seq_len=4,
            step_size=2,
            filter_empty=True,
        ),
        data=raw_data,
    )

    # 00000000_11111100_00111111_00000000 --> 0001_11101111_000
    assert len(dataset) == 8
    assert dataset[0].tolist() == raw_data[6:10].tolist()
    assert dataset[3].tolist() == raw_data[12:16].tolist()
    assert dataset[4].tolist() == raw_data[16:20].tolist()
    assert dataset[7].tolist() == raw_data[22:26].tolist()
