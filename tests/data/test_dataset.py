import numpy as np
import pytest

from src.data.datasets.hov_dataset import HOVDataset, HOVDatasetConfig, HOVEncoderDecoderDataset
from tests.data.utils import create_dummy_dataset, create_dummy_hov, create_dummy_pos_enc


def test_dumm_hov_generator():
    # Make sure hov generator is deterministic
    d1 = create_dummy_hov(64, 9)
    d2 = create_dummy_hov(64, 9)
    assert np.all(d1 == d2)


def test_dataset_aligned_sample_stride():
    dataset = create_dummy_dataset(
        num_steps=32,
        num_instruments=9,
        seq_len=16,
        sample_stride=8,
        filter_empty=False,
    )
    hov0, pos0 = dataset[0]
    hov1, pos1 = dataset[1]
    hov2, pos2 = dataset[2]

    assert len(dataset) == 3
    assert hov0.numpy().tolist() == dataset.raw_data[0:16].tolist()
    assert hov1.numpy().tolist() == dataset.raw_data[8:24].tolist()
    assert hov2.numpy().tolist() == dataset.raw_data[16:32].tolist()

    # pos_encoding alignment
    assert pos0.numpy().tolist() == dataset.pos_enc[0:16].tolist()
    assert pos1.numpy().tolist() == dataset.pos_enc[8:24].tolist()
    assert pos2.numpy().tolist() == dataset.pos_enc[16:32].tolist()


def test_dataset_misaligned_sample_stride():
    dataset = create_dummy_dataset(
        num_steps=31,
        num_instruments=9,
        seq_len=16,
        sample_stride=8,
        filter_empty=False,
    )
    assert len(dataset) == 2
    hov0, pos0 = dataset[0]
    hov1, pos1 = dataset[1]

    assert hov0.numpy().tolist() == dataset.raw_data[0:16].tolist()
    assert hov1.numpy().tolist() == dataset.raw_data[8:24].tolist()

    dataset = create_dummy_dataset(
        num_steps=33,
        num_instruments=9,
        seq_len=16,
        sample_stride=8,
        filter_empty=False,
    )
    assert len(dataset) == 3
    hov0, pos0 = dataset[0]
    hov1, pos1 = dataset[1]
    hov2, pos2 = dataset[2]

    assert hov0.numpy().tolist() == dataset.raw_data[0:16].tolist()
    assert hov1.numpy().tolist() == dataset.raw_data[8:24].tolist()
    assert hov2.numpy().tolist() == dataset.raw_data[16:32].tolist()


@pytest.mark.skip(reason="Filtering is currently not implemented")
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
            sample_stride=2,
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


def test_encoder_decoder():
    raw_data = create_dummy_hov(
        num_steps=24,
        num_instruments=9,
    )
    pos_enc = create_dummy_pos_enc(num_steps=24)

    dataset = HOVEncoderDecoderDataset(
        HOVDatasetConfig(
            dir="dummy",
            seq_len=16,
            sample_stride=8,
            filter_empty=False,
        ),
        data=raw_data,
        pos_enc=pos_enc,
    )

    assert dataset.start_token().tolist() == [[0.0, 0.0]] * 9

    # Note: only 2 full sequences should fit here, because the encoder/decoder requires one more step for the shifting
    # (and we are only supplying 32 steps, not 33)
    assert len(dataset) == 2

    # Check timestep 0
    encoder_input, decoder_input, decoder_target, pos = dataset[0]
    assert encoder_input.tolist() == raw_data[0:16, :, 0].tolist()
    assert decoder_input.tolist() == [[[0.0, 0.0]] * 9, *raw_data[0:15, :, 1:3].tolist()]
    assert decoder_target.tolist() == raw_data[0:16, :, 1:3].tolist()
    assert pos.tolist() == pos_enc[0:16].tolist()

    # Check timestep 1
    encoder_input, decoder_input, decoder_target, pos = dataset[1]
    assert encoder_input.tolist() == raw_data[8:24, :, 0].tolist()
    assert decoder_input.tolist() == [[[0.0, 0.0]] * 9, *raw_data[8:23, :, 1:3].tolist()]
    assert decoder_target.tolist() == raw_data[8:24, :, 1:3].tolist()
    assert pos.tolist() == pos_enc[8:24].tolist()
