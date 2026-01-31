import numpy as np

def test_pos_encoding_range():
    f = np.load(".data/dataset/test/egmd.npz", allow_pickle=True)
    assert "pos_en" in f.files
    pos = f["pos_en"][0]
    print(pos.shape)

    assert pos.ndim == 2 and pos.shape[1] == 4  # (T, 4)

    # values in range [-1, 1]
    assert np.max(pos) <= 1.0 + 1e-6
    assert np.min(pos) >= -1.0 - 1e-6

def test_pos_encoding_bar_level_periodicity():
    f = np.load(".data/dataset/test/egmd.npz", allow_pickle=True)
    pos = f["pos_en"][0]

    # repeats every 16 steps (steps_per_beat=4).
    if pos.shape[0] >= 32:
        assert np.allclose(pos[0, 0:2], pos[16, 0:2], atol=1e-6)

def test_pos_encoding_bar_change_over_time():
    f = np.load(".data/dataset/test/egmd.npz", allow_pickle=True)
    pos = f["pos_en"][0]

    if pos.shape[0] >= 64:
        assert not np.allclose(pos[0, 2:4], pos[32, 2:4], atol=1e-6)

