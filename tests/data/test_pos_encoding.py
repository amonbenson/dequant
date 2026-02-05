import numpy as np

from tests.data.utils import compute_pos_enc


def test_dummy_pos_encoding_properties():
    T = 64
    steps_per_bar = 16
    pos = compute_pos_enc(T, steps_per_bar)

    assert pos.shape == (T, 4)
    assert np.max(pos) <= 1.0
    assert np.min(pos) >= -1.0

    # beat repeats every bar (16)
    assert np.allclose(pos[0, 0:2], pos[16, 0:2], atol=1e-6)
    # not a constant beat within a bar
    assert not np.allclose(pos[0, 0:2], pos[1, 0:2], atol=1e-6)
    # bar changes across bars
    assert not np.allclose(pos[0, 2:4], pos[32, 2:4], atol=1e-6)

    # bar is constant within the same bar window
    assert np.allclose(pos[0:16, 2:4], pos[0, 2:4], atol=1e-6)
    assert np.allclose(pos[16:32, 2:4], pos[16, 2:4], atol=1e-6)
