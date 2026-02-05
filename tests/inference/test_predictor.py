import pytest
import torch

from src.config import ModelConfig
from src.inference.predictor import Predictor, PredictorConfig


def create_dummy_predictor(model_config: ModelConfig = ModelConfig()):
    predictor = Predictor(PredictorConfig(None, model_config))
    return predictor


@pytest.mark.parametrize("seq_len", (5, 200, 2000))
def test_prediction_no_uninitialized_values(seq_len):
    predictor = create_dummy_predictor()
    predictor._sequence[...] = -1.234
    predictor._pos_enc[...] = -1.234

    sequence = torch.zeros((seq_len, 9))
    result = predictor.process_sequence(sequence)

    # Check the result shape
    assert result.shape == (seq_len, 9, 3)

    # Check that no uninitialized values make it thru to the result
    assert not (result == -1.234).any()


def test_prediction_keeps_values():
    predictor = create_dummy_predictor()

    # Set all model weights to a known value
    with torch.no_grad():
        for param in predictor.model.parameters():
            param.zero_()

    # Initialize predictor internal states to a known value so we can debug access of uninitialized memory
    predictor._sequence[...] = -1.234
    predictor._pos_enc[...] = -1.234

    ######## STEP 1 ########

    # Run the prediction step with a custom hits array
    predictor.process_step(torch.tensor([1, 0, 0, 0, 1, 0, 0, 1, 0]))

    s0_generated = predictor.get_generated_sequence()
    assert s0_generated.shape == (1, 9, 3)

    # Make sure the hits got applied
    assert s0_generated[0, 0, 0] == 1
    assert s0_generated[0, 1, 0] == 0
    assert s0_generated[0, 7, 0] == 1
    assert s0_generated[0, 8, 0] == 0

    # Make sure some OV values got generated
    # At positions where no hits were played, the OV should also be zeroed out
    assert -0.5 <= s0_generated[0, 0, 1] <= 0.5
    assert s0_generated[0, 1, 1] == 0
    assert -0.5 <= s0_generated[0, 7, 1] <= 0.5
    assert s0_generated[0, 8, 1] == 0

    assert 0 <= s0_generated[0, 0, 2] <= 1
    assert s0_generated[0, 1, 2] == 0
    assert 0 <= s0_generated[0, 7, 2] <= 1
    assert s0_generated[0, 8, 2] == 0

    s0_generated = s0_generated.clone()

    ######## STEP 2 ########

    # Process another step
    predictor.process_step(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 1]))

    s1_generated = predictor.get_generated_sequence()
    assert s1_generated.shape == (2, 9, 3)

    # Make sure the complete s0 step is still in the sequence
    assert s1_generated[0].tolist() == s0_generated[0].tolist()

    # Make sure the hits got applied
    assert s1_generated[1, 0, 0] == 0
    assert s1_generated[1, 1, 0] == 1
    assert s1_generated[1, 7, 0] == 0
    assert s1_generated[1, 8, 0] == 1

    # Make sure some OV values got generated
    # At positions where no hits were played, the OV should also be zeroed out
    assert s1_generated[1, 0, 1] == 0
    assert -0.5 <= s1_generated[1, 1, 1] <= 0.5
    assert s1_generated[1, 7, 1] == 0
    assert -0.5 <= s1_generated[1, 8, 1] <= 0.5

    assert s1_generated[1, 0, 2] == 0
    assert 0 <= s1_generated[1, 1, 2] <= 1
    assert s1_generated[1, 7, 2] == 0
    assert 0 <= s1_generated[1, 8, 2] <= 1
