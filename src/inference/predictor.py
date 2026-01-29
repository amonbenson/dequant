from pathlib import Path
import torch
import logging
from ..model import DequantTransformer, DequantTransformerConfig
from ..config import CONFIG
from ..utils.checkpoint import Checkpoint

logger = logging.getLogger("predictor")


class Predictor:
    def __init__(self, checkpoint: Path):
        # Create model from a stored checkpoint
        self.model = DequantTransformer(
            DequantTransformerConfig(
                max_seq_len=CONFIG.model.max_seq_len,
                num_instruments=CONFIG.model.drums.num_instruments,
                d_model=CONFIG.model.transformer.d_model,
                dropout=CONFIG.model.transformer.dropout,
            )
        )
        Checkpoint.load(
            checkpoint,
            device="cpu",
            config=CONFIG,
            model=self.model,
        )

    def predict_sequence(self, hits: torch.Tensor):
        with torch.no_grad():
            seq_len, num_instruments = hits.shape
            max_len = self.model.config.max_seq_len

            assert num_instruments == self.model.config.num_instruments, "Incompatible number of instruments"

            if seq_len > max_len:
                logger.warning(f"Sequence length ({seq_len}) is higher than the maximum ({max_len}). Some context will be dropped.")

            # Use all zeros as the start roken
            generated = torch.zeros((1, num_instruments, 2))

            # Generate the sequence step-by-step
            for step in range(len(hits)):
                # Truncate encoder input
                encoder_start = max(0, step + 1 - max_len)
                encoder_input = hits[encoder_start : step + 1, :].unsqueeze(0)

                # Truncate decoder input
                decoder_start = max(0, generated.shape[0] - max_len)
                decoder_input = generated[decoder_start:, :].unsqueeze(0)

                # Run the model to get a full sequence prediction
                prediction = self.model(encoder_input, decoder_input)[0]

                # Extract just the latest predicted timestep and mask it by the hits array
                # (If the transformer generated offset/velocity data for instruments that were
                # not playing, this will clear those values from feeding back into the decoder_input
                # on the next step)
                prediction_masked = prediction[-1:, :, :] * hits[step : step + 1, :].unsqueeze(2)

                # Append the generated step
                generated = torch.cat([generated, prediction_masked], dim=0)

            # Concat the sequence with the hits to get the full HOV matrix
            hov = torch.cat([hits.unsqueeze(2), generated[1:, :]], dim=2)
            return hov
