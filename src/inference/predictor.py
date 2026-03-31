import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import ModelConfig
from ..data.converters.hov_converter import HOVConverter, HOVConverterConfig
from ..model import DequantTransformer, DequantTransformerConfig
from ..utils.checkpoint import Checkpoint

logger = logging.getLogger("predictor")


@dataclass
class PredictorConfig:
    checkpoint: Optional[Path]
    model: ModelConfig


class Predictor:
    def __init__(self, config: PredictorConfig):
        self.config = config

        # Create model from a stored checkpoint
        self.model = DequantTransformer(
            DequantTransformerConfig(
                max_seq_len=self.config.model.max_seq_len,
                num_instruments=self.config.model.drums.num_instruments,
                d_model=self.config.model.transformer.d_model,
                n_heads=self.config.model.transformer.n_heads,
                n_layers=self.config.model.transformer.n_layers,
                dropout=self.config.model.transformer.dropout,
            )
        )

        # Load checkpoint. For testing, we might not provide a checkpoint, so we just skip this
        if self.config.checkpoint is not None:
            Checkpoint.load(
                self.config.checkpoint,
                device="cpu",
                config=None,
                model=self.model,
            )

        self.converter = HOVConverter(
            HOVConverterConfig(steps_per_beat=self.config.model.drums.steps_per_beat, categories=self.config.model.drums.categories, max_seq_len=config.model.max_seq_len)
        )

        self._max_seq_len = self.config.model.max_seq_len
        self._num_instruments = self.config.model.drums.num_instruments

        self._sequence = torch.zeros((self._max_seq_len, self._num_instruments, 3))
        self._pos_enc = torch.empty((self._max_seq_len, 4))

        self._playhead_position = 0

        self.reset()

    @property
    def context_start(self) -> int:
        # Ideally we start the context from the beginning of the sequence, but we have to
        # limit it to the max sequence length
        return max(0, self._playhead_position - self.config.model.max_seq_len + 1)

    @property
    def context_end(self) -> int:
        # Always end context at the current playhead position
        return self._playhead_position

    def reset(self):
        # Reset position and clear the sequence
        self._playhead_position = 0
        self._sequence = torch.zeros((self._max_seq_len, self._num_instruments, 3))
        self._update_pos_enc()

    def _update_pos_enc(self):
        # Generate positional encoding for the whole sequence.
        # Must match the formula used in hov_converter.py during preprocessing.
        steps_per_bar = self.config.model.drums.steps_per_beat * 4  # 4/4 time: 4 beats × 4 steps
        total_bars = max(1, self._max_seq_len // steps_per_bar)
        step_idx = np.arange(0, len(self._sequence)) % self._max_seq_len
        pos_in_bar = step_idx % steps_per_bar
        bar_idx = step_idx // steps_per_bar

        self._pos_enc = torch.from_numpy(self.converter.positional_encoding(bar_idx, pos_in_bar, total_bars))

    def _adjust_capacity(self):
        if self._playhead_position > 16000:
            logger.error("Cannot increase capacity past 16000! This would result in massive performance issues.")
            return

        # If the playhead reaches the end of the sequence, double its capacity
        if self._playhead_position + 2 >= len(self._sequence):
            self._sequence = torch.cat([self._sequence, torch.zeros_like(self._sequence)], dim=0)
            self._update_pos_enc()

    def process_step(self, step_hits: torch.Tensor):
        """Process one grid step of hit activations and advance the playhead.

        Args:
            step_hits: Binary tensor of shape (num_instruments,) indicating which instruments fired.

        The predicted offset and velocity values are stored in the internal sequence buffer
        and can be retrieved via get_generated_sequence().
        """
        with torch.no_grad():
            # Adjust the capacity (if necessary)
            self._adjust_capacity()

            # Decoder input will be a concatenation of the start token and
            default_ov = torch.zeros((self._num_instruments, 2))
            decoder_input = torch.cat(
                [
                    default_ov.unsqueeze(0),  # Start token
                    self._sequence[self.context_start : self.context_end, :, 1:3],  # Use OV-components (without hits)
                ],
                dim=0,
            )

            # Store the hits for the new step
            self._sequence[self.context_end, :, 0] = step_hits

            # Cut out the current sequence area from the hits and positional encoding.
            # Pos enc always starts from index 0 - training used sample_stride == max_seq_len,
            # so every training window had pos_enc starting at enc(0). We must match that here.
            encoder_input = self._sequence[self.context_start : self.context_end + 1, :, 0]  # Hits only
            window_len = self.context_end - self.context_start + 1
            pos_enc_input = self._pos_enc[0:window_len]

            # Run the model to get a full sequence prediction
            prediction = self.model(
                encoder_input.unsqueeze(0),
                decoder_input.unsqueeze(0),
                pos_enc_input.unsqueeze(0),
            )[0]

            # Store the OV component of last predicted timestep into the generated sequence
            self._sequence[self.context_end, :, 1:3] = torch.where(
                step_hits.unsqueeze(-1) > 0.5,  # If there was a hit
                prediction[-1],  # Then store the predicted OV-values
                default_ov,  # Else, store the default OV (zeros)
            )

            # Update the position
            self._playhead_position += 1

    def process_sequence(self, hits: torch.Tensor) -> torch.Tensor:
        """Process a full sequence of hit activations and return the generated HOV sequence.

        Args:
            hits: Tensor of shape (seq_len, num_instruments) with binary hit activations.

        Returns:
            Generated sequence tensor of shape (seq_len, num_instruments, 3).
        """
        logger.info(f"Predicting {len(hits)} steps ...")

        # Reset the current time
        self.reset()

        # Start processing each step
        for step_hits in hits:
            self.process_step(step_hits)

        # Return the full generated sequence
        return self.get_generated_sequence()

    def seek(self, step_position: int):
        # Move the playhead
        self._playhead_position = step_position
        self._adjust_capacity()

    def get_cached_sequence(self) -> torch.Tensor:
        return self._sequence

    def get_context_sequence(self) -> torch.Tensor:
        return self._sequence[self.context_start : self.context_end]

    def get_generated_sequence(self) -> torch.Tensor:
        return self._sequence[: self._playhead_position]

    def get_position(self) -> int:
        return self._playhead_position
