import torch
import torch.nn as nn
from dataclasses import dataclass
from .encoder import Encoder, EncoderConfig
from .decoder import Decoder, DecoderConfig


@dataclass
class DequantConfig:
    max_seq_len: int = 128
    num_instruments: int = 7


class Dequant(nn.Module):
    def __init__(self, config: DequantConfig):
        super().__init__()
        self.config = config

        # HOV representation requires 3 values per instrument
        d_model = config.num_instruments * 3

        # Encoder and decoder blocks
        self.encoder = Encoder(EncoderConfig(d_model=d_model))
        self.decoder = Decoder(DecoderConfig(d_model=d_model))

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, num_instruments, hov_size = encoder_input.shape

        assert seq_len <= self.config.max_seq_len, "max_seq_len exceeded."
        assert num_instruments == self.config.num_instruments
        assert hov_size == 3

        # Flatten to d_model dimension
        encoder_flat = encoder_input.flatten(start_dim=2)
        decoder_flat = decoder_input.flatten(start_dim=2)

        # Encode the input sequence
        encoder_output: torch.Tensor = self.encoder(encoder_flat)

        # Decode with cross-attention to encoder
        y: torch.Tensor = self.decoder(decoder_flat, encoder_output)

        # Project and reshape
        y = self.output_proj(y)
        y = y.reshape(batch_size, seq_len, num_instruments, hov_size)

        # Apply different activations to hits, offsets, and velocities
        hits: torch.Tensor = self.sigmoid(y[..., 0:1])
        offsets: torch.Tensor = 0.5 * self.tanh(y[..., 1:2])
        velocities: torch.Tensor = self.sigmoid(y[..., 2:3])

        # Concatenate along last dimension
        return torch.cat([hits, offsets, velocities], dim=-1)
