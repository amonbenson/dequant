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
        d_encoder = config.num_instruments  # only hits
        d_decoder = config.num_instruments * 2  # offsets + velocities per instrument

        # Encoder and decoder blocks
        self.encoder = Encoder(EncoderConfig(d_model=d_encoder))
        self.decoder = Decoder(DecoderConfig(d_model=d_decoder, d_cross=d_encoder))

        # Output projection
        self.output_proj = nn.Linear(d_decoder, d_decoder)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, num_instruments = encoder_input.shape

        assert seq_len <= self.config.max_seq_len, "max_seq_len exceeded."
        assert num_instruments == self.config.num_instruments

        assert decoder_input.shape[0] == encoder_input.shape[0], (
            "inconsistent batch size"
        )
        assert decoder_input.shape[1] == encoder_input.shape[1], (
            "inconsistent sequence length"
        )
        assert decoder_input.shape[2] == encoder_input.shape[2], (
            "inconsistent number of instruments"
        )
        assert decoder_input.shape[3] == 2, "decoder input must have OV representations"

        # Flatten to d_model dimension
        encoder_flat = encoder_input.flatten(start_dim=2)
        decoder_flat = decoder_input.flatten(start_dim=2)

        # Encode the input sequence
        encoder_output: torch.Tensor = self.encoder(encoder_flat)

        # Decode with cross-attention to encoder
        y: torch.Tensor = self.decoder(decoder_flat, encoder_output)

        # Project and reshape
        y = self.output_proj(y)
        y = y.reshape(batch_size, seq_len, num_instruments, 2)

        # Apply different activations to hits, offsets, and velocities
        # hits: torch.Tensor = self.sigmoid(y[..., 0:1])
        offsets: torch.Tensor = 0.5 * self.tanh(y[..., 0:1])
        velocities: torch.Tensor = self.sigmoid(y[..., 1:2])

        # Concatenate along last dimension
        return torch.cat([offsets, velocities], dim=-1)
