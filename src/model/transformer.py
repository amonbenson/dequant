import torch
import torch.nn as nn
from dataclasses import dataclass
from .encoder import Encoder, EncoderConfig
from .decoder import Decoder, DecoderConfig


@dataclass
class DequantTransformerConfig:
    max_seq_len: int = 128
    num_instruments: int = 9
    d_model: int = 128
    dropout: float = 0.2


class DequantTransformer(nn.Module):
    def __init__(self, config: DequantTransformerConfig):
        super().__init__()
        self.config = config

        # Encoder receives only hits, while decoder receives offsets + velocities
        d_encoder_input = config.num_instruments * 1
        d_decoder_input = config.num_instruments * 2
        d_model = config.d_model
        dropout = config.dropout

        self.pos_proj = nn.Linear(4, d_model)

        # Input projections
        self.encoder_input_proj = nn.Linear(d_encoder_input, d_model)
        self.decoder_input_proj = nn.Linear(d_decoder_input, d_model)

        # Dropout after projections
        self.dropout = nn.Dropout(config.dropout)

        # Encoder and decoder blocks
        self.encoder = Encoder(EncoderConfig(d_model=d_model, dropout=dropout))
        self.decoder = Decoder(DecoderConfig(d_model=d_model, dropout=dropout))

        # Output projection
        self.output_proj = nn.Linear(d_model, d_decoder_input)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        # Validate inputs
        assert encoder_input.dtype == torch.float32, f"encoder_input must be float32, not {encoder_input.dtype}"
        assert decoder_input.dtype == torch.float32, f"decoder_input must be float32, not {encoder_input.dtype}"

        assert len(encoder_input.shape) == 3, f"Expected encoder_input to be of shape (batch_size, seq_len, num_instruments), but got shape ({encoder_input.shape})"
        assert len(decoder_input.shape) == 4, f"Expected decoder_input to be of shape (batch_size, seq_len, num_instruments, 2), but got shape ({decoder_input.shape})"

        assert encoder_input.shape[0] == decoder_input.shape[0], f"batch_size mismatch: {encoder_input.shape[0]} (encoder_input) != {decoder_input.shape[0]} (decoder_input)"
        assert encoder_input.shape[1] == decoder_input.shape[1], f"seq_len mismatch: {encoder_input.shape[1]} (encoder_input) != {decoder_input.shape[1]} (decoder_input)"
        assert encoder_input.shape[2] == decoder_input.shape[2], f"n_instruments mismatch: {encoder_input.shape[2]} (encoder_input) != {decoder_input.shape[2]} (decoder_input)"
        assert decoder_input.shape[3] == 2, f"expected decoder_input HOV dimension to only include offsets + velocities (size 2), but was {decoder_input.shape[3]}"

        batch_size, seq_len, num_instruments = encoder_input.shape
        assert seq_len <= self.config.max_seq_len, "max_seq_len exceeded."
        assert num_instruments == self.config.num_instruments, f"received num_instruments ({num_instruments}) is different from the configuration ({self.config.num_instruments})"
        # PE
        assert pos_enc.dtype == torch.float32
        assert pos_enc.shape[:2] == (batch_size, seq_len)
        assert pos_enc.shape[2] == 4

        # Flatten instrument dimension
        encoder_flat = encoder_input.flatten(start_dim=2)  # this is a nop, but we might add other info later
        decoder_flat = decoder_input.flatten(start_dim=2)

        # Project to model dimension
        encoder_emb = self.dropout(self.encoder_input_proj(encoder_flat))
        decoder_emb = self.dropout(self.decoder_input_proj(decoder_flat))

        pos_emb = self.pos_proj(pos_enc)
        encoder_emb = encoder_emb + pos_emb
        decoder_emb = decoder_emb + pos_emb

        # Transformer forward pass
        encoder_output: torch.Tensor = self.encoder(encoder_emb)
        y: torch.Tensor = self.decoder(decoder_emb, encoder_output)

        # Project back to output dimension
        y = self.output_proj(y)
        y = y.reshape(batch_size, seq_len, num_instruments, 2)

        # Apply different activations to hits, offsets, and velocities
        # hits: torch.Tensor = self.sigmoid(y[..., 0:1])
        offsets: torch.Tensor = 0.5 * self.tanh(y[..., 0:1])
        velocities: torch.Tensor = self.sigmoid(y[..., 1:2])

        # Concatenate along last dimension
        return torch.cat([offsets, velocities], dim=-1)
