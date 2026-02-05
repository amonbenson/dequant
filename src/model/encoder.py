from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    d_model: int
    dropout: float = 0.2


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # Self-attention (bidirectional, no causal mask)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)  # on attention weights
        self.resid_dropout1 = nn.Dropout(config.dropout)  # after attention
        self.resid_dropout2 = nn.Dropout(config.dropout)  # after FFN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bidirectional self-attention
        residual = x
        x = self.ln1(x)

        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        attn: torch.Tensor = (q @ k.transpose(-2, -1)) / np.sqrt(self.config.d_model)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = attn @ v
        x = self.attn_out(x)
        x = residual + self.resid_dropout1(x)

        # Feed-forward
        x = x + self.resid_dropout2(self.ffn(self.ln2(x)))

        return x
