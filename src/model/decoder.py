import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class DecoderConfig:
    d_model: int
    dropout: float = 0.2


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # Causal self-attention
        self.self_qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)

        # Cross-attention to encoder
        self.cross_q_proj = nn.Linear(d_model, d_model)
        self.cross_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # Dropout layers
        self.self_attn_dropout = nn.Dropout(config.dropout)
        self.cross_attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout1 = nn.Dropout(config.dropout)
        self.resid_dropout2 = nn.Dropout(config.dropout)
        self.resid_dropout3 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        _batch_size, seq_len, _ = x.shape

        # Causal self-attention on decoder sequence
        residual = x
        x = self.ln1(x)

        qkv: torch.Tensor = self.self_qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        attn: torch.Tensor = (q @ k.transpose(-2, -1)) / np.sqrt(self.config.d_model)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.self_attn_dropout(attn)

        x = attn @ v
        x = self.self_attn_out(x)
        x = residual + self.resid_dropout1(x)

        # Cross-attention to encoder output (no causal mask)
        residual = x
        x = self.ln2(x)

        q: torch.Tensor = self.cross_q_proj(x)  # Q from decoder
        kv: torch.Tensor = self.cross_kv_proj(encoder_output)  # K, V from encoder
        k, v = kv.chunk(2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.config.d_model)
        attn = torch.softmax(attn, dim=-1)
        attn = self.cross_attn_dropout(attn)

        x = attn @ v
        x = self.cross_attn_out(x)
        x = residual + self.resid_dropout2(x)

        # Feed-forward
        x = x + self.resid_dropout3(self.ffn(self.ln3(x)))

        return x
