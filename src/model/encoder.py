from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    d_model: int
    n_heads: int = 1
    dropout: float = 0.2


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_head = d_model // n_heads

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
        B, T, C = x.shape
        n_heads = self.config.n_heads
        d_head = self.d_head

        # Bidirectional self-attention
        residual = x
        x = self.ln1(x)

        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, n_heads, d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.view(B, T, n_heads, d_head).transpose(1, 2)
        v = v.view(B, T, n_heads, d_head).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0.0)  # (B, n_heads, T, d_head)
        x = x.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        x = self.attn_out(x)
        x = residual + self.resid_dropout1(x)

        # Feed-forward
        x = x + self.resid_dropout2(self.ffn(self.ln2(x)))

        return x
