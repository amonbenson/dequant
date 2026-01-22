import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Self-attention (bidirectional, no causal mask)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Bidirectional self-attention
        residual = x
        x = self.ln1(x)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn = torch.softmax(attn, dim=-1)

        x = attn @ v
        x = self.attn_out(x)
        x = residual + x

        # Feed-forward
        x = x + self.ffn(self.ln2(x))

        return x
