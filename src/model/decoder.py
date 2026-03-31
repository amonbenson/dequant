from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
    d_model: int
    n_heads: int = 1
    dropout: float = 0.2


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_head = d_model // n_heads

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
        """Apply one decoder block (causal self-attention + cross-attention + FFN).

        Args:
            x: Decoder sequence of shape (B, T, C).
            encoder_output: Encoder output of shape (B, T_enc, C).
        """
        B, T, C = x.shape
        T_enc = encoder_output.shape[1]
        n_heads = self.config.n_heads
        d_head = self.d_head

        # Causal self-attention on decoder sequence
        residual = x
        x = self.ln1(x)

        qkv: torch.Tensor = self.self_qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, n_heads, d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.view(B, T, n_heads, d_head).transpose(1, 2)
        v = v.view(B, T, n_heads, d_head).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )  # (B, n_heads, T, d_head)
        x = x.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        x = self.self_attn_out(x)
        x = residual + self.resid_dropout1(x)

        # Cross-attention to encoder output (no causal mask)
        residual = x
        x = self.ln2(x)

        q: torch.Tensor = self.cross_q_proj(x)  # Q from decoder
        kv: torch.Tensor = self.cross_kv_proj(encoder_output)  # K, V from encoder
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, T, n_heads, d_head).transpose(1, 2)
        k = k.view(B, T_enc, n_heads, d_head).transpose(1, 2)
        v = v.view(B, T_enc, n_heads, d_head).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.config.dropout if self.training else 0.0,
        )  # (B, n_heads, T, d_head)
        x = x.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        x = self.cross_attn_out(x)
        x = residual + self.resid_dropout2(x)

        # Feed-forward
        x = x + self.resid_dropout3(self.ffn(self.ln3(x)))

        return x
