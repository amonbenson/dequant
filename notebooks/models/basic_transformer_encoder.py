import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    
    def __ini__(self, d_model, max_seq_length=32, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create poistional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_0000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # shape : (1, max_seq_length, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        Args:
            x : Tensor of shape (batch_size, seq_length, d_model)
        """

        x = x + self.pe[:, :x.size(1), :] # slicing for shorter sequences
        # PE is actually just added on top of the values
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    """Multihead self-attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projection
        self.W_q = nn.Linear(d_model, d_model) # Preserves dimensions
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model) 

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """Split the model dimension (d_model) into num_heads, d_k
        
        Output shape: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)
    
    def combine_heads(self, x):
        """Combines heads back together"""
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, x, mask=None):
        """

        Args:
            x : Input tensor (batch_size, seq_length, d_model)
            mask : Optional attention mask (batch_size, 1, 1, seq_length). Defaults to None.
        """
        batch_size = x.size(0)

        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split into multiple heads
        Q = self.split_heads(Q) # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Attenuate values
        attention_output = torch.matmul(attention_weights, V)

        # Combine Heads
        attention_output = self.combine_heads(attention_output)

        # Final linear projection
        output = self.W_o(attention_output)

        return output
    
class FeedForwards(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # Fully Connected / Dense Layer
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self.linear2(self.dropout(F.relu(self.linear1(x))))