import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_length=32, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create poistional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100000.0) / d_model))

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
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

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
    
class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # Fully Connected / Dense Layer
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

    
class EncoderLayer(nn.Module):
    """Single Transformer encoder layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """

        Args:
            x : (batch_size, seq_length, d_model)
            mask : Optional Attention Mask.

        Returns:
            (batch_size, seq_length, d_model)
        """

        # self-attention with skip connection and layer normalization
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # feed-forward with skip connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x
    
class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder"""
    def __init__(
            self, 
            input_dim=27, # HVO matrices (9+9+9)
            d_model = 512,
            num_heads = 4,
            num_layers = 6,
            d_ff = 16,
            max_seq_length = 32,
            dropout = 0.1 
        ):
        
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim

        # Input projection (from input_dim to d_model)
        self.input_projection = nn.Linear(input_dim, d_model)

        # Pos Enc
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection 
        self.output_projection = nn.Linear(d_model, input_dim)

        # Init Params
        self._init_parameters()

    def _init_parameters(self):
        """Initialize trainable parameters with Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)

        # Project back to output dimension
        output = self.output_projection(x)

        return output
    


# Example for the paper's Model 1 configuration

def create_model_1():
    """Create Model 1 from the Pompeo Fabra paper"""
    return TransformerEncoder(
        input_dim=27,
        d_model=512,
        num_heads=4,
        num_layers=6,
        d_ff=16,
        max_seq_length=32,
        dropout=0.1
    )



# ==========================
# LOSS FUNCTION
# ==========================

class DrumLoss(nn.Module):
    """Custom Loss Function combines BCE for hits and MSE for velocity and offsets"""

    def __init__(self, hit_loss_weight=1.0, vel_loss_weight=1.0, offset_loss_weight=1.0):
        super().__init__()
        self.hit_loss_weight = hit_loss_weight
        self.vel_loss_weight = vel_loss_weight
        self.offset_loss_weight = offset_loss_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """

        Args:
            predictions (batch_size, 32, 27): 
            targets (batch_size, 32, 27): 
        """

        # split predictions and targets into HVO
        pred_hits = predictions[:, :, :9]
        pred_vels = predictions[:, :, 9:18]
        pred_offsets = predictions[:, :, 18:]

        target_hits = targets[:, :, :9]
        target_vels = targets[:, :, 9:18]
        target_offsets = targets[:, :, 18:]

        # calc losses
        hit_loss = self.bce_loss(pred_hits, target_hits)
        vel_loss = self.mse_loss(pred_vels, target_vels)
        offset_loss = self.mse_loss(pred_offsets, target_offsets)

        # Weighted Sum
        total_loss = (
            self.hit_loss_weight * hit_loss + 
            self.vel_loss_weight * vel_loss + 
            self.offset_loss_weight * offset_loss
        )

        return total_loss, hit_loss, vel_loss, offset_loss

    