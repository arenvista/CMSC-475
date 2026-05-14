import torch
import torch.nn as nn
from transformer import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Use your Multi-Head Attention module
        self.self_attn = MultiHeadAttention(d_model, num_heads, proj_drop=dropout, atten_drop=dropout)
        
        # Define position-wise feed-forward network
        # Usually, the inner layer is 4x the d_model size
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Define two layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Define dropout layers if needed
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1) Attention Sub-layer
        # Applying "Pre-Norm" architecture (more stable for training)
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attn_output) # Residual connection
        
        # 2) Feed-Forward Sub-layer
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        output = x + self.dropout2(ff_output) # Residual connection
        
        return output
