from transformer import MultiHeadAttention
import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, proj_drop=0.1, atten_drop=0.1):
        super(DecoderLayer, self).__init__()
        
        # 1. Masked self-attention (attends to previous tokens in the output)
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, proj_drop, atten_drop)
        
        # 2. Encoder-decoder attention (attends to the encoder's output)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, proj_drop, atten_drop)
        
        # 3. Position-wise feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Define three layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(proj_drop)
        self.dropout2 = nn.Dropout(proj_drop)
        self.dropout3 = nn.Dropout(proj_drop)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1) Masked Multi-Head Self-Attention
        # x is the target sequence
        x_norm = self.norm1(x)
        # Self-attention uses x for Q, K, and V
        self_attn_out = self.masked_self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(self_attn_out)
        
        # 2) Encoder-Decoder (Cross) Attention
        # Here, Query comes from the decoder (x), 
        # but Key and Value come from the encoder (enc_out)
        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(x_norm, enc_out, enc_out, src_mask)
        x = x + self.dropout2(cross_attn_out)
        
        # 3) Position-wise Feed-Forward Network
        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        output = x + self.dropout3(ff_out)
        
        return output
