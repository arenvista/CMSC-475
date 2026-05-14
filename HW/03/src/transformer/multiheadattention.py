from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, proj_drop=0.1, atten_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # Linear layers to project input to q, k, v
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output linear layer
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout for attention weights and projections
        self.attn_dropout = nn.Dropout(atten_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1) Project inputs to q, k, v and reshape for multi-head processing
        # Reshape from (B, L, D) to (B, L, H, d_k) and transpose to (B, H, L, d_k)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2) Calculate scaled dot-product attention scores
        # scores = (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Convert scores to probabilities
        attn_weights = F.softmax(scores, dim=-1)
        
        # 3) Attention dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # 4) Calculate context vector and concatenate heads
        # x = attention_weights @ V
        x = torch.matmul(attn_weights, v)
        
        # Reshape back to (B, L, D)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # Apply final linear projection and projection dropout
        output = self.proj_dropout(self.out_proj(x))
        
        return output
