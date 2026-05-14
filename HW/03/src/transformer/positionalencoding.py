import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_seq_length: int):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model) # type: ignore
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # type: ignore
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # type: ignore
        
        pe[:, 0::2] = torch.sin(position * div_term) # type: ignore
        pe[:, 1::2] = torch.cos(position * div_term) # type: ignore
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length, :]
