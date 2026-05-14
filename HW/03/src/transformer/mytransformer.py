import torch.nn as nn
from transformer import *

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1, max_seq_length=512):
        super(Transformer, self).__init__()
        
        # 1. Token embedding and Positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 2. Encoder stack (N layers)
        # Using nn.ModuleList to hold the stack of layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # 3. Decoder stack (N layers)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dropout, dropout) 
            for _ in range(num_layers)
        ])
        
        # 4. Output projection layer
        # Maps the d_model dimension back to vocab_size for prediction
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # --- Encoder Pass ---
        # Embed the source and add positional encoding
        enc_out = self.dropout(self.pos_encoding(self.embedding(src)))
        
        # Pass through each encoder layer
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
            
        # --- Decoder Pass ---
        # Embed the target and add positional encoding
        dec_out = self.dropout(self.pos_encoding(self.embedding(tgt)))
        
        # Pass through each decoder layer
        # Note: Decoder layers take enc_out (the context) and masks
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
        # --- Final Projection ---
        output = self.fc_out(dec_out)
        
        return output
