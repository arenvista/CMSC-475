from transformer import *
def main():
    print("Hello from 03!")

if __name__ == "__main__":
# Model Hyperparameters
    VOCAB_SIZE = 5000
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    MAX_SEQ_LEN = 100

# Initialize Model
    model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, max_seq_length=MAX_SEQ_LEN)

# Create dummy input data (Batch Size=2, Sequence Length=10)
# LongTensors represent token indices (like from a tokenizer)
    src_data = torch.randint(0, VOCAB_SIZE, (2, 10)) # type: ignore
    tgt_data = torch.randint(0, VOCAB_SIZE, (2, 10)) # type: ignore

# Generate a Causal Mask for the Decoder 
# (This prevents the decoder from "cheating" by looking at future tokens)
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) # type: ignore
        return mask == 0

    tgt_mask = create_causal_mask(10)

# Forward Pass
    output = model(src_data, tgt_data, tgt_mask=tgt_mask)

    print(f"Output Shape: {output.shape}") # Expected: [2, 10, 5000]
