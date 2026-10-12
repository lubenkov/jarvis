import torch.nn as nn

from model.config import d_model, d_ff, dropout
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


if __name__ == "__main__":
    import torch
    from model.config import context_length

    block = TransformerBlock()
    dummy_input = torch.randn(4, context_length, d_model)
    output = block(dummy_input)
    print("Входной shape:", dummy_input.shape)
    print("Выходной shape:", output.shape)