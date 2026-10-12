import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import d_model, n_heads, context_length, dropout


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)


if __name__ == "__main__":
    attn = MultiHeadAttention()
    dummy_input = torch.randn(4, context_length, d_model)
    output = attn(dummy_input)
    print("Входной shape:", dummy_input.shape)
    print("Выходной shape:", output.shape)