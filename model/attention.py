import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MODEL_PRESETS, ModelConfig
from model.rope import RoPE


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        cfg = cfg or MODEL_PRESETS["main"]
        assert cfg.d_model % cfg.n_heads == 0

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RoPE(self.head_dim, cfg.rope_theta)

    def forward(self, x, cache=None):
        """x: (batch, seq, d_model). cache: (k, v) прошлых шагов или None.

        Возвращает (выход, новая пара (k, v)).
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        offset = 0 if cache is None else cache[0].shape[2]
        q, k = self.rope.rotate(q, k, offset)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # is_causal корректен, когда запросов столько же, сколько ключей (префилл).
        # При генерации по одному токену маска не нужна: все ключи уже в прошлом.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(seq_len == k.shape[2]),
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output), (k, v)


if __name__ == "__main__":
    from model.config import MODEL_PRESETS

    cfg = MODEL_PRESETS["dev"]
    attn = MultiHeadAttention(cfg)
    dummy_input = torch.randn(4, cfg.context_length, cfg.d_model)
    output, cache = attn(dummy_input)
    print("Входной shape:", dummy_input.shape)
    print("Выходной shape:", output.shape)
    print("KV-кэш:", cache[0].shape, cache[1].shape)
