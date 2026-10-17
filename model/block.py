import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.config import MODEL_PRESETS, ModelConfig
from model.norm import RMSNorm


class SwiGLU(nn.Module):
    """Feed-forward с gated-активацией: лучше обычного MLP при том же числе параметров."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm блок: x + attention(norm(x)), x + ffn(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ff = SwiGLU(cfg)

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_cache


if __name__ == "__main__":
    import torch

    cfg = MODEL_PRESETS["dev"]
    block = TransformerBlock(cfg)
    dummy_input = torch.randn(4, cfg.context_length, cfg.d_model)
    output, cache = block(dummy_input)
    print("Входной shape:", dummy_input.shape)
    print("Выходной shape:", output.shape)
