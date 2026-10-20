import torch
import torch.nn as nn

from model.block import TransformerBlock
from model.config import MODEL_PRESETS, ModelConfig
from model.norm import RMSNorm


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        cfg = cfg or MODEL_PRESETS["main"]
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.ln_final = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        self.apply(_init_weights)

    def forward(self, idx, caches=None, use_cache=False):
        x = self.dropout(self.token_embedding(idx))

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, new_cache = block(x, cache)
            if use_cache:
                new_caches.append(new_cache)

        logits = self.lm_head(self.ln_final(x))
        if use_cache:
            return logits, new_caches
        return logits

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n


if __name__ == "__main__":
    for name in ("dev", "main"):
        cfg = MODEL_PRESETS[name]
        model = TransformerLM(cfg)
        dummy_input = torch.randint(0, cfg.vocab_size, (4, cfg.context_length))
        output = model(dummy_input)
        n_params = model.n_params()
        print(f"--- preset {name} ---")
        print("Input shape:", dummy_input.shape)
        print("Output shape:", output.shape)
        print(f"Parameters: {n_params / 1e6:.1f}M (non-embedding: {model.n_params(True) / 1e6:.1f}M)")

    cfg = MODEL_PRESETS["dev"]
    model = TransformerLM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    full_logits = model(ids)

    caches = None
    step_logits = []
    for t in range(16):
        logits, caches = model(ids[:, t:t + 1], caches=caches, use_cache=True)
        step_logits.append(logits)
    incremental_logits = torch.cat(step_logits, dim=1)

    diff = (full_logits - incremental_logits).abs().max().item()
    print("Max difference KV cache vs full pass:", diff)
    assert diff < 1e-5, "KV cache is incorrect"
    print("OK")
