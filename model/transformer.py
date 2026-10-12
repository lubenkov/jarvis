import torch
import torch.nn as nn

from model.config import vocab_size, context_length, d_model, dropout, n_layers
from model.block import TransformerBlock


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        self.apply(_init_weights)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    model = TransformerLM()
    dummy_input = torch.randint(0, vocab_size, (4, context_length))
    output = model(dummy_input)
    print("Входной shape:", dummy_input.shape)
    print("Выходной shape:", output.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print("Количество параметров модели:", n_params)
    print("В миллионах:", round(n_params / 1e6, 2), "M")