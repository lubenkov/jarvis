import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.config import context_length, checkpoint_path, max_new_tokens, temperature, top_k, top_p, repetition_penalty
from model.transformer import TransformerLM


def load_model(device):
    model = TransformerLM().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, device,
             max_new_tokens=max_new_tokens, temperature=temperature,
             top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty):
    ids = tokenizer.encode(prompt).ids
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    endoftext_id = tokenizer.token_to_id("<|endoftext|>")

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -context_length:]
        logits = model(ids_cond)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            for token_id in set(ids_cond[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(1)
            logits = torch.where(logits < min_value, torch.full_like(logits, float("-inf")), logits)

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == endoftext_id:
            break

        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    model = load_model(device)

    prompt = "- Привет! Как дела?\n"
    result = generate(model, tokenizer, prompt, device)
    print(result)