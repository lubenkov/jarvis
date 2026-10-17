"""Сэмплирование из обученной модели: KV-кэш, temperature, top-k, top-p, repetition penalty.

KV-кэш — главная оптимизация инференса: без него каждый новый токен требует
полного пересчёта по всему контексту (O(n^2)), с ним — O(1) по прошлым токенам.

Пример:
    python generate.py --ckpt checkpoints/main/inference.pt --prompt "Привет!"
"""

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from chat_format import build_prompt, check_tokenizer, decode_ids, stop_ids
from model.config import (MAX_NEW_TOKENS, MODEL_PRESETS, ModelConfig, REPETITION_PENALTY,
                          TEMPERATURE, TOKENIZER_PATH, TOP_K, TOP_P)
from model.transformer import TransformerLM


def load_model(path, device, dtype=None):
    """Загружает чекпоинт train.py или сырой state_dict (в т.ч. inference.pt в fp16)."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        cfg_dict = state.get("model_config")
        model_cfg = MODEL_PRESETS["main"] if cfg_dict is None else ModelConfig(**cfg_dict)
        state_dict = state["model"]
    else:
        model_cfg = MODEL_PRESETS["main"]
        state_dict = state

    model = TransformerLM(model_cfg)
    model.load_state_dict(state_dict)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, model_cfg


@torch.no_grad()
def generate_ids(model, ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                 top_k=TOP_K, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY,
                 stop_token_ids=None, callback=None):
    """Генерирует продолжение. ids — список или тензор (1, T). Возвращает итоговый тензор."""
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    device = next(model.parameters()).device
    ids = ids.to(device)
    stop_token_ids = set(stop_token_ids or [])

    model.eval()
    caches = None
    for _ in range(max_new_tokens):
        x = ids if caches is None else ids[:, -1:]
        logits, caches = model(x, caches=caches, use_cache=True)
        logits = logits[:, -1, :].float()

        if repetition_penalty and repetition_penalty != 1.0:
            for token_id in set(ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = torch.where(logits < values[:, -1:], torch.full_like(logits, float("-inf")), logits)
            if top_p and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                sorted_logits = sorted_logits.masked_fill(cumulative - sorted_probs > top_p, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        ids = torch.cat([ids, next_id], dim=1)
        if callback is not None:
            callback(ids)
        if next_id.item() in stop_token_ids:
            break
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/main/inference.pt")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--prompt", default="Привет! Как дела?")
    parser.add_argument("--max-new", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--rep-penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer(tokenizer)
    model, model_cfg = load_model(args.ckpt, device)
    print(f"Модель загружена ({model.n_params() / 1e6:.1f}M параметров, {device.type})")

    ids = build_prompt(tokenizer, [{"role": "user", "content": args.prompt}])
    max_prompt_tokens = max(8, model_cfg.context_length - args.max_new)
    if len(ids) > max_prompt_tokens:  # длинный промпт не должен пробивать контекст/RoPE
        ids = ids[-max_prompt_tokens:]
    out = generate_ids(model, ids, max_new_tokens=args.max_new, temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.rep_penalty,
                       stop_token_ids=stop_ids(tokenizer))
    print(decode_ids(tokenizer, ids) + decode_ids(tokenizer, out[0].tolist()[len(ids):]))


if __name__ == "__main__":
    main()
