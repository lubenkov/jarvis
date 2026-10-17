"""Консольный чат с моделью. Диалог ведётся в том же формате, что и при обучении.

Пример:
    python chat.py --ckpt checkpoints/pretrain/inference.pt
"""

import argparse

import torch
from tokenizers import Tokenizer

from chat_format import build_prompt, check_tokenizer, decode_ids, stop_ids, truncate_messages
from generate import generate_ids, load_model
from model.config import (MAX_NEW_TOKENS, REPETITION_PENALTY, TEMPERATURE, TOKENIZER_PATH,
                          TOP_K, TOP_P)

MAX_HISTORY_MESSAGES = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/pretrain/inference.pt")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--rep-penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--max-new", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer(tokenizer)
    model, model_cfg = load_model(args.ckpt, device)
    stops = stop_ids(tokenizer)
    max_prompt_tokens = max(64, model_cfg.context_length - args.max_new)

    print(f"Мини-Джарвис готов: {args.ckpt} ({model.n_params() / 1e6:.1f}M параметров, {device.type})")
    print("Команды: /reset — очистить историю, /temp X — температура, exit — выход")

    history = []
    while True:
        try:
            user = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "выход"):
            break
        if user == "/reset":
            history = []
            print("История очищена.")
            continue
        if user.startswith("/temp"):
            try:
                args.temperature = float(user.split()[1])
            except (IndexError, ValueError):
                print("Пример: /temp 0.9")
                continue
            print(f"temperature = {args.temperature}")
            continue

        history.append({"role": "user", "content": user})
        while len(history) > MAX_HISTORY_MESSAGES:
            history.pop(0)
        messages = truncate_messages(tokenizer, history, max_prompt_tokens)
        ids = build_prompt(tokenizer, messages)
        if len(ids) > max_prompt_tokens:  # страховка от экзотической токенизации
            ids = ids[-max_prompt_tokens:]

        printed = {"n": 0}

        def show(full_ids):
            text = decode_ids(tokenizer, full_ids[0].tolist()[len(ids):])
            if len(text) > printed["n"]:
                print(text[printed["n"]:], end="", flush=True)
                printed["n"] = len(text)

        print("Джарвис: ", end="", flush=True)
        out = generate_ids(model, torch.tensor([ids], dtype=torch.long),
                           max_new_tokens=args.max_new, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p,
                           repetition_penalty=args.rep_penalty,
                           stop_token_ids=stops, callback=show)
        print()
        reply = decode_ids(tokenizer, out[0].tolist()[len(ids):]).strip() or "..."
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
