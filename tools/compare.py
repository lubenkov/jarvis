import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer

from chat_format import build_prompt, check_tokenizer, decode_ids, stop_ids
from generate import generate_ids, load_model
from model.config import (MAX_NEW_TOKENS, REPETITION_PENALTY, TEMPERATURE, TOKENIZER_PATH, TOP_K, TOP_P)


def short_name(path):
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="checkpoints/pretrain/inference.pt", help="checkpoint before")
    parser.add_argument("--b", default="checkpoints/sft/inference.pt", help="checkpoint after")
    parser.add_argument("--prompts", default="golden_prompts.json")
    parser.add_argument("--out", default=None, help="default: samples/compare_<a>_vs_<b>.txt")
    parser.add_argument("--limit", type=int, default=0, help="take only the first N prompts")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="0 — greedy generation")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--rep-penalty", type=float, default=REPETITION_PENALTY)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer(tokenizer)
    stops = stop_ids(tokenizer)

    with open(args.prompts, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    models = {}
    cfg = None
    for name, path in (("A", args.a), ("B", args.b)):
        models[name], cfg = load_model(path, device)
        print(f"{name}: {path} — {models[name].n_params() / 1e6:.1f}M params, {device.type}", flush=True)
    max_prompt = max(8, cfg.context_length - args.max_new)

    a_name, b_name = short_name(args.a), short_name(args.b)
    out_path = args.out or os.path.join("samples", f"compare_{a_name}_vs_{b_name}.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines = [
        f"# comparison: {a_name} ({args.a}) vs {b_name} ({args.b})",
        f"# temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"rep_penalty={args.rep_penalty} max_new={args.max_new} seed={args.seed}",
        "",
    ]
    summary = []
    spent = {"A": 0.0, "B": 0.0}

    for idx, item in enumerate(items, start=1):
        ids = build_prompt(tokenizer, item["messages"])
        if len(ids) > max_prompt:
            ids = ids[-max_prompt:]
        prompt_text = " | ".join(m["content"] for m in item["messages"])
        lines.append(f"=== [{idx}/{len(items)}] USER: {prompt_text}")

        row = [idx]
        for name in ("A", "B"):
            torch.manual_seed(args.seed + idx)
            started = time.time()
            out = generate_ids(models[name], torch.tensor([ids], dtype=torch.long),
                               max_new_tokens=args.max_new, temperature=args.temperature,
                               top_k=args.top_k, top_p=args.top_p,
                               repetition_penalty=args.rep_penalty, stop_token_ids=stops)
            elapsed = time.time() - started
            spent[name] += elapsed
            reply_ids = out[0].tolist()[len(ids):]
            reply = decode_ids(tokenizer, reply_ids).strip()
            lines.append(f"--- {a_name if name == 'A' else b_name}: {len(reply_ids)} tokens, {elapsed:.1f} s")
            lines.append(reply or "(empty)")
            lines.append("")
            row.append(len(reply_ids))
            print(f"[{idx}/{len(items)}] {name} {a_name if name == 'A' else b_name}: "
                  f"{len(reply_ids)} tokens, {elapsed:.1f} s", flush=True)
        summary.append(row)

    mean_a = sum(r[1] for r in summary) / max(1, len(summary))
    mean_b = sum(r[2] for r in summary) / max(1, len(summary))
    lines.append("# summary")
    lines.append(f"{'#':>3} {a_name:>6} {b_name:>6}")
    for idx, len_a, len_b in summary:
        lines.append(f"{idx:>3} {len_a:>6} {len_b:>6}")
    lines.append(f"average reply length: {a_name} {mean_a:.0f} tokens, {b_name} {mean_b:.0f} tokens")
    lines.append(f"time: {a_name} {spent['A']:.0f} s, {b_name} {spent['B']:.0f} s")

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
