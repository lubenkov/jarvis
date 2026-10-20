import argparse
import glob
import gzip
import json
import lzma
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tokenizers import Tokenizer

from chat_format import check_tokenizer, decode_ids
from generate import load_model
from model.config import TOKENIZED_DIR, TOKENIZER_PATH

LN2 = math.log(2)
BASELINE_ID = "ai-forever/rugpt3small_based_on_gpt2"


def short_name(path):
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or os.path.splitext(os.path.basename(path))[0]


def load_val_ids(tokenized_dir, source):
    shards = sorted(glob.glob(os.path.join(tokenized_dir, "val", source, "shard_*.bin")))
    shards = [s for s in shards if not s.endswith(".mask.bin")]
    if not shards:
        raise FileNotFoundError(f"No validation shards for '{source}' in {tokenized_dir}/val/{source}")
    ids = []
    for path in shards:
        ids.append(np.fromfile(path, dtype=np.uint16).astype(np.int64))
    return np.concatenate(ids)


def window_batches(ids, ctx, batch):
    xs, ys = [], []
    for start in range(0, len(ids) - 1, ctx):
        piece = ids[start:start + ctx + 1]
        xs.append(piece[:-1])
        ys.append(piece[1:])
    out = []
    for i in range(0, len(xs), batch):
        chunk_x, chunk_y = xs[i:i + batch], ys[i:i + batch]
        width = max(len(x) for x in chunk_x)
        padded_x, padded_y, mask = [], [], []
        for x, y in zip(chunk_x, chunk_y):
            pad = width - len(x)
            padded_x.append(np.concatenate([x, np.zeros(pad, dtype=np.int64)]))
            padded_y.append(np.concatenate([y, np.zeros(pad, dtype=np.int64)]))
            mask.append(np.concatenate([np.ones(len(x), dtype=np.float32), np.zeros(pad, dtype=np.float32)]))
        out.append((np.stack(padded_x), np.stack(padded_y), np.stack(mask)))
    return out


@torch.no_grad()
def total_nats(model, ids, ctx, batch, device, label="", every=20):
    nats = 0.0
    tokens = 0
    batches = window_batches(ids, ctx, batch)
    for idx, (xs, ys, mask) in enumerate(batches, start=1):
        out = model(torch.from_numpy(xs).to(device))
        logits = out if torch.is_tensor(out) else out.logits
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), torch.from_numpy(ys).view(-1).to(device), reduction="none")
        losses = losses.view(mask.shape).double() * torch.from_numpy(mask).double().to(device)
        nats += losses.sum().item()
        tokens += int(mask.sum())
        if every and idx % every == 0:
            print(f"    {label}: {idx}/{len(batches)} windows, {tokens / 1000:.0f}k tokens", flush=True)
    return nats, tokens


def compression_bits(text):
    raw = text.encode("utf-8")
    return {"gzip": 8 * len(gzip.compress(raw, 6)) / len(text),
            "xz": 8 * len(lzma.compress(raw, preset=6)) / len(text)}


def score(nats, tokens, chars):
    return {"bits_per_token": nats / LN2 / tokens,
            "chars_per_token": chars / tokens,
            "bpc": nats / LN2 / chars,
            "ppl": math.exp(nats / tokens),
            "tokens": tokens,
            "chars": chars}


def main():
    parser = argparse.ArgumentParser(description="BPC on held-out Russian text")
    parser.add_argument("--ckpt", nargs="+", default=["checkpoints/sft/inference.pt"],
                        help="custom checkpoints; 'none' — baseline only")
    parser.add_argument("--baseline", default=BASELINE_ID, help="HF model for comparison, 'none' — skip")
    parser.add_argument("--sources", default="web_ru,texts_ru,chat_ru")
    parser.add_argument("--chars", type=int, default=600_000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokenized-dir", default=TOKENIZED_DIR)
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/bpc.json")
    parser.add_argument("--md", default="results/bpc.md")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer(tokenizer)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    texts = {}
    for source in sources:
        ids = load_val_ids(args.tokenized_dir, source)
        text = decode_ids(tokenizer, ids.tolist())[:args.chars]
        if len(text) < 1000:
            raise ValueError(f"Text too short for '{source}': {len(text)} chars")
        texts[source] = text

    payload = {"config": {"chars": args.chars, "batch": args.batch,
                          "baseline": args.baseline, "tokenizer": args.tokenizer},
               "systems": {}}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("config", {}).get("chars") == args.chars:
            payload["systems"] = old.get("systems", {})
            payload["compression"] = old.get("compression", {})
            print(f"{args.out}: appending to existing systems", flush=True)
        else:
            print(f"{args.out}: different sample size, starting over", flush=True)

    for path in [p for p in args.ckpt if p.lower() != "none"]:
        name = short_name(path)
        print(f"[{name}] {path}", flush=True)
        model, cfg = load_model(path, device)
        rows = {}
        for source in sources:
            text = texts[source]
            ids = np.array(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int64)
            started = time.time()
            nats, tokens = total_nats(model, ids, cfg.context_length, args.batch, device, f"{name}/{source}")
            rows[source] = score(nats, tokens, len(text))
            rows[source]["seconds"] = round(time.time() - started, 1)
            print(f"  {source}: {rows[source]['bpc']:.3f} bits/char "
                  f"({rows[source]['ppl']:.2f} ppl, {rows[source]['chars_per_token']:.2f} chars/token, "
                  f"{rows[source]['seconds']} s)", flush=True)
        payload["systems"][name] = rows
        del model

    if args.baseline and args.baseline.lower() != "none":
        try:
            import transformers

            print(f"[{args.baseline}] loading from Hugging Face", flush=True)
            baseline_tok = transformers.AutoTokenizer.from_pretrained(args.baseline)
            try:
                baseline_model = transformers.AutoModelForCausalLM.from_pretrained(
                    args.baseline, dtype=torch.float32)
            except TypeError:
                baseline_model = transformers.AutoModelForCausalLM.from_pretrained(
                    args.baseline, torch_dtype=torch.float32)
            baseline_model.to(device).eval()
            baseline_model.config.use_cache = False
            ctx = getattr(baseline_model.config, "n_positions", None) or \
                getattr(baseline_model.config, "max_position_embeddings", 1024)
            ctx = min(int(ctx), 1024)
            rows = {}
            for source in sources:
                text = texts[source]
                ids = np.array(baseline_tok(text)["input_ids"], dtype=np.int64)
                started = time.time()
                nats, tokens = total_nats(baseline_model, ids, ctx, args.batch, device, f"baseline/{source}")
                rows[source] = score(nats, tokens, len(text))
                rows[source]["seconds"] = round(time.time() - started, 1)
                print(f"  {source}: {rows[source]['bpc']:.3f} bits/char "
                      f"({rows[source]['ppl']:.2f} ppl, {rows[source]['chars_per_token']:.2f} chars/token, "
                      f"{rows[source]['seconds']} s)", flush=True)
            payload["systems"][args.baseline.split("/")[-1]] = rows
        except ImportError:
            print("transformers is not installed — HF comparison skipped", flush=True)
        except Exception as error:
            print(f"HF model failed to load: {error}", flush=True)
            print("try another id, for example sberbank-ai/rugpt3small_based_on_gpt2", flush=True)

    payload["compression"] = {source: compression_bits(texts[source]) for source in sources}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    header = ["system"] + sources
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, rows in payload["systems"].items():
        cells = [f"{rows[s]['bpc']:.2f}" if s in rows else "—" for s in sources]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    for name in ("gzip", "xz"):
        cells = [f"{payload['compression'][s][name]:.2f}" for s in sources]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    with open(args.md, "w", encoding="utf-8") as f:
        f.write("BPC (bits per character, lower is better)\n\n" + table + "\n")

    print("\n" + table)
    print(f"\nwritten: {args.out}, {args.md}")


if __name__ == "__main__":
    main()
