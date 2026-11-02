import argparse
import glob
import hashlib
import json
import os
import shutil

import numpy as np
from tokenizers import Tokenizer

from chat_format import check_tokenizer, conversation_mask, format_conversation, prepare_messages
from model.config import (CORPUS_DIR, SYSTEM_PROMPTS_FILE, TOKENIZED_DIR, TOKENIZER_PATH,
                          VAL_SKIP_TOKENS, VAL_TOKENS_PER_SOURCE)

SHARD_TOKENS = 25_000_000
ENCODE_BATCH = 512
MAX_DOC_CHARS = 40000
MAX_DIALOGUE_TOKENS = 4096

FILES = {
    "web_ru": ("web_ru.jsonl", "text"),
    "texts_ru": ("texts_ru.jsonl", "text"),
    "wiki_ru": ("wiki_ru.jsonl", "text"),
    "chat_ru": ("chat_ru.jsonl", "chat"),
}


class ShardWriter:
    def __init__(self, out_dir, with_mask):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.with_mask = with_mask
        self.pending = np.zeros(0, dtype=np.uint16)
        self.pending_mask = np.zeros(0, dtype=np.uint8)
        self.index = 0
        self.total_tokens = 0

    def add(self, ids, mask=None):
        self.pending = np.concatenate([self.pending, np.asarray(ids, dtype=np.uint16)])
        if self.with_mask:
            mask = mask if mask is not None else [1] * len(ids)
            self.pending_mask = np.concatenate([self.pending_mask, np.asarray(mask, dtype=np.uint8)])
        while len(self.pending) >= SHARD_TOKENS:
            self._write_shard(SHARD_TOKENS)

    def _write_shard(self, n):
        path = os.path.join(self.out_dir, f"shard_{self.index:04d}.bin")
        self.pending[:n].tofile(path)
        if self.with_mask:
            self.pending_mask[:n].tofile(path[:-4] + ".mask.bin")
        self.pending = self.pending[n:]
        self.pending_mask = self.pending_mask[n:]
        self.index += 1
        self.total_tokens += n

    def close(self):
        if len(self.pending) > 0:
            self._write_shard(len(self.pending))
        print(f"  {self.out_dir}: {self.total_tokens / 1e6:.1f}M tokens, shards: {self.index}")


class Router:
    def __init__(self, train_writer, val_writer, val_skip, val_tokens):
        self.train = train_writer
        self.val = val_writer
        self.val_skip = val_skip
        self.val_tokens = val_tokens
        self.seen = 0

    def add(self, ids, mask=None):
        i, n = 0, len(ids)
        while i < n:
            if self.seen < self.val_skip:
                take = min(n - i, self.val_skip - self.seen)
                i += take
                self.seen += take
            elif self.seen < self.val_skip + self.val_tokens:
                take = min(n - i, self.val_skip + self.val_tokens - self.seen)
                self.val.add(ids[i:i + take], mask[i:i + take] if mask is not None else None)
                i += take
                self.seen += take
            else:
                self.train.add(ids[i:], mask[i:] if mask is not None else None)
                self.seen += n - i
                i = n


def load_system_prompts(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f
                if line.strip() and not line.lstrip().startswith("#")]


def pick_system(text, prompts, probability, seed):
    """Deterministic per-document choice, so re-tokenizing gives identical shards."""
    if not prompts or probability <= 0:
        return ""
    digest = hashlib.md5(f"{seed}:{text[:500]}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big")
    if value % 1000 >= int(probability * 1000):
        return ""
    return prompts[value % len(prompts)]


def source_files(corpus_dir, name):
    single = os.path.join(corpus_dir, f"{name}.jsonl")
    parts = sorted(p for p in glob.glob(os.path.join(corpus_dir, f"{name}.*.jsonl"))
                   if os.path.basename(p) != f"{name}.jsonl")
    if parts:
        return parts
    return [single] if os.path.exists(single) else []


def process_source(source_name, paths, router, tokenizer, system_prompts=None,
                   system_prob=0.0, system_seed=0, dry_run_docs=0):
    kind = FILES[source_name][1]
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    pending_texts = []
    count = 0

    def flush():
        nonlocal pending_texts
        if not pending_texts:
            return
        for encoding in tokenizer.encode_batch(pending_texts):
            ids = encoding.ids[:MAX_DIALOGUE_TOKENS] if kind == "chat" else encoding.ids
            if kind == "chat":
                router.add(ids, conversation_mask(ids, tokenizer))
            else:
                router.add(ids + [eos_id], None)
        pending_texts = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind == "chat":
                    prepared = prepare_messages(ex.get("messages") or [])
                    if not prepared:
                        continue
                    text = format_conversation(prepared)
                    system = pick_system(text, system_prompts or [], system_prob, system_seed)
                    if system:
                        text = format_conversation([("system", system)] + prepared)
                    pending_texts.append(text)
                else:
                    text = (ex.get("text") or "").strip()
                    if not text:
                        continue
                    pending_texts.append(text[:MAX_DOC_CHARS])
                count += 1
                if len(pending_texts) >= ENCODE_BATCH:
                    flush()
                    if count % 50000 < ENCODE_BATCH:
                        print(f"  ... {count} records, total tokens: {router.seen / 1e6:.1f}M")
                if dry_run_docs and count >= dry_run_docs:
                    break
            if dry_run_docs and count >= dry_run_docs:
                break
    flush()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(FILES) + ["all"], default="all")
    parser.add_argument("--corpus-dir", default=CORPUS_DIR)
    parser.add_argument("--out-dir", default=TOKENIZED_DIR)
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--val-skip", type=int, default=VAL_SKIP_TOKENS)
    parser.add_argument("--val-tokens", type=int, default=VAL_TOKENS_PER_SOURCE)
    parser.add_argument("--system-prompts", default=SYSTEM_PROMPTS_FILE,
                        help="file with one short system prompt per line")
    parser.add_argument("--system-prob", type=float, default=0.5,
                        help="share of dialogues that get a <|system|> prefix")
    parser.add_argument("--system-seed", type=int, default=7)
    parser.add_argument("--keep-parts", action="store_true",
                        help="keep the .partNNNN.jsonl corpus chunks after tokenizing")
    parser.add_argument("--dry-run", type=int, default=0, help="take only N records from each source")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer(tokenizer)
    system_prompts = load_system_prompts(args.system_prompts)
    if system_prompts:
        print(f"System prompts: {len(system_prompts)} from {args.system_prompts}, "
              f"used in {args.system_prob:.0%} of dialogues")
    else:
        print(f"System prompts: none ({args.system_prompts} missing or empty) — "
              f"<|system|> will not be used")

    sources = list(FILES) if args.source == "all" else [args.source]
    for name in sources:
        paths = source_files(args.corpus_dir, name)
        if not paths:
            print(f"Skipping {name}: no corpus files in {args.corpus_dir}")
            continue
        print(f"=== tokenizing {name} ({len(paths)} file(s)) ===")

        train_dir = os.path.join(args.out_dir, "train", name)
        val_dir = os.path.join(args.out_dir, "val", name)
        for stale in (train_dir, val_dir):
            if os.path.isdir(stale):
                shutil.rmtree(stale)

        with_mask = FILES[name][1] == "chat"
        train_writer = ShardWriter(train_dir, with_mask)
        val_writer = ShardWriter(val_dir, with_mask)
        router = Router(train_writer, val_writer, args.val_skip, args.val_tokens)
        count = process_source(name, paths, router, tokenizer, system_prompts,
                               args.system_prob, args.system_seed, args.dry_run)
        train_writer.close()
        val_writer.close()
        print(f"  Records processed: {count}")

        if not args.keep_parts and not args.dry_run:
            for path in paths:
                os.remove(path)
                print(f"  removed {path}")

    sample_dir = os.path.join(args.out_dir, "train", sources[0])
    sample_path = os.path.join(sample_dir, "shard_0000.bin")
    if os.path.exists(sample_path):
        ids = np.fromfile(sample_path, dtype=np.uint16)[:200].tolist()
        print(f"\nSample from {sample_path}:")
        print(tokenizer.decode(ids, skip_special_tokens=False)[:500])


if __name__ == "__main__":
    main()
