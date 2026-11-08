"""Build tokenized shards straight from Hugging Face streams.

A 4B-token corpus is ~36 GB of JSONL, while a Kaggle CPU notebook has 20 GB of
disk. So web/wiki/news text is filtered, deduplicated and tokenized in one pass:
only the final uint16 shards (2 bytes per token) ever touch the disk. The chat
corpus is small enough to keep using prepare_data.py + tokenize_corpus.py.

Usage:
    python prepare_stream.py --source web --target-tokens 4.2e9 --web-shard-stride 40
    python prepare_stream.py --source wiki --target-tokens 2e8
    python prepare_stream.py --source texts --target-tokens 1.5e8
"""

import argparse
import hashlib
import os
import shutil

from tokenizers import Tokenizer

from chat_format import check_tokenizer
from model.config import TOKENIZED_DIR, TOKENIZER_PATH, VAL_SKIP_TOKENS, VAL_TOKENS_PER_SOURCE
from prepare_data import MAX_CHARS, cyrillic_ratio
from tokenize_corpus import Router, ShardWriter

WEB_REPO = "HuggingFaceFW/fineweb-2"
WEB_PATH = "data/rus_Cyrl/train"
WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.ru"
NEWS_REPO = "IlyaGusev/gazeta"

MAX_DOC_CHARS = 40000
ENCODE_BATCH = 512
BATCH_DOCS = 2048

DEFAULT_TARGETS = {"web": 4_200_000_000, "wiki": 200_000_000, "texts": 150_000_000}

# the phase mix in model/config.py refers to sources by these names
SOURCE_NAMES = {"web": "web_ru", "wiki": "wiki_ru", "texts": "texts_ru"}


def web_texts(stride, seed, stats):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = [(f.path, f.size or 0)
             for f in HfApi().list_repo_tree(WEB_REPO, path_in_repo=WEB_PATH, repo_type="dataset")
             if f.path.endswith(".parquet")]
    files.sort(key=lambda item: item[0])
    stride = max(1, stride)
    # Spread first (every Nth shard), then fall back to the remaining ones, so a
    # large stride can never leave us short of the token target.
    spread = list(range(seed % stride, len(files), stride))
    picked = [files[i] for i in spread] + [files[i] for i in range(len(files))
                                          if i not in set(spread)]
    print(f"  shards in dataset: {len(files)}, starting with {len(spread)} spread "
          f"shards (stride {stride}, seed {seed}), more if the target needs them")

    seen = set()
    work = os.path.join("data", "_web_tmp")
    os.makedirs(work, exist_ok=True)
    try:
        for position, (name, size) in enumerate(picked, start=1):
            print(f"  shard {position}: {os.path.basename(name)} ({size / 1e9:.1f} GB)",
                  flush=True)
            local = hf_hub_download(WEB_REPO, name, repo_type="dataset", local_dir=work)
            parquet = None
            try:
                parquet = pq.ParquetFile(local)
                for batch in parquet.iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
                    for text in batch.column("text").to_pylist():
                        stats["read"] += 1
                        text = (text or "").strip()
                        if not (400 <= len(text) <= MAX_CHARS) or cyrillic_ratio(text) < 0.6:
                            stats["filtered"] += 1
                            continue
                        key = hashlib.md5(text[:2000].encode("utf-8")).digest()
                        if key in seen:
                            stats["filtered"] += 1
                            continue
                        seen.add(key)
                        yield text
            finally:
                # drop the 5 GB parquet, its download metadata and any xet chunks
                # before the next shard lands, otherwise the 20 GB disk overflows
                if parquet is not None:
                    parquet.close()
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(work, exist_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def wiki_texts(stats):
    from datasets import load_dataset

    seen = set()
    dataset = load_dataset(WIKI_REPO, WIKI_CONFIG, split="train", streaming=True)
    for ex in dataset:
        stats["read"] += 1
        text = (ex.get("text") or "").strip()
        if not (300 <= len(text) <= MAX_CHARS) or cyrillic_ratio(text) < 0.5:
            stats["filtered"] += 1
            continue
        key = hashlib.md5(text[:2000].encode("utf-8")).digest()
        if key in seen:
            stats["filtered"] += 1
            continue
        seen.add(key)
        yield text


def news_texts(stats):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    # gazeta is a script-based dataset: `datasets` streaming hangs on it on the
    # Kaggle runner, so the two parquet files are read directly.
    files = [f.path for f in HfApi().list_repo_tree(NEWS_REPO, path_in_repo="default/train",
                                                    repo_type="dataset")
             if f.path.endswith(".parquet")]
    files.sort()
    seen = set()
    work = os.path.join("data", "_news_tmp")
    os.makedirs(work, exist_ok=True)
    try:
        for name in files:
            print(f"  news file: {os.path.basename(name)}", flush=True)
            local = hf_hub_download(NEWS_REPO, name, repo_type="dataset", local_dir=work)
            parquet = None
            try:
                parquet = pq.ParquetFile(local)
                for batch in parquet.iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
                    for text in batch.column("text").to_pylist():
                        stats["read"] += 1
                        text = (text or "").strip()
                        if not (300 <= len(text) <= MAX_CHARS) or cyrillic_ratio(text) < 0.5:
                            stats["filtered"] += 1
                            continue
                        key = hashlib.md5(text[:2000].encode("utf-8")).digest()
                        if key in seen:
                            stats["filtered"] += 1
                            continue
                        seen.add(key)
                        yield text
            finally:
                if parquet is not None:
                    parquet.close()
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(work, exist_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build(source_name, texts, target_tokens, tokenizer, out_dir, val_skip, val_tokens,
          stats, dry_run_docs=0):
    source_name = SOURCE_NAMES.get(source_name, source_name)
    free = shutil.disk_usage(out_dir if os.path.isdir(out_dir) else ".").free / 1e9
    print(f"  free disk: {free:.1f} GB, target {target_tokens / 1e9:.2f}B tokens "
          f"(~{target_tokens * 2 / 1e9:.1f} GB of shards)", flush=True)
    train_dir = os.path.join(out_dir, "train", source_name)
    val_dir = os.path.join(out_dir, "val", source_name)
    for stale in (train_dir, val_dir):
        if os.path.isdir(stale):
            shutil.rmtree(stale)

    train_writer = ShardWriter(train_dir, with_mask=False)
    val_writer = ShardWriter(val_dir, with_mask=False)
    router = Router(train_writer, val_writer, val_skip, val_tokens)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    pending = []
    batches = 0
    for text in texts:
        pending.append(text[:MAX_DOC_CHARS])
        stats["written"] += 1
        if len(pending) < ENCODE_BATCH:
            continue
        for encoding in tokenizer.encode_batch(pending):
            router.add(encoding.ids + [eos_id], None)
        pending = []
        batches += 1
        if batches % 20 == 0:
            print(f"  ... {stats['written']:,} docs, {router.seen / 1e9:.3f}B tokens", flush=True)
        if target_tokens and router.seen >= target_tokens:
            break
        if dry_run_docs and stats["written"] >= dry_run_docs:
            break
    if pending:
        for encoding in tokenizer.encode_batch(pending):
            router.add(encoding.ids + [eos_id], None)

    train_writer.close()
    val_writer.close()
    return router.seen


def main():
    parser = argparse.ArgumentParser(description="Stream, filter and tokenize web/wiki/news corpora")
    parser.add_argument("--source", choices=["web", "wiki", "texts", "all"], default="all")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--out-dir", default=TOKENIZED_DIR)
    parser.add_argument("--target-tokens", type=float, default=0,
                        help="stop after ~N tokens (0 = use the per-source default)")
    parser.add_argument("--web-shard-stride", type=int, default=1,
                        help="take every Nth FineWeb-2 shard instead of the first ones")
    parser.add_argument("--web-shard-seed", type=int, default=0)
    parser.add_argument("--val-skip", type=int, default=VAL_SKIP_TOKENS)
    parser.add_argument("--val-tokens", type=int, default=VAL_TOKENS_PER_SOURCE)
    parser.add_argument("--dry-run", type=int, default=0, help="stop after N documents")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer(tokenizer)

    sources = ["web", "wiki", "texts"] if args.source == "all" else [args.source]
    iterators = {"web": lambda stats: web_texts(args.web_shard_stride, args.web_shard_seed, stats),
                 "wiki": wiki_texts,
                 "texts": news_texts}

    for name in sources:
        target = args.target_tokens or DEFAULT_TARGETS[name]
        print(f"=== {name}: target {target / 1e9:.2f}B tokens ===", flush=True)
        stats = {"read": 0, "written": 0, "filtered": 0}
        produced = build(name, iterators[name](stats), target, tokenizer, args.out_dir,
                         args.val_skip, args.val_tokens, stats, args.dry_run)
        print(f"  {name}: read {stats['read']:,}, kept {stats['written']:,}, "
              f"filtered {stats['filtered']:,}, tokens {produced / 1e9:.3f}B", flush=True)


if __name__ == "__main__":
    main()
