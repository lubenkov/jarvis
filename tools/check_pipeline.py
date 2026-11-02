"""Offline check of the shard writer / router / streaming builder.

These paths only get exercised with multi-million-token corpora on Kaggle, so a
boundary bug there would silently corrupt the 4B-token run. Here the shard size
is shrunk so the same code splits and flushes many times on a tiny corpus.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tokenizers import Tokenizer

import tokenize_corpus
from tokenize_corpus import Router, ShardWriter
from train_tokenizer import build_tokenizer, iter_texts

SHARD = 1000
tokenize_corpus.SHARD_TOKENS = SHARD

work = tempfile.mkdtemp(prefix="jarvis_shards_")
corpus = os.path.join(work, "corpus")
os.makedirs(corpus)
words = "джарвис привет как дела сегодня погода хорошая хочу расскажи шутку спасибо друг".split()

with open(os.path.join(corpus, "web_ru.jsonl"), "w", encoding="utf-8") as f:
    for i in range(3000):
        f.write(json.dumps({"text": " ".join(words[i % 5:] + words[:i % 5]) + f" {i}"},
                           ensure_ascii=False) + "\n")

tokenizer, trainer = build_tokenizer(512, min_frequency=1, show_progress=False)
tokenizer.train_from_iterator(iter_texts(corpus, 500_000), trainer=trainer)
tok_path = os.path.join(work, "tokenizer.json")
tokenizer.save(tok_path)
tokenizer = Tokenizer.from_file(tok_path)
eos = tokenizer.token_to_id("<|endoftext|>")

texts = [f"{' '.join(words)} номер {i}" for i in range(400)]
expected = []
for text in texts:
    expected.extend(tokenizer.encode(text, add_special_tokens=False).ids + [eos])

# --- 1. ShardWriter alone: every token lands in exactly one shard, in order ---
out = os.path.join(work, "direct")
writer = ShardWriter(os.path.join(out, "train"), with_mask=True)
mask_expected = []
for idx, text in enumerate(texts):
    ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
    mask = [1 if idx % 2 else 0] * len(ids)
    writer.add(ids, mask)
    mask_expected.extend(mask)
writer.close()

paths = sorted(p for p in os.listdir(os.path.join(out, "train")) if p.endswith(".bin")
                and not p.endswith(".mask.bin"))
assert len(paths) == int(np.ceil(len(expected) / SHARD)), f"unexpected shard count {len(paths)}"
rebuilt, rebuilt_mask = [], []
for name in paths:
    rebuilt.extend(np.fromfile(os.path.join(out, "train", name), dtype=np.uint16).tolist())
    rebuilt_mask.extend(np.fromfile(os.path.join(out, "train", name[:-4] + ".mask.bin"),
                                    dtype=np.uint8).tolist())
assert rebuilt == expected, "shards do not reproduce the token stream"
assert rebuilt_mask == mask_expected, "masks are misaligned with the tokens"
print(f"shard writer: {len(paths)} shards, {len(rebuilt)} tokens, masks aligned")

# --- 2. Router: val split lands on exact token offsets ---
out2 = os.path.join(work, "routed")
train_w, val_w = ShardWriter(os.path.join(out2, "train"), False), ShardWriter(os.path.join(out2, "val"), False)
router = Router(train_w, val_w, val_skip=137, val_tokens=421)
step = 97
for start in range(0, len(expected), step):
    router.add(expected[start:start + step])
train_w.close()
val_w.close()

def read_all(folder):
    ids = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".bin"):
            ids.extend(np.fromfile(os.path.join(folder, name), dtype=np.uint16).tolist())
    return ids

val_ids = read_all(os.path.join(out2, "val"))
train_ids = read_all(os.path.join(out2, "train"))
assert val_ids == expected[137:137 + 421], "validation split is wrong"
# the first val_skip tokens are dropped on purpose (they are a warm-up buffer)
assert train_ids == expected[137 + 421:], "training split is wrong"
assert len(expected) - len(val_ids) - len(train_ids) == 137, "skipped tokens are missing"
print(f"router: val {len(val_ids)} tokens at the right offset, train {len(train_ids)}, "
      f"skipped 137")

# --- 3. prepare_stream.build: target stops the stream, val split works ---
from prepare_stream import build

out3 = os.path.join(work, "stream")
stats = {"read": 0, "written": 0, "filtered": 0}
produced = build("web", iter(texts * 5), 3 * SHARD + 17, tokenizer, out3,
                 100, 200, stats)
assert 0 < stats["written"] <= 512, "the stream should stop after the first batch"
assert read_all(os.path.join(out3, "val", "web")) == expected[100:300], "stream val split is wrong"
train_stream = read_all(os.path.join(out3, "train", "web"))
source = (expected * 5)[300:]
assert train_stream == source[:len(train_stream)], "stream shards do not continue the stream"
assert produced == len(train_stream) + 200 + 100, "token accounting does not match the shards"
assert produced >= 3 * SHARD + 17, "target token count not reached"
print(f"stream builder: stopped at {produced} tokens, {stats['written']} docs")

shutil.rmtree(work, ignore_errors=True)
print("ALL PIPELINE CHECKS OK")
