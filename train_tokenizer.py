"""Обучение BPE-токенизатора (ByteLevel) на подготовленном корпусе data/corpus/*.jsonl.

Спецтокены ролей (<|user|>, <|assistant|>, <|endoftext|>, <|pad|>) добавляются
на этапе обучения и получают фиксированные id — их нельзя «дорисовать» позже,
иначе модель и формат диалога разъедутся.

Запуск: python train_tokenizer.py [--vocab-size 16384] [--max-chars 400000000]
"""

import argparse
import glob
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from chat_format import SPECIAL_TOKENS, check_tokenizer, format_conversation, prepare_messages
from model.config import CORPUS_DIR, TOKENIZER_PATH, TOKENIZER_VOCAB_SIZE

DEFAULT_MAX_CHARS = 400_000_000  # сколько символов корпуса дать токенизатору
TEXT_SHARE = 0.6                 # доля текстовых источников (остальное — чаты)


def _iter_text_file(path, budget_chars):
    spent = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = ex.get("text")
            if not text:
                continue
            yield text
            spent += len(text)
            if spent >= budget_chars:
                return


def _iter_chat_file(path, budget_chars):
    spent = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            prepared = prepare_messages(ex.get("messages") or [])
            if not prepared:
                continue
            text = format_conversation(prepared)
            yield text
            spent += len(text)
            if spent >= budget_chars:
                return


def iter_texts(corpus_dir, max_chars):
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"В {corpus_dir} нет jsonl-файлов — сначала запусти prepare_data.py")

    text_files = [p for p in files if os.path.basename(p) != "chat_ru.jsonl"]
    text_budget = int(max_chars * TEXT_SHARE / max(1, len(text_files)))
    chat_path = os.path.join(corpus_dir, "chat_ru.jsonl")

    for path in text_files:
        yield from _iter_text_file(path, text_budget)
    if os.path.exists(chat_path):
        yield from _iter_chat_file(chat_path, int(max_chars * (1 - TEXT_SHARE)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=CORPUS_DIR)
    parser.add_argument("--vocab-size", type=int, default=TOKENIZER_VOCAB_SIZE)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--out", default=TOKENIZER_PATH)
    args = parser.parse_args()

    print(f"Обучаю токенизатор на сэмпле до {args.max_chars / 1e6:.0f}M символов...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]"] + SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_texts(args.corpus_dir, args.max_chars), trainer=trainer)

    tokenizer.save(args.out)
    print("Токенизатор сохранён в", args.out)
    check_tokenizer(tokenizer)

    demo = [
        {"role": "user", "content": "Привет! Как дела?"},
        {"role": "assistant", "content": "Привет! Всё хорошо, а у тебя как?"},
    ]
    text = format_conversation(prepare_messages(demo))
    encoded = tokenizer.encode(text)
    print("Пример:", text)
    print("Токены:", encoded.tokens[:30])
    print("Кусочков на символ:", round(len(encoded.ids) / max(1, len(text)), 3))


if __name__ == "__main__":
    main()
