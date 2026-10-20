import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))

WORDS = ["джарвис", "привет", "как", "дела", "сегодня", "погода", "хорошая", "хочу",
         "расскажи", "шутку", "спасибо", "пожалуйста", "друг", "работа", "фильм",
         "вечер", "интересно", "здорово", "конечно", "давай", "поговорим", "чем",
         "занимаешься", "люблю", "кофе", "утро", "книга", "музыка", "гулять", "парк",
         "смешно", "хорошо", "вопрос", "ответ", "думаю", "можно", "помоги", "совет", "лучше"]


def run(cmd, cwd):
    print("\n$", " ".join(str(c) for c in cmd[1:]))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True,
                            encoding="utf-8", errors="replace", env=env)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"Command failed with exit code {result.returncode}: {cmd}")
    return result.stdout


def build_corpus(corpus_dir, rng):
    os.makedirs(corpus_dir, exist_ok=True)

    def sentence(n):
        return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."

    def paragraph():
        return " ".join(sentence(rng.randint(5, 12)) for _ in range(rng.randint(3, 8)))

    with open(os.path.join(corpus_dir, "web_ru.jsonl"), "w", encoding="utf-8") as f:
        for _ in range(1500):
            f.write(json.dumps({"text": paragraph()}, ensure_ascii=False) + "\n")
    with open(os.path.join(corpus_dir, "texts_ru.jsonl"), "w", encoding="utf-8") as f:
        for _ in range(900):
            f.write(json.dumps({"text": paragraph()}, ensure_ascii=False) + "\n")
    with open(os.path.join(corpus_dir, "chat_ru.jsonl"), "w", encoding="utf-8") as f:
        for _ in range(2200):
            messages = []
            for _ in range(rng.randint(1, 3)):
                messages.append({"role": "user", "content": sentence(rng.randint(3, 8))})
                messages.append({"role": "assistant", "content": sentence(rng.randint(4, 12))})
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


def train_tiny_tokenizer(corpus_dir, out_path, vocab_size=512):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from chat_format import SPECIAL_TOKENS
    from train_tokenizer import iter_texts

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1,
                                  special_tokens=["[UNK]"] + SPECIAL_TOKENS, show_progress=False)
    tokenizer.train_from_iterator(iter_texts(corpus_dir, 100_000), trainer=trainer)
    tokenizer.save(out_path)
    return tokenizer


def main():
    import torch

    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    work = tempfile.mkdtemp(prefix="jarvis_selftest_")
    corpus_dir = os.path.join(work, "corpus")
    tokenized_dir = os.path.join(work, "tokenized")
    out_dir = os.path.join(work, "checkpoints")
    tokenizer_path = os.path.join(work, "tokenizer.json")
    rng = random.Random(1234)
    print("Work dir:", work)

    try:
        from dataclasses import replace

        from model.config import MODEL_PRESETS
        from model.transformer import TransformerLM

        cfg = replace(MODEL_PRESETS["dev"], vocab_size=512)
        model = TransformerLM(cfg)
        ids = torch.randint(0, 512, (2, 64))
        model.train()
        model(ids).square().mean().backward()
        model.eval()
        with torch.no_grad():
            full = model(ids)
            caches, parts = None, []
            for t in range(ids.shape[1]):
                logits_t, caches = model(ids[:, t:t + 1], caches=caches, use_cache=True)
                parts.append(logits_t)
            diff = (full - torch.cat(parts, dim=1)).abs().max().item()
        assert diff < 1e-4, "KV cache is incorrect"
        print(f"  model and KV cache: ok (diff {diff:.2e})")

        build_corpus(corpus_dir, rng)
        tokenizer = train_tiny_tokenizer(corpus_dir, tokenizer_path)
        from tokenizers import Tokenizer

        from chat_format import check_tokenizer, encode_conversation
        check_tokenizer(Tokenizer.from_file(tokenizer_path), verbose=False)
        conv_ids, conv_mask = encode_conversation(tokenizer, [
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "привет как дела"},
        ])
        assert conv_ids and any(conv_mask) and not conv_mask[0], "loss mask is incorrect"
        print(f"  tokenizer: {tokenizer.get_vocab_size()} tokens, dialogue format: ok")

        run([sys.executable, os.path.join(ROOT, "tokenize_corpus.py"), "--source", "all",
             "--corpus-dir", corpus_dir, "--out-dir", tokenized_dir, "--tokenizer", tokenizer_path,
             "--val-skip", "2000", "--val-tokens", "6000"], cwd=ROOT)
        print("  tokenization: ok")

        with open(os.path.join(work, "golden_prompts.json"), "w", encoding="utf-8") as f:
            json.dump([{"messages": [{"role": "user", "content": "привет"}]},
                       {"messages": [{"role": "user", "content": "расскажи шутку"}]}], f, ensure_ascii=False)

        train_cmd = [
            sys.executable, os.path.join(ROOT, "train.py"),
            "--preset", "dev", "--phase", "pretrain",
            "--data-dir", tokenized_dir, "--out-dir", out_dir, "--tokenizer", tokenizer_path,
            "--max-tokens", "120000", "--micro-batch", "4", "--grad-accum", "2",
            "--session-hours", "1", "--no-ddp", "--num-workers", "0",
            "--log-every", "5", "--val-every", "10", "--sample-every", "20",
            "--val-batch", "2", "--val-count", "2", "--peak-lr", "3e-3", "--seed", "1",
        ]
        output = run(train_cmd, cwd=work)
        assert "Stopped:" in output, "training did not finish normally"
        assert os.path.exists(os.path.join(out_dir, "latest.pt"))
        assert os.path.exists(os.path.join(out_dir, "inference.pt"))
        print("  training: ok")

        train_cmd[train_cmd.index("--max-tokens") + 1] = "150000"
        output = run(train_cmd, cwd=work)
        assert "Resuming from" in output, "resume did not work"
        print("  resume: ok")

        sft_dir = os.path.join(work, "sft")
        sft_cmd = [
            sys.executable, os.path.join(ROOT, "train.py"),
            "--preset", "dev", "--phase", "sft",
            "--data-dir", tokenized_dir, "--out-dir", sft_dir, "--tokenizer", tokenizer_path,
            "--resume", os.path.join(out_dir, "latest.pt"),
            "--max-tokens", "40000", "--micro-batch", "4", "--grad-accum", "2",
            "--session-hours", "1", "--no-ddp", "--num-workers", "0",
            "--log-every", "5", "--val-every", "10", "--sample-every", "1000",
            "--val-batch", "2", "--val-count", "2", "--peak-lr", "3e-4", "--seed", "1",
        ]
        output = run(sft_cmd, cwd=work)
        assert "Resuming from" in output and "phase complete" in output, \
            "SFT after pretrain did not finish normally"
        with open(os.path.join(sft_dir, "metrics.jsonl"), encoding="utf-8") as f:
            sft_records = [json.loads(line) for line in f if line.strip()]
        assert sft_records[-1].get("done") and sft_records[-1].get("phase_tokens", 0) > 0, \
            "SFT after pretrain did not train any phase tokens"
        print("  SFT after pretrain: ok")

        from chat_format import build_prompt, stop_ids
        from generate import generate_ids, load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_model(os.path.join(out_dir, "inference.pt"), device)
        prompt_ids = build_prompt(tokenizer, [{"role": "user", "content": "привет"}])
        out_ids = generate_ids(model, prompt_ids, max_new_tokens=16, temperature=0.8,
                               top_p=0.95, stop_token_ids=stop_ids(tokenizer))
        assert out_ids.shape[1] > len(prompt_ids), "model generated nothing"
        print("  generation: ok")

        print("\nALL OK")
    finally:
        if "--keep" not in sys.argv:
            import shutil
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
