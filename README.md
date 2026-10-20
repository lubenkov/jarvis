# jarvis

A 112M-parameter GPT-style language model for Russian, implemented from scratch in PyTorch — with a custom BPE tokenizer, a two-stage training pipeline and local inference.

![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
[![Hugging Face model](https://img.shields.io/badge/Hugging%20Face-jarvis--112m-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/Eragon1/jarvis-112m)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache_2.0-3DA639?style=flat-square)](LICENSE)

## About

`jarvis` is a compact, decoder-only Transformer focused primarily on Russian text and dialogue. The repository contains the model implementation, tokenizer, corpus preparation, training and evaluation code, plus a console chat and an optional Telegram bot. The trained checkpoint is hosted separately on [Hugging Face](https://huggingface.co/Eragon1/jarvis-112m); model weights are not committed to this repository.

## Features

| Component | What it includes |
| :-- | :-- |
| Model | 14-layer causal Transformer with RoPE, RMSNorm, SwiGLU, tied input/output embeddings and PyTorch scaled-dot-product attention |
| Tokenizer | Custom byte-level BPE tokenizer with a 16,384-token vocabulary and role/end-of-turn tokens |
| Training | Pretraining followed by supervised fine-tuning (SFT); mixed-source data, assistant-only loss masks for chat examples, validation and sample generation |
| Training runtime | Automatic mixed precision, multi-GPU DDP, time-limited sessions, periodic checkpoints and resume support |
| Inference | Local generation with a KV cache, configurable temperature/top-k/top-p and repetition penalty |
| Interfaces | Interactive terminal chat, one-shot text generation and an optional Telegram bot |
| Evaluation | Bits-per-character (BPC) benchmark against a Russian GPT baseline, with results committed under `results/` |

## Model

The default `main` preset is configured as follows:

| Parameter | Value |
| :-- | :-- |
| Architecture | Decoder-only autoregressive Transformer |
| Parameters | Approximately 111.7M (about 112M) |
| Layers | 14 |
| Attention heads | 12 |
| Hidden size | 768 |
| Feed-forward size | 2,048 |
| Context length | 768 tokens |
| Vocabulary | 16,384 tokens |
| Position encoding | RoPE |
| Normalization / MLP | RMSNorm / SwiGLU |

Chat examples use the special tokens `<|user|>`, `<|assistant|>` and `<|endoftext|>`; `<|pad|>` is also reserved by the tokenizer. For dialogue SFT, the loss is applied to assistant responses rather than user messages.

## Get started

### Run the published checkpoint

Install PyTorch for your platform first. For CUDA, choose the matching build from the [official PyTorch installer](https://pytorch.org/get-started/locally/). Then clone the repository and install its dependencies:

```bash
git clone https://github.com/lubenkov/jarvis.git
cd jarvis
python -m venv .venv
# Activate the virtual environment for your shell, then:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade huggingface_hub
```

Download the checkpoint and its matching tokenizer from Hugging Face:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Eragon1/jarvis-112m",
    local_dir="checkpoints/hf-jarvis-112m",
    allow_patterns=["jarvis-112m-sft-fp16.pt", "tokenizer.json"],
)
PY
```

Start the interactive chat:

```bash
python chat.py \
  --ckpt checkpoints/hf-jarvis-112m/jarvis-112m-sft-fp16.pt \
  --tokenizer checkpoints/hf-jarvis-112m/tokenizer.json
```

Or generate a single completion:

```bash
python generate.py \
  --ckpt checkpoints/hf-jarvis-112m/jarvis-112m-sft-fp16.pt \
  --tokenizer checkpoints/hf-jarvis-112m/tokenizer.json \
  --prompt "Привет! Расскажи короткую историю."
```

In `chat.py`, enter `exit` to quit, `/reset` to clear the conversation and `/temp 0.7` to change the sampling temperature. CUDA is selected automatically when available; CPU inference is supported but slower.

## Prepare data and train

Training from scratch requires internet access and enough disk space for the downloaded corpora and tokenized shards. The source datasets are fetched by `prepare_data.py`; they are not bundled with this repository.

```bash
# Optional schema/data smoke test; this still accesses the source datasets.
python prepare_data.py --what all --dry-run

# Build the corpora, tokenizer and tokenized shards.
python prepare_data.py --what all
python train_tokenizer.py
python tokenize_corpus.py --source all
```

Train the two phases:

```bash
python train.py --preset main --phase pretrain
python train.py --preset main --phase sft \
  --resume checkpoints/pretrain/latest.pt
```

The first SFT session starts from the pretraining checkpoint. Later sessions can resume automatically from `checkpoints/sft`:

```bash
python train.py --preset main --phase sft --resume auto
```

The checked-in defaults target 1.2B tokens for pretraining and 150M for SFT. Training sessions default to an 11-hour limit and save checkpoints periodically; actual completion time depends on the accelerator and available data. CUDA multi-GPU runs use DDP automatically. Use `--preset dev` for a smaller model, or `--no-ddp`, `--micro-batch`, `--grad-accum` and `--session-hours` to adjust a run. CPU training is supported but is not recommended for the full preset.

### Data sources and mixture

| Source | Dataset(s) | Use |
| :-- | :-- | :-- |
| `web_ru` | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), Russian Cyrillic training shards | Russian web text |
| `texts_ru` | [Gazeta](https://huggingface.co/datasets/IlyaGusev/gazeta) | News articles |
| `chat_ru` | [Saiga Scored](https://huggingface.co/datasets/IlyaGusev/saiga_scored), [GrandMaster-PRO-MAX](https://huggingface.co/datasets/Vikhrmodels/GrandMaster-PRO-MAX), [Veles-2.5](https://huggingface.co/datasets/Vikhrmodels/Veles-2.5), [Russian Instructions](https://huggingface.co/datasets/Den4ikAI/russian_instructions_2), [Jokes Dialogues](https://huggingface.co/datasets/inkoziev/jokes_dialogues) and the Russian subset of [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Dialogue and instruction examples |

The configured pretraining mixture is 68% web, 27% chat and 5% articles. SFT uses 95% chat and 5% web. These are source-sampling weights in the training configuration, not guarantees about the size of the underlying datasets. Dataset revisions are not pinned by the preparation script; review each source's terms and card before using or redistributing derived artifacts.

## Telegram bot

The bot is optional and has separate dependencies:

```bash
python -m pip install -r requirements-bot.txt
cp .env.example .env
# Put your token in .env as TELEGRAM_BOT_TOKEN=...
python bot.py --ckpt checkpoints/hf-jarvis-112m/jarvis-112m-sft-fp16.pt \
  --tokenizer checkpoints/hf-jarvis-112m/tokenizer.json
```

It supports `/start`, `/help`, `/reset`, `/stats` and `/temp 0.7`. 

## Evaluation

The repository includes a BPC benchmark over held-out text. Lower is better.

| Model | `web_ru` | `texts_ru` | `chat_ru` |
| :-- | --: | --: | --: |
| jarvis pretraining checkpoint | 1.18 | 1.02 | 0.92 |
| jarvis SFT checkpoint | 1.22 | 1.07 | 0.91 |
| `ai-forever/rugpt3small_based_on_gpt2` | 1.07 | 0.93 | 0.98 |

Each source uses a 600,000-character held-out sample. This is a small, domain-specific language-modeling check, not a comprehensive dialogue-quality or safety evaluation. The baseline uses a different tokenizer and context length, so treat the comparison as approximate. Full numbers and timing are in [`results/bpc.json`](results/bpc.json); the benchmark is [`tools/bench_bpc.py`](tools/bench_bpc.py).

## Tests

Run the end-to-end smoke test with the project dependencies installed:

```bash
python selftest.py
```

It builds a synthetic corpus and exercises the model/KV cache, tokenizer, data pipeline, a tiny pretraining run, checkpoint resume, the pretrain-to-SFT transition and generation. It does not download the training corpora or require the published checkpoint.

## How the code is organised

| File / directory | Responsibility |
| :-- | :-- |
| `model/` | Transformer configuration, attention, RoPE, RMSNorm, SwiGLU blocks and model |
| `prepare_data.py` | Download, filter, normalize and deduplicate source corpora |
| `train_tokenizer.py` | Train the byte-level BPE tokenizer |
| `tokenize_corpus.py` | Encode corpora and write train/validation shards |
| `dataset.py` | Stream token windows and assistant-response masks |
| `train.py` | Pretraining/SFT loop, validation, checkpoints, AMP and DDP |
| `chat.py`, `generate.py` | Interactive chat and one-shot generation |
| `bot.py` | Optional Telegram integration |
| `tools/bench_bpc.py`, `tools/compare.py` | BPC evaluation and checkpoint sample comparison |
| `selftest.py` | Synthetic end-to-end smoke test |
| `results/` | Committed BPC results |

## Limitations

`jarvis` is an experimental 112M-parameter model, not a general-purpose assistant or a production-ready service. It can produce incorrect, repetitive, biased or unsafe text; its reasoning, factuality and safety have not been comprehensively evaluated. It is intended primarily for Russian, has a 768-token context window, and should not be used for medical, legal, financial or other high-stakes decisions. The public datasets have their own licenses and possible privacy/content issues; check their terms before use.

## License

The source code is licensed under the Apache License 2.0 — see [LICENSE](LICENSE). Dataset terms remain those of their respective sources. The model card and checkpoint are published separately on [Hugging Face](https://huggingface.co/Eragon1/jarvis-112m).

<details>
<summary><b>🇷🇺 По-русски</b></summary>

<br>

`jarvis` — экспериментальная GPT-подобная авторегрессионная модель примерно на 112 млн параметров, ориентированная прежде всего на русский язык. В репозитории есть собственная byte-level BPE-токенизация, подготовка данных, обучение, оценка BPC, генерация с KV-cache, консольный чат и Telegram-бот.

**Запуск опубликованной модели:** установите зависимости из `requirements.txt`, скачайте `jarvis-112m-sft-fp16.pt` и `tokenizer.json` из [Eragon1/jarvis-112m](https://huggingface.co/Eragon1/jarvis-112m), затем выполните:

```bash
python chat.py --ckpt checkpoints/hf-jarvis-112m/jarvis-112m-sft-fp16.pt \
  --tokenizer checkpoints/hf-jarvis-112m/tokenizer.json
```

**Обучение:** `python prepare_data.py --what all` → `python train_tokenizer.py` →
`python tokenize_corpus.py --source all` → `python train.py --phase pretrain` →
`python train.py --phase sft --resume checkpoints/pretrain/latest.pt`.

В настройках обучения заложено 1,2 млрд токенов для pretrain и 150 млн для SFT.

</details>
