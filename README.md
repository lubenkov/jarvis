# jarvis

![status](https://img.shields.io/badge/status-work%20in%20progress-yellow?style=flat-square)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![trained on](https://img.shields.io/badge/trained%20on-Kaggle-20BEFF?style=flat-square)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/license-Apache_2.0-3DA639?style=flat-square)](LICENSE)

A small GPT-style language model written **from scratch** in PyTorch: my own multi-head causal
attention, transformer blocks, BPE tokenizer and training loop.

## Project structure

```
model/                  # the model itself
├── config.py           # all hyperparameters
├── attention.py        # causal multi-head attention
├── block.py            # feed-forward network and transformer block
└── transformer.py      # TransformerLM (embeddings, blocks, tied lm_head)
prepare_data.py         # dataset download → data/raw/dialogues.txt
train_tokenizer.py      # BPE tokenizer training → tokenizer.json
dataset.py              # sliding-window dataset
train.py                # training loop with checkpoints
generate.py             # sampling (temperature, top-k, top-p, repetition penalty)
chat.py                 # console chat wrapper
main.py                 # CUDA/GPU check
```
