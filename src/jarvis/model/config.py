from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    context_length: int = 1024
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    d_ff: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    depth_scaled_init: bool = True


MODEL_PRESETS = {
    # v1 (jarvis-112m): kept so old checkpoints and side-by-side benchmarks still load
    "v1": ModelConfig(vocab_size=16384, context_length=768, n_layers=14,
                      depth_scaled_init=False),
    # v2 (jarvis-138m)
    "main": ModelConfig(),
    "dev": ModelConfig(context_length=512, d_model=384, n_layers=8, n_heads=6, d_ff=1024),
}


@dataclass
class PhaseConfig:
    name: str
    mix: List[Tuple[str, float, bool]]
    total_tokens: int
    peak_lr: float
    min_lr_ratio: float
    warmup_steps: int
    micro_batch: int = 8
    grad_accum: int = 5
    weight_decay: float = 0.1
    grad_clip: float = 1.0


PRETRAIN = PhaseConfig(
    name="pretrain",
    mix=[("web_ru", 0.88, False), ("chat_ru", 0.05, False),
         ("wiki_ru", 0.04, False), ("texts_ru", 0.03, False)],
    total_tokens=4_000_000_000,
    peak_lr=6e-4,
    min_lr_ratio=0.1,
    warmup_steps=1500,
)


SFT = PhaseConfig(
    name="sft",
    mix=[("chat_ru", 0.95, True), ("web_ru", 0.05, True)],
    total_tokens=200_000_000,
    peak_lr=3e-4,
    min_lr_ratio=0.0,
    warmup_steps=100,
)


PHASES = {"pretrain": PRETRAIN, "sft": SFT}


SESSION_HOURS = 8.5
CKPT_MINUTES = 15.0
VAL_EVERY_STEPS = 1000
SAMPLE_EVERY_STEPS = 4000
SAMPLE_FIRST_STEPS = 20
LOG_EVERY_STEPS = 20
NUM_WORKERS = 2


TOKENIZER_PATH = "tokenizer.json"
TOKENIZER_VOCAB_SIZE = 32768

CORPUS_DIR = "data/corpus"
TOKENIZED_DIR = "data/tokenized"
CHECKPOINT_DIR = "checkpoints"

VAL_SKIP_TOKENS = 100_000
VAL_TOKENS_PER_SOURCE = 300_000


MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_K = 0
TOP_P = 0.95
REPETITION_PENALTY = 1.1


# System prompt is not hardcoded: it comes from a file or from --system / JARVIS_SYSTEM_PROMPT.
# An empty string means "no <|system|> segment is added".
SYSTEM_PROMPTS_FILE = "system_prompts.txt"
SYSTEM_PROMPT = ""
