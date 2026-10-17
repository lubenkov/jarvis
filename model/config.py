"""Все гиперпараметры проекта в одном месте."""

from dataclasses import dataclass
from typing import List, Tuple


# ------------------------------------------------------------------ модель


@dataclass
class ModelConfig:
    """Архитектура TransformerLM."""

    vocab_size: int = 16384      # размер словаря BPE, должен совпадать с tokenizer.json
    context_length: int = 768    # длина контекста в токенах
    d_model: int = 768           # размерность эмбеддингов и скрытого состояния
    n_layers: int = 14           # число трансформерных блоков
    n_heads: int = 12            # число голов внимания (d_model / n_heads = head_dim)
    d_ff: int = 2048             # скрытая размерность SwiGLU (~8/3 * d_model)
    dropout: float = 0.0         # токенов сильно больше параметров, дропаут не нужен
    rope_theta: float = 10000.0  # база частот RoPE
    norm_eps: float = 1e-5


MODEL_PRESETS = {
    # быстрый пресет для отладки всего пайплайна (~21M параметров)
    "dev": ModelConfig(context_length=512, d_model=384, n_layers=8, n_heads=6, d_ff=1024),
    # боевая конфигурация (~112M параметров). Для ~150M: d_model=896, n_heads=14, d_ff=2400
    "main": ModelConfig(),
}


# ------------------------------------------------------------------ обучение


@dataclass
class PhaseConfig:
    """Параметры одного этапа обучения."""

    name: str
    mix: List[Tuple[str, float, bool]]  # (источник, вес, маскировать ли лосс по ассистенту)
    total_tokens: int                   # сколько токенов пройти за этап
    peak_lr: float
    min_lr_ratio: float                 # до какой доли от peak_lr опускается cosine
    warmup_steps: int
    micro_batch: int = 24               # последовательностей на GPU за микрошаг
    grad_accum: int = 2                 # микрошагов до optimizer.step
    weight_decay: float = 0.1
    grad_clip: float = 1.0


# Этап 1: претрейн на смеси веба и чатов, обычный LM-лосс.
# Объём подобран под собранный корпус (~0.98B токенов: web 807M, chat 110M,
# texts 67M). При таких весах по данным получается примерно web 1 проход,
# chat 3, texts 1. При старых весах (chat 0.45) диалогов в корпусе не хватало
# и они прокручивались бы ~10 раз — это уже зазубривание, а не обучение.
PRETRAIN = PhaseConfig(
    name="pretrain",
    mix=[("web_ru", 0.68, False), ("chat_ru", 0.27, False), ("texts_ru", 0.05, False)],
    total_tokens=1_200_000_000,
    peak_lr=6e-4,
    min_lr_ratio=0.1,
    warmup_steps=1000,
)

# Этап 2: дожим/дотюн на чатах, лосс только на ответах ассистента.
# В корпусе 110M токенов диалогов, поэтому 150M токенов SFT — это ~1.3 прохода
# (300M дали бы ~2.6 в дополнение к ~3 проходам на претрейне).
SFT = PhaseConfig(
    name="sft",
    mix=[("chat_ru", 0.95, True), ("web_ru", 0.05, True)],
    total_tokens=150_000_000,
    peak_lr=3e-4,
    min_lr_ratio=0.0,
    warmup_steps=100,
)

PHASES = {"pretrain": PRETRAIN, "sft": SFT}

# Значения по умолчанию для train.py
SESSION_HOURS = 11.0        # мягкая остановка сессии (сессия Kaggle живёт максимум 12ч)
CKPT_MINUTES = 15.0         # как часто сохранять чекпоинт
VAL_EVERY_STEPS = 1000      # как часто считать валидацию
SAMPLE_EVERY_STEPS = 4000   # как часто писать примеры ответов на golden prompts
LOG_EVERY_STEPS = 20
NUM_WORKERS = 2


# ------------------------------------------------------------------ данные


TOKENIZER_PATH = "tokenizer.json"
TOKENIZER_VOCAB_SIZE = 16384

CORPUS_DIR = "data/corpus"        # jsonl-файлы после prepare_data.py
TOKENIZED_DIR = "data/tokenized"  # uint16-шарды после tokenize_corpus.py
CHECKPOINT_DIR = "checkpoints"    # сюда пишутся чекпоинты

# Размер валидации: первые VAL_SKIP_TOKENS каждого источника пропускаются,
# следующие VAL_TOKENS_PER_SOURCE токенов уходят в валидацию.
VAL_SKIP_TOKENS = 100_000
VAL_TOKENS_PER_SOURCE = 300_000


# ------------------------------------------------------------------ генерация


MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_K = 0            # 0 = выключено
TOP_P = 0.95
REPETITION_PENALTY = 1.1
