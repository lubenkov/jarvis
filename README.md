# jarvis

![status](https://img.shields.io/badge/status-work%20in%20progress-yellow?style=flat-square)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![trained on](https://img.shields.io/badge/trained%20on-Kaggle-20BEFF?style=flat-square)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/license-Apache_2.0-3DA639?style=flat-square)](LICENSE)

Русскоязычная GPT-модель **с нуля** (свой код, без готовых LLM-библиотек):
собственное causal-внимание с RoPE, блоки с RMSNorm и SwiGLU, BPE-токенизатор,
цикл обучения с DDP на 2×T4 и генерация с KV-кэшем.

Цель проекта — не рекорды бенчмарков, а живой собеседник на бытовые темы,
который обучается на Kaggle и запускается на ноутбуке.

## Как устроено

Два этапа обучения:

1. **pretrain** — обычное языковое моделирование на смеси веб-текста и диалогов
   (веб ~50%, чаты ~45%, новости/статьи ~5%), цель ~2.6B токенов;
2. **SFT** — дожим на диалогах: лосс считается **только на ответах ассистента**
   (токены пользователя маскируются), цель ~0.3B токенов.

Формат диалога (одинаковый при обучении и генерации):

```
<|user|>привет<|assistant|>привет! как дела?<|endoftext|><|user|>...
```

Модель (`--preset main`, ~112M параметров): d_model 768, 14 слоёв, 12 голов,
SwiGLU d_ff 2048, контекст 768, словарь 16384, tied embeddings, без bias,
fp16 AMP на 2×T4 через DDP.

## Быстрый старт

```bash
pip install -r requirements.txt

# 1) проверка окружения (torch, GPU)
python main.py

# 2) самопроверка всей пайплайна на синтетических данных (~пара минут, без интернета)
python selftest.py

# 3) чат с обученной моделью
python chat.py --ckpt checkpoints/main/inference.pt
```

`selftest.py` прогоняет весь путь: мини-корпус → токенизатор → шарды →
несколько шагов обучения → resume → генерация. Если он зелёный — пайплайн рабочий,
можно запускать боевое обучение.

## Полный пайплайн на Kaggle

Нужны 2 сессии: **CPU** (подготовка данных, интернет включён) и **GPU T4×2** (обучение).

### Сессия 1 (CPU + Internet): корпус и токенизатор

```bash
git clone https://github.com/lubenkov/jarvis.git && cd jarvis
pip install -q -r requirements.txt

# схемы датасетов на месте?
python prepare_data.py --what all --dry-run

# веб-текст (~1.2B токенов, 2-4 часа), новости, диалоги
python prepare_data.py --what web  --max-web-docs 1000000
python prepare_data.py --what texts --max-texts-docs 150000
python prepare_data.py --what chats

# токенизатор 16k с спецтокенами ролей
python train_tokenizer.py

# нарезка корпуса в uint16-шарды + отложенная валидация
python tokenize_corpus.py --source all

# посмотреть объём
python dataset.py
```

> Готовый `tokenizer.json` (16k, со спецтокенами ролей, ~1.6 МБ) уже лежит в репозитории.
> `train_tokenizer.py` нужен, только если корпус собран заново.

Теперь сохрани результат (**Save Version** → output как Kaggle Dataset
`jarvis-corpus`): нужны `tokenizer.json` и папка `data/tokenized/`.

### Сессия 2 (GPU T4×2): pretrain

Добавь к ноутбуку два Input Dataset: `jarvis-corpus` и (после первой сессии) `jarvis-checkpoints`.

```bash
git clone https://github.com/lubenkov/jarvis.git && cd jarvis
pip install -q -r requirements.txt
cp -r /kaggle/input/jarvis-corpus/tokenizer.json .
mkdir -p data && cp -r /kaggle/input/jarvis-corpus/tokenized data/

mkdir -p checkpoints
cp -r /kaggle/input/jarvis-checkpoints/* checkpoints/ 2>/dev/null || true

# сессия до 11 часов, потом мягкая остановка с полным чекпоинтом
python train.py --preset main --phase pretrain --session-hours 11
```

Чекпоинты складываются в `checkpoints/pretrain/`:

- `latest.pt` — полный чекпоинт (модель + оптимизатор + счётчики) для продолжения;
- `ckpt_XXXXXXXX.pt` — последние два шага на всякий случай;
- `best.pt` — лучший по валидации;
- `inference.pt` — компактная fp16-копия весов для чата (~220 МБ);
- `metrics.jsonl` — лог (loss, lr, скорость, валидация);
- `samples/step_*.txt` — ответы модели на фиксированные golden-промпты.

Между сессиями: **Save Version** (или скачай чекпоинты), в новой сессии положи их в
`checkpoints/` и запусти ту же команду — обучение продолжится с того же шага
(`--resume auto` по умолчанию; `--resume none` — начать заново).

### Сессия 3 (GPU T4×2): SFT

```bash
cp -r /kaggle/input/jarvis-checkpoints/pretrain checkpoints/   # или как удобно
python train.py --preset main --phase sft \
    --resume checkpoints/pretrain/latest.pt \
    --out-dir checkpoints/sft
```

### Локально: чат

Скачай `inference.pt` (и `tokenizer.json`), затем:

```bash
python chat.py --ckpt inference.pt
```

`chat.py` держит историю диалога, режет её по длине контекста и печатает ответ
по мере генерации. Команды: `/reset`, `/temp 0.9`, `exit`.

## Локальное обучение (маленький пресет)

На слабой GPU или CPU можно проверить всё на пресете `dev` (~21M параметров):

```bash
python train.py --preset dev --phase pretrain --max-tokens 50000000 --session-hours 2
python chat.py --ckpt checkpoints/pretrain/inference.pt
```

## Что смотреть во время обучения

- `loss`/`ppl` на претрейне должны ровно падать (первые часы — быстро, дальше медленнее);
- `[валидация]` — ppl по каждому источнику (web/chat): если train падает, а val растёт — переобучение
  (для нашего объёма данных случается редко);
- `samples/step_*.txt` — самое честное: сравнивай ответы на одни и те же промпты между шагами;
- `tokens_per_sec` — если сильно ниже ~8k ток/с на 2×T4, проверь, что GPU не греется в throttle;
  при необходимости уменьши `--micro-batch`.

## Структура

```
model/
├── config.py           # все гиперпараметры (пресеты dev/main, фазы pretrain/sft)
├── norm.py             # RMSNorm
├── rope.py             # RoPE
├── attention.py        # causal multi-head attention + KV-кэш
├── block.py            # SwiGLU и трансформерный блок
└── transformer.py      # TransformerLM (tied embeddings)
chat_format.py          # формат диалога: спецтокены ролей, маски лосса, промпты
prepare_data.py         # скачивание датасетов → data/corpus/*.jsonl
train_tokenizer.py      # BPE-токенизатор 16k со спецтокенами → tokenizer.json
tokenize_corpus.py      # корпус → uint16-шарды + маски (data/tokenized)
dataset.py              # memmap-датасет со случайными окнами + валидация
train.py                # обучение: AMP, DDP, расписание LR, resume, val, samples
generate.py             # генерация: KV-кэш, temperature/top-k/top-p/rep. penalty
chat.py                 # консольный чат с историей
selftest.py             # проверка всего пайплайна на синтетике
golden_prompts.json     # фиксированные промпты для сравнения чекпоинтов
main.py                 # проверка окружения
```

## Если что-то пошло не так

- **`ValueError: В tokenizer.json нет спецтокенов...`** — пересобери токенизатор:
  `python train_tokenizer.py` (готовый файл из репозитория проверку проходит);
- **CUDA out of memory** — уменьши батч: `--micro-batch 16` (или 12);
- **обучение зависло на инициализации DDP** — попробуй `NCCL_P2P_DISABLE=1 python train.py ...`
  или `--no-ddp` (будет одна GPU);
- **скорость сильно ниже ожидаемой** — проверь, что python-процесс видит обе GPU
  (`python main.py`), и что интернет в Kaggle-сессии включён (для `pip install`);
- **сессия оборвалась по лимиту** — ничего страшного: чекпоинты сохраняются каждые
  15 минут, в новой сессии просто запусти ту же команду (`--resume auto`).

## Ограничения

- ~112M параметров и ~3B токенов — это уровень «приятный собеседник на бытовые темы»:
  модель отвечает связно и поддерживает разговор, но плохо знает факты, путается в
  математике и коде, может выдумывать детали;
- максимальная длина ответа и контекста ограничена (`context_length=768`);
- качество напрямую зависит от объёма прогонов: каждый дополнительный миллиард
  токенов заметно улучшает связность.

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
