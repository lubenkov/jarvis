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
   (веб ~68%, чаты ~27%, новости/статьи ~5%), цель ~1.2B токенов;
2. **SFT** — дожим на диалогах: лосс считается **только на ответах ассистента**
   (токены пользователя маскируются), цель ~0.15B токенов.

Объём подобран под собранный корпус (~1B токенов: веб 807M, чаты 110M, новости
67M). При таких весах получается примерно один проход по вебу, три по диалогам и
один по новостям. Диалогов в корпусе мало, поэтому большие веса на чаты (как было
раньше) означали бы зазубривание конкретных диалогов, а не обучение.

Формат диалога (одинаковый при обучении и генерации):

```
<|user|>привет<|assistant|>привет! как дела?<|endoftext|><|user|>...
```

Модель (`--preset main`, ~112M параметров): d_model 768, 14 слоёв, 12 голов,
SwiGLU d_ff 2048, контекст 768, словарь 16384, tied embeddings, без bias,
fp16 AMP на 2×T4 через DDP.

## Результаты

Обучение на 2×T4 заняло ~12.6 GPU-часов: претрейн — 1.142B токенов за 15 491 шаг,
SFT — 0.150B за 2 035 шагов. Веса для чата — `checkpoints/sft/inference.pt`.

Сравнивать модели с разными словарями по перплексии «на токен» нельзя (у нас 16k
токенов, у ruGPT-3 — 50k), поэтому общая мера ниже — **биты на символ (BPC)** на
отложенном тексте, которого модель не видела. Все системы получают один и тот же
текст; окна идут без перекрытия, длиной в контекст каждой модели.

| система | web_ru | texts_ru | chat_ru |
|---|---|---|---|
| jarvis pretrain | 1.18 | 1.02 | 0.92 |
| jarvis sft | 1.22 | 1.07 | 0.91 |
| ruGPT-3 small (125M) | 1.07 | 0.93 | 0.98 |
| gzip | 4.12 | 4.12 | 3.66 |
| xz | 3.08 | 3.01 | 2.74 |

Воспроизведение (пишет `results/bpc.md` и `results/bpc.json`):

```bash
python tools/bench_bpc.py --ckpt checkpoints/pretrain/inference.pt checkpoints/sft/inference.pt
```

Скрипт сам скачает ruGPT-3 (~500 МБ, разово); чтобы считать без сравнения —
`--baseline none`.

Что видно по цифрам:

- модель сжимает русский текст в 3-4 раза плотнее `gzip` — для 112M параметров это
  заметно лучше, чем просто запоминание частот;
- **ruGPT-3 small (125M) впереди на вебе и новостях**: 1.07 против наших 1.18 и 0.93
  против 1.02, то есть мы отстаём примерно на 10%. Это ожидаемо: ruGPT-3 обучалась на
  заметно большем объёме текста, а у нас всего ~1.3B токенов. Зато **на диалогах мы
  впереди** (0.91 против 0.98): в нашем корпусе есть диалоги в нашем формате, которых
  ruGPT-3 не видела;
- **SFT не улучшил языковую метрику**: на вебе и новостях стало чуть хуже (1.22
  против 1.18 и 1.07 против 1.02), на диалогах — без изменений. Улучшилось ровно то,
  что SFT и оптимизировал: лосс по токенам ассистента 6.96 → 6.63, плюс поведение —
  пропали циклы, оскорбления и обрывы ответа. Посмотреть «до/после» глазами:
  `python tools/compare.py` (пишет `samples/compare_pretrain_vs_sft.txt`).

Сравнение с ruGPT-3: тот же текст, у каждой модели — свой токенизатор, окна по её
контексту. Словарь ruGPT-3 крупнее (50k против 16k), поэтому символов на токен у неё
больше (4.3 против 3.8 на вебе) — именно поэтому считаем биты на символ, а не
перплексию на токен.

> Числа из лога обучения («val ppl 22.9 → 19.7») с этими не сходятся, и сравнивать их
> нельзя: лог считает ppl по случайной выборке окон из того же отложенного набора
> (на другой выборке порядок может перевернуться), а для диалогов — ещё и по маске,
> то есть это не то же самое, что немаскированный претрейн.

## Быстрый старт

```bash
pip install -r requirements.txt

# 1) проверка окружения (torch, GPU)
python main.py

# 2) самопроверка всей пайплайна на синтетических данных (~пара минут, без интернета)
python selftest.py

# 3) чат с обученной моделью
python chat.py --ckpt checkpoints/pretrain/inference.pt
```

`selftest.py` прогоняет весь путь: мини-корпус → токенизатор → шарды →
несколько шагов обучения → resume → генерация. Если он зелёный — пайплайн рабочий,
можно запускать боевое обучение.

## Полный пайплайн на Kaggle

Нужны 2 сессии: **CPU** (подготовка данных, интернет включён) и **GPU T4×2** (обучение).

Рабочая ветка — `develop` (на GitHub веткой по умолчанию остался устаревший
`master`), поэтому клонируй явно: `git clone -b develop ...`.

В актуальных сессиях Kaggle датасеты монтируются в `/kaggle/input/datasets/<ник>/<слаг>/`,
поэтому в командах ниже ник заменён на `*` — так пути не зависят от имени аккаунта.

### Сессия 1 (CPU + Internet): корпус и токенизатор

```bash
git clone -b develop https://github.com/lubenkov/jarvis.git && cd jarvis
pip install -q -r requirements.txt

# схемы датасетов на месте?
python prepare_data.py --what all --dry-run

# веб-текст (~0.8B токенов, 2-4 часа), новости, диалоги
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

> Веб-загрузчик датасетов не принимает папку целиком, зато Kaggle сам распаковывает
> ZIP. Собери архив, в корне которого лежат `tokenizer.json` и папка `tokenized/`:

```powershell
Compress-Archive -Path tokenizer.json, data\tokenized -DestinationPath jarvis-corpus.zip
```

### Сессия 2 (GPU T4×2): pretrain

Добавь к ноутбуку два Input Dataset: `jarvis-corpus` и (после первой сессии) `jarvis-checkpoints`.

```bash
git clone -b develop https://github.com/lubenkov/jarvis.git && cd jarvis
pip install -q -r requirements.txt

mkdir -p data
cp -r /kaggle/input/datasets/*/jarvis-corpus/tokenized data/
cp /kaggle/input/datasets/*/jarvis-corpus/tokenizer.json .

mkdir -p checkpoints
cp -r /kaggle/input/datasets/*/jarvis-checkpoints/* checkpoints/ 2>/dev/null || true

# контроль: в train должно быть ~1B токенов по трём источникам
python dataset.py

# дым-тест: 10M токенов, ~136 шагов, минут десять — ловит ошибки до длинного прогона
python train.py --preset main --phase pretrain --max-tokens 10000000 \
    --session-hours 0.7 --micro-batch 16 --grad-accum 3 --sample-every 20

# сессия до 11 часов, потом мягкая остановка с полным чекпоинтом
python train.py --preset main --phase pretrain --session-hours 11 \
    --micro-batch 16 --grad-accum 3
```

> `--micro-batch 16 --grad-accum 3` — те же 73 728 токенов на шаг, что и дефолтные
> 24×2, но с запасом по памяти на 16 ГБ T4. Точность выбирается автоматически:
> fp16 на Turing/Volta, bf16 на Ampere+ (принудительно — `--dtype`).

Чекпоинты складываются в `checkpoints/pretrain/`:

- `latest.pt` — полный чекпоинт (модель + оптимизатор + счётчики) для продолжения, ~1.4 ГБ;
- `ckpt_XXXXXXXX.pt` — последние два шага на всякий случай;
- `best.pt` — лучший по валидации;
- `inference.pt` — компактная fp16-копия весов для чата (~250 МБ);
- `metrics.jsonl` — лог (loss, lr, скорость, валидация);
- `samples/step_*.txt` — ответы модели на фиксированные golden-промпты.

Между сессиями: **Save Version** (или скачай чекпоинты), в новой сессии положи их в
`checkpoints/` и запусти ту же команду — обучение продолжится с того же шага
(`--resume auto` по умолчанию; `--resume none` — начать заново).

Претрейн 1.2B токенов при ~24k ток/с на 2×T4 — это около 14 часов, то есть две
сессии по 11 ч: первая мягко остановится на ~0.95B, вторая добьёт остаток и сама
завершит этап.

### Сессия 3 (GPU T4×2): SFT

```bash
cp -r /kaggle/input/datasets/*/jarvis-checkpoints/pretrain checkpoints/
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

### Локально: телеграм-бот

```bash
pip install -r requirements-bot.txt
cp .env.example .env      # токен от @BotFather
python bot.py --selftest  # проверка генерации без телеграма и токена
python bot.py
```

Бот держит историю отдельно для каждого чата и обрабатывает сообщения по одному:
инференс блокирующий, поэтому очередь общая, а не на пользователя. Белого списка
нет: бот ответит любому, кто его найдёт. Чтобы локальный инференс не положили,
есть лимиты: `--max-active-chats` (по умолчанию 8), `--chat-ttl` (30 минут),
`--max-queue` (8 апдейтов в очереди) и `--max-concurrent` (1 генерация).
Частоту ответов ограничивает `--min-interval` (по умолчанию 3 секунды).
Команды: `/reset`, `/temp 0.7`, `/stats`.

Устройство выбирается автоматически: CUDA, если доступна, иначе CPU. На CUDA
модель грузится в fp16. Проверить окружение можно через `python main.py`, а
принудительно выбрать GPU — так:

```bash
python bot.py --device cuda
```

Если в venv стоит CPU-сборка torch, GPU не включится: нужна CUDA-сборка PyTorch.

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
- `samples/step_*.txt` — самое честное: сравнивай ответы на одни и те же промпты между шагами
  (первый снимок пишется в начале каждой сессии, а не через тысячи шагов);
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
bot.py                  # телеграм-бот поверх той же модели
tools/compare.py        # сравнение двух чекпоинтов на одних промптах
tools/bench_bpc.py      # BPC на отложенном тексте против ruGPT-3 и архиваторов
results/                # измеренные BPC (bpc.md, bpc.json)
```

## Если что-то пошло не так

- **`ValueError: В tokenizer.json нет спецтокенов...`** — пересобери токенизатор:
  `python train_tokenizer.py` (готовый файл из репозитория проверку проходит);
- **CUDA out of memory** — уменьши батч и добери накоплением, чтобы токенов на шаг
  осталось столько же: `--micro-batch 12 --grad-accum 4`;
- **странно медленно или `dtype: bf16` на T4** — точность подбирается автоматически
  (fp16 на Turing/Volta, bf16 на Ampere+); принудительно: `--dtype fp16`;
- **обучение зависло на инициализации DDP** — попробуй `NCCL_P2P_DISABLE=1 python train.py ...`
  или `--no-ddp` (будет одна GPU);
- **скорость сильно ниже ожидаемой** — проверь, что python-процесс видит обе GPU
  (`python main.py`), и что интернет в Kaggle-сессии включён (для `pip install`);
- **сессия оборвалась по лимиту** — ничего страшного: чекпоинты сохраняются каждые
  15 минут, в новой сессии просто запусти ту же команду (`--resume auto`).

## Ограничения

- ~112M параметров и ~1.3B токенов (1.14B претрейн + 0.15B SFT) — это уровень
  «приятный собеседник на бытовые темы»:
  модель отвечает связно и поддерживает разговор, но плохо знает факты, путается в
  математике и коде, может выдумывать детали;
- максимальная длина ответа и контекста ограничена (`context_length=768`);
- качество напрямую зависит от объёма прогонов: каждый дополнительный миллиард
  токенов заметно улучшает связность, но упирается в объём корпуса (сейчас ~1B
  токенов, дальше придётся докачивать данные).

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
