import argparse
import asyncio
import logging
import os
import time

import torch
from tokenizers import Tokenizer

from chat_format import build_prompt, check_tokenizer, decode_ids, stop_ids, truncate_messages
from generate import generate_ids, load_model
from model.config import REPETITION_PENALTY, TEMPERATURE, TOKENIZER_PATH, TOP_K, TOP_P

MAX_HISTORY_MESSAGES = 16
TELEGRAM_LIMIT = 4000
BUSY_TEXT = "Бот сейчас занят: слишком много активных диалогов. Попробуй позже."
START_TEXT = (
    "Привет! Я джарвис и у меня 300 iq\n\n"
    "Бот отвечает по очереди; при высокой нагрузке может быть занят.\n\n"
    "Команды:\n"
    "/reset — очистить историю\n"
    "/temp 0.7 — температура (0 — жадный режим)\n"
    "/stats — текущие параметры генерации"
)


def resolve_device(name):
    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is unavailable: the venv has a CPU-only torch build "
                f"({torch.__version__}, CUDA {torch.version.cuda}) or no GPU. "
                "Install a CUDA PyTorch build or run with --device cpu."
            )
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise SystemExit(
                f"CUDA device cuda:{index} not found "
                f"(available devices: {torch.cuda.device_count()})."
            )
        device = torch.device("cuda", index)
    return device


def device_label(device):
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return device.type


class Jarvis:
    def __init__(self, args):
        self.device = resolve_device(args.device)
        self.tokenizer = Tokenizer.from_file(args.tokenizer)
        check_tokenizer(self.tokenizer)
        self.model, cfg = load_model(args.ckpt, self.device)
        self.stops = stop_ids(self.tokenizer)
        self.max_new = args.max_new
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.rep_penalty = args.rep_penalty
        self.temperature = args.temperature
        self.min_interval = args.min_interval
        self.max_prompt_tokens = max(64, cfg.context_length - args.max_new)
        self.max_active = args.max_active_chats
        self.chat_ttl = args.chat_ttl * 60.0
        self.max_queue = args.max_queue
        self.max_concurrent = args.max_concurrent
        self.sem = asyncio.Semaphore(self.max_concurrent)
        self.chats = {}

    def _prune(self, now):
        stale = [chat_id for chat_id, state in self.chats.items()
                 if now - state["seen"] > self.chat_ttl]
        for chat_id in stale:
            self.chats.pop(chat_id, None)

    def state(self, chat_id):
        now = time.time()
        self._prune(now)
        state = self.chats.get(chat_id)
        if state is None:
            if len(self.chats) >= self.max_active:
                return None
            state = {
                "history": [],
                "temperature": self.temperature,
                "last": 0.0,
                "seen": now,
                "lock": asyncio.Lock(),
            }
            self.chats[chat_id] = state
        state["seen"] = now
        return state

    def peek(self, chat_id):
        now = time.time()
        self._prune(now)
        state = self.chats.get(chat_id)
        if state is not None:
            state["seen"] = now
        return state

    def generate(self, chat_id, text, state=None):
        if state is None:
            state = self.state(chat_id)
        if state is None:
            raise RuntimeError("no free slot for a new conversation")
        state["history"].append({"role": "user", "content": text})
        while len(state["history"]) > MAX_HISTORY_MESSAGES:
            state["history"].pop(0)
        messages = truncate_messages(self.tokenizer, state["history"], self.max_prompt_tokens)
        ids = build_prompt(self.tokenizer, messages)
        if len(ids) > self.max_prompt_tokens:
            ids = ids[-self.max_prompt_tokens:]
        out = generate_ids(self.model, torch.tensor([ids], dtype=torch.long), max_new_tokens=self.max_new,
                           temperature=state["temperature"], top_k=self.top_k, top_p=self.top_p,
                           repetition_penalty=self.rep_penalty, stop_token_ids=self.stops)
        reply = decode_ids(self.tokenizer, out[0].tolist()[len(ids):]).strip() or "..."
        state["history"].append({"role": "assistant", "content": reply})
        return reply[:TELEGRAM_LIMIT]


def build_handlers(jarvis):
    from telegram import Update
    from telegram.constants import ChatAction
    from telegram.ext import CommandHandler, MessageHandler, filters

    async def start(update: Update, context):
        await update.message.reply_text(START_TEXT)

    async def reset(update: Update, context):
        state = jarvis.peek(update.effective_chat.id)
        if state is not None:
            async with state["lock"]:
                state["history"] = []
        await update.message.reply_text("История очищена.")

    async def stats(update: Update, context):
        state = jarvis.peek(update.effective_chat.id)
        temperature = state["temperature"] if state is not None else jarvis.temperature
        history = len(state["history"]) if state is not None else 0
        await update.message.reply_text(
            f"temperature={temperature} top_k={jarvis.top_k} top_p={jarvis.top_p} "
            f"rep_penalty={jarvis.rep_penalty} max_new={jarvis.max_new} "
            f"реплик в истории={history} активных диалогов={len(jarvis.chats)}/{jarvis.max_active}")

    async def temp(update: Update, context):
        state = jarvis.peek(update.effective_chat.id)
        if state is None:
            await update.message.reply_text("Сначала напиши сообщение — потом настрою температуру.")
            return
        try:
            temperature = float(context.args[0].replace(",", "."))
        except (IndexError, ValueError, TypeError):
            await update.message.reply_text("Пример: /temp 0.7")
            return
        async with state["lock"]:
            state["temperature"] = temperature
        await update.message.reply_text(f"temperature = {state['temperature']}")

    async def answer(update: Update, context):
        chat_id = update.effective_chat.id
        state = jarvis.state(chat_id)
        if state is None:
            await update.message.reply_text(BUSY_TEXT)
            return
        wait = jarvis.min_interval - (time.time() - state["last"])
        if wait > 0:
            await update.message.reply_text(f"Слишком часто — подожди {wait:.0f} с.")
            return
        state["last"] = time.time()
        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        async with state["lock"]:
            async with jarvis.sem:
                reply = await asyncio.to_thread(jarvis.generate, chat_id, update.message.text, state)
        await update.message.reply_text(reply)

    return [CommandHandler("start", start), CommandHandler("help", start),
            CommandHandler("reset", reset), CommandHandler("stats", stats),
            CommandHandler("temp", temp), MessageHandler(filters.TEXT & ~filters.COMMAND, answer)]


def run_telegram(jarvis, token):
    from telegram.ext import Application

    builder = Application.builder().token(token).update_queue(asyncio.Queue(maxsize=jarvis.max_queue))
    if jarvis.max_concurrent > 1:
        builder = builder.concurrent_updates(True)
    application = builder.build()
    for handler in build_handlers(jarvis):
        application.add_handler(handler)
    print("Bot is running. Press Ctrl+C to stop.")
    application.run_polling()


def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        if os.path.exists(".env"):
            with open(".env", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
        return
    load_dotenv()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return number


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a number > 0")
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/sft/inference.pt")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--rep-penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--max-new", type=positive_int, default=120)
    parser.add_argument("--min-interval", type=float, default=3.0)
    parser.add_argument("--max-active-chats", type=positive_int, default=8,
                        help="maximum number of concurrent conversations")
    parser.add_argument("--chat-ttl", type=positive_float, default=30.0,
                        help="minutes of inactivity before a slot is released")
    parser.add_argument("--max-queue", type=positive_int, default=8,
                        help="maximum number of Telegram updates to queue")
    parser.add_argument("--max-concurrent", type=positive_int, default=1,
                        help="maximum number of concurrent generations")
    parser.add_argument("--selftest", action="store_true", help="test generation without Telegram")
    args = parser.parse_args()

    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    jarvis = Jarvis(args)
    dtype = next(jarvis.model.parameters()).dtype
    print(f"Model: {args.ckpt} ({jarvis.model.n_params() / 1e6:.1f}M params, "
          f"{device_label(jarvis.device)}, {dtype})")
    print(f"Limits: active chats {jarvis.max_active}, queue {jarvis.max_queue}, "
          f"concurrent {jarvis.max_concurrent}, TTL {args.chat_ttl:g} min")

    if args.selftest:
        for text in ("Привет! Как дела?", "Расскажи, что ты умеешь."):
            print(f"\nYou: {text}\nJarvis: {jarvis.generate(1, text)}")
        print("\nMessage receiving and sending were not tested — a token is required.")
        return

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        parser.error("TELEGRAM_BOT_TOKEN is missing: put the token in .env (see .env.example)")
    print("Whitelist disabled: anyone who knows the username can use the bot. "
          "Protection is provided by limits and min-interval.")
    run_telegram(jarvis, token)


if __name__ == "__main__":
    main()
