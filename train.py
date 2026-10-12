import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import (
    batch_size, learning_rate, grad_accum_steps,
    max_train_seconds, log_interval, checkpoint_path, vocab_size,
    checkpoint_interval_seconds, progress_path,
)
from model.transformer import TransformerLM
from dataset import TextDataset


def load_total_trained_seconds():
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r") as f:
                return json.load(f)["total_trained_seconds"]
        except (json.JSONDecodeError, KeyError):
            print("Файл прогресса повреждён, начинаю отсчёт заново.")
            return 0.0
    return 0.0


def save_total_trained_seconds(seconds):
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"total_trained_seconds": seconds}, f)
    os.replace(tmp_path, progress_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)

    dataset = TextDataset("tokenizer.json", "data/raw/dialogues.txt")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = TransformerLM().to(device)

    if os.path.exists(checkpoint_path):
        print("Найден чекпоинт, загружаю веса и продолжаю обучение...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    total_trained_seconds = load_total_trained_seconds()
    print(f"Уже накоплено обучения: {total_trained_seconds / 3600:.2f} ч из {max_train_seconds / 3600:.1f} ч")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device="cuda")

    session_start_time = time.time()
    last_checkpoint_time = time.time()
    step = 0
    optimizer.zero_grad()

    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        session_elapsed = time.time() - session_start_time
        total_elapsed = total_trained_seconds + session_elapsed

        if step % log_interval == 0:
            print(f"Шаг {step} | loss {loss.item() * grad_accum_steps:.4f} | сессия {session_elapsed / 60:.1f} мин | всего {total_elapsed / 3600:.2f} ч")

        if time.time() - last_checkpoint_time > checkpoint_interval_seconds:
            torch.save(model.state_dict(), checkpoint_path)
            save_total_trained_seconds(total_elapsed)
            last_checkpoint_time = time.time()
            print(f"Промежуточное сохранение на шаге {step}")

        step += 1

        if total_elapsed > max_train_seconds:
            print("Достигнут суммарный лимит времени, останавливаюсь.")
            break

    final_total = total_trained_seconds + (time.time() - session_start_time)
    torch.save(model.state_dict(), checkpoint_path)
    save_total_trained_seconds(final_total)
    print("Модель сохранена в", checkpoint_path)
    print(f"Всего накоплено обучения: {final_total / 3600:.2f} ч")


if __name__ == "__main__":
    main()