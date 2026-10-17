"""Обучение TransformerLM: AMP, расписание LR, resume, DDP на нескольких GPU.

Примеры:
    # быстрая проверка (одна GPU или CPU)
    python train.py --preset dev --phase pretrain --max-tokens 50000000 --session-hours 1
    # боевой запуск на Kaggle (2xT4)
    python train.py --preset main --phase pretrain
    # продолжение с последнего чекпоинта
    python train.py --preset main --phase pretrain --resume auto
    # начать заново, игнорируя чекпоинты
    python train.py --preset main --phase pretrain --resume none

Особенности под Kaggle: обучение идёт сессиями, каждая сессия мягко останавливается
по времени (--session-hours) и сохраняет ПОЛНЫЙ чекпоинт (модель + оптимизатор +
scaler + счётчики), поэтому между сессиями ничего не теряется.
"""

import argparse
import contextlib
import glob
import inspect
import json
import math
import os
import shutil
import signal
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from chat_format import build_prompt, check_tokenizer, decode_ids, stop_ids
from dataset import TokenStreamDataset, collate_batch, load_val_batches
from generate import generate_ids
from model.config import (CHECKPOINT_DIR, CKPT_MINUTES, LOG_EVERY_STEPS, MODEL_PRESETS,
                          NUM_WORKERS, PHASES, SAMPLE_EVERY_STEPS, SESSION_HOURS,
                          TOKENIZED_DIR, TOKENIZER_PATH, VAL_EVERY_STEPS)
from model.transformer import TransformerLM

_STOP = False
VAL_BATCH = 8
# device_id в init_process_group появился только в torch 2.6: подставляем его
# условно, чтобы проект не терял совместимость со старыми torch
_PG_HAS_DEVICE_ID = "device_id" in inspect.signature(dist.init_process_group).parameters


def _on_signal(signum, _frame):
    global _STOP
    _STOP = True
    print(f"\n[сигнал {signum}] завершусь после текущего шага и сохранюсь", flush=True)


def _autocast(enabled, dtype):
    if enabled:
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # старые версии torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location=map_location)


def _set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(MODEL_PRESETS), default="main")
    parser.add_argument("--phase", choices=list(PHASES), default="pretrain")
    parser.add_argument("--data-dir", default=TOKENIZED_DIR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--session-hours", type=float, default=SESSION_HOURS)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", default="auto", help="auto | none | путь к .pt")
    parser.add_argument("--no-ddp", action="store_true", help="использовать только одну GPU")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ckpt-minutes", type=float, default=CKPT_MINUTES)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY_STEPS)
    parser.add_argument("--val-batch", type=int, default=VAL_BATCH)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY_STEPS)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_STEPS)
    return parser.parse_args()


def resolve_resume(args):
    if args.resume.lower() in ("none", ""):
        return None
    if args.resume.lower() != "auto":
        return args.resume
    if not os.path.isdir(args.out_dir):
        return None
    latest = os.path.join(args.out_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    # inference.pt не подходит для продолжения: в нём нет оптимизатора и счётчиков
    candidates = [p for p in glob.glob(os.path.join(args.out_dir, "*.pt"))
                  if os.path.basename(p) != "inference.pt"]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    # страховка от фрагментации памяти: без неё к концу длинной сессии
    # аллокатор чаще падает с OOM. Значение можно переопределить до запуска.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    model_cfg = MODEL_PRESETS[args.preset]
    phase = PHASES[args.phase]
    if args.out_dir is None:
        args.out_dir = os.path.join(CHECKPOINT_DIR, args.phase)
    if args.max_tokens:
        phase = replace(phase, total_tokens=args.max_tokens)
    if args.micro_batch:
        phase = replace(phase, micro_batch=args.micro_batch)
    if args.grad_accum:
        phase = replace(phase, grad_accum=args.grad_accum)
    if args.peak_lr:
        phase = replace(phase, peak_lr=args.peak_lr)

    world_size = torch.cuda.device_count() if (torch.cuda.is_available() and not args.no_ddp) else 1
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        mp.spawn(worker, nprocs=world_size, args=(world_size, args, model_cfg, phase), join=True)
    else:
        worker(0, 1, args, model_cfg, phase)


def worker(rank, world_size, args, model_cfg, phase):
    global _STOP
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if world_size > 1:
        torch.cuda.set_device(rank)
        extra = {"device_id": device} if _PG_HAS_DEVICE_ID else {}
        dist.init_process_group("nccl", rank=rank, world_size=world_size, **extra)
    is_main = rank == 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    try:  # чтобы успеть сохраниться, если Kaggle пришлёт SIGTERM
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
    except (ValueError, AttributeError):
        pass

    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer(tokenizer, verbose=is_main)
    if tokenizer.get_vocab_size() != model_cfg.vocab_size:
        if is_main:
            print(f"vocab_size: беру {tokenizer.get_vocab_size()} из tokenizer.json "
                  f"(в пресете было {model_cfg.vocab_size})")
        model_cfg = replace(model_cfg, vocab_size=tokenizer.get_vocab_size())

    os.makedirs(args.out_dir, exist_ok=True)
    model = TransformerLM(model_cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False, gradient_as_bucket_view=True)
    raw = model.module if isinstance(model, DDP) else model
    if is_main:
        print(f"Модель: {raw.n_params() / 1e6:.1f}M параметров | GPU: {world_size} | фаза: {phase.name}")

    decay, no_decay = [], []
    for name, param in raw.named_parameters():
        if param.dim() >= 2 and "embedding" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": phase.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=phase.peak_lr, betas=(0.9, 0.95), eps=1e-8)

    # is_bf16_supported() по умолчанию учитывает и эмуляцию: на Turing/Volta (T4)
    # она вернёт True, и обучение уйдёт на эмулированный bf16 — а это медленные
    # ядра внимания и лишние гигабайты памяти (на T4 такой прогон падает с OOM).
    # Спрашиваем строго про аппаратную поддержку.
    use_fp16 = use_cuda and not torch.cuda.is_bf16_supported(including_emulation=False)
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = _make_scaler(use_fp16)

    step, tokens_seen, phase_tokens, total_seconds = 0, 0, 0, 0.0
    best_val = float("inf")
    resume_path = resolve_resume(args)
    if resume_path and not os.path.exists(resume_path):
        raise SystemExit(f"Чекпоинт не найден: {resume_path} (для старта с нуля: --resume none)")
    if resume_path:
        state = _torch_load(resume_path, device)
        if not (isinstance(state, dict) and "model" in state):
            raise SystemExit(f"{resume_path} — не полный чекпоинт train.py (нет ключа 'model'). "
                             "Начать заново: --resume none")
        if state.get("model_config", {}).get("vocab_size") not in (None, model_cfg.vocab_size):
            raise SystemExit("vocab_size чекпоинта не совпадает с tokenizer.json — переобучи чекпоинт")
        raw.load_state_dict(state["model"])
        if state.get("optim"):
            optimizer.load_state_dict(state["optim"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step = state.get("step", 0)
        tokens_seen = state.get("tokens", 0)
        total_seconds = state.get("total_seconds", 0.0)
        best_val = state.get("best_val", float("inf"))
        saved_phase = state.get("phase", phase.name)
        if saved_phase == phase.name:
            # та же фаза: продолжаем её счётчик (в старых чекпоинтах его нет)
            phase_tokens = state.get("phase_tokens", tokens_seen)
        else:
            phase_tokens = 0  # новая фаза (например, pretrain -> sft): расписание и лимит с нуля
        if is_main:
            print(f"Продолжаю с {resume_path}: шаг {step}, {tokens_seen / 1e9:.3f}B токенов "
                  f"({phase.name}: {phase_tokens / 1e6:.0f}M)")

    tokens_per_step = phase.micro_batch * phase.grad_accum * world_size * model_cfg.context_length
    dataset = TokenStreamDataset(os.path.join(args.data_dir, "train"), phase.mix,
                                 model_cfg.context_length, seed=args.seed + step,
                                 rank=rank, world_size=world_size)
    loader = DataLoader(dataset, batch_size=phase.micro_batch, num_workers=args.num_workers,
                        collate_fn=collate_batch, pin_memory=use_cuda, drop_last=True,
                        prefetch_factor=4 if args.num_workers > 0 else None)

    val_batches = {}
    golden = []
    if is_main:
        source_names = [name for name, _, _ in phase.mix]
        masked_sources = {name for name, _, use_mask in phase.mix if use_mask}
        val_batches = load_val_batches(os.path.join(args.data_dir, "val"), source_names,
                                       model_cfg.context_length, args.val_batch, args.val_count,
                                       masked_sources=masked_sources)
        if os.path.exists("golden_prompts.json"):
            with open("golden_prompts.json", encoding="utf-8") as f:
                golden = json.load(f)

    warmup_tokens = phase.warmup_steps * tokens_per_step

    def lr_at(tokens):
        if tokens < warmup_tokens:
            return phase.peak_lr * max(tokens, 1) / warmup_tokens
        progress = min(1.0, (tokens - warmup_tokens) / max(1, phase.total_tokens - warmup_tokens))
        return phase.peak_lr * (phase.min_lr_ratio + (1 - phase.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    if is_main:
        planned_steps = max(1, (phase.total_tokens - phase_tokens) // tokens_per_step)
        print(f"Токенов на шаг: {tokens_per_step:,} | шагов в плане: ~{planned_steps:,} | "
              f"устройств: {world_size} | dtype: {'fp16' if use_fp16 else (amp_dtype if use_cuda else 'fp32')}")

    def barrier():
        if world_size > 1:
            dist.barrier()

    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")

    def write_metrics(record):
        if is_main:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint(best=False):
        if not is_main:
            return
        state = {
            "model": raw.state_dict(),
            "optim": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_fp16 else None,
            "step": step,
            "tokens": tokens_seen,
            "phase_tokens": phase_tokens,
            "best_val": best_val,
            "total_seconds": total_seconds + (time.time() - session_start),
            "phase": phase.name,
            "preset": args.preset,
            "model_config": asdict(model_cfg),
            "phase_config": asdict(phase),
        }

        def write_atomic(path):
            tmp = path + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, path)

        if best:
            write_atomic(os.path.join(args.out_dir, "best.pt"))
        else:
            latest = os.path.join(args.out_dir, "latest.pt")
            write_atomic(latest)
            numbered_path = os.path.join(args.out_dir, f"ckpt_{step:08d}.pt")
            if os.path.exists(numbered_path):
                os.remove(numbered_path)
            try:
                os.link(latest, numbered_path)  # снапшот без повторной сериализации
            except OSError:
                shutil.copyfile(latest, numbered_path)
        numbered = sorted(glob.glob(os.path.join(args.out_dir, "ckpt_*.pt")))
        for old in numbered[:-2]:
            os.remove(old)

        half_state = {k: v.half() for k, v in raw.state_dict().items()}
        torch.save({"model": half_state, "model_config": asdict(model_cfg)},
                   os.path.join(args.out_dir, "inference.pt"))
        print(f"[чекпоинт] шаг {step}, {tokens_seen / 1e9:.3f}B токенов", flush=True)

    @torch.no_grad()
    def run_val():
        raw.eval()
        results = {}
        for name, batch in val_batches.items():
            x, y, mask = [t.to(device) for t in batch]
            with _autocast(use_cuda, amp_dtype):
                logits = raw(x)
                losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none")
            weights = mask.view(-1).float()
            loss = ((losses * weights).sum() / weights.sum().clamp_min(1)).item()
            results[name] = round(loss, 4)
        raw.train()
        return results

    def run_samples():
        raw.eval()
        lines = [f"# шаг {step}, {tokens_seen / 1e9:.3f}B токенов\n"]
        for item in golden:
            messages = item["messages"]
            ids = build_prompt(tokenizer, messages)
            out = generate_ids(raw, torch.tensor([ids], dtype=torch.long), max_new_tokens=96,
                               temperature=0.7, top_k=0, top_p=0.95, repetition_penalty=1.1,
                               stop_token_ids=stop_ids(tokenizer))
            reply = decode_ids(tokenizer, out[0].tolist()[len(ids):]).strip()
            prompt_text = " | ".join(m["content"] for m in messages)
            lines.append(f"\nUSER: {prompt_text}\nBOT: {reply}\n")
        sample_dir = os.path.join(args.out_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, f"step_{step:08d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        raw.train()

    model.train()
    data_iter = iter(loader)
    session_start = time.time()
    last_ckpt = time.time()
    loss_window = []
    log_tokens = tokens_seen
    log_time = time.time()
    stop_reason = "цикл завершён"

    while True:
        local_stop = _STOP or phase_tokens >= phase.total_tokens or \
            (time.time() - session_start) >= args.session_hours * 3600
        local_ckpt = (time.time() - last_ckpt) >= args.ckpt_minutes * 60
        flags = torch.tensor([1 if local_stop else 0, 1 if local_ckpt else 0], device=device)
        if world_size > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        do_ckpt = bool(flags[1].item())
        if flags[0].item():
            if _STOP:
                stop_reason = "получен сигнал остановки"
            elif phase_tokens >= phase.total_tokens:
                stop_reason = "этап завершён — набрано плановое число токенов"
            else:
                stop_reason = f"сессия достигла лимита {args.session_hours} ч"
            break

        # LR задаём ДО шага оптимизатора, а не после: иначе первая итерация
        # обучения шла бы с peak_lr из конструктора AdamW (для main-пресета
        # ~6e-4 вместо warmup ~6e-7), а вся кривая расписания была бы сдвинута
        # на один шаг вперёд.
        lr = lr_at(phase_tokens)
        _set_lr(optimizer, lr)

        loss_accum = 0.0
        grad_norm = torch.tensor(0.0)
        for micro in range(phase.grad_accum):
            x, y, mask = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            no_sync = model.no_sync() if (world_size > 1 and micro < phase.grad_accum - 1) \
                else contextlib.nullcontext()
            with no_sync:
                with _autocast(use_cuda, amp_dtype):
                    logits = model(x)
                    losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none")
                    weights = mask.view(-1).float()
                    loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
                scaler.scale(loss / phase.grad_accum).backward()
            loss_accum += loss.item()

        if use_fp16:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), phase.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), phase.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += tokens_per_step
        phase_tokens += tokens_per_step
        loss_window.append(loss_accum / phase.grad_accum)

        if step % args.log_every == 0:
            now = time.time()
            avg_loss = sum(loss_window) / len(loss_window)
            toks_per_sec = (tokens_seen - log_tokens) / max(1e-6, now - log_time)
            remaining = max(0, phase.total_tokens - phase_tokens) / max(1e-6, toks_per_sec)
            session_min = (now - session_start) / 60
            if is_main:
                print(f"шаг {step:>7} | {tokens_seen / 1e9:.3f}B ({100 * phase_tokens / phase.total_tokens:5.1f}%) "
                      f"| loss {avg_loss:.4f} | ppl {math.exp(avg_loss):7.1f} | lr {lr:.2e} "
                      f"| |grad| {float(grad_norm):6.2f} | {toks_per_sec / 1000:5.1f}k ток/с "
                      f"| сессия {session_min:5.0f} мин | ETA {remaining / 60:4.0f} мин", flush=True)
                write_metrics({"step": step, "tokens": tokens_seen, "phase_tokens": phase_tokens,
                               "loss": round(avg_loss, 4),
                               "lr": lr, "grad_norm": round(float(grad_norm), 3),
                               "tokens_per_sec": round(toks_per_sec),
                               "session_seconds": round(now - session_start)})
            loss_window.clear()
            log_tokens = tokens_seen
            log_time = now

        if step % args.val_every == 0:
            barrier()
            if is_main and val_batches:
                results = run_val()
                summary = " | ".join(f"{name}: ppl {math.exp(loss):.1f}" for name, loss in results.items())
                print(f"[валидация] шаг {step}: {summary}", flush=True)
                write_metrics({"step": step, "val": results})
                mean_loss = sum(results.values()) / len(results)
                if mean_loss < best_val:
                    best_val = mean_loss
                    save_checkpoint(best=True)
            barrier()

        if step % args.sample_every == 0:
            barrier()
            if is_main and golden:
                run_samples()
                print(f"[примеры] ответы записаны в {args.out_dir}/samples/step_{step:08d}.txt", flush=True)
            barrier()

        if do_ckpt:
            barrier()
            save_checkpoint()
            last_ckpt = time.time()
            barrier()

    barrier()
    save_checkpoint()
    barrier()
    if is_main:
        print(f"\nОстановка: {stop_reason}")
        print(f"Шагов: {step} | токенов: {tokens_seen / 1e9:.3f}B "
              f"(фаза: {phase_tokens / 1e9:.3f}B) | "
              f"время сессии: {(time.time() - session_start) / 3600:.2f} ч")
        write_metrics({"step": step, "tokens": tokens_seen, "phase_tokens": phase_tokens,
                       "done": True, "reason": stop_reason})
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
