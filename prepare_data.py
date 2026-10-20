import argparse
import hashlib
import json
import os
import shutil

from chat_format import prepare_messages
from model.config import CORPUS_DIR

MIN_CHARS = 40
MAX_CHARS = 20000
MAX_TURNS = 24
MIN_CYRILLIC_RATIO = 0.5

LIST_KEYS = ("messages", "conversation", "conversations", "dialog", "dialogue", "dialogues", "chats")
PAIR_KEYS = (("instruction", "output"), ("question", "answer"), ("prompt", "response"),
             ("query", "answer"), ("input", "output"))

WEB_REPO = "HuggingFaceFW/fineweb-2"
WEB_BATCH_DOCS = 2048


def cyrillic_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return cyr / len(letters)


def extract_generic(ex):
    for key in LIST_KEYS:
        value = ex.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            messages = []
            for m in value:
                role = m.get("role") or m.get("from") or m.get("speaker")
                content = m.get("content") or m.get("value") or m.get("text") or m.get("message")
                if role is None or content is None:
                    return None
                messages.append({"role": role, "content": content})
            return messages

    for q_key, a_key in PAIR_KEYS:
        question, answer = ex.get(q_key), ex.get(a_key)
        if isinstance(question, str) and isinstance(answer, str) and question.strip() and answer.strip():
            extra = ex.get("input") if q_key == "instruction" else None
            if isinstance(extra, str) and extra.strip():
                question = question.strip() + "\n" + extra.strip()
            return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return None


def extract_jokes(ex):
    context = (ex.get("context") or "").strip()
    utterance = (ex.get("utterance") or "").strip()
    if not utterance:
        return None
    messages = []
    for line in context.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            messages.append({"role": "user", "content": line.lstrip("- ").strip()})
        else:
            messages.append({"role": "assistant", "content": line})
    messages.append({"role": "assistant", "content": utterance})
    return messages


CHAT_SOURCES = [
    {
        "name": "saiga_scored",
        "dataset": "IlyaGusev/saiga_scored",
        "filter": lambda ex: ex.get("language") in (None, "Russian")
                             and (10 if ex.get("opus_score") is None else ex.get("opus_score")) >= 7
                             and not ex.get("is_bad_by_regex", False),
    },
    {
        "name": "grandmaster",
        "dataset": "Vikhrmodels/GrandMaster-PRO-MAX",
        "filter": lambda ex: ex.get("prompt_lang") in (None, "ru") and ex.get("answer_lang") in (None, "ru"),
    },
    {"name": "veles", "dataset": "Vikhrmodels/Veles-2.5"},
    {"name": "den4ik_instructions", "dataset": "Den4ikAI/russian_instructions_2", "limit": 120_000},
    {"name": "jokes", "dataset": "inkoziev/jokes_dialogues", "extract": extract_jokes},
]


def _write_conversation(f, messages, seen_hashes, stats):
    prepared = prepare_messages(messages)
    if not prepared:
        stats["filtered"] += 1
        return False
    if len(prepared) > MAX_TURNS:
        prepared = prepared[:MAX_TURNS]
    joined = "\n".join(content for _, content in prepared)
    if not (MIN_CHARS <= len(joined) <= MAX_CHARS):
        stats["filtered"] += 1
        return False
    if cyrillic_ratio(joined) < MIN_CYRILLIC_RATIO:
        stats["filtered"] += 1
        return False
    if prepared[-1][0] != "assistant":
        stats["filtered"] += 1
        return False
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    if digest in seen_hashes:
        stats["filtered"] += 1
        return False
    seen_hashes.add(digest)

    line = {"messages": [{"role": role, "content": content} for role, content in prepared]}
    f.write(json.dumps(line, ensure_ascii=False) + "\n")
    stats["written"] += 1
    return True


def _web_shards():
    from huggingface_hub import HfApi

    files = [(f.path, f.size or 0)
             for f in HfApi().list_repo_tree(WEB_REPO, path_in_repo="data/rus_Cyrl/train",
                                             repo_type="dataset")
             if f.path.endswith(".parquet")]
    files.sort(key=lambda item: item[0])
    return files


def _open_web_corpus(building_path, stats):
    if not os.path.exists(building_path):
        return open(building_path, "w", encoding="utf-8"), set()

    print(f"  found partial corpus ({os.path.getsize(building_path) / 1e9:.1f} GB) — restoring hashes")
    seen_hashes = set()
    good_bytes = 0
    with open(building_path, "rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            try:
                text = json.loads(raw)["text"]
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                break
            seen_hashes.add(hashlib.md5(text[:2000].encode("utf-8")).hexdigest())
            good_bytes += len(raw)
            stats["written"] += 1
    if good_bytes != os.path.getsize(building_path):
        os.truncate(building_path, good_bytes)
    return open(building_path, "a", encoding="utf-8"), seen_hashes


def build_web(out_path, max_docs, dry_run=False):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    stats = {"read": 0, "written": 0, "filtered": 0}
    limit = 100 if dry_run else max_docs
    building_path = out_path + ".building"
    shard_dir = out_path + ".parts"

    files = _web_shards()
    print(f"  shards in dataset: {len(files)}, reading from the first one")
    os.makedirs(shard_dir, exist_ok=True)
    f, seen_hashes = _open_web_corpus(building_path, stats)
    try:
        for idx, (name, size) in enumerate(files, start=1):
            if stats["written"] >= limit:
                break
            print(f"  shard {idx}/{len(files)}: {os.path.basename(name)} ({size / 1e9:.1f} GB)")
            try:
                local_path = hf_hub_download(WEB_REPO, name, repo_type="dataset",
                                             local_dir=shard_dir)
                parquet_file = pq.ParquetFile(local_path)
                stop = False
                for batch in parquet_file.iter_batches(batch_size=WEB_BATCH_DOCS,
                                                       columns=["text"]):
                    for text in batch.column("text").to_pylist():
                        stats["read"] += 1
                        text = (text or "").strip()
                        if not (400 <= len(text) <= MAX_CHARS):
                            stats["filtered"] += 1
                            continue
                        if cyrillic_ratio(text) < 0.6:
                            stats["filtered"] += 1
                            continue
                        digest = hashlib.md5(text[:2000].encode("utf-8")).hexdigest()
                        if digest in seen_hashes:
                            stats["filtered"] += 1
                            continue
                        seen_hashes.add(digest)

                        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        stats["written"] += 1
                        if stats["written"] % 50000 == 0:
                            print(f"  ... wrote {stats['written']} documents")
                        if stats["written"] >= limit:
                            stop = True
                            break
                    if stop:
                        break
                parquet_file.close()
                os.remove(local_path)
                print(f"  ... shard done: read {stats['read']}, wrote {stats['written']}")
                f.flush()
            except Exception as error:
                print(f"  ! shard {name} failed: {type(error).__name__}: {error}")
    finally:
        f.close()

    os.replace(building_path, out_path)
    shutil.rmtree(shard_dir, ignore_errors=True)
    return stats


def build_texts(out_path, max_docs, dry_run=False):
    from datasets import load_dataset

    stats = {"read": 0, "written": 0, "filtered": 0}
    seen_hashes = set()
    dataset = load_dataset("IlyaGusev/gazeta", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    limit = 100 if dry_run else max_docs

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            stats["read"] += 1
            text = (ex.get("text") or "").strip()
            if not (300 <= len(text) <= MAX_CHARS - 100):
                stats["filtered"] += 1
                continue
            if cyrillic_ratio(text) < 0.5:
                stats["filtered"] += 1
                continue
            digest = hashlib.md5(text[:2000].encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                stats["filtered"] += 1
                continue
            seen_hashes.add(digest)

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if stats["written"] >= limit:
                break
    return stats


def process_chat_source(f, spec, seen_hashes, dry_run=False):
    from datasets import load_dataset

    stats = {"read": 0, "written": 0, "filtered": 0}
    try:
        dataset = load_dataset(spec["dataset"], split=spec.get("split", "train"))
    except Exception as error:
        print(f"  ! failed to load {spec['dataset']}: {error}")
        return stats

    limit = spec.get("limit") or 10 ** 9
    if dry_run:
        limit = 50

    for ex in dataset:
        stats["read"] += 1
        source_filter = spec.get("filter")
        if source_filter and not source_filter(ex):
            stats["filtered"] += 1
            continue
        messages = spec["extract"](ex) if spec.get("extract") else extract_generic(ex)
        if _write_conversation(f, messages or [], seen_hashes, stats) and stats["written"] >= limit:
            break
    return stats


def build_oasst(f, seen_hashes, dry_run=False):
    from datasets import load_dataset

    stats = {"read": 0, "written": 0, "filtered": 0}
    rows = [r for r in load_dataset("OpenAssistant/oasst1", split="train")
            if r.get("lang") == "ru" and not r.get("deleted")]
    by_id = {r["message_id"]: r for r in rows}
    children = {}
    for r in rows:
        children.setdefault(r.get("parent_id"), []).append(r["message_id"])

    def walk(message_id, path):
        node = by_id[message_id]
        path = path + [node]
        kids = [k for k in children.get(message_id, []) if k in by_id]
        if not kids:
            yield path
            return
        for k in kids:
            yield from walk(k, path)

    limit = 50 if dry_run else 10 ** 9
    for r in rows:
        if r.get("parent_id") is not None:
            continue
        for path in walk(r["message_id"], []):
            stats["read"] += 1
            messages = [{"role": "user" if n["role"] == "prompter" else "assistant", "content": n["text"]}
                        for n in path]
            _write_conversation(f, messages, seen_hashes, stats)
            if stats["written"] >= limit:
                return stats
    return stats


def save_stats(out_dir, name, stats):
    path = os.path.join(out_dir, f"{name}.stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Summary {name}: read {stats['read']}, wrote {stats['written']}, "
          f"filtered {stats['filtered']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["web", "texts", "chats", "all"], default="all")
    parser.add_argument("--out-dir", default=CORPUS_DIR)
    parser.add_argument("--max-web-docs", type=int, default=1_000_000)
    parser.add_argument("--max-texts-docs", type=int, default=150_000)
    parser.add_argument("--dry-run", action="store_true", help="take 50-100 examples and validate schemas")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.what in ("web", "all"):
        print("=== web_ru (fineweb-2 rus_Cyrl) ===")
        stats = build_web(os.path.join(args.out_dir, "web_ru.jsonl"), args.max_web_docs, args.dry_run)
        save_stats(args.out_dir, "web_ru", stats)

    if args.what in ("texts", "all"):
        print("=== texts_ru (gazeta) ===")
        stats = build_texts(os.path.join(args.out_dir, "texts_ru.jsonl"), args.max_texts_docs, args.dry_run)
        save_stats(args.out_dir, "texts_ru", stats)

    if args.what in ("chats", "all"):
        print("=== chat_ru ===")
        path = os.path.join(args.out_dir, "chat_ru.jsonl")
        seen_hashes = set()
        total = {"read": 0, "written": 0, "filtered": 0}
        with open(path, "w", encoding="utf-8") as f:
            for spec in CHAT_SOURCES:
                print(f"--- {spec['name']} ---")
                stats = process_chat_source(f, spec, seen_hashes, args.dry_run)
                save_stats(args.out_dir, "chat_ru." + spec["name"], stats)
                for k in total:
                    total[k] += stats[k]
            print("--- oasst1 (ru) ---")
            stats = build_oasst(f, seen_hashes, args.dry_run)
            save_stats(args.out_dir, "chat_ru.oasst1", stats)
            for k in total:
                total[k] += stats[k]
        save_stats(args.out_dir, "chat_ru", total)

    print("Done.")


if __name__ == "__main__":
    main()
