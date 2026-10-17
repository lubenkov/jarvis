"""Единый формат диалога: спецтокены ролей и кодирование разговора в токены.

Формат (используется и при обучении, и при генерации):

    <|user|>привет<|assistant|>привет! как дела?<|endoftext|><|user|>...

После каждой реплики ассистента ставится <|endoftext|> — модель учится
заканчивать ответ. Реплики пользователя в лосс не попадают (это задаёт маска).
"""

from typing import Dict, List, Optional, Tuple

EOS_TOKEN = "<|endoftext|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
PAD_TOKEN = "<|pad|>"
SPECIAL_TOKENS = [EOS_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, PAD_TOKEN]

# разные датасеты называют роли по-разному
ROLE_ALIASES = {
    "assistant": "assistant", "bot": "assistant", "gpt": "assistant", "ai": "assistant",
    "user": "user", "human": "user", "prompter": "user",
    "system": "system",
}


def check_tokenizer(tokenizer, verbose=True):
    """Проверяет, что токенизатор содержит все спецтокены формата."""
    missing = [t for t in SPECIAL_TOKENS if tokenizer.token_to_id(t) is None]
    if missing:
        raise ValueError(
            f"В tokenizer.json нет спецтокенов {missing}. "
            "Переобучи токенизатор: python train_tokenizer.py"
        )
    if verbose:
        print(f"Токенизатор: {tokenizer.get_vocab_size()} токенов, спецтокены на месте")
    return tokenizer


def special_ids(tokenizer) -> Dict[str, int]:
    check_tokenizer(tokenizer, verbose=False)
    return {name: tokenizer.token_to_id(tok) for name, tok in (
        ("eos", EOS_TOKEN), ("user", USER_TOKEN), ("assistant", ASSISTANT_TOKEN), ("pad", PAD_TOKEN))}


def normalize_role(role) -> Optional[str]:
    return ROLE_ALIASES.get(str(role).strip().lower())


def prepare_messages(messages: List[dict], min_messages: int = 2) -> List[Tuple[str, str]]:
    """Приводит список сообщений к чистому виду: (role, content).

    - системные указания приклеиваются к первой реплике пользователя;
    - подряд идущие реплики одной роли склеиваются;
    - пустые реплики и реплики до первого сообщения пользователя отбрасываются;
    - если после чистки осталось меньше min_messages реплик, возвращается [].
      Для датасетов это min_messages=2 (диалог из одной реплики бесполезен для обучения),
      а для промптов генерации — min_messages=1 (одиночный вопрос — норма).
    """
    out: List[Tuple[str, str]] = []
    system_prefix: List[str] = []

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = normalize_role(m.get("role", ""))
        content = m.get("content") or m.get("value") or m.get("text") or ""
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role == "system":
            system_prefix.append(content)
            continue
        if role not in ("user", "assistant"):
            continue
        if not out and role == "assistant":
            continue  # диалог не может начинаться с ответа
        if not out and role == "user" and system_prefix:
            content = "\n\n".join(system_prefix + [content])
            system_prefix = []
        if out and out[-1][0] == role:
            out[-1] = (role, out[-1][1] + "\n" + content)
        else:
            out.append((role, content))

    if len(out) < min_messages:
        return []
    return out


def format_conversation(messages: List[Tuple[str, str]], add_eos=True) -> str:
    parts = []
    for role, content in messages:
        parts.append(USER_TOKEN if role == "user" else ASSISTANT_TOKEN)
        parts.append(content)
        if role == "assistant" and add_eos:
            parts.append(EOS_TOKEN)
    return "".join(parts)


def conversation_mask(ids: List[int], tokenizer) -> List[int]:
    """1 для токенов ответа ассистента (и закрывающего его EOS), иначе 0."""
    ids_map = special_ids(tokenizer)
    all_special = set(ids_map.values())
    mask = [0] * len(ids)
    in_assistant = False
    for pos, t in enumerate(ids):
        if t == ids_map["user"]:
            in_assistant = False
        elif t == ids_map["assistant"]:
            in_assistant = True
        elif t == ids_map["eos"]:
            if in_assistant:
                mask[pos] = 1
            in_assistant = False
        elif in_assistant and t not in all_special:
            mask[pos] = 1
    return mask


def encode_conversation(tokenizer, messages: List[dict], add_eos=True,
                        min_messages: int = 2) -> Tuple[List[int], List[int]]:
    """messages: [{"role": ..., "content": ...}] -> (ids, loss_mask)."""
    prepared = prepare_messages(messages, min_messages=min_messages)
    if not prepared:
        return [], []
    text = format_conversation(prepared, add_eos=add_eos)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return ids, conversation_mask(ids, tokenizer)


def build_prompt(tokenizer, messages: List[dict]) -> List[int]:
    """История диалога -> ids, заканчивающиеся маркером <|assistant|> для генерации.

    min_messages=1: одиночная реплика пользователя — допустимый промпт
    (по умолчанию prepare_messages отбрасывает всё короче двух реплик,
    из-за чего текст вопроса терялся).
    """
    ids, _ = encode_conversation(tokenizer, messages, add_eos=True, min_messages=1)
    ids.append(tokenizer.token_to_id(ASSISTANT_TOKEN))
    return ids


def truncate_messages(tokenizer, messages: List[dict], max_tokens: int) -> List[dict]:
    """Выкидывает самые старые реплики, пока промпт не влезет в max_tokens.

    Если и одиночная реплика не влезает — урезает её начало по токенам,
    иначе промпт может пробить context_length и буфер RoPE.
    """
    messages = list(messages)
    while len(messages) > 1:
        if len(build_prompt(tokenizer, messages)) <= max_tokens:
            return messages
        messages.pop(0)
    if not messages or len(build_prompt(tokenizer, messages)) <= max_tokens:
        return messages

    first = messages[0]
    content = first.get("content") if isinstance(first, dict) else None
    if not isinstance(content, str):
        return messages
    budget = max(1, max_tokens - 3)  # <|user|>, <|assistant|> и возможный <|endoftext|>
    if len(tokenizer.encode(content, add_special_tokens=False).ids) > budget:
        # бинарный поиск по числу отбрасываемых символов
        lo, hi = 0, len(content)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tokenizer.encode(content[mid:], add_special_tokens=False).ids) <= budget:
                hi = mid
            else:
                lo = mid + 1
        messages[0] = {**first, "content": content[lo:]}
    return messages


def decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def stop_ids(tokenizer) -> List[int]:
    ids_map = special_ids(tokenizer)
    return [ids_map["eos"], ids_map["user"]]


if __name__ == "__main__":
    import os

    from tokenizers import Tokenizer

    from model.config import TOKENIZER_PATH

    if not os.path.exists(TOKENIZER_PATH):
        print(f"Нет {TOKENIZER_PATH} — сначала обучи токенизатор (train_tokenizer.py)")
        raise SystemExit(0)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    check_tokenizer(tokenizer)

    demo = [
        {"role": "system", "content": "Ты дружелюбный собеседник."},
        {"role": "user", "content": "Привет! Как дела?"},
        {"role": "assistant", "content": "Привет! Всё отлично, а у тебя как?"},
        {"role": "user", "content": "Тоже хорошо. Расскажи шутку."},
        {"role": "assistant", "content": "Колобок повесился. — Шутка."},
    ]
    ids, mask = encode_conversation(tokenizer, demo)
    print("Токенов:", len(ids), "| в лоссе:", sum(mask))
    print("Текст:", decode_ids(tokenizer, ids))
    prompt = build_prompt(tokenizer, demo[:-1])
    print("Промпт для генерации:", decode_ids(tokenizer, prompt))
