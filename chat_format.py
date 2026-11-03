import os
from typing import Dict, List, Optional, Tuple

EOS_TOKEN = "<|endoftext|>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
PAD_TOKEN = "<|pad|>"
SPECIAL_TOKENS = [EOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, PAD_TOKEN]


ROLE_ALIASES = {
    "assistant": "assistant", "bot": "assistant", "gpt": "assistant", "ai": "assistant",
    "user": "user", "human": "user", "prompter": "user",
    "system": "system",
}


def check_tokenizer(tokenizer, verbose=True, required=None):
    """Verify the special tokens this tokenizer is expected to carry.

    Older (v1) tokenizers have no <|system|>, so benchmark tools can pass a
    narrower ``required`` list instead of rebuilding the checkpoint's tokenizer.
    """
    required = SPECIAL_TOKENS if required is None else required
    missing = [t for t in required if tokenizer.token_to_id(t) is None]
    if missing:
        raise ValueError(
            f"tokenizer.json is missing special tokens {missing}. "
            "Rebuild the tokenizer: python train_tokenizer.py"
        )
    if verbose:
        print(f"Tokenizer: {tokenizer.get_vocab_size()} tokens, special tokens OK")
    return tokenizer


def special_ids(tokenizer) -> Dict[str, int]:
    check_tokenizer(tokenizer, verbose=False)
    return {name: tokenizer.token_to_id(tok) for name, tok in (
        ("eos", EOS_TOKEN), ("system", SYSTEM_TOKEN), ("user", USER_TOKEN),
        ("assistant", ASSISTANT_TOKEN), ("pad", PAD_TOKEN))}


def normalize_role(role) -> Optional[str]:
    return ROLE_ALIASES.get(str(role).strip().lower())


def prepare_messages(messages: List[dict], min_messages: int = 2,
                     keep_system: bool = False) -> List[Tuple[str, str]]:
    """Normalize a raw message list into (role, content) pairs.

    With keep_system=False (default, v1 behaviour) system prompts are merged into
    the first user turn. With keep_system=True they stay a separate first turn and
    are rendered as a <|system|> segment.
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
            continue
        if not out and role == "user" and system_prefix and not keep_system:
            content = "\n\n".join(system_prefix + [content])
            system_prefix = []
        if out and out[-1][0] == role:
            out[-1] = (role, out[-1][1] + "\n" + content)
        else:
            out.append((role, content))

    if len(out) < min_messages:
        return []
    if keep_system and system_prefix:
        return [("system", "\n\n".join(system_prefix))] + out
    return out


def format_conversation(messages: List[Tuple[str, str]], add_eos=True) -> str:
    parts = []
    for role, content in messages:
        if role == "system":
            parts.append(SYSTEM_TOKEN)
            parts.append(content)
            continue
        parts.append(USER_TOKEN if role == "user" else ASSISTANT_TOKEN)
        parts.append(content)
        if role == "assistant" and add_eos:
            parts.append(EOS_TOKEN)
    return "".join(parts)


def conversation_mask(ids: List[int], tokenizer) -> List[int]:
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
                        min_messages: int = 2, system_prompt: str = "") -> Tuple[List[int], List[int]]:
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + list(messages or [])
    prepared = prepare_messages(messages, min_messages=min_messages, keep_system=bool(system_prompt))
    if not prepared:
        return [], []
    text = format_conversation(prepared, add_eos=add_eos)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return ids, conversation_mask(ids, tokenizer)


def build_prompt(tokenizer, messages: List[dict], system_prompt: str = "") -> List[int]:
    ids, _ = encode_conversation(tokenizer, messages, add_eos=True, min_messages=1,
                                 system_prompt=system_prompt)
    ids.append(tokenizer.token_to_id(ASSISTANT_TOKEN))
    return ids


def truncate_messages(tokenizer, messages: List[dict], max_tokens: int) -> List[dict]:
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
    budget = max(1, max_tokens - 3)
    if len(tokenizer.encode(content, add_special_tokens=False).ids) > budget:
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


def resolve_system_prompt(cli_value: str = "") -> str:
    """--system wins, then the JARVIS_SYSTEM_PROMPT env var; empty disables it."""
    return (cli_value or os.environ.get("JARVIS_SYSTEM_PROMPT", "")).strip()


if __name__ == "__main__":
    import os

    from tokenizers import Tokenizer

    from model.config import TOKENIZER_PATH

    if not os.path.exists(TOKENIZER_PATH):
        print(f"Missing {TOKENIZER_PATH} — train the tokenizer first (train_tokenizer.py)")
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
    print("Tokens:", len(ids), "| in loss:", sum(mask))
    print("Text:", decode_ids(tokenizer, ids))
    prompt = build_prompt(tokenizer, demo[:-1])
    print("Generation prompt:", decode_ids(tokenizer, prompt))
