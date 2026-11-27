import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from jarvis.chat_format import (
    ASSISTANT_TOKEN,
    BASE_TOKENS,
    EOS_TOKEN,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN,
    USER_TOKEN,
    build_prompt,
    check_tokenizer,
    conversation_mask,
    encode_conversation,
    format_conversation,
    prepare_messages,
    supports_system,
)

WORDS = [
    "привет", "как", "дела", "джарвис", "расскажи", "шутку", "пожалуйста", "спасибо",
    "сегодня", "погода", "хорошая", "друг", "кофе", "книга", "музыка", "вечер",
]


def _corpus():
    for i in range(300):
        yield " ".join(WORDS[(i + j) % len(WORDS)] for j in range(10)) + f" {i % 97}"


def _train_tokenizer(special_tokens):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=["[UNK]"] + special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_corpus(), trainer=trainer)
    return tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return _train_tokenizer(SPECIAL_TOKENS)


@pytest.fixture(scope="module")
def legacy_tokenizer():
    return _train_tokenizer(BASE_TOKENS)


def _ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def test_check_tokenizer_accepts_both_vocabularies(tokenizer, legacy_tokenizer):
    check_tokenizer(tokenizer, verbose=False)
    check_tokenizer(legacy_tokenizer, verbose=False)
    with pytest.raises(ValueError):
        check_tokenizer(legacy_tokenizer, verbose=False, required=SPECIAL_TOKENS)


def test_assistant_only_loss_mask(tokenizer):
    messages = [
        {"role": "user", "content": "привет как дела"},
        {"role": "assistant", "content": "привет хорошо"},
    ]

    ids, mask = encode_conversation(tokenizer, messages)

    assert len(ids) == len(mask)
    assert ids[0] == tokenizer.token_to_id(USER_TOKEN)
    first_assistant = ids.index(tokenizer.token_to_id(ASSISTANT_TOKEN))
    assert not any(mask[:first_assistant + 1])
    assert all(mask[first_assistant + 1:-1])
    assert ids[-1] == tokenizer.token_to_id(EOS_TOKEN)
    assert mask[-1] == 1


def test_system_prompt_rendered_when_token_is_available(tokenizer):
    messages = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "привет как дела"},
    ]

    ids, mask = encode_conversation(tokenizer, messages, system_prompt="ты джарвис")
    system_id = tokenizer.token_to_id(SYSTEM_TOKEN)
    first_user = ids.index(tokenizer.token_to_id(USER_TOKEN))

    assert supports_system(tokenizer)
    assert ids[0] == system_id
    assert not any(mask[:first_user])
    assert any(mask)


def test_system_prompt_is_ignored_without_system_token(legacy_tokenizer):
    messages = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "привет как дела"},
    ]

    plain_ids, plain_mask = encode_conversation(legacy_tokenizer, messages)
    system_ids, system_mask = encode_conversation(
        legacy_tokenizer, messages, system_prompt="ты джарвис")

    assert not supports_system(legacy_tokenizer)
    assert system_ids == plain_ids
    assert system_mask == plain_mask


def test_multi_turn_masks_only_assistant_turns(tokenizer):
    messages = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "привет как дела"},
        {"role": "user", "content": "расскажи шутку"},
        {"role": "assistant", "content": "шутка про кофе"},
    ]

    ids, mask = encode_conversation(tokenizer, messages)
    assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
    user_id = tokenizer.token_to_id(USER_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    in_assistant = False
    for token_id, keep in zip(ids, mask, strict=True):
        if token_id == assistant_id:
            assert not in_assistant
            in_assistant = True
        elif token_id == user_id:
            assert not in_assistant
        elif token_id == eos_id:
            assert keep == int(in_assistant)
            in_assistant = False
        else:
            assert keep == int(in_assistant)

    assert sum(mask) > 0
    assert mask[0] == 0


def test_prepare_messages_normalizes_roles_and_merges_system():
    messages = [
        {"role": "system", "content": "ты джарвис"},
        {"role": "human", "content": "привет"},
        {"role": "assistant", "content": "здравствуй"},
        {"role": "assistant", "content": "чем помочь"},
        {"role": "user", "content": "   "},
    ]

    merged = prepare_messages(messages)
    kept = prepare_messages(messages, keep_system=True)

    assert merged == [("user", "ты джарвис\n\nпривет"), ("assistant", "здравствуй\nчем помочь")]
    assert kept == [
        ("system", "ты джарвис"),
        ("user", "привет"),
        ("assistant", "здравствуй\nчем помочь"),
    ]


def test_prepare_messages_drops_leading_assistant_and_short_dialogues():
    assert prepare_messages([{"role": "assistant", "content": "привет"}]) == []
    assert prepare_messages([{"role": "user", "content": "привет"}]) == []
    assert prepare_messages([{"role": "user", "content": "привет"}], min_messages=1) == [
        ("user", "привет")
    ]


def test_format_conversation_adds_eos_after_each_assistant_turn(tokenizer):
    messages = [("user", "привет"), ("assistant", "здравствуй"), ("user", "как дела")]
    text = format_conversation(messages)

    assert text.count(ASSISTANT_TOKEN) == 1
    assert text.count(EOS_TOKEN) == 1
    assert text.startswith(USER_TOKEN)
    assert text.endswith(USER_TOKEN + "как дела")


def test_build_prompt_ends_with_assistant_marker(tokenizer):
    messages = [{"role": "user", "content": "привет"}]

    ids = build_prompt(tokenizer, messages)
    expected, _ = encode_conversation(tokenizer, messages, add_eos=True, min_messages=1)

    assert ids[-1] == tokenizer.token_to_id(ASSISTANT_TOKEN)
    assert ids[:-1] == expected
    assert len(ids) > 1


def test_build_prompt_without_messages_is_just_the_marker(tokenizer):
    assert build_prompt(tokenizer, []) == [tokenizer.token_to_id(ASSISTANT_TOKEN)]


def test_build_prompt_includes_system_when_supported(tokenizer):
    ids = build_prompt(tokenizer, [{"role": "user", "content": "привет"}], system_prompt="джарвис")
    assert ids[0] == tokenizer.token_to_id(SYSTEM_TOKEN)


def test_conversation_mask_is_all_zero_for_empty_ids(tokenizer):
    assert conversation_mask([], tokenizer) == []


def test_mask_marks_assistant_continuation_after_user_token(tokenizer):
    user_id = tokenizer.token_to_id(USER_TOKEN)
    assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    ids = [user_id] + _ids(tokenizer, "привет") + [assistant_id] + _ids(tokenizer, "ответ") + [eos_id]

    mask = conversation_mask(ids, tokenizer)

    last_assistant = max(i for i, token_id in enumerate(ids) if token_id == assistant_id)
    assert not any(mask[:last_assistant + 1])
    assert all(mask[last_assistant + 1:])
