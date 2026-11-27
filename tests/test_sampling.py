import torch
from torch import nn

from jarvis.generate import generate_ids


class FixedLogitsModel(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.float32))
        self.anchor = nn.Parameter(torch.zeros(1))
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def forward(self, idx, caches=None, use_cache=False):
        row = self.rows[min(self.cursor, self.rows.shape[0] - 1)]
        self.cursor += 1
        logits = row.view(1, 1, -1).expand(idx.shape[0], idx.shape[1], -1).clone()
        if use_cache:
            return logits, torch.zeros(1, 1, 1, 1)
        return logits


def sample(model, prompt, seed, **kwargs):
    torch.manual_seed(seed)
    model.reset()
    options = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    options.update(kwargs)
    out = generate_ids(model, [prompt], **options)
    return out[0, -1].item()


def test_greedy_is_deterministic_and_seed_independent():
    model = FixedLogitsModel([[0.2, 3.0, 1.0, -2.0]])

    results = [sample(model, 0, seed, temperature=0.0) for seed in (0, 1, 2)]

    assert results == [1, 1, 1]


def test_greedy_ignores_top_k_and_top_p():
    model = FixedLogitsModel([[0.2, 3.0, 1.0, -2.0]])

    greedy = sample(model, 0, 0, temperature=0.0)
    with_filters = sample(model, 0, 0, temperature=0.0, top_k=1, top_p=0.1)

    assert greedy == with_filters == 1


def test_top_k_one_always_picks_the_argmax():
    model = FixedLogitsModel([[5.0, 0.0, 0.0, 0.0]])

    results = [sample(model, 1, seed, temperature=1.0, top_k=1) for seed in range(5)]

    assert results == [0, 0, 0, 0, 0]


def test_top_k_restricts_the_sampling_support():
    model = FixedLogitsModel([[2.0, 2.0, -8.0, -8.0]])

    results = [sample(model, 0, seed, temperature=1.0, top_k=2) for seed in range(20)]

    assert set(results) <= {0, 1}


def test_top_p_keeps_only_the_nucleus():
    model = FixedLogitsModel([[10.0, -10.0, -10.0, -10.0]])

    results = [sample(model, 0, seed, temperature=1.0, top_p=0.5) for seed in range(5)]

    assert results == [0, 0, 0, 0, 0]


def test_top_p_can_keep_several_tokens():
    model = FixedLogitsModel([[2.0, 2.0, -8.0, -8.0]])

    results = [sample(model, 0, seed, temperature=1.0, top_p=0.99) for seed in range(20)]

    assert set(results) <= {0, 1}


def test_repetition_penalty_changes_the_greedy_choice():
    model = FixedLogitsModel([[3.0, 2.0, -8.0, -8.0]])

    plain = sample(model, 0, 0)
    penalized = sample(model, 0, 0, repetition_penalty=3.0)

    assert plain == 0
    assert penalized == 1


def test_repetition_penalty_of_one_is_a_no_op():
    model = FixedLogitsModel([[3.0, 2.0, -8.0, -8.0]])

    assert sample(model, 0, 0, repetition_penalty=1.0) == sample(model, 0, 0)


def test_sampling_is_reproducible_with_a_fixed_seed():
    model = FixedLogitsModel([[1.0, 1.0, 1.0, 1.0]])

    torch.manual_seed(7)
    model.reset()
    first = generate_ids(model, [0], max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0,
                         repetition_penalty=1.0)
    torch.manual_seed(7)
    model.reset()
    second = generate_ids(model, [0], max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0,
                          repetition_penalty=1.0)

    assert torch.equal(first, second)


def test_stop_token_truncates_generation():
    model = FixedLogitsModel([[0.0, 5.0, 0.0, 0.0]])

    out = generate_ids(model, [0], max_new_tokens=16, temperature=0.0, stop_token_ids=[1])

    assert out.shape == (1, 2)


def test_callback_sees_every_generated_step():
    model = FixedLogitsModel([[0.0, 5.0, 0.0, 0.0]])
    seen = []

    generate_ids(model, [0], max_new_tokens=4, temperature=0.0, callback=seen.append)

    assert len(seen) == 4
    assert [ids.shape[1] for ids in seen] == [2, 3, 4, 5]
