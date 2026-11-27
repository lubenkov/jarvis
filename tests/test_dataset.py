import numpy as np
import pytest
import torch

import tokenize_corpus
from jarvis.dataset import TokenStreamDataset, collate_batch, format_stats, load_val_batches
from tokenize_corpus import Router, ShardWriter

TOKENS = np.arange(1, 501, dtype=np.uint16)


def _write_binary(path, values, dtype=np.uint16):
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype(dtype).tofile(str(path))


def _read_shards(folder):
    values = []
    for path in sorted(folder.iterdir()):
        if path.name.endswith(".bin") and not path.name.endswith(".mask.bin"):
            values.extend(np.fromfile(path, dtype=np.uint16).tolist())
    return values


def _read_mask_shards(folder):
    values = []
    for path in sorted(folder.iterdir()):
        if path.name.endswith(".mask.bin"):
            values.extend(np.fromfile(path, dtype=np.uint8).tolist())
    return values


@pytest.fixture
def train_dir(tmp_path):
    train = tmp_path / "train"
    _write_binary(train / "web_ru" / "shard_0000.bin", TOKENS)
    _write_binary(train / "chat_ru" / "shard_0000.bin", TOKENS)
    _write_binary(train / "chat_ru" / "shard_0000.mask.bin", TOKENS % 2, np.uint8)
    return train


def test_token_stream_dataset_reads_uint16_shards(train_dir):
    dataset = TokenStreamDataset(str(train_dir), [("web_ru", 1.0, False)], seq_len=16, seed=3)

    x, y, mask = next(iter(dataset))

    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert mask.dtype == torch.bool
    assert x.shape == y.shape == mask.shape == (16,)
    assert torch.equal(y[:-1], x[1:])
    assert y[-1].item() == x[-1].item() + 1
    assert mask.all()


def test_token_stream_dataset_aligns_masks_with_targets(train_dir):
    dataset = TokenStreamDataset(str(train_dir), [("chat_ru", 1.0, True)], seq_len=16, seed=5)
    iterator = iter(dataset)

    for _ in range(10):
        _, y, mask = next(iterator)
        assert torch.equal(mask, (y % 2 == 1))


def test_token_stream_dataset_is_deterministic(train_dir):
    def sample():
        dataset = TokenStreamDataset(str(train_dir), [("web_ru", 1.0, False)], seq_len=16, seed=11)
        iterator = iter(dataset)
        return [tuple(tensor.tolist() for tensor in next(iterator)) for _ in range(5)]

    assert sample() == sample()


def test_token_stream_dataset_rejects_missing_or_short_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenStreamDataset(str(tmp_path), [("web_ru", 1.0, False)], seq_len=8)

    _write_binary(tmp_path / "web_ru" / "shard_0000.bin", np.arange(1, 5, dtype=np.uint16))
    with pytest.raises(ValueError):
        TokenStreamDataset(str(tmp_path), [("web_ru", 1.0, False)], seq_len=8)


def test_collate_batch_stacks_windows():
    batch = [
        (torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long),
         torch.zeros(4, dtype=torch.bool))
        for _ in range(3)
    ]

    x, y, mask = collate_batch(batch)

    assert x.shape == y.shape == mask.shape == (3, 4)
    assert y.dtype == torch.int64
    assert mask.dtype == torch.bool


def test_load_val_batches_aligns_masks(tmp_path):
    val = tmp_path / "val"
    _write_binary(val / "web_ru" / "shard_0000.bin", TOKENS)
    _write_binary(val / "web_ru" / "shard_0000.mask.bin", TOKENS % 2, np.uint8)

    result = load_val_batches(str(val), ["web_ru"], seq_len=16, batch_size=4, n_batches=3, seed=0)

    x, y, mask = result["web_ru"]
    assert x.shape == y.shape == mask.shape == (3, 16)
    assert torch.equal(mask, (y % 2 == 1))


def test_load_val_batches_defaults_to_ones_without_mask(tmp_path):
    val = tmp_path / "val"
    _write_binary(val / "web_ru" / "shard_0000.bin", TOKENS)

    result = load_val_batches(str(val), ["web_ru"], seq_len=16, batch_size=2, n_batches=2)
    result_unmasked = load_val_batches(str(val), ["web_ru"], seq_len=16, batch_size=2,
                                       n_batches=2, masked_sources=set())

    assert result["web_ru"][2].all()
    assert result_unmasked["web_ru"][2].all()


def test_format_stats_reports_splits(tmp_path):
    _write_binary(tmp_path / "train" / "web_ru" / "shard_0000.bin", TOKENS)
    _write_binary(tmp_path / "val" / "web_ru" / "shard_0000.bin", TOKENS)

    report = format_stats(str(tmp_path))

    assert "[train]" in report
    assert "[val]" in report
    assert report.count("web_ru") == 2


def test_router_splits_offsets_without_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenize_corpus, "SHARD_TOKENS", 100)
    tokens = np.arange(1, 1001)
    train_dir = tmp_path / "train" / "web_ru"
    val_dir = tmp_path / "val" / "web_ru"
    train_writer = ShardWriter(str(train_dir), False)
    val_writer = ShardWriter(str(val_dir), False)
    router = Router(train_writer, val_writer, val_skip=137, val_tokens=421)

    for start in range(0, len(tokens), 97):
        router.add(tokens[start:start + 97].tolist())
    train_writer.close()
    val_writer.close()

    train_ids = _read_shards(train_dir)
    val_ids = _read_shards(val_dir)

    assert val_ids == tokens[137:137 + 421].tolist()
    assert train_ids == tokens[137 + 421:].tolist()
    assert set(train_ids).isdisjoint(val_ids)
    assert len(train_ids) + len(val_ids) + 137 == len(tokens)


def test_router_routes_masks_with_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenize_corpus, "SHARD_TOKENS", 100)
    tokens = np.arange(1, 401)
    mask = (tokens % 2).astype(np.uint8)
    train_writer = ShardWriter(str(tmp_path / "train" / "chat_ru"), True)
    val_writer = ShardWriter(str(tmp_path / "val" / "chat_ru"), True)
    router = Router(train_writer, val_writer, val_skip=50, val_tokens=120)

    router.add(tokens.tolist(), mask.tolist())
    train_writer.close()
    val_writer.close()

    assert _read_mask_shards(tmp_path / "val" / "chat_ru") == mask[50:170].tolist()
    assert _read_mask_shards(tmp_path / "train" / "chat_ru") == mask[170:].tolist()


def test_shard_writer_exact_multiple_has_no_empty_trailing_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenize_corpus, "SHARD_TOKENS", 10)
    writer = ShardWriter(str(tmp_path), False)

    writer.add(list(range(20)))
    writer.close()

    assert sorted(path.name for path in tmp_path.iterdir()) == ["shard_0000.bin", "shard_0001.bin"]
    assert writer.total_tokens == 20
    assert _read_shards(tmp_path) == list(range(20))


def test_shard_writer_flushes_across_partial_adds(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenize_corpus, "SHARD_TOKENS", 5)
    writer = ShardWriter(str(tmp_path), True)

    writer.add([1, 2, 3])
    assert list(tmp_path.iterdir()) == []

    writer.add([4, 5])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shard_0000.bin",
                                                                "shard_0000.mask.bin"]

    writer.add([6], [0])
    writer.close()

    assert _read_shards(tmp_path) == [1, 2, 3, 4, 5, 6]
    assert _read_mask_shards(tmp_path) == [1, 1, 1, 1, 1, 0]


def test_shard_writer_defaults_mask_to_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenize_corpus, "SHARD_TOKENS", 4)
    writer = ShardWriter(str(tmp_path), True)

    writer.add([1, 2, 3, 4, 5, 6])
    writer.close()

    assert _read_mask_shards(tmp_path) == [1, 1, 1, 1, 1, 1]


def test_shard_writer_empty_close_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenize_corpus, "SHARD_TOKENS", 4)
    writer = ShardWriter(str(tmp_path), False)

    writer.add([])
    writer.close()

    assert list(tmp_path.iterdir()) == []
