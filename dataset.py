import glob
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset


class TokenStreamDataset(IterableDataset):
    def __init__(self, data_dir, mix, seq_len, seed=0, rank=0, world_size=1):
        super().__init__()
        self.seq_len = seq_len
        self.sources = []
        for name, weight, use_mask in mix:
            shards = sorted(glob.glob(os.path.join(data_dir, name, "shard_*.bin")))
            shards = [s for s in shards if not s.endswith(".mask.bin")]
            if not shards:
                raise FileNotFoundError(f"No shards for source '{name}' in {data_dir}/{name}")

            usable = [s for s in shards if os.path.getsize(s) // 2 > seq_len + 1]
            if not usable:
                raise ValueError(
                    f"All shards for source '{name}' are shorter than the window ({seq_len + 2} tokens) — "
                    f"check the data in {data_dir}/{name}")
            self.sources.append({"name": name, "use_mask": use_mask, "shards": usable})

        total_weight = sum(w for _, w, _ in mix)
        self.probs = np.array([w / total_weight for _, w, _ in mix], dtype=np.float64)
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng((self.seed * 1_000_003 + self.rank * 1_000_033 + worker_id * 1_000 + 7) % (2 ** 63))

        memmaps = {}
        mask_memmaps = {}
        while True:
            src = self.sources[int(rng.choice(len(self.sources), p=self.probs))]
            shard_path = src["shards"][int(rng.integers(len(src["shards"])))]

            mm = memmaps.get(shard_path)
            if mm is None:
                mm = np.memmap(shard_path, dtype=np.uint16, mode="r")
                memmaps[shard_path] = mm
            if len(mm) <= self.seq_len + 1:
                continue

            start = int(rng.integers(0, len(mm) - self.seq_len - 1))
            x = torch.from_numpy(np.asarray(mm[start:start + self.seq_len], dtype=np.int64))
            y = torch.from_numpy(np.asarray(mm[start + 1:start + 1 + self.seq_len], dtype=np.int64))

            mask = torch.ones(self.seq_len, dtype=torch.bool)
            if src["use_mask"]:
                mask_path = shard_path[:-4] + ".mask.bin"
                if os.path.exists(mask_path):
                    mmm = mask_memmaps.get(mask_path)
                    if mmm is None:
                        mmm = np.memmap(mask_path, dtype=np.uint8, mode="r")
                        mask_memmaps[mask_path] = mmm
                    length = min(self.seq_len, len(mmm) - 1 - start)
                    if length > 0:
                        mask[:length] = torch.from_numpy(
                            np.asarray(mmm[start + 1:start + 1 + length], dtype=np.uint8)).bool()
            yield x, y, mask


def collate_batch(batch):
    xs, ys, masks = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


def load_val_batches(val_dir, source_names, seq_len, batch_size=8, n_batches=8, seed=1234,
                     masked_sources=None):
    rng = np.random.default_rng(seed)
    result = {}
    for name in source_names:
        shards = sorted(glob.glob(os.path.join(val_dir, name, "shard_*.bin")))
        shards = [s for s in shards if not s.endswith(".mask.bin")]
        if not shards:
            continue

        xs, ys, masks = [], [], []
        for _ in range(n_batches):
            path = shards[int(rng.integers(len(shards)))]
            mm = np.memmap(path, dtype=np.uint16, mode="r")
            if len(mm) <= seq_len + 1:
                continue
            start = int(rng.integers(0, len(mm) - seq_len - 1))
            xs.append(torch.from_numpy(np.asarray(mm[start:start + seq_len], dtype=np.int64)))
            ys.append(torch.from_numpy(np.asarray(mm[start + 1:start + 1 + seq_len], dtype=np.int64)))

            mask = np.ones(seq_len, dtype=np.uint8)
            use_mask = masked_sources is None or name in masked_sources
            mask_path = path[:-4] + ".mask.bin"
            if use_mask and os.path.exists(mask_path):
                mmm = np.memmap(mask_path, dtype=np.uint8, mode="r")
                length = min(seq_len, len(mmm) - 1 - start)
                if length > 0:
                    mask[:length] = np.asarray(mmm[start + 1:start + 1 + length], dtype=np.uint8)
            masks.append(torch.from_numpy(mask.astype(bool)))

        if xs:
            result[name] = (torch.stack(xs), torch.stack(ys), torch.stack(masks))
    return result


def format_stats(tokenized_dir):
    lines = []
    for split in ("train", "val"):
        split_dir = os.path.join(tokenized_dir, split)
        if not os.path.isdir(split_dir):
            continue
        lines.append(f"[{split}]")
        for name in sorted(os.listdir(split_dir)):
            tokens = 0
            mask_tokens = 0
            for path in glob.glob(os.path.join(split_dir, name, "shard_*.bin")):
                if path.endswith(".mask.bin"):
                    mask_tokens += os.path.getsize(path)
                else:
                    tokens += os.path.getsize(path) // 2
            mask_info = f", includes mask: {'yes' if mask_tokens else 'no'}" if split == "train" else ""
            lines.append(f"  {name}: {tokens / 1e6:.1f}M tokens{mask_info}")
    return "\n".join(lines)


if __name__ == "__main__":
    from model.config import TOKENIZED_DIR

    print(format_stats(TOKENIZED_DIR))
