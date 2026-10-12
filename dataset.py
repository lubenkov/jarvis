import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from model.config import context_length


class TextDataset(Dataset):
    def __init__(self, tokenizer_path, text_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = self.tokenizer.encode(text).ids
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - context_length

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


if __name__ == "__main__":
    dataset = TextDataset("tokenizer.json", "data/raw/dialogues.txt")
    print("Размер датасета (количество возможных окон):", len(dataset))

    x, y = dataset[0]
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x[:10]:", x[:10])
    print("y[:10]:", y[:10])