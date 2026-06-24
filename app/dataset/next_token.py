import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from app.dataset.common import build_memmap_batch_getter
from app.preprocessing import prepare_model_files


class MemMappedNextTokenDataset(Dataset):
    """
    Now your dataset size is basically limited by disk space instead of RAM.

    Expects the data to be a file containing a sequence of uint16 tokens.
    """

    def __init__(self, path, block_size):
        self.data = np.memmap(
            path,
            dtype=np.uint16,
            mode="r"
        )
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(
            self.data[idx:idx + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1:idx + self.block_size + 1].astype(np.int64)
        )
        return x, y


class SequentialMemMappedNextTokenDataset(Dataset):
    """
    Deterministic next-token evaluation dataset over a memmapped token stream.

    Windows advance by `stride` tokens so evaluation is stable across runs and
    does not sample with replacement.
    """

    def __init__(self, path: str, block_size: int, stride: int | None = None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.stride = stride or block_size

        max_start = len(self.data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(f"Expected more than {block_size + 1} tokens, found {len(self.data)}")

        self.starts = list(range(0, max_start + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        x = torch.from_numpy(
            self.data[start:start + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[start + 1:start + self.block_size + 1].astype(np.int64)
        )
        return x, y


def load_bpe_text_memmapped(model_dir: str, block_size, batch_size, device, model="roneneldan/TinyStories"):
    enc = tiktoken.get_encoding("gpt2")

    train_path, val_path = prepare_model_files(model, model_dir, enc)
    # print(f"Dataset: {len(train_tokens):,} tokens, vocab size: {enc.n_vocab}. Batch size: {batch_size}.")
    print(f"Datasets: {train_path}, {val_path}")
    train_loader = build_memmap_batch_getter(train_path, block_size, batch_size, device)
    val_loader = build_memmap_batch_getter(val_path, block_size, batch_size, device)

    return train_loader, val_loader, enc.n_vocab, enc


def build_memmap_eval_loader(path: str, block_size: int, batch_size: int, stride: int | None = None):
    dataset = SequentialMemMappedNextTokenDataset(path, block_size, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_bpe_text_eval_loader(model_dir: str, block_size: int, batch_size: int,
                              model="roneneldan/TinyStories", stride: int | None = None):
    enc = tiktoken.get_encoding("gpt2")
    _, val_path = prepare_model_files(model, model_dir, enc)
    val_loader = build_memmap_eval_loader(val_path, block_size, batch_size, stride=stride)
    return val_loader, enc.n_vocab, enc
