from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def populate_loss_mask(full_tokens, loss_mask: Tensor, prompt_tokens, seq_len: int):
    response_start = min(len(prompt_tokens), len(full_tokens))
    first_response_target = max(response_start - 1, 0)
    if first_response_target < seq_len:
        loss_mask[first_response_target:seq_len] = 1.0


def split_train_val(items: Iterable, train_ratio: float = 0.9):
    """Split a sequence or tensor into train/validation slices by prefix."""
    n = int(train_ratio * len(items))
    return items[:n], items[n:]


def build_batch_getter(loader: DataLoader, device):
    """
    Adapt a DataLoader to the existing `get_batch()` training interface.

    The returned closure cycles through the loader indefinitely and moves every
    tensor in the batch onto the requested device before returning it.
    """
    iterator = iter(loader)

    def get_batch():
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        if isinstance(batch, (tuple, list)):
            return tuple(part.to(device) for part in batch)
        return batch.to(device)

    return get_batch


def build_memmap_batch_getter(path: str, block_size: int, batch_size: int, device):
    """
    Replicates the MemMappedNextTokenDataset. Performs random sampling.

    Beware if you want to use this to consume the entire dataset, as it will sample with replacement.
    """
    data = np.memmap(path, dtype=np.uint16, mode="r")
    max_start = len(data) - block_size - 1

    if max_start <= 0:
        raise ValueError(f"Expected more than {block_size + 1} tokens, found {len(data)}")

    def get_batch():
        starts = torch.randint(0, max_start, (batch_size,)).tolist()

        x = torch.empty((batch_size, block_size), dtype=torch.long)
        y = torch.empty((batch_size, block_size), dtype=torch.long)

        for row, start in enumerate(starts):
            x[row] = torch.from_numpy(
                data[start:start + block_size].astype(np.int64)
            )
            y[row] = torch.from_numpy(
                data[start + 1:start + block_size + 1].astype(np.int64)
            )

        return x.to(device), y.to(device)

    return get_batch
