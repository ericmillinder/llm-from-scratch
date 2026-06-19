"""
Dataset helpers for the two training modes used in this project.

Pretraining uses a single token stream and learns next-token prediction from
random contiguous windows. Supervised fine-tuning uses Alpaca-style instruction
examples, where prompts and responses are tokenized separately so batches can
mask out prompt tokens from the loss.

The module now uses `Dataset` and `DataLoader` internally, but still returns
closure-based `get_train_batch` / `get_val_batch` functions so the existing
training loop in `train.py` does not need a broader interface change.
"""

import json
from typing import Iterable

import numpy as np
import tiktoken
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from alpaca_formatting import format_alpaca_prompt_and_response, format_alpaca_example
from preprocessing import prepare_model_files


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


class InstructionDataset(Dataset):
    """
    SFT dataset that stores raw instruction examples and tokenizes on access.

    Each item returns prompt and response token lists separately so the collator
    can assemble padded sequences and build a response-only loss mask.
    """

    def __init__(self, examples: list[dict], block_size: int, enc):
        self.examples = examples
        self.block_size = block_size
        self.enc = enc

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, response = format_alpaca_prompt_and_response(self.examples[idx])
        prompt_tokens = self.enc.encode(prompt)
        response_tokens = self.enc.encode(response)
        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "example": self.examples[idx],
        }


class InstructionCollator:
    """
    Converts tokenized instruction examples into fixed-width training tensors.

    The collator concatenates prompt + response + EOT, truncates to the model
    context window, pads to `block_size`, and marks only response tokens in the
    loss mask so prompt tokens do not contribute to the SFT loss.
    """

    def __init__(self, block_size: int, pad_token: int):
        self.block_size = block_size
        self.pad_token = pad_token

    def __call__(self, batch):
        x_list = []
        y_list = []
        loss_mask_list = []

        for item in batch:
            prompt_tokens = item["prompt_tokens"]
            response_tokens = item["response_tokens"]
            full_tokens = (prompt_tokens + response_tokens + [self.pad_token])[:self.block_size + 1]

            total_sequence_len = len(prompt_tokens) + len(response_tokens)
            if len(prompt_tokens) > self.block_size:
                print(
                    "\nPrompt exceeds block size and will truncate before the full response fits. "
                    f"Prompt: {len(prompt_tokens)}\n"
                    f"Response: {len(response_tokens)}\n"
                    f"{item['example']}"
                )
            elif total_sequence_len > self.block_size:
                kept_response_tokens = max(self.block_size - len(prompt_tokens), 0)
                print(
                    "\nResponse is being truncated by the prompt length. "
                    f"Prompt: {len(prompt_tokens)}\n"
                    f"Response: {len(response_tokens)}\n"
                    f"Kept response tokens: {kept_response_tokens}\n"
                    f"{item['example']}"
                )

            x_tokens = full_tokens[:-1]
            y_tokens = full_tokens[1:]
            seq_len = len(x_tokens)

            x = torch.full((self.block_size,), self.pad_token, dtype=torch.long)
            y = torch.full((self.block_size,), self.pad_token, dtype=torch.long)
            loss_mask = torch.zeros(self.block_size, dtype=torch.float32)

            x[:seq_len] = torch.tensor(x_tokens, dtype=torch.long)
            y[:seq_len] = torch.tensor(y_tokens, dtype=torch.long)
            populate_loss_mask(full_tokens, loss_mask, prompt_tokens, seq_len)

            x_list.append(x)
            y_list.append(y)
            loss_mask_list.append(loss_mask)

        return torch.stack(x_list), torch.stack(y_list), torch.stack(loss_mask_list)


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


def load_alpaca_instruction_json(filepath, block_size, batch_size, device):
    """
    Instructions are loaded from a JSON file then formatted into a shape like
    ### Instruction:
    What are the three primary colors?

    ### Response:
    The three primary colors are red, blue, and yellow.

    This is then used to fine-tune a language model.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    formatted_examples = [format_alpaca_example(item) for item in data]
    enc = tiktoken.get_encoding("gpt2")
    total_tokens = sum(len(enc.encode(example)) for example in formatted_examples)

    print(f"Instruction dataset: {len(data):,} examples, "
          f"{total_tokens:,} tokens, vocab size: {enc.n_vocab}")

    train_examples, val_examples = split_train_val(data)
    collator = InstructionCollator(block_size=block_size, pad_token=enc.eot_token)
    train_loader = DataLoader(
        InstructionDataset(train_examples, block_size, enc),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        InstructionDataset(val_examples, block_size, enc),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    return build_batch_getter(train_loader, device), build_batch_getter(val_loader, device), enc.n_vocab, enc


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

def load_bpe_text_memmapped(model_dir: str, block_size, batch_size, device, model="roneneldan/TinyStories"):
    enc = tiktoken.get_encoding("gpt2")

    train_path, val_path = prepare_model_files(model, model_dir, enc)
    # print(f"Dataset: {len(train_tokens):,} tokens, vocab size: {enc.n_vocab}. Batch size: {batch_size}.")
    print(f"Datasets: {train_path}, {val_path}")
    train_loader = build_memmap_batch_getter(train_path, block_size, batch_size, device)
    val_loader = build_memmap_batch_getter(val_path, block_size, batch_size, device)

    return train_loader, val_loader, enc.n_vocab, enc


def populate_loss_mask(full_tokens, loss_mask: Tensor, prompt_tokens, seq_len: int):
    response_start = min(len(prompt_tokens), len(full_tokens))
    first_response_target = max(response_start - 1, 0)
    if first_response_target < seq_len:
        loss_mask[first_response_target:seq_len] = 1.0
