import json
from dataclasses import dataclass

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from app.dataset.common import populate_loss_mask, split_train_val, build_batch_getter


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



@dataclass
class PromptData:
    instruction: str
    input: str
    output: str


def format_alpaca_example(item):
    if item["input"].strip():
        return (
            f"### Instruction:\n{item['instruction'].strip()}\n\n"
            f"### Input:\n{item['input'].strip()}\n\n"
            f"### Response:\n{item['output'].strip()}"
        )
    return (
        f"### Instruction:\n{item['instruction'].strip()}\n\n"
        f"### Response:\n{item['output'].strip()}"
    )


def format_alpaca_prompt_and_response(item):
    if "input" in item and item["input"].strip():
        prompt = (
            f"### Instruction:\n{item['instruction'].strip()}\n\n"
            f"### Input:\n{item['input'].strip()}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{item['instruction'].strip()}\n\n"
            f"### Response:\n"
        )

    response = item["output"].strip()
    return prompt, response


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



def load_alpaca_instruction_eval_loader(filepath: str, block_size: int, batch_size: int):
    with open(filepath, "r") as f:
        data = json.load(f)

    enc = tiktoken.get_encoding("gpt2")
    _, val_examples = split_train_val(data)
    collator = InstructionCollator(block_size=block_size, pad_token=enc.eot_token)
    val_loader = DataLoader(
        InstructionDataset(val_examples, block_size, enc),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return val_loader, enc.n_vocab, enc
