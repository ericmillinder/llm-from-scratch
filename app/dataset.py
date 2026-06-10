import json
from dataclasses import dataclass

import tiktoken
import torch
from torch import Tensor


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
    if item["input"].strip():
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


def build_token_mappings(text: str):
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return vocab_size, stoi, itos


def build_bpe_batchers(tokens, block_size, batch_size, device):
    def get_batch(split_tokens):
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    n = int(0.9 * len(tokens))
    return lambda: get_batch(tokens[:n]), lambda: get_batch(tokens[n:])


def build_char_batchers(tokens, block_size, batch_size, device):
    def get_batch(split_tokens):
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    return get_train, get_val


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

    get_train, get_val = build_instruction_batchers(data, block_size, batch_size, device, enc)
    return get_train, get_val, enc.n_vocab, enc


def load_bpe_text(filepath, block_size, batch_size, device):
    enc = tiktoken.get_encoding("gpt2")

    with open(filepath, "r") as f:
        text = f.read()

    tokens = torch.tensor(enc.encode(text), dtype=torch.long)
    print(f"Dataset: {len(tokens):,} tokens, vocab size: {enc.n_vocab}")

    get_train, get_val = build_bpe_batchers(tokens, block_size, batch_size, device)
    return get_train, get_val, enc.n_vocab, enc


def build_instruction_batchers(examples, block_size, batch_size, device, enc):
    n = int(0.9 * len(examples))
    train_examples = examples[:n]
    val_examples = examples[n:]

    get_train = lambda: get_batch(train_examples, block_size, batch_size, device, enc)
    get_val = lambda: get_batch(val_examples, block_size, batch_size, device, enc)
    return get_train, get_val


def get_batch(examples, block_size, batch_size, device, enc):
    ix = torch.randint(len(examples), (batch_size,))
    pad_token = enc.eot_token

    x_list = []
    y_list = []
    loss_mask_list = []

    for i in ix.tolist():
        prompt, response = format_alpaca_prompt_and_response(examples[i])
        prompt_tokens = enc.encode(prompt)
        response_tokens = enc.encode(response)

        # consider truncating the prompt or preserving a minimum amount of the response
        if len(prompt_tokens) + len(response_tokens) > block_size:
            print(f"Example is too long. Prompt: {len(prompt_tokens)}\nResponse: {len(response_tokens)}\n{examples[i]}")

        full_tokens = (prompt_tokens + response_tokens + [enc.eot_token])[:block_size + 1]
        x_tokens = full_tokens[:-1]
        y_tokens = full_tokens[1:]
        seq_len = len(x_tokens)

        x = torch.full((block_size,), pad_token, dtype=torch.long)
        y = torch.full((block_size,), pad_token, dtype=torch.long)
        loss_mask = torch.zeros(block_size, dtype=torch.float32)

        x[:seq_len] = torch.tensor(x_tokens, dtype=torch.long)
        y[:seq_len] = torch.tensor(y_tokens, dtype=torch.long)

        populate_loss_mask(full_tokens, loss_mask, prompt_tokens, seq_len)

        x_list.append(x)
        y_list.append(y)
        loss_mask_list.append(loss_mask)

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    loss_mask = torch.stack(loss_mask_list).to(device)
    return x, y, loss_mask


def populate_loss_mask(full_tokens, loss_mask: Tensor, prompt_tokens, seq_len: int):
    response_start = min(len(prompt_tokens), len(full_tokens))
    first_response_target = max(response_start - 1, 0)
    if first_response_target < seq_len:
        loss_mask[first_response_target:seq_len] = 1.0
