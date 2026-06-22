import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from dataset import load_alpaca_instruction_eval_loader, load_bpe_text_eval_loader
from generate import generate
from model import GPT
from train import get_device


DEFAULT_PROMPTS = [
    "Once upon a time",
    "The fox looked at the moon and",
    "Write a short poem about rain.",
    "List 5 colors?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT checkpoint on held-out validation data.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--dataset", required=True, help="Dataset directory for pretraining or JSON file for SFT")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True,
                        help="Evaluation mode matching the checkpoint and dataset")
    parser.add_argument("--pretrain_dataset_name", default="roneneldan/TinyStories",
                        help="Hugging Face dataset name used for pretraining token files")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional cap on validation batches for quicker checks")
    parser.add_argument("--pretrain_stride", type=int, default=None,
                        help="Stride for deterministic pretrain evaluation windows; default is block_size")
    parser.add_argument("--prompts_file", help="Optional text file containing one prompt per line")
    parser.add_argument("--sample_prompts", action="store_true",
                        help="Generate text for a fixed prompt suite after scoring")
    parser.add_argument("--sample_only", action="store_true",
                        help="Skip loss evaluation and only emit prompt generations")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling cutoff")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible generation")
    parser.add_argument("--save_json", help="Optional path to save the full evaluation report as JSON")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer_name = checkpoint.get("tokenizer", "gpt2")
    config = checkpoint["config"]

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoder = tiktoken.get_encoding(tokenizer_name)
    return checkpoint, model, encoder


def build_eval_loader(args, config):
    if args.mode == "pretrain":
        val_loader, vocab_size, _ = load_bpe_text_eval_loader(
            args.dataset,
            config.block_size,
            args.batch_size,
            model=args.pretrain_dataset_name,
            stride=args.pretrain_stride,
        )
    else:
        val_loader, vocab_size, _ = load_alpaca_instruction_eval_loader(
            args.dataset,
            config.block_size,
            args.batch_size,
        )

    if config.vocab_size != vocab_size:
        raise ValueError(f"Checkpoint vocab size ({config.vocab_size}) does not match dataset vocab size ({vocab_size}).")

    return val_loader


def compute_batch_loss(logits, targets, loss_mask=None):
    B, T, V = logits.shape
    per_token_loss = F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        reduction="none",
    ).view(B, T)

    if loss_mask is not None:
        weights = loss_mask.to(per_token_loss.dtype)
    else:
        weights = torch.ones_like(per_token_loss)

    weighted_loss_sum = (per_token_loss * weights).sum().item()
    token_weight_sum = weights.sum().item()
    return weighted_loss_sum, token_weight_sum


@torch.no_grad()
def evaluate_validation(model: GPT, val_loader, device: torch.device, max_batches: int | None = None):
    total_loss_sum = 0.0
    total_token_weight = 0.0
    batches_evaluated = 0

    for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
        if max_batches is not None and batches_evaluated >= max_batches:
            break

        if len(batch) == 3:
            x, y, loss_mask = batch
            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)
            logits, _ = model(x)
            loss_sum, token_weight = compute_batch_loss(logits, y, loss_mask=loss_mask)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss_sum, token_weight = compute_batch_loss(logits, y)

        total_loss_sum += loss_sum
        total_token_weight += token_weight
        batches_evaluated += 1

    if total_token_weight == 0:
        raise ValueError("No validation tokens were evaluated.")

    mean_loss = total_loss_sum / total_token_weight
    perplexity = math.exp(mean_loss)
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "token_count": int(total_token_weight),
        "batches_evaluated": batches_evaluated,
    }


def load_prompts(prompts_file: str | None):
    if prompts_file is None:
        return DEFAULT_PROMPTS

    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)

    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")

    return prompts


@torch.no_grad()
def sample_prompts(model: GPT, enc, prompts: list[str], seed: int, max_new_tokens: int,
                   temperature: float, top_k: int):
    outputs = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(seed + i)
        text = generate(
            model,
            prompt,
            enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outputs.append({
            "prompt": prompt,
            "completion": text,
        })
    return outputs


def main():
    args = parse_args()
    device = get_device()
    checkpoint, model, enc = load_checkpoint(args.checkpoint, device)

    report = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "dataset": args.dataset,
        "device": str(device),
        "step": checkpoint.get("step"),
        "config": {
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
        },
    }

    if not args.sample_only:
        val_loader = build_eval_loader(args, model.config)
        report["validation"] = evaluate_validation(model, val_loader, device, max_batches=args.max_batches)

    if args.sample_prompts or args.sample_only:
        prompts = load_prompts(args.prompts_file)
        report["generations"] = sample_prompts(
            model,
            enc,
            prompts,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(json.dumps(report, indent=2))

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
