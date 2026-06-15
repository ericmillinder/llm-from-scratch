import sys

import tiktoken
import torch

from app.dataset import format_alpaca_prompt_and_response
from model import GPT


def generate_greedy(model, idx, max_new_tokens):
    """
    Should be very deterministic and boring
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


@torch.no_grad()
def generate(model, prompt: str, enc, max_new_tokens=200, temperature=0.8, top_k=40):
    """
    There is no distinction between generation from a pretrain or post-trained model yet.

    The pre-train model is just a next-token predictor and does not understand instructions.

    So, until something changes, we will treat an input with some punctuation as an instruction
    that needs alpaca instruction formatting.
    """
    if prompt.endswith(".") or prompt.endswith("?"):
        prompt = format_alpaca_prompt_and_response(prompt)

    device = next(model.parameters()).device
    prompt_tokens = enc.encode(prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        print(f"next_token = {next_token.item()}")
        if next_token.item() == enc.eot_token:
            break

        idx = torch.cat([idx, next_token], dim=1)

    return enc.decode(idx[0, len(prompt_tokens):].tolist())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint")
    # parser.add_argument("model", default="round1", help="Model 'size' you built: tiny, small, etc")
    parser.add_argument("--checkpoint", default="small-gpt2/checkpoint_final.pt",
                        help="Path to checkpoint file (e.g. checkpoint_final.pt)")
    parser.add_argument("--prompt", default="Shall I compare thee to a", help="Starting text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=40, help="Only sample from top-k most likely tokens")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    sys.path.append('../')  # lame workaround to get the 'app' seen as a module for unpickle
    print(f"PYTHONPATH = {sys.path}")

    # unpickling fails because the checkpoints are in a 'size' directory. How can I fix that?
    checkpoint = torch.load(f"{args.checkpoint}", weights_only=False)
    config = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    encoder = tiktoken.get_encoding(tokenizer)

    output = generate(model, args.prompt, encoder,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k)
    print(output)
