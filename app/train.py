import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file, save_model
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from dataset import load_alpaca_instruction_json, load_bpe_text
from generate import generate
from loss import plot_loss_curve
from model import GPTConfig, GPT
from gguf import GGUFWriter


@dataclass
class TrainingJob:
    config: GPTConfig
    model: GPT
    optimizer: torch.optim.Optimizer
    get_train_batch: callable
    get_val_batch: callable
    tokenizer: object
    sample_prompt: str
    checkpoint_meta: dict = None
    batches_include_loss_mask: bool = False
    max_steps: int = 4000
    max_lr: float = 1e-3
    min_lr: float | None = None
    warmup_steps: int = 100


def get_device():
    """
    On a MacBook with Apple Silicon, MPS gives roughly 2-3x speedup over CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    return torch.device("cpu")


def build_lr_scheduler(optimizer, max_steps: int, max_lr: float, min_lr: float | None, warmup_steps: int):
    """
    Build the LR schedule used for both fresh runs and resumed runs.

    Warmup is linear from `max_lr / warmup_steps` to `max_lr`. After warmup,
    cosine decay moves toward `min_lr`. For very short runs, the function falls
    back to a single-stage scheduler rather than constructing an invalid chain.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    min_lr = min_lr if min_lr is not None else max_lr * 0.1
    warmup_steps = max(0, min(warmup_steps, max_steps))

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_lr)

    if warmup_steps >= max_steps:
        return LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=min_lr,
    )
    hold = ConstantLR(optimizer, factor=1.0, total_iters=0)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, hold, cosine],
        milestones=[warmup_steps, warmup_steps],
    )


def init_model_from_scratch(config,
                            data_path,
                            output_dir,
                            max_steps=4000,
                            batch_size=64):
    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, enc = load_bpe_text(
        data_path, config.block_size, batch_size, device
    )

    prompt = "To be or not"
    config.vocab_size = vocab_size

    model = GPT(config).to(device)
    print(f"Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}D, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    training_job = TrainingJob(config,
                               model,
                               optimizer,
                               get_train_batch,
                               get_val_batch,
                               enc,
                               prompt,
                               max_steps=max_steps,
                               max_lr=1e-3)

    run_training(output_dir, training_job)


def init_model_from_checkpoint(checkpoint_path,
                               data_path,
                               output_dir,
                               max_steps=4000,
                               batch_size=64
                               ):
    """
    For SFT after a pretraining model is completed.
    """
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint["tokenizer"] != "gpt2":
        raise ValueError(f"Unsupported tokenizer: {checkpoint['tokenizer']}. Only 'gpt2' is currently supported.")

    config = checkpoint["config"]

    get_train_batch, get_val_batch, vocab_size, enc = load_alpaca_instruction_json(
        data_path, config.block_size, batch_size, device
    )
    if config.vocab_size != vocab_size:
        raise ValueError(f"Checkpoint vocab size ({config.vocab_size}) does not "
                         f"match dataset vocab size ({vocab_size}).")

    prompt = "Which are the primary colors?"

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    training_job = TrainingJob(config, model, optimizer,
                               get_train_batch, get_val_batch, enc, prompt, batches_include_loss_mask=True,
                               max_steps=max_steps,
                               max_lr=1e-4)

    run_training(output_dir, training_job)


def resume_pretrain_run(path, output_dir, data_path, batch_size, max_steps=4000):
    """
    Resume training from a checkpoint. Config and training setup comes from
    the checkpoint payload.

    Only resume numbered checkpoints. The optimizer state is not stored in the final checkpoint as
    it is not necessary for further fine-tuning.

    Also, the max_steps should be > than the current step in the checkpoint.

    :param path:
    :param data_path: path to the training data because I dont want to store it in the checkpoint
    :param batch_size: batch size to use for training
    :return:
    """
    device = get_device()
    checkpoint = torch.load(path, map_location=device)
    if checkpoint["tokenizer"] != "gpt2":
        raise ValueError(f"Unsupported tokenizer: {checkpoint['tokenizer']}. Only 'gpt2' is currently supported.")

    config = checkpoint['config']
    if max_steps <= checkpoint['step']:
        raise ValueError(f"max_steps must be greater than the current checkpoint step ({checkpoint['step']})")

    # training_type = checkpoint['training_type']

    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, enc = load_bpe_text(
        data_path, config.block_size, batch_size, device
    )

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer.load_state_dict(optimizer_state_dict)

    scheduler_config = checkpoint.get("scheduler_config", {})

    training_job = TrainingJob(config, model, optimizer,
                               get_train_batch, get_val_batch, enc,
                               "To be or not",
                               max_steps=max_steps,
                               max_lr=scheduler_config.get("max_lr", optimizer.param_groups[0]["lr"]),
                               min_lr=scheduler_config.get("min_lr"),
                               warmup_steps=scheduler_config.get("warmup_steps", 100))

    checkpoint_step = checkpoint['step']
    starting_step = checkpoint_step + 1

    run_training(
        output_dir,
        training_job,
        starting_step,
        scheduler_state_dict=checkpoint.get("scheduler_state_dict"),
    )


def run_training(output_dir, job: TrainingJob, starting_step=0, scheduler_state_dict=None):
    loss_log = {"steps": [], "train": [], "val": []}

    model = job.model
    optimizer = job.optimizer

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=job.max_lr, weight_decay=0.01)

    scheduler = build_lr_scheduler(
        optimizer,
        max_steps=job.max_steps,
        max_lr=job.max_lr,
        min_lr=job.min_lr,
        warmup_steps=job.warmup_steps,
    )
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    if starting_step > 0:
        tqdm.write(f"Resuming training from step {starting_step}")

    progress_bar = tqdm(range(starting_step, job.max_steps), desc="Training")
    for step in progress_bar:
        batch = job.get_train_batch()
        if job.batches_include_loss_mask:
            x, y, loss_mask = batch
            _, loss = model(x, y, loss_mask=loss_mask)
        else:
            x, y = batch
            _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # --- log loss ---
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            # --- validation loss ---
            val_loss = validate_training(job, model, step)
            loss_log["val"].append(val_loss)

        # --- generate sample ---
        if step > 0 and step % 100 == 0:
            model.eval()
            sample = generate(model, job.sample_prompt, job.tokenizer, max_new_tokens=100, temperature=0.8)
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        # --- save checkpoint ---
        if step % 1000 == 0:  # and step > 0:
            save_checkpoint(job, model, optimizer, scheduler, output_dir, step)

    # --- save final checkpoint and loss log ---
    torch.save({
        "step": job.max_steps,
        "model_state_dict": model.state_dict(),
        "config": job.config,
        "tokenizer": "gpt2"
    }, f"{output_dir}/checkpoint_final.pt")
    save_file(model.state_dict(), f"{output_dir}/checkpoint_final.safetensors")

    with open(f"{output_dir}/loss_log.json", "w") as f:
        json.dump(loss_log, f)

    plot_loss_curve(output_dir)

    return model


def save_checkpoint(job: TrainingJob, model: GPT, optimizer, scheduler, output_dir, step: int):
    torch.save({
        "step": step,
        "config": job.config,
        "tokenizer": "gpt2",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler_config": {
            "max_lr": job.max_lr,
            "min_lr": job.min_lr,
            "warmup_steps": job.warmup_steps,
        },
    }, f"{output_dir}/checkpoint_{step}.pt")

    metadata = {
        "step": str(step),
        "tokenizer": "gpt2",
        "vocab_size": str(job.config.vocab_size),
        "block_size": str(job.config.block_size),
        "n_layer": str(job.config.n_layer),
        "n_head": str(job.config.n_head),
        "n_embd": str(job.config.n_embd),
    }
    save_model(model, f"{output_dir}/checkpoint_{step}.safetensors", metadata=metadata)

    save_gguf(job, model, optimizer, output_dir, step)


def save_gguf(job: TrainingJob, model: GPT, optimizer, output_dir, step: int):
    # 1. Define your PyTorch model or state dict
    # (e.g., model = YourTorchModel() or torch.load("model.pth"))
    state_dict = model.state_dict()

    # 2. Initialize the GGUF writer
    outputPath = f"{output_dir}/checkpoint_{step}.gguf"
    writer = GGUFWriter(outputPath, arch="llama")

    # 3. Add metadata descriptive fields
    writer.add_name("A PyTorch Model")
    writer.add_author("A. I. Developer")

    # 4. Loop through weights, convert to NumPy, and write
    for tensor_name, tensor in state_dict.items():
        # GGUF requires data as numpy arrays
        numpy_array = tensor.detach().cpu().numpy()

        # Write tensor to file
        writer.add_tensor(tensor_name, numpy_array)

    # 5. Build and close the file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()

    print(f"Successfully saved to {outputPath}")


def validate_training(job: TrainingJob, model, step: int) -> float:
    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(20):
            batch = job.get_val_batch()
            if job.batches_include_loss_mask:
                x, y, loss_mask = batch
                _, loss = model(x, y, loss_mask=loss_mask)
            else:
                x, y = batch
                _, loss = model(x, y)

            val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f}")
    model.train()
    return val_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--dataset", default="../data/shakespeare.txt", help="Path to training dataset (file for now)")
    parser.add_argument("--model_output_dir", default="unnamed", help="Output directory for model checkpoints")
    parser.add_argument("--mode", default="pretrain", help="Mode to run in: pretrain, finetune, resume")
    parser.add_argument("--checkpoint", help="Path to checkpoint to resume from. Required if resuming training.")
    parser.add_argument("--max-samples", help="Number of samples to train on.")

    args = parser.parse_args()

    sys.path.append('../')  # lame workaround to get the 'app' seen as a module for unpickle
    print(f"PYTHONPATH = {sys.path}")

    training_data_path = args.dataset
    model_output_dir = args.model_output_dir

    Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "resume":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for resume mode")

        resume_pretrain_run(args.checkpoint, model_output_dir, training_data_path, 64, max_steps=args.max_steps)
    elif args.mode == "finetune":
        init_model_from_checkpoint(args.checkpoint, training_data_path, model_output_dir, 4000, 64)
    else:

        # train(data_path, max_steps=3000)
        # block_size = context window size and has considerable impact on performance
        # subzero: 2, 2, 64
        # tiny: 4, 4, 128
        # (defaults) small: 6, 6, 384 -- 12M char encoding | 78M BPE
        # medium: 8, 8, 512
        # BPE token encoding has affected the names for those sizes.

        # config's vocab size will be set during the setup
        config = GPTConfig(
            block_size=384,
            n_layer=6,
            n_head=6,
            n_embd=384,
        )

        init_model_from_scratch(config, training_data_path, model_output_dir)
