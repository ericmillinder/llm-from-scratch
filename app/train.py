import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from gguf import GGUFWriter
from safetensors.torch import save_model
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from dataset import load_alpaca_instruction_json, load_bpe_text_memmapped
from generate import generate
from loss import plot_loss_curve
from model import GPTConfig, GPT

PRETRAIN_PROMPT = "There are five flowers in a field. What colors are they? Blue, "

# dumps decoded training batch tokens to see what is getting trained.
DEBUG_BATCH = False


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
    lr_decay_horizon: int = 20000  # the largest planned training run. Will skew LR of shorter runs but allow longer resumes.
    max_steps: int = 4000
    max_lr: float = 1e-3
    min_lr: float | None = None
    warmup_steps: int = 100
    batch_size: int = 16
    accum_steps: int = 1


def get_device():
    """
    On a MacBook with Apple Silicon, MPS gives roughly 2-3x speedup over CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    return torch.device("cpu")


def build_lr_scheduler(optimizer, lr_decay_steps: int, max_lr: float, min_lr: float | None, warmup_steps: int):
    """
    Build the LR schedule used for both fresh runs and resumed runs.

    Warmup is linear from `max_lr / warmup_steps` to `max_lr`. After warmup,
    cosine decay moves toward `min_lr`. For very short runs, the function falls
    back to a single-stage scheduler rather than constructing an invalid chain.

    :param:lr_decay_steps: A large horizon to base the scheduler's decay on. Separates the scheduler from experimental training runs.

    """
    if lr_decay_steps <= 0:
        raise ValueError("max_steps must be positive")

    min_lr = min_lr if min_lr is not None else max_lr * 0.1
    warmup_steps = max(0, min(warmup_steps, lr_decay_steps))

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=lr_decay_steps, eta_min=min_lr)

    if warmup_steps >= lr_decay_steps:
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

        T_max=lr_decay_steps - warmup_steps,
        eta_min=min_lr,
    )
    hold = ConstantLR(optimizer, factor=1.0, total_iters=0)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, hold, cosine],
        milestones=[warmup_steps, warmup_steps]
    )


def init_model_from_scratch(config,
                            data_path,
                            output_dir,
                            max_steps=4000,
                            batch_size=8,
                            accum_steps=8):
    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, enc = load_bpe_text_memmapped(
        data_path, config.block_size, batch_size, device
    )

    config.vocab_size = vocab_size

    model = GPT(config).to(device)
    print(f"\nModel: {config.n_layer}L/{config.n_head}H/{config.n_embd}D, block_size: {config.block_size}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    training_job = TrainingJob(config,
                               model,
                               optimizer,
                               get_train_batch,
                               get_val_batch,
                               enc,
                               PRETRAIN_PROMPT,
                               max_steps=max_steps,
                               max_lr=1e-3,
                               batch_size=batch_size,
                               accum_steps=accum_steps)

    run_training(output_dir, training_job)


def init_model_from_checkpoint(checkpoint_path,
                               data_path,
                               output_dir,
                               max_steps=4000,
                               batch_size=8,
                               accum_steps=8
                               ):
    """
    For SFT after a pretraining model is completed.
    """
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["tokenizer"] != "gpt2":
        raise ValueError(f"Unsupported tokenizer: {checkpoint['tokenizer']}. Only 'gpt2' is currently supported.")

    config = checkpoint["config"]

    get_train_batch, get_val_batch, vocab_size, enc = load_alpaca_instruction_json(
        data_path, config.block_size, batch_size, device
    )
    if config.vocab_size != vocab_size:
        raise ValueError(f"Checkpoint vocab size ({config.vocab_size}) does not "
                         f"match dataset vocab size ({vocab_size}).")

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    training_job = TrainingJob(config,
                               model,
                               optimizer,
                               get_train_batch,
                               get_val_batch,
                               enc,
                               "List 5 colors?",
                               batches_include_loss_mask=True,
                               max_steps=max_steps,
                               max_lr=1e-4,
                               batch_size=batch_size,
                               accum_steps=accum_steps)

    run_training(output_dir, training_job)


def resume_pretrain_run(path, output_dir, data_path, max_steps=4000):
    """
    Resume training from a checkpoint. Config and training setup comes from the checkpoint payload.

    Only resume numbered checkpoints. The optimizer state is not stored in the final checkpoint as
    it is not necessary for further fine-tuning.

    Also, the max_steps should be > than the current step in the checkpoint.

    When max_steps changes from what is stored in the model, the learning rate schedule will be
    rebuilt based on the new max_steps.

    :param path:
    :param max_steps: total number of optimization steps to perform
    :param output_dir: where the model checkpoints will be saved
    :return:
    """
    device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["tokenizer"] != "gpt2":
        raise ValueError(f"Unsupported tokenizer: {checkpoint['tokenizer']}. Only 'gpt2' is currently supported.")

    config = checkpoint['config']
    if max_steps <= checkpoint['step']:
        raise ValueError(f"max_steps must be greater than the current checkpoint step ({checkpoint['step']})")

    batch_size = checkpoint.get('batch_size', 8)
    accum_steps = checkpoint.get('accum_steps', 8)

    device = get_device()
    print(f"Using device: {device}")

    get_train_batch, get_val_batch, vocab_size, enc = load_bpe_text_memmapped(
        data_path, config.block_size, batch_size, device
    )

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer.load_state_dict(optimizer_state_dict)

    print("Model and optimizer loaded from checkpoint")
    # print(f"Checkpoint state\n\t{checkpoint}")

    scheduler_config = checkpoint.get("scheduler_config", {})
    schedule_state_dict = checkpoint.get("scheduler_state_dict")

    training_job = TrainingJob(config, model, optimizer,
                               get_train_batch, get_val_batch, enc,
                               PRETRAIN_PROMPT,
                               max_steps=max_steps,
                               max_lr=scheduler_config.get("max_lr", optimizer.param_groups[0]["lr"]),
                               min_lr=scheduler_config.get("min_lr"),
                               warmup_steps=scheduler_config.get("warmup_steps", 100),
                               batch_size=batch_size,
                               accum_steps=accum_steps)

    completed_steps = checkpoint['step']

    run_training(
        output_dir,
        training_job,
        completed_steps,
        scheduler_state_dict=schedule_state_dict,
    )


def run_training(output_dir, job: TrainingJob, completed_steps=0, scheduler_state_dict=None):
    """
    The training loop for a given training job.
    Saves the model checkpoints at intervals.
    Saves a final checkpoint at the job.max_steps.

    :param output_dir: (str) The directory to save the model checkpoints.
    :param job: (TrainingJob) The training job configuration.
    :param completed_steps: (int) The number of steps already completed.
    :param scheduler_state_dict: (dict) The state dictionary of the learning rate scheduler.

    Returns: None
    """
    loss_log = {"steps": [], "train": [], "val": []}

    model = job.model
    optimizer = job.optimizer
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=job.max_lr, weight_decay=0.01)

    scheduler = build_lr_scheduler(
        optimizer,
        lr_decay_steps=job.lr_decay_horizon,
        max_lr=job.max_lr,
        min_lr=job.min_lr,
        warmup_steps=job.warmup_steps,
    )
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    if completed_steps > 0:
        tqdm.write(f"Resuming training with {completed_steps} completed steps.")
        if Path(f"{output_dir}/loss_log.json").exists():
            with open(f"{output_dir}/loss_log.json", "r") as f:
                loss_log = json.load(f)
            print("Loaded existing loss log.")
        else:
            print("No existing loss log found.")

    progress_bar = tqdm(range(completed_steps, job.max_steps), desc="Training")
    optimizer.zero_grad(set_to_none=True)
    for _ in progress_bar:
        accum_loss = 0.0  # gradient accumulation loss

        # micro-batching to reduce memory usage
        for _ in range(job.accum_steps):
            batch = job.get_train_batch()
            if job.batches_include_loss_mask:
                x, y, loss_mask = batch
                _, loss = model(x, y, loss_mask=loss_mask)
            else:
                x, y = batch
                _, loss = model(x, y)

                # This is incredibly inefficient and only for understanding what the training batch looks like.
                if DEBUG_BATCH and completed_steps % 100 == 0:
                    for token in x:
                        print(f"\n---train token batch\n{job.tokenizer.decode(token.tolist())}\n---")

            accum_loss += loss.item()
            (loss / job.accum_steps).backward()  # does loss have a divide operand?

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # completed_steps is separated from the progress bar. It tracks the number of optimizer steps completed.
        completed_steps += 1

        lr = optimizer.param_groups[0]["lr"]
        mean_loss = accum_loss / job.accum_steps

        progress_bar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.2e}")

        # --- log loss ---
        loss_log["steps"].append(completed_steps)
        loss_log["train"].append(mean_loss)
        if completed_steps % 200 == 0:
            # --- validation loss ---
            val_loss = validate_training(job, model, completed_steps)
            loss_log["val"].append(val_loss)


        # --- generate sample ---
        if completed_steps % 500 == 0:
            model.eval()
            sample = generate(model, job.sample_prompt, job.tokenizer, max_new_tokens=100, temperature=0.8)
            tqdm.write(f"\n--- Step {completed_steps} sample ---\n{job.sample_prompt}\n{sample}\n---\n")
            model.train()

        # --- save checkpoint ---
        if completed_steps % 1000 == 0:
            save_checkpoint(job, model, optimizer, scheduler, output_dir, completed_steps)

    # --- save final checkpoint (without full checkpoint state) and loss log ---
    torch.save({
        "step": completed_steps,
        "model_state_dict": model.state_dict(),
        "config": job.config,
        "tokenizer": "gpt2"
    }, f"{output_dir}/checkpoint_final.pt")
    # save_file(model.state_dict(), f"{output_dir}/checkpoint_final.safetensors")

    with open(f"{output_dir}/loss_log.json", "w") as f:
        json.dump(loss_log, f)

    plot_loss_curve(output_dir)

    # save_gguf(job, model, optimizer, output_dir, job.max_steps)

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
            "max_steps": job.max_steps,
            "max_lr": job.max_lr,
            "min_lr": job.min_lr,
            "warmup_steps": job.warmup_steps,
        },
        "batch_size": job.batch_size,
        "accum_steps": job.accum_steps,
    }, f"{output_dir}/checkpoint_{step}.pt")


def save_checkpoint_safetensors(job: TrainingJob, model: GPT, optimizer, scheduler, output_dir, step: int):
    """
    Saves the model checkpoint in the safetensors format. The metadata is flat
    """
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

    # save_gguf(job, model, optimizer, output_dir, step)


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


def get_args():
    parser = argparse.ArgumentParser(description="Train a GPT2ish model")
    parser.add_argument("--dataset", default="./data",
                        help="Path to training dataset directory.")
    parser.add_argument("--model_output_dir", default="unnamed", help="Output directory for model checkpoints")
    parser.add_argument("--mode", default="pretrain", help="Mode to run in: pretrain, finetune, resume")
    parser.add_argument("--checkpoint", help="Path to checkpoint to resume from. Required if resuming training.")
    parser.add_argument("--max_steps", help="Number of samples to train on.", type=int, default=5000)
    parser.add_argument("--size",
                        help="Size of the model (tiny, small, medium, large). Affects block size, layers, etc..",
                        type=str, default="tiny")

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = get_args()

    sys.path.append('../')  # lame workaround to get the 'app' seen as a module for unpickle
    print(f"PYTHONPATH = {sys.path}")

    training_data_path = args.dataset
    model_output_dir = args.model_output_dir

    Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "resume":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for resume mode")

        resume_pretrain_run(args.checkpoint,
                            model_output_dir,
                            training_data_path,
                            max_steps=args.max_steps)
    elif args.mode == "finetune":
        init_model_from_checkpoint(args.checkpoint,
                                   training_data_path,
                                   model_output_dir,
                                   max_steps=args.max_steps,
                                   batch_size=8,
                                   accum_steps=8)
    else:

        # train(data_path, max_steps=3000)
        # block_size = context window size and has considerable impact on performance
        # subzero: 2, 2, 64
        # tiny: 4, 4, 128
        # (defaults) small: 6, 6, 384 -- 12M char encoding | 78M BPE
        # medium: 8, 8, 512
        # BPE token encoding has affected the names for those sizes.

        # config's vocab size will be set during the setup

        large = GPTConfig(
            block_size=512,
            n_layer=8,
            n_head=8,
            n_embd=512,
        )
        medium = GPTConfig(
            block_size=384,
            n_layer=6,
            n_head=6,
            n_embd=384,
        )
        small = GPTConfig(
            block_size=128,
            n_layer=4,
            n_head=4,
            n_embd=128,
        )
        tiny = GPTConfig(
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        configs = {"tiny": tiny, "medium": medium, "small": small, "large": large}
        if args.size not in configs:
            raise ValueError(f"Invalid config: {args.size}")

        config = configs[args.size]

        init_model_from_scratch(config,
                                training_data_path,
                                model_output_dir,
                                max_steps=args.max_steps,
                                batch_size=8,
                                accum_steps=8)
