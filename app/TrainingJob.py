from dataclasses import dataclass

import torch

from app.model import GPTConfig, GPT


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
