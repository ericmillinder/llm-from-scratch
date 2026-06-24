"""
Dataset helpers for the two training modes used in this project.

Pretraining uses a single token stream and learns next-token prediction from
random contiguous windows. Supervised fine-tuning uses Alpaca-style instruction
examples, where prompts and responses are tokenized separately so batches can
mask out prompt tokens from the loss.

The module now uses `Dataset` and `DataLoader` internally but still returns
closure-based `get_train_batch` / `get_val_batch` functions, so the existing
training loop in `train.py` does not need a broader interface change.

The actual datasets and loading have been moved into a datasets module.

"""

from app.dataset.instruction import load_alpaca_instruction_json, load_alpaca_instruction_eval_loader
from app.dataset.next_token import load_bpe_text_memmapped, load_bpe_text_eval_loader

__all__ = ["load_alpaca_instruction_json",
           "load_bpe_text_memmapped",
           "load_alpaca_instruction_eval_loader",
           "load_bpe_text_eval_loader"]
