from pathlib import Path

from datasets import load_dataset
import numpy as np
from tiktoken import Encoding


def load_and_tokenize(model_name: str, enc: Encoding):
    ds = load_dataset(model_name, split="train") # should be cached after first run
    training_tokens = tokenize_text_dataset(ds, enc, f"data/{model_name.replace('/','_')}-train.bin")
    ds = load_dataset(model_name, split="validation")
    validation_tokens = tokenize_text_dataset(ds, enc, f"data/{model_name.replace('/','_')}-valid.bin")

    return training_tokens, validation_tokens


def tokenize_text_dataset(ds, enc: Encoding, outfile):
    # File path
    a = Path(outfile)
    if a.exists():
        print(f"{outfile} exists. Skipping tokenization. Delete it to retokenize.")
        return np.fromfile(outfile, dtype=np.long)

    print("Tokenizing...")
    tokens = []

    for row in ds:
        tokens.extend(enc.encode_ordinary(row["text"]))

    array = np.array(tokens, dtype=np.long)
    print(f"Writing tokens to {outfile}")
    array.tofile(outfile)
    return array
