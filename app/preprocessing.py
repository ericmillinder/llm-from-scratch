from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tiktoken import Encoding


def load_and_tokenize(model_name: str, enc: Encoding):
    training_tokens = tokenize_text_dataset(model_name, "train", enc)
    validation_tokens = tokenize_text_dataset(model_name, "validation", enc)

    return training_tokens, validation_tokens


def tokenize_text_dataset(model_name: str, split: str, enc: Encoding, force=False, debug=False):
    outfile = f"data/{model_name.replace('/', '_')}-{split}.bin"
    # File path
    a = Path(outfile)
    if a.exists() and not force:
        print(f"{outfile} exists. Skipping tokenization. Delete it to retokenize.")
        return np.fromfile(outfile, dtype=np.long)

    print(f"Fetching {model_name} ({split}) from huggingface..")
    ds = load_dataset(model_name, split=split)  # should be cached after first run

    print("Tokenizing...")
    tokens = []
    for row in ds:
        if debug:
            print(f"---\n{row['text']}<|endoftext|>\n---")
        tokens.extend(enc.encode_ordinary(row["text"]))
        tokens.append(enc.eot_token)

    print(f"Writing tokens to {outfile}")
    array = np.array(tokens, dtype=np.long)
    array.tofile(outfile)

    return array


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess a hf dataset")
    parser.add_argument("--force", default=False, help="Force it even if the files are already there")
    parser.add_argument("--dataset", help="The hf dataset name.", type=str, default='roneneldan/TinyStories')
    parser.add_argument("--split", help="The split to load", type=str, default='train')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    tokenize_text_dataset(args.dataset, args.split, tiktoken.get_encoding("gpt2"))
