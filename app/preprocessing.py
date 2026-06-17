from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tiktoken import Encoding


def load_and_tokenize(model_name: str, enc: Encoding, output_dir="data"):
    training_tokens = tokenize_text_dataset(model_name, "train", enc)
    validation_tokens = tokenize_text_dataset(model_name, "validation", enc)

    return training_tokens, validation_tokens


def tokenize_text_dataset(model_name: str, split: str, enc: Encoding, output_dir: str, force=False, debug=False):
    outfile = build_model_path(model_name, output_dir, split)
    # File path
    a = Path(outfile)
    if a.exists() and not force:
        print(f"{outfile} exists. Skipping tokenization. Delete it to retokenize.")
        return np.fromfile(outfile, dtype=np.uint16)

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
    array = np.array(tokens, dtype=np.uint16)
    array.tofile(outfile)

    return array


def prepare_model_files(model_name: str, output_dir: str, enc: Encoding):
    if not Path(output_dir).exists():
        raise ValueError(f"Output directory {output_dir} does not exist! Choose a different directory.")

    print(f"Looking for {model_name} files in {output_dir}")
    training_path = tokenize_text_dataset_if_missing(model_name, "train", enc, output_dir)
    validation_path = tokenize_text_dataset_if_missing(model_name, "validation", enc, output_dir)

    return training_path, validation_path


def tokenize_text_dataset_if_missing(model_name: str, split: str, enc: Encoding, output_dir: str, force=False,
                                     debug=False):

    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"Output directory '{output_path.absolute()}' does not exist! Choose a different directory.")

    outfile = build_model_path(model_name, output_dir, split)

    if Path(outfile).exists() and not force:
        print(f"{outfile} exists. Skipping tokenization. Delete it to retokenize.")
        return outfile

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
    array = np.array(tokens, dtype=np.uint16)
    array.tofile(outfile)

    return outfile


def build_model_path(model_name: str, output_dir: str, split: str) -> str:
    outfile = f"{output_dir}/{model_name.replace('/', '_')}-{split}.bin"
    return outfile


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess a hf dataset")
    parser.add_argument("--force", default=False, help="Force it even if the files are already there")
    parser.add_argument("--dataset", help="The hf dataset name.", type=str, default='roneneldan/TinyStories')
    parser.add_argument("--split", help="The split to load", type=str, default='train')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    tokenize_text_dataset_if_missing(args.dataset, args.split, tiktoken.get_encoding("gpt2"), "../data")
