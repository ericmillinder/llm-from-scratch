# Starting Point

Initially based on the documentation and info in https://github.com/angelos-p/llm-from-scratch.

See the docs over there for very detailed info to get started. https://github.com/angelos-p/llm-from-scratch/tree/main/docs

# Expanded Goal

This has an updated goal of being able to handle instructions like "write a sonnet about dogs".

# Training sets

Pretraining is being done against a set of [Shakespeare plays](https://github.com/angelos-p/llm-from-scratch/blob/main/data/shakespeare.txt). This looks like [this](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) may be another source.

Fine-tuning will be done against an alpaca instruction dataset like https://huggingface.co/datasets/yahma/alpaca-cleaned/tree/main.
