# Starting Point

Initially based on the documentation and info in https://github.com/angelos-p/llm-from-scratch.

See the docs over there for very detailed info to get started. https://github.com/angelos-p/llm-from-scratch/tree/main/docs

# Expanded Goal

This has an updated goal of being able to handle instructions like "write a sonnet about dogs." Or some kind of instruction
handling that produces logical, coherent, and creative responses. 

# Training sets

Pretraining is being done against the roneneldan/TinyStories dataset. It will be pulled from huggingface.

This started with pretraining on Shakespeare plays. Shakespearean text was generated. It turns out that Shakespearean
responses are hard to reason about. They are also hard to tell if they are coherent. So I moved on to TinyStories. That
dataset is producing seemingly good results.  

Fine-tuning will be done against an alpaca instruction dataset like https://huggingface.co/datasets/yahma/alpaca-cleaned/tree/main.
