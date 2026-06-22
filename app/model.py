import torch
import torch.nn as nn

from dataclasses import dataclass
import math


@dataclass
class GPTConfig:
    """
    Configuration for the GPT model.

    Attributes:
        vocab_size (int): The size of the vocabulary, default is 65 for character-level models.
        block_size (int): The maximum sequence length (context window), default is 256.
        n_layer (int): The number of transformer blocks, default is 6.
        n_head (int): The number of attention heads, default is 6.
        n_embd (int): The embedding dimension, default is 384.

    A good mental model:
        * increase block_size when you need longer dependencies.
        * increase n_embd when you need more representational capacity.
        * increase n_layer when you need more depth/computation per token.
        * increase n_head mostly to improve how attention is partitioned--but keep per-head dimension reasonable.

    When should they differ?
     n_embd != block_size: almost always. These represent different things. There is no reason for them to match.
     n_layer != n_head: also common. Depth and number of heads solve different problems.
    """
    vocab_size: int = 65  # character-level: 65 unique chars in Shakespeare
    block_size: int = 256  # max sequence length (context window)
    n_layer: int = 6  # number of transformer blocks
    n_head: int = 6  # number of attention heads
    n_embd: int = 384  # embedding dimension


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model architecture. Extends the base neural-net module.

    Implements the GPT model with a configurable number of layers, embedding dimensions, and vocabulary size.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # The transformer ModuleDict ends up in the model checkpoint pickles like
        # transformer.wte.weight
        # transformer.wpe.weight
        # transformer.h.0.ln_1.weight
        # transformer.ln_f.weight
        # lm_head.weight
        # It provides a structured way to manage and access different components of the transformer model.

        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the output projection shares weights with the token embeddings
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, loss_mask=None, output_attentions=False, output_hidden_states=False):
        """
        Forward pass through the transformer model. This method computes the logits for the given input tokens
        and optionally computes the loss if targets are provided.
        """

        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)

        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd) — broadcasting adds position info

        hidden_states = [] if output_hidden_states else None
        attentions = [] if output_attentions else None

        # The type is ModuleList. The ModuleDict could be replaced with a small type safe module
        for block in self.transformer.h:
            if output_attentions:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x)

            if output_hidden_states:
                hidden_states.append(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            per_token_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none'
            ).view(B, T)

            if loss_mask is not None:
                loss_mask = loss_mask.to(per_token_loss.dtype)
                loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
            else:
                loss = per_token_loss.mean()

        outputs = {
            "logits": logits,
            "loss": loss,
        }
        if output_hidden_states:
            outputs["hidden_states"] = hidden_states
        if output_attentions:
            outputs["attentions"] = attentions

        if output_attentions or output_hidden_states:
            return outputs

        return logits, loss


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism for transformer models.

    Query, key, and value projections are performed using a single linear layer.
    Output is projected back to the original embedding dimension.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Q, K, V projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # output projection
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, return_attention=False):
        """
        Q = query projection
        K = key projection
        V = value projection
        """
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape for multi-head: (B, T, C) → (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        attn_weights = None
        if return_attention:
            scale = 1.0 / math.sqrt(head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            attn_weights = torch.softmax(att, dim=-1)
            y = attn_weights @ v
        else:
            # attention with causal mask (each token can only attend to previous tokens)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if return_attention:
            return y, attn_weights
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) module for transformer models.

    Performs feed-forward transformations on the input tensor.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)  # project up: 384 → 1536
        # GELU = Gaussian Error Linear Unit
        x = self.gelu(x)  # non-linearity means applying a smooth, non-linear transformation
        return self.c_proj(x)  # project back down: 1536 → 384


class Block(nn.Module):
    """
    Transformer block with attention and MLP layers.

    Implements a single transformer block with residual connections and layer normalization.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out  # attention with residual connection
            x = x + self.mlp(self.ln_2(x))  # MLP with residual connection
            return x, attn_weights

        x = x + self.attn(self.ln_1(x))  # attention with residual connection
        x = x + self.mlp(self.ln_2(x))  # MLP with residual connection
        return x


class Transformer(nn.Module):
    """
    Replacement for the original transformer module, using ModuleList for blocks.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

# config = GPTConfig()
# model = GPT(config)
# n_params = sum(p.numel() for p in model.parameters())
# print(f"Parameters: {n_params / 1e6:.1f}M")  # ~10.8M
