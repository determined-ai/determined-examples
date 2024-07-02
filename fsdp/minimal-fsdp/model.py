from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Minimal transformer model code adapted from gpt-fast: https://github.com/pytorch-labs/gpt-fast
"""


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.w0 = nn.Linear(d_model, 4 * d_model, device=device)
        self.relu = nn.ReLU()
        self.w1 = nn.Linear(4 * d_model, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(self.relu(self.w0(x)))


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "n_heads must divide d_model evenly"
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False, device=device)
        self.wo = nn.Linear(d_model, d_model, bias=False, device=device)

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = inputs.shape

        # Get queries, keys, and values
        q, k, v = self.wqkv(inputs).split([self.d_model, self.d_model, self.d_model], dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)
        q, k, v = map(lambda inputs: inputs.transpose(1, 2), (q, k, v))

        # Compute attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        y = self.wo(y)

        return y


class TransformerBlock(nn.Module):
    """
    The transformer blocks.

    Forward pass schematic:

    ┌──────┐
    │inputs│
    └┬─┬───┘
     │┌▽─────────┐
     ││norm, attn│
     │└┬─────────┘
    ┌▽─▽──┐
    │add  │
    └┬─┬──┘
     │┌▽────────┐
     ││norm, ffn│
     │└┬────────┘
    ┌▽─▽──┐
    │add  │
    └┬────┘
    ┌▽──────┐
    │outputs│
    └───────┘
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.attention = Attention(d_model=d_model, n_heads=n_heads, device=device)
        self.feed_forward = FeedForward(d_model=d_model, device=device)
        self.ffn_norm = nn.LayerNorm(d_model, device=device)
        self.attention_norm = nn.LayerNorm(d_model, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        h = inputs + self.attention(self.attention_norm(inputs))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EmbedAndEncode(nn.Module):
    """
    Embedding layer with learned positional encodings.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        # Learned positional encoding and embedding layer:
        self.max_seq_len = max_seq_len
        self.learned_pos_enc = nn.Parameter(torch.zeros(max_seq_len, d_model, device=device))
        self.tok_embeddings = nn.Embedding(vocab_size, d_model, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, seq_len = inputs.shape
        assert seq_len <= self.max_seq_len
        outputs = self.tok_embeddings(inputs) + self.learned_pos_enc[None, :seq_len]
        return outputs


class LMHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_model, device=device)
        self.output = nn.Linear(d_model, vocab_size, bias=False, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.output(self.norm(inputs))
        return logits


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        vocab_size: int,
        n_layers: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # Embed/encode
        self.embed_and_encode = EmbedAndEncode(d_model, vocab_size, max_seq_len, device=device)

        # Transformer blocks
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, device=device) for _ in range(n_layers)
        )

        # Final norm and language model head:
        self.lm_head = LMHead(d_model, vocab_size, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embed_and_encode(inputs)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits
