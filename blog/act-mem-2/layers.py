from typing import Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Minimal multi-head attention layer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_head, remainder = divmod(self.d_model, self.n_heads)
        assert not remainder, f"{n_heads=} must divide {d_model=} evenly"

        self.lin_qkv = nn.Linear(
            self.d_model,
            3 * self.d_model,
            **factory_kwargs,
        )

        self.lin_out = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = inputs.size()

        # Create the queries, keys, values
        qkv = einops.rearrange(
            self.lin_qkv(inputs),
            "b s (three n_h d_h) -> three b s n_h d_h",
            b=bsz,
            s=seq_len,
            three=3,
            n_h=self.n_heads,
            d_h=self.d_head,
        )
        q, k, v = qkv

        bsz, seq_len, n_heads, d_head = q.shape

        shape_kwargs = dict(b=bsz, n_h=n_heads, s=seq_len, d_h=d_head)
        q = einops.rearrange(q, "b s n_h d_h -> b n_h s d_h", **shape_kwargs)
        k = einops.rearrange(k, "b s n_h d_h -> b n_h s d_h", **shape_kwargs)
        v = einops.rearrange(v, "b s n_h d_h -> b n_h s d_h", **shape_kwargs)

        # Multi-head self-attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = einops.rearrange(
            attn_output,
            "b n_h s d_h -> b s (n_h d_h)",
            b=bsz,
            n_h=n_heads,
            s=seq_len,
            d_h=d_head,
        )

        # Final projection
        out = self.lin_out(attn_output)

        return out


class MLP(nn.Module):
    """
    Basic MLP layer with optional Dropout.
    """

    def __init__(
        self,
        d_model: int,
        act_fn: nn.Module,
        dropout_prob: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.act_fn = act_fn
        self.dropout_prob = dropout_prob
        factory_kwargs = {"device": device, "dtype": dtype}

        self.lin_0 = nn.Linear(self.d_model, 4 * self.d_model, **factory_kwargs)
        self.lin_1 = nn.Linear(4 * self.d_model, self.d_model, **factory_kwargs)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.lin_0(inputs)
        x = self.act_fn(x)
        x = self.lin_1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Basic transformer block.

    Schematic:
    ┌──────┐
    │inputs│
    └┬─┬───┘
     │┌▽───────────┐
     ││norm_0, attn│
     │└┬───────────┘
    ┌▽─▽──┐
    │ add │
    └┬─┬──┘
     │┌▽──────────┐
     ││norm_1, mlp│
     │└┬──────────┘
    ┌▽─▽──┐
    │ add │
    └┬────┘
    ┌▽──────┐
    │outputs│
    └───────┘
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        act_fn: nn.Module,
        dropout_prob: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.attn = Attention(d_model=d_model, n_heads=n_heads, **factory_kwargs)
        self.mlp = MLP(d_model=d_model, act_fn=act_fn, dropout_prob=dropout_prob, **factory_kwargs)
        self.norm_0 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm_1 = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.attn(self.norm_0(inputs)) + inputs
        outputs = self.mlp(self.norm_1(outputs)) + outputs
        return outputs
