from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn


class MLP(nn.Module):
    """
    Basic MLP (multi-layer perceptron) layer. Dropout is neglected.
    """

    def __init__(
        self,
        d_model: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.lin_0 = nn.Linear(d_model, 4 * d_model, device=device, dtype=dtype)
        self.act_fn = nn.GELU()
        self.lin_1 = nn.Linear(4 * d_model, d_model, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.lin_0(inputs)
        x = self.act_fn(x)
        x = self.lin_1(x)
        return x


class AllReduceFwdIdentityBwd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, group: Optional[dist.ProcessGroup] = None
    ) -> torch.Tensor:
        inputs = inputs.clone()
        dist.all_reduce(inputs, group=group)
        return inputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_outputs, None


class IdentityFwdAllReduceBwd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, group: Optional[dist.ProcessGroup] = None
    ) -> torch.Tensor:
        ctx.group = group
        return inputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_outputs = grad_outputs.clone()
        dist.all_reduce(grad_outputs, group=ctx.group)
        return grad_outputs, None


class LinearShardedOutputs(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: dist.ProcessGroup,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        sharded_out_features, remainder = divmod(out_features, group.size())
        assert not remainder, "out_features must be divisible by the ProcessGroup size"
        super().__init__(
            in_features=in_features,
            out_features=sharded_out_features,
            device=device,
            dtype=dtype,
        )

        self.group = group

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Wrap the unsharded inputs for backwards-pass correctness.
        x = IdentityFwdAllReduceBwd.apply(inputs, self.group)
        x = super().forward(x)
        return x


class LinearShardedInputs(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: dist.ProcessGroup,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        sharded_in_features, remainder = divmod(in_features, group.size())
        assert not remainder, "in_features must be divisible by the ProcessGroup size"
        super().__init__(
            in_features=sharded_in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
        )
        self.group = group

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs @ self.weight.T
        # Wrap the mat-mul in an all-reduce forwards-pass correctness.
        x = AllReduceFwdIdentityBwd.apply(x, self.group)
        # Crucial: add the bias _after_ the all-reduce.
        x = x + self.bias
        return x


class MLPTP(MLP):
    """
    Basic Tensor Parallel MLP (multi-layer perceptron) layer. Dropout is neglected.
    """

    def __init__(
        self,
        d_model: int,
        group: Optional[dist.ProcessGroup] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        nn.Module.__init__(self)
        # Fallback to the WORLD process group, if None provided
        group = group or dist.group.WORLD

        self.lin_0 = LinearShardedOutputs(
            d_model, 4 * d_model, group=group, device=device, dtype=dtype
        )
        self.act_fn = nn.GELU()
        self.lin_1 = LinearShardedInputs(
            4 * d_model, d_model, group=group, device=device, dtype=dtype
        )
