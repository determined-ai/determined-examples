from typing import Optional

import pytest
import torch
import torch.nn as nn

import act_mem
import layers

BATCH_SIZES = (1, 2)
D_MODELS = (128, 256)
SEQ_LENS = (64, 128)
N_HEADS = (2, 4)


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


ZERO_MEM_ACT_FNS = [
    nn.ReLU(),
    nn.Sigmoid(),
    nn.Tanh(),
    nn.LeakyReLU(inplace=True),
    nn.Sigmoid(),
]
ALL_ACT_FNS = ZERO_MEM_ACT_FNS + [
    nn.ELU(),
    nn.GELU(),
    nn.Hardshrink(),
    nn.Hardsigmoid(),
    nn.Hardswish(),
    nn.Hardtanh(),
    nn.LeakyReLU(),
    nn.SELU(),
    nn.SiLU(),
]


class TestSavedTensorContext:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("d_model", D_MODELS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_linear(self, device: str, d_model: int, batch_size: int) -> None:
        """
        Test a simple linear layer. The inputs should be saved for backwards
        """
        inputs = torch.randn(batch_size, d_model, requires_grad=True, device=device)
        lin = nn.Linear(d_model, d_model, device=device)
        with act_mem.SavedTensorContext(ignored_tensors=lin.parameters()) as saved:
            _ = lin(inputs)
        assert saved.saved_tensor_mem == inputs.numel() * inputs.element_size()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("d_model", D_MODELS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_linear_amp(self, device: str, d_model: int, batch_size: int) -> None:
        """
        Test a linear layer with AMP. The saved tensors should now be a low-precision version of the
        inputs and the low-precision version of the weights version of the weights
        """
        inputs = torch.randn(batch_size, d_model, requires_grad=True, device=device)
        lin = nn.Linear(d_model, d_model, device=device)
        dtype = torch.bfloat16
        with torch.autocast(device_type=device, dtype=dtype):
            with act_mem.SavedTensorContext(ignored_tensors=lin.parameters()) as saved:
                out = lin(inputs)
        assert (
            saved.saved_tensor_mem
            == out.numel() * out.element_size() + lin.weight.numel() * dtype.itemsize
        )

    @pytest.mark.parametrize("act_fn", ALL_ACT_FNS)
    @pytest.mark.parametrize("dropout_prob", (None, 0.5))
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("d_model", D_MODELS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_mlp(
        self,
        act_fn: nn.Module,
        dropout_prob: Optional[float],
        device: str,
        d_model: int,
        batch_size: int,
        seq_len: int,
    ) -> None:
        """
        For the transformer MLP layer with a ReLU non-linearity, the initial inputs and the inputs
        to the final linear layer (which are four times as large) must always be saved. If the
        derivative of the activation function cannot be expressed in terms of the activation
        function's *outputs*, then the activation inputs must also be saved (which are again four
        times as large as the MLP's inputs). The MLP activation memory can be nearly halved by a
        choice of activation function.
        """
        inputs = torch.randn(
            batch_size, seq_len, d_model, requires_grad=True, device=device
        )
        expansion_factor = 4
        mlp = layers.MLP(
            d_model=d_model, act_fn=act_fn, dropout_prob=dropout_prob, device=device
        )
        with act_mem.SavedTensorContext(ignored_tensors=mlp.parameters()) as saved:
            _ = mlp(inputs)

        # Compare measured memory against expected
        first_lin_input_mem = act_mem.get_tensor_bytes(inputs)
        second_lin_input_mem = expansion_factor * first_lin_input_mem
        # Only some activations require additional activation memory
        activation_input_mem = 0 if act_fn in ZERO_MEM_ACT_FNS else second_lin_input_mem
        dropout_act_mem = (
            0 if not dropout_prob else inputs.numel() * (4 if device == "cpu" else 1)
        )

        expected_mem = (
            first_lin_input_mem
            + second_lin_input_mem
            + activation_input_mem
            + dropout_act_mem
        )
        assert saved.saved_tensor_mem == expected_mem

    @pytest.mark.parametrize("act_fn", ALL_ACT_FNS)
    @pytest.mark.parametrize("dropout_prob", (None, 0.5))
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("d_model", D_MODELS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_mlp_amp(
        self,
        act_fn: nn.Module,
        dropout_prob: Optional[float],
        device: str,
        d_model: int,
        batch_size: int,
        seq_len: int,
    ) -> None:
        """
        Similar story with AMP. The only changes come from the modified dtypes and needing to also
        save references to the low-precision weights in the Linear layers.
        """
        inputs = torch.randn(
            batch_size, seq_len, d_model, requires_grad=True, device=device
        )
        expansion_factor = 4
        mlp = layers.MLP(
            d_model=d_model, act_fn=act_fn, dropout_prob=dropout_prob, device=device
        )
        dtype = torch.bfloat16
        with torch.autocast(device_type=device, dtype=dtype):
            with act_mem.SavedTensorContext(ignored_tensors=mlp.parameters()) as saved:
                _ = mlp(inputs)

        # Compare measured memory against expected
        amp_weight_mem = 2 * expansion_factor * d_model**2 * dtype.itemsize
        first_lin_input_mem = inputs.numel() * dtype.itemsize
        second_lin_input_mem = expansion_factor * inputs.numel() * dtype.itemsize
        # Only some activations require additional activation memory
        activation_input_mem = 0 if act_fn in ZERO_MEM_ACT_FNS else second_lin_input_mem
        dropout_act_mem = (
            0
            if not dropout_prob
            else inputs.numel() * (dtype.itemsize if device == "cpu" else 1)
        )

        expected_mem = (
            amp_weight_mem
            + first_lin_input_mem
            + second_lin_input_mem
            + activation_input_mem
            + dropout_act_mem
        )
        assert (
            saved.saved_tensor_mem == expected_mem
        ), f"Failed on {act_fn=}, {dropout_prob=}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
class TestCUDAMemReadings:
    @pytest.mark.parametrize("d_model", D_MODELS)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    @pytest.mark.parametrize("act_fn", ALL_ACT_FNS)
    def test_mlp(
        self, d_model: int, batch_size: int, seq_len: int, act_fn: nn.Module
    ) -> None:
        """
        Track saved tensors and allocated memory and verify they agree.
        """

        inputs = torch.randn(batch_size, seq_len, d_model, device="cuda")
        mlp = layers.MLP(d_model=d_model, act_fn=act_fn, device="cuda")

        with act_mem.AllocatedMemContext() as mem, act_mem.SavedTensorContext(
            ignored_tensors=mlp.parameters()
        ) as saved:
            outputs = mlp(inputs)

        # AllocatedMemContext captures the outputs, but not inputs, while SavedTensorContext
        # captures inputs and not outputs. Nevertheless, the readings agree because inputs and
        # outputs are tensors of the same size and `dtype`.
        assert mem.delta["current"] == saved.saved_tensor_mem
