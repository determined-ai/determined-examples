"""
Testing the correctness of the TP implementation. Uses CPU, the gloo backend, and multi-processing,
so that the code can run anywhere.

"""

import os
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.distributed as dist

import layers

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29500
WORLD_SIZE = 4
BATCH_SIZE = 2
SEQ_LEN = 64
D_MODEL = 128

# Environment variables expected by torch.distributed.
os.environ["MASTER_ADDR"] = MASTER_ADDR
os.environ["MASTER_PORT"] = str(MASTER_PORT)
os.environ["WORLD_SIZE"] = str(WORLD_SIZE)


def test_mlp(rank: int):
    # More torch.distributed env vars.
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)

    assert (
        not D_MODEL % WORLD_SIZE
    ), f"Choose D_MODEL to be divisible by WORLD_SIZE {D_MODEL % WORLD_SIZE=}."

    # Create two sets of equivalent inputs, both requiring gradients.
    torch.manual_seed(42)
    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)
    inputs_tp = inputs.detach().clone().requires_grad_()

    # Create TP and non-TP MLP layers
    dist.init_process_group(backend="gloo")
    mlp = layers.MLP(D_MODEL)
    mlp_tp = layers.MLPTP(D_MODEL)

    # Give the TP model the same weights as the non-TP model:
    with torch.no_grad():
        mlp_tp.lin_0.weight.data = mlp.lin_0.weight.data.tensor_split(WORLD_SIZE, dim=0)[rank]
        mlp_tp.lin_0.bias.data = mlp.lin_0.bias.data.tensor_split(WORLD_SIZE, dim=0)[rank]
        mlp_tp.lin_1.weight.data = mlp.lin_1.weight.data.tensor_split(WORLD_SIZE, dim=1)[rank]
        mlp_tp.lin_1.bias.data = mlp.lin_1.bias.data

    # The outputs should be the same:
    outputs = mlp(inputs)
    outputs_tp = mlp_tp(inputs_tp)
    with torch.no_grad():
        torch.testing.assert_close(outputs, outputs_tp)

    # Perform a backwards pass on a simple loss function.
    outputs.pow(2).sum().backward()
    outputs_tp.pow(2).sum().backward()

    # Check that the input gradients are the same
    with torch.no_grad():
        assert inputs.grad is not None
        torch.testing.assert_close(inputs.grad, inputs_tp.grad)

    # And finally check that the parameter gradients are the same:
    # Give the TP model the same weights as the non-TP model:
    with torch.no_grad():
        mlp_tp.lin_0.weight.grad.data = mlp.lin_0.weight.grad.data.tensor_split(WORLD_SIZE, dim=0)[
            rank
        ]
        mlp_tp.lin_0.bias.grad.data = mlp.lin_0.bias.grad.data.tensor_split(WORLD_SIZE, dim=0)[rank]
        mlp_tp.lin_1.weight.grad.data = mlp.lin_1.weight.grad.data.tensor_split(WORLD_SIZE, dim=1)[
            rank
        ]
        mlp_tp.lin_1.bias.grad.data = mlp.lin_1.bias.grad.data

    return f"Correct results on {rank=}"


def run():
    with ProcessPoolExecutor(max_workers=WORLD_SIZE) as pool:
        ranks_list = [r for r in range(WORLD_SIZE)]
        results = pool.map(test_mlp, ranks_list)
        for r in results:
            print(r)


if __name__ == "__main__":
    run()
