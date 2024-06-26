"""
A sharded dot-product computed over multiple processes. Uses CPU, the gloo backend, and
multi-processing, so that the code can run anywhere.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.distributed as dist

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29500
WORLD_SIZE = 4
D_MODEL = 128

# Environment variables expected by torch.distributed.
os.environ["MASTER_ADDR"] = MASTER_ADDR
os.environ["MASTER_PORT"] = str(MASTER_PORT)
os.environ["WORLD_SIZE"] = str(WORLD_SIZE)


def compute_dot_product(rank: int):
    # More torch.distributed env vars.
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)

    assert (
        not D_MODEL % WORLD_SIZE
    ), f"Choose D_MODEL to be divisible by WORLD_SIZE {D_MODEL % WORLD_SIZE=}."

    # Setup: populate the same tensors on all devices. The full tensors will be used to check
    # correctness.
    torch.manual_seed(42)
    a = torch.randn(D_MODEL)
    b = torch.randn(D_MODEL)

    # Each rank uses a different shard for the sharded dot-product
    a_sharded = a.reshape(WORLD_SIZE, D_MODEL // WORLD_SIZE)[rank]
    b_sharded = b.reshape(WORLD_SIZE, D_MODEL // WORLD_SIZE)[rank]

    # Compute the dot-product via collectives.

    # Each rank first computes their local dot-product using the available shards:
    c = a_sharded @ b_sharded

    # The computation is completed by summing over processes:
    dist.init_process_group(backend="gloo")
    dist.all_reduce(c)

    # Test correctness:
    torch.testing.assert_close(c, a @ b)
    return f"Correct results on {rank=}"


def run():
    with ProcessPoolExecutor(max_workers=WORLD_SIZE) as pool:
        ranks_list = [r for r in range(WORLD_SIZE)]
        results = pool.map(compute_dot_product, ranks_list)
        for r in results:
            print(r)


if __name__ == "__main__":
    run()
