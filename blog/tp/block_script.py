"""
Prints out the ratio of activation memory for the a transformer Block when using ReLU vs GELU.
"""

import torch
import torch.nn as nn

import act_mem
import layers

if __name__ == "__main__":
    batch_size, seq_len, d_model, n_heads = 2, 4096, 1024, 2
    dtype = torch.bfloat16
    inputs = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    )

    act_fn_dict = {"ReLU": nn.ReLU(), "GELU": nn.GELU()}
    # Append outputs to a list to keep tensors alive
    outputs = []
    mem_bytes = []

    for name, act_fn in act_fn_dict.items():
        block = layers.Block(
            d_model=d_model,
            act_fn=act_fn,
            n_heads=n_heads,
            device="cuda",
            dtype=dtype,
        )
        with act_mem.AllocatedMemContext() as mem, act_mem.SavedTensorContext(
            ignored_tensors=block.parameters()
        ) as saved:
            out = block(inputs)
            outputs.append(out)
        print(f"{name} block bytes: {saved.saved_tensor_mem}")
        mem_bytes.append(saved.saved_tensor_mem)

    print(f"ReLU/GeLU block act mem ratio: {mem_bytes[0]/mem_bytes[1]}")
