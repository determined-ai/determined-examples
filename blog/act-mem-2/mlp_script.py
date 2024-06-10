import torch
import torch.nn as nn

import act_mem
import layers

if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 4096, 1024
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
        mlp = layers.MLP(
            d_model=d_model,
            act_fn=act_fn,
            device="cuda",
            dtype=dtype,
        )
        with act_mem.AllocatedMemContext() as mem, act_mem.SavedTensorContext(
            ignored_tensors=mlp.parameters()
        ) as saved:
            out = mlp(inputs)
            outputs.append(out)
        assert mem.delta["current"] == saved.saved_tensor_mem
        print(f"{name} bytes: {saved.saved_tensor_mem}")
        mem_bytes.append(saved.saved_tensor_mem)

    print(f"ReLU/GeLU act mem ratio: {mem_bytes[0]/mem_bytes[1]}")
