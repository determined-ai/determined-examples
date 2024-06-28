"""
Demonstrating the equivalence of a basic dot product with intermediate activation function and a
sharded-version of the same calculation.
"""

import torch

D_MODEL = 128
RANKS = 4

if __name__ == "__main__":
    a = torch.randn(D_MODEL)
    b = torch.randn(D_MODEL)

    act_fn = torch.nn.GELU()
    # The dot-product, two different ways
    dot_0 = a @ act_fn(b)
    dot_1 = (a * act_fn(b)).sum()

    a_sharded = a.reshape(RANKS, D_MODEL // RANKS)
    b_sharded = b.reshape(RANKS, D_MODEL // RANKS)

    # More equivalent dot-products, using the sharded tensors.
    dot_2 = (a_sharded * act_fn(b_sharded)).sum()
    dot_3 = (a_sharded @ act_fn(b_sharded).T).trace()
    dot_4 = (a_sharded.T @ act_fn(b_sharded)).trace()

    for dot_prod in (dot_1, dot_2, dot_3, dot_4):
        torch.testing.assert_close(dot_0, dot_prod)
    print("Correct results")
