"""
Demonstrating the equivalence of a basic dot product and a sharded-version of the same dot product.
"""

import torch

D_MODEL = 128

if __name__ == "__main__":
    a = torch.randn(D_MODEL)
    b = torch.randn(D_MODEL)

    # The dot-product, two different ways
    c_0 = a @ b
    c_1 = (a * b).sum()

    a_sharded = a.reshape(2, D_MODEL // 2)
    b_sharded = b.reshape(2, D_MODEL // 2)

    # More equivalent dot-products, using the sharded tensors.
    c_2 = (a_sharded * b_sharded).sum()
    c_3 = (a_sharded @ b_sharded.T).trace()
    c_4 = (a_sharded.T @ b_sharded).trace()

    for dot_prod in (c_1, c_2, c_3, c_4):
        torch.testing.assert_close(c_0, dot_prod)
