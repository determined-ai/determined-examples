# Tensor Parallelism

Code accompanying the deep-dive [blog post on Tensor Parallelism](https://determined.ai/blog/tp).

- The MLP and TP MLP layers are in `layer.py`
- Matmul profiling code in `matmul_profiling.py`
- MLP TP profiling code in `tp_profiling.py`
- Tests of the rearranging tensor sums are in `test_dot_product_{local,distributed}.py`


## Contributors

- [Garrett Goon](https://github.com/garrett361)