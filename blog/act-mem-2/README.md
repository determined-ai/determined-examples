# Activation Memory: Part 2

Code accompanying the deep-dive [blog post on activation memory](https://determined.ai/blog/act-mem-2).

- The main utility code is in `act_mem.py`. 
- Basic transformer layers are implemented in `layers.py`.
- The scripts `{block,mlp}_script.py` demonstrate how replacing `GELU` with `ReLU` affects activation
memory. 
- `attn_script.py` shows the cost of activation memory in the attention layer. 
- Tests of the code are in `test.py`. 
- See `requirements.txt` for versions the code was built against.


## Contributors

- [Garrett Goon](https://github.com/garrett361)