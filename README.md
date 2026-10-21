# Lab Notebook

A working engineer's path into AI research engineering, built in the open.

Each module is self-contained: a small artifact, the numbers behind it, and the reasoning that
produced them. The point is the working, not the polish.

## Modules

| # | Module | Demonstrates | Status |
|---|--------|--------------|--------|
| [00](modules/00-gpus/) | Working on GPUs | Measured memory bandwidth on real hardware, and why LLM generation is memory-bound | In progress |

## How to read this

Every module directory holds the code plus a `notes.md` carrying the working , the arithmetic,
the wrong turns, and the number that came out at the end.

## Reproduce the first experiment

The bandwidth script requires an NVIDIA CUDA GPU and a compatible PyTorch installation. It allocates two 1 GiB buffers. CPU-only machines and Apple Silicon cannot run this CUDA experiment as written.

```sh
python modules/00-gpus/bandwidth.py
```

Install PyTorch for your CUDA environment using its official installation instructions. The notes record the original environment and measurements. `SPEC_GB_S` refers to that original GPU; update it before comparing a different card with its specification. Recorded measurements are historical results, not results automatically reproduced on your machine.

The optional `wandb_smoke.py` script requires the `wandb` package and your own `WANDB_API_KEY`. It uploads a synthetic loss curve to your Weights & Biases account; it does not train a model.

This is a learning notebook. Hardware assumptions and estimates are documented alongside results and may be revised as the experiments progress.

## License

MIT. See [LICENSE](LICENSE).
