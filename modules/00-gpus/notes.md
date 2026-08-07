# Module 00 — Working on GPUs

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation |
| VRAM | 15957 MiB usable (16380 MiB reported by `nvidia-smi`) |
| SMs | 22 |
| Bandwidth, spec | TODO — look it up and verify |
| Bandwidth, measured | TODO — from `bandwidth.py` |

## Measuring bandwidth

TODO — what the benchmark does, what came out wrong on the first attempt and why,
measured vs. spec as a percentage of peak.

## The 7B arithmetic

Using the **measured** bandwidth, not the spec.

- Weights at fp16: 7e9 params x 2 bytes = 14 GB
- Time for one full read of the weights: TODO
- Ceiling in tokens/s at batch size 1: TODO
- Memory-bound or compute-bound, and by what margin: TODO
- Batch size at which we stop being memory-bound: TODO

## What this changes

TODO — the consequences worth stating: batching, quantization, prefill vs. generation,
and what 14 GB of weights on a 16 GB card means for the KV cache.
