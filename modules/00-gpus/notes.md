# Module 00 — Working on GPUs

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation |
| VRAM | 15957 MiB usable (16380 MiB reported by `nvidia-smi`) |
| SMs | 22 |
| Bandwidth, spec | 224 GB/s |
| Bandwidth, measured | **207 GB/s** — 92% of spec |

Host: RunPod, EU-RO-1, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
Dev loop: VS Code Remote-SSH, editing files that live on the pod, work kept on a persistent
network volume at `/workspace`.

## Measuring bandwidth

`bandwidth.py` allocates two 1 GiB fp16 buffers in VRAM, copies one into the other 50 times, and
divides the bytes that crossed the memory bus by the elapsed seconds.

It produced two wrong answers before the right one, and the wrong ones taught more than the
result did.

### Wrong answer 1 — 199 TB/s

Timing the loop with `time.perf_counter()` reported 0.270 ms to move 50 GiB. That is roughly 890x
the physical maximum of the card, which is the useful part: a number that impossible is not an
optimistic measurement, it is a measurement of something else entirely.

CUDA calls are asynchronous. `dst.copy_(src)` returns as soon as the CPU has queued the kernel on
a stream, not when the GPU has executed it. 0.270 ms across 50 iterations is ~5.4 us per
iteration — the cost of the CPU writing down an order, not of the GPU carrying it out. The copies
did happen; they happened after the clock had stopped.

Fix: `torch.cuda.synchronize()` before reading the clock, so the CPU waits for the queue to drain.

### Wrong answer 2 — 103 GB/s

With the clock fixed, 50 GiB in 519.8 ms → 103.3 GB/s. Slower than expected by a factor of very
close to exactly 2.

An exact 2x is almost never a performance problem. Contention, thermal throttling or clock limits
produce dirty ratios — 0.7x, 1.3x. A clean 2.0x means one side of the division is miscounted.

A copy touches every byte twice: once reading `src`, once writing `dst`. The byte crosses the
memory bus in both directions. Counting only the bytes copied ignores half the traffic.

### The number

```
traffic     107.4 GB  (53.7 GB copied, read + write)
elapsed     519.8 ms
bandwidth   206.6 GB/s
of spec     92%
```

92% of peak is above the 70-85% band usually quoted, and that is expected here rather than
suspicious. A contiguous device-to-device copy is the friendliest possible workload for memory:
perfectly sequential access, no compute, no strided reads. It measures the ceiling, not a
realistic workload. Real kernels with strided access or mixed compute land lower.

The script records the read+write decision explicitly as `TRAFFIC_PER_BYTE = 2` rather than
folding it into a magic number, because that factor of 2 is exactly what went wrong once already.

## The 7B arithmetic

Using the **measured** 207 GB/s, not the spec.

- Weights at fp16: 7e9 params x 2 bytes = 14 GB
- Time for one full read of the weights: TODO
- Ceiling in tokens/s at batch size 1: TODO
- Memory-bound or compute-bound, and by what margin: TODO
- Batch size at which we stop being memory-bound: TODO

Note: reading weights is read-only. The 2x traffic factor from the copy benchmark does **not**
carry over here — check this before using the number.

## What this changes

TODO — the consequences worth stating: batching, quantization, prefill vs. generation,
and what 14 GB of weights on a 16 GB card means for the KV cache.
