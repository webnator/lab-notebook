"""Measure achievable memory bandwidth on this GPU.

Method: allocate two large buffers in VRAM, copy one into the other repeatedly, and divide the
bytes that crossed the memory bus by the seconds it took.

Two things that make this read wrong, both of which this script handles explicitly:

  1. CUDA calls are asynchronous. `dst.copy_(src)` returns as soon as the CPU has queued the
     kernel, not when the GPU has run it. Timing the loop without torch.cuda.synchronize()
     measures how fast the CPU can write down 50 orders -- roughly 5 us each, which reports an
     impossible ~199 TB/s.

  2. A copy touches every byte twice: once reading src, once writing dst. Counting only the
     bytes copied halves the answer. TRAFFIC_PER_BYTE below is that decision, made in the open.

Buffers are sized well past the L2 cache so this measures VRAM and not cache.
"""

import time

import torch

N = 512 * 1024 * 1024  # elements; at fp16 that is 1 GiB per buffer
ITERS = 50
WARMUP = 5
TRAFFIC_PER_BYTE = 2  # a copy reads src and writes dst
SPEC_GB_S = 224.0  # NVIDIA RTX 2000 Ada datasheet

src = torch.empty(N, dtype=torch.float16, device="cuda")
dst = torch.empty(N, dtype=torch.float16, device="cuda")

bytes_per_copy = src.numel() * src.element_size()

# The first copies pay for allocation and kernel compilation; keep them out of the measurement.
for _ in range(WARMUP):
    dst.copy_(src)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(ITERS):
    dst.copy_(src)
torch.cuda.synchronize()  # wait for the queue to drain before stopping the clock
elapsed = time.perf_counter() - start

copied = bytes_per_copy * ITERS
traffic = copied * TRAFFIC_PER_BYTE
bandwidth = traffic / elapsed / 1e9

print(f"buffer      {bytes_per_copy / 1e9:.2f} GB ({bytes_per_copy / 1024**3:.2f} GiB)")
print(f"iterations  {ITERS}")
print(f"copied      {copied / 1e9:.1f} GB")
print(f"traffic     {traffic / 1e9:.1f} GB  (read + write)")
print(f"elapsed     {elapsed * 1e3:.1f} ms")
print(f"bandwidth   {bandwidth:.1f} GB/s")
print(f"of spec     {100 * bandwidth / SPEC_GB_S:.0f}%  (spec {SPEC_GB_S:.0f} GB/s)")
