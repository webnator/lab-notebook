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

The gap between the two VRAM figures is the driver and the CUDA context. The number I can
actually allocate against is the smaller one.

## The dev loop

VS Code Remote-SSH, editing files that live on the pod. No sync step, so there is no version of
"which copy am I running" to get wrong. Work lives on a persistent network volume at `/workspace`;
everything outside it is ephemeral and gets recycled with the pod.

Two things cost me time and are worth writing down:

**SSH asked for a password and no password exists.** The pod had been created with `PUBLIC_KEY`
unset, so nothing was written to `authorized_keys`; key auth failed and sshd fell through to
password auth, which on this image is a dead end. RunPod is key-only. The fix was setting the env
var and restarting.

**The SSH port changes on every stop/start.** RunPod remaps the container port 22 to a new public
port each time. An alias in `~/.ssh/config` turns a reconnect into editing one line instead of
rebuilding the command. I also used `ForwardAgent yes` so git can push to GitHub from the pod
without a private key ever living on an ephemeral machine.

## Measuring bandwidth

`bandwidth.py` allocates two 1 GiB fp16 buffers in VRAM, copies one into the other 50 times, and
divides the bytes that crossed the memory bus by the elapsed seconds. It took two wrong answers
to get there, and the wrong ones taught more than the result.

### First run — 199 TB/s

50 GiB in 0.270 ms. Roughly 890x the physical maximum of the card.

My first read was that it had only moved pointers on the CPU and never involved the GPU, so no
data had actually moved. Half right, and the wrong half is the interesting one.

Right: the CPU did not move any data. 0.270 ms across 50 iterations is ~5.4 us per iteration,
the cost of issuing an order rather than of carrying it out.

Wrong: the data did move. `dst.copy_(src)` is not a pointer swap, it is a real device-to-device
copy, and all 50 executed. They executed after the clock had stopped.

The model I was missing: `dst.copy_(src)` is the CPU **queuing** a kernel on a CUDA stream. The
call returns in microseconds without waiting. The GPU drains that queue on its own schedule.
`time.perf_counter()` measured how long the CPU took to write down 50 orders. The GPU was never
idle — only the clock was wrong.

Fix: `torch.cuda.synchronize()` before reading the clock.

### Prediction before the second run

Written down before measuring, on purpose:

> elapsed ~220 ms, bandwidth ~200 GB/s

The method was right, time = bytes / bandwidth, but I divided 50 by 224 while the 50 was in
**GiB** and the 224 in **GB**. 50 GiB is 53.7 GB, so the prediction should have been 240 ms. A 7%
error, harmless here, and exactly the kind that decides whether a model fits in VRAM or OOMs.

Rule to keep: GiB to GB, multiply by 1.074.

### Second run — 103 GB/s

519.8 ms. Twice my predicted time, half my predicted bandwidth.

The useful move was noticing the ratio was *exactly* 2. Contention, thermal throttling or clock
limits produce dirty ratios, 0.7x or 1.3x. A clean 2.0x is not a performance problem, it is a
counting problem: one side of the division is wrong. The clock was now correct, so the error had
to be in the byte count.

A copy touches every byte twice, once reading `src` and once writing `dst`. The byte crosses the
memory bus in each direction. Counting only the bytes copied ignores half the traffic.

Honest note: I first saw "read and write" in the editor autocomplete before I had reasoned it
out. The derivation came after — the byte sits in VRAM and has to end up elsewhere in VRAM, so it
travels the path twice.

### The number

```
traffic     107.4 GB  (53.7 GB copied, read + write)
elapsed     519.8 ms
bandwidth   206.6 GB/s
of spec     92%
```

92% is above the 70-85% band usually quoted, and that is expected rather than suspicious. A
contiguous device-to-device copy is the friendliest workload memory can get: perfectly sequential
access, no compute, no strided reads. It measures the ceiling, not anything realistic. Kernels
with strided access or mixed compute land lower.

`bandwidth.py` records the read+write choice as an explicit `TRAFFIC_PER_BYTE = 2` rather than
folding it into a magic number, because that factor is what went wrong once already.

## The 7B arithmetic

Using the **measured** 207 GB/s, not the spec.

- Weights at fp16: 7e9 params x 2 bytes = 14 GB
- Time for one full read of the weights: TODO
- Ceiling in tokens/s at batch size 1: TODO
- Memory-bound or compute-bound, and by what margin: TODO
- Batch size at which we stop being memory-bound: TODO

Reading weights is read-only. The 2x traffic factor from the copy benchmark does **not** carry
over here — verify before using the number.

## What this changes

TODO — batching, quantization, prefill vs. generation, and what 14 GB of weights on a 16 GB card
means for the KV cache.
