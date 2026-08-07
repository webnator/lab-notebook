# Module 00 — Working on GPUs

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation |
| VRAM | 15957 MiB usable (16380 MiB reported by `nvidia-smi`) |
| SMs | 22 |
| Bandwidth, spec | 224 GB/s (NVIDIA datasheet, checked) |
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

All of it on the **measured** 207 GB/s, not the spec.

### One full read of the weights

```
7e9 params x 2 bytes (fp16)  =  14 GB
14 GB / 207 GB/s             =  67.6 ms
```

Reading weights is read-only, so the 2x traffic factor from the copy benchmark does not carry
over. That factor existed because a copy writes back what it reads; a forward pass does not write
the weights anywhere.

### Ceiling in tokens/s at batch size 1

Producing one token requires reading every weight once, so the ceiling is the inverse of the read:

```
1 / 0.0676 s  =  14.8 tokens/s
```

Theoretical, with the hardware perfectly saturated. Slow enough to watch arrive word by word.

### Memory-bound or compute-bound

Arithmetic per token is ~2 FLOPs per parameter: `2 x 7e9 = 14 GFLOP`.

Card figure used: **30.0 TFLOP/s, Tensor Core**, taken as the dense number. Deliberately not the
other two on the datasheet — 12.0 TFLOP/s FP32 is the CUDA cores, which LLM inference does not run
on, and 24.0 TFLOP/s RT is ray-triangle intersection hardware, unrelated to any of this. The sheet
does not label the 30.0 as dense explicitly; if it turns out to be the structured-sparsity figure,
the dense one is half and the ridge point below halves with it.

```
1.4e10 FLOP / 3.0e13 FLOP/s  =  0.47 ms of compute
67.6 ms of memory / 0.47 ms of compute  =  145x
```

**Memory-bound by 145x.** The tensor cores are idle about 99.3% of the time, waiting for weights
to arrive over the bus.

First pass at this gave 0.47 as 0.047 — one order of magnitude out, from dividing 14 by 30 and
placing the exponent afterwards instead of normalising to 1.4e10 / 3.0e13 first. The qualitative
conclusion survived either way, which is exactly why the slip went unnoticed. What caught it was
the next number refusing to agree.

### Batch size at which we stop being memory-bound

```
ridge point = 3.0e13 FLOP/s / 2.07e11 B/s = 145 FLOPs per byte
```

At batch size B the arithmetic intensity is ~B FLOPs/byte, so the ridge is reached around
**batch 145**.

### Why those two numbers are the same

The 145x margin and the batch-145 ridge point are not a coincidence and not two measurements.
They are the same quantity:

```
margin  =  t_memory / t_compute  =  (W / BW) / (2P / F)

    W = 2P              fp16: 2 bytes per parameter
    FLOPs per token = 2P    2 FLOPs per parameter

        =  (2P / BW) / (2P / F)  =  F / BW      <- the definition of the ridge point
```

The `2P` cancels. It cancels because in fp16 every parameter contributes 2 bytes to read and
2 FLOPs to do, which puts the arithmetic intensity at exactly **1 FLOP per byte** at batch size 1.
When intensity is 1, the ratio of the two times is just the ratio of the machine's two capacities.

Two things fall out of that:

**The 7B never mattered.** It cancelled out. A 70B in fp16 on this same card gives the same 145 —
ten times slower in absolute terms, equally memory-bound, same ridge point. 145 is a property of
the hardware, not of the model.

**And it only holds in fp16.** In int8 the bytes halve but the FLOPs do not, so intensity becomes
2 FLOPs/byte and the ridge arrives at batch ~72. That is the less-told half of quantization: it
speeds up generation *and* moves you toward the compute-bound regime sooner.

## What this changes

**Batching.** Read the weights once, serve 32 sequences. Same bytes moved, 32x the output tokens,
intensity up 32x. On this card there is headroom all the way to batch ~145 before the tensor cores
become the constraint — every batch slot below that is close to free throughput.

**Quantization is not optional here, it is the entry ticket.** 14 GB of fp16 weights on a card
with 15.96 GB usable leaves under 2 GB. Assuming Llama-2-7B geometry — 32 layers, hidden 4096, no GQA, not
verified against a specific checkpoint — the KV cache costs
`2 (K and V) x 32 x 4096 x 2 bytes` per token, about 512 KB. Two spare gigabytes buy roughly **4,000 tokens of context**, and that is before activations
or fragmentation. The weights fit; there is barely room left to think. int8 halves the weights and
buys back most of the context window.

**Prefill and generation are opposite regimes on the same hardware.** Prefill pushes many tokens
through one read of the weights, so its intensity scales with prompt length and it lands
compute-bound. Generation is one token per full read of the weights: intensity 1, memory-bound by
145x. Same model, same card, two different machines depending on which phase you are in.

**And the practical one.** The measured 207 GB/s, not the 224 on the datasheet, is what these
numbers rest on. The gap is small here; the habit is not. Every estimate above is only as good as
the number it was built from, which is why the module measures rather than quotes.
