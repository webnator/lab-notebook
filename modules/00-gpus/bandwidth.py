"""Measure achievable memory bandwidth on this GPU.

Method: allocate a large tensor in VRAM, copy it repeatedly, divide bytes moved by seconds.

Two things that will give a wrong answer:

  1. CUDA calls are asynchronous. Timing with time.time() around a copy measures kernel *launch*,
     not execution. The symptom is unmistakable: an impossible number, tens of TB/s. Fix with
     torch.cuda.synchronize() before stopping the clock, or use torch.cuda.Event.

  2. The first iteration includes allocation and kernel compilation. Warm up before measuring.

A copy both reads and writes. Decide whether that counts as 1x or 2x the tensor size and write
the choice down here — it is a factor-of-2 difference in the result.

Expect to land at 70-85% of the spec sheet number. That gap is real, and the measured number is
the one worth reasoning with.
"""

# TODO
