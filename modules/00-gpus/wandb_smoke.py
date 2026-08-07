"""Smoke test for the Weights & Biases pipeline.

Logs a fake loss curve. The point is not the curve -- it is proving that auth, network egress
from the pod, and the run lifecycle all work now, in the cold, rather than discovering they
don't in the middle of a real training run.

Auth comes from the WANDB_API_KEY environment variable, injected by RunPod at pod start.
"""

import math
import random

import wandb

STEPS = 100

run = wandb.init(
    project="lab-notebook",
    name="module-00-smoke",
    config={"steps": STEPS, "note": "fake curve, no model involved"},
)

for step in range(STEPS):
    # Exponential decay plus noise: shaped like a loss curve, means nothing.
    loss = math.exp(-step / 30) + random.uniform(-0.02, 0.02)
    wandb.log({"loss": loss, "lr": 1e-3}, step=step)

run.finish()
print(f"done -- {STEPS} steps logged, see the run URL above")
