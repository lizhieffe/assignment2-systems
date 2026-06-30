# Shared training harness for FSDP assignments.
# Import and call run_main() with your FSDP class to reuse this loop.
#
# Run the code:
# uv run cs336_systems/assignment_7_fsdp_main.py


import os
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics import model, optimizer

import assignment_7_fsdp_v1 as fsdp_lib
#import assignment_7_fsdp_v2 as fsdp_lib

from cs336_systems.model_configs import (  # noqa: F401
  MODEL_CONFIG_S,
  MODEL_CONFIG_S_LC,
  MODEL_CONFIG_S_SC,
  MODEL_CONFIG_M,
  MODEL_CONFIG_M_SC,
  MODEL_CONFIG_M_LC,
  MODEL_CONFIG_L_SC,
  MODEL_CONFIG_XL_LC,
  MODEL_CONFIG_XL_SC,
)

MODEL_CONFIG = MODEL_CONFIG_S
MODEL_CONFIG_NAME = next(
  name
  for name, value in globals().items()
  if name.startswith("MODEL_CONFIG_") and value is MODEL_CONFIG
)

BATCH_SIZE = 2
USE_GPU = True
WORLD_SIZE = 2


def setup(
  rank: int,
  world_size: int,
  backend: str,
  device_id: torch.device | None = None,
):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29500"
  dist.init_process_group(
    backend=backend, world_size=world_size, rank=rank, device_id=device_id
  )


def distributed_train(
  rank: int,
  world_size: int,
  use_gpu: bool,
  model_config: dict[str, Any],
  batch_size: int,
  fsdp_class: type,
):
  try:
    if use_gpu:
      assert torch.cuda.is_available(), "GPU is not available"
      device_count = torch.cuda.device_count()
      assert device_count >= world_size, (
        f"Less GPU {device_count} than world size {world_size}"
      )
      device = f"cuda:{rank}"
      torch.cuda.set_device(device)
    else:
      device = "cpu"
    if rank == 0:
      print(f"{device=}")

    setup(
      rank=rank,
      world_size=world_size,
      backend="nccl" if use_gpu else "gloo",
      device_id=torch.device(device) if use_gpu else None,
    )

    vocab_size = model_config["vocab_size"]
    context_length = model_config["context_length"]
    m = model.BasicsTransformerLM(
      vocab_size=vocab_size,
      context_length=context_length,
      d_model=model_config["d_model"],
      num_layers=model_config["num_layers"],
      num_heads=model_config["num_heads"],
      d_ff=model_config["d_ff"],
    ).to(device, dtype=torch.bfloat16)
    fsdp_m = fsdp_class(m)
    opt = optimizer.AdamW(fsdp_m.parameters())

    x = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)

    torch.cuda.reset_peak_memory_stats()
    y = fsdp_m(x)
    loss = y.sum()
    hbm_fwd_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{hbm_fwd_mb=:.2f}")

    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    hbm_bwd_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{hbm_bwd_mb=:.2f}")

    torch.cuda.reset_peak_memory_stats()
    fsdp_m.finish_gradient_synchronization()
    hbm_finish_gradient_synchronization_mb = (
      torch.cuda.max_memory_allocated() / 1024**2
    )
    print(f"{hbm_finish_gradient_synchronization_mb=:.2f}")

    torch.cuda.reset_peak_memory_stats()
    opt.step()
    hbm_opt_step_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{hbm_opt_step_mb=:.2f}")

  finally:
    dist.destroy_process_group()


def main():
  print(
    f"USE_GPU={USE_GPU} WORLD_SIZE={WORLD_SIZE} "
    f"MODEL_CONFIG_NAME={MODEL_CONFIG_NAME} BATCH_SIZE={BATCH_SIZE}"
  )
  mp.spawn(
    fn=distributed_train,
    args=(WORLD_SIZE, USE_GPU, MODEL_CONFIG, BATCH_SIZE, fsdp_lib.FSDP),
    nprocs=WORLD_SIZE,
    join=True,
  )


if __name__ == "__main__":
  main()
