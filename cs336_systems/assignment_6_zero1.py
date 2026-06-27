# Assignment 6 - ZeRO1
#
# Run the code:
# uv run cs336_systems/assignment_6_zero1.py

from typing import Any

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics import model, optimizer
from cs336_systems import assignment_52_naive_ddp

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

MODEL_CONFIG = MODEL_CONFIG_S_SC
MODEL_CONFIG_NAME = next(
  name
  for name, value in globals().items()
  if name.startswith("MODEL_CONFIG_") and value is MODEL_CONFIG
)

BATCH_SIZE = 1
USE_GPU = False
WORLD_SIZE = 2


class ZeRO1(torch.optim.Optimizer):
  def __init__(
    self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any
  ):
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
    self.param_counter = 0

    self.local_optimizer = None
    self.optimizer_cls = optimizer_cls
    self.kwargs = kwargs
    self.param_to_owner: dict[torch.Tensor, int] = {}

    super().__init__(params=params, defaults={})

  def add_param_group(self, param_group: dict[str, Any]):
    """
    Args:
      param_group: a dict like {"params": [...], "lr": 0.1, "weight_decay": 0.1, ...}
      (the full list of params plus any per-group hyperparameter overrides)."""
    super().add_param_group(param_group)

    local_params = []
    for p in param_group["params"]:
      owner = self.param_counter % self.world_size
      self.param_counter += 1
      self.param_to_owner[p] = owner
      if owner == self.rank:
        local_params.append(p)

    if not local_params:
      return

    local_group = {**param_group, "params": local_params}
    if self.local_optimizer is None:
      self.local_optimizer = self.optimizer_cls([local_group], **self.kwargs)
    else:
      self.local_optimizer.add_param_group(local_group)

  def step(self, closure=None, **kwargs):
    # In the BWD, the grad are already sync across all devices.
    # Thus in the optimizer, it can directly run step(), and then
    # broadcast the updated param.
    loss = self.local_optimizer.step(closure, **kwargs)

    # This is easy to get confused; every node should call
    # broadcast no matter it is the sender or receiver.
    for p, owner in self.param_to_owner.items():
      dist.broadcast(p.data, src=owner)
    return loss


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
  rank: int, world_size: int, use_gpu: bool, model_config: dict[str, Any]
):
  if use_gpu:
    assert torch.cuda.is_available(), "GPU is not available"
    device_count = torch.cuda.device_count()
    assert device_count >= world_size, (
      f"Less GPU {device_count} than world size {world_size}"
    )
    device = f"cuda:{rank}"
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
    d_model=MODEL_CONFIG["d_model"],
    num_layers=MODEL_CONFIG["num_layers"],
    num_heads=MODEL_CONFIG["num_heads"],
    d_ff=MODEL_CONFIG["d_ff"],
  ).to(device, dtype=torch.bfloat16)
  ddp_m = assignment_52_naive_ddp.DDP(m)
  opt = optimizer.AdamW(m.parameters())

  x = torch.randint(0, vocab_size, (1, context_length))

  # FWD + BWD
  torch.cuda.reset_peak_memory_stats()

  y = ddp_m(x)
  loss = y.sum()
  loss.backward()
  ddp_m.finish_gradient_synchronization()

  peak = torch.cuda.max_memory_allocated() / 1024**2
  print(f"Peak memory usage for FWD + BWD = {peak:.2f} MB")

  # optimizer step
  torch.cuda.reset_peak_memory_stats()
  opt.step()
  peak = torch.cuda.max_memory_allocated() / 1024**2
  print(f"Peak memory usage for optimizer step = {peak:.2f} MB")


def main():
  print(f"{USE_GPU=} {WORLD_SIZE=} {MODEL_CONFIG_NAME=}")
  world_size = WORLD_SIZE
  mp.spawn(
    fn=distributed_train,
    args=(world_size, USE_GPU, MODEL_CONFIG),
    nprocs=world_size,
    join=True,
  )


if __name__ == "__main__":
  main()
