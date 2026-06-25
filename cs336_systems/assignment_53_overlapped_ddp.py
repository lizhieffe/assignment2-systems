# Assignment 5.2 - Naive DDP impl.
#
# Run the code:
# uv run cs336_systems/assignment_52_naive_ddp.py

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cs336_basics import model
from cs336_systems.model_configs import (  # noqa: F401
  MODEL_CONFIG_S,
  MODEL_CONFIG_S_LC,
  MODEL_CONFIG_S_SC,
  MODEL_CONFIG_M,
  MODEL_CONFIG_M_LC,
  MODEL_CONFIG_M_SC,
  MODEL_CONFIG_L_SC,
  MODEL_CONFIG_XL_LC,
  MODEL_CONFIG_XL_SC,
)

MODEL_CONFIG = MODEL_CONFIG_S_SC
MODEL_CONFIG_NAME = next(
  name for name, value in globals().items()
  if name.startswith("MODEL_CONFIG_") and value is MODEL_CONFIG
)
BATCH_SIZE = 1
USE_GPU = True
WORLD_SIZE = 2
USE_ASYNC_COMM = False


class DDP:
  """A wrapper to convert a nn.Module to a DDP module."""

  def __init__(self, module: nn.Module, use_async_comm: bool = True):
    """
    Args:
      module: the model to wrap to convert to DDP model.
      use_async_comm: if true, use async comm to overlap with the grad
        computation in bwd pass.

    """
    self.module = module
    self.use_async_comm = use_async_comm

    self._handles: list[dist.Work] = []
    self._comm_latencies: list[float] = []  # sec

    for param in self.module.parameters():
      # The device 0 broadcast the initial params.
      with torch.no_grad():
        dist.broadcast(param.data, src=0)

      # Skip the params that doesn't needs grad.
      if param.requires_grad:
        param.register_post_accumulate_grad_hook(self._grad_all_reduce)

  def _grad_all_reduce(self, param: nn.Parameter) -> None:
    is_cuda = param.grad.is_cuda
    if not self.use_async_comm:
      if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
      else:
        start_time = time.time()

    handle = dist.all_reduce(
      param.grad, op=dist.ReduceOp.AVG, async_op=self.use_async_comm
    )

    if not self.use_async_comm:
      if is_cuda:
        # The GPU time is more accurate compared to time.time() CPU time.
        end_event.record()
        end_event.synchronize()
        self._comm_latencies.append(start_event.elapsed_time(end_event) / 1000)
      else:
        self._comm_latencies.append(time.time() - start_time)

    if handle is not None:
      self._handles.append(handle)

  def named_parameters(self):
    return self.module.named_parameters()

  def parameters(self):
    return self.module.parameters()

  def __call__(self, x):
    return self.module(x)

  def finish_gradient_synchronization(self):
    for handle in self._handles:
      handle.wait()
    self._handles.clear()

  def get_comm_latencies(self) -> list[float]:
    """Get comm latencies.

    Only valid when the self.use_async_comm is False.
    """
    return self._comm_latencies


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


def warmup(device: str, n: int) -> None:
  """Warm up the dist comm.

  Args:
    device: the device
    n: # of comm in the warmup.
  """
  for _ in range(n):
    x = torch.randint(0, 10, (3,)).to(device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)


def distributed_training(
  rank: int,
  world_size: int,
  use_gpu: bool,
  use_async_comm: bool,
  batch_size: int,
) -> None:
  if use_gpu:
    assert torch.cuda.is_available(), "GPU is not available"
    device_count = torch.cuda.device_count()
    assert device_count >= world_size, (
      f"Less GPU {device_count} than world size {world_size}"
    )
    device = f"cuda:{rank}"
  else:
    device = "cpu"

  setup(
    rank=rank,
    world_size=world_size,
    backend="nccl" if use_gpu else "gloo",
    device_id=torch.device(device) if use_gpu else None,
  )

  try:
    # Warmup
    dist.barrier()
    start_time = time.time()
    warmup(device=device, n=5)
    if use_gpu:
      torch.cuda.synchronize()
    if rank == 0:
      print(f"Warm up used {time.time() - start_time:.5f} sec")

    vocab_size = MODEL_CONFIG["vocab_size"]
    context_length = MODEL_CONFIG["context_length"]

    lm = model.BasicsTransformerLM(
      vocab_size=vocab_size,
      context_length=context_length,
      d_model=MODEL_CONFIG["d_model"],
      num_layers=MODEL_CONFIG["num_layers"],
      num_heads=MODEL_CONFIG["num_heads"],
      d_ff=MODEL_CONFIG["d_ff"],
    ).to(device, dtype=torch.bfloat16)
    ddp_lm = DDP(lm, use_async_comm=use_async_comm)

    x = torch.randint(0, vocab_size, (batch_size, context_length)).to(
      device=device
    )

    # Forward
    dist.barrier()
    start_time = time.time()
    y = ddp_lm(x)
    if use_gpu:
      torch.cuda.synchronize()
    if rank == 0:
      print(f"FWD used {time.time() - start_time:.5f} sec")

    loss = y.sum()

    # Backward
    dist.barrier()
    start_time = time.time()
    loss.backward()
    if use_gpu:
      torch.cuda.synchronize()
    if rank == 0:
      print(f"BWD used {time.time() - start_time:.5f} sec")

    comm_latencies = ddp_lm.get_comm_latencies()
    if rank == 0:
      print(f"Comm latencies sum: {sum(comm_latencies):.5f} sec")

  finally:
    dist.destroy_process_group()


def main():
  print(
    f"{MODEL_CONFIG_NAME=} {USE_GPU=} {WORLD_SIZE=} {USE_ASYNC_COMM=} "
    f"{BATCH_SIZE=}"
  )

  mp.spawn(
    fn=distributed_training,
    args=(WORLD_SIZE, USE_GPU, USE_ASYNC_COMM, BATCH_SIZE),
    nprocs=WORLD_SIZE,
  )


if __name__ == "__main__":
  main()
