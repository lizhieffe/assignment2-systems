# Assignment 5.2 - Naive DDP impl.
#
# The naive DDP sync the grads at the end of BWD layer by layer.
# It doesn't have comm overlap or the flattened grads.
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
  name
  for name, value in globals().items()
  if name.startswith("MODEL_CONFIG_") and value is MODEL_CONFIG
)
BATCH_SIZE = 1
USE_GPU = True
WORLD_SIZE = 2


class DDP:
  """A wrapper to convert a nn.Module to a DDP module."""

  def __init__(self, module: nn.Module):
    """
    Args:
      module: the model to wrap to convert to DDP model.
    """
    self.module = module

    self._comm_latencies: list[float] = []  # sec
    self._comm_bytes: list[int] = []  # total bytes all-reduced per call
    self._comm_num_calls: list[int] = []  # # of all_reduce calls per step

    for param in self.module.parameters():
      # The device 0 broadcast the initial params.
      with torch.no_grad():
        dist.broadcast(param.data, src=0)

  def named_parameters(self):
    return self.module.named_parameters()

  def parameters(self):
    return self.module.parameters()

  def __call__(self, x):
    return self.module(x)

  def finish_gradient_synchronization(self):
    """Only add_reduce once for the concat grad tensor."""
    params = [p for p in self.module.parameters() if p.requires_grad]
    handles = []

    is_cuda = params[0].is_cuda
    if is_cuda:
      torch.cuda.synchronize()

    dist.barrier()

    if is_cuda:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
    else:
      start_time = time.time()

    for p in params:
      handle = dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=False)
      if handle is not None:
        handles.append(handle)

    for handle in handles:
      handle.wait()
    handles.clear()

    # dist.all_reduce(async_op=False) only enqueues the NCCL kernel and
    # returns without blocking the host, so a CUDA event (or an explicit
    # torch.cuda.synchronize()) is required here to time actual completion
    # rather than just kernel-dispatch time.
    if is_cuda:
      end_event.record()
      end_event.synchronize()
      self._comm_latencies.append(start_event.elapsed_time(end_event) / 1000)
    else:
      self._comm_latencies.append(time.time() - start_time)

    self._comm_bytes.append(
      sum(p.grad.numel() * p.grad.element_size() for p in params)
    )
    self._comm_num_calls.append(len(params))

  def get_comm_latencies(self) -> list[float]:
    """Get comm latencies."""
    return self._comm_latencies

  def get_comm_bytes(self) -> list[int]:
    """Get total bytes all-reduced per `finish_gradient_synchronization` call."""
    return self._comm_bytes

  def get_comm_num_calls(self) -> list[int]:
    """Get # of `dist.all_reduce` calls per `finish_gradient_synchronization` call."""
    return self._comm_num_calls

  def reset_comm_stats(self) -> None:
    """Clear comm stats recorded so far, e.g. after a warmup run."""
    self._comm_latencies.clear()
    self._comm_bytes.clear()
    self._comm_num_calls.clear()


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


def forward_backward_step(
  ddp_lm: DDP, x: torch.Tensor, use_gpu: bool
) -> torch.Tensor:
  """Run one forward + backward step through the DDP-wrapped model."""
  y = ddp_lm(x)
  if use_gpu:
    torch.cuda.synchronize()

  loss = y.sum()
  loss.backward()
  ddp_lm.finish_gradient_synchronization()
  if use_gpu:
    torch.cuda.synchronize()

  return loss


def distributed_training(
  rank: int,
  world_size: int,
  use_gpu: bool,
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
    ddp_lm = DDP(lm)

    x = torch.randint(0, vocab_size, (batch_size, context_length)).to(
      device=device
    )

    # Warmup: run a real forward/backward step (untimed) so that CUDA
    # lazy kernel/library loading happens before the timed run below.
    dist.barrier()
    forward_backward_step(ddp_lm, x, use_gpu)
    for param in ddp_lm.parameters():
      param.grad = None
    ddp_lm.reset_comm_stats()

    # Forward
    dist.barrier()
    start_time = time.time()
    y = ddp_lm(x)
    if use_gpu:
      torch.cuda.synchronize()
    if rank == 0:
      print(f"FWD used {time.time() - start_time:.5f} sec")

    loss = y.sum()
    if use_gpu:
      torch.cuda.synchronize()

    # Backward
    dist.barrier()
    start_time = time.time()
    loss.backward()
    ddp_lm.finish_gradient_synchronization()
    if use_gpu:
      torch.cuda.synchronize()
    if rank == 0:
      print(f"BWD used {time.time() - start_time:.5f} sec")

    comm_latencies = ddp_lm.get_comm_latencies()
    comm_bytes = sum(ddp_lm.get_comm_bytes())
    comm_num_calls = sum(ddp_lm.get_comm_num_calls())
    comm_time = sum(comm_latencies)
    if rank == 0:
      print(f"Comm latencies sum: {comm_time:.5f} sec")
      print(f"Comm # all_reduce calls: {comm_num_calls}")
      print(f"Comm bytes: {comm_bytes / 1e6:.3f} MB")
      print(
        f"Comm effective bandwidth: {comm_bytes / comm_time / 1e9:.3f} GB/s"
      )

  finally:
    dist.destroy_process_group()


def main():
  print(f"{MODEL_CONFIG_NAME=} {USE_GPU=} {WORLD_SIZE=} {BATCH_SIZE=}")

  mp.spawn(
    fn=distributed_training,
    args=(WORLD_SIZE, USE_GPU, BATCH_SIZE),
    nprocs=WORLD_SIZE,
  )


if __name__ == "__main__":
  main()
