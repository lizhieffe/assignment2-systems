# Assignment 5.2 - Naive DDP impl.
#
# Run the code:
# uv run cs336_systems/assignment_52_naive_ddp.py

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP:
  """A wrapper to convert a nn.Module to a DDP module."""

  def __init__(self, module: nn.Module):
    self.module = module
    self._handles: list[dist.Work] = []

    for param in self.module.parameters():
      # The device 0 broadcast the initial params.
      with torch.no_grad():
        dist.broadcast(param.data, src=0)

      # Skip the params that doesn't needs grad.
      if param.requires_grad:
        param.register_post_accumulate_grad_hook(self._grad_all_reduce)

  def _grad_all_reduce(self, param: nn.Parameter) -> None:
    """Async comm to overlap with the grad computation in bwd pass."""
    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
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
