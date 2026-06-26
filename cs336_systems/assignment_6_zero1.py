# Assignment 6 - ZeRO1
#
# Run the code:
# uv run cs336_systems/assignment_6_zero1.py

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from typing import Any


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
