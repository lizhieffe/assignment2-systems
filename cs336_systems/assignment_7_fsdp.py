# Assignment 7 - FSDP
#
# Run the code:
# uv run cs336_systems/assignment_7_fsdp.py


from collections.abc import Mapping
from typing import Any

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics import model, optimizer

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

BATCH_SIZE = 16
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


def is_norm_layer(module: nn.Module) -> bool:
  # 组合所有常见的归一化层基类
  norm_classes = (
    nn.modules.batchnorm._BatchNorm,  # 包含 BatchNorm1d, 2d, 3d
    nn.LayerNorm,  # LayerNorm
    nn.GroupNorm,  # GroupNorm
    nn.modules.instancenorm._InstanceNorm,  # 包含 InstanceNorm1d, 2d, 3d
    nn.RMSNorm,  # PyTorch 新版本原生支持的 RMSNorm
  )
  return isinstance(module, norm_classes)


def get_named_layers(module: nn.Module) -> list[tuple[str, nn.Module]]:
  """Get all layers in sequence.

  For ModuleList, extract the layers it contains.
  """
  named_layers = []
  for name, layer in module.named_children():
    # For ModuleList, recursively get its sub layers.
    if isinstance(layer, nn.ModuleList):
      named_layers.extend(get_named_layers(layer))
    else:
      named_layers.append((name, layer))
  return named_layers


def get_shard_layers(all_layers: list[nn.Module]) -> list[nn.Module]:
  return [layer for layer in all_layers if not is_norm_layer(layer)]


def get_local_layers(
  named_layers: list[tuple[str, nn.Module]], world_size: int, rank: int
):
  # layer to is_sharded
  local_layers: Mapping[nn.Module, bool] = {}
  layer_idx = 0

  for _, layer in named_layers:
    if is_norm_layer(layer):
      local_layers[layer] = False
    else:
      if layer_idx % world_size == rank:
        local_layers[layer] = True
        layer_idx += 1

  return local_layers


class FSDP(torch.nn.Module):
  """FSDP impl.

  Weights: each GPU stores an equal slice of every weight tensor.
    In FWD, the weights are sync layer by layer; the sync overlays with computation.
    In the end of BWD, the full weight are sharded back.
  Grads: each GPU stores an equal slice of grad corresponding to every weight tensor.
    In BWD, the grad are reduce-scatter layer by layer; the sync overlays with computation.
    In the beginning of next FWD, it is expected the caller will zero_grad().
  Optimizer state: the caller should maintain that. This FSDP impl works with either
    non-sharded optimizer or sharded optimizer. The optimizer state sharding is
    controlled by the optimizer.
  """

  def __init__(
    self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None
  ):
    """ctor.


    Args:
      module: the module to wrap.
      compute_dtype: if present, use this type to comm and compute.
        The master weight type is not affected.
    """
    super().__init__()

    # TODO(lizhi): support compute_dtype.

    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()

    self.module = module
    self.shard_params: set[nn.Parameter] = set()

    # Step 0 - get all layers that need to shard.
    all_named_layers = get_named_layers(self.module)
    all_layers = [p for (_, p) in all_named_layers]
    self.all_shard_layers = get_shard_layers(all_layers)
    self.all_shard_layers_to_idx = {}
    for i, layer in enumerate(self.all_shard_layers):
      self.all_shard_layers_to_idx[layer] = i

    # Step 1 - for every param, scatter it to all GPUs.
    for layer in self.all_shard_layers:
      for p in layer.parameters():
        chunks = list(p.detach().chunk(chunks=self.world_size, dim=0))
        # print(f"{len(chunks)=}")
        local_chunk = torch.empty_like(chunks[0])
        if self.rank == 0:
          dist.scatter(local_chunk, scatter_list=chunks, src=0)
        else:
          dist.scatter(local_chunk, scatter_list=None, src=0)
        self.shard_params.add(p)

        # Step 2 - delete params. Replace each param's full-size data with just
        # its local shard, so the rest of the original storage can be freed.
        p.data = local_chunk

    # Step 3 - get all layers for forward layer sync.
    for layer in self.all_shard_layers:
      layer.register_forward_pre_hook(self.fwd_params_sync_hook)

    for param in self.module.parameters():
      if param.requires_grad:
        param.register_post_accumulate_grad_hook(self.bwd_grads_sync_hook)

    # Step
    # local_layers = get_local_layers(
    #   layers, world_size=self.world_size, rank=self.rank
    # )
    # print(f"{local_layers=}")

  def forward(self, *inputs, **kwargs):
    # Sync the params for the first 2 layers.
    assert len(self.all_shard_layers) >= 2
    self._all_gather_params_for_layer(self.all_shard_layers[0])
    self._all_gather_params_for_layer(self.all_shard_layers[1])

    return self.module(*inputs, **kwargs)

  def parameters(self, recurse=True):
    return self.module.parameters(recurse)

  def finish_gradient_synchronization(self) -> None:
    """Release the parameters that don't belong to this GPU.

    Called after BWD and before optimizer.step().
    """
    for layer in self.all_shard_layers:
      for param in layer.parameters():
        chunks = param.data.detach().chunk(chunks=self.world_size, dim=0)
        param.data = chunks[self.rank]

  def _all_gather_params_for_layer(self, layer: nn.Module) -> None:
    # Sync for each parameter in the layer.
    for p in layer.parameters():
      assert p in self.shard_params, "non shared param is used in comm."
      local_chunk = p.data.detach()

      chunks = []
      for i in range(self.world_size):
        chunks.append(torch.empty_like(local_chunk))
      dist.all_gather(chunks, local_chunk)

      # Replace the parameter data with the synced tensor.
      p.data = torch.cat(chunks, dim=0)

  def fwd_params_sync_hook(self, module: nn.Module, input):
    if module not in self.all_shard_layers_to_idx:
      raise ValueError("fwd_params_sync_hook is triggered on non shard layer.")
    layer_idx = self.all_shard_layers_to_idx[module]

    # sync the params of the layer that is 2 layers ahead.
    comm_idx = layer_idx + 2
    if comm_idx < len(self.all_shard_layers):
      comm_layer = self.all_shard_layers[comm_idx]
      self._all_gather_params_for_layer(comm_layer)

  def bwd_grads_sync_hook(self, param: nn.Parameter) -> None:
    assert param.requires_grad
    chunks = list(param.grad.detach().chunk(chunks=self.world_size, dim=0))
    local_chunk = torch.empty_like(chunks[0])
    dist.reduce_scatter(
      output=local_chunk, input_list=chunks, op=dist.ReduceOp.AVG
    )
    param.grad.data = local_chunk

  def gather_full_params(self) -> dict[str, torch.Tensor]:
    """All-gather sharded params to reconstruct full parameter tensors.

    Replicated parameters are returned as-is."""
    ret = {}
    for name, param in self.module.named_parameters():
      if param not in self.shard_params:
        ret[name] = param.data
      else:
        chunks = []
        for _ in range(self.world_size):
          chunks.append(torch.empty_like(param.data))
        dist.all_gather(chunks, param.data.detach())
        ret[name] = torch.concat(chunks, dim=0)
    return ret


def distributed_train(
  rank: int,
  world_size: int,
  use_gpu: bool,
  model_config: dict[str, Any],
  batch_size: int,
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
    fsdp_m = FSDP(m)
    opt = optimizer.AdamW(fsdp_m.parameters())

    x = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)

    y = fsdp_m(x)
    loss = y.sum()

    loss.backward()
    fsdp_m.finish_gradient_synchronization()

    opt.step()

  finally:
    dist.destroy_process_group()


def main():
  print(f"{USE_GPU=} {WORLD_SIZE=} {MODEL_CONFIG_NAME=} {BATCH_SIZE=}")
  world_size = WORLD_SIZE
  mp.spawn(
    fn=distributed_train,
    args=(world_size, USE_GPU, MODEL_CONFIG, BATCH_SIZE),
    nprocs=world_size,
    join=True,
  )


if __name__ == "__main__":
  main()
