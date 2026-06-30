from __future__ import annotations

import torch

from cs336_systems import (
  assignment_52_naive_ddp,
  assignment_53_overlapped_ddp,
  assignment_6_zero1,
  assignment_7_fsdp_v2 as assignment_7_fsdp,
  triton_kernels_flash_attention_2,
)


def get_flashattention_autograd_function_pytorch() -> type:
  """
  Returns a torch.autograd.Function subclass that implements FlashAttention2.
  The expectation is that this class will implement FlashAttention2
  using only standard PyTorch operations (no Triton!).

  Returns:
      A class object (not an instance of the class)
  """
  # For example: return MyFlashAttnAutogradFunctionClass
  return triton_kernels_flash_attention_2.TorchFlashAttention2Func


def get_flashattention_autograd_function_triton() -> type:
  """
  Returns a torch.autograd.Function subclass that implements FlashAttention2
  using Triton kernels.
  The expectation is that this class will implement the same operations
  as the class you return in get_flashattention_autograd_function_pytorch(),
  but it should do so by invoking custom Triton kernels in the forward
  and backward passes.

  Returns:
      A class object (not an instance of the class)
  """
  # For example: return MyTritonFlashAttentionAutogradFunctionClass
  return triton_kernels_flash_attention_2.TritonFlashAttention2Func


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
  """
  Returns a torch.nn.Module container that handles
  parameter broadcasting and gradient synchronization for
  distributed data parallel training.

  This container should overlaps communication with backprop computation
  by asynchronously communicating gradients as they are ready
  in the backward pass. The gradient for each parameter tensor
  is individually communicated.

  Args:
      module: torch.nn.Module
          Underlying model to wrap with DDP.
  Returns:
      Instance of a DDP class.
  """
  # For example: return DDP(module)
  return assignment_53_overlapped_ddp.DDP(module)


def ddp_on_after_backward(
  ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
  """
  Code to run after the backward pass is completed, but before we take
  an optimizer step.

  Args:
      ddp_model: torch.nn.Module
          DDP-wrapped model.
      optimizer: torch.optim.Optimizer
          Optimizer being used with the DDP-wrapped model.
  """
  ddp_model.finish_gradient_synchronization()


def get_fsdp(
  module: torch.nn.Module, compute_dtype: torch.dtype | None = None
) -> torch.nn.Module:
  """
  Returns a torch.nn.Module container that handles
  fully-sharded data parallel training, including weight sharding,
  all-gather for forward/backward, and gradient reduce-scatter.

  Args:
      module: torch.nn.Module
          Underlying model to wrap with FSDP.
      compute_dtype: optional torch.dtype
          If provided, weights are cast to this dtype before communication
          and compute, saving bandwidth. Master weights stay in fp32.
  Returns:
      Instance of an FSDP class.
  """
  # For example: return FSDP(module, compute_dtype=compute_dtype)
  return assignment_7_fsdp.FSDP(module=module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(
  fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
  """
  Code to run after the backward pass is completed, but before we take
  an optimizer step.

  Args:
      fsdp_model: torch.nn.Module
          FSDP-wrapped model.
      optimizer: torch.optim.Optimizer
          Optimizer being used with the FSDP-wrapped model.
  """
  fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(
  fsdp_model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
  """
  All-gather sharded parameters from the FSDP model to reconstruct full
  parameter tensors. Replicated parameters are returned as-is.

  Args:
      fsdp_model: torch.nn.Module
          FSDP-wrapped model.
  Returns:
      State dictionary mapping parameter names to full (unsharded) tensors.
  """
  return fsdp_model.gather_full_params()


def get_sharded_optimizer(
  params, optimizer_cls: type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
  """
  Returns a torch.optim.Optimizer that handles optimizer state sharding
  of the given optimizer_cls on the provided parameters.

  Arguments:
      params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
          or :class:`dict` s giving all parameters, which will be sharded
          across ranks.
      optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
          optimizer.
  Keyword arguments:
      kwargs: keyword arguments to be forwarded to the optimizer constructor.
  Returns:
      Instance of sharded optimizer.
  """
  return assignment_6_zero1.ZeRO1(
    params=params, optimizer_cls=optimizer_cls, **kwargs
  )
