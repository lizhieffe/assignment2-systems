# Assignment 7 - FSDP
#
# Run the code:
# uv run cs336_systems/assignment_7_fsdp_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from cs336_basics import model, optimizer

Linear = model.Linear
Embedding = model.Embedding


def get_world_size():
  return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
  return dist.get_rank() if dist.is_initialized() else 0


class FSDPFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, wrapper, input_tensor, weight, bias):
    # 1. Wait for pre-fetched weights if gathering asynchronously
    if wrapper.gather_work is not None:
      work_w, work_b = wrapper.gather_work
      if work_w is not None:
        work_w.wait()
      if work_b is not None:
        work_b.wait()
      wrapper.gather_work = None

    # 2. Fallback sync gather (if not pre-fetched)
    if wrapper.full_weight is None:
      wrapper._sync_gather()

    fw = wrapper.full_weight
    fb = wrapper.full_bias

    # Save variables for the backward pass
    ctx.wrapper = wrapper
    ctx.save_for_backward(input_tensor, weight, bias)

    # 3. Compute Forward using the compute_dtype
    if wrapper.original_module_class == Linear:
      dtype = wrapper.compute_dtype
      input_cast = input_tensor.to(dtype) if dtype is not None else input_tensor
      out = F.linear(input_cast, fw, fb)
    else:  # Embedding
      out = F.embedding(input_tensor, fw)

    # 4. CRITICAL: Free gathered memory immediately after compute
    wrapper.full_weight = None
    wrapper.full_bias = None

    return out

  @staticmethod
  def backward(ctx, grad_output):
    input_tensor, weight, bias = ctx.saved_tensors
    wrapper = ctx.wrapper
    dtype = wrapper.compute_dtype

    # 1. Synchronously gather weights for the backward pass
    wrapper._sync_gather()
    fw = wrapper.full_weight

    if dtype is not None:
      grad_output = grad_output.to(dtype)

    grad_input = grad_weight = grad_bias = None

    # 2. Compute gradients
    if wrapper.original_module_class == Linear:
      if ctx.needs_input_grad[1]:  # dJ/dx
        grad_input = grad_output @ fw
        if grad_input is not None:
          grad_input = grad_input.to(input_tensor.dtype)

      if ctx.needs_input_grad[2]:  # dJ/dW
        grad_out_flat = grad_output.reshape(-1, grad_output.shape[-1])
        input_cast = (
          input_tensor.to(dtype) if dtype is not None else input_tensor
        )
        input_flat = input_cast.reshape(-1, input_cast.shape[-1])

        # Gradient computed in compute_dtype, but cast to FP32 for accumulation
        grad_weight_full = (grad_out_flat.t() @ input_flat).to(torch.float32)
        grad_weight_full /= wrapper.world_size

        # Reduce-scatter asynchronously
        grad_weight = torch.empty_like(weight.data)
        if wrapper.world_size > 1:
          work_w = dist.reduce_scatter_tensor(
            grad_weight, grad_weight_full, async_op=True
          )
          # NOTE: grad_weight is returned to autograd below AND held here for
          # later .wait(). Autograd's AccumulateGrad clones any returned
          # gradient tensor that has other live references (refcount > 1),
          # so `weight.grad` ends up being a *different* tensor object than
          # `grad_weight`, frozen at clone time (before the async op below
          # has actually written real data into it). Waiting on `work_w`
          # later only updates `grad_weight`'s storage, not `weight.grad`'s
          # -- so we must explicitly copy the finished result into
          # `weight.grad` ourselves once the wait completes, rather than
          # relying on tensor-object identity.
          wrapper.fsdp_parent.gradient_sync_works.append(
            (work_w, grad_weight_full, grad_weight, weight)
          )
        else:
          grad_weight.copy_(grad_weight_full)

      if bias is not None and ctx.needs_input_grad[3]:  # dJ/db
        grad_out_flat = grad_output.reshape(-1, grad_output.shape[-1])
        grad_bias_full = grad_out_flat.sum(dim=0).to(torch.float32)
        grad_bias_full /= wrapper.world_size

        grad_bias = torch.empty_like(bias.data)
        if wrapper.world_size > 1:
          work_b = dist.reduce_scatter_tensor(
            grad_bias, grad_bias_full, async_op=True
          )
          wrapper.fsdp_parent.gradient_sync_works.append(
            (work_b, grad_bias_full, grad_bias, bias)
          )
        else:
          grad_bias.copy_(grad_bias_full)

    elif wrapper.original_module_class == Embedding:
      if ctx.needs_input_grad[2]:  # dJ/dW
        grad_out_flat = grad_output.reshape(-1, grad_output.shape[-1]).to(
          torch.float32
        )
        input_flat = input_tensor.reshape(-1)

        # Dense manual accumulation for embeddings to prepare for reduce_scatter
        grad_weight_full = torch.zeros(
          fw.shape, dtype=torch.float32, device=fw.device
        )
        grad_weight_full.index_add_(0, input_flat, grad_out_flat)
        grad_weight_full /= wrapper.world_size

        grad_weight = torch.empty_like(weight.data)
        if wrapper.world_size > 1:
          work_w = dist.reduce_scatter_tensor(
            grad_weight, grad_weight_full, async_op=True
          )
          wrapper.fsdp_parent.gradient_sync_works.append(
            (work_w, grad_weight_full, grad_weight, weight)
          )
        else:
          grad_weight.copy_(grad_weight_full)

    # 3. CRITICAL: Free gathered memory again
    wrapper.full_weight = None
    wrapper.full_bias = None

    # ctx.needs_input_grad maps to: (wrapper, input_tensor, weight, bias)
    return None, grad_input, grad_weight, grad_bias


class _FSDPShardedMixin:
  """Shared sharding/gather/forward logic for the Sharded* wrapper classes.

  Not an nn.Module itself: each concrete wrapper subclasses BOTH this mixin
  AND the original Linear/Embedding class, so isinstance(wrapper, Linear) /
  isinstance(wrapper, Embedding) keeps holding for callers (including
  test_fsdp.py's gradient-sync check, which relies on that to tell sharded
  layers apart from replicated ones) -- the layer is genuinely replaced in
  the module tree with a same-type-family object, not swapped for an
  unrelated wrapper class.
  """

  def _init_sharding(self, original_module, fsdp_parent, index, compute_dtype):
    self.original_module_class = type(original_module)
    # Plain attribute assignment would make nn.Module.__setattr__ register
    # fsdp_parent (the whole FSDP instance, including the rest of the model)
    # as a submodule of this layer -- creating a reference cycle and leaking
    # every other layer's params (including replicated ones like RMSNorm)
    # into this layer's .parameters(). Bypass that registration explicitly.
    object.__setattr__(self, "fsdp_parent", fsdp_parent)
    self.index = index
    self.compute_dtype = compute_dtype

    self.rank = get_rank()
    self.world_size = get_world_size()

    chunks = original_module.weight.data.chunk(chunks=self.world_size, dim=0)
    self.weight = nn.Parameter(chunks[self.rank].clone())
    self.register_parameter("bias", None)

    self.full_weight = None
    self.full_bias = None
    self.gather_work = None

  def gather_async(self):
    if self.full_weight is not None or self.gather_work is not None:
      return

    # Cast to compute_dtype BEFORE communicating to save bandwidth
    comm_dtype = (
      self.compute_dtype
      if self.compute_dtype is not None
      else self.weight.dtype
    )
    weight_shard = self.weight.data.to(comm_dtype)

    if self.world_size == 1:
      self.full_weight = weight_shard
      self.full_bias = (
        self.bias.data.to(comm_dtype) if self.bias is not None else None
      )
      self.gather_work = (None, None)
      return

    weight_full_shape = [self.world_size * self.weight.shape[0]] + list(
      self.weight.shape[1:]
    )
    self.full_weight = torch.empty(
      weight_full_shape, dtype=comm_dtype, device=self.weight.device
    )
    work_w = dist.all_gather_into_tensor(
      self.full_weight, weight_shard, async_op=True
    )

    if self.bias is not None:
      bias_shard = self.bias.data.to(comm_dtype)
      bias_full_shape = [self.world_size * self.bias.shape[0]]
      self.full_bias = torch.empty(
        bias_full_shape, dtype=comm_dtype, device=self.bias.device
      )
      work_b = dist.all_gather_into_tensor(
        self.full_bias, bias_shard, async_op=True
      )
    else:
      work_b = None

    self.gather_work = (work_w, work_b)

  def _sync_gather(self):
    if self.full_weight is not None:
      return

    comm_dtype = (
      self.compute_dtype
      if self.compute_dtype is not None
      else self.weight.dtype
    )
    weight_shard = self.weight.data.to(comm_dtype)

    if self.world_size == 1:
      self.full_weight = weight_shard
      self.full_bias = (
        self.bias.data.to(comm_dtype) if self.bias is not None else None
      )
      return

    weight_full_shape = [self.world_size * self.weight.shape[0]] + list(
      self.weight.shape[1:]
    )
    self.full_weight = torch.empty(
      weight_full_shape, dtype=comm_dtype, device=self.weight.device
    )
    dist.all_gather_into_tensor(self.full_weight, weight_shard, async_op=False)

    if self.bias is not None:
      bias_shard = self.bias.data.to(comm_dtype)
      bias_full_shape = [self.world_size * self.bias.shape[0]]
      self.full_bias = torch.empty(
        bias_full_shape, dtype=comm_dtype, device=self.bias.device
      )
      dist.all_gather_into_tensor(self.full_bias, bias_shard, async_op=False)

  def forward(self, input_tensor):
    # 1. Execute math and autograd tracking using our custom FSDPFunction
    out = FSDPFunction.apply(
      self, input_tensor, self.weight, getattr(self, "bias", None)
    )

    # 2. Prefetch the layer exactly 2 steps ahead. By scheduling layer (i + 2) right
    # after layer i concludes, we strictly satisfy the memory limiting requirement.
    next_idx = self.index + 2
    if next_idx < len(self.fsdp_parent.sharded_layers):
      self.fsdp_parent.sharded_layers[next_idx].gather_async()

    return out


class FSDPLinearWrapper(_FSDPShardedMixin, Linear):
  """A model.Linear whose weight is permanently just the local shard.

  _FSDPShardedMixin MUST come first in the base list: Linear defines its own
  `forward` (the unsharded einsum), and Python's MRO resolves left-to-right,
  so if Linear came first its `forward` would silently win over the mixin's
  -- which is exactly what happened before this fix (it ran the unsharded
  Embedding.forward on a weight that was already cut down to one shard,
  producing out-of-bounds indexing).
  """

  def __init__(self, original_module, fsdp_parent, index, compute_dtype):
    nn.Module.__init__(self)  # skip Linear.__init__'s full-size weight init
    self._init_sharding(original_module, fsdp_parent, index, compute_dtype)


class FSDPEmbeddingWrapper(_FSDPShardedMixin, Embedding):
  """A model.Embedding whose weight is permanently just the local (vocab-dim) shard.

  See FSDPLinearWrapper's docstring: _FSDPShardedMixin must come first.
  """

  def __init__(self, original_module, fsdp_parent, index, compute_dtype):
    nn.Module.__init__(self)  # skip Embedding.__init__'s full-size weight init
    self._init_sharding(original_module, fsdp_parent, index, compute_dtype)


class FSDP(nn.Module):
  def __init__(
    self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None
  ):
    super().__init__()
    self.module = module
    self.compute_dtype = compute_dtype

    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()

    self.gradient_sync_works = []

    # Track replaced layers to orchestrate exact execution ordering/prefetching
    self.sharded_layers = []

    self.shard_params: set[nn.Parameter] = set()

    # Recursively parse and wrap Linear and Embedding; skip LayerNorms
    def _replace_modules(mod):
      for name, child in mod.named_children():
        if isinstance(child, Linear):
          wrapper = FSDPLinearWrapper(
            child, self, len(self.sharded_layers), compute_dtype
          )
          setattr(mod, name, wrapper)
          self.sharded_layers.append(wrapper)
        elif isinstance(child, Embedding):
          wrapper = FSDPEmbeddingWrapper(
            child, self, len(self.sharded_layers), compute_dtype
          )
          setattr(mod, name, wrapper)
          self.sharded_layers.append(wrapper)
        else:
          _replace_modules(child)

    _replace_modules(self.module)

    for layer in self.sharded_layers:
      self.shard_params.update(layer.parameters())

    # Replicated params (RMSNorm, etc. -- anything not wrapped above) see
    # different data per rank, so their gradients must be averaged across
    # ranks. Without this, each rank silently keeps its own locally-computed
    # gradient and the ranks' models drift apart step over step.
    for param in self.module.parameters():
      if param.requires_grad and param not in self.shard_params:
        param.register_post_accumulate_grad_hook(
          self._replicated_grad_sync_hook
        )

  def _replicated_grad_sync_hook(self, param: nn.Parameter) -> None:
    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

  def forward(self, *inputs, **kwargs):
    # Kick off gathering for the first two layers before data flows through the model
    if len(self.sharded_layers) > 0:
      self.sharded_layers[0].gather_async()
    if len(self.sharded_layers) > 1:
      self.sharded_layers[1].gather_async()

    return self.module(*inputs, **kwargs)

  def finish_gradient_synchronization(self):
    # Waits for all pending asynchronous reduce-scatter calls to finish, then
    # explicitly writes the finished result into param.grad. We can't rely on
    # param.grad already being the same tensor object we reduce-scattered
    # into: autograd's AccumulateGrad clones any returned gradient tensor
    # that still has other live references (it did here, since we also hold
    # one in this list for the .wait() below), so param.grad is a separate
    # clone taken before the async op finished writing real data.
    for work, _grad_full, grad_local, param in self.gradient_sync_works:
      if work is not None:
        work.wait()
      if param.grad is None:
        param.grad = grad_local
      else:
        param.grad.copy_(grad_local)
    # Clean up the references to the intermediate tensors
    self.gradient_sync_works = []

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
