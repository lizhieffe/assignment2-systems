# Triton kernels.
#
# Run the code:
# uv run cs336_systems/triton_kernels_weight_sum.py

import einops
import jaxtyping
import math
import triton
import triton.language as tl
import torch

from cs336_systems import gpu_lib, profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def weighted_sum_forward(
  x: jaxtyping.Float[torch.Tensor, "... d"],
  weight: jaxtyping.Float[torch.Tensor, "d"],
) -> jaxtyping.Float[torch.Tensor, "..."]:
  """CPU wrapper."""
  assert x.is_cuda
  assert x.is_contiguous

  rows_tile_size = 1024

  # Flat the multi-dim x into 2-dim.
  # The weight sum will happen on the last dim.
  flat_x = einops.rearrange(x, "... d -> (...) d")

  num_rows = flat_x.shape[0]
  num_d = weight.numel()
  num_blocks = math.ceil(num_rows / rows_tile_size)
  x_stride_row = num_d

  # Sum over the last dim or the output doesn't have the last dimension.
  output = torch.empty(num_rows, dtype=x.dtype, device=x.device)

  weighted_sum_forward_kernel[(num_blocks,)](
    x,
    weight,
    output,
    x_stride_row=x_stride_row,
    x_stride_dim=1,
    weight_stride_dim=1,
    output_stirde_row=1,
    NUM_ROWS=num_rows,
    D=num_d,
    ROWS_TILE_SIZE=rows_tile_size,
    D_TILE_SIZE=32,
  )

  return output.view(x.shape[:-1])


@triton.jit
def weighted_sum_forward_kernel(
  x_ptr,
  weight_ptr,  # (N, D)
  output_ptr,  # (D)
  x_stride_row,
  x_stride_dim,  # Stride tells us how to move one element in each axis of a tensor.
  weight_stride_dim,
  output_stirde_row,
  NUM_ROWS,  # Num of rows for the parent tensor.
  D,  # Num of columns (embedding dim) for the parent tensor.
  ROWS_TILE_SIZE: tl.constexpr,
  D_TILE_SIZE: tl.constexpr,
):
  """Note: x_ptr input tensor needs to be 2D."""
  # Each thread handles the weighted sum of an **entire** row.
  # Each thread block handles ROWS_TILE_SIZE rows.
  row_tile_idx = tl.program_id(axis=0)

  # Block pointers give us a way to select from an ND region of memory
  # and move our selection around.
  #
  # order: the order of the dimensions in memory from major to minor axes.
  x_block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(NUM_ROWS, D),
    strides=(x_stride_row, x_stride_dim),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(0, 1),
  )

  weight_block_ptr = tl.make_block_ptr(
    base=weight_ptr,
    shape=(D,),
    strides=(weight_stride_dim,),
    offsets=(0,),
    block_shape=(D_TILE_SIZE,),
    order=(0,),
  )

  output_block_ptr = tl.make_block_ptr(
    base=output_ptr,
    shape=(NUM_ROWS,),
    strides=(output_stirde_row,),
    offsets=(row_tile_idx * ROWS_TILE_SIZE,),
    block_shape=(ROWS_TILE_SIZE,),
    order=(0,),
  )

  # Initialize a buffer to write to
  output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)  # (ROWS_TILE_SIZE,)

  for i in range(tl.cdiv(D, D_TILE_SIZE)):
    # Load a block from x.
    row = tl.load(
      x_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )  # (ROW_TILE_SIZE, D_TILE_SIZE)

    weight = tl.load(
      weight_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (D_TILE_SIZE,)

    output += tl.sum(row * weight[None, :], axis=1)

    # Move the block ptr along the D direction.
    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
    weight_block_ptr = weight_block_ptr.advance(
      (D_TILE_SIZE,),
    )

  tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward_kernel(
  x_ptr,  # Input. (NUM_ROWS, D)
  weight_ptr,  # Input. (D,)
  grad_output_ptr,  # Grad Input. (NUM_ROWS,)
  grad_x_ptr,  # Grad output. (NUM_ROWS, D)
  partial_grad_weight_ptr,  # Grad output. (n_row_tiles, D)
  stride_xr,
  stride_xd,
  stride_wd,
  stride_gr,
  stride_gxr,
  stride_gxd,
  stride_gwb,
  stride_gwd,
  NUM_ROWS,
  D,
  ROWS_TILE_SIZE: tl.constexpr,
  D_TILE_SIZE: tl.constexpr,
):
  row_tile_idx = tl.program_id(axis=0)
  n_row_tiles = tl.cdiv(NUM_ROWS, ROWS_TILE_SIZE)
  n_d_tiles = tl.cdiv(D, D_TILE_SIZE)

  # TODO(lizhi): I think having each thread to handle one D is more efficient.
  # But here we follow the instruction to have each thread to handle one ROW.
  x_block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(NUM_ROWS, D),
    strides=(stride_xr, stride_xd),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(0, 1),
  )
  weight_block_ptr = tl.make_block_ptr(
    base=weight_ptr,
    shape=(D,),
    strides=(stride_wd,),
    offsets=(0,),
    block_shape=(D_TILE_SIZE,),
    order=(0,),
  )
  grad_output_block_ptr = tl.make_block_ptr(
    base=grad_output_ptr,
    shape=(NUM_ROWS,),
    strides=(stride_gr,),
    offsets=(row_tile_idx * ROWS_TILE_SIZE,),
    block_shape=(ROWS_TILE_SIZE,),
    order=(0,),
  )
  grad_x_block_ptr = tl.make_block_ptr(
    base=grad_x_ptr,
    shape=(NUM_ROWS, D),
    strides=(stride_gxr, stride_gxd),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(0, 1),
  )
  partial_grad_weight_block_ptr = tl.make_block_ptr(
    base=partial_grad_weight_ptr,
    shape=(n_row_tiles, D),
    strides=(stride_gwb, stride_gwd),
    offsets=(row_tile_idx, 0),
    block_shape=(1, D_TILE_SIZE),
    order=(0, 1),
  )

  for i in range(n_d_tiles):
    grad_output = tl.load(
      grad_output_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (ROWS_TILE_SIZE,)

    # Calculate the grad for x
    weight = tl.load(
      weight_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (D_TILE_SIZE,)
    grad_x_row = (
      grad_output[:, None] * weight[None, :]
    )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
    tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

    # Calculate the grad for w
    row = tl.load(
      x_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
    grad_weight_row = tl.sum(
      grad_output[:, None] * row,
      axis=0,
    )[None, :]  # (1, D_TILE_SIZE)
    tl.store(
      partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)
    )

    # Advance the blocks.
    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
    weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
    partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance(
      (0, D_TILE_SIZE)
    )


class WeightedSumFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight):
    input_shape = x.shape

    D, output_dim = (
      input_shape[-1],
      input_shape[:-1],
    )  # D is the embedding dimension.

    # Reshape x because the kernel operates on 2-D tensors.
    x = einops.rearrange(x, "... D -> (...) D")

    ctx.save_for_backward(x, weight)

    assert len(weight.shape) == 1 and D == weight.shape[0], (
      "Dimension mismatch."
    )
    assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous(), "X is not contiguous"

    ctx.D_TILE_SIZE = (
      triton.next_power_of_2(D) // 16
    )  # Rougly 16 loops through the embedding dimension
    ctx.ROW_TILE_SIZE = (
      16  # Each thread block processes 16 batch elements at a time.
    )
    ctx.input_shape = input_shape

    num_rows = x.shape[0]
    num_blocks = triton.cdiv(num_rows, ctx.ROW_TILE_SIZE)
    y = torch.empty((num_rows,), device=x.device)

    weighted_sum_forward_kernel[(num_blocks,)](
      x,
      weight,
      y,
      x_stride_row=x.stride()[0],
      x_stride_dim=x.stride()[1],
      weight_stride_dim=weight.stride()[0],
      output_stirde_row=y.stride()[0],
      NUM_ROWS=num_rows,
      D=D,
      ROWS_TILE_SIZE=ctx.ROW_TILE_SIZE,
      D_TILE_SIZE=ctx.D_TILE_SIZE,
    )
    return y.view(output_dim)

  @staticmethod
  def backward(ctx, grad_out):
    x, weight = ctx.saved_tensors

    assert len(x.shape) == 2, "Wrong x dim"
    assert len(weight.shape) == 1, "Wrong weight dim"
    assert len(grad_out.shape) == len(ctx.input_shape[:-1]), (
      "Wrong grad_out dim"
    )
    assert ctx.input_shape[:-1] == grad_out.shape, (
      "Mismatched x & grad_out dimensions."
    )
    assert x.shape[1] == weight.shape[0], "Mismatched x & weight dimensions."

    assert grad_out.device == x.device, "Device mismatch between grad_out & x"
    assert grad_out.device == weight.device, (
      "Device mismatch between grad_out & weight"
    )

    grad_out = einops.rearrange(grad_out, "... -> (...)")
    assert x.shape[0] == grad_out.shape[0], "Wrong dim"

    num_rows = x.shape[0]
    D = x.shape[1]

    num_blocks = triton.cdiv(num_rows, ctx.ROW_TILE_SIZE)

    grad_x = torch.empty_like(x)
    partial_grad_weight = torch.empty(
      (num_blocks, D), dtype=x.dtype, device=x.device
    )

    weighted_sum_backward_kernel[(num_blocks,)](
      x_ptr=x,
      weight_ptr=weight,
      grad_output_ptr=grad_out,
      grad_x_ptr=grad_x,
      partial_grad_weight_ptr=partial_grad_weight,
      stride_xr=x.stride(0),
      stride_xd=x.stride(1),
      stride_wd=weight.stride(0),
      stride_gr=grad_out.stride(0),
      stride_gxr=grad_x.stride(0),
      stride_gxd=grad_x.stride(1),
      stride_gwb=partial_grad_weight.stride(0),
      stride_gwd=partial_grad_weight.stride(1),
      NUM_ROWS=num_rows,
      D=D,
      ROWS_TILE_SIZE=ctx.ROW_TILE_SIZE,
      D_TILE_SIZE=ctx.D_TILE_SIZE,
    )

    return grad_x.view(ctx.input_shape), partial_grad_weight.sum(axis=0)


def main() -> None:
  x = torch.rand((8, 16, 32), dtype=torch.float32, device=DEVICE)
  weight = torch.rand(32, dtype=torch.float32, device=DEVICE)
  x.requires_grad_()
  weight.requires_grad_()
  print(f"{x.shape=}, {weight.shape=}")
  print(f"{x.stride()=}, {weight.stride()=}")

  x_dup = x.detach().clone()
  weight_dup = weight.detach().clone()
  x_dup.requires_grad_()
  weight_dup.requires_grad_()

  print("======================= The forward fn =========================")
  result = WeightedSumFunc.apply(x, weight)
  print(f"{result=}")

  expected = torch.sum(x_dup * weight_dup, dim=-1)
  assert result.shape == expected.shape, (
    f"shape mismatch: {result.shape} != {expected.shape}"
  )
  assert torch.allclose(result, expected), (
    "weighted_sum result does not match expected"
  )
  print(f"weighted_sum forward passed!!! {result.shape=}")

  print("======================= The backward fn =========================")
  loss = result.sum()
  loss.backward()
  print(f"Backward result {x.grad.shape=}, {weight.grad.shape=}")

  expected_loss = expected.sum()
  assert torch.allclose(loss, expected_loss), f"{loss=}, {expected_loss=}"
  expected_loss.backward()
  assert torch.allclose(x.grad, x_dup.grad), f"{x.grad=}, {x_dup.grad=}"
  assert torch.allclose(weight.grad, weight_dup.grad), (
    f"{weight.grad=}, {weight_dup.grad=}"
  )
  print(
    f"weighted_sum backward passed!!! {x.grad.shape=}, {weight.grad.shape=}"
  )


if __name__ == "__main__":
  main()
