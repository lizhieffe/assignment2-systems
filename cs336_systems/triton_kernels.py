# Triton kernels.
#
# Run the code:
# uv run cs336_systems/triton_kernels.py

import einops
import jaxtyping
import math
import triton
import triton.language as tl
import torch

from cs336_systems import gpu_lib, profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def weighted_sum(
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

  weighted_sum_kernel[(num_blocks,)](
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
def weighted_sum_kernel(
  x_ptr,
  weight_ptr,
  output_ptr,
  x_stride_row,
  x_stride_dim,  # Stride tells us how to move one element in each axis of a tensor.
  weight_stride_dim,
  output_stirde_row,
  NUM_ROWS,
  D,
  ROWS_TILE_SIZE: tl.constexpr,
  D_TILE_SIZE: tl.constexpr,
):
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


def main() -> None:
  x = torch.rand((8, 16, 32), dtype=torch.float32, device=DEVICE)
  print(f"{x.stride()=}")

  weight = torch.rand(32, dtype=torch.float32, device=DEVICE)
  result = weighted_sum(x, weight)
  expected = torch.sum(x * weight, dim=-1)
  assert result.shape == expected.shape, (
    f"shape mismatch: {result.shape} != {expected.shape}"
  )
  assert torch.allclose(result, expected), (
    "weighted_sum result does not match expected"
  )
  print("weighted_sum passed", result)


if __name__ == "__main__":
  main()
