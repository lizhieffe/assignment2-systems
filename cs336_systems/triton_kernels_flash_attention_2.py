# Triton kernels.
#
# Run the code:
# uv run cs336_systems/triton_kernels_flash_attention_2.py

import einops
import jaxtyping
import math
import triton
import triton.language as tl
import torch

from cs336_systems import gpu_lib, profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def torch_plain_attention(
  Q: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
  K: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
  V: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
) -> jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"]:
  D = Q.shape[-1]
  S = Q @ K.transpose(-1, -2) / D**0.5  # (B, N, N)
  P = torch.softmax(S, dim=-1)  # (B, N, N)
  output = P @ V  # (B, N, D)
  return output


class TorchFlashAttention2Func(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx,
    Q: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
    K: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
    V: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
    is_causal: bool,
  ) -> tuple[
    jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
    jaxtyping.Float[torch.Tensor, "*LEADING_DIM"],
  ]:
    """
    Args:

    Returns:
      [0]: the output of softmax.
      [1]: the denominator of softmax.
    """
    assert Q.shape == K.shape
    assert Q.shape == V.shape
    assert len(Q.shape) >= 2

    input_shape = Q.shape

    # Convert to 2D for ease of processing.
    Q = einops.rearrange(Q, "... D -> (...) D")
    K = einops.rearrange(K, "... D -> (...) D")
    V = einops.rearrange(V, "... D -> (...) D")

    num_rows, D = Q.shape[0], Q.shape[-1]

    # Inheritate from weight sum kernel.
    ctx.Q_TILE_SIZE = (
      16  # Each thread block processes 16 batch elements at a time.
    )
    ctx.K_TILE_SIZE = (
      16  # Each thread block processes 16 batch elements at a time.
    )

    NUM_Q_TILES = triton.cdiv(num_rows, ctx.Q_TILE_SIZE)
    NUM_K_TILES = triton.cdiv(num_rows, ctx.K_TILE_SIZE)

    q_tiles = torch.split(Q, split_size_or_sections=ctx.Q_TILE_SIZE)
    k_tiles = torch.split(K, split_size_or_sections=ctx.K_TILE_SIZE)
    v_tiles = torch.split(V, split_size_or_sections=ctx.K_TILE_SIZE)

    o_tiles = []  # softmax results
    l_tiles = []  # denominator of softmax

    NUM_Q_TILES = len(q_tiles)
    NUM_K_TILES = len(k_tiles)
    NUM_V_TILES = NUM_K_TILES

    for i in range(NUM_Q_TILES):
      q_tile = q_tiles[i]  # (ctx.Q_TILE_SIZE, D)
      o_tile = torch.zeros_like(q_tile)  # (ctx.Q_TILE_SIZE, D)

      # l_tile is the running proxy of the softmax denominator.
      l_tile = torch.zeros(
        (ctx.Q_TILE_SIZE,), device=Q.device, dtype=Q.dtype
      )  # (ctx.Q_TILE_SIZE,)

      # m_tile is the running maxium.
      m_tile = torch.full(
        (ctx.Q_TILE_SIZE,), float("-inf"), device=Q.device, dtype=Q.dtype
      )  # (ctx.Q_TILE_SIZE,)

      for j in range(NUM_K_TILES):
        k_tile = k_tiles[j]  # (ctx.K_TILE_SIZE, D)
        v_tile = v_tiles[j]  # (ctx.K_TILE_SIZE, D)

        s = (
          q_tile @ k_tile.transpose(0, 1) / D**0.5
        )  # (ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE)
        m_tile_next = torch.maximum(
          m_tile, torch.max(s, dim=-1).values
        )  # (ctx.Q_TILE_SIZE,)
        p = torch.exp(
          s - m_tile_next[:, None]
        )  # (ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE)
        l_tile_next = torch.exp(m_tile - m_tile_next) * l_tile + torch.sum(
          p, dim=-1
        )  # (ctx.Q_TILE_SIZE,)

        # diag() is to scale the each row of o_tile by the corresponding element of
        # torch.exp(m_tile - m_tile_next)
        diag = torch.diag(
          torch.exp(m_tile - m_tile_next)
        )  # (ctx.Q_TILE_SIZE,ctx.Q_TILE_SIZE)
        o_tile_next = diag @ o_tile + p @ v_tile  # (ctx.Q_TILE_SIZE, D)

        # Update m, l & o
        m_tile = m_tile_next
        l_tile = l_tile_next
        o_tile = o_tile_next

      o_tiles.append(
        torch.inverse(torch.diag(l_tile)) @ o_tile
      )  # (ctx.Q_TILE_SIZE, D)
      l_tiles.append(m_tile + torch.log(l_tile))  # (ctx.Q_TILE_SIZE,)

    return torch.stack(o_tiles, dim=0).view(input_shape), torch.stack(
      l_tiles, dim=0
    ).view(input_shape[:-1])

  @staticmethod
  def backward(ctx, grad_out):
    raise NotImplementedError("backward pass not yet implemented")


def main() -> None:
  input_shape = (32, 64)
  Q = torch.rand(input_shape, dtype=torch.float32, device=DEVICE)
  K = torch.rand(input_shape, dtype=torch.float32, device=DEVICE)
  V = torch.rand(input_shape, dtype=torch.float32, device=DEVICE)

  forward_res, softmax_denominator = TorchFlashAttention2Func.apply(
    Q, K, V, False
  )
  assert forward_res.shape == input_shape
  assert softmax_denominator.shape == input_shape[:-1]

  expected_forward_res = torch_plain_attention(Q, K, V)
  assert torch.allclose(forward_res, expected_forward_res), (
    f"{forward_res=}, {expected_forward_res=}"
  )
  print("TorchFlashAttention2Func -> Passed!!!")


if __name__ == "__main__":
  main()
