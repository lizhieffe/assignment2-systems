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
  ) -> jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"]:
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

    # Add batch dim if it is not there.
    if len(Q.shape) == 2:
      Q = einops.rearrange(Q, "... -> () ...")
      K = einops.rearrange(K, "... -> () ...")
      V = einops.rearrange(V, "... -> () ...")

    # Convert to 2D for ease of processing.
    Q = einops.rearrange(Q, "... N D -> (...) N D")
    K = einops.rearrange(K, "... N D -> (...) N D")
    V = einops.rearrange(V, "... N D -> (...) N D")

    B, num_rows, D = Q.shape

    # Inheritate from weight sum kernel.
    ctx.Q_TILE_SIZE = (
      16  # Each thread block processes 16 batch elements at a time.
    )
    ctx.K_TILE_SIZE = (
      16  # Each thread block processes 16 batch elements at a time.
    )

    NUM_Q_TILES = triton.cdiv(num_rows, ctx.Q_TILE_SIZE)
    NUM_K_TILES = triton.cdiv(num_rows, ctx.K_TILE_SIZE)

    o_tiles = []  # softmax results
    l_tiles = []  # denominator of softmax

    for bi in range(B):  # The outer loop is for (batch * head) dim.
      q_tiles = torch.split(Q[bi], split_size_or_sections=ctx.Q_TILE_SIZE)
      k_tiles = torch.split(K[bi], split_size_or_sections=ctx.K_TILE_SIZE)
      v_tiles = torch.split(V[bi], split_size_or_sections=ctx.K_TILE_SIZE)

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
          )  # (ctx.Q_TILE_SIZE,ctx. Q_TILE_SIZE)
          o_tile_next = diag @ o_tile + p @ v_tile  # (ctx.Q_TILE_SIZE, D)

          # Update m, l & o
          m_tile = m_tile_next
          l_tile = l_tile_next
          o_tile = o_tile_next

        o_tiles.append(
          torch.inverse(torch.diag(l_tile)) @ o_tile
        )  # (ctx.Q_TILE_SIZE, D)
        l_tiles.append(m_tile + torch.log(l_tile))  # (ctx.Q_TILE_SIZE,)

    output = torch.stack(o_tiles, dim=0)  # (NUM_Q_TILES, ctx.Q_TILE_SIZE, D)
    output = output.view(input_shape)

    softmax_denominator = torch.stack(
      l_tiles, dim=0
    )  # (NUM_Q_TILES, ctx.Q_TILE_SIZE)
    softmax_denominator = softmax_denominator.view(input_shape[:-1])

    ctx.save_for_backward(softmax_denominator, Q, K, V, output)

    return output

  @staticmethod
  def backward(ctx, grad_out):
    raise NotImplementedError("backward pass not yet implemented")


@triton.jit
def flash_fwd_kernel(
  Q_ptr,  # (B, Q, D)
  K_ptr,  # (B, K, D)
  V_ptr,  # (B, K, D)
  O_ptr,  # (B, Q, D)
  L_ptr,  # (B, Q)
  stride_qb,
  stride_qq,
  stride_qd,
  stride_kb,
  stride_kk,
  stride_kd,
  stride_vb,
  stride_vk,
  stride_vd,
  stride_ob,
  stride_oq,
  stride_od,
  stride_lb,
  stride_lq,
  N_QUERIES,
  N_KEYS,
  scale,  # 1 / sqrt(D)
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr,
):
  query_tile_idx = tl.program_id(axis=0)
  batch_idx = tl.program_id(axis=1)

  # Absolute query positions handled by this program (fixed across the K loop).
  q_offsets = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

  Q_block_ptr = tl.make_block_ptr(
    base=Q_ptr + batch_idx * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(0, 1),
  )
  # Load K transposed as (D, N_KEYS) so we can do Q @ K^T without using .T
  K_block_ptr = tl.make_block_ptr(
    base=K_ptr + batch_idx * stride_kb,
    shape=(D, N_KEYS),
    strides=(stride_kd, stride_kk),
    offsets=(0, 0),
    block_shape=(D, K_TILE_SIZE),
    order=(0, 1),
  )
  V_block_ptr = tl.make_block_ptr(
    base=V_ptr + batch_idx * stride_vb,
    shape=(N_KEYS, D),
    strides=(stride_vk, stride_vd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(0, 1),
  )
  O_block_ptr = tl.make_block_ptr(
    base=O_ptr + batch_idx * stride_ob,
    shape=(N_QUERIES, D),
    strides=(stride_oq, stride_od),
    offsets=(query_tile_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(0, 1),
  )
  L_block_ptr = tl.make_block_ptr(
    base=L_ptr + batch_idx * stride_lb,
    shape=(N_QUERIES,),
    strides=(stride_lq,),
    offsets=(query_tile_idx * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,),
  )

  Q_i = tl.load(
    Q_block_ptr,
    boundary_check=(
      0,
    ),  # boundary check the dim 0 only becuase it never goes out of boundary on dim 1
    padding_option="zero",
  )  # (Q_TILE_SIZE, D)
  O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (Q_TILE_SIZE, D)
  l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (Q_TILE_SIZE,)
  m_i = tl.full(
    (Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32
  )  # (Q_TILE_SIZE,)

  for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
    K_j = tl.load(
      K_block_ptr,
      boundary_check=(1,),  # boundary check dim 1 (N_KEYS direction)
      padding_option="zero",
    )  # (D, K_TILE_SIZE)

    V_j = tl.load(
      V_block_ptr,
      boundary_check=(0,),
      padding_option="zero",
    )  # (K_TILE_SIZE, D)

    S_i = tl.dot(Q_i, K_j) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

    m_i_next = tl.maximum(m_i, tl.max(S_i, axis=1))  # (Q_TILE_SIZE,)
    P_i = tl.exp(S_i - m_i_next[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

    l_i_next = tl.exp(m_i - m_i_next) * l_i + tl.sum(
      P_i, axis=1
    )  # (Q_TILE_SIZE,)
    O_i_next = tl.exp(m_i - m_i_next)[:, None] * O_i + tl.dot(
      P_i.to(tl.float32),
      V_j.to(tl.float32),
    )  # (Q_TILE_SIZE, D)

    # Update the running values.
    m_i = m_i_next
    l_i = l_i_next
    O_i = O_i_next

    # Advance block pointers. advance() returns a NEW pointer; it does not
    # mutate in place, so the result must be reassigned.
    K_block_ptr = K_block_ptr.advance(
      (0, K_TILE_SIZE)
    )  # advance in the N_KEYS direction (dim 1 of K^T)
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

  O_i = O_i / l_i[:, None]  # (Q_TILE_SIZE, D)
  l_i = m_i + tl.log(l_i)  # (Q_TILE_SIZE,)

  tl.store(
    O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0,)
  )
  tl.store(
    L_block_ptr, l_i.to(L_block_ptr.type.element_ty), boundary_check=(0,)
  )


class TritonFlashAttention2Func(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx,
    Q: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
    K: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
    V: jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"],
    is_causal: bool,
  ) -> jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"]:
    assert len(Q.shape) == 3
    assert len(Q.shape) == len(K.shape)
    assert len(Q.shape) == len(V.shape)
    assert K.shape == V.shape

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

    N_BATCHES, N_QUERIES, D = Q.shape
    N_KEYS = K.shape[-2]
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    num_query_tiles = math.ceil(N_QUERIES / Q_TILE_SIZE)

    output = torch.empty_like(Q)
    softmax_denominator = torch.empty(
      (N_BATCHES, N_QUERIES), device=Q.device, dtype=Q.dtype
    )

    flash_fwd_kernel[(num_query_tiles, N_BATCHES)](
      Q,
      K,
      V,
      output,
      softmax_denominator,
      Q.stride()[0],
      Q.stride()[1],
      Q.stride()[2],
      K.stride()[0],
      K.stride()[1],
      K.stride()[2],
      V.stride()[0],
      V.stride()[1],
      V.stride()[2],
      output.stride()[0],
      output.stride()[1],
      output.stride()[2],
      softmax_denominator.stride()[0],
      softmax_denominator.stride()[1],
      N_QUERIES,
      N_KEYS,
      1 / math.sqrt(D),
      D,
      Q_TILE_SIZE,
      K_TILE_SIZE,
    )

    ctx.save_for_backward(softmax_denominator, Q, K, V, output)

    return output


def main() -> None:
  input_shape = (32, 128, 256)
  Q = torch.rand(input_shape, dtype=torch.float32, device=DEVICE)
  K = torch.rand(input_shape, dtype=torch.float32, device=DEVICE)
  V = torch.rand(input_shape, dtype=torch.float32, device=DEVICE)

  expected_forward_res = torch_plain_attention(Q, K, V)

  # Assert torch Flash Attention
  torch_forward_res = TorchFlashAttention2Func.apply(Q, K, V, False)
  assert torch_forward_res.shape == input_shape
  assert torch.allclose(torch_forward_res, expected_forward_res), (
    f"{torch_forward_res=}, {expected_forward_res=}"
  )
  print("TorchFlashAttention2Func -> Passed!!!")

  # Assert torch Flash Attention
  triton_forward_res = TritonFlashAttention2Func.apply(Q, K, V, False)
  assert triton_forward_res.shape == input_shape
  # The tolerance is the same as _test_flash_forward_pass.
  assert torch.allclose(
    triton_forward_res, expected_forward_res, rtol=1e-2, atol=1e-2
  ), f"{triton_forward_res=}, {expected_forward_res=}"
  print("TritonFlashAttention2Func -> Passed!!!")


if __name__ == "__main__":
  main()
