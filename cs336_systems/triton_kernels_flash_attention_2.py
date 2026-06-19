# Triton kernels.
#
# Run the code:
# uv run cs336_systems/triton_kernels_flash_attention_2.py

import einops
import functools
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
  is_causal: bool = False,
) -> jaxtyping.Float[torch.Tensor, "*LEADING_DIM D"]:
  D = Q.shape[-1]
  N = Q.shape[-2]
  S = Q @ K.transpose(-1, -2) / D**0.5  # (B, N, N)
  if is_causal:
    causal_mask = (
      torch.arange(N, device=S.device)[:, None]
      >= torch.arange(N, device=S.device)[None, :]
    )
    S = S.masked_fill(~causal_mask, float("-inf"))
  P = torch.softmax(S, dim=-1)  # (B, N, N)
  output = P @ V  # (B, N, D)
  return output


@torch.compile
def torch_flash_backward(
  Q: jaxtyping.Float[torch.Tensor, "B Q D"],
  K: jaxtyping.Float[torch.Tensor, "B K D"],
  V: jaxtyping.Float[torch.Tensor, "B K D"],
  output: jaxtyping.Float[torch.Tensor, "B Q D"],
  d_output: jaxtyping.Float[torch.Tensor, "B Q D"],
  softmax_denominator: jaxtyping.Float[torch.Tensor, "B Q"],
) -> tuple[
  jaxtyping.Float[torch.Tensor, "B Q D"],  # dQ
  jaxtyping.Float[torch.Tensor, "B K D"],  # dK
  jaxtyping.Float[torch.Tensor, "B V D"],  # dV
]:
  """Plain backward without recomputation.

  Followed Eq 4 - 6 & 12."""
  assert len(Q.shape) == 3
  _, _, d = Q.shape

  d_sqrt = d**0.5
  S = Q @ torch.transpose(K, -1, -2) / d_sqrt  # (B, Q, K)
  P = torch.exp(S - softmax_denominator.unsqueeze(-1))  # (B, Q, K)
  dV = P.transpose(-1, -2) @ d_output  # (B, K, D)
  dP = d_output @ V.transpose(-1, -2)  # (B, Q, K)
  D = torch.sum(output * d_output, dim=-1)  # (B, Q)
  dS = P * (dP - D.unsqueeze(-1))  # (B, Q, K)
  dQ = dS @ K / d_sqrt  # (B, Q, D)
  dK = dS.transpose(-1, -2) @ Q / d_sqrt  # (B, K, D)
  return dQ, dK, dV, None


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
    softmax_denominator, Q, K, V, output = ctx.saved_tensors
    return torch_flash_backward(
      Q=Q,
      K=K,
      V=V,
      output=output,
      d_output=grad_out,
      softmax_denominator=softmax_denominator,
    )


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
  is_causal: tl.constexpr,
):
  query_tile_idx = tl.program_id(axis=0)
  batch_idx = tl.program_id(axis=1)

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

    if is_causal:
      q_indexes = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
      k_indexes = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
      mask = q_indexes[:, None] >= k_indexes[None, :]
      mask = mask.to(
        tl.int1
      )  # Convert to bool; 0 means the positions that is after the queiry position and thus no attention.
      S_i = tl.where(mask, S_i, -1e6)

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


@triton.jit
def flash_bwd_kernel(
  Q_ptr,  # (B, Q, D)
  K_ptr,  # (B, K, D)
  V_ptr,  # (B, K, D)
  O_ptr,  # (B, Q, D)
  dO_ptr,  # (B, Q, D)
  L_ptr,  # (B, Q)
  dQ_ptr,  # (B, Q, D)
  dK_ptr,  # (B, K, D)
  dV_ptr,  # (B, K, D)
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
  stride_dob,
  stride_doq,
  stride_dod,
  stride_lb,
  stride_lq,
  stride_dqb,
  stride_dqq,
  stride_dqd,
  stride_dkb,
  stride_dkk,
  stride_dkd,
  stride_dvb,
  stride_dvk,
  stride_dvd,
  N_QUERIES,
  N_KEYS,
  scale,  # 1 / sqrt(D)
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr,
  is_causal: tl.constexpr,
):
  query_tile_idx = tl.program_id(axis=0)
  key_tile_idx = tl.program_id(axis=1)
  batch_idx = tl.program_id(axis=2)

  num_Q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
  num_K_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

  # First loop - calculate dK & dV

  K_block_ptr = tl.make_block_ptr(
    base=K_ptr + batch_idx * stride_kb,
    shape=(N_KEYS, D),
    strides=(stride_kk, stride_kd),
    offsets=(key_tile_idx * K_TILE_SIZE, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(0, 1),
  )
  V_block_ptr = tl.make_block_ptr(
    base=V_ptr + batch_idx * stride_vb,
    shape=(N_KEYS, D),
    strides=(stride_vk, stride_vd),
    offsets=(key_tile_idx * K_TILE_SIZE, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(0, 1),
  )
  dK_block_ptr = tl.make_block_ptr(
    base=dK_ptr + batch_idx * stride_dkb,
    shape=(N_KEYS, D),
    strides=(stride_dkk, stride_dkd),
    offsets=(key_tile_idx * K_TILE_SIZE, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(0, 1),
  )
  dV_block_ptr = tl.make_block_ptr(
    base=dV_ptr + batch_idx * stride_dvb,
    shape=(N_KEYS, D),
    strides=(stride_dvk, stride_dvd),
    offsets=(key_tile_idx * K_TILE_SIZE, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(0, 1),
  )

  K_j = tl.load(
    K_block_ptr, boundary_check=(0,), padding_option="zero"
  )  # (K_TILE_SIZE, D)
  V_j = tl.load(
    V_block_ptr, boundary_check=(0,), padding_option="zero"
  )  # (K_TILE_SIZE, D)
  dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # (K_TILE_SIZE, D)
  dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # (K_TILE_SIZE, D)

  Q_block_ptr = tl.make_block_ptr(
    base=Q_ptr + batch_idx * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(0, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(0, 1),
  )
  O_block_ptr = tl.make_block_ptr(
    base=O_ptr + batch_idx * stride_ob,
    shape=(N_QUERIES, D),
    strides=(stride_oq, stride_od),
    offsets=(0, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(0, 1),
  )
  dO_block_ptr = tl.make_block_ptr(
    base=dO_ptr + batch_idx * stride_dob,
    shape=(N_QUERIES, D),
    strides=(stride_doq, stride_dod),
    offsets=(0, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(0, 1),
  )
  L_block_ptr = tl.make_block_ptr(
    base=L_ptr + batch_idx * stride_lb,
    shape=(N_QUERIES,),
    strides=(stride_lq,),
    offsets=(0,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,),
  )

  for i in range(num_Q_tiles):
    Q_i = tl.load(
      Q_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE, D)
    O_i = tl.load(
      O_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE, D)
    dO_i = tl.load(
      dO_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE, D)
    D_i = tl.sum(O_i * dO_i, axis=-1)  # (Q_TILE_SIZE,)

    L_i = tl.load(
      L_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE,)

    S_i = tl.dot(Q_i, K_j.T) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
    if is_causal:
      q_range = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
      k_range = key_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
      mask = q_range[:, None] >= k_range[None, :]
      S_i = tl.where(mask, S_i, -1e6)

    P_i = tl.exp(S_i - L_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
    dV_j_next = dV_j + tl.dot(P_i.T, dO_i)  # (K_TILE_SIZE, D)
    dP_i = tl.dot(dO_i, V_j.T)  # (Q_TILE_SIZE, K_TILE_SIZE)
    dS_i = P_i * (dP_i - D_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
    dK_j_next = dK_j + tl.dot(dS_i.T, Q_i) * scale  # (K_TILE_SIZE, D)

    dV_j = dV_j_next
    dK_j = dK_j_next

    Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
    O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
    dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
    L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))

  tl.store(dK_block_ptr, dK_j, boundary_check=(0,))
  tl.store(dV_block_ptr, dV_j, boundary_check=(0,))

  # Second loop - calculate dQ
  Q_block_ptr = tl.make_block_ptr(
    base=Q_ptr + batch_idx * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
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
  dO_block_ptr = tl.make_block_ptr(
    base=dO_ptr + batch_idx * stride_dob,
    shape=(N_QUERIES, D),
    strides=(stride_doq, stride_dod),
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
    Q_block_ptr, boundary_check=(0,), padding_option="zero"
  )  # (Q_TILE_SIZE, D)
  O_i = tl.load(
    O_block_ptr, boundary_check=(0,), padding_option="zero"
  )  # (Q_TILE_SIZE, D)
  dO_i = tl.load(
    dO_block_ptr, boundary_check=(0,), padding_option="zero"
  )  # (Q_TILE_SIZE, D)
  D_i = tl.sum(O_i * dO_i, axis=-1)  # (Q_TILE_SIZE,)
  L_i = tl.load(
    L_block_ptr, boundary_check=(0,), padding_option="zero"
  )  # (Q_TILE_SIZE,)
  dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (Q_TILE_SIZE, D)

  K_block_ptr = tl.make_block_ptr(
    base=K_ptr + batch_idx * stride_kb,
    shape=(N_KEYS, D),
    strides=(stride_kk, stride_kd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
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
  for j in range(num_K_tiles):
    K_j = tl.load(
      K_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (K_TILE_SIZE, D)
    V_j = tl.load(
      V_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (K_TILE_SIZE, D)

    S_i = tl.dot(Q_i, K_j.T) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
    if is_causal:
      q_range = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
      k_range = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
      mask = q_range[:, None] >= k_range[None, :]
      S_i = tl.where(mask, S_i, -1e6)

    P_i = tl.exp(S_i - L_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
    dP_i = tl.dot(dO_i, V_j.T)  # (Q_TILE_SIZE, K_TILE_SIZE)
    dS_i = P_i * (dP_i - D_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
    dQ_i_next = dQ_i + tl.dot(dS_i, K_j) * scale  # (Q_TILE_SIZE, DS)

    dQ_i = dQ_i_next

    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

  dQ_block_ptr = tl.make_block_ptr(
    base=dQ_ptr + batch_idx * stride_dqb,
    shape=(N_QUERIES, D),
    strides=(stride_dqq, stride_dqd),
    offsets=(query_tile_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(0, 1),
  )
  tl.store(dQ_block_ptr, dQ_i, boundary_check=(0,))


@torch.compile
def torch_flash_backward_recomputation(
  Q: jaxtyping.Float[torch.Tensor, "B Q D"],
  K: jaxtyping.Float[torch.Tensor, "B K D"],
  V: jaxtyping.Float[torch.Tensor, "B K D"],
  output: jaxtyping.Float[torch.Tensor, "B Q D"],
  d_output: jaxtyping.Float[torch.Tensor, "B Q D"],
  softmax_denominator: jaxtyping.Float[torch.Tensor, "B Q"],
  is_causal: bool = False,
) -> tuple[
  jaxtyping.Float[torch.Tensor, "B Q D"],  # dQ
  jaxtyping.Float[torch.Tensor, "B K D"],  # dK
  jaxtyping.Float[torch.Tensor, "B V D"],  # dV
]:
  """Backward with recomputation.

  Followed Eq 13 - 19."""
  assert len(Q.shape) == 3
  _, N_QUERIES, d = Q.shape

  assert N_QUERIES == K.shape[1]

  d_sqrt = d**0.5

  S = Q @ torch.transpose(K, -1, -2) / d_sqrt  # (B, Q, K)
  if is_causal:
    causal_mask = (
      torch.arange(N_QUERIES, device=S.device)[:, None]
      >= torch.arange(N_QUERIES, device=S.device)[None, :]
    )
    S = S.masked_fill(~causal_mask, float("-inf"))

  P = torch.exp(S - softmax_denominator.unsqueeze(-1))  # (B, Q, K)
  dV = P.transpose(-1, -2) @ d_output  # (B, K, D)
  dP = d_output @ V.transpose(-1, -2)  # (B, Q, K)
  D = torch.sum(output * d_output, dim=-1)  # (B, Q)
  dS = P * (dP - D.unsqueeze(-1))  # (B, Q, K)
  dQ = dS @ K / d_sqrt  # (B, Q, D)
  dK = dS.transpose(-1, -2) @ Q / d_sqrt  # (B, K, D)
  return dQ, dK, dV, None


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
    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32

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
      is_causal,
    )

    ctx.save_for_backward(softmax_denominator, Q, K, V, output)
    ctx.is_causal = is_causal

    return output

  @staticmethod
  def backward(ctx, grad_out):
    use_torch_flash_backward_recomputation = False

    softmax_denominator, Q, K, V, output = ctx.saved_tensors
    assert len(Q.shape) == 3
    assert Q.shape == K.shape
    assert Q.shape == V.shape

    if use_torch_flash_backward_recomputation:
      return torch_flash_backward_recomputation(
        Q=Q,
        K=K,
        V=V,
        output=output,
        d_output=grad_out,
        softmax_denominator=softmax_denominator,
        is_causal=ctx.is_causal,
      )
    else:
      N_BATCHES, N_QUERIES, D = Q.shape
      N_KEYS = K.shape[1]

      Q_TILE_SIZE = 32
      K_TILE_SIZE = 32

      num_query_tiles = math.ceil(N_QUERIES / Q_TILE_SIZE)
      num_key_tiles = math.ceil(N_KEYS / K_TILE_SIZE)

      dQ = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
      dK = torch.empty(K.shape, device=K.device, dtype=K.dtype)
      dV = torch.empty(V.shape, device=V.device, dtype=V.dtype)

      flash_bwd_kernel[(num_query_tiles, num_key_tiles, N_BATCHES)](
        Q,  # (B, Q, D)
        K,  # (B, K, D)
        V,  # (B, K, D)
        output,  # (B, Q, D)
        grad_out,  # (B, Q, D)
        softmax_denominator,  # (B, Q)
        dQ,  # (B, Q, D)
        dK,  # (B, K, D)
        dV,  # (B, K, D)
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
        grad_out.stride()[0],
        grad_out.stride()[1],
        grad_out.stride()[2],
        softmax_denominator.stride()[0],
        softmax_denominator.stride()[1],
        dQ.stride()[0],
        dQ.stride()[1],
        dQ.stride()[2],
        dK.stride()[0],
        dK.stride()[1],
        dK.stride()[2],
        dV.stride()[0],
        dV.stride()[1],
        dV.stride()[2],
        N_QUERIES,
        N_KEYS,
        1 / math.sqrt(D),
        D,
        Q_TILE_SIZE,
        K_TILE_SIZE,
        ctx.is_causal,
      )
    return dQ, dK, dV, None


def create_benchmark_inputs(seq_len, emb_dim, dtype):
  Q = torch.rand(
    (1, seq_len, emb_dim),
    device=DEVICE,
    dtype=dtype,
    requires_grad=True,
  )
  K = torch.rand(
    (1, seq_len, emb_dim),
    device=DEVICE,
    dtype=dtype,
    requires_grad=True,
  )
  V = torch.rand(
    (1, seq_len, emb_dim),
    device=DEVICE,
    dtype=dtype,
    requires_grad=True,
  )
  grad_out = torch.rand((1, seq_len, emb_dim), device=DEVICE, dtype=dtype)
  return Q, K, V, grad_out


def benchmark() -> None:
  WARMUP = 1000
  REP = 3000
  DTYPE = torch.bfloat16

  # Triton impl
  for seq_len in [
    2**i for i in range(int(math.log2(128)), int(math.log2(65536 * 2)))
  ]:
    for emb_dim in [16, 32, 64, 128]:
      try:
        Q, K, V, grad_out = create_benchmark_inputs(
          seq_len=seq_len, emb_dim=emb_dim, dtype=DTYPE
        )
        fwd_fn = functools.partial(
          TritonFlashAttention2Func.apply, Q, K, V, True
        )
        fwd_runtime = triton.testing.do_bench(fwd_fn, warmup=WARMUP, rep=REP)

        Q, K, V, grad_out = create_benchmark_inputs(
          seq_len=seq_len, emb_dim=emb_dim, dtype=DTYPE
        )

        def fwd_bwd_fn():
          TritonFlashAttention2Func.apply(Q, K, V, True).backward(grad_out)

        fwd_bwd_runtime = triton.testing.do_bench(
          fwd_bwd_fn, warmup=WARMUP, rep=REP
        )

        print(
          f"Triton {seq_len=} {emb_dim=}: fwd={fwd_runtime:.4f}ms, "
          f"bwd={fwd_bwd_runtime - fwd_runtime:.4f}ms, "
          f"fwd-bwd={fwd_bwd_runtime:.4f}ms"
        )
      except torch.OutOfMemoryError:
        print(f"Triton {seq_len=} {emb_dim=}: OOM")

  # Torch impl
  for seq_len in [
    2**i for i in range(int(math.log2(128)), int(math.log2(65536 * 2)))
  ]:
    for emb_dim in [16, 32, 64, 128]:
      try:
        Q, K, V, grad_out = create_benchmark_inputs(
          seq_len=seq_len, emb_dim=emb_dim, dtype=DTYPE
        )
        fwd_fn = functools.partial(torch_plain_attention, Q, K, V, True)
        fwd_runtime = triton.testing.do_bench(fwd_fn, warmup=WARMUP, rep=REP)

        Q, K, V, grad_out = create_benchmark_inputs(
          seq_len=seq_len, emb_dim=emb_dim, dtype=DTYPE
        )

        def fwd_bwd_fn():
          output = torch_plain_attention(Q, K, V, True)
          output.backward(grad_out)

        fwd_bwd_runtime = triton.testing.do_bench(
          fwd_bwd_fn, warmup=WARMUP, rep=REP
        )

        print(
          f"Torch {seq_len=} {emb_dim=}: fwd={fwd_runtime:.4f}ms, "
          f"bwd={fwd_bwd_runtime - fwd_runtime:.4f}ms, "
          f"fwd-bwd={fwd_bwd_runtime:.4f}ms"
        )
      except torch.OutOfMemoryError:
        print(f"Torch {seq_len=} {emb_dim=}: OOM")


def main() -> None:
  benchmark()
  return

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
