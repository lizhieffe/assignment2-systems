# Library for benchmark.
#
# Run the code:
# uv run cs336_systems/profile_pytorch_attn.py

import functools
import torch


from cs336_basics import model
from cs336_systems import profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_model_then_profile(d_model: int, seq_length: int):
  """Helper fn to create model, and then do profile.

  This makes sure the profiling focuses on the computation, instead
  of model/data copy to HBM.
  """
  bs = 8
  rotary_embedding = model.RotaryEmbedding(
    context_length=seq_length, dim=d_model
  ).to(DEVICE)
  attn = model.CausalMultiHeadSelfAttention(
    d_model=d_model,
    num_heads=1,
    positional_encoder=rotary_embedding,
  ).to(DEVICE)

  x = torch.rand((bs, seq_length, d_model)).to(DEVICE)

  profile_lib.profile(
    f"torch_attn_forward_dim:{d_model}_seq:{seq_length}",
    functools.partial(attn, x),
  )


def main():
  for d_model in [16, 32, 64, 128]:
    for seq_length in [256, 1024, 4096, 8192, 16384]:
      create_model_then_profile(d_model=d_model, seq_length=seq_length)


if __name__ == "__main__":
  main()
