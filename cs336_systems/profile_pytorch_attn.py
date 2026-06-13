# Library for benchmark.
#
# Run the code:
# uv run cs336_systems/profile_pytorch_attn.py

import functools
import torch
import torch.nn as nn
import torch._dynamo


from cs336_basics import model
from cs336_systems import profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_TORCH_COMPILE = True

# Increase the limit from 8 to something higher
torch._dynamo.config.recompile_limit = 128


def profile_nn_module(
  description: str,
  module: nn.Module,
  x: torch.Tensor,
  num_warmups: int = 1,
  profile_memory: bool = False,
  with_stack: bool = False,
):
  """Profile the CPU + GPU time of fn and print a summary table.

  Args:
    description: Label printed as the table header.
    fn: Callable taking the input tensor `x` and returning the output tensor.
      Called num_warmups times before the measured run.
    num_warmups: Number of warm-up calls before profiling starts.
    with_stack: If True, export a CUDA stack trace to var/stacks_<description>.txt.
  """
  print(f"============= {description} ===============")

  try:
    for _ in range(num_warmups):
      module(x)

    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()
      torch.cuda.synchronize()  # Wait for all previous CUDA work to finish before starting the timer.

    # Forward pass
    with torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
      ],
      record_shapes=False,
      profile_memory=profile_memory,
      with_stack=with_stack,
    ) as prof:
      y = module(x)
      if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all previous CUDA work to finish before ending the timer.

    forward_peak = (
      torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    forward_current = (
      torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    print(
      f"[HBM forward] current={forward_current / 1024**2:.2f} MiB "
      f"peak={forward_peak / 1024**2:.2f} MiB"
    )

    table = prof.key_averages().table(
      sort_by="self_cuda_time_total", row_limit=10, max_name_column_width=80
    )
    print("Forward pass: ")
    print(table)

    loss = y.sum()

    # Backpward pass
    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()
      torch.cuda.synchronize()  # Wait for all previous CUDA work to finish before starting the timer.

    with torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
      ],
      record_shapes=False,
      profile_memory=profile_memory,
      with_stack=with_stack,
    ) as prof:
      y = loss.backward()
      if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all previous CUDA work to finish before ending the timer.

    backward_peak = (
      torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    backward_current = (
      torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    print(
      f"[HBM backward] current={backward_current / 1024**2:.2f} MiB "
      f"peak={backward_peak / 1024**2:.2f} MiB"
    )

    table = prof.key_averages().table(
      sort_by="self_cuda_time_total", row_limit=10, max_name_column_width=80
    )
    print("Backward pass: ")
    print(table)

  except RuntimeError as exc:
    message = str(exc).lower()
    if "out of memory" in message or "cuda out of memory" in message:
      print(f"[OOM] {description} skipped due to CUDA OOM\n")
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      return
    raise


def main():
  for d_model in [16, 32, 64, 128]:
    for seq_length in [256, 1024, 4096, 8192, 16384]:
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

      if ENABLE_TORCH_COMPILE:
        attn = torch.compile(attn, mode="reduce-overhead")

      profile_nn_module(
        f"torch_attention_dim:{d_model}_seq:{seq_length}",
        attn,
        x,
        profile_memory=True,
      )


if __name__ == "__main__":
  main()
