# Library for profiling GPU.

import functools
import torch

from collections.abc import Callable


def profile(
  description: str,
  fn: Callable,
  num_warmups: int = 1,
  with_stack: bool = False,
):
  """Profile the CPU + GPU time of fn and print a summary table.

  Args:
    description: Label printed as the table header.
    fn: Zero-argument callable to profile. Called num_warmups times before
      the measured run.
    num_warmups: Number of warm-up calls before profiling starts.
    with_stack: If True, export a CUDA stack trace to var/stacks_<description>.txt.
  """
  for _ in range(num_warmups):
    fn()

  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for all previous CUDA work to finish before starting the timer.

  # The key to profile CPU + GPU time.
  with torch.profiler.profile(
    activities=[
      torch.profiler.ProfilerActivity.CPU,
      torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=False,
    profile_memory=False,
    with_stack=with_stack,
  ) as prof:
    fn()
    if torch.cuda.is_available():
      torch.cuda.synchronize()  # Wait for all previous CUDA work to finish before ending the timer.

  print(f"============= {description} ===============")
  table = prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=10, max_name_column_width=80
  )
  print(table)

  if with_stack:
    text_path = f"var/stacks_{description}.txt"
    svg_path = f"var/stacks_{description}.svg"
    prof.export_stacks(text_path, "self_cuda_time_total")


def run_operation1(dim: int, fn: Callable, device: str = "cpu") -> Callable:
  """Helper fn to run an op requires two 2D metrics input."""
  a = torch.rand((dim, dim)).to(device)
  return functools.partial(fn, a)


def run_operation2(dim: int, fn: Callable, device: str = "cpu") -> Callable:
  """Helper fn to run an op requires two 2D metrics input."""
  a = torch.rand((dim, dim)).to(device)
  b = torch.rand((dim, dim)).to(device)
  return functools.partial(fn, a, b)
