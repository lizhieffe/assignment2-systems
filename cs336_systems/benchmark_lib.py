# Library for benchmark.

import time
import statistics
import torch

from collections.abc import Callable


def benchmark(
  description: str, fn: Callable, num_warmups: int = 5, num_trials: int = 3
):
  """Benchmark a fn."""
  for _ in range(num_warmups):
    fn()

  if torch.cuda.is_available():
    torch.cuda.synchronize()

  times = []
  for _ in range(num_trials):
    start_time = time.time()

    fn()

    if torch.cuda.is_available():
      torch.cuda.synchronize()

    times.append(time.time() - start_time)

  return statistics.mean(times)
