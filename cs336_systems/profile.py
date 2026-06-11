# Profile CPU & GPU for various ops
#
# Run the code:
# uv run cs336_systems/profile.py

import time
import torch

from cs336_systems import profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sleep():
  time.sleep(50 / 1000)


def add_function(x, y):
  return x + y


def matmul(x, y):
  return x @ y


def cdist(x, y):
  return torch.cdist(x, y)


def main():
  profile_lib.profile("Sleep", fn=sleep)
  profile_lib.profile(
    "Add function",
    profile_lib.run_operation2(dim=2048, fn=add_function, device=DEVICE),
  )
  profile_lib.profile(
    "MatMul dim=2048",
    profile_lib.run_operation2(dim=128, fn=matmul, device=DEVICE),
  )
  profile_lib.profile(
    "MatMul dim=2048",
    profile_lib.run_operation2(dim=2048, fn=matmul, device=DEVICE),
  )
  profile_lib.profile(
    "cdist",
    profile_lib.run_operation2(dim=2048, fn=cdist, device=DEVICE),
  )


if __name__ == "__main__":
  main()
