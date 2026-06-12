# Profile CPU & GPU for various ops
#
# Run the code:
# uv run cs336_systems/profile_ops.py

import time
import torch

from cs336_systems import gpu_lib, profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sleep():
  time.sleep(50 / 1000)


def add_function(x, y):
  return x + y


def matmul(x, y):
  return x @ y


def cdist(x, y):
  return torch.cdist(x, y)


def gelu(x, y):
  return torch.nn.functional.gelu(x + y)


def softmax(x, y):
  return torch.nn.functional.softmax(x + y)


def mlp(x, y):
  z = x + y
  z = torch.nn.functional.linear(input=z, weight=torch.rand_like(z))
  z = torch.nn.functional.gelu(z)
  return z


def main():
  gpu_lib.print_gpu_specs()

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
  profile_lib.profile(
    "gelu", profile_lib.run_operation2(dim=2048, fn=gelu, device=DEVICE)
  )
  profile_lib.profile(
    "softmax", profile_lib.run_operation2(dim=2048, fn=softmax, device=DEVICE)
  )
  profile_lib.profile(
    "mlp", profile_lib.run_operation2(dim=2048, fn=mlp, device=DEVICE)
  )


if __name__ == "__main__":
  main()
