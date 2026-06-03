# Section 2: Profiling
#
# Run the code:
# uv run cs336_systems/precision.py

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_steps(steps: int) -> None:
  s = torch.tensor(0, dtype=torch.float32, device=DEVICE)
  for _ in range(steps):
    s += torch.tensor(0.01, dtype=torch.float32, device=DEVICE)
  print(f"===lizhi float32 sum: {s.item()}")

  s = torch.tensor(0, dtype=torch.float16, device=DEVICE)
  for _ in range(steps):
    s += torch.tensor(0.01, dtype=torch.float16, device=DEVICE)
  print(f"===lizhi float16 sum: {s.item()}")

  s = torch.tensor(0, dtype=torch.bfloat16, device=DEVICE)
  for _ in range(steps):
    s += torch.tensor(0.01, dtype=torch.bfloat16, device=DEVICE)
  print(f"===lizhi bfloat16 sum: {s.item()}")

  s = torch.tensor(0, dtype=torch.float32, device=DEVICE)
  for _ in range(steps):
    s += torch.tensor(0.01, dtype=torch.float16, device=DEVICE)
  print(f"===lizhi mixed float32/float16 sum: {s.item()}")

  s = torch.tensor(0, dtype=torch.float32, device=DEVICE)
  for _ in range(steps):
    x = torch.tensor(0.01, dtype=torch.float16, device=DEVICE)
    s += x.type(torch.float32)
  print(f"===lizhi mixed float32/upcast-float32 sum: {s.item()}")


def main():
  print(f"Using device: {DEVICE}")
  run_steps(1000)
  run_steps(2000)


if __name__ == "__main__":
  main()
