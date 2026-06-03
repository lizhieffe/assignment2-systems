# Section 2: Profiling
#
# Run the code:
# uv run cs336_systems/section_2_profiling.py
#
# Detailed profiling (using Nvidia Nsight Systems):
# uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python cs336_systems/benchmark_with_precision.py
#
# Less-detailed profiling:
# uv run nsys profile -- python cs336_systems/benchmark_with_precision.py

from contextlib import nullcontext
import torch
import torch.cuda.nvtx as nvtx
from torch import nn

from cs336_basics import model, nn_utils, optimizer

BATCH_SIZE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_ITERS = 5
PROFILE_ITERS = 20

MODEL_CONFIG_S = {
  "vocab_size": 10_000,
  "context_length": 512,
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072,
}

MODEL_CONFIG_S_SC = {
  "vocab_size": 10_000,
  "context_length": 128,
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072,
}

MODEL_CONFIG_M = {
  "vocab_size": 10_000,
  "context_length": 512,
  "d_model": 1024,
  "num_layers": 24,
  "num_heads": 16,
  "d_ff": 4096,
}

MODEL_CONFIG_M_SC = {
  "vocab_size": 10_000,
  "context_length": 128,
  "d_model": 1024,
  "num_layers": 24,
  "num_heads": 16,
  "d_ff": 4096,
}

model_config = MODEL_CONFIG_M

IN_FEATURES = 20
OUT_FEATURES = 2


class ToyModel(nn.Module):
  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    print(f"Forward after fc1: {x.dtype=}")
    x = self.ln(x)
    print(f"Forward after ln: {x.dtype=}")
    x = self.fc2(x)
    print(f"Forward after fc2: {x.dtype=}")
    return x


@nvtx.range("training loop")
def main():
  print(f"Using device: {DEVICE}")
  lm = ToyModel(in_features=IN_FEATURES, out_features=OUT_FEATURES).to(DEVICE)
  inp = torch.rand((BATCH_SIZE, IN_FEATURES)).to(DEVICE)

  with torch.autocast(device_type=DEVICE, dtype=torch.float16):
    print("Params dtypes:")
    for name, param in lm.named_parameters():
      print(f"param {name}: {param.dtype}")
    print()
    with nvtx.range("forward pass"):
      logits = lm(inp)
      print()
      print(f"logits dtype: {logits.dtype}")
      print()

    with nvtx.range("backward pass"):
      # Use inp as target for simplicity, since the model is untrained and we just want to profile the forward and backward pass.
      loss = nn_utils.cross_entropy(
        inputs=logits,
        targets=torch.randint(0, OUT_FEATURES, (BATCH_SIZE,)).to(DEVICE),
      )
      print(f"loss dtype: {loss.dtype}")
      print()

      loss.backward()
    
    for name, param in lm.named_parameters():
      print(f"grad for param {name}: {param.grad.dtype}")


if __name__ == "__main__":
  main()
