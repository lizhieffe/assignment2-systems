# Section 2: Profiling
#
# Run the code:
# uv run cs336_systems/section_2_profiling.py
#
# Detailed profiling (using Nvidia Nsight Systems):
# uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python cs336_systems/benchmark.py
#
# Less-detailed profiling:
# uv run nsys profile -- python benchmark.py

from contextlib import nullcontext
import torch
import torch.cuda.nvtx as nvtx

from cs336_basics import model, nn_utils, optimizer

BATCH_SIZE = 4
# CONTEXT_LENGTH = 512
# CONTEXT_LENGTH = 512 * 1

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


@nvtx.range("training loop")
def main():
  print(f"Using device: {DEVICE}")
  lm = model.BasicsTransformerLM(
    vocab_size=model_config["vocab_size"],
    context_length=model_config["context_length"],
    d_model=model_config["d_model"],
    num_layers=model_config["num_layers"],
    num_heads=model_config["num_heads"],
    d_ff=model_config["d_ff"],
  ).to(DEVICE)

  optim = optimizer.AdamW(lm.parameters(), lr=1e-3)

  for it in range(WARMUP_ITERS + PROFILE_ITERS):
    optim.zero_grad()

    inp = torch.randint(
      0, 10_000, (BATCH_SIZE, model_config["context_length"])
    ).to(DEVICE)

    forward_context = (
      nullcontext() if it < WARMUP_ITERS else nvtx.range("forward pass")
    )
    with forward_context:
      logits = lm(inp)

    backward_context = (
      nullcontext() if it < WARMUP_ITERS else nvtx.range("backward pass")
    )
    with backward_context:
      # Use inp as target for simplicity, since the model is untrained and we just want to profile the forward and backward pass.
      loss = nn_utils.cross_entropy(inputs=logits, targets=inp)
      loss.backward()

    optimizer_context = (
      nullcontext() if it < WARMUP_ITERS else nvtx.range("optimizer step")
    )
    with optimizer_context:
      optim.step()


if __name__ == "__main__":
  main()
