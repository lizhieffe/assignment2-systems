# Section 2: Profiling
# Include both 1) CUDA kernel execution time and CPU time, as well as 2) memory usage.
# For 1) use `nsys profile` to run, for 2) run regularly
#
# Run the code:
# uv run cs336_systems/benchmark.py
#
# Detailed profiling (using Nvidia Nsight Systems):
# uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python cs336_systems/benchmark.py
#
# Less-detailed profiling:
# uv run nsys profile -- python cs336_systems/benchmark.py

from contextlib import nullcontext
import torch
import torch.cuda.nvtx as nvtx

from cs336_basics import model, nn_utils, optimizer
from cs336_systems.model_configs import (  # noqa: F401
  MODEL_CONFIG_S,
  MODEL_CONFIG_S_LC,
  MODEL_CONFIG_S_SC,
  MODEL_CONFIG_M,
  MODEL_CONFIG_M_LC,
  MODEL_CONFIG_M_SC,
  MODEL_CONFIG_XL_LC,
  MODEL_CONFIG_XL_SC,
)

BATCH_SIZE = 4
# CONTEXT_LENGTH = 512
# CONTEXT_LENGTH = 512 * 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_ITERS = 5
PROFILE_ITERS = 20

model_config = MODEL_CONFIG_M
enable_mixed_precision = False
inference_only = False
enable_torch_compile = True


@nvtx.range("training loop")
def main():
  print(f"Using device: {DEVICE}")

  precision_context = (
    torch.autocast(device_type=DEVICE, dtype=torch.float16)
    if enable_mixed_precision
    else nullcontext()
  )
  # Avoid saving activations when not necessary.
  grad_context = torch.no_grad() if inference_only else nullcontext()

  with precision_context, grad_context:
    # start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    lm = model.BasicsTransformerLM(
      vocab_size=model_config["vocab_size"],
      context_length=model_config["context_length"],
      d_model=model_config["d_model"],
      num_layers=model_config["num_layers"],
      num_heads=model_config["num_heads"],
      d_ff=model_config["d_ff"],
    ).to(DEVICE)

    if enable_torch_compile:
      lm = torch.compile(lm)

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

      if not inference_only:
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

        optim.zero_grad(set_to_none=True)

    # Save and finish the memory recording
    memory_history_file_name = (
      "memory_profile_fwd_bwd_optim_torch_compile.pickle"
      if not inference_only
      else "memory_profile_fwd_torch_compile.pickle"
    )
    torch.cuda.memory._dump_snapshot(memory_history_file_name)
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
  main()
