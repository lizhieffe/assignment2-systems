# Section 2: Profiling
#
# Run the code:
# uv run cs336_systems/section_2_profiling.py
#
# Detailed profiling (using Nvidia Nsight Systems):
# uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autogradshapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py
#
# Less-detailed profiling:
# uv run nsys profile -- python benchmark.py

import timeit
import torch

from cs336_basics import model, nn_utils, optimizer

BATCH_SIZE = 4
CONTEXT_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_ITERS = 5
PROFILE_ITERS = 20


def main():
  print(f"Using device: {DEVICE}")
  lm = model.BasicsTransformerLM(
    vocab_size=10_000,
    context_length=CONTEXT_LENGTH,
    d_model=1024,
    num_layers=24,
    num_heads=16,
    d_ff=4096,
  ).to(DEVICE)

  optim = optimizer.AdamW(lm.parameters(), lr=1e-3)

  profile_forward = []
  profile_forward_and_backward = []
  profile_forward_and_backward_and_optimizer = []

  for it in range(WARMUP_ITERS + PROFILE_ITERS):
    optim.zero_grad()

    inp = torch.randint(0, 10_000, (BATCH_SIZE, CONTEXT_LENGTH)).to(DEVICE)

    start = timeit.default_timer()
    logits = lm(inp)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the timer.
    if it >= WARMUP_ITERS:
      profile_forward.append(timeit.default_timer() - start)

    # Use inp as target for simplicity, since the model is untrained and we just want to profile the forward and backward pass.
    loss = nn_utils.cross_entropy(inputs=logits, targets=inp)
    loss.backward()
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the timer.
    if it >= WARMUP_ITERS:
      profile_forward_and_backward.append(timeit.default_timer() - start)

    optim.step()
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the timer.
    if it >= WARMUP_ITERS:
      profile_forward_and_backward_and_optimizer.append(
        timeit.default_timer() - start
      )

  print(
    f"Average forward pass time: {sum(profile_forward) / len(profile_forward):.4f} seconds"
  )
  print(
    f"Average forward and backward pass time: {sum(profile_forward_and_backward) / len(profile_forward_and_backward):.4f} seconds"
  )
  print(
    f"Average forward, backward, and optimizer time: {sum(profile_forward_and_backward_and_optimizer) / len(profile_forward_and_backward_and_optimizer):.4f} seconds"
  )


if __name__ == "__main__":
  main()
