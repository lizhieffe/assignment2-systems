# Section 3: activation checkpointing with XL model
#
# Run the code:
# uv run cs336_systems/activation_checkpointing_xl.py --granularity 4

import argparse
from enum import IntEnum

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics import model
from cs336_systems.model_configs import MODEL_CONFIG_L

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

total_size_bytes = 0


def pack_hook(t):
  """A simple hook to print the saving of the activation."""
  global total_size_bytes
  if isinstance(t, torch.nn.Parameter):
    # avoid double counting parameters, which are also saved as activations
    return t
  total_size_bytes += t.numel() * t.element_size()
  return t


# Checkpoint every N blocks: more blocks per segment = less memory, more recompute.
# Valid values must divide num_layers (36 for MODEL_CONFIG_L).
# Powers of 2 that divide 36: 1, 2, 4. Plus 36 for all-layers-in-one-checkpoint.
class CheckpointGranularity(IntEnum):
  BLOCKS_1 = 1
  BLOCKS_2 = 2
  BLOCKS_4 = 4
  BLOCKS_6 = 6
  BLOCKS_9 = 9
  BLOCKS_12 = 12
  BLOCKS_18 = 18
  BLOCKS_36 = 36  # all layers in one checkpoint


def _make_segment_fn(block, n):
  def run_n_blocks(x):
    for _ in range(n):
      x = block(x)
    return x

  return run_n_blocks


def run_with_checkpoint(block, x, num_layers, granularity):
  segment_size = int(granularity)
  assert num_layers % segment_size == 0, (
    f"num_layers ({num_layers}) must be divisible by segment_size ({segment_size})"
  )
  segment_fn = _make_segment_fn(block, segment_size)
  for _ in range(num_layers // segment_size):
    x = checkpoint(segment_fn, x, use_reentrant=False)
  x.sum().backward()
  return x


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--granularity",
    type=int,
    choices=[g.value for g in CheckpointGranularity],
    default=CheckpointGranularity.BLOCKS_2.value,
    help="Number of blocks per checkpoint segment (must be a power of 2 up to num_layers).",
  )
  args = parser.parse_args()
  checkpoint_granularity = CheckpointGranularity(args.granularity)

  cfg = MODEL_CONFIG_L
  d_model = cfg["d_model"]
  d_ff = cfg["d_ff"]
  num_heads = cfg["num_heads"]
  num_layers = cfg["num_layers"]
  context_length = cfg["context_length"]

  block = model.TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    positional_encoder=model.RotaryEmbedding(
      dim=d_model // num_heads, context_length=context_length
    ),
  ).to(DEVICE)
  block = torch.compile(block)

  x = torch.rand(
    (4, context_length, d_model), requires_grad=True, device=DEVICE
  )

  # Warmup: trigger CUDA/compile initialization before entering checkpoint context,
  # which forbids device state initialization inside it.
  with torch.no_grad():
    _ = block(x)

  global total_size_bytes
  total_size_bytes = 0

  memory_history_file_name = (
    f"memory_profile_checkpoint_{checkpoint_granularity.name.lower()}.pickle"
  )

  # start recording memory history
  torch.cuda.memory._record_memory_history(max_entries=1_000_000)

  with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
    _ = run_with_checkpoint(block, x, num_layers, checkpoint_granularity)

  # Store the memory history for later analysis.
  torch.cuda.memory._dump_snapshot(memory_history_file_name)
  torch.cuda.memory._record_memory_history(enabled=None)

  num_segments = num_layers // int(checkpoint_granularity)
  print(
    f"CheckpointGranularity: {checkpoint_granularity.name} ({num_segments} segments)"
  )
  print(
    f"Total size of saved activations: {total_size_bytes / (1024**3):.2f} GB"
  )
  expected_gb = num_segments * 4 * context_length * d_model * 4 / (1024**3)
  print(f"Expected (segment inputs only): {expected_gb:.2f} GB")


if __name__ == "__main__":
  main()
