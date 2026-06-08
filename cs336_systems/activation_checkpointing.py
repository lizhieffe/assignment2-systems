# Section 3: autograd residules
#
# Run the code:
# uv run cs336_systems/activation_checkpointing.py

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics import model

total_size_bytes = 0


def pack_hook(t):
  """A simple hook to print the saving of the activation."""
  global total_size_bytes
  if isinstance(t, torch.nn.Parameter):
    # avoid double counting parameters, which are also saved as activations
    return t
  shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
  total_size_bytes += t.numel() * t.element_size()
  print(f"Saving activation: {shape=}, {dtype=}, {grad_fn=}")
  return t


# When enabled, the memory used to save the activations during is 0.16GB;
# it only contains the input activations -> 2 x [4, 2048, 2560] x 4 bytes = 0.16GB.
# when disabled, it is 14GB.
enable_checkpointing = True


def main():
  d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
  block = model.TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    positional_encoder=model.RotaryEmbedding(
      dim=d_model // num_heads, context_length=context_length
    ),
  )

  # Fuse as many
  block = torch.compile(block)

  def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x

  def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

  def four_blocks_checkpoint(x):
    # Checkpointing saves the input, and throws away the intermediate
    # activations. During backward, it will recompute the intermediate
    # activations by running the forward function again.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

  x = torch.rand((4, context_length, d_model), requires_grad=True)

  # Log the total bytes saved during fwd pass for autograd residules.
  with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
    if enable_checkpointing:
      y = four_blocks_checkpoint(x)
    else:
      y = four_blocks(x)
    # y.sum().backward()

  print(
    f"Total size of saved activations: {total_size_bytes / (1024**3):.2f} GB"
  )


if __name__ == "__main__":
  main()
