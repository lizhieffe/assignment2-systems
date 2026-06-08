# Section 2: autograd residules
#
# Run the code:
# uv run cs336_systems/autograd_residule.py

import torch

from cs336_basics import model


def pack_hook(t):
  """A simple hook to print the saving of the activation."""
  shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
  print(f"Saving activation: {shape=}, {dtype=}, {grad_fn=}")
  return t


def unpack_hook(t):
  """A simple hook to print the loading of the activation."""
  shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
  print(f"Loading activation: {shape=}, {dtype=}, {grad_fn=}")
  return t


use_compile = True


def main():
  #   x = torch.rand((4, 512, 2560))
  x = torch.rand((4, 512, 2560), requires_grad=True)
  ln = model.RMSNorm(x.shape[-1])
  if use_compile:
    ln = torch.compile(ln)

  with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()


if __name__ == "__main__":
  main()
