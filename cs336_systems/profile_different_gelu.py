# Profile CPU & GPU for various implementation of GeLU, from
# naive python impl to more advanced CUDA & Triton impls.
#
# Run the code:
# uv run cs336_systems/profile_different_gelu.py

import os
import triton
import triton.language as tl
import torch
import torch.utils.cpp_extension as _cpp_ext
from torch.utils.cpp_extension import load_inline

from cs336_systems import gpu_lib, profile_lib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_LAUNCH_BLOCKING = (
  False  # Enable to capture debug log, but it will be slow.
)


def pytorch_gelu(x):
  return torch.nn.functional.gelu(x)


def manual_gelu(x: torch.Tensor) -> torch.Tensor:
  return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def manual_gelu_with_pow(x: torch.Tensor) -> torch.Tensor:
  return (
    0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))
  )


def _find_cuda_home() -> str:
  import site

  for d in site.getsitepackages():
    cuda = os.path.join(d, "nvidia", "cu13")
    if os.path.isdir(cuda):
      return cuda
  raise RuntimeError("Could not find pip-installed nvidia/cu13 package")


def create_cuda_gelu():
  if CUDA_LAUNCH_BLOCKING:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
  if _cpp_ext.CUDA_HOME is None:
    _cpp_ext.CUDA_HOME = _find_cuda_home()
  # Prepend pip nvcc's bin dir so its ptxas is found before any system ptxas
  cuda_bin = os.path.join(_cpp_ext.CUDA_HOME, "bin")
  os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
  src = os.path.join(os.path.dirname(__file__), "gelu.cu")
  # Strip PYBIND11_MODULE block — load_inline generates its own via functions=
  cuda_gelu_src = open(src).read()
  cuda_gelu_src = cuda_gelu_src[: cuda_gelu_src.index("PYBIND11_MODULE")]
  cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

  if not torch.cuda.is_available():
    return None

  os.makedirs("var/cuda_gelu", exist_ok=True)
  module = load_inline(
    name="inline_gelu",
    cpp_sources=[cpp_gelu_src],
    cuda_sources=[cuda_gelu_src],
    functions=["gelu"],
    extra_cflags=["-O2"],
    verbose=True,
    with_cuda=True,
    build_directory="var/cuda_gelu",
  )
  return module.gelu


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
  assert x.is_cuda
  assert x.is_contiguous

  # Allocate output tensor
  y = torch.empty_like(x)

  # Determine grid (elements divided into blocks)
  num_elements = x.numel()
  block_size = 1024  # num of threads
  num_blocks = triton.cdiv(num_elements, block_size)

  # Invoke GPU kernel
  triton_gelu_kernel[(num_blocks,)](
    x, y, num_elements=num_elements, BLOCK_SIZE=block_size
  )

  return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
  """
  GPU kernel for triton gelu.

  Args:
    x_ptr: input
    y_ptr: output
    num_elements: total number of elements
    BLOCK_SIZE: num of threads in each block
  """
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE

  # Indicate where this thread block should operate.
  offset = block_start + tl.arange(0, BLOCK_SIZE)

  # Handle boundary
  mask = offset < num_elements

  # Read
  x = tl.load(x_ptr + offset, mask=mask)

  # Compute
  a = 0.7978845608 * (x + 0.044715 * x * x * x)
  exp = tl.exp(2 * a)
  tanh = (exp - 1) / (exp + 1)
  y = 0.5 * x * (1 + tanh)

  # Write
  tl.store(y_ptr + offset, y, mask=mask)


def main():
  gpu_lib.print_gpu_specs()

  profile_lib.profile(
    "pytorch_gelu",
    profile_lib.run_operation1(dim=2048, fn=pytorch_gelu, device=DEVICE),
  )
  profile_lib.profile(
    "manual_gelu",
    profile_lib.run_operation1(dim=2048, fn=manual_gelu, device=DEVICE),
  )
  profile_lib.profile(
    "manual_gelu_torch_compile",
    profile_lib.run_operation1(
      dim=2048, fn=torch.compile(manual_gelu), device=DEVICE
    ),
  )
  profile_lib.profile(
    "manual_gelu_with_pow",
    profile_lib.run_operation1(
      dim=2048, fn=manual_gelu_with_pow, device=DEVICE
    ),
  )

  cuda_gelu = create_cuda_gelu()
  profile_lib.profile(
    "cuda_gelu",
    profile_lib.run_operation1(dim=2048, fn=cuda_gelu, device=DEVICE),
  )

  profile_lib.profile(
    "triton_gelu",
    profile_lib.run_operation1(dim=2048, fn=triton_gelu, device=DEVICE),
  )


if __name__ == "__main__":
  main()
