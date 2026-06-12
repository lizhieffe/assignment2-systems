// CUDA kernel for GELU.

#include <c10/cuda/CUDAException.h>
#include <math.h>
#include <torch/extension.h>

// This is the Kernel code which runs on GPU.
__global__ void gelu_kernel(float* in, float* out, int num_elements) {
  // Get the index into the tensor
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_elements) {  // To handle the case when n < numBlocks * blockDim
    out[i] = 0.5f * in[i] *
             (1.0f +
              tanhf(0.79788456f * (in[i] + 0.044715f * in[i] * in[i] * in[i])));
  }
}

// Utility: Compute ceil(a/b)
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

// Wrapper that runs on CPU.
torch::Tensor gelu(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");

  // Allocate empty tensor
  auto out = torch::empty_like(x);

  // Determine grid (elements divided into blocks)
  int num_elements = x.numel();

  const int block_size = 1024;  // # of threads

  const int num_blocks = cdiv(num_elements, block_size);
  gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(), num_elements);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu", &gelu, "GeLU forward pass (CUDA)");
}
