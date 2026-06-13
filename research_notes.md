# Research Notes — Assignment 2: Systems

## Section 2: Profiling

### Experiment: Training Iteration Timing (timeit)

**Setup:** 5 warmup iterations, 20 profiling iterations

| Phase | Avg Time (s) |
|---|---|
| Forward pass | 0.1448 |
| Forward + backward | 0.4430 |
| Forward + backward + optimizer | 0.5205 |

**Derived:**
- Backward pass alone: 0.4430 − 0.1448 = 0.2982 s (~2.06× forward)
- Optimizer step alone: 0.5205 − 0.4430 = 0.0775 s
- Backward is the dominant cost (~57% of total iteration time)

---

**Setup:** 0 warmup iterations, 20 profiling iterations

| Phase | Avg Time (s) |
|---|---|
| Forward pass | 0.1567 |
| Forward + backward | 0.4620 |
| Forward + backward + optimizer | 0.5400 |

**Derived:**
- Backward pass alone: 0.4620 − 0.1567 = 0.3053 s (~1.95× forward)
- Optimizer step alone: 0.5400 − 0.4620 = 0.0780 s
- Backward is the dominant cost (~56% of total iteration time)

**Comparison (no warmup vs. 5 warmup iters):**
- Forward: +0.0119 s (+8.2%) without warmup
- Backward: +0.0071 s (+2.4%) without warmup
- Warmup effect is most visible in the forward pass, likely due to CUDA kernel compilation/caching on first runs

---

**Setup:** 0 warmup iterations, 5 profiling iterations

| Phase | Avg Time (s) |
|---|---|
| Forward pass | 0.1899 |
| Forward + backward | 0.5099 |
| Forward + backward + optimizer | 0.5882 |

**Derived:**
- Backward pass alone: 0.5099 − 0.1899 = 0.3200 s (~1.68× forward)
- Optimizer step alone: 0.5882 − 0.5099 = 0.0783 s
- Backward is the dominant cost (~54% of total iteration time)

**Comparison (no warmup: 5 iters vs. 20 iters):**
- Forward: +0.0332 s (+21%) with only 5 iters — high variance from small sample, early iters include cold-start overhead
- The 5-iter average is noisier and skewed upward; 20 iters gives a more stable estimate

---

## NVTX Profiling (Nsight Systems)

Four configurations profiled: two model sizes (S, M) × two context lengths (512, 128).

| Config | context_length | d_model | num_layers | Forward (ms) | Backward (ms) | Optimizer (ms) | Total (ms) |
|---|---|---|---|---|---|---|---|
| MODEL_CONFIG_M    | 512 | 1024 | 24 |  88 | 250 | 160 | 498 |
| MODEL_CONFIG_M_SC | 128 | 1024 | 24 |  87 | 170 |  94 | 351 |
| MODEL_CONFIG_S    | 512 |  768 | 12 |  44 |  85 |  46 | 175 |
| MODEL_CONFIG_S_SC | 128 |  768 | 12 |  44 |  85 |  46 | 175 |

**Key takeaways:**
- Backward is consistently the dominant cost (1.9–2.8× forward depending on config)
- Reducing context 4× (512→128) saves ~30% on MODEL_CONFIG_M but nothing on MODEL_CONFIG_S — at the S model size, attention cost is negligible and compute is feedforward-bound
- Optimizer cost scales with parameter count: ~3.5× more parameters in M vs S, ~3.5× longer optimizer step (160ms vs 46ms)

---

### Experiment: NVTX Profiling (Nsight Systems) — MODEL_CONFIG_M

**Model config (MODEL_CONFIG_M):** vocab_size=10,000 · context_length=512 · d_model=1024 · num_layers=24 · num_heads=16 · d_ff=4096

**Setup:** NVTX ranges applied only after warmup iterations (`it > WARMUP_ITERS`); profiled with Nsight Systems

| Phase | Time (ms) |
|---|---|
| Forward pass | 88 |
| Backward pass | 250 |
| Optimizer step | 160 |

**Derived:**
- Backward is ~2.84× forward
- Backward is the dominant cost (~50% of total iteration time)
- Optimizer is notably more expensive here (~32%) than in timeit measurements (~15%) — NVTX captures GPU-side work more precisely, whereas timeit can undercount async optimizer kernels

---

### Experiment: NVTX Profiling (Nsight Systems) — MODEL_CONFIG_M_SC

**Model config (MODEL_CONFIG_M_SC):** vocab_size=10,000 · context_length=128 · d_model=1024 · num_layers=24 · num_heads=16 · d_ff=4096

**Setup:** same as MODEL_CONFIG_M above; context_length reduced from 512 → 128

| Phase | Time (ms) |
|---|---|
| Forward pass | 87 |
| Backward pass | 170 |
| Optimizer step | 94 |

**Derived:**
- Backward is ~1.95× forward (vs ~2.84× for MODEL_CONFIG_1)
- Total iteration: 351ms vs 498ms for MODEL_CONFIG_1 (~30% faster)
- Forward is nearly unchanged (87 vs 88ms) — suggests forward cost is dominated by MLP/feedforward ops, not attention (which scales quadratically with context length)
- Backward savings (~32%) and optimizer savings (~41%) are larger, consistent with fewer activations stored for backprop at shorter context

---

### Experiment: NVTX Profiling (Nsight Systems) — MODEL_CONFIG_S

**Model config (MODEL_CONFIG_S):** vocab_size=10,000 · context_length=512 · d_model=768 · num_layers=12 · num_heads=12 · d_ff=3072

**Setup:** same as MODEL_CONFIG_M above

| Phase | Time (ms) |
|---|---|
| Forward pass | 44 |
| Backward pass | 85 |
| Optimizer step | 46 |

**Derived:**
- Backward is ~1.93× forward
- Total iteration: 175ms vs 498ms for MODEL_CONFIG_M (~65% faster)
- Forward is ~2× faster than MODEL_CONFIG_M (44 vs 88ms), consistent with half the layers and smaller d_model/d_ff
- Optimizer step scales closely with model size (46ms vs 160ms) — proportional to number of parameters

---

### Experiment: NVTX Profiling (Nsight Systems) — MODEL_CONFIG_S_SC

**Model config (MODEL_CONFIG_S_SC):** vocab_size=10,000 · context_length=128 · d_model=768 · num_layers=12 · num_heads=12 · d_ff=3072

**Setup:** same as MODEL_CONFIG_S above; context_length reduced from 512 → 128

| Phase | Time (ms) |
|---|---|
| Forward pass | 44 |
| Backward pass | 85 |
| Optimizer step | 46 |

**Derived:**
- Timings are identical to MODEL_CONFIG_S (512 context) — shortening context had no measurable effect on this smaller model
- Contrast with MODEL_CONFIG_M where SC reduced total time by ~30%; suggests the S model's compute is fully dominated by feedforward (MLP) ops, and the attention cost at 512 context was already negligible relative to MLP cost at this model size

---

### Experiment: GeLU Kernel Profiling

**Setup:** single call of each GPU GeLU variant, profiling with CUDA/Nsight-style breakdown

| Variant | Self CPU time | Self CUDA time | Kernel | Kernel time | Launch/sync overhead |
|---|---|---|---|---|---|
| `cuda_gelu` | 634.848 µs | 50.654 µs | `gelu_kernel(float*, float*, int)` | 50.654 µs | `cudaLaunchKernel` 315.359 µs + `cudaDeviceSynchronize` 115.043 µs |
| `triton_gelu` | 586.368 µs | 40.319 µs | `triton_gelu_kernel` | 40.319 µs | `cuLaunchKernelEx` 295.349 µs + `cudaDeviceSynchronize` 19.791 µs |

**Observations:**
- The Triton kernel is measurably faster than the handwritten CUDA kernel, with `triton_gelu_kernel` at ~40.3 µs vs `gelu_kernel` at ~50.7 µs.
- Both variants incur significant non-kernel overhead from tensor allocation and GPU launch/synchronization; the kernel-only portion is a small fraction of total wall time.
- `cuda_gelu` spends more time in `cudaDeviceSynchronize` (115 µs vs 20 µs for Triton), suggesting the Triton path has lower blocking overhead in this profile.
- The allocation path (`aten::empty_like`, `aten::empty_strided`) is also visible in both cases, but the execution-level advantage belongs to Triton for this GeLU workload.
- If the goal is pure kernel performance for a simple elementwise activation, Triton appears to be the better choice here, though overall GPU timing still depends on launch/sync costs.

---

### Experiment: torch.compile vs plain PyTorch attention

**Setup:** same attention module, compare baseline PyTorch vs `torch.compile` on forward and backward profiling.

| Config | Forward peak (MiB) | Backward peak (MiB) | Forward CUDA | Backward CUDA | Key behavior |
|---|---|---|---|---|---|
| dim16_seq256 | 16.73 → 12.93 | 28.98 → 24.80 | 0.123 ms → 0.067 ms | 0.421 ms → 0.333 ms | compile reduces both mem and forward latency |
| dim16_seq4096 | 2088.91 → 1053.00 | 3114.88 → 2073.01 | 11.801 ms → 3.971 ms | 26.914 ms → 11.103 ms | large forward and memory wins, backward also improves |
| dim32_seq8192 | 8306.08 → 4162.27 | 12410.02 → 8242.27 | 39.808 ms → 14.891 ms | 95.842 ms → 223.221 ms | forward wins, backward runtime regresses heavily |
| dim128_seq8192 | 8405.31 → 4309.50 | 12533.31 → 8341.63 | 44.523 ms → 19.121 ms | 99.909 ms → 227.494 ms | same forward gain, backward runtime spike |

**Key findings:**

- `torch.compile` consistently reduced forward peak HBM and forward CUDA time across measured attention shapes.
- For small-to-medium shapes, compile also reduced backward memory and improved backward throughput.
- For larger sequence lengths and model widths, compile produced much faster forward execution but introduced expensive backward kernels, extra DtoD copies, and longer CPU-side profiler totals.
- The compiled backward phase often shows Triton fused kernels and heavier device copy activity, suggesting the new backward graph is more memory- and compute-intensive.
- In effect, `torch.compile` can trade forward kernel fusion for a backward graph that is less efficient: the compiled backward path may produce larger fused kernels, more intermediate buffers, and more device-to-device transfers than the plain PyTorch backward path.
- Because the backward pass dominates the training iteration, it is essential to validate end-to-end forward+backward performance, not just forward gains.

**Practical conclusion:**

Use `torch.compile` when it lowers both forward and backward cost, but be cautious for large attention shapes. The best result is not guaranteed by forward speed alone; if backward runtime or peak activation spikes, retain plain PyTorch or explore shape-specialized compilation, lower precision, or activation checkpointing.

---

## CUDA GPU Kernel Summary

### MODEL_CONFIG_M — Forward Pass (Top 10 Kernels)

| Time% | Total Time | Instances | Avg | Med | Min | Max | StdDev | Kernel |
|---|---|---|---|---|---|---|---|---|
| 60.6% | 50.964 ms | 108 | 471.885 μs | 237.041 μs | 207.745 μs | 906.982 μs | 295.027 μs | ampere_sgemm_128x64_tn |
|  4.4% |  3.690 ms | 190 |  19.421 μs |  18.736 μs |  16.416 μs |  26.752 μs |   2.177 μs | elementwise_kernel (MulFunctor) |
|  4.3% |  3.658 ms |  30 | 121.919 μs | 121.041 μs | 120.033 μs | 135.009 μs |   2.788 μs | vectorized_elementwise_kernel (MulFunctor) |
|  4.3% |  3.655 ms |  15 | 243.671 μs | 256.034 μs | 223.394 μs | 258.593 μs |  15.259 μs | ampere_sgemm_128x128_nn |
|  2.9% |  2.450 ms |  15 | 163.320 μs | 163.009 μs | 160.129 μs | 170.593 μs |   2.780 μs | vectorized_elementwise_kernel (BUnaryFunctor/MulFunctor) |
|  2.9% |  2.432 ms |  15 | 162.100 μs | 161.345 μs | 160.257 μs | 169.473 μs |   2.373 μs | vectorized_elementwise_kernel (exp_kernel) |
|  2.9% |  2.416 ms |  15 | 161.048 μs | 160.705 μs | 158.209 μs | 168.289 μs |   3.017 μs | elementwise_kernel (DivFunctor) |
|  2.8% |  2.382 ms |  15 | 158.797 μs | 158.145 μs | 156.609 μs | 161.857 μs |   1.683 μs | elementwise_kernel (CUDAFunctor_add) |
|  2.7% |  2.231 ms |  15 | 148.700 μs | 150.753 μs | 140.833 μs | 155.297 μs |   4.633 μs | ampere_sgemm_128x128_tn |
|  2.2% |  1.879 ms |  15 | 125.274 μs | 124.800 μs | 122.817 μs | 131.073 μs |   2.233 μs | elementwise_kernel (where_kernel / causal mask) |

**Notes:**
- `ampere_sgemm_128x64_tn` alone is 60.6% of forward time — a single cuBLAS kernel dominates; 108 instances across 24 layers (×2 projections per layer + feedforward)
- Ranks 2–3 are both MulFunctor elementwise kernels with different tile configs (128-wide vs vectorized-4) — likely attention score scaling and dropout
- Ranks 4, 9: `ampere_sgemm_128x128_nn/tn` handle larger square-ish matmul shapes, likely the feedforward up-projection (d_model→d_ff)
- Ranks 6–8 (exp, div, add) each have exactly 15 instances = one per layer — these are the softmax steps (exp → sum+div → weighted sum)
- Rank 10 (`where_kernel`, 15 instances) is the causal mask application before softmax, one per attention layer
- Contrast with backward: forward has one dominant SGEMM kernel (60.6%) whereas backward spreads across 5 SGEMM variants (~56%) — backprop needs multiple transposition layouts to compute both input and weight gradients

---

### MODEL_CONFIG_M — Backward Pass (Top 10 Kernels)

| Time% | Total Time | Instances | Avg | Med | Min | Max | StdDev | Kernel |
|---|---|---|---|---|---|---|---|---|
| 13.7% | 32.486 ms |  61 | 532.564 μs | 236.834 μs | 233.953 μs |   1.9CZ84 ms | 356.836 μs | ampere_sgemm_128x64_tn |
| 12.1% | 28.671 ms |  33 | 868.817 μs | 829.286 μs | 820.326 μs |   2.055 ms | 213.203 μs | cutlass_80_simt_sgemm_256x128_8x4_nn_align1 |
| 10.8% | 25.528 ms |  77 | 331.526 μs | 215.809 μs | 213.538 μs | 796.357 μs | 226.403 μs | ampere_sgemm_128x64_nn |
| 10.1% | 23.917 ms |  77 | 310.606 μs | 204.545 μs | 201.313 μs | 739.013 μs | 208.593 μs | cutlass_80_simt_sgemm_128x256_8x4_nt_align1 |
|  9.1% | 21.523 ms |  32 | 672.597 μs | 669.524 μs | 663.364 μs | 696.612 μs |  10.662 μs | cutlass_80_simt_sgemm_128x64_8x5_nt_align1 |
|  8.7% | 20.531 ms | 241 |  85.189 μs |  32.833 μs |   1.376 μs | 270.626 μs |  75.783 μs | vectorized_elementwise_kernel (MulFunctor) |
|  4.8% | 11.389 ms | 252 |  45.196 μs |  30.080 μs |  12.544 μs | 315.234 μs |  61.133 μs | vectorized_elementwise_kernel (CUDAFunctor_add) |
|  3.9% |  9.178 ms |  56 | 163.888 μs | 163.025 μs | 160.577 μs | 183.937 μs |   5.036 μs | elementwise_kernel (DivFunctor) |
|  3.1% |  7.441 ms |  31 | 240.022 μs | 239.201 μs | 236.993 μs | 248.450 μs |   3.413 μs | ampere_sgemm_128x128_nt |
|  3.1% |  7.265 ms | 219 |  33.173 μs |  22.945 μs |  19.328 μs | 172.193 μs |  22.266 μs | elementwise_kernel (direct_copy) |

**Notes:**
- Top 5 kernels (~56% of backward time) are all SGEMM variants — backward pass is dominated by matrix multiplications for gradient computation (weight gradients and input gradients)
- cuBLAS kernels (`ampere_sgemm_*`) handle standard tile sizes; cutlass `*_align1` variants handle non-aligned shapes (likely feedforward gradient projections with d_ff=4096)
- The `tn`, `nn`, `nt` suffixes indicate transposition: backward requires both A·Bᵀ and Aᵀ·B products to compute input and weight gradients respectively, hence more kernel variants than forward
- High StdDev on `ampere_sgemm_128x64_tn` (357 μs vs 237 μs median) suggests bimodal dispatch across different layer sizes
- Rank 6 (MulFunctor, 241 instances, 8.7%) — elementwise multiply for gradient masking (dropout, attention mask backward)
- Ranks 7–8: elementwise add (residual gradient accumulation) and div (softmax backward normalization)
- Rank 10 (direct_copy, 219 instances) — tensor copies for gradient buffers and saved activations needed for backward

---

### MODEL_CONFIG_M — Optimizer Step (Top 10 Kernels)

| Time% | Total Time | Instances | Avg | Med | Min | Max | StdDev | Kernel |
|---|---|---|---|---|---|---|---|---|
| 16.5% | 25.744 ms | 860 |  29.935 μs |  12.448 μs |   1.248 μs | 244.609 μs |  34.407 μs | vectorized_elementwise_kernel (CUDAFunctor_add) |
| 14.1% | 21.921 ms | 1156 |  18.962 μs |  10.784 μs |   1.184 μs |  99.201 μs |  17.701 μs | vectorized_elementwise_kernel (AUnaryFunctor/MulFunctor) |
|  8.8% | 13.644 ms |  43 | 317.299 μs | 214.497 μs | 213.601 μs | 774.757 μs | 216.547 μs | ampere_sgemm_128x64_nn |
|  8.5% | 13.223 ms |  16 | 826.449 μs | 825.094 μs | 818.854 μs | 836.454 μs |   5.639 μs | cutlass_80_simt_sgemm_256x128_8x4_nn_align1 |
|  8.2% | 12.799 ms |  43 | 297.654 μs | 203.170 μs | 201.537 μs | 717.381 μs | 199.891 μs | cutlass_80_simt_sgemm_128x256_8x4_nt_align1 |
|  6.9% | 10.693 ms |  16 | 668.338 μs | 667.188 μs | 663.396 μs | 676.100 μs |   4.793 μs | cutlass_80_simt_sgemm_128x64_8x5_nt_align1 |
|  6.3% |  9.842 ms | 118 |  83.407 μs |  32.176 μs |   1.376 μs | 251.106 μs |  79.947 μs | vectorized_elementwise_kernel (MulFunctor) |
|  3.2% |  5.011 ms | 186 |  26.939 μs |  12.128 μs |   1.504 μs | 146.497 μs |  26.969 μs | vectorized_elementwise_kernel (DivFunctor) |
|  2.8% |  4.345 ms | 393 |  11.055 μs |   6.016 μs |   1.024 μs |  87.521 μs |  13.663 μs | vectorized_elementwise_kernel (FillFunctor) |
|  2.6% |  4.069 ms |  25 | 162.744 μs | 162.753 μs | 160.929 μs | 164.961 μs |     982 ns | elementwise_kernel (DivFunctor) |

**Notes:**
- Top 2 kernels are elementwise ops (add and scalar-mul) — Adam's parameter update rule (`p += -lr * m_hat / (v_hat + eps)`) is entirely elementwise and dominates at ~31% combined
- Ranks 3–6 are SGEMM kernels (~32% combined) — unexpected for an optimizer step; these are likely gradient all-reduce or gradient norm computation rather than weight updates (AdamW has no matmuls)
- Rank 9 (FillFunctor, 393 instances) — zeroing gradient buffers (`optim.zero_grad()` call, which runs at the top of the loop just before the optimizer step is measured, or lazy gradient reset)
- Ranks 8 (DivFunctor, 186 instances) and 10 (DivFunctor, 25 instances) — bias-correction division in Adam: `m_hat = m / (1 - β₁ᵗ)` and `v_hat = v / (1 - β₂ᵗ)`
- High instance counts (860, 1156) for the top two kernels reflect one call per parameter tensor per optimizer step across all 24 layers

---

## Self-Attention Analysis — MODEL_CONFIG_M (Forward Pass, Single Layer)

**Config:** B=4, T=512, H=16, d_head=64, d_model=1024

### Kernel Table (all kernels in one self-attention layer)

| Time% | Total Time | Instances | Kernel | Role |
|---|---|---|---|---|
| 34.7% | 636.781 μs | 3 | ampere_sgemm_128x64_tn | matmul (Q·Kᵀ, scores·V, output proj) |
|  8.7% | 159.107 μs | 1 | ampere_sgemm_128x128_tn | matmul (QKV projection) |
| 10.2% | 186.146 μs | 10 | elementwise_kernel (MulFunctor) | attention score scaling (÷√d_head) |
|  9.1% | 166.660 μs |  1 | vectorized_elementwise_kernel (BUnaryFunctor/Mul) | softmax: scale |
|  9.0% | 164.676 μs |  1 | vectorized_elementwise_kernel (exp_kernel) | softmax: exp |
|  8.7% | 160.259 μs |  1 | elementwise_kernel (CUDAFunctor_add) | softmax: sum accumulation |
|  6.7% | 123.618 μs |  1 | elementwise_kernel (where_kernel) | causal mask application |
|  4.8% |  87.938 μs |  1 | reduce_kernel (MaxOps) | softmax: max reduction (numerical stability) |
|  3.0% |  54.336 μs |  2 | CatArrayBatchedCopy | tensor concat (likely K/V cache or head split) |
|  2.9% |  53.632 μs |  4 | vectorized_elementwise_kernel (CUDAFunctor_add) | residual add |
|  1.1% |  20.992 μs |  1 | vectorized_elementwise_kernel (pow_kernel) | layernorm: x² |
|  0.6% |  10.465 μs |  1 | reduce_kernel (MeanOps) | layernorm: mean reduction |
|  0.2% |   3.232 μs |  1 | elementwise_kernel (CompareFunctor) | mask comparison |
|  0.1% |   1.536 μs |  1 | vectorized_elementwise_kernel (FillFunctor) | zero fill |
|  0.1% |   1.536 μs |  1 | vectorized_elementwise_kernel (rsqrt_kernel) | layernorm: 1/√var |
|  0.1% |   1.408 μs |  1 | vectorized_elementwise_kernel (CUDAFunctorOnSelf_add) | layernorm: bias add |
|  0.1% |   1.376 μs |  1 | elementwise_kernel_with_index (arange) | position index generation |

### Runtime Comparison: Matmul vs Softmax

| Group | Kernels | Total Time |
|---|---|---|
| Matmul | sgemm_128x64_tn (×3) + sgemm_128x128_tn (×1) | 636.781 + 159.107 = **795.9 μs** |
| Softmax pipeline | scale + exp + sum + causal mask + max reduction | 166.660 + 164.676 + 160.259 + 123.618 + 87.938 = **703.2 μs** |

Runtime ratio (matmul : softmax) ≈ **1.13 : 1**

### FLOP Comparison: Matmul vs Softmax

**Matmul FLOPs** (all linear projections + attention products, per layer):
- QKV projection: 2 · B·T · d_model · 3·d_model = 2 · 2048 · 1024 · 3072 ≈ **12.88 GFLOPs**
- Q·Kᵀ: 2 · B·H · T · T · d_head = 2 · 64 · 512 · 512 · 64 ≈ **2.15 GFLOPs**
- scores·V: same ≈ **2.15 GFLOPs**
- Output projection: 2 · B·T · d_model · d_model = 2 · 2048 · 1024 · 1024 ≈ **4.29 GFLOPs**
- **Total: ~21.5 GFLOPs**

**Softmax FLOPs** (operating on attention score matrix of shape B·H·T·T = 4·16·512·512 = 16.8M elements):
- Per row of length T: max(T) + subtract(T) + exp(T) + sum(T) + divide(T) ≈ 5·T ops
- Total rows: B·H·T = 4·16·512 = 32,768
- **Total: ~5 · 512 · 32,768 ≈ 84 MFLOPs**

FLOP ratio (matmul : softmax) ≈ **21,500 : 84 ≈ 255 : 1**

### Key Takeaway

| | Matmul | Softmax | Ratio |
|---|---|---|---|
| FLOPs | ~21.5 GFLOPs | ~84 MFLOPs | **255 : 1** |
| Runtime | ~796 μs | ~703 μs | **1.13 : 1** |

Matmul has **255× more FLOPs** than softmax but takes only **1.13× more time**. This gap exists because:
- Matmuls are **compute-bound**: data is reused heavily in registers/shared memory; FLOP throughput is near peak GPU compute (~312 TFLOPS for A100)
- Softmax is **memory-bandwidth-bound**: each step (max, exp, sum, div) reads and writes the full attention matrix (B·H·T·T ≈ 67 MB at fp32); with ~900 GB/s bandwidth, 5 passes ≈ 370 μs — matching observations

This is the core motivation for **FlashAttention**: by fusing the softmax steps and tiling over the T×T matrix, it avoids multiple full reads/writes of the attention matrix, making softmax effectively bandwidth-free relative to the surrounding matmuls.

---

## Mixed Precision Observations

**Verified with `benchmark_with_precision.py`:**

```
param fc1.weight:  torch.float32
param ln.weight:   torch.float32
param ln.bias:     torch.float32
param fc2.weight:  torch.float32

Forward after fc1:  x.dtype = torch.float16
Forward after ln:   x.dtype = torch.float32
Forward after fc2:  x.dtype = torch.float16

logits dtype:  torch.float16
loss dtype:    torch.float32

grad for fc1.weight:  torch.float32
grad for ln.weight:   torch.float32
grad for ln.bias:     torch.float32
grad for fc2.weight:  torch.float32
```

**Key findings:**

1. **Model parameters are always stored as FP32.** `torch.autocast` does not modify the stored weights. At the point of a matmul, weights are downcast to FP16 on-the-fly (a temporary cast, not written back). The FP32 master copy is the source of truth for the optimizer.

2. **Matmul: FP16 input → FP16 output.** Both the (transiently-cast) weight and the activation are FP16 entering the linear layer; the output activation is FP16. Tensor cores accumulate in FP32 internally but deliver FP16 output by default under autocast.

3. **Reductions and accumulations output FP32.** LayerNorm (mean/variance reduction) and the loss computation both output FP32 — autocast keeps these in FP32 to avoid the accumulation precision failure. The FP32 activation is then downcast to FP16 on entry to the next matmul.

4. **Gradients are FP32.** Even though activations flow as FP16 during the forward pass, all `.grad` tensors are FP32. Autograd upcasts during the backward pass so that gradient accumulation is numerically stable — same reason as the optimizer: gradients are summed across many terms and must not stall.

**Data flow (forward + backward):**
```
FP32 weights ──(on-the-fly cast)──► FP16
                                      │
                                   matmul ──► FP16 activations
                                                   │
                                              LayerNorm (FP32) ──► FP32
                                                                     │
                                                             (on-the-fly cast)
                                                                     │
                                                                  matmul ──► FP16 logits
                                                                                  │
                                                                             loss (FP32)
                                                                                  │
                                                                    backward ──► FP32 gradients
                                                                                  │
                                                                    optimizer updates FP32 weights
```

---

## Memory Profiling (PyTorch Memory Visualizer)

### Memory breakdown by phase (Adam optimizer, fp32)

Let P = parameter memory. Adam stores two moment buffers (m, v), each the same size as parameters.

| Phase | Contents | Formula |
|---|---|---|
| Forward pass peak (end of fwd) | params + optimizer states + all saved activations | P + 2P + A |
| End of backward | params + optimizer states + gradients (activations freed) | P + 2P + P = 4P |
| Optimizer step | same as end of backward | 4P |
| Inference (`torch.no_grad()`) | params only | P |

### Memory over a training iteration

- **Peak occurs at the forward/backward boundary** — not strictly "during forward." Autograd saves intermediate activations layer by layer as the forward pass runs, accumulating them all simultaneously so backward can use them. The very first step of backward sees the same memory as the last step of forward. This is the highest-memory point of the iteration.
- **Backward decreases memory monotonically.** Backward runs in reverse layer order; each layer consumes its saved activations to compute gradients then frees them. By the time the optimizer step runs, only params + gradients + optimizer states remain (~4P).
- **Whether optimizer phase is ~50% of peak depends on activation size.** If A ≈ P, optimizer/peak = 4/6 ≈ 67%. If A ≈ 5P, it's 4/10 = 40%. The observed ~50% puts activations at roughly the same magnitude as all non-activation memory combined.

Memory timeline within one training iteration:
```
forward (growing) ──► fwd/bwd boundary (peak: 3P+A) ──► backward (shrinking) ──► optimizer (4P, ~50% of peak)
```

### Inference vs. training memory

- **Inference only (`torch.no_grad()`) uses ~10% of the training peak** and is flat/stable throughout.
- The entire activation memory is absent — PyTorch skips saving intermediate tensors when gradient tracking is disabled. No gradients or optimizer states exist either.
- Remaining memory is parameters only (~P). If training peak is ~10P (e.g., A = 7P, non-activation = 3P), inference at P is exactly 10%.

| Mode | Relative Memory | Stable? |
|---|---|---|
| Training (fwd/bwd boundary peak) | 100% | No — grows through fwd, drops through bwd |
| Training (optimizer phase) | ~50% | Briefly stable |
| Inference (`torch.no_grad()`) | ~10% | Yes — flat throughout |

### Measured: MODEL_CONFIG_M

**Model:** vocab_size=10,000 · d_model=1024 · num_layers=24 · num_heads=16 · d_ff=4096

| Config | context_length | Training peak | Activation % of peak | Activation (abs) | Non-activation (abs) | Inference peak | Inference / Training |
|---|---|---|---|---|---|---|---|
| MODEL_CONFIG_M     | 512 | 13.5 GB | ~50% | ~6.75 GB | ~6.75 GB | 2.0 GB | ~15% |
| MODEL_CONFIG_M_SC  | 128 |  6.5 GB | ~70% | ~4.55 GB | ~1.95 GB | 1.7 GB | ~26% |

**Derived observations:**

- **Inference ≈ params only (~2 GB).** Both configs show inference at ~1.7–2.0 GB regardless of context length, consistent with parameters alone (P ≈ 2 GB; no gradients, optimizer states, or saved activations).
- **Shortening context 4× (512→128) cuts training peak by ~52%** (13.5→6.5 GB), driven almost entirely by fewer saved activations (6.75→4.55 GB, −33%). Non-activation memory should be fixed at ~4P ≈ 8 GB after backward, but the measured ~6.75 GB vs ~1.95 GB non-activation split suggests the visualizer is measuring the fwd/bwd peak where optimizer states may not yet be fully allocated, or some temporary buffers scale with context.
- **Activation fraction rises from 50% → 70% with shorter context** — the absolute activation shrinks, but the non-activation floor (params + optimizer states) stays fixed, so activations become proportionally smaller relative to total peak.
- **Activation scales sub-linearly with context** (−33% for 4× context reduction). If activations were purely linear in T (MLP-dominated), we'd expect −75%; the shallower drop suggests either attention's T²-scaled buffers still contribute at 512, or residual/embedding activations add a context-independent baseline.

### Effect of `torch.compile` on Memory (MODEL_CONFIG_M, context_length=512)

| Phase | Memory |
|---|---|
| Forward peak (fwd/bwd boundary) | 12 GB |
| Backward peak (end of backward) | 6.5 GB |

**Comparison to no-compile baseline (13.5 GB forward peak):**
- Forward peak drops from 13.5 GB → 12 GB (−1.5 GB, −11%). This matches the reduction in saved activations: kernel fusion eliminates duplicate intermediate saves (e.g., rsqrt and x saved once instead of twice in RMSNorm), cutting activation memory across all 24 layers.
- Backward "peak" of 6.5 GB is the floor after all activations have been freed, leaving params + gradients + optimizer states ≈ 4P. This is consistent with P ≈ 1.5–2 GB for this model size.
- The two peaks are no longer equal: in eager mode the fwd/bwd boundary is both the end of forward and start of backward, so one number describes both. With `torch.compile` the fwd peak (12 GB, all activations resident) and the bwd floor (6.5 GB, activations freed) differ by the amount of activation memory saved by the fused kernels (~5.5 GB), demonstrating how much activation memory is consumed and released during the backward pass.

**Why the 1.5 GB savings is modest (and expected):**

`torch.compile` only helps ops that were previously split into multiple eager graph nodes. The dominant activation memory consumers are already single ops and are unaffected:

- **Attention score matrix** `[B, H, T, T]` — already a single op (`scaled_dot_product_attention`). Compile cannot fuse further; the full ~1.6 GB across 24 layers is still saved for backward.
- **MLP activations** `[B, T, d_ff]` — the large intermediate after the up-projection (~32 MB × 24 layers ≈ 768 MB) still must be saved for backward.

What compile does help: RMSNorm and other pointwise chains. With ~48 RMSNorm layers (2 per block × 24 layers), each saving ~16 MB fewer tensors (eliminating the duplicate x and rsqrt saves at d_model=1024), that is ~750 MB — roughly half the observed 1.5 GB total savings, with the remainder from other fused elementwise ops.

**To get larger activation savings, compile alone is insufficient.** The right tools are:
- **Gradient checkpointing** — recompute activations during backward instead of saving them; trades compute for memory.
- **FlashAttention** — eliminates the T×T attention matrix entirely via online softmax tiling, saving ~1.6 GB of the dominant per-layer allocation.

---

### Effect of Mixed Precision (`torch.autocast`)

**Inference:** memory *increased* by ~0.3 GB.
- Counterintuitive because FP16 activations are smaller, but parameters are still stored as FP32 master copies. PyTorch caches FP16 casts of the weights within the autocast context to avoid redundant recasting — that cache adds memory. Meanwhile, activation savings during inference are near zero: with `torch.no_grad()` activations are consumed and freed layer by layer, so there is almost nothing to halve.
- Net result: FP16 weight cache overhead > activation savings → memory increases slightly.
- **Mixed precision is a training optimization, not an inference one.** To actually reduce inference memory, store the model directly in FP16 (`model.half()`), which cuts parameter memory 2× with no cache overhead. `torch.autocast` is designed for the training case where the FP32 master weights must be preserved for the optimizer.

**Training — gradients: unchanged.**
- Gradients are always FP32 regardless of autocast, so gradient memory is identical to full-precision training. The GradScaler scales the loss value but does not change gradient dtype.

**Training — activations: reduced by ~50%.**
- Saved activations (stored at the fwd/bwd boundary for use during backprop) are now FP16 instead of FP32 — half the size. This is the main memory benefit of mixed precision training.
- Updated formula for training peak: P + 2P + A/2 = 3P + A/2 (vs. 3P + A full-precision).

| Component | Full Precision | Mixed Precision | Change |
|---|---|---|---|
| Parameters | FP32 (P) | FP32 (P) | none |
| Optimizer states | FP32 (2P) | FP32 (2P) | none |
| Gradients | FP32 (P) | FP32 (P) | none |
| Saved activations | FP32 (A) | FP16 (A/2) | **−50%** |
| Inference overhead | P | P + ~0.3 GB | **+0.3 GB** |

### Largest single allocations: `scaled_dot_product_attention`

**Observation:** The PyTorch memory visualizer shows the largest individual allocations during the forward pass come from `scaled_dot_product_attention`. FlashAttention is NOT being used — the naive math backend is active, which fully materializes the attention score matrix in HBM.

**This is expected.** Every other tensor in the transformer scales as O(B·T·d_model) — linear in sequence length. The attention score matrix is the only O(B·H·T²) tensor in the model:

| Tensor | Shape | Size per layer (B=4, T=512) |
|---|---|---|
| MLP activation | `[B, T, d_ff]` = `[4, 512, 4096]` | ~32 MB |
| Attention score matrix | `[B, H, T, T]` = `[4, 16, 512, 512]` | ~67 MB |

At T=512, the attention matrix is **2× larger than the MLP activation per layer** — purely from the T² factor, despite d_ff=4096 >> d_head=64. Across 24 layers: ~67 MB × 24 ≈ **1.6 GB** just for attention score matrices.

With the naive backend, `scaled_dot_product_attention` materializes the full `[B, H, T, T]` softmax output and saves it for the backward pass (needed to compute the gradient through softmax). That saved tensor is the largest per-layer allocation visible in the profiler — it is the direct cost of not using FlashAttention.

**This is precisely why FlashAttention exists.** It fuses the attention computation into tiles and never materializes the full T×T matrix in HBM. The backward pass recomputes attention scores on-the-fly from saved per-row log-sum-exp statistics (O(B·H·T)), bringing attention memory from O(T²) back to O(T). The compute cost increases slightly (recomputation), but the memory savings are the entire T×T matrix per layer.

---

## Autograd Saved Tensors — Effect of `torch.compile` on RMSNorm

**Setup:** `RMSNorm(d_model=2560)` on input `x` of shape `[4, 512, 2560]` with `requires_grad=True`. Saved tensors observed via `torch.autograd.graph.saved_tensors_hooks`.

### Without `torch.compile`

```
Saving activation: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None         # x
Saving activation: shape=torch.Size([4, 512, 1]),    dtype=torch.float32, grad_fn=<RsqrtBackward0>  # rsqrt(mean(x²)+ε)
Saving activation: shape=torch.Size([4, 512, 1]),    dtype=torch.float32, grad_fn=<RsqrtBackward0>  # same rsqrt, saved again for second backward node
Saving activation: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None         # x (saved again)
Saving activation: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0>   # x_normalized = x * rsqrt
Saving activation: shape=torch.Size([2560]),         dtype=torch.float32, grad_fn=None         # weight
```

6 tensors saved — one per autograd graph node that needs inputs for its backward function.

### With `torch.compile`

```
Saving activation: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None   # x
Saving activation: shape=torch.Size([2560]),         dtype=torch.float32, grad_fn=None   # weight
Saving activation: shape=torch.Size([4, 512, 1]),    dtype=torch.float32, grad_fn=None   # rsqrt (no grad_fn — fused)
```

3 tensors saved — half the storage.

### Key Takeaway

`torch.compile` fuses the RMSNorm forward pass into a single kernel. In the eager (unfused) path, autograd inserts a separate graph node for each op (mul, rsqrt, mul again, etc.), and each node independently saves the tensors it needs for its own backward — resulting in duplicate saves (rsqrt saved twice, x saved twice). With compilation, the entire forward is one fused op with one combined backward, so autograd only saves what is strictly necessary once: x, weight, and rsqrt. Kernel fusion reduces saved activation count by ~50% for RMSNorm.

### Additional Observations

**Loading order no longer reverses saving order.**
In eager mode the autograd graph is a DAG of individual ops. Backward traverses that DAG in reverse topological order, so the last forward op's backward node runs first and loads its saved tensors first — this is what produces the reversed loading pattern. With `torch.compile`, all ops are collapsed into a single fused kernel with a single custom backward. That backward loads its saved tensors in whatever order the compiled code accesses them; there is no sub-op DAG to impose a reverse-order constraint.

**All saved tensors have `grad_fn=None`.**
In eager mode, saved intermediates like `rsqrt(mean(x²)+ε)` carry a `grad_fn` because autograd may need to differentiate *through* them — they are live nodes in the computation graph, and their gradient history must be preserved. With `torch.compile`, the backward is a pre-derived fused kernel that directly implements the analytic gradient formula for the whole RMSNorm operation. The saved tensors are just raw numerical inputs to that formula; the kernel does not backprop through them. They are therefore stored as detached values (`grad_fn=None`), replacing graph structure with compiled code.

---

## Gradient Checkpointing

**Setup:** 4× TransformerBlock (d_model=2560, d_ff=10240, num_heads=32, context_length=2048) with `torch.compile`. Activation memory measured via `saved_tensors_hooks` over the forward pass. Checkpointing applied to 2-block segments: `checkpoint(two_blocks, x)` × 2.

| Config | Saved activation memory |
|---|---|
| No checkpointing | 14 GB |
| Checkpointing (`use_reentrant=False`) | 0.16 GB |
| Checkpointing (`use_reentrant=True`) | ~14 GB (same as no checkpointing) |

**~87× reduction** from no-checkpointing to `use_reentrant=False`.

### Why `use_reentrant=True` fails with `torch.compile`

`use_reentrant=True` suppresses intermediate saves by running the checkpointed forward under `torch.no_grad()`. With `torch.compile`, the function is compiled into a fused kernel that bypasses the standard autograd save mechanism — the compiled graph does not properly respond to the `no_grad` suppression, so intermediates are still recorded. Result: no memory savings.

`use_reentrant=False` uses `torch.autograd.graph.saved_tensors_hooks` internally to intercept and discard saved tensors during the checkpointed forward. This operates at a lower level that `torch.compile` respects, so discarding actually takes effect.

**Practical rule: always use `use_reentrant=False`, especially with `torch.compile`.**

### Why the reduction is so large (~87×)

Without checkpointing, all intermediate activations from 4 blocks are saved: attention score matrices `[B, H, T, T]` = `[4, 32, 2048, 2048]` (~2 GB each at fp32) plus MLP and residual activations across all 4 blocks account for the 14 GB total.

With `use_reentrant=False`, each `checkpoint(two_blocks, x)` call saves only the input `x` to that 2-block segment (`[4, 2048, 2560]` ≈ 0.08 GB) and discards all intermediates inside. Two checkpoints = ~0.16 GB total. During backward, the 2-block forward is recomputed on-the-fly to recover the needed intermediates — trading compute for memory.

---

### Experiment: Varying Checkpoint Granularity — MODEL_CONFIG_L

**Model:** d_model=1280 · d_ff=5120 · num_layers=36 · num_heads=20 · context_length=512 · batch_size=4 · `torch.compile` enabled.

Granularity = number of blocks per checkpoint segment. More blocks per segment → fewer segments → less saved activation memory → more recomputation during backward.

#### Saved activation memory (forward pass)

| Granularity | Segments | Saved activations | Expected (inputs only) |
|---|---|---|---|
| BLOCKS_1  | 36 | 0.35 GB | 0.35 GB |
| BLOCKS_2  | 18 | 0.18 GB | 0.18 GB |
| BLOCKS_4  |  9 | 0.09 GB | 0.09 GB |
| BLOCKS_6  |  6 | 0.06 GB | 0.06 GB |
| BLOCKS_9  |  4 | 0.04 GB | 0.04 GB |
| BLOCKS_12 |  3 | 0.03 GB | 0.03 GB |
| BLOCKS_18 |  2 | 0.02 GB | 0.02 GB |
| BLOCKS_36 |  1 | 0.01 GB | 0.01 GB |

Saved activations = segment inputs only. Measured == Expected in every case, confirming that `torch.compile` + `use_reentrant=False` discards all intermediates inside each segment and retains only the segment boundary tensor.

Input tensor size: `[4, 512, 1280]` × 4 bytes = 10,485,760 bytes ≈ 0.01 GB per segment. Total = 0.01 GB × num_segments, so saved memory scales exactly linearly with the number of segments (inversely with segment size).

#### Peak memory during backward pass

| Granularity | Segments | Peak memory |
|---|---|---|
| BLOCKS_1  | 36 |  1.0 GB |
| BLOCKS_2  | 18 |  1.1 GB |
| BLOCKS_4  |  9 |  1.8 GB |
| BLOCKS_6  |  6 |  2.4 GB |
| BLOCKS_9  |  4 |  3.4 GB |
| BLOCKS_12 |  3 |  4.5 GB |
| BLOCKS_18 |  2 |  6.5 GB |
| BLOCKS_36 |  1 | 13.0 GB |

**Key takeaway:** BLOCKS_36 (one big checkpoint) peaks at 13 GB — the entire forward is recomputed inside backward, materializing all 36 blocks' activations simultaneously. BLOCKS_1 (one block per segment) peaks at only 1 GB — during backward, only one block's activations are live at a time (recomputed, used for grad, freed). The peak is dominated by that one recomputed block, not the full model.

This is the fundamental memory–compute tradeoff of gradient checkpointing: finer granularity (smaller segments) reduces peak memory but multiplies recompute work, since each segment is recomputed once during backward.

---

## CPU/GPU Time Profiling (`torch.profiler`)

**Setup:** `torch.profiler.profile` with CPU + CUDA activities, 1 warmup iteration. Operations run on 1D tensors of size 2048.

**Note on CUDA total double-counting:** The `CUDA total` column for CPU-side ops (e.g. `aten::add`, `aten::dot`) is inflated 2× relative to actual GPU time. The same kernel duration is attributed both directly to the op and again via the `Activity Buffer Request` child event. The correct GPU time is the `Self CUDA` value on the raw kernel rows (e.g. `vectorized_elementwise_kernel`, `dot_kernel`), or equivalently the `Self CUDA time total` footer.

### Baseline: Sleep (no operation)

```
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------
      cudaDeviceSynchronize         0.78%      21.531us         0.78%      21.531us      10.766us             2
    Activity Buffer Request        99.22%       2.734ms        99.22%       2.734ms       2.734ms             1
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.755ms
```

No GPU work. Total CPU time ~2.8 ms — almost entirely the profiler's CUDA activity buffer flush. This is the instrumentation floor present in every run.

### Add function (`aten::add`, dim=2048)

```
Self CPU time total: 2.101ms    Self CUDA time total: 1.920us

aten::add            Self CPU: 53 μs   Self CUDA: 1.920 μs
Activity Buffer Req  Self CPU: 1.958ms  Self CUDA: 1.920 μs  (profiler overhead)
vectorized_elementwise_kernel (CUDAFunctor_add)  Self CUDA: 1.920 μs
cudaLaunchKernel     Self CPU: 21 μs
cudaDeviceSynchronize Self CPU: 68 μs
```

One GPU kernel dispatched. Actual GPU work: **1.920 μs**.

### MatMul / dot product (`aten::dot`, dim=2048) — run 1

```
Self CPU time total: 2.722ms    Self CUDA time total: 3.840us

aten::dot            Self CPU: 228 μs  Self CUDA: 3.840 μs
Activity Buffer Req  Self CPU: 2.438ms Self CUDA: 3.840 μs  (profiler overhead)
dot_kernel           Self CUDA: 2.016 μs  (52.5%)
reduce_1Block_kernel Self CUDA: 1.824 μs  (47.5%)
cudaLaunchKernel     Self CPU: 27 μs  (2 calls)
```

Two cuBLAS kernels dispatched. Actual GPU work: **3.840 μs** (2.016 + 1.824).

### MatMul / dot product (`aten::dot`, dim=2048) — run 2

```
Self CPU time total: 3.026ms    Self CUDA time total: 4.352us

aten::dot            Self CPU: 25 μs   Self CUDA: 4.352 μs
Activity Buffer Req  Self CPU: 2.558ms Self CUDA: 4.352 μs  (profiler overhead)
dot_kernel           Self CUDA: 2.336 μs  (53.7%)
reduce_1Block_kernel Self CUDA: 2.016 μs  (46.3%)
cudaLaunchKernel     Self CPU: 359 μs  (2 calls — CPU jitter spike)
```

### Key observations

- **`Activity Buffer Request` (~2–2.7 ms) is the profiler floor** — present even in Sleep with no GPU work. It dominates 80–99% of reported CPU time and is pure instrumentation cost; ignore it when reasoning about real performance.
- **Actual GPU kernel time is ~2–4 μs** for these 2048-element ops. The GPU is barely utilized at this scale.
- **`aten::add` dispatches 1 kernel; `aten::dot` dispatches 2** — `dot_kernel` (main multiply-accumulate) and `reduce_1Block_kernel` (final partial-sum reduction). The reduction adds ~47% overhead vs. a pure elementwise op.
- **`cudaLaunchKernel` CPU time is ~20–27 μs normally but can spike to ~360 μs** (run 2) due to OS scheduler jitter. This is the dominant variable in CPU-side dispatch cost.
- **`aten::dot` CPU self time (228 μs run 1 vs. 25 μs run 2)** shows similar jitter in the PyTorch dispatch path itself — the two runs are the same operation with no algorithmic difference.

### `torch.cdist` (pairwise Euclidean distance)

```
Self CUDA time total: 1.172ms

aten::mm + ampere_sgemm_128x64_tn   Self CUDA: 834 μs   (71.2%)
aten::cat + CatArrayBatchedCopy     Self CUDA:  87 μs   ( 7.4%)
aten::pow + vectorized_elementwise  Self CUDA:  78 μs   ( 6.6%)
aten::sum + reduce_kernel           Self CUDA:  48 μs   ( 4.1%)
aten::sqrt_ + sqrt_kernel           Self CUDA:  42 μs   ( 3.6%)
```

**Algorithm:** `cdist` does not compute differences elementwise. Instead it uses the identity:

```
||x - y||² = ||x||² + ||y||² - 2 · x·yᵀ
```

- `pow` + `sum` — compute per-row squared norms ||x||² and ||y||²
- `mm` — compute the cross-term matrix x·Yᵀ  (one matmul over all pairs at once)
- combine norms and cross-terms, then `sqrt_` in-place
- `cat` — concatenate intermediate norm vectors before combining

**Key takeaway:** 71% of GPU time is the single `ampere_sgemm_128x64_tn` matmul. The norm computation (pow+sum+sqrt) adds ~14% total. `cdist` is effectively a matmul — the pairwise distance structure collapses into a single GEMM via the algebraic identity, which is why it can be highly efficient on GPU despite computing O(N²) distances.

### Softmax (`aten::softmax`, dim=2048)

```
Self CUDA time total: 105.854us

aten::add              Self CUDA: 60.959 μs  (57.59%)
  vectorized_elementwise_kernel (CUDAFunctor_add)       Self CUDA: 60.959 μs
aten::_softmax         Self CUDA: 44.895 μs  (42.41%)
  softmax_warp_forward<float, float, float, 11, false>  Self CUDA: 44.895 μs
cudaLaunchKernel       Self CPU: 364 μs  (2 calls)
cudaDeviceSynchronize  Self CPU:   8 μs  (2 calls)
```

Two GPU kernels dispatched. Actual GPU work: **105.854 μs**.

**Key observations:**

- **Same add-vs-activation pattern as gelu:** add (binary, 3N bytes) is ~1.36× slower than softmax (unary, 2N bytes), for the same memory-bandwidth-bound reason.
- **`softmax_warp_forward<float, float, float, 11, false>`:** the template parameter `11 = log₂(2048)` encodes the row length. PyTorch selects a warp-specialized kernel based on the softmax dimension size — each warp handles one row of 2^11 elements using warp-level shuffle reductions (no shared memory needed), making a single-pass fused max+exp+sum+divide.
- **Softmax dispatches only 1 kernel** despite being a 3-step computation (max reduction, exp, normalize). The warp-forward kernel fuses all steps, unlike `aten::dot` which dispatches 2 kernels. This fusion matters at larger T where a naive 3-pass softmax would read the T×T attention matrix 3× from HBM.
- **GPU time is nearly identical to gelu (103 μs vs 106 μs)** — both are bandwidth-bound with the same add overhead.

---

### MLP block (`aten::mm` + gelu + add + dropout)

```
Self CUDA time total: 948.447us

aten::mm (ampere_sgemm_128x64_tn)     Self CUDA: 826.207 μs  (87.11%)
aten::add (CUDAFunctor_add)            Self CUDA:  60.480 μs  ( 6.38%)
aten::gelu (GeluCUDAKernelImpl)        Self CUDA:  40.352 μs  ( 4.25%)
aten::uniform_ (distribution_grid)     Self CUDA:  21.408 μs  ( 2.26%)
Memset (Device)                        Self CUDA:   0.832 μs  ( 0.09%)
```

**Key observations:**

- **The matmul (`aten::mm`) consumes 87% of MLP GPU time.** Add + gelu + dropout together account for only ~13%. At realistic model sizes the MLP block is overwhelmingly compute-bound by the linear projection.
- **`aten::uniform_` is the dropout random-number kernel.** It uses a grid-stride distribution kernel to fill the dropout mask in-place. At 2.26% it is non-trivial; at larger batch/sequence sizes it could become more significant.
- **`Memset (Device)` (0.09%)** zeroes the dropout mask buffer before `uniform_` fills it.
- **Add and gelu retain their relative sizes from the isolated benchmarks** (add ≈ 1.5× gelu), confirming they are still memory-bandwidth-bound and unaffected by the surrounding matmul context.
- **This directly explains why shortening context length doesn't speed up MODEL_CONFIG_S:** the MLP block is 87% matmul, whose flop count is `2·B·T·d_model·d_ff` — linear in T. But the matmul at small model sizes is already occupying nearly all GPU SM throughput, so the ~4× reduction in T only yields ~4× fewer flops with near-proportional time savings. The attention block, which scales as T², is a small fraction of total time at MODEL_CONFIG_S sizes — so reducing T gives diminishing returns relative to the fixed MLP cost per token.

### GeLU implementation comparison (`profile_different_gelu.py`, dim=2048)

**Setup:** 1D tensor of shape `[2048]`, 1 warmup iteration. Five implementations compared.

| Implementation | CUDA kernels | Self CUDA time | Notes |
|---|---|---|---|
| `pytorch_gelu` (`F.gelu`) | 1 | **41.215 µs** | Built-in fused `GeluCUDAKernelImpl` |
| `manual_gelu` (eager, no compile) | 9 | **452.221 µs** | Separate mul/add/tanh/mul kernels |
| `manual_gelu_torch_compile` | 1 | **42.176 µs** | Triton fused: `triton_poi_fused_add_mul_tanh_0` |
| `manual_gelu_with_pow` (eager) | ~8 | **370.853 µs** | Same as manual_gelu + extra pow kernel |
| `cuda_gelu` (custom CUDA kernel) | 1 | **49.441 µs** | Hand-written `gelu_kernel(float*, float*, int)` |

**Key observations:**

- **All single-kernel implementations (~41–49 µs) perform equivalently.** `pytorch_gelu`, `torch.compile`, and the custom CUDA kernel all take essentially the same wall-clock GPU time. The operation is memory-bandwidth-bound at this size — one pass over the tensor at ~900 GB/s for 2×2048×4 bytes ≈ 16 KB takes ~18 ns in principle, but kernel launch overhead and profiler effects bring observed time to ~40–50 µs.

- **Eager Python `manual_gelu` is 11× slower (452 µs)** despite identical arithmetic. It dispatches 9 separate kernels (6× mul, 2× add, 1× tanh) — each kernel independently reads and writes the full tensor from HBM. Each round-trip is ~41–62 µs at this size, and 9 kernels multiply that. The compute itself is trivial; the bottleneck is memory traffic from kernel fragmentation.

- **`torch.compile` recovers all the lost performance** by fusing the 9-op eager graph into a single Triton kernel (`triton_poi_fused_add_mul_tanh_0`). The result is indistinguishable from a hand-written CUDA kernel or PyTorch's built-in. This demonstrates the primary value of `torch.compile` for pointwise op chains: not smarter math, but elimination of redundant HBM round-trips.

- **`manual_gelu_with_pow` is slightly faster than `manual_gelu` (370 µs vs 452 µs)** despite having an extra `aten::pow` call. The pow kernel replaces the 3× `aten::mul` chain for `x³`, which PyTorch dispatches as three separate binary mul kernels (9 total for manual_gelu vs. ~8 for with_pow). The trade-off: `aten::pow` is a single kernel for `x³`, but dispatches its own overhead; net effect is slightly fewer kernel launches.

- **Custom CUDA kernel is 20% slower than pytorch_gelu (49 µs vs 41 µs)** — likely attributable to: (1) `tanhf` in CUDA is slightly more expensive than PyTorch's `erf`-based GELU approximation used by `GeluCUDAKernelImpl`; (2) no `--use_fast_math` flag; (3) minor block-size tuning differences. Still well within the memory-bandwidth-bound regime — the gap is not algorithmic.

- **The lesson:** for pointwise operations over small-to-medium tensors, implementation efficiency is dominated by the number of kernel dispatches (memory-bandwidth-bound), not arithmetic complexity. Writing a custom CUDA kernel achieves the same result as `torch.compile` but requires more effort. For any chain of pointwise ops, `torch.compile` is the zero-effort path to optimal performance.

#### CUDA vs Triton custom GeLU profiling

| Variant | Self CPU time | Self CUDA time | Kernel | Kernel time | Launch/sync overhead |
|---|---|---|---|---|---|
| `cuda_gelu` | 634.848 µs | 50.654 µs | `gelu_kernel(float*, float*, int)` | 50.654 µs | `cudaLaunchKernel` 315.359 µs + `cudaDeviceSynchronize` 115.043 µs |
| `triton_gelu` | 586.368 µs | 40.319 µs | `triton_gelu_kernel` | 40.319 µs | `cuLaunchKernelEx` 295.349 µs + `cudaDeviceSynchronize` 19.791 µs |

**Additional observations:**
- Triton's kernel is faster than the handwritten CUDA kernel on this benchmark, with `triton_gelu_kernel` running in ~40.3 µs vs `gelu_kernel` at ~50.7 µs.
- Both implementations show significant dispatch/synchronization overhead; the kernel-only time is a small fraction of the overall CPU-side profile.
- `cuda_gelu` has higher synchronize overhead than `triton_gelu`, suggesting the Triton path can be more efficient for the full-profile call.
- This reinforces the teardown in the table above: the fastest GeLU implementations are the ones that minimize kernel dispatches and fuse work into a single GPU kernel.

---

### GELU activation (`aten::gelu` + `aten::add`)

```
Self CUDA time total: 103.072us

aten::add              Self CUDA: 60.256 μs  (58.46%)
  vectorized_elementwise_kernel (CUDAFunctor_add)  Self CUDA: 60.256 μs
aten::gelu             Self CUDA: 42.816 μs  (41.54%)
  vectorized_elementwise_kernel (GeluCUDAKernelImpl)  Self CUDA: 42.816 μs
cudaLaunchKernel       Self CPU: 351 μs  (2 calls)
cudaDeviceSynchronize  Self CPU:  39 μs  (2 calls)
```

Two GPU kernels dispatched. Actual GPU work: **103.072 μs** (60.256 + 42.816).

**Key observations:**

- **`aten::add` (60.3 μs, 58%) is more expensive than `aten::gelu` (42.8 μs, 42%)** despite GELU being a more complex transcendental computation. Both are elementwise kernels over the same tensor size, so they are both memory-bandwidth-bound — the runtime difference reflects GELU's more complex per-element work (erf/tanh approximation) being offset by its slightly smaller share.
- **Both ops dispatch a single `vectorized_elementwise_kernel`** — PyTorch uses the same vectorized kernel template for both simple adds and complex activation functions, with the math functor as a template parameter.
- **`cudaLaunchKernel` CPU overhead: ~175 μs per kernel** (351 μs / 2 calls) — substantially higher than the ~20–27 μs seen for add/dot in isolation. This is likely a combination of profiler overhead and the fact that GELU requires more CUDA driver setup for the transcendental math path.
- **Ratio: add is ~1.41× slower than gelu on GPU** — because both are memory-bandwidth-bound, and `add` (binary op) moves **50% more data** than `gelu` (unary op): add reads 2 tensors + writes 1 = 3N bytes vs. gelu reads 1 + writes 1 = 2N bytes. The ~1.41× runtime gap tracks the ~1.5× memory traffic ratio. GELU's extra transcendental arithmetic (erf/tanh approximation) is irrelevant — the GPU hides it entirely while waiting for HBM fetches. For memory-bound elementwise kernels, **op arity (how many tensors are read) matters; arithmetic complexity does not**.

---

## Attention Profiling

### Plain PyTorch Attention — Latency Profiling

**Setup:** Single-head causal self-attention with batch_size=8, varying d_model ∈ [16, 32, 64, 128] and seq_length ∈ [256, 1024, 4096, 8192, 16384]. Profiled with `torch.profiler` (1 warmup, single measurement run). Total GPU time measured via `Self CUDA time total` from profiler output.

#### GPU Latency Summary (Self CUDA time, μs)

| d_model | seq=256 | seq=1024 | seq=4096 | seq=8192 | seq=16384 |
|---|---|---|---|---|---|
| 16 | 124.6 | 735.2 | 11,548 | 42,166 | OOM |
| 32 | 119.0 | 720.1 | 11,581 | 39,848 | OOM |
| 64 | 128.5 | 774.8 | 11,958 | 41,037 | OOM |
| 128 | 163.6 | 939.6 | 13,084 | 44,767 | OOM |

**Key observations:**

1. **Quadratic scaling with sequence length.** GPU time grows ~100× from seq=256 → seq=4096 (4× sequence), consistent with O(T²) attention score matrix. At seq=8192 (8× sequence), GPU time is ~360× baseline, matching T² scaling.
   - d_model=16: 124.6 μs → 42,166 μs = 338× increase for 32× sequence (expected: 1024×, but dominated by fixed overhead at small seq)
   - Trend is clear: seq=256 → 1024 is ~6× GPU time (16² = 256, 1024² = 1,048,576; ratio ~4,096×, but overhead masks this)

2. **Minimal d_model effect.** Changing d_model from 16 → 128 (8×) increases GPU time by only ~30% at the same sequence length, not 8×. The attention computation is dominated by the T² score matrix, not the embedding dimension.
   - seq=4096: d_model=16 (11.5 ms) vs d_model=128 (13.1 ms) — only 13% difference
   - This aligns with scaled_dot_product_attention: Q·Kᵀ is O(B·H·T²·d_k) where d_k is small per-head, while T² dominates

3. **Top GPU kernels remain consistent.**
   - `aten::bmm` (matmul for Q·Kᵀ, scores·V): ~20–30% of GPU time
   - `aten::div` (softmax normalization): ~23–26% at larger seq
   - `aten::sub`, `aten::exp`, `aten::where` (softmax: max shift, exp, mask apply): collectively ~40–50% of GPU time
   - Softmax is the dominant bottleneck, not the matmul — this is the classic "memory-bound softmax" issue: one pass over the attention matrix for exp, another for sum reduction, another for normalization, vs. a single fused matmul.

4. **Attention memory is the OOM bottleneck.** All configurations OOM at seq=16384:
   - Attention score matrix shape: `[B, num_heads, T, T] = [8, 1, 16384, 16384]` = 2 GB at fp32 (4 bytes × 268M elements)
   - With a single head and small d_model, the model itself is tiny (~KB), so the attention matrix dominates memory
   - Gradient checkpointing or FlashAttention (online softmax) would be necessary to fit this on current 40 GB GPUs

#### Kernel Breakdown: d_model=16, seq=4096 (11.5 ms total GPU time)

| Kernel | Time | % | Purpose |
|---|---|---|---|
| `aten::div` | 2.679 ms | 23.2% | Softmax: divide by sum |
| `aten::bmm` | 2.593 ms | 22.4% | Q·Kᵀ (query-key scores) |
| `ampere_sgemm_128x128_nn` | 1.908 ms | 16.5% | Scores·V (apply attention weights) |
| `aten::where` | 3.760 ms | 32.6% | Causal mask application + extra softmax |
| Other elementwise | ~1.0 ms | ~8.7% | mul, sub, exp, max, reduce |

**Analysis:** The softmax pipeline (`where` for masking + `div` for normalization) alone consumes ~56% of GPU time. The two matmuls (Q·Kᵀ and scores·V) together are only ~39%. This is the hallmark of a **memory-bandwidth-limited attention implementation** — the T×T softmax matrix is read multiple times (once for max, once for exp, once for sum, once for divide), and masking adds another full read/write.

#### Comparison: Fixed overhead at small sequence lengths

At seq=256 and seq=1024, `aten::bmm` (matmul) dominates the profile (~27–47% of GPU time), while at larger seq, softmax (`aten::div`, `aten::where`) takes over (~56%). This crossover reflects that:
- Matmul is **compute-bound** — scales with T² and O(T²) flops per element
- Softmax is **memory-bandwidth-bound** — scales with T² but only O(1) flops per element after max+shift

At small T, the launch overhead of softmax kernels relative to their 1 μs runtime becomes significant, so matmul appears larger in the profile. At T=4096, the softmax matrix (67 MB) requires multiple full reads/writes, dominating.

#### Latency implications for long sequences

- **Token generation (autoregressive decoding).** For seq=1024 with a 40 GB GPU, a single attention forward pass takes ~720 μs (d_model=32). Decoding 1000 tokens would incur ~720 ms in attention alone, not counting embedding projection or MLP. With a 100-token cache (in practice), attention is ~3 ms per token.
- **Training with long context.** At seq=8192, even a forward pass is ~40 ms per batch. With backward and optimizer overhead, a 160ms iteration becomes 200+ ms, yielding ~5 tokens/sec throughput for a single GPU.

#### OOM boundary observed

- **Effective OOM limit at seq=16384** for d_model ≥ 16 with bs=8 on a 40 GB GPU suggests attention matrix memory alone occupies ~2 GB (67 MB per head × 32 heads ≈ 2 GB). Model parameters are negligible (~KB).
- **Mitigation strategies:**
  - **FlashAttention:** Fuse softmax and matmul, eliminate full T×T matrix materialization in HBM (saves ~2 GB)
  - **Gradient checkpointing:** Save only the input and recompute attention in backward
  - **Sparse attention:** Reduce T² to T·log(T) or T via local/strided patterns
  - **Multi-GPU / distributed attention:** Shard T dimension across GPUs