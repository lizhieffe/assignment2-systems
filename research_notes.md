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