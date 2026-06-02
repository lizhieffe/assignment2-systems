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