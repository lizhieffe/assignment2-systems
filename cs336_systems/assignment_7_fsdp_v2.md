# FSDP v2 Diagrams

## Class structure

```mermaid
classDiagram
    class FSDP {
        +module: nn.Module
        +sharded_layers: list
        +shard_params: set
        +gradient_sync_works: list
        +__init__(module, compute_dtype)
        +forward(...)
        +finish_gradient_synchronization()
        +gather_full_params()
        -_replicated_grad_sync_hook(param)
    }

    class _FSDPShardedMixin {
        +weight: Parameter (local shard only)
        +full_weight: Tensor|None
        +gather_work: tuple|None
        +fsdp_parent: FSDP (back-ref, not registered as submodule)
        +_init_sharding(...)
        +gather_async()
        +_sync_gather()
        +forward(input_tensor)
    }

    class Linear { +forward(x) }
    class Embedding { +forward(token_ids) }

    class FSDPLinearWrapper
    class FSDPEmbeddingWrapper

    class FSDPFunction {
        <<autograd.Function>>
        +forward(ctx, wrapper, input, weight, bias)
        +backward(ctx, grad_output)
    }

    FSDP "1" *-- "many" _FSDPShardedMixin : replaces Linear/Embedding\nleaves in module tree
    _FSDPShardedMixin <|-- FSDPLinearWrapper : mixin first (MRO)
    Linear <|-- FSDPLinearWrapper
    _FSDPShardedMixin <|-- FSDPEmbeddingWrapper : mixin first (MRO)
    Embedding <|-- FSDPEmbeddingWrapper
    FSDPLinearWrapper ..> FSDPFunction : forward() calls .apply()
    FSDPEmbeddingWrapper ..> FSDPFunction : forward() calls .apply()
```

## Runtime flow (one layer's forward → backward)

```mermaid
sequenceDiagram
    participant Model as model.forward()
    participant W as FSDPLinearWrapper (layer i)
    participant Next as layer i+2 wrapper
    participant Fn as FSDPFunction
    participant NCCL as collective (all_gather/reduce_scatter)
    participant Acc as AccumulateGrad (autograd)
    participant FSDP as FSDP.finish_gradient_synchronization()

    Note over Model,W: --- forward pass ---
    Model->>W: layer(x)
    W->>Fn: apply(wrapper, x, weight, bias)
    Fn->>Fn: wait on gather_work (prefetched 2 layers ago)
    Fn->>NCCL: (fallback) sync all_gather if not prefetched
    Fn->>Fn: out = F.linear(x, full_weight, full_bias)
    Fn->>Fn: free full_weight/full_bias immediately
    Fn-->>W: out
    W->>Next: gather_async() -- kick off prefetch for i+2
    W-->>Model: out

    Note over Model,Acc: --- backward pass (reverse order) ---
    Model->>Fn: backward(grad_output)  [via autograd engine]
    Fn->>NCCL: sync all_gather (re-materialize full_weight)
    Fn->>Fn: compute grad_input, grad_weight_full
    Fn->>NCCL: reduce_scatter_tensor(grad_weight, grad_weight_full, async_op=True)
    Fn->>FSDP: gradient_sync_works.append((work, grad_full, grad_local, weight))
    Fn->>Fn: free full_weight again
    Fn-->>Acc: return grad_input, grad_weight, grad_bias
    Acc->>Acc: weight.grad = clone(grad_weight)  [refcount>1 => cloned!]

    Note over FSDP: --- after loss.backward() ---
    FSDP->>NCCL: work.wait() for every pending reduce_scatter
    FSDP->>Acc: param.grad.copy_(grad_local)  [explicit fix for the clone]
```

## Key things the diagrams make visible

- **MRO matters**: `_FSDPShardedMixin` must precede `Linear`/`Embedding` in the base list so its `forward` wins over the original einsum/indexing -- that's why the class diagram shows the mixin first.
- **`fsdp_parent` is a deliberately unregistered back-reference** (via `object.__setattr__`) -- shown as a plain association, not composition, to avoid the submodule-leak bug that was fixed.
- **The async reduce-scatter + later `.wait()` + explicit `.copy_()`** is the one non-obvious sequence: autograd clones the returned gradient (refcount > 1), so the real fix isn't waiting -- it's writing the result into `param.grad` *after* waiting, since the clone and the original buffer are different tensors after that point.
