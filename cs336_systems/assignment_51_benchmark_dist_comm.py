# Assignment 5.1 - benchmark distributed communication.
#
# Run the code:
# uv run cs336_systems/assignment_51_benchmark_dist_comm.py


import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

USE_GPU = True
WORLD_SIZE = 2
MATRIX_SIZE = 1000_000_000


def setup(
  rank: int,
  world_size: int,
  backend: str,
  device_id: torch.device | None = None,
):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29500"
  dist.init_process_group(
    backend=backend, world_size=world_size, rank=rank, device_id=device_id
  )


def warmup(device: str, n: int):
  for _ in range(n):
    x = torch.randint(0, 10, (3,)).to(device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)


def distributed_demo(rank: int, world_size: int):
  if USE_GPU:
    assert torch.cuda.is_available(), "GPU is not available"
    device_count = torch.cuda.device_count()
    assert device_count >= world_size, (
      f"Less GPU {device_count} than world size {world_size}"
    )
    device = f"cuda:{rank}"
  else:
    device = "cpu"

  setup(
    rank=rank,
    world_size=world_size,
    backend="nccl" if USE_GPU else "gloo",
    device_id=torch.device(device) if USE_GPU else None,
  )

  try:
    warmup(device=device, n=5)

    x = torch.randint(0, 10, (MATRIX_SIZE,)).to(device=device)
    print(f"Before all_reduce: {rank=}, x={x}")

    if USE_GPU:
      torch.cuda.synchronize()
    dist.barrier()

    start_time = time.time()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)

    # synchronize is needed even When async_op=False, which returns once the op
    # is scheduled on GPU instead of waiting for comm to finish.
    if USE_GPU:
      torch.cuda.synchronize()

    print(
      f"After all_reduce: {rank=}, x={x}, latency={time.time() - start_time:.5f}"
    )
  finally:
    dist.destroy_process_group()


def main():
  print(f"{USE_GPU=} {WORLD_SIZE=} {MATRIX_SIZE=}")
  world_size = WORLD_SIZE
  mp.spawn(
    fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True
  )


if __name__ == "__main__":
  main()
