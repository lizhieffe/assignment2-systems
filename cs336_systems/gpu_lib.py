import torch


def print_gpu_specs():
  num_devices = torch.cuda.device_count()
  print("=" * 80)
  for i in range(num_devices):
    properties = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {properties}")
  print("=" * 80)
  print()
