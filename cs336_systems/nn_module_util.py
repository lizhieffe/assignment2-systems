import torch.nn as nn


def get_named_leaf_modules(root: nn.Module) -> list[tuple[str, nn.Module]]:
  ret = []

  for name, child in root.named_children():
    if child == root:
      continue
    if len(list(child.children())) == 0:
      ret.append((name, child))
    else:
      ret.extend(get_named_leaf_modules(child))

  return ret


def get_leaf_modules(root: nn.Module) -> list[nn.Module]:
  named_leaf_modules = get_named_leaf_modules(root)
  return [m for (_, m) in named_leaf_modules]


def get_leaf_module_types(root: nn.Module) -> set[type[nn.Module]]:
  leaf_modules = get_leaf_modules(root)
  leaf_module_types = [type(m) for m in leaf_modules]
  return set(leaf_module_types)
