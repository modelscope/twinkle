# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from types import MethodType
from typing import Union, Callable, Any, List, Optional, Literal
from ..utils import torch_util, framework_util
from ..utils import exists


torch_kernel_mapping = {
}


def apply_kernel(module: Any,
                 mode: Literal['train', 'inference', 'compile', None] = 'train',
                 kernel: "Optional[Union[str, Callable, 'torch.nn.Module']]"=None,
                 target_modules: Union[str, List[str]]=None,
                 device: Optional[Union[str, Any]] = None,
                ) -> Any:
    if framework_util.get_framework(module) == 'torch':
        if torch_util.get_library(module) == 'transformers':
            if exists('kernels'):
                from kernels import kernelize, Mode
                kernel_mode = Mode.TRAINING
                if mode == 'inference':
                    kernel_mode = Mode.INFERENCE
                elif mode == 'compile':
                    kernel_mode = Mode.TORCH_COMPILE
                from kernels import kernelize
                return kernelize(module, mode=kernel_mode, device=device)

        assert target_modules is not None and kernel is not None

        if torch_util.get_library(module) == 'megatron':
            ...

        return apply_kernel_torch(module, kernel, target_modules=target_modules)
    else:
        raise NotImplementedError(f'Unsupported applying kernels for: {module.__class__}')


def apply_kernel_torch(module: Any,
                     kernel: "Optional[Union[str, Callable, 'torch.nn.Module']]",
                     target_modules: Union[str, List[str]]):
    if kernel in torch_kernel_mapping:
        kernel = torch_kernel_mapping[kernel]

    kernel_fn = kernel
    import torch
    if isinstance(kernel_fn, torch.nn.Module):
        kernel_fn = kernel_fn.forward

    if target_modules is None:
        raise ValueError(f'Module patching needs a valid `target_modules` parameter,'
                         f'but current is: {target_modules}')

    if isinstance(target_modules, str):
        pattern = re.compile(target_modules)
        for name, submodule in module.named_modules():
            if pattern.search(name):
                if not hasattr(submodule, '__origin_forward__'):
                    submodule.__origin_forward__ = submodule.forward
                    submodule.forward = MethodType(kernel_fn, submodule)

    elif isinstance(target_modules, list):
        for name, submodule in module.named_modules():
            if any(name.endswith(target) for target in target_modules):
                if not hasattr(submodule, '__origin_forward__'):
                    submodule.__origin_forward__ = submodule.forward
                    submodule.forward = MethodType(kernel_fn, submodule)
    return module


