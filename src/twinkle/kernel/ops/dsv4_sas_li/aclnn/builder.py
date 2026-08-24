# Copyright (c) ModelScope Contributors. All rights reserved.
"""JIT-compile ACLNN C++ binding extensions on first use.

Adapted from ``torchtitan_npu.ops.aclnn.builder``.  Each ``build_op`` call
loads (and caches) a pybind11 extension whose ``binding.cpp`` invokes the
ACLNN kernel macro ``ACLNN_CMD``.
"""
import os
import torch

_loaded_ops = {}


def build_op(name, sources, verbose=False):
    if name in _loaded_ops:
        return _loaded_ops[name]

    cwd = os.path.dirname(os.path.abspath(__file__))
    ascend_home = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/cann')

    try:
        import torch_npu
        tnpu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    except ImportError:
        raise ImportError('torch_npu is required to compile ACLNN extensions. '
                          'Install it for your Ascend NPU environment.')

    torch_path = os.path.dirname(os.path.abspath(torch.__file__))

    cflags = [
        '-D_FORTIFY_SOURCE=2',
        '-O2',
        f'-I{cwd}',
        f'-I{ascend_home}/include',
        f'-I{torch_path}/include',
        f'-I{torch_path}/include/torch/csrc/api/include',
        f'-I{tnpu_path}/include',
        f'-I{tnpu_path}/include/torch_npu/csrc/framework/utils',
        f'-I{tnpu_path}/include/torch_npu/csrc/aten',
        f'-I{tnpu_path}/include/third_party/hccl/inc',
        f'-I{tnpu_path}/include/third_party/acl/inc',
    ]

    ldflags = [
        f'-L{ascend_home}/lib64',
        f'-L{tnpu_path}/lib',
        '-ltorch_npu',
        '-lascendcl',
    ]

    from torch.utils.cpp_extension import load

    op_module = load(
        name=name,
        sources=[os.path.join(cwd, s) for s in sources],
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        verbose=verbose,
    )
    _loaded_ops[name] = op_module
    return op_module
