# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lazy re-exports: importing ``twinkle.utils`` must stay dependency-light.

The server-config import path (``twinkle_server.config`` -> ``hub.model_alias`` ->
``get_logger``) is guarded by tests/server/test_dependencies.py to never load torch/ray;
eagerly re-exporting the torch-dependent helpers (kl/torch_utils/safetensors/...) here
dragged torch into that path. Same _LazyModule pattern as the twinkle root package.
"""
import sys
from typing import TYPE_CHECKING

from twinkle._lazy_module import _LazyModule
# Import-time side effect that must stay eager: the detected platform's one-shot runtime
# fixups (see platforms/__init__). The vLLM EngineCore subprocess only *imports*
# twinkle.utils through the worker extension and never accesses an attribute, yet it needs
# XPU's torch.xpu stub installed before vLLM probes for an Intel XPU. platforms itself is
# torch-free at module level (torch imports live inside methods), so this stays light.
from . import platforms  # noqa: F401

# Renamed re-exports (``Framework as framework_util``) cannot go through _import_structure:
# _LazyModule resolves a name with getattr(source_module, name), and the source module only
# defines the original class name. framework/device_mesh are torch-free at module level, so
# binding the aliases eagerly keeps the package light.
from .framework import Framework as framework_util
from .framework import Torch as torch_util

if TYPE_CHECKING:
    from .dequantizer import Fp8Dequantizer, MxFp4Dequantizer
    from .device_mesh import DeviceGroup, DeviceMesh, is_last_rank, is_master
    from .exceptions import FullWeightLoadError
    from .framework import Framework as framework_util
    from .framework import Torch as torch_util
    from .import_utils import exists, requires
    from .kl import KLEstimator
    from .loader import Plugin, construct_class, load_module
    from .logger import get_logger
    from .network import find_free_port, find_node_ip, is_valid_ipv6_address
    from .parallel import processing_lock
    from .platforms import (
        GPU,
        NPU,
        XPU,
        Metax,
        Platform,
        ensure_hccl_socket_env,
        ensure_npu_backend,
        is_metax_available,
    )
    from .safetensors import LazyTensor, SafetensorLazyLoader, StreamingSafetensorSaver
    from .torch_utils import (
        clone_state_dict_to_cpu,
        pad_and_stack_tensors,
        pad_sequence_to_length,
        prepare_replayed_selective_log_softmax,
        replayed_selective_log_softmax,
        selective_log_softmax,
        split_cp_inputs,
        stateless_init_process_group,
        to_device,
    )
    from .transformers_utils import find_all_linears, find_layers, get_modules_to_not_convert
    from .unsafe import check_unsafe, trust_remote_code
    from .utils import copy_files_by_pattern, deep_getattr, get_runtime_meta, run_sync, seed_from_env
    from .vision_tools import load_image, load_mm_file
else:
    _import_structure = {
        'dequantizer': ['Fp8Dequantizer', 'MxFp4Dequantizer'],
        'device_mesh': ['DeviceGroup', 'DeviceMesh', 'is_last_rank', 'is_master'],
        'exceptions': ['FullWeightLoadError'],
        'framework': [],
        'import_utils': ['exists', 'requires'],
        'kl': ['KLEstimator'],
        'loader': ['Plugin', 'construct_class', 'load_module'],
        'logger': ['get_logger'],
        'network': ['find_free_port', 'find_node_ip', 'is_valid_ipv6_address'],
        'parallel': ['processing_lock'],
        'platforms': [
            'GPU', 'is_metax_available', 'Metax', 'NPU', 'Platform', 'ensure_hccl_socket_env', 'ensure_npu_backend',
            'XPU'
        ],
        'safetensors': ['LazyTensor', 'SafetensorLazyLoader', 'StreamingSafetensorSaver'],
        'torch_utils': [
            'clone_state_dict_to_cpu', 'pad_and_stack_tensors', 'pad_sequence_to_length',
            'prepare_replayed_selective_log_softmax', 'replayed_selective_log_softmax', 'selective_log_softmax',
            'split_cp_inputs', 'stateless_init_process_group', 'to_device'
        ],
        'transformers_utils': ['find_all_linears', 'find_layers', 'get_modules_to_not_convert'],
        'unsafe': ['check_unsafe', 'trust_remote_code'],
        'utils': ['copy_files_by_pattern', 'deep_getattr', 'get_runtime_meta', 'run_sync', 'seed_from_env'],
        'vision_tools': ['load_image', 'load_mm_file'],
    }

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,  # noqa
        extra_objects={
            'framework_util': framework_util,
            'torch_util': torch_util,
        },
    )
