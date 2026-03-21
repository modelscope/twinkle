# Copyright (c) ModelScope Contributors. All rights reserved.
"""MindSpeed bootstrap helpers for NPU Megatron runs."""

import importlib
import inspect
from argparse import Namespace
from typing import Any, Dict, Optional

from twinkle import Platform

from ._mindspeed_args import build_mindspeed_namespace, get_mindspeed_signature, sanitize_mindspeed_values

_DEFAULT_MINDSPEED_VALUES: Optional[Dict[str, Any]] = None
_RUNTIME_MINDSPEED_ARGS: Optional[Namespace] = None
_LAST_REPATCH_SIGNATURE: Optional[str] = None


def _get_mindspeed_defaults(args_utils) -> Dict[str, Any]:
    global _DEFAULT_MINDSPEED_VALUES

    if _DEFAULT_MINDSPEED_VALUES is None:
        defaults = args_utils.get_mindspeed_args(get_defaults=True)
        _DEFAULT_MINDSPEED_VALUES = sanitize_mindspeed_values(vars(defaults).copy())
    return _DEFAULT_MINDSPEED_VALUES


def _install_full_args_provider(args_utils) -> None:
    if getattr(args_utils, '_TWINKLE_RUNTIME_PROVIDER_INSTALLED', False):
        return

    def get_full_args():
        if _RUNTIME_MINDSPEED_ARGS is None:
            raise RuntimeError('MindSpeed runtime args are not initialized before bootstrap.')
        return _RUNTIME_MINDSPEED_ARGS

    args_utils.get_full_args = get_full_args
    args_utils._TWINKLE_RUNTIME_PROVIDER_INSTALLED = True


def _set_runtime_args(args_utils, runtime_args: Namespace) -> None:
    global _RUNTIME_MINDSPEED_ARGS

    _RUNTIME_MINDSPEED_ARGS = runtime_args
    args_utils._MINDSPEED_ARGS = runtime_args


def _import_mindspeed_adaptor(args_utils):
    patch_utils = importlib.import_module('mindspeed.patch_utils')
    if not hasattr(patch_utils, 'inspect'):
        patch_utils.inspect = inspect
    return importlib.import_module('mindspeed.megatron_adaptor')


def bootstrap_mindspeed_for_npu(args: Any) -> Optional[Dict[str, Any]]:
    global _LAST_REPATCH_SIGNATURE

    if Platform.device_prefix() != 'npu':
        return None

    try:
        args_utils = importlib.import_module('mindspeed.args_utils')
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'MindSpeed is required for Twinkle NPU Megatron runs. '
            'Please install MindSpeed in the current environment.'
        ) from exc
    # Fetch MindSpeed defaults here, then merge them with Twinkle args to
    # build the final MindSpeed runtime args.
    runtime_args = build_mindspeed_namespace(args, _get_mindspeed_defaults(args_utils))
    # Replace get_full_args in mindspeed.args_utils so it returns the runtime
    # args constructed by Twinkle.
    _install_full_args_provider(args_utils)
    # Store the constructed runtime args in mindspeed.args_utils so later
    # MindSpeed modules can consume them.
    _set_runtime_args(args_utils, runtime_args)

    signature = get_mindspeed_signature(runtime_args)
    adaptor = _import_mindspeed_adaptor(args_utils)
    if signature != _LAST_REPATCH_SIGNATURE:
        if _LAST_REPATCH_SIGNATURE is not None:
            adaptor.repatch(vars(runtime_args).copy())
        _LAST_REPATCH_SIGNATURE = signature

    return vars(runtime_args).copy()
