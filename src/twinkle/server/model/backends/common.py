# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Shared helpers and base classes for backend model implementations.
"""
import numpy as np
import re
import torch
from collections.abc import Mapping
from numbers import Number
from tinker import types
from typing import Any, List

from twinkle import DeviceMesh
from twinkle.template import Template


def collect_forward_backward_results(results, device_mesh: DeviceMesh):
    """Custom collect function for forward_backward that handles list [outputs, loss]."""
    if not results:
        return results

    pp_last_ranks = None
    if device_mesh.pp_world_size > 1:
        pp_last_ranks = set(device_mesh.get_pp_last_ranks())

    tp_last_ranks = None
    if device_mesh.tp_world_size > 1:
        tp_last_ranks = set(device_mesh.get_tp_last_ranks())

    mesh_flat = device_mesh.mesh.flatten()

    all_outputs = []
    all_losses = []
    for i, result in enumerate(results):
        rank = mesh_flat[i] if i < len(mesh_flat) else -1

        if pp_last_ranks is not None:
            if rank not in pp_last_ranks:
                continue

        if tp_last_ranks is not None:
            if rank not in tp_last_ranks:
                continue

        if result is None:
            continue

        outputs, loss = result
        if outputs is None or loss is None:
            continue
        all_outputs.extend(outputs)
        all_losses.append(loss)

    if all_losses:
        avg_loss = float(np.mean(all_losses))
    else:
        avg_loss = 0.0

    return [all_outputs, avg_loss]


def to_cpu_safe_output(obj: Any) -> Any:
    """Convert nested model outputs into CPU-safe Python objects for HTTP transport.

    Recursively walks tensors, numpy arrays, mappings and sequences,
    converting each tensor/array to a plain Python scalar or list so
    Ray can serialise the result without requiring CUDA on the driver.
    """
    from twinkle.utils import torch_util

    if isinstance(obj, torch.Tensor):
        tensor = torch_util.to_local_tensor(obj).detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {key: to_cpu_safe_output(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_cpu_safe_output(value) for value in obj]
    return obj


def clean_metrics(metrics: dict) -> dict:

    def _to_float(v):
        if isinstance(v, (float, int, Number, np.generic, str)):
            try:
                return float(v)
            except Exception:
                return None
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            try:
                return float(v.item())
            except Exception:
                return None
        return None

    cleaned = {}
    for key, value in metrics.items():
        fv = _to_float(value)
        if fv is not None:
            cleaned[key] = fv
            continue

        if isinstance(value, str):
            s = value.strip()
            if s:
                try:
                    head, unit = s.split(maxsplit=1)
                    cleaned[f'{key}/{unit}'] = float(head)
                except Exception:
                    m = re.match(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', s)
                    if m:
                        cleaned[key] = float(m.group(1))

    return cleaned


class TwinkleCompatModelBase:
    """Base class containing common logic for Twinkle compatibility wrappers."""

    def get_template(self, adapter_name: str) -> Template:
        return self.optimizer_group[adapter_name].template

    @staticmethod
    def _get_forward_output(inputs: List[types.Datum], logits: torch.Tensor, logps: torch.Tensor) -> List[dict]:
        """Convert raw logits to the expected output format with logprobs and elementwise_loss."""
        from twinkle.utils.torch_utils import selective_log_softmax
        if logps is not None:
            device = logps.device
        elif logits is not None:
            device = logits.device
        else:
            raise ValueError('At least one of logits or logps must be provided.')
        results = []
        if logits is None:
            logits = [None] * len(inputs)
        for idx, (feature, logit) in enumerate(zip(inputs, logits)):
            labels = feature.loss_fn_inputs['target_tokens'].to_torch().long().view(-1).to(device)
            weights = feature.loss_fn_inputs['weights'].to_torch().view(-1).to(device)

            seq_len = labels.numel()

            if logps is None:
                assert logit is not None, 'logit must not be None when logps is None'
                feature_logits = logit[:seq_len, :]
                token_log_probs = selective_log_softmax(feature_logits, labels)
            else:
                token_log_probs = logps[idx, :seq_len]

            # elementwise_loss: positive NLL loss (0.0 where masked)
            token_log_probs = token_log_probs.to(weights.device)
            elementwise_loss = -token_log_probs * weights

            results.append({
                'logprobs': types.TensorData.from_torch(token_log_probs.cpu()),
                'elementwise_loss': types.TensorData.from_torch(elementwise_loss.cpu())
            })
        return results
