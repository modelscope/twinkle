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

    def _tinker_setup_loss(self, loss_fn: str, inputs, adapter_name: str, kwargs: dict):
        """Set up loss function based on loss_fn; pops DPO/GRPO-specific params from kwargs in-place."""
        if loss_fn == 'cross_entropy':
            self.set_loss('CrossEntropyLoss', adapter_name=adapter_name)
        elif loss_fn == 'importance_sampling':
            has_ref_logps = any('ref_logps' in d.loss_fn_inputs for d in inputs)
            if has_ref_logps:
                beta = kwargs.pop('dpo_beta', 0.1)
                loss_type = kwargs.pop('dpo_loss_type', 'sigmoid')
                sft_weight = kwargs.pop('dpo_sft_weight', 0.0)
                self.set_loss(
                    'DPOLoss', adapter_name=adapter_name, beta=beta, loss_type=loss_type, sft_weight=sft_weight)
                # Only add DPOMetric if not already present for this adapter
                self._ensure_dpo_metric(adapter_name, beta)
            else:
                epsilon = kwargs.pop('epsilon', 0.2)
                grpo_beta = kwargs.pop('beta', 0.0)
                self.set_loss('GRPOLoss', adapter_name=adapter_name, epsilon=epsilon, beta=grpo_beta)
        else:
            self.set_loss('CrossEntropyLoss', adapter_name=adapter_name)

    def _ensure_dpo_metric(self, adapter_name: str, beta: float):
        """Add DPOMetric for the adapter if not already present.

        This prevents duplicate metric accumulation across training steps.
        """
        from twinkle.metric.dpo import DPOMetric
        optimizer_config = self.optimizer_group[adapter_name]
        # Check if DPOMetric already exists in training metrics
        for metric in optimizer_config.train_status.metrics:
            if isinstance(metric, DPOMetric):
                return
        self.add_metric('DPOMetric', adapter_name=adapter_name, beta=beta)

    def _apply_ref_outputs(self, loss_values: dict, loss_kwargs: dict, adapter_name: str) -> None:
        """Pop ref_outputs from loss_values into loss_kwargs and propagate to train_status.

        DPOMetric reads ref_outputs from train_status.forward_kwargs during accumulate_metrics,
        so it must be set here before the subsequent loss calculation.
        """
        if 'ref_outputs' not in loss_values:
            return
        ref_outputs_dict = loss_values.pop('ref_outputs')
        loss_kwargs['ref_outputs'] = ref_outputs_dict
        self.optimizer_group[adapter_name].train_status.forward_kwargs['ref_outputs'] = ref_outputs_dict

    def _tinker_build_output(self, inputs, outputs, return_full_logprobs: bool = False):
        """Extract logits/logps from model outputs and build per-datum output list."""
        seq_lens = [feature.loss_fn_inputs['target_tokens'].to_torch().long().view(-1).numel() for feature in inputs]
        logits = self._tensor_output_to_rows(outputs.get('logits'), seq_lens, kind='logits')
        logps = self._tensor_output_to_rows(outputs.get('logps'), seq_lens, kind='logps')
        if logits is None and logps is None:
            # non-last PP stage: no outputs produced, collector will discard this
            return []
        return self._get_forward_output(inputs, logits, logps, return_full_logprobs=return_full_logprobs)

    @staticmethod
    def _tensor_output_to_rows(value, seq_lens: list[int], *, kind: str) -> list[torch.Tensor] | None:
        """Normalize backend tensors to one row per Tinker datum.

        Accepted backend shapes:
        - per-datum lists: ``[[T], [T]]`` for logps, ``[[T, V], [T, V]]`` for logits.
        - batch-major tensors: ``[B, T]`` for logps, ``[B, T, V]`` for logits.
        - packed tensors: ``[sum(T)]`` for logps, ``[sum(T), V]`` for logits.
        - Megatron wrappers: an extra leading singleton dimension around any of the above.
        """
        if value is None:
            return None

        tensors = value
        if isinstance(value, torch.Tensor):
            tensors = [value]
        elif isinstance(value, list) and not value:
            # Non-last PP stages can legitimately produce no logits/logps.
            return None
        elif not (isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value)):
            tensors = [torch.as_tensor(value, dtype=torch.float32)]

        tensors = [tensor.detach().cpu() for tensor in tensors]
        if len(tensors) == len(seq_lens) and all(tensor.dim() <= 1 or tensor.shape[0] == 1 for tensor in tensors):
            return [TwinkleCompatModelBase._normalize_single_row_tensor(tensor, kind=kind) for tensor in tensors]

        rows = []
        for tensor in tensors:
            chunk_rows = TwinkleCompatModelBase._tensor_chunk_to_rows(tensor, seq_lens[len(rows):], kind=kind)
            rows.extend(chunk_rows)
        return rows

    @staticmethod
    def _normalize_single_row_tensor(tensor: torch.Tensor, *, kind: str) -> torch.Tensor:
        if kind == 'logits':
            if tensor.dim() >= 3 and tensor.shape[0] == 1:
                return tensor.squeeze(0)
            return tensor

        while tensor.dim() > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    @staticmethod
    def _split_packed_tensor(tensor: torch.Tensor, seq_lens: list[int]) -> list[torch.Tensor] | None:
        if tensor.dim() == 0 or not seq_lens:
            return None
        split_rows = []
        offset = 0
        for seq_len in seq_lens:
            if offset + seq_len > tensor.shape[0]:
                break
            split_rows.append(tensor[offset:offset + seq_len])
            offset += seq_len
            if offset == tensor.shape[0]:
                break
        return split_rows or None

    @staticmethod
    def _tensor_chunk_to_rows(tensor: torch.Tensor, seq_lens: list[int], *, kind: str) -> list[torch.Tensor]:
        if tensor.dim() == 0 or len(seq_lens) <= 1:
            return [TwinkleCompatModelBase._normalize_single_row_tensor(tensor, kind=kind)]

        if kind == 'logps':
            while tensor.dim() > 1 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() >= 2:
                batch_size = tensor.shape[0]
                if batch_size <= len(seq_lens) and tensor.shape[1] >= max(seq_lens[:batch_size]):
                    return [
                        TwinkleCompatModelBase._normalize_single_row_tensor(row, kind=kind) for row in tensor.unbind(0)
                    ]
            packed_rows = TwinkleCompatModelBase._split_packed_tensor(tensor, seq_lens)
            if packed_rows is not None:
                return packed_rows
            return [tensor]

        if kind == 'logits':
            if tensor.dim() >= 3 and tensor.shape[0] == 1:
                packed_rows = TwinkleCompatModelBase._split_packed_tensor(tensor.squeeze(0), seq_lens)
                if packed_rows is not None:
                    return packed_rows
                tensor = tensor.squeeze(0)
            if tensor.dim() >= 3:
                batch_size = tensor.shape[0]
                if batch_size <= len(seq_lens) and tensor.shape[1] >= max(seq_lens[:batch_size]):
                    return [
                        TwinkleCompatModelBase._normalize_single_row_tensor(row, kind=kind) for row in tensor.unbind(0)
                    ]
            packed_rows = TwinkleCompatModelBase._split_packed_tensor(tensor, seq_lens)
            if packed_rows is not None:
                return packed_rows
            return [tensor]

        raise ValueError(f'Unsupported tensor output kind: {kind}')

    @staticmethod
    def _get_forward_output(inputs: list[types.Datum],
                            logits: list[torch.Tensor] | None,
                            logps: list[torch.Tensor] | None,
                            return_full_logprobs: bool = False) -> list[dict]:
        """Convert raw logits to the expected output format with logprobs and elementwise_loss.

        When return_full_logprobs is True (forward_only / reference pass), logprobs is returned
        at the full TP/CP-padded sequence length so that when the client sends it back as
        ref_logps in the DPO forward_backward step the shape already matches the padded labels.
        When return_full_logprobs is False (default, forward_backward pass), logprobs is
        truncated to the original unpadded sequence length.
        elementwise_loss is always computed on the original (unpadded) length because the
        per-datum weights tensor has that length.
        """
        if logps is not None:
            if len(logps) != len(inputs):
                raise ValueError(f'Expected {len(inputs)} logps rows, got {len(logps)}')
            device = logps[0].device
        elif logits is not None:
            if len(logits) != len(inputs):
                raise ValueError(f'Expected {len(inputs)} logits rows, got {len(logits)}')
            device = logits[0].device
        else:
            raise ValueError('At least one of logits or logps must be provided.')
        results = []
        if logits is None:
            logits = [None] * len(inputs)
        if logps is None:
            logps = [None] * len(inputs)
        for feature, logit, feature_logps in zip(inputs, logits, logps):
            labels = feature.loss_fn_inputs['target_tokens'].to_torch().long().view(-1).to(device)
            weights = feature.loss_fn_inputs['weights'].to_torch().view(-1).to(device)

            seq_len = labels.numel()  # original unpadded length

            if feature_logps is None:
                assert logit is not None, 'logit must not be None when logps is None'
                from twinkle.utils.torch_utils import selective_log_softmax
                feature_logits = logit[:seq_len, :]
                token_log_probs_orig = selective_log_softmax(feature_logits, labels)
                if return_full_logprobs:
                    # Extend to the full logit length (TP/CP-padded) by padding with 0.
                    # Padded positions have label -100 so they are masked out by DPOLoss.
                    padded_len = logit.shape[0]
                    if padded_len > seq_len:
                        import torch.nn.functional as F
                        token_log_probs_full = F.pad(token_log_probs_orig, (0, padded_len - seq_len), value=0.0)
                    else:
                        token_log_probs_full = token_log_probs_orig
                else:
                    token_log_probs_full = token_log_probs_orig
            else:
                feature_logps = feature_logps.to(device)
                token_log_probs_orig = feature_logps[:seq_len]
                # When return_full_logprobs is True, retain the full TP/CP-padded slice.
                # Positions beyond seq_len have label -100 and are masked by _compute_sequence_logps.
                token_log_probs_full = feature_logps if return_full_logprobs else token_log_probs_orig

            # elementwise_loss: positive NLL loss (0.0 where masked)
            token_log_probs_orig = token_log_probs_orig.to(weights.device)
            elementwise_loss = -token_log_probs_orig * weights

            results.append({
                'logprobs': types.TensorData.from_torch(token_log_probs_full.cpu()),
                'elementwise_loss': types.TensorData.from_torch(elementwise_loss.cpu())
            })
        return results
