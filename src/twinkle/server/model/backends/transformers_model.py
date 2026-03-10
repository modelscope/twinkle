# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Backend model implementations for the unified model deployment.

Contains two classes:
- TwinkleCompatTransformersModel: tinker-compat wrapper (Datum-based I/O),
  moved from tinker/common/transformers_model.py.
- TwinkleCompatTransformersModelNative: twinkle-native wrapper
  (InputFeature/Trajectory-based I/O), moved from twinkle/common/transformers_model.py.
"""
import numpy as np
import torch
from collections.abc import Mapping
from tinker import types
from typing import Any, List, Union

# ---------------------------------------------------------------------------
# Shared helpers (moved from tinker/common/compat_base.py)
# ---------------------------------------------------------------------------
from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.model import MultiLoraTransformersModel
from twinkle.server.common.datum import datum_to_input_feature, extract_rl_feature
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


def clean_metrics(metrics: dict) -> dict:
    import re
    from numbers import Number

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
                    head, unit = s.split()
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
        device = logits.device if logits is not None else logps.device
        results = []
        if logits is None:
            logits = [None] * len(inputs)
        for idx, (feature, logit) in enumerate(zip(inputs, logits)):
            labels = feature.loss_fn_inputs['target_tokens'].to_torch().long().view(-1).to(device)
            weights = feature.loss_fn_inputs['weights'].to_torch().view(-1).to(device)

            seq_len = labels.numel()

            if logps is None:
                assert logits is not None
                feature_logits = logit[:seq_len, :]
                token_log_probs = selective_log_softmax(feature_logits, labels)
            else:
                token_log_probs = logps[idx, :seq_len]

            elementwise_loss = -token_log_probs * weights

            results.append({
                'logprobs': types.TensorData.from_torch(token_log_probs.cpu()),
                'elementwise_loss': types.TensorData.from_torch(elementwise_loss.cpu())
            })
        return results


# ---------------------------------------------------------------------------
# Tinker-compat Transformers model (Datum-based I/O)
# ---------------------------------------------------------------------------


@remote_class()
class TwinkleCompatTransformersModel(MultiLoraTransformersModel, TwinkleCompatModelBase):
    """Tinker-compatible wrapper around MultiLoraTransformersModel.

    Input/output is in tinker Datum / TensorData format.
    Moved from tinker/common/transformers_model.py.
    """

    @remote_function(dispatch='slice_dp', collect='flatten')
    def forward_only(self, *, inputs: List[types.Datum], **kwargs):
        template = self.get_template(**kwargs)
        input_features = datum_to_input_feature(inputs, template)
        outputs = super().forward_only(inputs=input_features, **kwargs)
        logits = outputs['logits'].detach().cpu()
        logps = outputs.get('logps', None)
        if logps is not None:
            logps = logps.detach().cpu()
        results = self._get_forward_output(inputs, logits, logps)
        return results

    @remote_function(dispatch='slice_dp', collect=collect_forward_backward_results)
    def forward_backward(self, *, inputs: List[types.Datum], adapter_name: str, loss_fn: str, **kwargs):
        if loss_fn == 'cross_entropy':
            super().set_loss('CrossEntropyLoss', adapter_name=adapter_name)
        elif loss_fn == 'importance_sampling':
            super().set_loss('GRPOLoss', adapter_name=adapter_name, epsilon=0.2, beta=0.0)
        else:
            super().set_loss('CrossEntropyLoss', adapter_name=adapter_name)
        template = self.get_template(adapter_name)
        input_features = datum_to_input_feature(inputs, template)
        outputs = super().forward(inputs=input_features, adapter_name=adapter_name, **kwargs)
        loss_values = extract_rl_feature(inputs)
        loss_kwargs = kwargs.copy()
        loss_kwargs.update(loss_values)
        loss = super().calculate_loss(adapter_name=adapter_name, **loss_kwargs)
        super().backward(adapter_name=adapter_name, **kwargs)
        logits = outputs['logits'].detach()
        logps = outputs.get('logps', None)
        if logps is not None:
            logps = logps.detach().cpu()
        results = self._get_forward_output(inputs, logits, logps)
        return [results, loss]

    @remote_function()
    def step(self, *, adam_params: types.AdamParams, **kwargs):
        grad_clip_norm = adam_params.grad_clip_norm
        if grad_clip_norm > 0.0:
            self.clip_grad_norm(max_grad_norm=grad_clip_norm, norm_type=2, **kwargs)
        optim_params = {
            'lr': adam_params.learning_rate,
            'eps': adam_params.eps,
            'betas': (adam_params.beta1, adam_params.beta2),
            'weight_decay': adam_params.weight_decay,
        }
        super().step(optim_params=optim_params, **kwargs)
        super().zero_grad(**kwargs)

    @remote_function(collect='first', lazy_collect=False)
    def calculate_metric(self, is_training, **kwargs):
        metric = super().calculate_metric(is_training, **kwargs)
        return clean_metrics(metric)

    @remote_function()
    def load(self, checkpoint_dir: str, **kwargs):
        """Load checkpoint with token-based isolation support."""
        token = kwargs.pop('token', None)
        if not token:
            raise ValueError('Token is required for loading checkpoints')
        from twinkle.server.common.io_utils import create_checkpoint_manager
        checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
        resolved = checkpoint_manager.resolve_load_path(checkpoint_dir)
        if resolved.is_twinkle_path:
            return super().load(name=resolved.checkpoint_name, output_dir=resolved.checkpoint_dir, **kwargs)
        else:
            return super().load(name=resolved.checkpoint_name, **kwargs)


# ---------------------------------------------------------------------------
# Twinkle-native Transformers model (InputFeature/Trajectory-based I/O)
# ---------------------------------------------------------------------------


@remote_class()
class TwinkleCompatTransformersModelNative(MultiLoraTransformersModel):
    """Twinkle-native wrapper around MultiLoraTransformersModel.

    Input/output is in native InputFeature / Trajectory format.
    Moved from twinkle/common/transformers_model.py.
    """

    @staticmethod
    def _to_cpu_safe_output(obj: Any) -> Any:
        """Convert nested outputs into CPU-safe Python objects for HTTP transport."""
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
            return {key: TwinkleCompatTransformersModelNative._to_cpu_safe_output(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [TwinkleCompatTransformersModelNative._to_cpu_safe_output(value) for value in obj]
        return obj

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        output = super().forward_backward(inputs=inputs, **kwargs)
        return self._to_cpu_safe_output(output)
