# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Backend model implementations for the unified model deployment.

Contains one unified class:
- TwinkleCompatTransformersModel: handles both tinker (Datum-based I/O) via /tinker/*
  endpoints and twinkle-native (InputFeature/Trajectory-based I/O) via /twinkle/* endpoints.
"""
from tinker import types
from typing import List, Union

from twinkle import remote_class, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.model import MultiLoraTransformersModel
from twinkle.server.common.datum import datum_to_input_feature, extract_rl_feature
from twinkle.server.model.backends.common import (TwinkleCompatModelBase, clean_metrics,
                                                  collect_forward_backward_results, to_cpu_safe_output)


@remote_class()
class TwinkleCompatTransformersModel(MultiLoraTransformersModel, TwinkleCompatModelBase):
    """Unified wrapper around MultiLoraTransformersModel.

    Handles both:
    - Tinker-compat I/O (Datum / TensorData) via /tinker/* endpoints.
    - Twinkle-native I/O (InputFeature / Trajectory) via /twinkle/* endpoints.
    """

    # ------------------------------------------------------------------
    # Tinker-compat methods (Datum-based I/O)
    # ------------------------------------------------------------------

    @remote_function(dispatch='slice_dp', collect='flatten')
    def tinker_forward_only(self, *, inputs: List[types.Datum], **kwargs):
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
    def tinker_forward_backward(self, *, inputs: List[types.Datum], adapter_name: str, loss_fn: str, **kwargs):
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
    def tinker_step(self, *, adam_params: types.AdamParams, **kwargs):
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
    def tinker_calculate_metric(self, is_training, **kwargs):
        metric = super().calculate_metric(is_training, **kwargs)
        return clean_metrics(metric)

    @remote_function()
    def tinker_load(self, checkpoint_dir: str, **kwargs):
        """Load checkpoint with token-based isolation support."""
        token = kwargs.pop('token', None)
        if not token:
            raise ValueError('Token is required for loading checkpoints')
        from twinkle.server.common.checkpoint_factory import create_checkpoint_manager
        checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
        resolved = checkpoint_manager.resolve_load_path(checkpoint_dir)
        if resolved.is_twinkle_path:
            return super().load(name=resolved.checkpoint_name, output_dir=resolved.checkpoint_dir, **kwargs)
        else:
            return super().load(name=resolved.checkpoint_name, **kwargs)

    # ------------------------------------------------------------------
    # Twinkle-native methods (InputFeature/Trajectory-based I/O)
    # ------------------------------------------------------------------

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        """Forward+backward for twinkle-native clients (InputFeature/Trajectory I/O)."""
        output = super().forward_backward(inputs=inputs, **kwargs)
        return to_cpu_safe_output(output)
