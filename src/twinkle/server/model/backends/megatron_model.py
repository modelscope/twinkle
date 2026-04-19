# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Megatron backend model for the unified model deployment.
"""
import torch
from tinker import types
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from twinkle import remote_class, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.infra import collect_tensor_dict
from twinkle.model.megatron import MultiLoraMegatronModel
from twinkle.server.common.datum import datum_to_input_feature, extract_rl_features_for_loss
from twinkle.server.model.backends.common import (TwinkleCompatModelBase, clean_metrics,
                                                  collect_forward_backward_results, to_cpu_safe_output)


@remote_class(execute='all')
class TwinkleCompatMegatronModel(MultiLoraMegatronModel, TwinkleCompatModelBase):
    """Compatibility wrapper around MultiLoraMegatronModel for Twinkle/Tinker.

    Moved from tinker/common/megatron_model.py — logic unchanged.
    """

    @remote_function(dispatch='slice_dp', collect=collect_forward_backward_results, sync=True)
    def tinker_forward_backward(self, *, inputs: List[types.Datum], adapter_name: str, loss_fn: str, **kwargs):
        """Combined forward and backward pass."""
        self._tinker_setup_loss(loss_fn, inputs, adapter_name, kwargs)
        template = self.get_template(adapter_name=adapter_name)
        input_features = datum_to_input_feature(inputs, template)
        loss_values = extract_rl_features_for_loss(inputs)
        loss_kwargs = kwargs.copy()
        self._apply_ref_outputs(loss_values, loss_kwargs, adapter_name)
        loss_kwargs.update(loss_values)

        outputs = super().forward_backward(inputs=input_features, adapter_name=adapter_name, **loss_kwargs)
        loss = outputs.get('loss', None)
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        else:
            loss = float(loss) if loss is not None else 0.0
        results = self._tinker_build_output(inputs, outputs)
        return [results, loss]

    @remote_function(dispatch='slice_dp', collect=collect_forward_backward_results)
    def tinker_forward_only(self, *, inputs: List[types.Datum], adapter_name: str = None, **kwargs):
        """Forward pass without gradient computation."""
        template = self.get_template(adapter_name)
        input_features = datum_to_input_feature(inputs, template)
        outputs = super().forward_only(inputs=input_features, adapter_name=adapter_name, **kwargs)
        results = self._tinker_build_output(inputs, outputs, return_full_logprobs=True)
        return [results, 0.0]

    @remote_function(dispatch='all')
    def tinker_step(self, *, adam_params: types.AdamParams, **kwargs):
        """Optimizer step with AdamParams configuration."""
        adapter_name = kwargs.get('adapter_name')
        optimizer_config = self.optimizer_group.get(adapter_name)

        if optimizer_config and optimizer_config.optimizer:
            opt = optimizer_config.optimizer
            if hasattr(opt, 'chained_optimizers'):
                for chained_opt in opt.chained_optimizers:
                    if hasattr(chained_opt, 'config'):
                        chained_opt.config.lr = adam_params.learning_rate
                        chained_opt.config.adam_eps = adam_params.eps
                        chained_opt.config.adam_beta1 = adam_params.beta1
                        chained_opt.config.adam_beta2 = adam_params.beta2
                        chained_opt.config.weight_decay = adam_params.weight_decay
                        if adam_params.grad_clip_norm > 0:
                            chained_opt.config.clip_grad = adam_params.grad_clip_norm

        super().step(**kwargs)
        super().zero_grad(**kwargs)

    @remote_function(collect='last_pp_first', lazy_collect=False)
    def tinker_calculate_metric(self, is_training, **kwargs):
        metric = super().calculate_metric(is_training, **kwargs)
        return clean_metrics(metric)

    @remote_function(dispatch='all', sync=True)
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

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        """Forward-only for twinkle-native clients (InputFeature/Trajectory I/O)."""
        output = super().forward_only(inputs=inputs, **kwargs)
        return to_cpu_safe_output(output)

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        """Forward+backward for twinkle-native clients (InputFeature/Trajectory I/O)."""
        output = super().forward_backward(inputs=inputs, **kwargs)
        return to_cpu_safe_output(output)
