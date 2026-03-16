# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Megatron backend model for the unified model deployment.
"""
import torch
from tinker import types
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from twinkle import remote_class, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.model.megatron import MultiLoraMegatronModel
from twinkle.server.common.datum import datum_to_input_feature, extract_rl_feature
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
        if loss_fn == 'importance_sampling':
            super().set_loss('GRPOLoss', adapter_name=adapter_name, epsilon=0.2, beta=0.0)
        template = self.get_template(adapter_name=adapter_name)
        input_features = datum_to_input_feature(inputs, template)
        loss_values = extract_rl_feature(inputs)
        loss_kwargs = kwargs.copy()
        loss_kwargs.update(loss_values)
        outputs = super().forward_backward(inputs=input_features, adapter_name=adapter_name, **loss_kwargs)
        loss = outputs.get('loss', None)
        logits_list = outputs.get('logits', [])
        logps = outputs.get('logps', [])
        if logits_list is None and logps is None:
            return [None, None]

        logits = None
        if logits_list is not None:
            if isinstance(logits_list, torch.Tensor):
                logits = logits_list.detach()
            else:
                logits = torch.cat([logit.detach() for logit in logits_list], dim=0)
        logps = logps.detach().cpu()
        results = self._get_forward_output(inputs, logits, logps)

        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        else:
            loss = float(loss)

        return [results, loss]

    @remote_function(dispatch='slice_dp', collect='flatten')
    def tinker_forward_only(self, *, inputs: List[types.Datum], **kwargs):
        """Forward pass without gradient computation."""
        template = self.get_template(**kwargs)
        input_features = datum_to_input_feature(inputs, template)
        outputs = super().forward_only(inputs=input_features, **kwargs)
        logits = outputs.get('logits', None)
        logps = outputs.get('logps', None)

        if logits is not None:
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu()
            elif isinstance(logits, list) and len(logits) > 0:
                logits = torch.cat([logit.detach().cpu() for logit in logits], dim=0)
            results = self._get_forward_output(inputs, logits, logps)
        else:
            results = [{'logprobs': None, 'elementwise_loss': None} for _ in inputs]

        return results

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

    @remote_function(collect='first', lazy_collect=False)
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

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        """Forward+backward for twinkle-native clients (InputFeature/Trajectory I/O)."""
        output = super().forward_backward(inputs=inputs, **kwargs)
        return to_cpu_safe_output(output)
