# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Backend model implementations for the unified model deployment.

Contains the shared Tinker/Twinkle compatibility mixin and two concrete
wrappers:
- TwinkleCompatTransformersModel: multi-LoRA training (default), wraps
  MultiLoraTransformersModel.
- TwinkleCompatFullTransformersModel: exclusive full-parameter training, wraps
  the plain TransformersModel (no PEFT/MultiLora).

Both handle tinker (Datum-based I/O) via /tinker/* endpoints and twinkle-native
(InputFeature/Trajectory-based I/O) via /twinkle/* endpoints.
"""
from tinker import types

from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.model import MultiLoraTransformersModel, SpectralHybridTransformersModel
from twinkle.model.transformers import TransformersModel
from twinkle.server.model.backends.common import (TwinkleCompatModelBase, clean_metrics,
                                                  collect_forward_backward_results, decision_forward_output,
                                                  to_cpu_safe_output)
from twinkle.server.model.tinker_datum import datum_to_input_feature, extract_rl_features_for_loss


class _TransformersTinkerCompatMixin(TwinkleCompatModelBase):
    """Tinker/Twinkle-compat methods shared by LoRA and full-parameter wrappers.

    Method bodies only call ``super()``, so they work against either
    ``MultiLoraTransformersModel`` (LoRA) or ``TransformersModel`` (full) as the
    next class in the MRO. For full-parameter training the ``adapter_name`` is
    the empty-string default optimizer group.
    """

    # ------------------------------------------------------------------
    # Tinker-compat methods (Datum-based I/O)
    # ------------------------------------------------------------------

    @remote_function(dispatch='slice_dp', collect=collect_forward_backward_results)
    def tinker_forward_only(self, *, inputs: list[types.Datum], adapter_name: str = None, **kwargs):
        template = self.get_template(adapter_name)
        input_features = datum_to_input_feature(inputs, template)
        outputs = super().forward_only(inputs=input_features, adapter_name=adapter_name, **kwargs)
        # Tinker contract: one logprob per input token. Return the unpadded slice;
        # the DPO reference round-trip stays intact because the server re-pads
        # ragged client-supplied ref_logps in extract_rl_features_for_loss.
        results = self._tinker_build_output(inputs, outputs)
        return [results, 0.0]

    @remote_function(dispatch='slice_dp', collect=collect_forward_backward_results)
    def tinker_forward_backward(self, *, inputs: list[types.Datum], adapter_name: str, loss_fn: str, **kwargs):
        self._tinker_setup_loss(loss_fn, inputs, adapter_name, kwargs)
        template = self.get_template(adapter_name)
        input_features = datum_to_input_feature(inputs, template)
        outputs = super().forward(inputs=input_features, adapter_name=adapter_name, **kwargs)
        loss_values = extract_rl_features_for_loss(inputs)
        loss_kwargs = kwargs.copy()
        self._apply_ref_outputs(loss_values, loss_kwargs, adapter_name)
        loss_kwargs.update(loss_values)
        loss = super().calculate_loss(adapter_name=adapter_name, **loss_kwargs)
        super().backward(adapter_name=adapter_name, **kwargs)
        results = self._tinker_build_output(inputs, outputs)
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
    def tinker_load(self, *, checkpoint_name: str, output_dir: str | None = None, **kwargs):
        """Load a checkpoint from an already-resolved location.

        Path resolution (token isolation, twinkle-vs-external path shapes) belongs to the
        handler layer: it is a server storage policy, and this class runs inside a Ray
        actor as a compute backend. Keeping it here meant a checkpoint-layout change had
        to touch GPU-side code -- the hardest layer to test -- and the resolution block
        was duplicated verbatim in the megatron backend.
        """
        if output_dir is not None:
            return super().load(name=checkpoint_name, output_dir=output_dir, **kwargs)
        return super().load(name=checkpoint_name, **kwargs)

    # ------------------------------------------------------------------
    # Twinkle-native methods (InputFeature/Trajectory-based I/O)
    # ------------------------------------------------------------------

    @remote_function(dispatch='slice_dp', collect='tensor_dict')
    def forward_only(self, *, inputs: InputFeature | list[InputFeature] | Trajectory | list[Trajectory],
                     distillation_topk: int | None = None, **kwargs):
        """Forward-only for native clients, with optional compact distillation output."""
        if distillation_topk is None:
            output = super().forward_only(inputs=inputs, **kwargs)
            if kwargs.get('task') == 'decision' and kwargs.get('return_logits'):
                return decision_forward_output(output)
            return to_cpu_safe_output(output)
        if self.device_mesh is not None and (self.device_mesh.tp_world_size > 1 or self.device_mesh.cp_world_size > 1):
            raise ValueError('Top-K distillation scoring requires TP=CP=1.')
        adapter_name = kwargs.get('adapter_name')
        if adapter_name is None:
            adapter_name = self._get_default_group()
        if self.optimizer_group[adapter_name].processor.padding_side != 'right':
            raise ValueError('Top-K sequence scoring requires right padding.')
        if (kwargs.get('task', 'causal_lm') != 'causal_lm' or kwargs.get('return_logits')
                or kwargs.get('output_fields') or kwargs.get('output_frame', 'sequence') != 'sequence'):
            raise ValueError('distillation_topk requests its own output for causal LM sequences.')
        if isinstance(inputs, dict):
            inputs = [inputs]
        if not inputs or any(not row.get('input_ids') or not isinstance(row['input_ids'][0], int) for row in inputs):
            raise ValueError('distillation_topk requires encoded token sequences.')
        lengths = [len(feature['input_ids']) for feature in inputs]
        kwargs['output_fields'] = ('logits',)
        logits = super().forward_only(inputs=inputs, **kwargs)['logits']
        if logits.ndim != 3 or not 2 <= distillation_topk <= logits.shape[-1] or logits.shape[1] < max(lengths):
            raise ValueError('Cannot select Top-K candidates from the teacher logits for this batch.')
        # Only mass relative to the retained candidates matters to GKDLoss, including at other temperatures.
        scores, indices = logits.topk(distillation_topk, dim=-1)
        logprobs = scores.float().log_softmax(dim=-1)
        return {
            'teacher_topk_indices': [row[:length] for row, length in zip(indices.tolist(), lengths, strict=True)],
            'teacher_topk_logprobs': [row[:length] for row, length in zip(logprobs.tolist(), lengths, strict=True)],
        }

    @remote_function(dispatch='slice_dp', collect='tensor_dict')
    def forward_backward(self, *, inputs: InputFeature | list[InputFeature] | Trajectory | list[Trajectory], **kwargs):
        """Forward+backward for twinkle-native clients (InputFeature/Trajectory I/O)."""
        self._normalize_ref_outputs(kwargs)
        output = super().forward_backward(inputs=inputs, **kwargs)
        return to_cpu_safe_output(output)

    @remote_function(collect='first', lazy_collect=False, sync=True, timeout=4)
    def ping(self) -> bool:
        """Lightweight liveness probe for watchdog health checks."""
        return True


@remote_class()
class TwinkleCompatTransformersModel(_TransformersTinkerCompatMixin, MultiLoraTransformersModel):
    """Unified multi-LoRA wrapper around MultiLoraTransformersModel."""


@remote_class()
class TwinkleCompatSpectralHybridTransformersModel(_TransformersTinkerCompatMixin, SpectralHybridTransformersModel):
    """Unified API wrapper for the Spectral Hybrid Transformers model."""


@remote_class()
class TwinkleCompatFullTransformersModel(_TransformersTinkerCompatMixin, TransformersModel):
    """Full-parameter (non-LoRA) wrapper around the plain TransformersModel.

    Used by exclusive full-parameter server deployments. All training operations
    target the empty-string default optimizer group created by
    ``TransformersModel.__init__``.
    """

    def __init__(self, *args, device_mesh: DeviceMesh = None, task=None, num_labels=None,
                 task_head_loader=None, **kwargs):
        super().__init__(*args, device_mesh=device_mesh, **kwargs)
        if task is not None or task_head_loader is not None:
            self._add_task_head('', task or 'decision', task_head_loader, num_labels)
