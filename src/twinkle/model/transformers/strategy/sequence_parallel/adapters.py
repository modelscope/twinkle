"""Model-family SP adapters and their small registry.

Adapters own model-family detection and the model-specific setup steps. The
sequence-parallel context remains responsible for generic padding, sharding and
collectives; adapters only install behavior that depends on model internals.
"""
from dataclasses import dataclass
from typing import Tuple

import torch

from twinkle.patch import apply_patch
from twinkle.patch.transformers_qwen3_vl_deepstack import Qwen3VLDeepstackSPPatch
from .data_keys import DataKeySpec
from .linear_attention_sp import Qwen3_5GatedDeltaNetUlyssesPatch, _iter_qwen35_gated_delta_net_classes
from .utils import GatherLoss, is_moe_config


class ModelSPAdapter:
    """Base contract for model-specific SP behavior."""

    name = 'generic'
    position_ids_mode = 'auto'

    def matches(self, model) -> bool:
        return False

    def input_key_specs(self) -> Tuple[DataKeySpec, ...]:
        return ()

    def output_key_specs(self) -> Tuple[DataKeySpec, ...]:
        return ()

    def configure(self, context, model) -> None:
        """Install hooks on the model's text root. Generic adapter is a no-op."""


class Qwen3VLSPAdapter(ModelSPAdapter):
    name = 'qwen3_vl'
    position_ids_mode = 'mrope'

    def matches(self, model) -> bool:
        config = getattr(model, 'config', None)
        model_type = str(getattr(config, 'model_type', '') or '')
        return 'qwen3_vl' in model_type

    def configure(self, context, model) -> None:
        apply_patch(model, Qwen3VLDeepstackSPPatch, sequence_parallel=context)


class Qwen35LinearAttentionSPAdapter(ModelSPAdapter):
    name = 'qwen3_5_linear_attention'

    @staticmethod
    def _matches_config_or_class(model) -> bool:
        config = getattr(model, 'config', None)
        model_type = str(getattr(config, 'model_type', '') or '')
        if model_type in {'qwen3_5', 'qwen3_5_moe'}:
            return True
        architectures = getattr(config, 'architectures', None) or []
        if any('Qwen3_5' in str(architecture) for architecture in architectures):
            return True
        module_name = getattr(model.__class__, '__module__', '') or ''
        return ('transformers.models.qwen3_5' in module_name
                or 'transformers.models.qwen3_5_moe' in module_name)

    def matches(self, model) -> bool:
        if self._matches_config_or_class(model):
            return True
        gated_delta_net_classes = tuple(_iter_qwen35_gated_delta_net_classes())
        return bool(gated_delta_net_classes) and any(
            isinstance(module, gated_delta_net_classes) for module in model.modules())

    def configure(self, context, model) -> None:
        if int(context.rp_world_size or 1) > 1:
            raise NotImplementedError(
                'SequenceParallel: Qwen3.5 linear attention sequence parallel does not support '
                'rp_world_size > 1 (derived ring attention).')
        apply_patch(None, Qwen3_5GatedDeltaNetUlyssesPatch, sequence_parallel=context)


class MoEAuxSPAdapter(ModelSPAdapter):
    name = 'moe_aux'

    def matches(self, model) -> bool:
        config = getattr(model, 'config', None)
        model_type = str(getattr(config, 'model_type', '') or '')
        return model_type.endswith('_moe') or is_moe_config(config)

    def configure(self, context, base_model) -> None:
        def gather_router_logits(module, args, kwargs, output):
            router_logits = getattr(output, 'router_logits', None)
            if router_logits is None:
                return output
            attention_mask = kwargs.get('attention_mask')
            batch_size = 1 if attention_mask is None else attention_mask.shape[0]
            assert router_logits[0].shape[0] % batch_size == 0
            local_seq_len = router_logits[0].shape[0] // batch_size
            gathered_per_sample = []
            for batch_idx in range(batch_size):
                seq_slice = slice(batch_idx * local_seq_len, (batch_idx + 1) * local_seq_len)
                sample_logits = [layer_logits[seq_slice] for layer_logits in router_logits]
                compute_device = sample_logits[0].device
                sample_logits = torch.stack([logits.to(compute_device) for logits in sample_logits], dim=0)
                sample_logits, _ = GatherLoss.apply(context, sample_logits, None, 1, context.real_position_ids)
                gathered_per_sample.append(sample_logits)
            router_logits = torch.stack(gathered_per_sample, dim=0)
            if context.real_position_ids is not None:
                router_logits = router_logits[:, :, :context.real_position_ids.shape[1], :]
            output['router_logits'] = tuple(
                logits.reshape(-1, logits.shape[-1]) for logits in router_logits.split(1, dim=1))
            return output

        base_model.register_forward_hook(gather_router_logits, with_kwargs=True)


_ADAPTERS = [Qwen3VLSPAdapter(), Qwen35LinearAttentionSPAdapter(), MoEAuxSPAdapter()]


def adapters_for_model(model):
    """Return every matching adapter; families can compose capabilities."""
    return tuple(adapter for adapter in _ADAPTERS if adapter.matches(model))
