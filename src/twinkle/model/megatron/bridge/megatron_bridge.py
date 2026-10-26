# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Dict, List


class _MCoreCompatBridgeShim:
    """mcore-compatible facade over a megatron-bridge ``AutoBridge``: load/save/export with
    twinkle's signatures. Per-adapter saves write only that adapter's LoRA delta; ``converter``
    hooks and ``only_master_rank=True`` are unsupported and fail-fast."""

    def __init__(self, auto_bridge: Any, hf_config: Any):
        self._bridge = auto_bridge
        self._hf_config = hf_config

    @staticmethod
    def _peft_state_dict(mg_models, adapter_name: str):
        """Exactly ``adapter_name``'s LoRA weights in HF PEFT layout (adapter name stripped), the
        content ``adapter_model.safetensors`` expects."""
        from peft import PeftModel
        from peft.utils.save_and_load import get_peft_model_state_dict

        peft_model = next((m for m in mg_models if isinstance(m, PeftModel)), None)
        if peft_model is None:
            raise RuntimeError('peft_format=True save/export expects a PeftModel; got '
                               f'{[type(m).__name__ for m in mg_models]}.')
        name = adapter_name or 'default'
        if name not in peft_model.peft_config:
            raise KeyError(f'adapter {name!r} not on the model; available: {list(peft_model.peft_config)}')
        return get_peft_model_state_dict(peft_model, adapter_name=name)

    @staticmethod
    def _reject_sharded_peft() -> None:
        """Fail-fast on per-adapter save/export under TP or PP > 1: LoRA weights are sharded there
        and this shim does not all-gather them, so rank 0 would write an incomplete adapter."""
        try:
            from megatron.core import parallel_state as mpu
        except Exception:
            return
        if not mpu.model_parallel_is_initialized():
            return
        tp = mpu.get_tensor_model_parallel_world_size()
        pp = mpu.get_pipeline_model_parallel_world_size()
        if tp > 1 or pp > 1:
            raise NotImplementedError(f'per-adapter save/export is only implemented for DP; got tensor_parallel={tp}, '
                                      f'pipeline_parallel={pp}. LoRA weights are sharded under TP/PP and are not '
                                      'all-gathered here, so the written adapter would be incomplete.')

    def load_weights(self,
                     mg_models,
                     hf_model_dir: str,
                     peft_format: bool = False,
                     adapter_name: str = 'default',
                     converter=None):
        """Forward to ``AutoBridge.load_hf_weights``; adapter reload is handled by twinkle's
        PeftModel, so ``peft_format=True`` never applies here."""
        if converter is not None:
            raise NotImplementedError('load_weights does not support a converter hook.')
        if peft_format:
            raise NotImplementedError('load_weights(peft_format=True) is not used on this backend.')
        self._bridge.load_hf_weights(mg_models, hf_model_dir)

    def save_weights(self,
                     mg_models,
                     output_dir: str,
                     peft_format: bool = False,
                     adapter_name: str = 'default',
                     converter=None,
                     max_shard_size: str = '5GB'):
        """peft_format=False: ``AutoBridge.save_hf_weights`` (full HF weights); True: write only
        ``adapter_name``'s LoRA delta as ``adapter_model.safetensors``."""
        if converter is not None:
            raise NotImplementedError('save_weights does not support a converter hook.')
        if max_shard_size != '5GB':
            raise NotImplementedError(
                'megatron-bridge AutoBridge does not expose max_shard_size. Use --bridge_backend mcore-bridge or '
                'remove the custom --max_shard_size value.')
        if peft_format:
            import torch.distributed as dist
            self._reject_sharded_peft()
            state_dict = self._peft_state_dict(mg_models, adapter_name)
            is_global_zero = (not dist.is_initialized()) or dist.get_rank() == 0
            if is_global_zero:
                import os
                from safetensors.torch import save_file
                os.makedirs(output_dir, exist_ok=True)
                cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
                save_file(cpu_sd, os.path.join(output_dir, 'adapter_model.safetensors'))
            if dist.is_initialized():
                dist.barrier()
            return
        # save_hf_weights decides sharding internally and exposes no shard-size override.
        self._bridge.save_hf_weights(mg_models, output_dir)

    def export_weights(self,
                       mg_models,
                       target_device=None,
                       only_master_rank: bool = False,
                       peft_format: bool = False,
                       adapter_name: str = 'default',
                       converter=None,
                       tqdm_desc: str = 'Exporting: ',
                       disable_tqdm: bool = True,
                       _is_saving: bool = False):
        """Yield ``(hf_name, tensor)``: full weights via ``AutoBridge.export_hf_weights``, or only
        ``adapter_name``'s LoRA delta when peft_format. ``only_master_rank=True`` is unsupported."""
        if converter is not None:
            raise NotImplementedError('export_weights does not support a converter hook.')
        if peft_format:
            self._reject_sharded_peft()
            state_dict = self._peft_state_dict(mg_models, adapter_name)
            to_cpu = (target_device == 'cpu')
            for name, tensor in state_dict.items():
                yield name, (tensor.detach().cpu() if to_cpu else tensor.detach())
            return
        if only_master_rank:
            raise NotImplementedError('export_weights only supports only_master_rank=False.')
        cpu = (target_device == 'cpu')
        for name, tensor in self._bridge.export_hf_weights(mg_models, cpu=cpu, show_progress=not disable_tqdm):
            yield name, tensor


class MegatronBridgeBackend:
    """Wraps NVIDIA megatron-bridge (AutoBridge) construction behind the BridgeBackend protocol."""

    backend_name = 'megatron-bridge'

    @property
    def is_multimodal(self) -> bool:
        # TODO: MLLM
        return False

    def build_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], strategy: Any, **kwargs) -> Any:
        """HF config -> a finalized GPTModelProvider with a shim attached as ``.bridge``."""
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge

        model_path = getattr(hf_config, 'name_or_path', None) or getattr(hf_config, '_name_or_path', None)
        if not model_path:
            raise ValueError('MegatronBridgeBackend needs hf_config.name_or_path (a local model dir).')
        trust_remote_code = bool(kwargs.pop('trust_remote_code', True))
        strict_model_kwargs = set(kwargs.pop('_strict_model_kwargs', ()))
        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=trust_remote_code)
        provider = bridge.to_megatron_provider(load_weights=False)

        unsupported = sorted(name for name in strict_model_kwargs if not hasattr(provider, name))
        if unsupported:
            raise NotImplementedError(
                f'The installed megatron-bridge provider for {type(provider).__name__} does not expose explicitly '
                f'requested model options {unsupported}. Use --bridge_backend mcore-bridge, remove these options, '
                'or upgrade megatron-bridge/Megatron-LM.')

        overrides: Dict[str, Any] = dict(parallel_kwargs)
        overrides['sequence_parallel'] = strategy.sequence_parallel
        overrides['variable_seq_lengths'] = strategy.variable_seq_lengths
        # attention_backend is resolved by the caller and arrives through kwargs; both backends
        # share it, so the same Config never picks a different kernel per bridge_backend.
        for k, v in kwargs.items():
            if hasattr(provider, k):
                overrides[k] = v

        # Match the mcore backend's backward/MoE defaults (calculate_per_token_loss, dispatcher
        # type gated on variable_seq_lengths) and fusion-flag alignment; see mcore.py for rationale.
        overrides.setdefault('calculate_per_token_loss', True)
        overrides.setdefault('moe_token_dispatcher_type', 'alltoall' if strategy.variable_seq_lengths else 'allgather')
        for _fusion_flag in ('bias_activation_fusion', 'masked_softmax_fusion', 'bias_dropout_fusion'):
            if hasattr(provider, _fusion_flag):
                overrides.setdefault(_fusion_flag, True)

        provider.apply_overrides_and_finalize(dtype=strategy.params_type, overrides=overrides)
        provider.bridge = _MCoreCompatBridgeShim(bridge, hf_config)
        return provider

    def create_model(self, config: Any, model_dir: str, *, load_weights: bool, move_to_gpu) -> List[Any]:
        """provide_distributed_model(wrap_with_ddp=False) -> bare models, move to GPU, then
        load HF weights; the strategy's wrap_model does the DDP wrap later."""
        import torch.distributed as dist

        provider = config
        mg_models = provider.provide_distributed_model(wrap_with_ddp=False, fp16=provider.fp16, bf16=provider.bf16)
        if not isinstance(mg_models, list):
            mg_models = [mg_models]
        if dist.is_initialized():
            dist.barrier()

        models = [move_to_gpu(m) for m in mg_models]

        if load_weights:
            # config.bridge is the shim; use the underlying AutoBridge for the raw load.
            config.bridge._bridge.load_hf_weights(mg_models, model_dir)
        return models
