# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSpeed training strategy for twinkle.

A peer of ``AccelerateStrategy`` and ``NativeFSDPStrategy``. All DeepSpeed
logic is self-contained here — the existing strategies are not modified.

Design: construct ``DeepSpeedPlugin`` + ``Accelerator`` in ``__init__``.
``wrap_model`` applies ZeRO-3 compatibility patches (from
``twinkle.patch.deepspeed_patches``) around ``accelerator.prepare``.
``get_full_state_dict`` / ``gather_parameters`` use DeepSpeed's
``GatheredParameters`` for ZeRO-3 parameter gathering.
"""
import os
from contextlib import nullcontext
from datetime import timedelta
from typing import Any, Dict, Literal, Mapping, Optional

from twinkle.patch import apply_context, apply_patch
# Re-export for convenience
from twinkle.patch.deepspeed_patches import DeepSpeedLeafModulesPatch  # noqa: F401
from twinkle.patch.deepspeed_patches import (DeepSpeedHookReorderPatch, DeepSpeedModulesToSavePatch,
                                             DeepSpeedParamWrapperPatch)


class DeepSpeedStrategy:
    """A training strategy backed by HuggingFace accelerate + DeepSpeed.

    Args:
        mixed_precision: Mixed precision type.
        deepspeed_config: DeepSpeed config dict. If ``None``, a minimal
            ZeRO-2 config is used.
        device_mesh: Model device mesh (unused by DeepSpeed but kept for
            interface parity with other strategies).
    """

    def __init__(
        self,
        mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
        deepspeed_config: Optional[Dict[str, Any]] = None,
        device_mesh=None,
    ):
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs

        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision

        config = deepspeed_config or {'zero_optimization': {'stage': 2}}
        self._deepspeed_plugin = self._build_plugin(config)

        self.accelerator = Accelerator(
            deepspeed_plugin=self._deepspeed_plugin,
            mixed_precision=mixed_precision,
            kwargs_handlers=[
                InitProcessGroupKwargs(
                    timeout=timedelta(seconds=int(os.environ.get('TWINKLE_DIST_TIMEOUT_SECONDS', '7200')), ), ),
            ],
        )

    # -- plugin construction ------------------------------------------------

    @staticmethod
    def _build_plugin(config: Dict[str, Any]):
        """Build a ``DeepSpeedPlugin`` from a raw config dict.

        Uses ``HfTrainerDeepSpeedConfig`` when available so that ``"auto"``
        values are resolved; falls back to a plain dict otherwise.
        """
        from accelerate.utils import DeepSpeedPlugin

        try:
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            hf_ds_config = HfTrainerDeepSpeedConfig(config)
            return DeepSpeedPlugin(hf_ds_config=hf_ds_config)
        except ImportError:
            return DeepSpeedPlugin(hf_ds_config=config)

    # -- state queries ------------------------------------------------------

    @property
    def is_deepspeed_enabled(self) -> bool:
        return True

    def is_zero_stage(self, stage: int) -> bool:
        cfg = self._deepspeed_plugin.deepspeed_config
        if not isinstance(cfg, dict):
            cfg = getattr(cfg, 'config', {})
        return cfg.get('zero_optimization', {}).get('stage', 0) == stage

    @property
    def zero_stage(self) -> int:
        cfg = self._deepspeed_plugin.deepspeed_config
        if not isinstance(cfg, dict):
            cfg = getattr(cfg, 'config', {})
        return cfg.get('zero_optimization', {}).get('stage', 0)

    # -- twinkle Strategy interface -----------------------------------------

    def pretrained_load_context(self):
        """Context manager for loading pretrained weights.

        Under ZeRO-3, returns ``deepspeed.zero.Init`` so that weights are
        partitioned immediately during ``from_pretrained``. For ZeRO-1/2,
        weights fit in full on each rank, so no special context is needed.
        """
        if self.is_zero_stage(3):
            import deepspeed

            cfg = self._deepspeed_plugin.deepspeed_config
            if not isinstance(cfg, dict):
                cfg = getattr(cfg, 'config', cfg)
            return deepspeed.zero.Init(config_dict_or_path=cfg)
        return nullcontext()

    def capture_pre_ep_state_if_needed(self, model, *, enable_ep: bool) -> None:
        pass

    def prepare_adapter_config(self, config_or_dir, *, enable_ep: bool):
        return config_or_dir

    def wrap_model(self, model, *args):
        """Wrap model with DeepSpeed via ``accelerator.prepare``.

        Patch ordering:
        1. ``DeepSpeedLeafModulesPatch`` (permanent) — ZeRO-3 MoE leaf modules
        2. ``DeepSpeedModulesToSavePatch`` (permanent) — PEFT modules_to_save
        3. ``DeepSpeedHookReorderPatch`` (temporary) — wrap ``deepspeed.initialize``
        4. ``DeepSpeedParamWrapperPatch`` (temporary) — PEFT param metadata

        Steps 3-4 are active only during ``accelerator.prepare``; the wrapper
        restores originals on exit.
        """
        apply_patch(model, DeepSpeedLeafModulesPatch)
        apply_patch(model, DeepSpeedModulesToSavePatch)

        with apply_context(model, DeepSpeedHookReorderPatch):
            with apply_context(model, DeepSpeedParamWrapperPatch):
                return self.accelerator.prepare(model, *args)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model, keep_torch_compile=False)

    def load_peft_weights(
        self,
        model,
        adapter_weights: Mapping[str, Any],
        adapter_name: str,
    ) -> None:
        from peft.utils import set_peft_model_state_dict

        set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)

    def needs_wrapped_optimizer_state(self) -> bool:
        return True

    def save_optimizer_checkpoint(
        self,
        model,
        optimizer,
        output_path: str,
    ) -> None:
        """Save optimizer state via DeepSpeed engine checkpoint.

        ``output_path`` is a file path (e.g. ``.../optimizer.pt``); DeepSpeed
        ``save_checkpoint`` takes a directory, so we use the parent dir.
        Both model and optimizer state are saved — the model portion is
        redundant with ``save()`` but is the standard DeepSpeed format.
        """
        engine = self.unwrap_model(model)
        save_dir = os.path.dirname(output_path)
        engine.save_checkpoint(save_dir)

    def load_optimizer_checkpoint(
        self,
        model,
        optimizer,
        input_path: str,
    ) -> None:
        """Load optimizer state from a DeepSpeed engine checkpoint."""
        from twinkle.utils import get_logger

        logger = get_logger()
        engine = self.unwrap_model(model)
        load_dir = os.path.dirname(input_path)
        try:
            engine.load_checkpoint(load_dir)
        except Exception as e:
            logger.warning(f'Failed to load deepspeed checkpoint: {e}')

    def get_full_state_dict(self, model) -> dict:
        """Collect full state dict, gathering ZeRO-3 partitioned params."""
        unwrapped = self.unwrap_model(model)
        if self.is_zero_stage(3):
            from deepspeed.runtime.zero import GatheredParameters

            params = list(unwrapped.parameters())
            with GatheredParameters(params):
                return {k: v.cpu() for k, v in unwrapped.named_parameters()}
        return {k: v.cpu() for k, v in unwrapped.named_parameters()}

    def load_full_state_dict(self, model, state_dict) -> None:
        """Load a full state dict into the (possibly partitioned) model."""
        unwrapped = self.unwrap_model(model)
        if self.is_zero_stage(3):
            from deepspeed.runtime.zero import GatheredParameters

            params = list(unwrapped.parameters())
            with GatheredParameters(params):
                unwrapped.load_state_dict(state_dict, strict=False)
        else:
            unwrapped.load_state_dict(state_dict, strict=False)

    def get_adapter_state_dict(self, model, adapter_name: str) -> dict:
        """Collect only LoRA adapter parameters, gathering under ZeRO-3."""
        unwrapped = self.unwrap_model(model)
        adapter_suffix = f'.{adapter_name}.'
        state_dict = {}

        if self.is_zero_stage(3):
            from deepspeed.runtime.zero import GatheredParameters

            params = [p for n, p in unwrapped.named_parameters() if _is_lora_state_key(n) and adapter_suffix in n]
            with GatheredParameters(params):
                for n, p in unwrapped.named_parameters():
                    if _is_lora_state_key(n) and adapter_suffix in n:
                        state_dict[n] = p.cpu()
        else:
            for n, p in unwrapped.named_parameters():
                if _is_lora_state_key(n) and adapter_suffix in n:
                    state_dict[n] = p.cpu()
        return state_dict

    # -- DeepSpeed-specific -------------------------------------------------

    def gather_parameters(self, params):
        """Context manager to gather ZeRO-3 partitioned parameters.

        Usage::

            with strategy.gather_parameters(list(model.parameters())):
                # params are fully gathered here
                ...
        """
        if self.is_zero_stage(3):
            from deepspeed.runtime.zero import GatheredParameters

            return GatheredParameters(params)
        return nullcontext()


def _is_lora_state_key(name: str) -> bool:
    return 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name
