# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import shutil
import torch
import torch.distributed as dist
from contextlib import contextmanager
from peft import PeftConfig
from safetensors.torch import load_file, save_file
from torch.optim import Optimizer
from typing import Any, Dict, Optional, Type, Union

from twinkle import DeviceMesh, Platform, remote_class, remote_function
from twinkle.utils.safetensors import StreamingSafetensorSaver
from ..multi_lora_transformers import MultiLoraTransformersModel
from .fft_slots import HybridFftSlots
from .spectral_allocation import load_spectral_allocation

_HYBRID_CONFIG_FIELDS = (
    'r',
    'lora_alpha',
    'lora_dropout',
    'use_rslora',
    'fan_in_fan_out',
    'use_dora',
    'bias',
    'rank_pattern',
    'alpha_pattern',
    'target_modules',
    'modules_to_save',
)
HYBRID_ADAPTER_MODE = 'hybrid'


@remote_class()
class SpectralHybridTransformersModel(MultiLoraTransformersModel):
    """Transformers MultiLoRA service extended with server-owned FFT slots."""

    def __init__(self,
                 hybrid: Dict[str, Any],
                 memory_efficient_init: bool = False,
                 device_mesh: Optional[DeviceMesh] = None,
                 **kwargs):
        if memory_efficient_init:
            raise ValueError(
                'Spectral Hybrid does not support memory_efficient_init because FFT slots require materialized '
                'base weights before FSDP wrapping.')
        config = dict(hybrid or {})
        allocation_path = config.get('allocation_path')
        if not allocation_path:
            raise ValueError('Spectral Hybrid requires allocation_path.')
        s_fft = load_spectral_allocation(allocation_path)
        self.default_lr_lora = float(config.get('default_lr_lora', 2.5e-5))
        self.default_lr_fft = float(config.get('default_lr_fft', 1e-6))
        super().__init__(memory_efficient_init=False, device_mesh=device_mesh, **kwargs)
        self.fft_slots = HybridFftSlots(self.multi_adapter, s_fft)
        self.fft_slots.install_fft_slots()

    @contextmanager
    def _adapter_context(self, adapter_name: str, disable_lora: bool = False):
        with super()._adapter_context(adapter_name, disable_lora=disable_lora) as slot_name:
            if disable_lora:
                self.fft_slots.deactivate_fft_slots()
            else:
                self.fft_slots.activate_fft_slot(adapter_name)
            try:
                yield slot_name
            finally:
                self.fft_slots.deactivate_fft_slots()

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        adapter_mode = kwargs.pop('adapter_mode', 'lora')
        if adapter_mode == 'lora':
            return super().add_adapter_to_model(adapter_name, config_or_dir, **kwargs)
        if adapter_mode != HYBRID_ADAPTER_MODE:
            raise ValueError(f'Unsupported adapter_mode {adapter_mode!r}; expected "lora" or {HYBRID_ADAPTER_MODE!r}.')
        config = self._copy_lora_config(config_or_dir)
        if config.modules_to_save:
            raise ValueError('Hybrid modules_to_save is controlled by the server allocation.')
        if getattr(config, 'target_parameters', None):
            raise ValueError('Hybrid target_parameters is not supported.')
        config.target_modules = set(self.fft_slots.resolve_lora_targets(config.target_modules))
        config.modules_to_save = list(self.fft_slots.s_fft)
        self._register_adapter(adapter_name, config, **kwargs)
        self.fft_slots.register_adapter(adapter_name)

    def _create_param_group(self, adapter_name: str, lr: float = 1e-5, weight_decay: float = 0.01, **kwargs):
        if not self.fft_slots.is_hybrid(adapter_name):
            return super()._create_param_group(adapter_name=adapter_name, lr=lr, weight_decay=weight_decay, **kwargs)
        params = self._get_trainable_parameters(adapter_name)
        fft_token = '.modules_to_save.fft_'
        lora_names = [name for name in params if fft_token not in name]
        fft_names = [name for name in params if fft_token in name]
        groups = []
        if lora_names:
            groups.append({
                'params': [params[name] for name in lora_names],
                'param_names': lora_names,
                'lr': kwargs.get('lr_lora', self.default_lr_lora),
                'weight_decay': weight_decay,
            })
        if fft_names:
            groups.append({
                'params': [params[name] for name in fft_names],
                'param_names': fft_names,
                'lr': kwargs.get('lr_fft', self.default_lr_fft),
                'weight_decay': weight_decay,
            })
        if not groups:
            raise ValueError(f'Spectral Hybrid adapter {adapter_name!r} has no trainable parameters.')
        return groups

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        adapter_name = kwargs.get('adapter_name')
        if not self.fft_slots.is_hybrid(adapter_name):
            return super().set_optimizer(optimizer_cls, **kwargs)
        if 'params' not in kwargs:
            lr_lora = kwargs.pop('lr_lora', kwargs.get('lr', self.default_lr_lora))
            lr_fft = kwargs.pop('lr_fft', self.default_lr_fft)
            kwargs['params'] = self._create_param_group(
                adapter_name,
                weight_decay=kwargs.get('weight_decay', 0.01),
                lr_lora=lr_lora,
                lr_fft=lr_fft,
            )
        return super().set_optimizer(optimizer_cls, **kwargs)

    def _get_trainable_parameters(self, adapter_name):
        params = super()._get_trainable_parameters(adapter_name)
        if not self.fft_slots.is_hybrid(adapter_name):
            return params
        target_modules = self.multi_adapter.find_lora_by_tenant(adapter_name).tenant_config.target_modules
        # MultiLoRA preallocates every slot; only collect S_LORA targets enabled for this tenant.
        params = {
            name: parameter
            for name, parameter in params.items() if self.multi_adapter.match_target_modules(name, target_modules)
        }
        known_parameter_ids = {id(parameter) for parameter in params.values()}
        for name, parameter in self.fft_slots.named_fft_parameters(adapter_name):
            if id(parameter) not in known_parameter_ids:
                params[name] = parameter
                known_parameter_ids.add(id(parameter))
        return params

    def _optimizer_param_name_mapping(self, adapter_name: str, optimizer: Optimizer) -> Dict[str, str]:
        """Extend the LoRA mapping with the Hybrid FFT ``default`` slot."""
        mapping = super()._optimizer_param_name_mapping(adapter_name, optimizer)
        tenant = self.multi_adapter.find_lora_by_tenant(adapter_name)
        physical_token = f'.modules_to_save.fft_{tenant.index}.'
        checkpoint_token = '.modules_to_save.default.'
        for group in optimizer.param_groups:
            for name in group.get('param_names', []):
                checkpoint_name = name.replace(physical_token, checkpoint_token)
                if checkpoint_name != name:
                    mapping[name] = checkpoint_name
        return mapping

    @remote_function(collect='first')
    def get_state_dict(self, **kwargs):
        adapter_name = kwargs.get('adapter_name')
        self._check_adapter_valid(adapter_name)
        state = self.multi_adapter.get_state_dict(adapter_name)
        if self.fft_slots.is_hybrid(adapter_name):
            state.update(self.fft_slots.get_fft_state_dict(adapter_name))
        return state

    def _validate_hybrid_training_checkpoint_boundary(self, adapter_name: str) -> None:
        """Require a quiescent optimizer-step boundary for a lossless checkpoint."""
        optimizer_group = self.optimizer_group[adapter_name]
        if optimizer_group.optimizer is None:
            raise ValueError('Spectral Hybrid optimizer must be configured before save_optimizer=True.')
        for group in optimizer_group.optimizer.param_groups:
            if len(group.get('param_names', [])) != len(group['params']):
                raise ValueError(
                    'Spectral Hybrid lossless checkpoints require optimizer param_names for every parameter.')
        train_status = optimizer_group.train_status
        if train_status.loss_value is not None or train_status.num_tokens != 0:
            raise ValueError('Spectral Hybrid training state can only be saved after the optimizer step and zero_grad.')
        if any(parameter.grad is not None for parameter in self._get_trainable_parameters(adapter_name).values()):
            raise ValueError(
                'Spectral Hybrid training state can only be saved after zero_grad cleared accumulated gradients.')
        if (optimizer_group.cur_step > 0 and optimizer_group.gradient_accumulation_steps > 1
                and not optimizer_group.do_grad_sync()):
            raise ValueError('Spectral Hybrid training state cannot be saved in the middle of gradient accumulation.')

    @staticmethod
    def _class_identity(instance) -> Optional[str]:
        if instance is None:
            return None
        cls = instance.__class__
        return f'{cls.__module__}.{cls.__qualname__}'

    @staticmethod
    def _normalize_hybrid_config(config) -> dict:
        raw = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        normalized = {}
        for field in _HYBRID_CONFIG_FIELDS:
            value = raw.get(field)
            if field in ('target_modules', 'modules_to_save'):
                value = sorted(value or [])
            elif field in ('rank_pattern', 'alpha_pattern'):
                value = dict(sorted((value or {}).items()))
            normalized[field] = value
        return normalized

    def _write_hybrid_training_manifest(self, training_dir: str, adapter_name: str) -> None:
        if not Platform.is_master():
            return
        optimizer_group = self.optimizer_group[adapter_name]
        trainer_state_path = os.path.join(training_dir, 'trainer_state.json')
        with open(trainer_state_path, encoding='utf-8') as handle:
            trainer_state = json.load(handle)
        trainer_state.update({
            'checkpoint_boundary': 'optimizer_step',
            'scheduler_class': self._class_identity(optimizer_group.lr_scheduler),
            'has_scaler': optimizer_group.scaler is not None,
        })
        with open(trainer_state_path, 'w', encoding='utf-8') as handle:
            json.dump(trainer_state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')

    def _validate_hybrid_resume_state(self, training_dir: str, adapter_name: str, saved_config: dict,
                                      trainer_state: dict) -> None:
        optimizer_group = self.optimizer_group[adapter_name]
        current_config = self._normalize_hybrid_config(optimizer_group.adapter_config)
        checkpoint_config = self._normalize_hybrid_config(saved_config)
        differences = {
            field: (checkpoint_config[field], current_config[field])
            for field in _HYBRID_CONFIG_FIELDS if checkpoint_config[field] != current_config[field]
        }
        if differences:
            raise ValueError(f'Spectral Hybrid adapter config does not match checkpoint: {differences}')
        if trainer_state.get('checkpoint_boundary') != 'optimizer_step':
            raise ValueError('Spectral Hybrid checkpoint was not saved at a supported optimizer-step boundary.')
        if optimizer_group.optimizer is None:
            raise ValueError('Spectral Hybrid optimizer must be configured before resuming training.')
        optimizer_path = os.path.join(training_dir, 'optimizer.pt')
        if not os.path.isfile(optimizer_path):
            raise ValueError('Spectral Hybrid training state is missing optimizer.pt.')
        saved_scheduler = trainer_state.get('scheduler_class')
        if saved_scheduler != self._class_identity(optimizer_group.lr_scheduler):
            raise ValueError('Spectral Hybrid scheduler configuration does not match the checkpoint.')
        if saved_scheduler and not os.path.isfile(os.path.join(training_dir, 'scheduler.pt')):
            raise ValueError('Spectral Hybrid training state is missing scheduler.pt.')
        saved_scaler = bool(trainer_state.get('has_scaler'))
        if saved_scaler != (optimizer_group.scaler is not None):
            raise ValueError('Spectral Hybrid grad scaler configuration does not match the checkpoint.')
        if saved_scaler and not os.path.isfile(os.path.join(training_dir, 'scaler.pt')):
            raise ValueError('Spectral Hybrid training state is missing scaler.pt.')
        rank = dist.get_rank() if dist.is_initialized() else 0
        rank_rng_path = os.path.join(training_dir, f'rng_state_rank_{rank}.pt')
        if not os.path.isfile(rank_rng_path):
            raise ValueError(f'Spectral Hybrid training state is missing rank RNG state: {rank_rng_path}')

    def _save_spectral_hybrid(self, name, output_dir: Optional[str], interval: int, adapter_name: str, **kwargs):
        optimizer_group = self.optimizer_group[adapter_name]
        save_optimizer = kwargs.get('save_optimizer', False)
        save_only_training_state = kwargs.get('save_only_training_state', False)
        if save_only_training_state and not save_optimizer:
            raise ValueError('save_only_training_state=True requires save_optimizer=True.')
        if name is None:
            name = f'checkpoint-step-{optimizer_group.cur_step}'
        output_dir = output_dir or 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        if optimizer_group.cur_step % interval != 0:
            return None
        if save_optimizer:
            self._validate_hybrid_training_checkpoint_boundary(adapter_name)
        training_dir = os.path.join(checkpoint_dir, 'twinkle_training_state')
        if Platform.is_master() and os.path.isdir(training_dir):
            shutil.rmtree(training_dir)

        full_state = None
        if not save_only_training_state:
            full_state = self.strategy.get_full_state_dict(self.model)
            saver = StreamingSafetensorSaver(
                checkpoint_dir,
                max_shard_size=kwargs.get('max_shard_size', '5GB'),
                save_rank='master',
            )
            if Platform.is_master():
                for key, value in self.fft_slots.iter_merged_state_dict(adapter_name, full_state):
                    saver.add_tensor(key, value)
            saver.finalize()

            model = self.strategy.unwrap_model(self.model)
            if Platform.is_master():
                self.hf_config.save_pretrained(checkpoint_dir)
                generation_config = getattr(model, 'generation_config', None)
                if generation_config is not None:
                    generation_config.save_pretrained(checkpoint_dir)
                else:
                    generation_config_path = os.path.join(checkpoint_dir, 'generation_config.json')
                    if os.path.exists(generation_config_path):
                        os.unlink(generation_config_path)
            self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)

        if save_optimizer:
            if save_only_training_state:
                tenant = self.multi_adapter.find_lora_by_tenant(adapter_name)
                full_state = self.strategy.get_adapter_state_dict(self.model, tenant.adapter_name)
                full_state.update(self.strategy.get_adapter_state_dict(self.model, f'fft_{tenant.index}'))
            if Platform.is_master():
                adapter_state = self.fft_slots.build_training_state_dict(adapter_name, full_state)
                os.makedirs(training_dir, exist_ok=True)
                optimizer_group.adapter_config.save_pretrained(training_dir)
                config_path = os.path.join(training_dir, 'adapter_config.json')
                with open(config_path, encoding='utf-8') as handle:
                    adapter_config = json.load(handle)
                adapter_config['twinkle_adapter_mode'] = HYBRID_ADAPTER_MODE
                with open(config_path, 'w', encoding='utf-8') as handle:
                    json.dump(adapter_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write('\n')
                save_file(
                    {
                        key: value.contiguous()
                        for key, value in adapter_state.items()
                    },
                    os.path.join(training_dir, 'adapter_model.safetensors'),
                )
            if dist.is_initialized():
                dist.barrier()
            self._save_training_state(
                training_dir,
                adapter_name=adapter_name,
                consumed_train_samples=kwargs.get('consumed_train_samples', 0),
            )
            self._write_hybrid_training_manifest(training_dir, adapter_name)
            rank = dist.get_rank() if dist.is_initialized() else 0
            torch.save(self._get_training_rng_state(), os.path.join(training_dir, f'rng_state_rank_{rank}.pt'))
            if dist.is_initialized():
                dist.barrier()
        return checkpoint_dir

    @remote_function(collect='first')
    def save(self, name, output_dir: Optional[str] = None, interval=1, **kwargs):
        adapter_name = kwargs.pop('adapter_name', None)
        self._check_adapter_valid(adapter_name)
        if not self.fft_slots.is_hybrid(adapter_name):
            return super().save(name, output_dir, interval, adapter_name=adapter_name, **kwargs)
        checkpoint_dir = self._save_spectral_hybrid(name, output_dir, interval, adapter_name, **kwargs)
        if dist.is_initialized():
            dist.barrier()
        return checkpoint_dir

    def _resume_spectral_hybrid(self, checkpoint_dir: str, adapter_name: str, resume_only_model: bool):
        training_dir = os.path.join(checkpoint_dir, 'twinkle_training_state')
        adapter_path = os.path.join(training_dir, 'adapter_model.safetensors')
        trainer_state_path = os.path.join(training_dir, 'trainer_state.json')
        if not os.path.isfile(adapter_path) or not os.path.isfile(trainer_state_path):
            raise ValueError('Cannot resume Spectral Hybrid training from a merged-only checkpoint. '
                             'Save the checkpoint with save_optimizer=True to create twinkle_training_state.')
        adapter_config_path = os.path.join(training_dir, 'adapter_config.json')
        if not os.path.isfile(adapter_config_path):
            raise ValueError('Spectral Hybrid training state is missing adapter_config.json.')
        with open(adapter_config_path, encoding='utf-8') as handle:
            saved_config = json.load(handle)
        if saved_config.get('twinkle_adapter_mode') != HYBRID_ADAPTER_MODE:
            raise ValueError('Checkpoint training state is not a Spectral Hybrid adapter.')
        with open(trainer_state_path, encoding='utf-8') as handle:
            trainer_state = json.load(handle)
        if not resume_only_model:
            self._validate_hybrid_resume_state(training_dir, adapter_name, saved_config, trainer_state)
        else:
            current_config = self._normalize_hybrid_config(self.optimizer_group[adapter_name].adapter_config)
            checkpoint_config = self._normalize_hybrid_config(saved_config)
            if current_config != checkpoint_config:
                raise ValueError('Spectral Hybrid adapter config does not match checkpoint.')

        rank = dist.get_rank() if dist.is_initialized() else 0
        adapter_state = load_file(adapter_path, device='cpu')
        self.multi_adapter.set_state_dict(adapter_name, adapter_state)
        self.fft_slots.set_fft_state_dict(adapter_name, adapter_state)
        if not resume_only_model:
            trainer_state = self._restore_training_state(training_dir, adapter_name=adapter_name)
            self._load_rng_state(os.path.join(training_dir, f'rng_state_rank_{rank}.pt'))
        return {
            'cur_step': trainer_state['cur_step'],
            'consumed_train_samples': trainer_state['consumed_train_samples'],
            'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
        }

    @remote_function(dispatch='all', collect='first', sync=True)
    def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
        adapter_name = kwargs.get('adapter_name', '')
        self._check_adapter_valid(adapter_name)
        if not self.fft_slots.is_hybrid(adapter_name):
            return super().resume_from_checkpoint(checkpoint_dir, resume_only_model=resume_only_model, **kwargs)
        result = self._resume_spectral_hybrid(checkpoint_dir, adapter_name, resume_only_model)
        if dist.is_initialized():
            dist.barrier()
        return result

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if self.fft_slots.is_hybrid(adapter_name):
            self.fft_slots.reset_adapter_slot(adapter_name)
            self.fft_slots.unregister_adapter(adapter_name)
        return super().remove_adapter(adapter_name)
