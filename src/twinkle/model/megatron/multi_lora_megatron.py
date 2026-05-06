# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import os
import random
import re
import torch
import torch.distributed as dist
import torch.nn as nn
from contextlib import contextmanager
from functools import partial
from peft import LoraConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig, PretrainedConfig
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from twinkle import DeviceMesh, remote_class, remote_function, requires, template, torch_util
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.infra import collect_tensor_dict
from twinkle.loss import Loss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from ..multi_lora import MultiLora
from ._mindspeed_runtime import ensure_mindspeed_adaptor_patched
from .megatron import MegatronModel
from .strategy import MegatronStrategy


@remote_class(execute='all')
class MultiLoraMegatronModel(MegatronModel):

    def __init__(
        self,
        model_id: str,
        config: Optional[PretrainedConfig] = None,
        ddp_config: Optional[Dict[str, Any]] = None,
        device_mesh: Optional[DeviceMesh] = None,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        load_weights: bool = True,
        recompute_granularity: Optional[str] = 'full',  # Activation checkpointing
        recompute_method: Optional[str] = 'uniform',
        recompute_num_layers: Optional[int] = 1,
        recompute_modules: Optional[list] = None,  # Modules to recompute
        max_loras: int = 5,
        max_r: int = 32,
        max_length: int = 8192,
        target_modules: Union[List[str], str] = 'all-linear',
        **kwargs,
    ):
        requires('megatron_core')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        nn.Module.__init__(self)
        from twinkle.patch.megatron_peft import MegatronPeft

        self.model_id = model_id
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision

        self._model_path = HubOperation.download_model(model_id)
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        self._default_tokenizer = None
        self.use_distributed_optimizer = False
        self.variable_seq_lengths = kwargs.get('variable_seq_lengths', True)
        self.optimizer_group = {}
        torch_util.set_device()
        self._try_init_process_group()
        ensure_mindspeed_adaptor_patched()
        requires('mcore_bridge')

        kwargs.update({
            'recompute_granularity': recompute_granularity,
            'recompute_modules': recompute_modules,
            'recompute_method': recompute_method,
            'recompute_num_layers': recompute_num_layers,
            'variable_seq_lengths': self.variable_seq_lengths,
        })
        seed = kwargs.pop('seed', None) or int(os.environ.get('TWINKLE_SEED', 42))
        if config is None:
            self.hf_config = AutoConfig.from_pretrained(self._model_path, trust_remote_code=True)
        else:
            self.hf_config = config
        self.strategy = MegatronStrategy(
            self._model_path,
            self.device_mesh,
            mixed_precision=mixed_precision,
            config=self.hf_config,
            ddp_config=ddp_config or {},
            seed=seed,
            use_distributed_optimizer=self.use_distributed_optimizer,
            **kwargs)
        self.model: List[nn.Module] = self.strategy.create_megatron_model(load_weights)
        MegatronPeft().__call__()
        self.multi_adapter = MultiLora(max_loras=max_loras, max_r=max_r, max_length=max_length)
        self.model = self.multi_adapter.patch(self.model, target_modules=target_modules)
        self.model = self.strategy.wrap_model(self.model)
        self.strategy.finish_param_config(self.model, None)
        self.multi_adapter.save_initial_weights()
        self._model_wrapped = True
        self._finish_config = True
        # Active group for compatibility with single adapter
        self.active_group = None
        self.multi_adapter.reset_adapter_status()

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name and adapter_name in self.optimizer_group, (f'Use a valid adapter_name first, '
                                                                       f'current is: {adapter_name}')

    def _lazy_wrap_model(self):
        pass

    def _lazy_finish_param_config(self):
        pass

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict, sync=True)
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        """Forward pass without gradient computation.

        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments including disable_lora.

        Returns:
            Model outputs.
        """
        adapter_name = kwargs.get('adapter_name')
        disable_lora = kwargs.get('disable_lora', False)
        self._check_adapter_valid(adapter_name)
        with self.multi_adapter.adapter(adapter_name, disable_lora=disable_lora):
            return super().forward_only(inputs=inputs, **kwargs)

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict, sync=True)
    def forward_backward(self,
                         *,
                         inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         num_microbatches: int = 1,
                         **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().forward_backward(inputs=inputs, num_microbatches=num_microbatches, **kwargs)

    @remote_function(dispatch='all')
    def step(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().step(**kwargs)

    @remote_function(dispatch='all')
    def zero_grad(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().lr_step(**kwargs)

    @remote_function(dispatch='all')
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        return super().set_loss(loss_cls, **kwargs)

    @contextmanager
    def optimizer_context(self, adapter_name: str):
        """Temporarily replace named_parameters on each module in self.model
        so that only parameters belonging to ``adapter_name`` are visible."""
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        originals = []
        for module in self.model:
            orig = module.named_parameters

            def make_filtered(orig_fn):

                def filtered(prefix: str = '', recurse: bool = True, **kwargs):
                    for name, param in orig_fn(prefix=prefix, recurse=recurse, **kwargs):
                        if param.requires_grad and pattern.search(name):
                            yield name, param

                return filtered

            module.named_parameters = make_filtered(orig)
            originals.append((module, orig))
        try:
            yield
        finally:
            for module, orig in originals:
                module.named_parameters = orig

    @remote_function(dispatch='all')
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')) as adapter_name:
            with self.optimizer_context(adapter_name):
                # Multi lora cannot config use_distributed_optimizer/loss_scale/mix_precision
                kwargs.pop('use_distributed_optimizer', None)
                kwargs.pop('loss_scale', None)
                kwargs['fp16'] = False
                kwargs['bf16'] = True
                super().set_optimizer(optimizer_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_lr_scheduler(scheduler_cls, **kwargs)

    @staticmethod
    def _rank_local_optimizer_path(checkpoint_dir: str) -> str:
        rank = dist.get_rank() if dist.is_initialized() else 0
        return os.path.join(checkpoint_dir, f'optimizer_rank_{rank}.pt')

    @staticmethod
    def _save_local_training_rng_state():
        from megatron.core import tensor_parallel

        rng_state = {
            'random_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state['cuda_rng_state'] = torch.cuda.get_rng_state()
        rng_state['rng_tracker_states'] = tensor_parallel.get_cuda_rng_tracker().get_states()
        return rng_state

    @staticmethod
    def _load_local_training_rng_state(rng_state):
        from megatron.core import tensor_parallel

        random.setstate(rng_state['random_rng_state'])
        np.random.set_state(rng_state['np_rng_state'])
        torch.set_rng_state(rng_state['torch_rng_state'])
        if 'cuda_rng_state' in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_state['rng_tracker_states'])

    def _save_multi_lora_optimizer(self, checkpoint_dir: str, optimizer_config, **kwargs):
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_dict = {
            'checkpoint_version': 1,
            'iteration': optimizer_config.cur_step,
            'rng_state': self._save_local_training_rng_state(),
        }
        if optimizer_config.optimizer is not None:
            state_dict['optimizer'] = optimizer_config.optimizer.state_dict()
        if optimizer_config.lr_scheduler is not None:
            state_dict['opt_param_scheduler'] = optimizer_config.lr_scheduler.state_dict()

        torch.save(state_dict, self._rank_local_optimizer_path(checkpoint_dir))

    def _load_multi_lora_optimizer(self, checkpoint_dir: str, adapter_name: str = '', **kwargs):
        no_load_optim = kwargs.pop('no_load_optim', False)
        no_load_rng = kwargs.pop('no_load_rng', False)
        optimizer_config = self.optimizer_group.get(adapter_name)
        state_dict = torch.load(self._rank_local_optimizer_path(checkpoint_dir), map_location='cpu', weights_only=False)

        if not no_load_optim and optimizer_config is not None:
            if optimizer_config.optimizer is not None and 'optimizer' in state_dict:
                optimizer_config.optimizer.load_state_dict(state_dict['optimizer'])
            if optimizer_config.lr_scheduler is not None and 'opt_param_scheduler' in state_dict:
                optimizer_config.lr_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        if not no_load_rng and 'rng_state' in state_dict:
            self._load_local_training_rng_state(state_dict['rng_state'])
        if optimizer_config is not None and 'iteration' in state_dict:
            optimizer_config.cur_step = state_dict['iteration']

    @remote_function(dispatch='all', collect='first', sync=True)
    def save(self, name, output_dir: Optional[str] = None, interval=1, **kwargs):
        adapter_name = kwargs.pop('adapter_name', None)
        self._check_adapter_valid(adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if optimizer_config.cur_step % interval != 0:
            return

        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        save_optimizer = kwargs.pop('save_optimizer', False)

        with self.multi_adapter.save_context(adapter_name) as real_adapter_name:
            save_format = kwargs.pop('save_format', 'hf')  # 'hf' or 'megatron'
            # Use partial to bind adapter_name to save_lora_converter
            lora_converter = partial(self.multi_adapter.save_lora_converter, adapter_name=real_adapter_name)
            # Mask non-target LoraParallelLinear modules so the bridge skips them,
            # avoiding Megatron-vs-HF key format mismatch in save_lora_converter.
            with self.multi_adapter.save_hf_key_context(real_adapter_name):
                if save_format == 'hf':
                    self._save_hf_format(checkpoint_dir, real_adapter_name, lora_converter=lora_converter)
                else:
                    self._save_megatron_format(checkpoint_dir, real_adapter_name, lora_converter=lora_converter)

            self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)
            if save_optimizer:
                with self.optimizer_context(real_adapter_name):
                    self._save_multi_lora_optimizer(
                        checkpoint_dir,
                        optimizer_config=optimizer_config,
                        **kwargs,
                    )
                trainer_state = {
                    'checkpoint_version': 1,
                    'cur_step': optimizer_config.cur_step,
                    'consumed_train_samples': kwargs.get('consumed_train_samples', 0),
                    'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
                }
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    with open(os.path.join(checkpoint_dir, 'trainer_state.json'), 'w') as f:
                        json.dump(trainer_state, f, indent=2)
            # Final synchronization to ensure all ranks complete save
            if dist.is_initialized():
                dist.barrier()

            return checkpoint_dir

    @remote_function(dispatch='all')
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        load_optimizer = kwargs.pop('load_optimizer', False)
        adapter_name = kwargs.pop('adapter_name', None)
        if output_dir is None:
            if os.path.exists(name):
                checkpoint_dir = name
            else:
                token = kwargs.pop('token', None)
                checkpoint_dir = HubOperation.download_model(name, token=token)
        else:
            checkpoint_dir = os.path.join(output_dir, name)
        bridge = self.strategy.bridge
        with self.multi_adapter.save_context(adapter_name) as real_adapter_name:
            model = self.strategy.unwrap_model(self.model)
            bridge.load_weights(
                model,
                checkpoint_dir,
                peft_format=True,
                adapter_name=real_adapter_name,
                converter=self.multi_adapter.load_lora_converter)

        if load_optimizer:
            with self.optimizer_context(real_adapter_name):
                self._load_multi_lora_optimizer(checkpoint_dir, adapter_name=adapter_name, **kwargs)

        if dist.is_initialized():
            dist.barrier()

    @remote_function(dispatch='all', collect='first', sync=True)
    def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
        adapter_name = kwargs.pop('adapter_name', None)
        self._check_adapter_valid(adapter_name)

        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)

        self.load(checkpoint_dir, load_optimizer=not resume_only_model, adapter_name=adapter_name, **kwargs)

        optimizer_config = self.optimizer_group.get(adapter_name)
        if not resume_only_model and optimizer_config is not None:
            optimizer_config.cur_step = trainer_state['cur_step']
            optimizer_config.gradient_accumulation_steps = trainer_state['gradient_accumulation_steps']

        return {
            'cur_step': trainer_state['cur_step'],
            'consumed_train_samples': trainer_state['consumed_train_samples'],
            'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
        }

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        return self.multi_adapter.get_state_dict(**kwargs)

    @remote_function(dispatch='all', sync=True)
    def add_adapter_to_model(
        self,
        adapter_name: str,
        config_or_dir: Union[Dict[str, Any], LoraConfig, str],
        **kwargs,
    ):
        # prevent opening requires_grad of the base model
        # prevent loading malicious code
        assert not isinstance(
            config_or_dir, str
        ), 'config_or_dir does not support str, because loading config from modelhub may causing unexpected behavior'
        assert isinstance(config_or_dir, LoraConfig), 'config_or_dir must be a LoraConfig instance'
        # Limit the max peft version in pyproject.toml, in case any newer version opens some untested module grad.
        config_or_dir.modules_to_save = None
        config_or_dir.bias = 'none'
        config_or_dir.init_lora_weights = False
        config_or_dir.modules_to_save = None
        config_or_dir.trainable_token_indices = None
        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config_or_dir
        self.optimizer_group[adapter_name].gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        self.multi_adapter.acquire_lora(tenant_adapter_name=adapter_name, config=config_or_dir)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_template(template_cls, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, Callable], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_processor(processor_cls, **kwargs)

    @remote_function()
    def add_metric(self, metric_cls: Union[Metric, str], is_training: Optional[bool] = None, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().add_metric(metric_cls, is_training, **kwargs)

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if adapter_name in self.optimizer_group:
            self.optimizer_group.pop(adapter_name)
        self.multi_adapter.release_lora(adapter_name)
