# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import Any, Dict, List, Literal, Optional, Type, Union, Callable

import torch
import torch.nn as nn
from peft import LoraConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PretrainedConfig, AutoConfig
import torch.distributed as dist
from twinkle import DeviceMesh, remote_class, remote_function, template
from twinkle import requires
from twinkle import torch_util
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from .args import TwinkleMegatronArgs, set_args
from .megatron import MegatronModel
from .strategy import MegatronStrategy
from ..multi_lora import MultiLora


@remote_class(execute='all')
class MultiLoraMegatronModel(MegatronModel):

    def __init__(self,
                 model_id: str,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
                 load_weights: bool = True,
                 recompute_granularity: Optional[str] = 'selective',  # Activation checkpointing
                 recompute_modules: Optional[list] = None,  # Modules to recompute
                 sequence_parallel: bool = False,
                 max_loras:int = 5,
                 max_r:int = 32,
                 max_length: int = 8192,
                 **kwargs,
                 ):
        requires('megatron_core')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        nn.Module.__init__(self)
        from twinkle.patch.megatron_peft import MegatronPeft

        self.model_id = model_id
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision

        self._model_path = HubOperation.download_model(model_id)
        self.hf_config = config or AutoConfig.from_pretrained(self._model_path)
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)

        self._seed = kwargs.pop('seed', None) or int(os.environ.get('TWINKLE_SEED', 42))
        self._default_tokenizer = None
        self.use_distributed_optimizer = kwargs.get('use_distributed_optimizer', True)
        self.variable_seq_lengths = kwargs.get('variable_seq_lengths', True)
        self.optimizer_group = {}
        torch_util.set_device()

        self.strategy = MegatronStrategy(self.device_mesh, sequence_parallel=sequence_parallel, mixed_precision=mixed_precision, **kwargs)

        # Determine params_dtype and activation checkpointing kwargs
        params_dtype = torch.bfloat16
        if self.mixed_precision == 'fp16':
            params_dtype = torch.float16
        elif self.mixed_precision == 'no':
            params_dtype = torch.float32

        ac_kwargs = {
            'recompute_granularity': recompute_granularity,
            'recompute_modules': recompute_modules,
        }
        if kwargs.get('recompute_method'):
            ac_kwargs['recompute_method'] = kwargs.get('recompute_method')
        if kwargs.get('recompute_num_layers'):
            ac_kwargs['recompute_num_layers'] = kwargs.get('recompute_num_layers')

        # Initialize TwinkleMegatronArgs BEFORE creating the model
        args = TwinkleMegatronArgs.from_hf_config(
            self.hf_config,
            model_dir=self._model_path,
            device_mesh=self.device_mesh,
            params_dtype=params_dtype,
            sequence_parallel=self.strategy.sequence_parallel,
            **ac_kwargs,
        )
        set_args(args)
        self._initialized = False
        self.model: List[nn.Module] = self._create_megatron_model(load_weights, **kwargs)

        MegatronPeft().patch()
        self.multi_adapter = MultiLora(max_loras=max_loras, max_r=max_r, max_length=max_length)
        self.model = self.multi_adapter.patch(self.model)
        self.model = self.strategy.wrap_model(self.model)
        self._model_wrapped = True
        self.multi_adapter.save_initial_weights()

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name and adapter_name in self.optimizer_group, f'Use a valid adapter_name first, current is: {adapter_name}'

    def _lazy_wrap_model(self):
        pass

    @remote_function(dispatch='slice_dp', collect='last_pp', sync=True)
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature],
                                            List[Trajectory]], **kwargs):
        """Forward pass without gradient computation.

        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments.

        Returns:
            Model outputs.
        """
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().forward_only(inputs=inputs, **kwargs)

    @remote_function(dispatch='slice_dp', collect='mean', sync=True)
    def forward_backward(self,
                         *,
                         inputs: Union[InputFeature, List[InputFeature],
                                       Trajectory, List[Trajectory]],
                         num_microbatches: int = 1,
                         **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().forward_backward(inputs=inputs, num_microbatches=num_microbatches, **kwargs)

    @remote_function(dispatch='all')
    def step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().step(**kwargs)

    @remote_function(dispatch='all')
    def zero_grad(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().lr_step(**kwargs)

    @remote_function(dispatch='all')
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        return super().set_loss(loss_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str],
                      **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().set_optimizer(optimizer_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str],
                         **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        return super().set_lr_scheduler(scheduler_cls, **kwargs)

    @remote_function(dispatch='all', sync=True)
    def save(self, name, output_dir: Optional[str] = None, interval=1, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        optimizer_config = self.optimizer_group[kwargs.get("adapter_name")]
        if optimizer_config.cur_step % interval != 0:
            return

        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)

        with self.multi_adapter.save_context(kwargs.get("adapter_name")) as real_adapter_name:
            save_format = kwargs.pop('save_format', 'hf')  # 'hf' or 'megatron'
            if save_format == 'hf':
                self._save_hf_format(checkpoint_dir, real_adapter_name, lora_converter=self.multi_adapter.save_lora_converter)
            else:
                self._save_megatron_format(checkpoint_dir, real_adapter_name, lora_converter=self.multi_adapter.save_lora_converter)

            self._save_tokenizer(checkpoint_dir, adapter_name=kwargs.get("adapter_name"))
            # Final synchronization to ensure all ranks complete save
            if dist.is_initialized():
                dist.barrier()

    @remote_function(dispatch='all')
    def load(self, name: Optional[str], output_dir: Optional[str] = None, **kwargs):
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        bridge = self._bridge
        with self.multi_adapter.save_context(kwargs.get("adapter_name")) as adapter_name:
            for _model in self.strategy.unwrap_model(self.model):
                bridge.load_weights(_model, checkpoint_dir, True, adapter_name=adapter_name, lora_converter=self.multi_adapter.load_lora_converter)

        if dist.is_initialized():
            dist.barrier()

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
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
        assert not isinstance(config_or_dir,
                              str), 'config_or_dir does not support str, because loading config from modelhub may causing unexpected behavior'
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
        self.optimizer_group[
            adapter_name].gradient_accumulation_steps = kwargs.get(
            'gradient_accumulation_steps', 1)
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        self.multi_adapter.acquire_lora(tenant_adapter_name=adapter_name, config=config_or_dir)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().set_template(template_cls, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, Callable], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().set_processor(processor_cls, **kwargs)

    def add_metric(self, metric_cls: Union[Metric, str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().add_metric(metric_cls, **kwargs)

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if adapter_name in self.optimizer_group:
            self.optimizer_group.pop(adapter_name)
        self.multi_adapter.release_lora(adapter_name)
