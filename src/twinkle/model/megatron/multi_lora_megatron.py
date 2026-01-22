# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch.nn as nn
from peft import LoraConfig
from peft import PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PretrainedConfig, AutoConfig

from twinkle import DeviceMesh, remote_class, remote_function, template
from twinkle import requires
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss
from twinkle.processor import InputProcessor
from .megatron import MegatronModel, MegatronOptimizerGroup
from .strategy import MegatronStrategy
from twinkle.metric import Metric


@remote_class(execute='all')
class MultiLoraMegatronModel(MegatronModel):

    DUMMY_ADAPTER_NAME = '__dummy_adapter__'

    def __init__(self,
                 model_id: str,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
                 load_weights: bool = True,
                 recompute_granularity: Optional[str] = 'selective',  # Activation checkpointing
                 recompute_modules: Optional[list] = None,  # Modules to recompute
                 **kwargs,
                 ):
        requires('megatron_core')
        nn.Module.__init__(self)
        from twinkle.patch.megatron_peft import MegatronPeft

        self.model_id = model_id
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.recompute_granularity = recompute_granularity
        self.recompute_modules = recompute_modules
        model_path = HubOperation.download_model(model_id)
        if config is None:
            # Load HuggingFace config first
            self.hf_config = AutoConfig.from_pretrained(model_path)
        else:
            self.hf_config = config
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        # Store model_path for later use
        self._model_path = model_path

        self._seed = kwargs.pop('seed', None)
        if self._seed is None and os.environ.get('TWINKLE_SEED'):
            self._seed = int(os.environ.get('TWINKLE_SEED'))
        if self._seed is None:
            self._seed = 42
        self._default_tokenizer = None
        self.use_distributed_optimizer = kwargs.get('use_distributed_optimizer', True)
        # Create Megatron strategy
        self.strategy = MegatronStrategy(self.device_mesh, mixed_precision=mixed_precision, **kwargs)

        self.model = self._create_megatron_model(model_path, load_weights, **kwargs)

        self._model_wrapped = False
        # This correctly handles vocab sharding in Tensor Parallelism
        self.optimizer_group: Dict[str, MegatronOptimizerGroup] = self._construct_megatron_optimizer_group()
        MegatronPeft().patch()
        self._inited = False
        self.model = self.strategy.wrap_model(self.model)
        self.add_adapter_to_model(MultiLoraMegatronModel.DUMMY_ADAPTER_NAME, LoraConfig(r=1, target_modules='all-linear'))
        self._inited = True

    def _check_adapter_valid(self, adapter_name: str):
        if self._inited:
            assert adapter_name and adapter_name != MultiLoraMegatronModel.DUMMY_ADAPTER_NAME and adapter_name in self.optimizer_group, f'Use a valid adapter_name first, current is: {adapter_name}'

    def _activate_adapter(self, adapter_name: str):
        self.multi_adapter.set_current_adapter_name(adapter_name)

    def _lazy_wrap_model(self):
        pass

    @remote_function()
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
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().forward_only(inputs=inputs, **kwargs)

    @remote_function(dispatch='all', collect='mean', sync=True)
    def forward_backward(self,
                         *,
                         inputs: Union[InputFeature, List[InputFeature],
                                       Trajectory, List[Trajectory]],
                         num_microbatches: int = 1,
                         **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().forward_backward(inputs=inputs, num_microbatches=num_microbatches, **kwargs)

    @remote_function(dispatch='all')
    def clip_grad_norm(self,
                       max_grad_norm: float = 1.0,
                       norm_type: int = 2,
                       **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().clip_grad_norm(max_grad_norm=max_grad_norm, norm_type=norm_type, **kwargs)

    @remote_function(dispatch='all')
    def step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().step(**kwargs)

    @remote_function(dispatch='all')
    def zero_grad(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().lr_step(**kwargs)

    @remote_function(dispatch='all')
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().set_loss(loss_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str],
                      **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().set_optimizer(optimizer_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str],
                         **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().set_lr_scheduler(scheduler_cls, **kwargs)

    @remote_function(dispatch='all', sync=True)
    def save(self, output_dir: str, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().save(output_dir, **kwargs)

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return self.get_state_dict(**kwargs)

    def _prepare_adapter(self, adapter_name: str):
        self._check_adapter_valid(adapter_name)
        pattern = re.compile(r'\.lora_\w+\.[^.]+\.')
        unwrapped_model = self.strategy.unwrap_model(self.model)
        for name, param in unwrapped_model.named_parameters():
            if pattern.search(name):
                param.requires_grad = True

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
        self._patch_adapter(adapter_name, config_or_dir, adapter_name, **kwargs)
        self._activate_adapter(adapter_name)
        self._prepare_adapter(adapter_name)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().set_template(template_cls, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().set_processor(processor_cls, **kwargs)

    def add_metric(self, metric_cls: Union[Metric, str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().add_metric(metric_cls, **kwargs)

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if adapter_name in self.optimizer_group:
            self.optimizer_group.pop(adapter_name)
        model = self.strategy.unwrap_model(self.model)
        if isinstance(model, PeftModel):
            model.base_model.delete_adapter(adapter_name=adapter_name)
