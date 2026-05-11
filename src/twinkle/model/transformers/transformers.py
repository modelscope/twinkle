# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import contextlib
import json
import numpy as np
import os
import random
import re
import threading
import torch
import torch.distributed as dist
import transformers
from copy import copy
from dataclasses import dataclass, field
from peft import PeftConfig, PeftModel, get_peft_model
from peft.utils import load_peft_weights, set_peft_model_state_dict
from safetensors.torch import save_file
from torch import GradScaler
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, overload

import twinkle
import twinkle.module.scheduler
from twinkle import DeviceMesh, Platform, remote_class, remote_function
from twinkle.checkpoint_engine import CheckpointEngine
from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin
from twinkle.data_format import InputFeature, ModelOutput, Trajectory
from twinkle.hub import HubOperation
from twinkle.infra import collect_tensor_dict
from twinkle.loss import CrossEntropyLoss, Loss
from twinkle.metric import Accuracy, LossMetric, Metric, TrainMetric
from twinkle.model.base import TwinkleModel
from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.model.transformers.moe import apply_expert_parallel
from twinkle.model.transformers.strategy import AccelerateStrategy, NativeFSDPStrategy
from twinkle.patch import Patch, apply_patch
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import construct_class, get_logger, selective_log_softmax, torch_util
from twinkle.utils.framework import Torch
from twinkle.utils.grad_clip import normalize_and_clip_grad_norm

logger = get_logger()


@dataclass
class OptimizerGroup(BaseOptimizerGroup):
    """Optimizer group for Transformers training."""
    adapter_config: PeftConfig = None
    loss_instance: Loss = CrossEntropyLoss
    scaler: GradScaler = None
    scaler_has_nan: bool = False
    checkpoint_engine: CheckpointEngine = None
    _handler: Any = None

    def __post_init__(self):
        self._ensure_dp_group()
        self._build_metrics()

    def _build_metrics(self):
        train_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            Accuracy(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]
        self.train_status = TrainStatus(metrics=train_metrics)

        eval_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            Accuracy(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]
        self.eval_status = TrainStatus(metrics=eval_metrics)

    def _ensure_dp_group(self):
        if self._dp_group is not None or self._device_mesh is None:
            return
        raw_world_size = self._device_mesh._get_dp_fsdp_world_size()
        if raw_world_size <= 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        if dist.get_world_size() < raw_world_size:
            # World size is smaller than the requested dp/fsdp group; skip to avoid crash.
            return
        dims = [dim for dim in ('dp', 'fsdp') if self._device_mesh.has_dim(dim)]
        if not dims:
            return
        self._dp_group = self._device_mesh.create_process_group(dims)

    def _get_lr(self):
        if self.optimizer is not None:
            _lrs = []
            _default_lr = self.optimizer.defaults.get('lr')
            for param_group in self.optimizer.param_groups:
                _lrs.append(param_group.get('lr', _default_lr))
            return _lrs
        else:
            return []

    def accumulate_metrics(self, is_training):
        self._ensure_dp_group()
        status = self.train_status if is_training else self.eval_status
        if len(status.metrics) > 0 and status.inputs is not None and status.outputs is not None:
            for metric in status.metrics:
                metric.accumulate(
                    status.inputs,
                    status.outputs,
                    lr=self._get_lr(),
                    step=self.cur_step - 1,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    grad_norm=self._last_grad_norm,
                    loss_reduction=getattr(self.loss_instance, 'reduction', 'mean'),
                    **status.forward_kwargs)


_default_adapter_name = ''
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01


@remote_class()
class TransformersModel(TwinkleModel, PreTrainedModel, CheckpointEngineMixin):
    """The transformers model wrapper.

    Args:
        model_cls: The PreTrainedModel model class, only needed when creating a blank(not pretrained) model.
        config: The config of the model.
        model_id: The model id or path, this argument will be used in `from_pretrained`.
        device_mesh: The model device mesh to follow.
        mixed_precision: The mixed precision type.
        strategy: The training strategy to use.
        ddp_config: The DDP config to use.
        fsdp_config: The fsdp config to use.
        grad_scaler_config: The gradient scaler config to use.
        kwargs: Any kwargs used in `from_pretrained` or `__init__`.

    If model_id is passed in, `from_pretrained` will be used, else `__init__` will be used.
    """

    @overload
    def __init__(self, *, model_cls: Type[PreTrainedModel], config: PretrainedConfig, remote_group, **kwargs) -> None:
        ...

    @overload
    def __init__(self, *, model_id: str, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        ...

    def __init__(
            self,  # noqa
            model_cls: Optional[Union[Type[PreTrainedModel], str, Type[_BaseAutoModelClass]]] = None,
            model_id: Optional[str] = None,
            config: Optional[PretrainedConfig] = None,
            device_mesh: Optional[DeviceMesh] = None,
            mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
            strategy: Literal['accelerate', 'native_fsdp'] = 'accelerate',
            ddp_config: Dict[str, Any] = None,
            fsdp_config: Dict[str, Any] = None,
            grad_scaler_config: Dict[str, Any] = None,
            memory_efficient_init: bool = False,
            **kwargs):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._try_init_process_group()
        super(PreTrainedModel, self).__init__()
        # The Default tokenizer will be used to save with a model if no template was set.
        self._default_tokenizer = None
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._fsdp_config = dict(fsdp_config or {})
        self._ddp_config = ddp_config or {}
        self._memory_efficient_init = memory_efficient_init
        self._decide_strategy(strategy)
        self.grad_scaler_config = grad_scaler_config
        if model_id is not None:
            model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        if config is None:
            from transformers import AutoConfig
            self.hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        else:
            self.hf_config = config
        if model_cls is None and hasattr(self.hf_config, 'architectures'):
            model_cls = self.hf_config.architectures[0]
        if model_cls is None:
            model_cls = AutoModelForCausalLM
        if isinstance(model_cls, str):
            model_cls = getattr(transformers, model_cls)
        if model_id is None:
            self.model = model_cls.from_config(self.hf_config, **kwargs)
        else:
            # Trigger transformers' FSDP-aware loading: meta-device init + rank-0-only weight load.
            with self.strategy.pretrained_load_context():
                self.model = model_cls.from_pretrained(model_id, config=self.hf_config, **kwargs)
        self.model.gradient_checkpointing_enable()
        self.sp_strategy = None
        self._model_wrapped = False
        self.optimizer_group: Dict[str, OptimizerGroup] = {
            _default_adapter_name: self._construct_default_optimizer_group()
        }
        self.optimizer_group[_default_adapter_name].adapter_name = _default_adapter_name
        self.active_group = _default_adapter_name

    def _decide_strategy(self, strategy: Literal['accelerate', 'native_fsdp']):
        self._expert_parallel_config = self._fsdp_config.pop('expert_parallel', None)
        self._enable_expert_parallel = self._should_enable_expert_parallel(self._expert_parallel_config,
                                                                           self.device_mesh)
        self._expert_parallel_applied = False

        use_native_fsdp = self._enable_expert_parallel or strategy == 'native_fsdp'
        if use_native_fsdp:
            ep_size = (self._expert_parallel_config.get('ep_size') if self._expert_parallel_config else None)
            if ep_size is None and self.device_mesh is not None:
                ep_size = getattr(self.device_mesh, 'ep_size', None)
            self.strategy = NativeFSDPStrategy(
                mixed_precision=self.mixed_precision,
                fsdp_config=self._fsdp_config,
                device_mesh=self.device_mesh,
                memory_efficient_init=self._memory_efficient_init,
                enable_ep=self._enable_expert_parallel,
                ep_size=ep_size,
            )
        else:
            self.strategy = AccelerateStrategy(
                mixed_precision=self.mixed_precision,
                ddp_config=self._ddp_config,
                fsdp_config=self._fsdp_config,
                device_mesh=self.device_mesh,
                memory_efficient_init=self._memory_efficient_init)

        # Sequence parallel ("ulysses") is derived from dp/fsdp ranks; it does not change world size.
        # We construct `sp_strategy` after the underlying HF model is initialized (see __init__).
        self._enable_sp = False
        if self.device_mesh is not None:
            sp_size = getattr(self.device_mesh, 'ulysses_size', None)
            self._enable_sp = bool(sp_size and sp_size > 1)

    def _ensure_sp_strategy(self) -> None:
        if not getattr(self, '_enable_sp', False):
            return
        if self.sp_strategy is not None:
            return
        from .strategy.sequence_parallel import SequenceParallelStrategy

        self.sp_strategy = SequenceParallelStrategy(
            self.device_mesh,
            {},
            model=self.model,
            tokenizer_id=self.tokenizer_id,
        )

    def _get_default_group(self):
        """Get the only group, else return the default one"""
        if len(self.optimizer_group) == 1:
            return next(iter(self.optimizer_group))
        return self.active_group

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return 'input_ids' not in inputs and 'input_embedding' not in inputs

    def _lazy_wrap_model(self):
        if not self._model_wrapped:
            optimizer_groups = [og for og in self.optimizer_group.values() if og.optimizer is not None]
            self._maybe_apply_expert_parallel()
            self._ensure_sp_strategy()
            if self.sp_strategy is not None:
                self.sp_strategy.initialize()

            if len(optimizer_groups) == 1:
                optimizer_group = optimizer_groups[0]
                optimizer = optimizer_group.optimizer
                assert optimizer is not None
                self.model, optimizer = self.strategy.wrap_model(self.model, optimizer)
                optimizer_group.optimizer = optimizer
                self.register_mm_forward_hook(optimizer_group)
            else:
                # maybe forward_only, no optimizer_group available
                result = self.strategy.wrap_model(self.model)
                if isinstance(result, tuple):
                    self.model = result[0]
                else:
                    self.model = result
            self._model_wrapped = True

    def register_mm_forward_hook(self, optimizer_group: OptimizerGroup):
        model = self.strategy.unwrap_model(self.model)
        template = optimizer_group.template
        assert template is not None
        optimizer_group._handler = model.register_forward_pre_hook(template.pre_forward_hook, with_kwargs=True)

    def unregister_mm_forward_hook(self, optimizer_group: OptimizerGroup):
        if optimizer_group._handler is not None:
            optimizer_group._handler.remove()
            optimizer_group._handler = None

    @staticmethod
    def _should_enable_expert_parallel(expert_parallel_config: Optional[Dict[str, Any]],
                                       device_mesh: Optional[DeviceMesh]) -> bool:
        if expert_parallel_config is None or device_mesh is None:
            return False
        # Check ep_size from config first, then from device_mesh.ep_size attribute
        ep_size = expert_parallel_config.get('ep_size') or getattr(device_mesh, 'ep_size', None) or 1
        if ep_size <= 1:
            return False
        return expert_parallel_config.get('enabled', True)

    def _maybe_apply_expert_parallel(self):
        if not self._enable_expert_parallel or self._expert_parallel_applied:
            return
        self._ensure_optimizer_dp_groups()
        model = self.strategy.unwrap_model(self.model)
        ep_fsdp_mesh = getattr(self.strategy, 'ep_fsdp_device_mesh', None)
        apply_expert_parallel(
            model,
            self.device_mesh,
            config=self._expert_parallel_config,
            ep_fsdp_device_mesh=ep_fsdp_mesh,
        )
        self._expert_parallel_applied = True

    def _ensure_optimizer_dp_groups(self):
        for optimizer_group in self.optimizer_group.values():
            if not isinstance(optimizer_group, OptimizerGroup):
                continue
            before = optimizer_group._dp_group
            optimizer_group._ensure_dp_group()
            if before is None and optimizer_group._dp_group is not None:
                optimizer_group._build_metrics()

    def _construct_default_optimizer_group(self):
        return OptimizerGroup(
            loss_instance=CrossEntropyLoss(reduction='sum'),
            template=Template(self.tokenizer_id),
            processor=InputProcessor(self.device_mesh),
            _device_mesh=self.device_mesh,
        )

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        """Call forward function and record the inputs and outputs.

        Args:
            inputs: The model inputs. Can be an encoded batch, or a list of `Trajectory`
            **kwargs:
                adapter_name: Lora adapter name.
        Returns:
            The output of the model forward.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        temperature = float(kwargs.pop('temperature', 1.0))
        return_logits = kwargs.pop('return_logits', False)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        self.model.train()
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        processor: InputProcessor = optimizer_config.processor
        loss_instance = optimizer_config.loss_instance
        loss_require_logits = (hasattr(loss_instance, 'require_logits') and loss_instance.require_logits)
        assert isinstance(processor, InputProcessor), 'Set a correct `InputProcessor` before forwarding'
        inputs: Dict[str, Any] = processor(
            inputs,
            sp_strategy=self.sp_strategy,
            model=self.model,
            hf_config=self.hf_config,
            enable_sp=getattr(self, '_enable_sp', False),
        )
        labels: torch.Tensor = inputs.pop('labels', None)
        optimizer_config.accumulate_metrics(True)
        outputs = self.model(**inputs)
        inputs['labels'] = labels
        if labels is not None:
            loss_mask = (labels != -100).bool()
            masked_labels = labels.clone()
            masked_labels[~loss_mask] = 0
            logits = outputs['logits']
            logits.div_(temperature)
            outputs['logps'] = selective_log_softmax(logits, masked_labels)
            del logits
        outputs['past_key_values'] = None
        if not (return_logits or loss_require_logits):
            outputs['logits'] = None
        inputs, outputs = processor.postprocess_tensor_sp(inputs, outputs, sp_strategy=self.sp_strategy)
        inputs, outputs = processor.unpack_packed_sequences(inputs, outputs)
        optimizer_config.train_status.inputs = inputs
        optimizer_config.train_status.outputs = outputs
        optimizer_config.train_status.forward_kwargs = kwargs
        optimizer_config.train_status.loss_value = outputs.get('aux_loss', 0)
        return_outputs = copy(outputs)
        if not return_logits:
            return_outputs['logits'] = None
        return return_outputs

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        """Call forward function without grad and record the inputs and outputs.

        Args:
            inputs: The model inputs. Can be an encoded batch, or a list of `Trajectory`
            **kwargs:
                adapter_name: Lora adapter name.
                disable_lora: If True, disable LoRA and use base model for inference.
        Returns:
            The output of the model forward.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        disable_lora = kwargs.pop('disable_lora', False)
        temperature = float(kwargs.pop('temperature', 1.0))
        return_logits = kwargs.pop('return_logits', False)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        self.model.eval()
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        with torch.no_grad():
            processor: InputProcessor = optimizer_config.processor
            assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
            loss_instance = optimizer_config.loss_instance
            loss_require_logits = (hasattr(loss_instance, 'require_logits') and loss_instance.require_logits)
            inputs: Dict[str, Any] = processor(
                inputs,
                sp_strategy=self.sp_strategy,
                model=self.model,
                hf_config=self.hf_config,
                enable_sp=getattr(self, '_enable_sp', False),
            )
            labels = inputs.pop('labels', None)
            optimizer_config.accumulate_metrics(False)
            unwrapped_model = self.strategy.unwrap_model(self.model)
            if disable_lora and isinstance(unwrapped_model, PeftModel):
                with unwrapped_model.disable_adapter():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            inputs['labels'] = labels
            if labels is not None:
                loss_mask = (labels != -100).bool()
                masked_labels = labels.clone()
                masked_labels[~loss_mask] = 0
                logits = outputs['logits']
                logits.div_(temperature)
                outputs['logps'] = selective_log_softmax(logits, masked_labels)
                del logits
            outputs['past_key_values'] = None
            if not (return_logits or loss_require_logits):
                outputs['logits'] = None
            inputs, outputs = processor.postprocess_tensor_sp(inputs, outputs, sp_strategy=self.sp_strategy)
            inputs, outputs = processor.unpack_packed_sequences(inputs, outputs)
            optimizer_config.eval_status.inputs = inputs
            optimizer_config.eval_status.outputs = outputs
            optimizer_config.eval_status.forward_kwargs = kwargs
            optimizer_config.eval_status.loss_value = outputs.get('aux_loss', 0)
            return_outputs = copy(outputs)
            if not return_logits:
                return_outputs['logits'] = None
            return return_outputs

    @remote_function(collect='mean')
    def calculate_loss(self, **kwargs):
        """Calculate loss

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for the specific loss type.
        Returns:
            A scalar loss value.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance: Loss = optimizer_config.loss_instance
        assert isinstance(loss_instance, Loss), 'Set a loss_instance before calculating loss'
        if self.model.training:
            status = optimizer_config.train_status
        else:
            status = optimizer_config.eval_status
        inputs = status.inputs
        outputs = status.outputs
        assert inputs is not None and outputs is not None, 'Cannot calculate loss of empty inputs and outputs'
        result = loss_instance(inputs, outputs, **kwargs)
        loss_value = result['loss']
        raw_counts = result['num_tokens']
        counts = raw_counts
        if not counts:
            counts = torch.tensor(1, device=loss_value.device)
        # Later will gather this value, so it becomes:
        # 1. SUM loss: gather_sum(local_num_tokens / dp_world_size) = global_num_tokens / dp_world_size
        # 2. PER TOKEN MEAN loss: gather_sum(1 * gradient_accumulation_steps / dp_world_size )
        #   = gradient_accumulation_steps
        # Then, grad will divided by this value:
        # 1. SUM loss: gather_mean(local_sum_grad) / (global_num_tokens / dp_world_size)
        #              = (global_sum_grad / dp_world_size) / (global_num_tokens / dp_world_size)
        #              = global_sum_grad/global_num_tokens
        # 2. PER TOKEN MEAN loss: gather_mean(per_token_grad * gradient_accumulation_steps)
        #                               / gradient_accumulation_steps
        #                         = (global_per_token_grad * gradient_accumulation_steps / dp_world_size )
        #                               / gradient_accumulation_steps
        #                         = global_per_token_grad / dp_world_size = avg_per_token_grad
        raw_dp_fsdp_world_size = self.device_mesh._get_dp_fsdp_world_size() if self.device_mesh is not None else 1
        counts = counts / raw_dp_fsdp_world_size
        optimizer_config = self.optimizer_group[adapter_name]
        status.num_tokens += counts.item()
        status.loss_value += loss_value
        outputs['loss'] = status.loss_value
        outputs['num_tokens'] = raw_counts.detach() if hasattr(raw_counts, 'detach') else raw_counts
        return status.loss_value.item()

    @remote_function()
    def backward(self, **kwargs):
        """Backward propagation.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                gradient_accumulation_steps: Number of gradient accumulation steps.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        loss_value = optimizer_config.train_status.loss_value
        assert loss_value is not None, 'Do forwarding and calculating loss before backward'
        scaler = optimizer_config.scaler
        if scaler is None and self.mixed_precision == 'fp16':
            # Auto set a grad scaler
            self.set_grad_scaler(adapter_name=adapter_name)
            scaler = optimizer_config.scaler

        optimizer_config.cur_step += 1
        should_sync = optimizer_config.do_grad_sync()

        import contextlib
        no_sync_ctx = contextlib.nullcontext()
        if not should_sync and hasattr(self.model, 'no_sync'):
            no_sync_ctx = self.model.no_sync()

        with no_sync_ctx:
            if scaler is not None:
                scaler.scale(loss_value).backward()
            else:
                loss_value.backward()

        optimizer_config.train_status.loss_value = None

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        """Do forward, calculate loss, and backward.

        Args:
            inputs: The model inputs. Can be an encoded batch, or a list of `Trajectory`
            **kwargs:
                adapter_name: Lora adapter name.
                gradient_accumulation_steps: Number of gradient accumulation steps.
                Any parameters needed for the specific loss type.
        Returns:
            The output of the model forward.
        """
        outputs = self.forward(inputs=inputs, **kwargs)
        loss = self.calculate_loss(**kwargs)
        outputs['loss'] = loss
        self.backward(**kwargs)
        return outputs

    @remote_function()
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        """ Clip the gradient norm

        Args:
            max_grad_norm: The maximum grad norm, default `1.0`.
            norm_type: Default `2`.
            **kwargs:
                adapter_name: Lora adapter name.
        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return

        optimizer = optimizer_config.optimizer
        scaler = optimizer_config.scaler
        context = contextlib.nullcontext
        if self.device_mesh is not None and self.device_mesh.tp_world_size > 1:
            from torch.distributed.tensor.experimental import implicit_replication
            context = implicit_replication

        with context():
            if scaler is not None:
                scaler.unscale_(optimizer)

            optimizer_config._ensure_dp_group()
            num_tokens = optimizer_config.train_status.num_tokens
            num_tokens = torch_util.gather_object([num_tokens], self.device_mesh, optimizer_config._dp_group)
            num_tokens = sum(num_tokens)
            parameters = list(self._get_trainable_parameters(adapter_name).values())

            ep_clip_kwargs = self.strategy.get_ep_clip_kwargs(self.model) if hasattr(
                self.strategy, 'get_ep_clip_kwargs') else {}

            grad_norm = normalize_and_clip_grad_norm(
                parameters,
                num_tokens=num_tokens,
                max_grad_norm=max_grad_norm,
                norm_type=norm_type,
                group=optimizer_config._dp_group,
                **ep_clip_kwargs,
            )
            optimizer_config._last_grad_norm = grad_norm
            optimizer_config.train_status.num_tokens = 0
            return grad_norm

    @remote_function(dispatch='all')
    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        self.clip_grad_norm(max_grad_norm, norm_type, **kwargs)
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)

    def _create_param_group(self,
                            adapter_name: str,
                            lr: float = DEFAULT_LEARNING_RATE,
                            weight_decay: float = DEFAULT_WEIGHT_DECAY,
                            **kwargs):
        # Some code borrowed from transformers

        def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
            forbidden_layer_patterns = ([re.compile(pattern) for pattern in forbidden_layer_names]
                                        if forbidden_layer_names is not None else [])
            result = []
            for name, child in model.named_children():
                child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
                result += [
                    f'{name}.{n}' for n in child_params
                    if not isinstance(child, tuple(forbidden_layer_types)) and not any(
                        pattern.search(f'{name}.{n}'.lower()) for pattern in forbidden_layer_patterns)
                ]
            # Add model specific parameters that are not in any child
            result += [
                k for k in model._parameters
                if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
            ]

            return result

        forbidden_name_patterns = [r'bias', r'layernorm', r'rmsnorm', r'(?:^|\.)norm(?:$|\.)', r'_norm(?:$|\.)']
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm], forbidden_name_patterns)
        params = self._get_trainable_parameters(adapter_name)
        decay_param_names = [n for n, p in params.items() if (n in decay_parameters and p.requires_grad)]
        no_decay_param_names = [n for n, p in params.items() if (n not in decay_parameters and p.requires_grad)]
        optimizer_grouped_parameters = [
            {
                'params': [params[n] for n in decay_param_names],
                'param_names': decay_param_names,
                'weight_decay': weight_decay,
                'lr': lr
            },
            {
                'params': [params[n] for n in no_decay_param_names],
                'param_names': no_decay_param_names,
                'weight_decay': 0.0,
                'lr': lr
            },
        ]
        return optimizer_grouped_parameters

    @remote_function()
    def step(self, **kwargs):
        """Optimizer step.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for `optimizer.step`.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        grad_accum_steps = kwargs.pop('gradient_accumulation_steps', None)
        if not optimizer_config.do_grad_sync(grad_accum_steps):
            return
        optimizer = optimizer_config.optimizer
        scaler = optimizer_config.scaler
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'

        context = contextlib.nullcontext
        if self.device_mesh is not None and self.device_mesh.tp_world_size > 1:
            from torch.distributed.tensor.experimental import implicit_replication
            context = implicit_replication

        optim_params = kwargs.pop('optim_params', {})
        if optim_params:
            assert isinstance(optimizer, (AdamW, Adam))
            for group in optimizer.param_groups:
                group['lr'] = optim_params['lr']
                if group['weight_decay'] > 0.0 and optim_params.get('weight_decay', None) is not None:
                    group['weight_decay'] = optim_params['weight_decay']
                if optim_params.get('eps') is not None:
                    group['eps'] = optim_params['eps']
                if optim_params.get('betas') is not None:
                    group['betas'] = optim_params['betas']

        with context():
            if scaler is not None:
                scaler.step(optimizer, **kwargs)
                scaler.update()
                optimizer_config.scaler_has_nan = sum(v.item()
                                                      for v in scaler._found_inf_per_device(optimizer).values()) > 0
            else:
                optimizer.step(**kwargs)

    @remote_function()
    def zero_grad(self, **kwargs):
        """Optimizer zero_grad.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for `optimizer.zero_grad`.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'
        optimizer.zero_grad(set_to_none=True)

    @remote_function()
    def lr_step(self, **kwargs):
        """Do lr_scheduler step.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for `lr_scheduler.step`.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return
        if optimizer_config.scaler_has_nan:
            return
        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...], torch.Tensor]],
                 **kwargs):
        """Set the loss instance.

        Args:
            loss_cls: A loss class name, a loss plugin id, or a loss class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the loss instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.loss_instance = construct_class(loss_cls, Loss, twinkle.loss, **kwargs)

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str, Optimizer], **kwargs):
        """Set the optimizer.

        Args:
            optimizer_cls: An optimizer class name, an optimizer plugin id, or an optimizer class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                lr: Learning rate
                weight_decay: Weight decay
                Any parameters needed to construct the optimizer instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if isinstance(optimizer_cls, Optimizer):
            optimizer_config.optimizer = optimizer_cls
            return

        params = kwargs.pop('params', None)
        if params is None:
            lr = kwargs.get('lr', DEFAULT_LEARNING_RATE)
            weight_decay = kwargs.get('weight_decay', DEFAULT_WEIGHT_DECAY)
            params = self._create_param_group(adapter_name, lr=lr, weight_decay=weight_decay)
        if hasattr(self.strategy, 'adjust_optimizer_kwargs'):
            kwargs = self.strategy.adjust_optimizer_kwargs(optimizer_cls, kwargs)
        optimizer_config.optimizer = construct_class(
            optimizer_cls,
            Optimizer,
            torch.optim,
            params=params,
            **kwargs,
        )

    def _get_trainable_parameters(self, adapter_name=_default_adapter_name):
        is_default = adapter_name == _default_adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        params = {}
        model = self.strategy.unwrap_model(self.model)
        for name, param in model.named_parameters():
            if param.requires_grad and (pattern.search(name) or is_default):
                params[name] = param
        return params

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str, LRScheduler], **kwargs):
        """Set the lr_scheduler.

        Args:
            scheduler_cls: An lr_scheduler class name, an lr_scheduler plugin id, or an lr_scheduler class type.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the lr_scheduler instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before setting lr_scheduler'
        kwargs['optimizer'] = optimizer
        scheduler = construct_class(scheduler_cls, LRScheduler, [torch.optim.lr_scheduler, twinkle.module.scheduler],
                                    **kwargs)
        optimizer_config.lr_scheduler = scheduler

    @remote_function()
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs):
        apply_patch(self.model, patch_cls, **kwargs)

    def __del__(self):
        HubOperation.wait_for()

    @remote_function(collect='first')
    def save(self, name: Optional[str] = None, output_dir: Optional[str] = None, interval: int = 1, **kwargs):
        """Save model.

        Args:
            name: The name of checkpoint to save.
            output_dir: An output_dir to save the model.
            interval: Save each interval steps.
            **kwargs:
                adapter_name: Lora adapter name.
                save_optimizer: Whether to save optimizer state.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        if optimizer_config.cur_step % interval != 0:
            return
        model = self.strategy.unwrap_model(self.model)
        processed_state_dict = {}
        save_kwargs = {}
        if adapter_name == _default_adapter_name:
            # Full model save
            processed_state_dict = self.strategy.get_full_state_dict(self.model)
        else:
            # LoRA adapter save
            state_dict = self.get_state_dict(adapter_name=adapter_name, **kwargs)
            for key, value in state_dict.items():
                key = key.replace(f'.{adapter_name}.', '.')
                processed_state_dict[key] = torch_util.to_local_tensor(value).cpu()

        if isinstance(model, PeftModel):
            if Platform.is_master():
                model.peft_config[adapter_name].save_pretrained(checkpoint_dir)
                save_file(processed_state_dict, os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
        else:
            model.save_pretrained(
                checkpoint_dir, state_dict=processed_state_dict, is_main_process=Platform.is_master(), **save_kwargs)

        self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)

        if kwargs.get('save_optimizer', False):
            self._save_training_state(
                checkpoint_dir,
                adapter_name=adapter_name,
                consumed_train_samples=kwargs.get('consumed_train_samples', 0),
            )

        return checkpoint_dir

    def _save_optimizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer = optimizer_config.optimizer
        lr_scheduler = optimizer_config.lr_scheduler

        if optimizer is not None:
            optimizer_path = os.path.join(output_dir, 'optimizer.pt')
            self.strategy.save_optimizer_checkpoint(self.model, optimizer, optimizer_path)
        if Platform.is_master():
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))

    def _save_training_state(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        self._save_optimizer(output_dir, adapter_name=adapter_name)

        optimizer_config = self.optimizer_group[adapter_name]

        if not Platform.is_master():
            return

        trainer_state = {
            'checkpoint_version': 1,
            'cur_step': optimizer_config.cur_step,
            'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
            'consumed_train_samples': kwargs.get('consumed_train_samples', 0),
        }
        with open(os.path.join(output_dir, 'trainer_state.json'), 'w', encoding='utf-8') as f:
            json.dump(trainer_state, f)

        if optimizer_config.scaler is not None:
            torch.save(
                {
                    'scaler_state_dict': optimizer_config.scaler.state_dict(),
                    'scaler_has_nan': optimizer_config.scaler_has_nan,
                },
                os.path.join(output_dir, 'scaler.pt'),
            )

        torch.save(self._get_training_rng_state(), os.path.join(output_dir, 'rng_state.pt'))

    def _save_tokenizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        template_ins = optimizer_config.template
        if Platform.is_master():
            if template_ins is not None:
                template_ins.processor.save_pretrained(output_dir)
            else:
                self._default_tokenizer.save_pretrained(output_dir)

    @remote_function()
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        """Load model state and optionally optimizer state from a checkpoint.

        Args:
            name: The name of checkpoint to load.
            output_dir: An output_dir to load the model.
            **kwargs:
                adapter_name: Adapter to load.
                load_optimizer: Whether to load optimizer and scheduler states.
        """
        load_optimizer = kwargs.get('load_optimizer', False)
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())

        if output_dir is None:
            if os.path.exists(name):
                checkpoint_dir = name
            else:
                # load from hub
                token = kwargs.pop('token', None)
                checkpoint_dir = HubOperation.download_model(name, token=token)
        else:
            checkpoint_dir = os.path.join(output_dir, name)
        model = self.strategy.unwrap_model(self.model)
        if isinstance(model, PeftModel):
            adapter_weights = load_peft_weights(checkpoint_dir, device='cpu')

            def load_peft_weights_for_fsdp2(model, adapter_weights, adapter_name='default'):
                from torch.distributed.tensor import DTensor, distribute_tensor

                model_sd = model.state_dict()
                converted_weights = {}
                for key, value in adapter_weights.items():
                    model_key = key
                    if f'.{adapter_name}.weight' not in model_key:
                        model_key = model_key.replace('.weight', f'.{adapter_name}.weight')
                    if model_key in model_sd:
                        param = model_sd[model_key]
                        if isinstance(param, DTensor) and not isinstance(value, DTensor):
                            value = distribute_tensor(value.to(param.device), param.device_mesh, param.placements)
                    converted_weights[key] = value

                set_peft_model_state_dict(model, converted_weights, adapter_name=adapter_name)

            if self.device_mesh.fsdp_world_size > 1:
                load_peft_weights_for_fsdp2(model, adapter_weights, adapter_name=adapter_name)
            else:
                set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)
        else:
            raise NotImplementedError

        if load_optimizer:
            self._load_optimizer(checkpoint_dir, adapter_name=adapter_name)

    def _load_optimizer(self, checkpoint_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        strict = kwargs.pop('strict', False)
        # assume optimizer and lr_scheduler are created
        optimizer_config = self.optimizer_group[adapter_name]

        optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
        scheduler_path = os.path.join(checkpoint_dir, 'scheduler.pt')

        if strict and not os.path.exists(optimizer_path):
            raise FileNotFoundError(optimizer_path)
        if strict and optimizer_config.lr_scheduler is not None and not os.path.exists(scheduler_path):
            logger.warning(
                f'Missing scheduler checkpoint {scheduler_path}; resuming without restoring lr scheduler state.', )

        if os.path.exists(optimizer_path) and optimizer_config.optimizer is not None:
            if self.strategy.needs_wrapped_optimizer_state() and not self._model_wrapped:
                self._lazy_wrap_model()
            self.strategy.load_optimizer_checkpoint(self.model, optimizer_config.optimizer, optimizer_path)

        if os.path.exists(scheduler_path) and optimizer_config.lr_scheduler is not None:
            state_dict = torch.load(scheduler_path, map_location='cpu', weights_only=True)
            optimizer_config.lr_scheduler.load_state_dict(state_dict)

    def _ensure_lora_dtype(self, model):
        """Force LoRA parameters to use the same dtype as base model for FSDP2 compatibility."""
        base_dtype = None
        for param in model.parameters():
            if param.dtype in (torch.float16, torch.bfloat16, torch.float32):
                base_dtype = param.dtype
                break
        if base_dtype is None:
            return

        # Convert all LoRA parameters to the base model dtype
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora_' in name.lower() and param.dtype != base_dtype:
                    param.data = param.data.to(base_dtype)

    def _load_scaler_state(self, scaler_path, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if optimizer_config.scaler is None:
            raise ValueError(f'Grad scaler is not configured for adapter {adapter_name!r}')

        scaler_state = torch.load(scaler_path, map_location='cpu', weights_only=True)
        optimizer_config.scaler.load_state_dict(scaler_state['scaler_state_dict'])
        optimizer_config.scaler_has_nan = scaler_state.get('scaler_has_nan', False)

    def _get_training_rng_state(self):
        state = {
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }

        device_prefix = Platform.device_prefix()
        device_module = getattr(torch, device_prefix, None)
        if device_module and hasattr(device_module, 'is_available') and device_module.is_available():
            state['device_type'] = device_prefix
            state['device_rng_state'] = device_module.get_rng_state()
        else:
            state['device_type'] = 'cpu'
            state['device_rng_state'] = None
        return state

    def _load_rng_state(self, rng_path):
        rng_state = torch.load(rng_path, map_location='cpu', weights_only=False)
        random.setstate(rng_state['python_rng_state'])
        np.random.set_state(rng_state['numpy_rng_state'])
        torch.set_rng_state(rng_state['torch_rng_state'])

        device_type = rng_state.get('device_type')
        device_rng_state = rng_state.get('device_rng_state')
        if device_type != 'cpu' and device_rng_state is not None:
            device_module = getattr(torch, device_type, None)
            if device_module and hasattr(device_module, 'is_available') and device_module.is_available():
                device_module.set_rng_state(device_rng_state)

    def _restore_training_state(self, checkpoint_dir, *, adapter_name=''):
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)

        adapter_name = adapter_name or self._get_default_group()
        optimizer_config = self.optimizer_group[adapter_name]
        self._load_optimizer(checkpoint_dir, adapter_name=adapter_name)
        scaler_path = os.path.join(checkpoint_dir, 'scaler.pt')
        if os.path.exists(scaler_path) and optimizer_config.scaler is not None:
            self._load_scaler_state(scaler_path, adapter_name=adapter_name)
        rng_path = os.path.join(checkpoint_dir, 'rng_state.pt')
        if os.path.exists(rng_path):
            self._load_rng_state(rng_path)
        optimizer_config.cur_step = trainer_state['cur_step']
        optimizer_config.gradient_accumulation_steps = trainer_state['gradient_accumulation_steps']

        return trainer_state

    @remote_function(dispatch='all', collect='first', sync=True)
    def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
        adapter_name = kwargs.get('adapter_name', '')

        has_adapter = (
            os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
            or os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.bin')))
        if has_adapter:
            self.load(checkpoint_dir, adapter_name=adapter_name)

        if not resume_only_model:
            trainer_state = self._restore_training_state(checkpoint_dir, adapter_name=adapter_name)
        else:
            with open(os.path.join(checkpoint_dir, 'trainer_state.json')) as f:
                trainer_state = json.load(f)

        return {
            'cur_step': trainer_state['cur_step'],
            'consumed_train_samples': trainer_state['consumed_train_samples'],
            'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
        }

    @remote_function(collect='first')
    def get_state_dict(self, **kwargs):
        return self._get_trainable_parameters(kwargs.pop('adapter_name', self._get_default_group()))

    @remote_function(collect='first')
    def get_peft_config_dict(self, adapter_name: str = None) -> dict:
        """Return the PEFT config as a dict for vLLM's PEFTHelper.

        Used by CheckpointEngineManager to pass peft_config to the sampler
        when doing LoRA-only weight sync.

        Returns:
            PEFT config dict, or None if the model has no LoRA adapter.
        """
        if adapter_name is None:
            adapter_name = self._get_default_group()
        optimizer_config = self.optimizer_group.get(adapter_name)
        if optimizer_config is None or optimizer_config.adapter_config is None:
            return None
        config = optimizer_config.adapter_config
        # PeftConfig can be a dict-like mapping (e.g. {adapter_name: LoraConfig})
        # or a single LoraConfig.  Normalize to a single config.
        if isinstance(config, dict):
            config = config.get(adapter_name, next(iter(config.values())))
        return config.to_dict() if hasattr(config, 'to_dict') else dict(config)

    @remote_function(collect='first', lazy_collect=False)
    def calculate_metric(self, is_training, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        return optimizer_config.calculate_metrics(is_training)

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        assert adapter_name, 'Use a different adapter_name, current is empty.'
        unwrapped_model = self.strategy.unwrap_model(self.model)
        if isinstance(config_or_dir, str):
            config_or_dir = HubOperation.download_model(config_or_dir)
            _adapted_model = PeftModel.from_pretrained(
                unwrapped_model,
                model_id=config_or_dir,
                adapter_name=adapter_name,
                is_trainable=kwargs.get('is_trainable', True))
            if unwrapped_model is self.model:
                self.model = _adapted_model
            else:
                # post check: unwrapped_model must be already a peft model before wrapping ddp
                assert isinstance(unwrapped_model, PeftModel)
            config = _adapted_model.peft_config
        else:
            config = config_or_dir
            if not isinstance(unwrapped_model, PeftModel):
                assert unwrapped_model is self.model, 'Cannot wrap model with peft after DDP/FSDP!'
                self.model = get_peft_model(unwrapped_model, config, adapter_name=adapter_name)
            else:
                unwrapped_model.add_adapter(adapter_name, config)

        self._ensure_lora_dtype(self.model)
        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config
        _gas_default = kwargs.get('gradient_accumulation_steps', 1)
        self.optimizer_group[adapter_name].gradient_accumulation_steps = _gas_default
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        self.active_group = adapter_name

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        """Add adapter to model.

        Args:
            adapter_name: The lora adapter name.
            config_or_dir:  The lora adapter config.
            **kwargs:
                is_trainable: Whether the adapter is trainable.
                gradient_accumulation_steps: The number of gradient accumulation steps
        """
        self._patch_adapter(adapter_name, config_or_dir, **kwargs)

    @remote_function()
    def set_template(self, template_cls: Union[Type[Template], str, Template], **kwargs):
        """Set template. This is optional, if you need to input `Trajectory`,
            you need to set the template to encode them.

        Args:
            template_cls: A template_cls class name, a template_cls plugin id, or a template_cls class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the template_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['model_id'] = self.tokenizer_id
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        optimizer_config.template = template

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, InputProcessor, Callable], **kwargs):
        """Set task processor to prepare the task inputs.
        Args:
            processor_cls: A processor_cls class name, a processor_cls plugin id,
                or a processor_cls class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the processor_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['device_mesh'] = self.device_mesh
        processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)
        optimizer_config.processor = processor

    @remote_function()
    def set_grad_scaler(self, **kwargs):
        """Set the grad scaler.
        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the GradScaler instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        from torch.amp.grad_scaler import GradScaler
        grad_scaler_config = self.grad_scaler_config.copy()
        grad_scaler_config.update(kwargs)
        optimizer_config.scaler = GradScaler(**grad_scaler_config)

    @remote_function()
    def add_metric(self, metric_cls: Union[Metric, str], is_training: Optional[bool] = None, **kwargs):
        """Add an eval metric

        Args:
            metric_cls: A metric class type or id.
            is_training: Whether the metric is for training. If None, it will be used for both training and evaluation.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the metric_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['device_mesh'] = self.device_mesh
        kwargs['process_group'] = optimizer_config._dp_group
        if is_training is None or is_training is True:
            optimizer_config.train_status.metrics.append(construct_class(metric_cls, Metric, twinkle.metric, **kwargs))
        if not is_training:
            optimizer_config.eval_status.metrics.append(construct_class(metric_cls, Metric, twinkle.metric, **kwargs))

    def _get_nb_trainable_parameters(self, adapter_name, model):
        return PeftModel.get_nb_trainable_parameters(model)

    def _get_trainable_parameters_example(self, adapter_name, model):
        trainable_param_names = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                trainable_param_names.append(name)
        trainable_param_names = trainable_param_names[:5] + ['...'] + trainable_param_names[-5:]
        trainable_param_names = '\n'.join(trainable_param_names)
        return trainable_param_names

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def get_train_configs(self, **kwargs) -> str:
        expr = ''
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if optimizer_config.adapter_config is not None:
            config = optimizer_config.adapter_config.__dict__
        else:
            config = {}
        config = {key: str(value) for key, value in config.items() if value is not None}
        trainable_params, all_param = self._get_nb_trainable_parameters(adapter_name, self.model)
        trainable_param_names = self._get_trainable_parameters_example(adapter_name, self.model)
        if optimizer_config.optimizer is not None:
            expr += (f'Adapter config:\n'
                     f'{json.dumps(config, indent=2, ensure_ascii=False)}\n'
                     f'Trainable parameters examples:\n'
                     f'{trainable_param_names}\n'
                     f'Trainable params: {trainable_params:,d} || all params: {all_param:,d} || '
                     f'trainable%: {100 * trainable_params / all_param:.4f}\n'
                     f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
                     f'Learning rate: {optimizer_config.optimizer.defaults.get("lr", "No default lr")}\n'
                     f'Lr scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
                     f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n')
        else:
            expr += (f'Adapter config:\n'
                     f'{json.dumps(config, indent=2, ensure_ascii=False)}\n'
                     f'Trainable parameters examples:\n'
                     f'{trainable_param_names}\n'
                     f'Trainable params: {trainable_params:,d} || all params: {all_param:,d} || '
                     f'trainable%: {100 * trainable_params / all_param:.4f}%\n')
        return expr

    # =========================================================================
    # Checkpoint Engine weight sync (from CheckpointEngineMixin)
    # =========================================================================
    # prepare_checkpoint_engine, init_checkpoint_process_group, and
    # finalize_checkpoint_engine are inherited from CheckpointEngineMixin.
    # Only send_weights_via_checkpoint_engine is model-specific.

    @remote_function(dispatch='all', lazy_collect=True)
    def send_weights(
        self,
        adapter_name: str = None,
        base_sync_done: bool = False,
        merge_and_sync: bool = False,
        model_keys: List[str] = None,
        **kwargs,
    ):
        if adapter_name is None:
            adapter_name = self._get_default_group()
        engine = self._get_or_create_checkpoint_engine()
        # Get state dict from unwrapped model
        model = self.strategy.unwrap_model(self.model)

        def _normalize(name: str, keep_base_layer: bool) -> str:
            name = name.replace('base_model.model.', '')
            if not keep_base_layer:
                name = name.replace('.base_layer', '')
            else:
                if 'conv1d.weight' in name:
                    if model_keys and any('conv1d.base_layer.weight' in name for name in model_keys):
                        name = name.replace('conv1d.weight', 'conv1d.base_layer.weight')
            return name

        def _print_weight_example(names):
            for name in names[:3]:
                logger.info(f'Sync weight: {name}')

        def _is_lora_key(name: str) -> bool:
            return 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name

        if base_sync_done and adapter_name:
            if merge_and_sync:
                # LoRA Training and sync full model(merge_adapter)
                # merge and skip lora weigts(already merged)
                # trim prefix(base_model.model.) and suffix(.base_layer)
                def weight_generator():
                    if isinstance(model, PeftModel):
                        model.merge_adapter()
                    names = []
                    for name, tensor in model.state_dict().items():
                        if _is_lora_key(name):
                            continue
                        tensor = Torch.to_local_tensor(tensor)
                        name = _normalize(name, keep_base_layer=False)
                        names.append(name)
                        yield name, tensor
                    _print_weight_example(names)
                    if isinstance(model, PeftModel):
                        model.unmerge_adapter()
            else:
                # LoRA-only mode: send only adapter weights.
                # Use PEFT's get_peft_model_state_dict for clean LoRA extraction
                from peft.utils import get_peft_model_state_dict
                lora_state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)

                def weight_generator():
                    names = []
                    for name, tensor in lora_state_dict.items():
                        tensor = Torch.to_local_tensor(tensor)
                        name = _normalize(name, keep_base_layer=True)
                        names.append(name)
                        yield name, tensor
                    _print_weight_example(names)

        else:
            # First full base-model sync.  Whether to keep ``.base_layer.``
            # depends on whether the sampler uses ``enable_lora``:
            #   merge_and_sync=True  → enable_lora=False → strip .base_layer
            #   merge_and_sync=False → enable_lora=True  → keep .base_layer
            keep_base_layer = not merge_and_sync
            state_dict = model.state_dict()

            def weight_generator():
                names = []
                for name, tensor in state_dict.items():
                    if _is_lora_key(name):
                        continue
                    tensor = Torch.to_local_tensor(tensor)
                    name = _normalize(name, keep_base_layer=keep_base_layer)
                    names.append(name)
                    yield name, tensor
                _print_weight_example(names)

        # Run async send_weights in a dedicated event loop thread.
        # We cannot use the Ray worker's event loop because it may already
        # be occupied, and send_weights uses run_in_executor internally.
        async def _send():
            await engine.send_weights(weight_generator())

        result_container = {'error': None}

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_send())
                finally:
                    loop.close()
            except Exception as e:
                result_container['error'] = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        if result_container['error'] is not None:
            raise result_container['error']
