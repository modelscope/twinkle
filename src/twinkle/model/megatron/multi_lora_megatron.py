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
from copy import deepcopy
from functools import partial
from peft import LoraConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig, PretrainedConfig
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from twinkle import DeviceMesh, Platform, remote_class, remote_function, requires, template, torch_util
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.utils import seed_from_env
from twinkle.loss import Loss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from twinkle.utils import get_logger
from twinkle.utils.safetensors import load_state_dict, save_state_dict
from ..base import ModelLoaderProtocol
from ..multi_lora import MultiLora, MultiLoraTaskHeadSlots
from ..tasks import get_task_spec_obj
from .utils._mindspeed_runtime import ensure_mindspeed_adaptor_patched
from .megatron import MegatronModel
from .mtp.mtp import MtpTrainingState
from .strategy import MegatronStrategy

logger = get_logger()


@contextmanager
def _seq_cls_head_lora_load_gate(bridge):
    """Let mcore-bridge restore a LoRA-only sequence classification head.

    mcore-bridge 1.6.4 checks for ``score.weight`` before entering its head loader,
    but a PEFT checkpoint contains only ``score.lora_A/B.weight``. Provide the key
    as a load-time gate; the LoRA branch never reads the corresponding base tensor.
    """
    required = ('_convert_post_process', '_peft_format', 'hf_score_key', 'config')
    if any(not hasattr(bridge, name) for name in required):
        raise RuntimeError('seq_cls head LoRA restore requires the mcore-bridge 1.6.4 loader API')
    original = bridge._convert_post_process

    def convert_post_process(mg_model, hf_state_dict, hf_prefix, to_mcore):
        if to_mcore and bridge._peft_format and bridge.config.task_type == 'seq_cls':
            head_key = bridge.hf_score_key
            module_key, _, param_key = head_key.rpartition('.')
            lora_key = f'{module_key}.lora_A.{param_key}'
            if head_key not in hf_state_dict and lora_key in hf_state_dict:
                hf_state_dict = dict(hf_state_dict)
                hf_state_dict[head_key] = hf_state_dict[lora_key]
        return original(mg_model, hf_state_dict, hf_prefix, to_mcore)

    bridge._convert_post_process = convert_post_process
    try:
        yield
    finally:
        bridge._convert_post_process = original


@remote_class(execute='all')
class MultiLoraMegatronModel(MegatronModel):
    # Class-level defaults (_head_task / task_head_slots) are inherited from TrainableModel
    # -- see the attached-head lifecycle there; a slot-less host reads as None either way.

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
        task: Optional[str] = None,
        task_head_loader: Optional[tuple] = None,
        num_labels: Optional[int] = None,
        dp_balance: bool = True,
        **kwargs,
    ):
        self._init_dp_balance(dp_balance)
        requires('megatron_core')
        new_special_tokens = kwargs.pop('new_special_tokens', None)
        if new_special_tokens:
            # Tenants share one frozen base tokenizer/embedding, so the vocab cannot be grown here;
            # pre-grow the tokenizer or use the single-tenant MegatronModel.
            raise ValueError('new_special_tokens is not supported on the multi-LoRA model.')
        nn.Module.__init__(self)
        from twinkle.patch.megatron_peft import MegatronPeft

        self.model_id = model_id
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision

        self._model_path = HubOperation.download_model(model_id, revision=kwargs.pop('revision', None))
        self.tokenizer_id = kwargs.pop('tokenizer_id', self.model_id)
        self._default_tokenizer = None
        self.use_distributed_optimizer = False
        kwargs.pop('use_distributed_optimizer', None)
        self.variable_seq_lengths = kwargs.get('variable_seq_lengths', True)
        self.optimizer_group = {}
        # Popped here so neither key reaches MegatronStrategy / ModelConfig; mirrors the parent.
        self._router_replay_enabled = bool(kwargs.pop('enable_router_replay', False))
        self._router_replay_applied = False
        if self._router_replay_enabled:
            kwargs['moe_enable_routing_replay'] = True
        # Same ordering constraint as MegatronModel: must precede the CUDA context creation below.
        if MegatronStrategy.uses_fsdp(ddp_config):
            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '32'
        torch_util.set_device()
        # mcore requires an initialised process group even at world_size==1.
        self._try_init_process_group(force=True)
        ensure_mindspeed_adaptor_patched()
        requires('mcore_bridge')

        kwargs.update({
            'recompute_granularity': recompute_granularity,
            'recompute_modules': recompute_modules,
            'recompute_method': recompute_method,
            'recompute_num_layers': recompute_num_layers,
            'variable_seq_lengths': self.variable_seq_lengths,
        })
        seed = kwargs.pop('seed', None) or seed_from_env(42)
        model_loader: Optional[ModelLoaderProtocol] = kwargs.pop('model_loader', None)
        if config is not None:
            self.hf_config = config
        elif model_loader is not None:
            # Same caller-supplied builder contract as MegatronModel (see ModelLoaderProtocol): only the
            # config hooks apply on this backend -- mcore builds the module and the bridge loads weights.
            self.hf_config = model_loader.process_config(model_loader.build_config(self._model_path))
        else:
            self.hf_config = AutoConfig.from_pretrained(self._model_path, trust_remote_code=True)
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
        # Head slots -- the head-side mirror of the LoRA slot pre-allocation above: one task
        # head per LoRA slot, built BEFORE the wrap below so every slot's parameters enter
        # the DDP / mcore machinery (reducers only discover parameters at wrap time). A
        # multi-tenant host serves ONE head task (the slot's head type is fixed here);
        # ``task`` names it ('decision' rebuilds the family loader's head from the shipped
        # NAME, the seq_cls family builds its standard head from the config). None = this
        # host serves no head-task tenant: no slots, no memory.
        self._head_task = task or ('decision' if task_head_loader is not None else None)
        self._task_head_loader = task_head_loader
        self._num_labels = num_labels
        self.task_head_slots = MultiLoraTaskHeadSlots(max_loras) if self._head_task is not None else None
        if self.task_head_slots is not None:
            self._validate_task_head_config()
            self._build_task_head_slots()
        self.model = self.strategy.wrap_model(self.model)
        self.strategy.finish_param_config(self.model, None)
        self.multi_adapter.save_initial_weights()
        self._model_wrapped = True
        self._finish_config = True
        # Active group for compatibility with single adapter
        self.active_group = None
        # Inherited forward_backward reads this before any tenant adapter is added.
        self._mtp = MtpTrainingState()
        self.multi_adapter.reset_adapter_status()

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name and adapter_name in self.optimizer_group, (f'Use a valid adapter_name first, '
                                                                       f'current is: {adapter_name}')

    def _lazy_wrap_model(self):
        pass

    def _lazy_finish_param_config(self):
        pass

    def _build_task_head_slots(self) -> None:
        """Pre-allocate the head slots on the last-PP chunk (see ``MultiLoraTaskHeadSlots``).

        Building and its validation live on the shared TrainableModel lifecycle
        (``_build_task_head``); only the megatron placement (place onto the last-PP
        chunk) is backend-specific and stays here. Only the last pipeline stage
        builds -- the other stages carry no heads, so their bind/release calls keep
        the bookkeeping only and the optimizer-side filter matches nothing there."""
        from megatron.core import mpu
        if not mpu.is_pipeline_last_stage():
            return
        spec_obj = get_task_spec_obj(self._head_task)

        def build_head():
            head = self._build_task_head(self._head_task, self._task_head_loader, self._num_labels)
            return spec_obj.place_megatron_head(head, self.model[-1])
        self.task_head_slots.build(self._task_head_host(), build_head)

    def _validate_task_head_config(self) -> None:
        """Fail at construction, on every rank, before any slot or memory is spent."""
        task = self._head_task
        spec_obj = get_task_spec_obj(task)
        if not spec_obj.row().readout:
            raise ValueError(f'task={task!r} carries no readout head; nothing to pre-allocate.')
        if task == 'decision':
            if self._task_head_loader is None:
                raise ValueError(
                    "task='decision' needs task_head_loader=(module, class) to rebuild its family head.")
        elif self._task_head_loader is not None:
            raise ValueError(
                'task_head_loader is the decision families\' head replay; '
                f'task={task!r} builds its standard head from the config instead.')
        elif self._num_labels is None:
            # Validates num_labels / hidden_size now (the actual build happens per slot).
            spec_obj.build_head(self.hf_config, None)

    def _bind_task_head_slot(self, adapter_name: str) -> None:
        """Bind the lowest free head slot to a tenant (see ``MultiLoraTaskHeadSlots.bind``)."""
        if self.task_head_slots is None:
            raise RuntimeError(
                'this MultiLoraMegatronModel was built without a head task (task= / task_head_loader=), '
                'so it carries no task head slots; construct it with task= to serve head-task '
                'tenants.')
        self.task_head_slots.bind(adapter_name, self._task_head_host())

    def _release_task_head_slot(self, adapter_name: str) -> None:
        """Unbind a tenant's head slot and restore its build-time weights in place."""
        if self.task_head_slots is None:
            return
        self.task_head_slots.release(adapter_name, self._task_head_host(),
                                     self.multi_adapter._write_param_tensor)

    @remote_function(dispatch='slice_dp', collect='tensor_dict', sync=True)
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

    @remote_function(dispatch='slice_dp', collect='tensor_dict', sync=True)
    def forward_backward(self,
                         *,
                         inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().forward_backward(inputs=inputs, **kwargs)

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
    def optimizer_context(self, tenant_adapter_name: str):
        """Temporarily replace named_parameters on each module in self.model
        so that only parameters belonging to the tenant are visible: its LoRA slot
        parameters plus its bound task head slot (a head-task tenant trains both).

        Takes the TENANT name. The LoRA pattern needs the physical slot name, the head
        pattern keys on the tenant (``MultiLoraTaskHeadSlots.bindings``); passing the
        physical name -- the value ``multi_adapter.adapter()`` yields -- silently dropped
        the head from every optimizer build: the head trained gradients that never
        updated."""
        physical = self.multi_adapter.find_lora_by_tenant(tenant_adapter_name).adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(physical)}\.')
        head_pattern = (self.task_head_slots.optimizer_pattern(tenant_adapter_name)
                        if self.task_head_slots is not None else None)
        originals = []
        for module in self.model:
            orig = module.named_parameters

            def make_filtered(orig_fn):

                def filtered(prefix: str = '', recurse: bool = True, **kwargs):
                    for name, param in orig_fn(prefix=prefix, recurse=recurse, **kwargs):
                        if param.requires_grad and (pattern.search(name)
                                                    or (head_pattern is not None and head_pattern.search(name))):
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
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            with self.optimizer_context(kwargs.get('adapter_name')):
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
        return os.path.join(checkpoint_dir, f'optimizer_rank_{rank}.safetensors')

    @staticmethod
    def _save_local_training_rng_state():
        from megatron.core import tensor_parallel

        rng_state = {
            'random_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }
        # Backend-agnostic device RNG capture (CUDA / NPU / MPS). Key is kept as
        # 'cuda_rng_state' for backward compatibility with existing checkpoints.
        device_rng = Platform.get_device_rng_state()
        if device_rng is not None:
            rng_state['cuda_rng_state'] = device_rng
        rng_state['rng_tracker_states'] = tensor_parallel.get_cuda_rng_tracker().get_states()
        return rng_state

    @staticmethod
    def _load_local_training_rng_state(rng_state):
        from megatron.core import tensor_parallel

        random.setstate(rng_state['random_rng_state'])
        np.random.set_state(rng_state['np_rng_state'])
        torch.set_rng_state(rng_state['torch_rng_state'])
        # Backend-agnostic device RNG restore: tolerates ckpt produced on different
        # backend (key absent or None) and avoids hard-coded torch.cuda on NPU.
        if 'cuda_rng_state' in rng_state:
            Platform.set_device_rng_state(rng_state['cuda_rng_state'])
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

        save_state_dict(state_dict, self._rank_local_optimizer_path(checkpoint_dir))

        if dist.is_initialized():
            dist.barrier()

    def _load_multi_lora_optimizer(self, checkpoint_dir: str, adapter_name: str = '', **kwargs):
        no_load_optim = kwargs.pop('no_load_optim', False)
        no_load_rng = kwargs.pop('no_load_rng', True)
        optimizer_config = self.optimizer_group.get(adapter_name)
        state_dict = load_state_dict(self._rank_local_optimizer_path(checkpoint_dir))

        if not no_load_optim and optimizer_config is not None:
            if optimizer_config.optimizer is not None and 'optimizer' in state_dict:
                optimizer_config.optimizer.load_state_dict(state_dict['optimizer'])
                device = Platform.get_local_device()
                for _, group_state in optimizer_config.optimizer.state.items():
                    if not isinstance(group_state, dict):
                        continue
                    for k, v in group_state.items():
                        if isinstance(v, torch.Tensor):
                            group_state[k] = v.to(device)
            if optimizer_config.lr_scheduler is not None and 'opt_param_scheduler' in state_dict:
                optimizer_config.lr_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        # RNG is intentionally not restored: the global RNG is shared across tenants, and
        # restoring it would silently affect other active adapters' dropout / init.
        if not no_load_rng and 'rng_state' in state_dict:
            logger.warning('Skipping RNG state restoration in multi-tenant mode. '
                           'Global RNG is shared across tenants; restoring it would '
                           'affect other active adapters.')
        if optimizer_config is not None and 'iteration' in state_dict:
            optimizer_config.cur_step = state_dict['iteration']

    def _save_task_head_checkpoint(self, checkpoint_dir: str, adapter_name: Optional[str]) -> None:
        """Slot-pool form of the shared head checkpoint (see TrainableModel)."""
        if self.task_head_slots is None:
            return
        super()._save_task_head_checkpoint(checkpoint_dir, adapter_name)

    def _task_head_state(self, adapter_name: Optional[str]) -> Dict[str, Any]:
        """The bound tenant's head weights out of its slot, under slot-independent
        'task_head.' keys; the LoRA base writes through the bridge and never sees the head."""
        if self.task_head_slots is None:
            return {}
        return self.task_head_slots.export_state(
            adapter_name, self._task_head_host(), self.multi_adapter._read_param_tensor)

    def _restore_task_head_state(self, adapter_name: Optional[str], state: Dict[str, Any]) -> None:
        """Restore a ``_task_head_state`` snapshot into the tenant's currently bound
        slot, in place. An empty snapshot passes (a rank without slots, a head-less
        checkpoint); a non-empty snapshot on a slot-less host is a mismatched deployment
        that would silently lose the trained head -- fail loudly instead."""
        if not state:
            return
        if self.task_head_slots is None:
            raise ValueError(
                'task_head.safetensors exists but this model was built without task-head slots '
                '(task= / task_head_loader=); it cannot serve the checkpoint\'s head task. Load it '
                'on a model constructed with the same task.')
        self.task_head_slots.import_state(
            adapter_name, self._task_head_host(), state, self.multi_adapter._write_param_tensor)

    # lazy_collect=False: a checkpoint write the caller acts on immediately (asserting the
    # directory, uploading it, loading it back) must block until it lands; the lazy default
    # would return the path while the actors are still inside _save_hf_format.
    @remote_function(dispatch='all', collect='first', sync=True, lazy_collect=False, timeout=3600)
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
            self._save_task_head_checkpoint(checkpoint_dir, adapter_name)

            self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)
            if save_optimizer:
                with self.optimizer_context(adapter_name):
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
                    'component_states': kwargs.get('component_states', {}),
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
        with self.multi_adapter.save_context(adapter_name) as real_adapter_name, _seq_cls_head_lora_load_gate(bridge):
            model = self.strategy.unwrap_model(self.model)
            bridge.load_weights(
                model,
                checkpoint_dir,
                peft_format=True,
                adapter_name=real_adapter_name,
                converter=self.multi_adapter.load_lora_converter)
        self._load_task_head_checkpoint(checkpoint_dir, adapter_name)

        optimizer_config = self.optimizer_group.get(adapter_name)
        if optimizer_config is not None and optimizer_config.optimizer is not None:
            optimizer_config.optimizer.reload_model_params()
        if load_optimizer:
            with self.optimizer_context(adapter_name):
                self._load_multi_lora_optimizer(checkpoint_dir, adapter_name=adapter_name, **kwargs)

        if dist.is_initialized():
            dist.barrier()

    # lazy_collect=False: the driver consumes the returned dict immediately (loop.resume
    # subscripts state['cur_step']); a lazy callable default would break it.
    @remote_function(dispatch='all', collect='first', sync=True, lazy_collect=False, timeout=3600)
    def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
        adapter_name = kwargs.pop('adapter_name', None)
        self._check_adapter_valid(adapter_name)
        optimizer_config = self.optimizer_group.get(adapter_name)

        self.load(checkpoint_dir, load_optimizer=not resume_only_model, adapter_name=adapter_name, **kwargs)
        if resume_only_model:
            return {
                'cur_step': 0,
                'consumed_train_samples': 0,
                'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
                'component_states': {},
            }

        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        if not os.path.isfile(trainer_state_path):
            raise FileNotFoundError(f'trainer_state.json not found in {checkpoint_dir}. '
                                    f'Ensure the checkpoint was saved with save_optimizer=True.')
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)

        if optimizer_config is not None:
            optimizer_config.cur_step = trainer_state['cur_step']
            optimizer_config.gradient_accumulation_steps = trainer_state['gradient_accumulation_steps']

        result = {
            'cur_step': trainer_state['cur_step'],
            'consumed_train_samples': trainer_state['consumed_train_samples'],
            'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
        }
        if 'component_states' in trainer_state:
            result['component_states'] = trainer_state['component_states']
        return result

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        # MultiLora.get_state_dict takes the tenant name positionally; kwargs here
        # only ever carries adapter_name.
        return self.multi_adapter.get_state_dict(kwargs.get('adapter_name'))

    @remote_function(dispatch='all', sync=True, timeout=3600)
    def add_adapter_to_model(
        self,
        adapter_name: str,
        config_or_dir: Union[Dict[str, Any], LoraConfig, str],
        **kwargs,
    ):
        # prevent opening requires_grad of the base model
        # prevent loading malicious code
        if isinstance(config_or_dir, str):
            raise ValueError('config_or_dir does not support str, '
                             'because loading config from modelhub may causing unexpected behavior')
        if not isinstance(config_or_dir, LoraConfig):
            raise ValueError('config_or_dir must be a LoraConfig instance')
        config_or_dir = deepcopy(config_or_dir)
        if config_or_dir.modules_to_save:
            raise ValueError('modules_to_save is not supported for a multi-tenant LoRA adapter.')
        self.multi_adapter.validate_tenant_target_modules(config_or_dir.target_modules,
                                                         getattr(config_or_dir, 'target_parameters', None))
        # Limit the max peft version in pyproject.toml, in case any newer version opens some untested module grad.
        config_or_dir.bias = 'none'
        config_or_dir.init_lora_weights = False
        config_or_dir.trainable_token_indices = None
        # ``task`` is an optional cross-check: the construction ``task=`` already declared this
        # host's head, so EVERY tenant carries it (a deployment names its task once -- the
        # yaml's ``task_type``); the forward task stays a per-call argument. Checked BEFORE
        # any state mutation so a mismatched add leaves no half-initialized tenant behind.
        task = kwargs.pop('task', None)
        if task is not None and task != self._head_task:
            raise ValueError(
                f'this host was built for task={self._head_task!r} head slots; '
                f'add_adapter_to_model(task={task!r}) does not match. Construct the host with '
                'the matching task to serve that head task.')
        if self._head_task is not None:
            # Bind BEFORE acquire so a host that cannot serve this tenant fails before any
            # state mutation; roll the binding back if the LoRA acquire fails.
            self._bind_task_head_slot(adapter_name)
        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config_or_dir
        self.optimizer_group[adapter_name].gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        try:
            self.multi_adapter.acquire_lora(tenant_adapter_name=adapter_name, config=config_or_dir)
        except Exception:
            self._release_task_head_slot(adapter_name)
            self.optimizer_group.pop(adapter_name, None)
            raise

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
        self._release_task_head_slot(adapter_name)
