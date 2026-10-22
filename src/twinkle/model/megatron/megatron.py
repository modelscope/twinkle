# Copyright (c) ModelScope Contributors. All rights reserved.
import contextlib
import inspect
import json
import numpy as np
import os
import random
import re
import torch
import torch.distributed as dist
import torch.nn as nn
from contextlib import contextmanager
from dataclasses import dataclass
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PretrainedConfig
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Type, Union

import twinkle
import twinkle.metric
import twinkle.patch
from twinkle import DeviceMesh, Platform, remote_class, remote_function, requires, torch_util
from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin
from twinkle.checkpoint_engine.weight_keys import add_base_layer_suffix, is_lora_key, normalize_param_name, print_weight_example
from twinkle.data_format import InputFeature, ModelOutput, Trajectory, is_encoded
from twinkle.hub import HubOperation
from twinkle.infra import collect_tensor_dict, remote_function
from twinkle.infra._torch_setup import seed_from_env
from twinkle.loss import CrossEntropyLoss, Loss
from twinkle.metric import LossMetric, Metric, TrainMetric
from twinkle.model.base import ModelLoaderProtocol, TrainableModel, copy_checkpoint_args, rotate_checkpoints
from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.patch import Patch, apply_context, apply_patch
from twinkle.perf import timed
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.template.utils import construct_template
from twinkle.utils import construct_class, get_logger, selective_log_softmax
from .moe.router_replay import RouterReplay
from .mtp.mtp import MtpTrainingState
from .utils._mindspeed_runtime import ensure_mindspeed_adaptor_patched
from .utils.utils import _find_embedding, _find_layers
from .strategy import MegatronStrategy

logger = get_logger()


def _resolve_task_context(model, task):
    """The per-forward Patch for ``task``: 'embedding' swaps the output_layer for identity
    (with TP/SP gather) and pools to ``[n_seqs, hidden]``; causal_lm/seq_cls/reranker/value
    need no patch (mcore-bridge already builds the right head; pooling happens in forward_step)."""
    if task in (None, 'causal_lm', 'seq_cls', 'reranker', 'generative_reranker', 'value'):
        return contextlib.nullcontext()
    if task == 'embedding':
        from twinkle.patch.megatron_emb import MegatronEmbeddingPatch
        return apply_context(model, MegatronEmbeddingPatch())
    raise ValueError(f'Unknown task={task!r}; expected one of: causal_lm, embedding, seq_cls, reranker, '
                     'generative_reranker, value.')


@dataclass
class MegatronOptimizerGroup(BaseOptimizerGroup):
    """Optimizer group for Megatron training.

    Similar to OptimizerGroup but adapted for Megatron's distributed training.
    """
    # Megatron-specific fields
    _last_step_success: bool = True

    def __post_init__(self):
        if self._device_mesh.data_world_size > 1:
            self._dp_group = self._device_mesh.create_process_group(['dp', 'fsdp'])
        train_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]
        self.train_status = TrainStatus(metrics=train_metrics)

        eval_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]
        self.eval_status = TrainStatus(metrics=eval_metrics)

    def _get_lr(self):
        _lrs = []
        if self.optimizer is None:
            return _lrs
        _default_lr = self.optimizer.chained_optimizers[0].config.lr
        for param_group in self.optimizer.param_groups:
            _lrs.append(param_group.get('lr', _default_lr))
        return _lrs


_default_adapter_name = ''


@remote_class(execute='all')
class MegatronModel(TrainableModel, nn.Module, CheckpointEngineMixin):

    _mcore_async_queue = None

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
        vit_gradient_checkpointing: Optional[bool] = None,
        vit_gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
        new_special_tokens: Optional[List[str]] = None,
        bridge_backend='mcore-bridge',
        align_grad_reduce: bool = True,
        nccl_comm_warmup: bool = False,
        attn_impl: Optional[str] = None,
        **kwargs,
    ):
        requires('megatron_core')
        nn.Module.__init__(self)
        from twinkle.patch.megatron_peft import MegatronPeft

        self.model_id = model_id
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._model_path = HubOperation.download_model(model_id, revision=kwargs.pop('revision', None))
        self.tokenizer_id = kwargs.pop('tokenizer_id', self.model_id)
        self._default_tokenizer = None
        self.use_distributed_optimizer = kwargs.pop('use_distributed_optimizer', True)
        self.variable_seq_lengths = kwargs.get('variable_seq_lengths', True)
        self._router_replay_enabled = bool(kwargs.pop('enable_router_replay', False))
        self._router_replay_applied = False
        if self._router_replay_enabled:
            kwargs['moe_enable_routing_replay'] = True
        if MegatronStrategy.uses_fsdp(ddp_config):
            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '32'
        torch_util.set_device()
        self._try_init_process_group()
        # MindSpeed must patch before mcore_bridge imports its patcher, otherwise
        # mcore_bridge pulls in megatron.core/TE too early on NPU.
        ensure_mindspeed_adaptor_patched()
        requires('mcore_bridge')
        # Route batched pipeline P2P through the WORLD group to avoid subgroup-P2P PP hangs.
        from twinkle.patch.megatron_p2p import MegatronBatchedP2PGroupPatch
        apply_patch(None, MegatronBatchedP2PGroupPatch())
        # Parallelize torch DCP checkpoint reads (mcore optimizer/RNG resume) for faster loading.
        from twinkle.patch.torch_dcp_reader import TorchDCPParallelReaderPatch
        apply_patch(None, TorchDCPParallelReaderPatch())
        # Skip slow, redundant torch DCP shard validation during mcore checkpoint save/load.
        from twinkle.patch.torch_dcp_validation import MegatronDCPValidationPatch
        apply_patch(None, MegatronDCPValidationPatch())

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
            self.hf_config = model_loader.process_config(model_loader.build_config(self._model_path))
        else:
            from transformers import AutoConfig
            # Fallback path (no ModelLoader): this config load may execute the checkpoint's own
            # Python. State it loudly -- the same advisory class as a hardcoded trust_remote_code.
            logger.warning_once('loading %s with trust_remote_code=True: the checkpoint\'s own Python code '
                                'will execute inside this process. Only use checkpoints you trust.',
                                self._model_path)
            self.hf_config = AutoConfig.from_pretrained(self._model_path, trust_remote_code=True)
        if new_special_tokens:
            self._extend_vocab_for_new_special_tokens(new_special_tokens)
        self.strategy = MegatronStrategy(
            self._model_path,
            self.device_mesh,
            mixed_precision=mixed_precision,
            config=self.hf_config,
            ddp_config=ddp_config or {},
            seed=seed,
            use_distributed_optimizer=self.use_distributed_optimizer,
            bridge_backend=bridge_backend,
            align_grad_reduce=align_grad_reduce,
            nccl_comm_warmup=nccl_comm_warmup,
            attn_impl=attn_impl,
            **kwargs)
        self.model: List[nn.Module] = self.strategy.create_megatron_model(load_weights)
        if vit_gradient_checkpointing is not None:
            self._apply_vit_gradient_checkpointing(vit_gradient_checkpointing, vit_gradient_checkpointing_kwargs)
        if new_special_tokens:
            # The rows added by the vocab raise above are zero-filled by mcore; give them real values before
            # the optimizer group snapshots the parameters below.
            self._initialize_new_special_token_embeddings()

        self._model_wrapped = False
        self._finish_config = False
        # MTP bookkeeping (pass gating, one-shot trainability warning, loss-tracker draining)
        # lives in the MtpTrainingState instance; the model keeps thin same-named shells below.
        self._mtp = MtpTrainingState()
        # This correctly handles vocab sharding in Tensor Parallelism
        self.optimizer_group: Dict[str, MegatronOptimizerGroup] = {
            _default_adapter_name: self._construct_default_optimizer_group()
        }
        self.optimizer_group[_default_adapter_name].adapter_name = _default_adapter_name
        self.active_group = _default_adapter_name
        MegatronPeft().__call__()

    def _apply_vit_gradient_checkpointing(self, enabled: bool, gc_kwargs: Optional[Dict[str, Any]]) -> None:
        """Checkpoint each HF vision tower of a multimodal mcore model.

        mcore builds the tower under ``visual`` on the first PP stage only; tower names come from
        ``model_meta.visual_cls._vision_tower`` so this stays model-agnostic. The global recompute
        knobs do not reach the external tower, hence its own switch. Worker-side: touches real
        modules; ``enable_input_require_grads`` keeps gradients flowing through a checkpointed
        tower when the input embedding is frozen (LoRA / quantized).
        """
        from twinkle.utils import deep_getattr
        for chunk in self.model:
            core = chunk
            while hasattr(core, 'module'):  # unwrap DDP / Float16Module down to the mcore model
                core = core.module
            visual = getattr(core, 'visual', None)
            if visual is None:  # not the first pipeline stage, or a plain LLM
                continue
            visual_cls = getattr(getattr(core, 'model_meta', None), 'visual_cls', None)
            for tower_name in (getattr(visual_cls, '_vision_tower', None) or []):
                tower = deep_getattr(visual, tower_name)
                if tower is None:
                    continue
                try:
                    if enabled:
                        if gc_kwargs:
                            tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
                        else:
                            tower.gradient_checkpointing_enable()
                        tower.enable_input_require_grads()
                    else:
                        tower.gradient_checkpointing_disable()
                except (NotImplementedError, AttributeError, ValueError) as exc:
                    logger.warning(f'vision-tower gradient checkpointing skipped for {tower_name}: {exc}')

    def _extend_vocab_for_new_special_tokens(self, new_special_tokens: List[str]) -> None:
        """Grow the tokenizer vocab and raise ``hf_config.vocab_size`` to match.

        mcore builds VocabParallelEmbedding FROM ``hf_config.vocab_size``, so the vocab must be
        raised BEFORE the strategy is constructed (unlike the transformers path, which resizes
        after load). The 128 alignment keeps the vocab evenly divisible across TP shards.
        """
        processor = self._default_tokenizer
        if processor is None:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(self.tokenizer_id, trust_remote_code=True)
            self._default_tokenizer = processor
        tokenizer = getattr(processor, 'tokenizer', processor)
        added = tokenizer.add_special_tokens({'additional_special_tokens': list(new_special_tokens)})
        if added:
            text_config = self.hf_config.get_text_config()
            new_vocab_size = ((len(tokenizer) + 127) // 128) * 128
            if text_config.vocab_size < new_vocab_size:
                text_config.vocab_size = new_vocab_size

    def _initialize_new_special_token_embeddings(self) -> None:
        """Re-initialize the zero-filled rows of the just-added tokens on the sharded modules.

        Uses mcore's own ``init_method`` with an all-zero-row mask, per TP shard and PP stage
        (missing modules yield None), skipping tied output layers; runs before the optimizer
        group snapshots the parameters.
        """
        from twinkle.utils import deep_getattr
        for chunk in self.model:
            model = chunk
            while hasattr(model, 'module'):  # unwrap DDP / Float16Module down to the mcore model
                model = model.module
            init_method = model.config.init_method
            if hasattr(model, 'language_model'):  # multimodal: the decoder carries the embeddings
                model = model.language_model
            for key in ('embedding.word_embeddings', 'output_layer'):
                if key == 'output_layer' and model.share_embeddings_and_output_weights:
                    continue  # tied to the input embedding, already initialized above
                module = deep_getattr(model, key)
                if module is None:
                    continue  # this pipeline stage does not carry the module
                initialize_mask = (module.weight == 0).all(dim=-1)
                num_to_initialize = int(initialize_mask.sum().item())
                if num_to_initialize == 0:
                    continue
                from megatron.core import mpu
                if mpu.get_data_parallel_rank() == 0:
                    logger.info(f'Initializing {num_to_initialize} zero rows of {key!r} for new special tokens.')
                tensor = module.weight.new_empty(num_to_initialize, module.weight.shape[1])
                module.weight.data[initialize_mask] = init_method(tensor)
                if getattr(module.weight, 'main_param', None) is not None:
                    module.weight.main_param.copy_(module.weight.view(-1))

    def _should_bind_device_id_for_process_group(self, backend: str) -> bool:
        # Keep NCCL's device binding behavior, but avoid binding HCCL's default
        # PG so Megatron's later Gloo DP groups stay decoupled on NPU.
        return backend == 'nccl'

    def _construct_default_optimizer_group(self):
        return MegatronOptimizerGroup(
            loss_instance=CrossEntropyLoss(reduction='sum'),
            template=Template(self.tokenizer_id),
            processor=InputProcessor(self.device_mesh, framework='megatron'),
            _device_mesh=self.device_mesh,
        )

    def _lazy_wrap_model(self):
        if not self._model_wrapped:
            self.model = self.strategy.wrap_model(self.model)
            self._model_wrapped = True

    def _maybe_apply_router_replay(self):
        """Lazily apply the (global, idempotent) dispatcher patch once, when routing replay is
        enabled; deferred so non-MoE or replay-disabled runs never touch it."""
        if not self._router_replay_enabled or self._router_replay_applied:
            return
        RouterReplay.apply_router_replay_patch()
        self._router_replay_applied = True

    def _lazy_finish_param_config(self):
        if self._finish_config:
            return
        self._finish_config = True
        optimizer = self.optimizer_group[self._get_default_group()].optimizer
        self.strategy.finish_param_config(self.model, optimizer)

    def _get_default_group(self):
        """Get the only group has optimizer, else return the default one"""
        if len(self.optimizer_group) == 1:
            return next(iter(self.optimizer_group))
        return self.active_group

    @staticmethod
    def _slice_value_for_microbatch(value, mb_start: int, mb_end: int, micro_batch_size: int):
        """Recursively slice a value for microbatch processing.

        Handles nested dicts (e.g. ref_outputs: {"logps": tensor}) by recursively slicing
        internal tensors; scalars and non-sliceable values pass through as-is.
        """
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] > micro_batch_size:
            return value[mb_start:mb_end]
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > micro_batch_size:
            return value[mb_start:mb_end]
        elif isinstance(value, (list, tuple)) and len(value) > micro_batch_size:
            return value[mb_start:mb_end]
        elif isinstance(value, dict):
            # Recursively slice dict values (e.g., ref_outputs: {"logps": tensor})
            return {
                k: MegatronModel._slice_value_for_microbatch(v, mb_start, mb_end, micro_batch_size)
                for k, v in value.items()
            }
        else:
            # Scalars, small tensors, or non-sliceable values pass through as-is
            return value

    @remote_function()
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        raise NotImplementedError('Megatron only supports `forward_backward` and `forward_only`')

    # MTP training: thin shells over the MtpTrainingState instance; the same-named methods stay
    # so subclass overrides and duck-typed callers (``getattr(model, 'pop_mtp_metrics')``) keep resolving.
    def _mtp_training_enabled(self, *, forward_only: bool, task: str, disable_lora: bool) -> bool:
        return self._mtp.training_enabled(self, forward_only=forward_only, task=task, disable_lora=disable_lora)

    @remote_function(collect='last_pp_first', lazy_collect=False)
    def pop_mtp_metrics(self, loss_scale: Optional[float] = None) -> dict[str, float]:
        """Drain megatron's MTP loss tracker; see :meth:`MtpTrainingState.pop`."""
        return self._mtp.pop(loss_scale)


    # sync=True orders the workers for PP's NCCL collectives; lazy_collect follows the global default
    # (True) so the return keeps the Future protocol. Note the sync collect itself still materializes
    # on the calling side -- the handle is a value wrapper there, resolved like any other.
    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict, sync=True)
    @timed
    def forward_only(self,
                     *,
                     inputs: Union[InputFeature, List[InputFeature], List[Trajectory]],
                     micro_batch_size: Optional[int] = None,
                     **kwargs):
        """Forward pass without gradient computation.

        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments.

        Returns:
            Model outputs.
        """
        return self.forward_backward(inputs=inputs, micro_batch_size=micro_batch_size, forward_only=True, **kwargs)

    def generate(self, *args, **kwargs):
        """Refuse in-place generation rather than approximate it.

        A ``TransformersSampler`` facade forwards ``generate`` to the model it wraps. That is sound for a
        ``TransformersModel`` whose weights an inference engine can read, but not here: Megatron keeps
        weights sharded across TP/PP ranks under Megatron's own parameter names, a layout no sampling
        engine understands. Failing loudly beats silently generating from a partial, renamed view.
        """
        raise NotImplementedError(
            'MegatronModel cannot generate in place: its weights are sharded across TP/PP ranks under '
            'Megatron parameter names, which no inference engine reads. Use a dedicated sampler '
            '(vLLM/SGLang/Transformers) that loads its own copy of the weights.')

    def generate_stream(self, *args, **kwargs):
        """Streaming counterpart of :meth:`generate`, refused for the same reason.

        Deliberately a plain method rather than a generator function: the refusal must fire when the method
        is called, not only when a returned iterator is consumed, so a facade surfaces it at the call site.
        """
        raise NotImplementedError(
            'MegatronModel cannot generate in place: its weights are sharded across TP/PP ranks under '
            'Megatron parameter names, which no inference engine reads. Use a dedicated sampler '
            '(vLLM/SGLang/Transformers) that loads its own copy of the weights.')

    @remote_function(collect='mean')
    @timed
    def calculate_loss(self, **kwargs):
        raise NotImplementedError('Megatron only supports `forward_backward` and `forward_only`')

    @remote_function()
    def backward(self, **kwargs):
        raise NotImplementedError('Megatron only supports `forward_backward` and `forward_only`')

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict, sync=True)
    @timed
    def forward_backward(self,
                         *,
                         inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         micro_batch_size: Optional[int] = None,
                         loss_key: Optional[str] = None,
                         loss_weight: float = 1.0,
                         finalize_gradients: bool = True,
                         **kwargs):
        """Combined forward and backward pass using Megatron's scheduler.

        sync=True is required in Ray mode: pipeline parallel's NCCL P2P needs all ranks to enter
        simultaneously. ``get_forward_backward_func()`` picks the schedule (1F1B / interleaved /
        no-pipeline) and accumulates gradients across microbatches.

        Args:
            inputs: A single batch dict, a list of batch dicts (num_microbatches=len(inputs)),
                or an iterator yielding batch dicts.
            micro_batch_size: split and trains by `micro_batch_size`

        Returns:
            Average loss value across all microbatches.
        """
        self._lazy_wrap_model()
        self._lazy_finish_param_config()
        from functools import partial
        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel import get_forward_backward_func

        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        disable_lora = kwargs.pop('disable_lora', False)
        temperature = float(kwargs.pop('temperature', 1.0))
        forward_only = kwargs.pop('forward_only', False)
        # Per-call opt-in to skip the configured loss and return logps only (an empty-loss probe:
        # a KTO z_KL / adapter-disabled reference batch carries no label/ref_logps, which the loss
        # would reject). eval/training omit it; megatron's calculate_loss is a raise, so eval_loss
        # depends on forward_only computing it here.
        compute_loss = kwargs.pop('compute_loss', True)
        return_logits = kwargs.pop('return_logits', False)
        task = kwargs.pop('task', 'causal_lm')
        # MoE routing replay (R2/R3): the driver passes the MODE scalar, resolved below to this
        # backend's action. Popped before loss_extra_kwargs_per_mb so it never reaches the loss.
        router_replay_mode = kwargs.pop('router_replay_mode', None)
        router_replay_manual_cleanup = kwargs.pop('router_replay_manual_cleanup', False)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance = optimizer_config.get_loss(loss_key)
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        if forward_only and not finalize_gradients:
            raise ValueError('forward_only does not support finalize_gradients=False')
        if not forward_only and not finalize_gradients and self.strategy.uses_fsdp(self.strategy.ddp_config):
            ddp_config = getattr(self.model[0], 'ddp_config', None)
            sharding_strategy = getattr(
                ddp_config,
                'data_parallel_sharding_strategy',
                self.strategy.ddp_config.get('data_parallel_sharding_strategy'),
            )
            if sharding_strategy in {'optim_grads', 'optim_grads_params'}:
                raise ValueError(
                    'Deferred backward is not supported by Megatron-FSDP with gradient sharding '
                    f'({sharding_strategy}); use data_parallel_sharding_strategy="no_shard" or "optim".')
        completing_pending = False
        if not forward_only:
            completing_pending = optimizer_config.prepare_backward_transaction(finalize_gradients, loss_weight)
        transactional = not forward_only and (not finalize_gradients or completing_pending)
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        if isinstance(inputs, dict):
            inputs = [inputs]
        local_batch_normalizer = loss_instance.batch_normalizer(inputs) if transactional else None
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'

        if micro_batch_size is None:
            # Compatible with DPO
            micro_batch_size = min(2, len(inputs))
        unwrapped_model = self.strategy.unwrap_model(self.model)[0]
        inputs = processor(
            inputs,
            micro_batch_size=micro_batch_size,
            variable_seq_lengths=self.variable_seq_lengths,
            attention_mask_type=getattr(unwrapped_model.config, 'attention_mask_type', None),
        )

        # Get parallelism settings for sequence padding and splitting
        cp_size = self.device_mesh.cp_world_size
        # MoE routing replay orchestration of this call (a null object when disabled or mode-less);
        # the per-microbatch dance lives in forward_step_func (pre_mb / post_mb).
        rr = RouterReplay(enabled=self._router_replay_enabled, mode=router_replay_mode,
                          forward_only=forward_only, cp_size=cp_size,
                          apply_patch=self._maybe_apply_router_replay)
        # Check actual sequence_parallel setting from model config
        # Bridge may auto-enable sequence_parallel for MoE models
        if self.variable_seq_lengths:
            seq_length = max(inp['input_ids'].shape[-1] for inp in inputs)
        else:
            original_seq_length = inputs[0]['input_ids'].shape[1] * (cp_size or 1)
            if cp_size > 1:
                divisor = 2 * cp_size
            elif self.strategy.sequence_parallel and self.device_mesh.tp_world_size > 1:
                divisor = self.device_mesh.tp_world_size
            else:
                divisor = 1

            if divisor > 1 and original_seq_length % divisor != 0:
                seq_length = original_seq_length + (divisor - original_seq_length % divisor)
            else:
                seq_length = original_seq_length

        num_microbatches = len(inputs)
        self._mtp.note_num_microbatches(num_microbatches)
        global_normalizer = None
        if transactional:
            local_normalizer = num_microbatches if local_batch_normalizer is None else local_batch_normalizer
            cp_size_for_normalizer = self.device_mesh.cp_world_size or 1
            global_normalizer = torch.as_tensor(
                float(local_normalizer) / cp_size_for_normalizer,
                device=Platform.get_local_device(),
                dtype=torch.float32,
            )
            dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            if dist.get_world_size(dp_cp_group) > 1:
                dist.all_reduce(global_normalizer, op=dist.ReduceOp.SUM, group=dp_cp_group)
        loss_extra_kwargs_per_mb = []
        if num_microbatches <= 1:
            loss_extra_kwargs_per_mb = [kwargs]
        else:
            # Only support extra kwargs length==total_batch_size
            for mb_idx in range(num_microbatches):
                mb_start = mb_idx * micro_batch_size
                mb_end = mb_start + micro_batch_size
                mb_kwargs = {
                    key: self._slice_value_for_microbatch(value, mb_start, mb_end, micro_batch_size)
                    for key, value in kwargs.items()
                }
                loss_extra_kwargs_per_mb.append(mb_kwargs)

        _mb_counter = [0]  # mutable counter for closure
        feed_mtp = self._mtp_training_enabled(forward_only=forward_only, task=task, disable_lora=disable_lora)

        def post_loss_function(output_tensor, inputs, logps, unpacked_logits=None, entropies=None, embeddings=None):
            mb_idx = _mb_counter[0]
            _mb_counter[0] += 1
            current_kwargs = loss_extra_kwargs_per_mb[mb_idx % len(loss_extra_kwargs_per_mb)]
            if embeddings is not None:
                outputs = ModelOutput(embeddings=embeddings)
            else:
                logits = unpacked_logits if unpacked_logits is not None else output_tensor
                outputs = ModelOutput(logits=logits, logps=logps)
                if entropies is not None:
                    outputs['entropies'] = entropies
            if not compute_loss:
                # Empty-loss probe: skip the loss but still route a zero loss through reduce_loss so
                # the logps reach its returned dict unchanged (the transformers forward_only contract).
                zero = torch.zeros((), device=Platform.get_local_device(), dtype=torch.float32)
                return self.strategy.reduce_loss(zero, zero.clone(), output_tensor, logps)
            result = loss_instance(inputs, outputs, **current_kwargs)
            if unpacked_logits is not None:
                outputs.pop('logits', None)
                del unpacked_logits
            losses = result['loss']
            counts = result['num_tokens']
            if not counts:
                # Later will gather this value, so it becomes:
                # 1. SUM loss: gather_sum(local_num_tokens) = global_num_tokens
                # 2. PER TOKEN MEAN loss: gather_sum(1 * gradient_accumulation_steps )
                #       = gradient_accumulation_steps * world_size
                # Then, grad will divided by this value:
                # 1. SUM loss: (global_sum_grad) / (global_num_tokens) = global_sum_grad/global_num_tokens
                # 2. PER TOKEN MEAN loss: (gather_sum(per_token_grad * gradient_accumulation_steps))
                #       / (gradient_accumulation_steps  * world_size ) = avg_per_token_grad
                counts = torch.tensor(1, device=losses.device)
            reduced = self.strategy.reduce_loss(losses, counts, output_tensor, logps)
            if result.get('channel_loss') is not None:
                reduced[2]['channel_loss'] = result['channel_loss']
            return reduced

        # Define forward step function for Megatron
        # forward_step_func(data_iterator, model) -> (output_tensor, partial(loss_func))
        def forward_step_func(data_iterator, model):
            batch = next(data_iterator)
            labels = batch.pop('labels', None)
            loss_scale = batch.pop('loss_scale', None)
            channel = batch.pop('channel', None)
            # Not a model argument; restored below so the loss can read it.
            completion_mask = batch.pop('completion_mask', None)
            # The whole-model routing to replay (R3, delivered by the sampler): popped so it never
            # reaches model(**model_kwargs); the replay dance below slices it to the local shard.
            routed_experts_mb = batch.pop('routed_experts', None)
            # Per-token mask selecting which tokens replay vs recompute natively; not a model argument.
            replay_mask_mb = batch.pop('replay_mask', None)
            # MTP joint training: labels stay out of the model call so the main loss stays external;
            # the MTP heads get next-token targets on a separate keyword, never left in ``batch``.
            model_kwargs = batch
            if feed_mtp and labels is not None:
                model_kwargs = dict(batch)
                model_kwargs['mtp_labels'] = labels
            unwrapped_model = self.strategy.unwrap_model([model])[0]
            rr.pre_mb(unwrapped_model, batch, routed_experts_mb, replay_mask_mb)
            if disable_lora and isinstance(unwrapped_model, PeftModel):
                with unwrapped_model.disable_adapter():
                    output_tensor = model(**model_kwargs)
            else:
                output_tensor = model(**model_kwargs)
            rr.post_mb(unwrapped_model, batch)

            batch['labels'] = labels
            if loss_scale is not None:
                batch['loss_scale'] = loss_scale
            if channel is not None:
                batch['channel'] = channel
            if completion_mask is not None:
                batch['completion_mask'] = completion_mask
            logps = None
            unpacked_logits = None
            entropies = None
            embeddings = None
            _loss_instance = loss_instance
            is_last_pp = mpu.is_pipeline_last_stage(False, unwrapped_model.vp_stage)
            try:
                if task == 'embedding':
                    # MegatronEmbeddingPatch already pooled output to [n_seqs, hidden] on last PP stage.
                    if is_last_pp:
                        embeddings = output_tensor
                elif task in ('seq_cls', 'reranker', 'generative_reranker'):
                    # mcore-bridge emits per-token head output; pool the last valid token per sequence
                    # here (transformers does this inside *ForSequenceClassification). These losses read
                    # logits (require_logps=False); the pool runs regardless of labels because a
                    # preference row carries no next-token labels at all.
                    if is_last_pp:
                        _packed = batch.get('packed_seq_params')
                        # CP reconstruct first (like the causal_lm branch), then take the last token.
                        cu_seqlens_q = getattr(_packed, 'cu_seqlens_q', None) if _packed is not None else None
                        output_tensor = processor.postprocess_tensor_cp(output_tensor, cu_seqlens=cu_seqlens_q)
                        if _packed is None:
                            _am = batch.get('attention_mask')
                            if _am is None:
                                _am = batch.get('attention_mask_2d')
                            if _am is None:
                                # No mask in the batch (the collator right-pads without emitting one): pool
                                # the final position, exactly what transformers does when handed no mask.
                                _last_idx = torch.full((output_tensor.shape[0], ),
                                                       output_tensor.shape[1] - 1,
                                                       dtype=torch.long,
                                                       device=output_tensor.device)
                            else:
                                # attention_mask_2d compatibility: a 4D mask collapses to per-row validity.
                                if _am.dim() > 2:
                                    _am = (~_am).sum(dim=(1, 2)) > 0
                                _last_idx = processor._last_valid_indices(_am.long())
                            unpacked_logits = output_tensor[
                                torch.arange(output_tensor.shape[0], device=output_tensor.device), _last_idx]
                        else:
                            # Packed: one row [1, total, C]; each segment ends at cu_seqlens_q[i]+seq_lens[i]-1.
                            _n = _packed.seq_lens.shape[0]
                            _last_idx = _packed.cu_seqlens_q[:_n] + _packed.seq_lens - 1
                            unpacked_logits = output_tensor[0, _last_idx]
                elif task == 'value':
                    # PPO critic: the same num_labels=1 seq_cls head, but keep its per-token V(s_t)
                    # riding the logits channel ([b, s]); PPOValueLoss masks to response tokens.
                    # Non-packed / non-SP only, mirroring the transformers TransformersValuePatch scope.
                    if is_last_pp and labels is not None:
                        _packed = batch.get('packed_seq_params')
                        cu_seqlens_q = getattr(_packed, 'cu_seqlens_q', None) if _packed is not None else None
                        output_tensor = processor.postprocess_tensor_cp(output_tensor, cu_seqlens=cu_seqlens_q)
                        # [b, s, 1] -> [b, s] (num_labels=1); already per-token, no pooling.
                        unpacked_logits = output_tensor.squeeze(-1) if output_tensor.dim() == 3 else output_tensor
                elif labels is not None and is_last_pp:
                    _loss_require_logps = getattr(_loss_instance, 'require_logps', True)
                    _loss_require_entropy = getattr(_loss_instance, 'require_entropy', False)
                    _packed = batch.get('packed_seq_params')
                    cu_seqlens_q = getattr(_packed, 'cu_seqlens_q', None) if _packed is not None else None
                    if _loss_require_logps:
                        loss_mask = (labels != -100).bool()
                        masked_labels = labels.clone()
                        masked_labels[~loss_mask] = 0
                        output_tensor.div_(temperature)
                        if _loss_require_entropy:
                            logps, entropies = selective_log_softmax(output_tensor, masked_labels, return_entropy=True)
                        else:
                            logps = selective_log_softmax(output_tensor, masked_labels)
                        # Reconstruct full-length tensors from CP-split shards
                        logps = processor.postprocess_tensor_cp(logps, cu_seqlens=cu_seqlens_q)
                        if entropies is not None:
                            entropies = processor.postprocess_tensor_cp(entropies, cu_seqlens=cu_seqlens_q)
                    batch['labels'] = processor.postprocess_tensor_cp(labels, cu_seqlens=cu_seqlens_q)
                    if batch.get('loss_scale') is not None:
                        batch['loss_scale'] = processor.postprocess_tensor_cp(
                            batch['loss_scale'], cu_seqlens=cu_seqlens_q)
                    if completion_mask is not None:
                        # Same index space as labels, so it needs the same CP reassembly.
                        batch['completion_mask'] = processor.postprocess_tensor_cp(
                            completion_mask, cu_seqlens=cu_seqlens_q)
                    if 'position_ids' in batch:
                        pos = batch['position_ids']
                        if pos.dim() == 3:
                            pos = pos[0]  # [2/3, 1, seq] → [1, seq]
                        batch['position_ids'] = processor.postprocess_tensor_cp(pos, cu_seqlens=cu_seqlens_q)
                    # Unpack packed sequences into per-sequence batch format
                    _outputs = {'logps': logps}
                    if entropies is not None:
                        _outputs['entropies'] = entropies
                    if hasattr(_loss_instance, 'require_logits') and _loss_instance.require_logits:
                        _outputs['logits'] = output_tensor
                    batch, _outputs = processor.unpack_packed_sequences(batch, _outputs)
                    logps = _outputs['logps']
                    entropies = _outputs.get('entropies', None)
                    unpacked_logits = _outputs.get('logits', None)
            except Exception as e:
                # Log the rank-attributed failure, then re-raise: swallowing it would let this DP
                # rank skip the PP collectives while the others wait, and a degraded value would
                # mask the failure.
                logger.warning('[nccl_safe] forward_step_func data processing error on rank: '
                               '%s: %s',
                               type(e).__name__, e)
                raise
            return output_tensor, partial(
                post_loss_function,
                inputs=batch,
                logps=logps,
                unpacked_logits=unpacked_logits,
                entropies=entropies,
                embeddings=embeddings,
            )

        # Get Megatron's forward-backward function
        # This automatically selects the right scheduler based on PP config:
        # - PP > 1: forward_backward_pipelining_without_interleaving (or with interleaving if VPP)
        # - PP = 1: forward_backward_no_pipelining
        forward_backward_func = get_forward_backward_func()
        vpp_size = self.device_mesh.vpp_size
        if vpp_size in (0, 1):
            vpp_size = None

        micro_batch_size = inputs[0]['input_ids'].shape[0]
        if vpp_size is None:
            data_iter = iter(inputs)
        else:
            data_iter = [iter(inputs) for _ in range(0, vpp_size)]

        optimizer_config.accumulate_metrics(is_training=not forward_only)

        # Run forward-backward with Megatron's scheduler; the gradient-hook surgery for a
        # transactional / deferred-finalize pass is owned by the strategy.
        rr.arm()
        with self.strategy.transactional_grad_schedule(
                self.model,
                forward_only=forward_only,
                transactional=transactional,
                loss_weight=loss_weight,
                finalize_gradients=finalize_gradients,
                completing_pending=completing_pending,
                global_normalizer=global_normalizer,
                **kwargs):
            with _resolve_task_context(self.model, task):
                losses = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iter,
                    model=self.model,
                    num_microbatches=len(inputs),
                    seq_length=seq_length,
                    micro_batch_size=micro_batch_size,
                    forward_only=forward_only,
                )

        # Extract loss from results (only last PP stage returns non-empty)
        loss = torch.tensor(0.0).to(Platform.get_local_device())
        logits = []
        logps = []
        channel_loss = {}
        count = 0
        if losses:
            for loss_dict in losses:
                if isinstance(loss_dict, dict):
                    if 'loss' in loss_dict:
                        loss += loss_dict['loss']
                    if 'logits' in loss_dict:
                        logits.append(loss_dict['logits'])
                    if 'logps' in loss_dict:
                        logps.append(loss_dict['logps'])
                    if 'num_tokens' in loss_dict:
                        count += loss_dict['num_tokens']
                    for channel, stats in loss_dict.get('channel_loss', {}).items():
                        if channel in channel_loss:
                            channel_loss[channel] = channel_loss[channel] + stats
                        else:
                            channel_loss[channel] = stats
                elif isinstance(loss_dict, torch.Tensor):
                    raise ValueError('Expected loss dict, got tensor')

        # Report the LOCAL token-sum and its denominator (transformers contract): LossMetric's
        # sum branch divides exactly once and its dp_group gather does the DP reduction, so no
        # pre-division or DP pre-average here; mcore normalizes the backward tensor itself.
        num_tokens = count

        # PP > 1: broadcast loss/count from the last stage so every PP rank reports consistently
        # (calculate_metric collects 'last_pp_first').
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            src_rank = mpu.get_pipeline_model_parallel_last_rank()
            pp_group = mpu.get_pipeline_model_parallel_group()
            loss_f = loss.detach().float() if isinstance(loss, torch.Tensor) else torch.tensor(
                float(loss), device=Platform.get_local_device())
            count_f = torch.as_tensor(float(num_tokens), device=loss_f.device, dtype=torch.float32)
            # One broadcast carries both scalars so the collective count stays identical on all stages.
            packed = torch.stack([loss_f, count_f])
            torch.distributed.broadcast(packed, src=src_rank, group=pp_group)
            loss = packed[0]
            num_tokens = packed[1]

        if not forward_only and finalize_gradients:
            optimizer_config.cur_step += 1

        if logps and not self.variable_seq_lengths:
            logps = torch.cat(logps, dim=0)
        elif not logps:
            logps = None
        if logits and not self.variable_seq_lengths:
            logits = torch.cat(logits, dim=0)
        if transactional:
            dp_world_size = self.device_mesh._get_dp_fsdp_world_size()
            loss = loss * (float(loss_weight) * dp_world_size / global_normalizer.clamp(min=1))
            num_tokens = 0
        elif not forward_only and loss_weight != 1.0:
            loss = loss * float(loss_weight)
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().float().item()
        if isinstance(num_tokens, torch.Tensor):
            num_tokens = num_tokens.detach().float().item()
        if not return_logits:
            logits = None
        inputs = processor.unpack_inputs(inputs, task=task)
        model_output = ModelOutput(logits=logits, loss=loss, logps=logps, num_tokens=num_tokens)
        if channel_loss:
            model_output['channel_loss'] = channel_loss
        # R2 RECORD: hand the captured whole-model routing back to the driver (symmetric with
        # TransformersModel.forward's return_outputs['routed_experts']).
        rr.finish(model_output, unwrapped_model.config, num_microbatches, vpp_size,
                  router_replay_manual_cleanup)
        if forward_only:
            optimizer_config.eval_status.inputs = inputs
            optimizer_config.eval_status.outputs = model_output
            optimizer_config.eval_status.forward_kwargs = kwargs
        else:
            status = optimizer_config.train_status
            status.inputs = inputs
            status.outputs = model_output
            status.forward_kwargs = {**kwargs, '_metric_loss_key': loss_key}
            status.loss_reduction = 'mean' if transactional else getattr(loss_instance, 'reduction', 'mean')
            if transactional:
                metric_step = optimizer_config.cur_step - 1 if finalize_gradients else optimizer_config.cur_step
                optimizer_config.accumulate_metrics(True, include_default=False, metric_step=metric_step)
                status.inputs = None
                status.outputs = None
                total_loss = optimizer_config.record_transaction_loss(model_output.get('loss', 0.0),
                                                                     finalize_gradients)
                if finalize_gradients:
                    status.inputs = {}
                    status.outputs = ModelOutput(loss=total_loss)
                    status.forward_kwargs = {}
                    status.loss_reduction = 'mean'
                    return ModelOutput(loss=total_loss)
        return model_output

    @remote_function(dispatch='all')
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type: int = 2, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        self.optimizer_group[adapter_name].require_finalized('clip gradients')
        # Megatron optimizer will cover this function.
        return 0

    @remote_function(dispatch='all')
    def step(self, **kwargs):
        """Optimizer step (dispatch='all' so every worker executes it). For non-DDP models
        (e.g. PEFT/LoRA) gradients are NOT synchronized across DP ranks; each replica trains
        independently on its own data."""
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.require_finalized('step the optimizer')

        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return

        optimizer = optimizer_config.optimizer
        assert optimizer is not None, 'Set optimizer correctly before stepping'
        # Megatron optimizer step() returns (success, grad_norm, num_zeros)

        optim_params = kwargs.pop('optim_params', {})
        if optim_params:
            for group in optimizer.param_groups:
                group['lr'] = optim_params['lr']
                if group['weight_decay'] > 0.0 and optim_params.get('weight_decay', None) is not None:
                    group['weight_decay'] = optim_params['weight_decay']
                if optim_params.get('eps') is not None:
                    group['eps'] = optim_params['eps']
                if optim_params.get('betas') is not None:
                    group['betas'] = optim_params['betas']

        success, grad_norm, num_zeros = optimizer.step()
        # Store grad_norm for later retrieval
        if grad_norm is not None:
            grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
        optimizer_config._last_grad_norm = float(grad_norm) if grad_norm is not None else 0.0
        optimizer_config._last_step_success = success

    def _is_model_ddp_wrapped(self) -> bool:
        """Check if model is wrapped with DDP.

        Returns:
            True if model is wrapped with DDP (either Megatron DDP, LoRA DDP, or PyTorch DDP).
        """
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        return isinstance(self.model[0], (MegatronDDP, TorchDDP))

    @remote_function(dispatch='all')
    def zero_grad(self, **kwargs):
        """Zero gradients. For DDP-wrapped models the gradient buffer is ALWAYS zeroed (the
        schedule tracks accumulation through it); do_grad_sync only gates optimizer.zero_grad()."""
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.require_finalized('zero gradients')

        # For DDP-wrapped models, ALWAYS zero the gradient buffer
        # This is essential because Megatron's forward_backward_func uses
        # the buffer's state to track gradient accumulation
        # (self.model is a list of chunks; zero_grad_buffer lives on each chunk)
        if self._is_model_ddp_wrapped():
            for model_chunk in self.model:
                if hasattr(model_chunk, 'zero_grad_buffer'):
                    model_chunk.zero_grad_buffer()

        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return

        optimizer = optimizer_config.optimizer
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

    @remote_function()
    def lr_step(self, **kwargs):
        """Learning rate scheduler step.

        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.require_finalized('step the learning-rate scheduler')

        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return

        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            # Megatron's OptimizerParamScheduler.step() requires increment argument
            increment = kwargs.pop('increment', 1)
            lr_scheduler.step(increment=increment)

    @remote_function(dispatch='all')
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...], torch.Tensor]],
                 **kwargs):
        """Register a loss used by Megatron's scheduler-side loss callback.

        Args:
            loss_cls: Loss class, instance, or registered name.
            **kwargs: Constructor arguments plus optional ``loss_key`` and ``adapter_name``.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        loss_key = kwargs.pop('loss_key', None)
        optimizer_config = self.optimizer_group[adapter_name]
        loss = construct_class(loss_cls, Loss, twinkle.loss, **kwargs)
        # Gather over the DP group, not WORLD: under PP the loss runs only on the last
        # pipeline stage, so a WORLD-group all-gather would deadlock on earlier-stage ranks.
        if getattr(loss, 'process_group', None) is None and getattr(optimizer_config, '_dp_group', None) is not None:
            loss.process_group = optimizer_config._dp_group
        optimizer_config.set_loss(loss, loss_key=loss_key)

    @remote_function(dispatch='all')
    def empty_cache(self, **kwargs):
        """Release the caching allocator's unused blocks on every worker; backend-agnostic
        through torch_util, mirrors TransformersModel.empty_cache."""
        torch_util.empty_cache()

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
        loss_key = kwargs.pop('loss_key', None)
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['device_mesh'] = self.device_mesh
        kwargs['process_group'] = optimizer_config._dp_group
        if is_training is None or is_training is True:
            metric = construct_class(metric_cls, Metric, twinkle.metric, **kwargs)
            metric._twinkle_loss_key = loss_key
            optimizer_config.train_status.metrics.append(metric)
        if not is_training:
            metric = construct_class(metric_cls, Metric, twinkle.metric, **kwargs)
            metric._twinkle_loss_key = loss_key
            optimizer_config.eval_status.metrics.append(metric)

    @remote_function(dispatch='all')
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str], **kwargs):
        """Set optimizer.

        Args:
            optimizer_cls: Optimizer class or string name.
                - Standard PyTorch optimizers: 'AdamW', 'Adam', 'SGD', etc.
                - 'MegatronDistributed': Use Megatron's distributed optimizer
            **kwargs: Additional arguments.
                - For standard optimizers: lr, weight_decay, etc.
                - For MegatronDistributed: use_distributed_optimizer, clip_grad, etc.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not self._model_wrapped:
            self.model = self.strategy.wrap_model(self.model)
            self._model_wrapped = True

        # Check if requesting Megatron distributed optimizer
        if not optimizer_cls or optimizer_cls in ('MegatronOptimizer', 'default', 'Adam'):
            optimizer_config.optimizer = self._create_megatron_optimizer(**kwargs)  # noqa
        else:
            raise NotImplementedError(
                f'Unsupported optimizer: {optimizer_cls}, only support MegatronOptimizer currently.')

    @remote_function(collect='last_pp_first', lazy_collect=False)
    @timed
    def calculate_metric(self, is_training, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.require_finalized('calculate metrics')
        return optimizer_config.calculate_metrics(is_training)

    def _create_megatron_optimizer(self, **kwargs):
        """Create Megatron's distributed optimizer (shards optimizer state across DP replicas).

        Args:
            **kwargs: Optimizer configuration options (lr, weight_decay, use_distributed_optimizer,
                clip_grad, bf16/fp16, adam_* ...).
                - config_overrides: Optional mcore ``{ParamKey: ParamGroupOverride}`` mapping for
                    per-parameter-group overrides (e.g. differential LR for vision / aligner params);
                    forwarded verbatim so callers describe their own parameter grouping.
                - param_groups_spec: Backend-neutral ordered grouping rules (per-tower LR, LoRA+);
                    translated into ``config_overrides`` as exclusive ``with_name_predicate`` rules
                    (first-match-wins) layered on top of the standard bias / wd-skip defaults.

        Returns:
            MegatronOptimizer instance.
        """
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

        # Build optimizer config
        lr = kwargs.pop('lr', 1e-4)
        self.use_distributed_optimizer: bool = kwargs.pop('use_distributed_optimizer', self.use_distributed_optimizer)
        # Pop before building OptimizerConfig: these are get_megatron_optimizer args, not config fields.
        config_overrides = kwargs.pop('config_overrides', None)
        param_groups_spec = kwargs.pop('param_groups_spec', None)

        opt_config = OptimizerConfig(
            optimizer=kwargs.pop('optimizer', 'adam'),
            lr=lr,
            min_lr=kwargs.pop('min_lr', 0.0),
            weight_decay=kwargs.pop('weight_decay', 0.01),
            adam_beta1=kwargs.pop('adam_beta1', 0.9),
            adam_beta2=kwargs.pop('adam_beta2', 0.999),
            adam_eps=kwargs.pop('adam_eps', 1e-8),
            clip_grad=kwargs.pop('clip_grad', 1.0),
            bf16=kwargs.pop('bf16', self.mixed_precision == 'bf16'),
            fp16=kwargs.pop('fp16', self.mixed_precision == 'fp16'),
            use_distributed_optimizer=self.use_distributed_optimizer,
            overlap_param_gather=kwargs.pop('overlap_param_gather', False),
            log_num_zeros_in_grad=kwargs.pop('log_num_zeros_in_grad', False),
            **kwargs,
        )

        if param_groups_spec:
            # Layer the spec's rules on top of the standard defaults (bias / length-1 wd skip) --
            # or a caller-supplied config_overrides -- so per-tower / LoRA+ LRs do not drop the skip.
            from megatron.core.optimizer import get_standard_config_overrides
            from twinkle.module.optimizer import build_param_group_overrides, resolve_megatron_tower_prefixes
            base_overrides = config_overrides if config_overrides is not None else get_standard_config_overrides(
                opt_config)
            spec_overrides = build_param_group_overrides(
                param_groups_spec,
                model_chunks=self.model,
                tower_prefixes=resolve_megatron_tower_prefixes(self.model),
                base_lr=opt_config.lr,
                base_min_lr=opt_config.min_lr)
            config_overrides = {**base_overrides, **spec_overrides}

        # Ensure each model chunk has ddp_config attached (required by Megatron optimizer)
        model_chunks = self.model
        for model_chunk in model_chunks:
            assert hasattr(model_chunk, 'ddp_config')
        optimizer = get_megatron_optimizer(
            config=opt_config,
            model_chunks=model_chunks,
            config_overrides=config_overrides,
        )
        return optimizer

    def _create_megatron_scheduler(self, optimizer, lr_decay_steps, max_lr=1e-4, **kwargs):
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
        return OptimizerParamScheduler(
            optimizer,
            init_lr=kwargs.pop('init_lr', 0.0),
            max_lr=max_lr,
            min_lr=kwargs.pop('min_lr', 0.0),
            lr_warmup_steps=kwargs.pop('lr_warmup_steps', 0),
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=kwargs.pop('lr_decay_style', 'cosine'),
            start_wd=kwargs.pop('start_wd', 0.01),
            end_wd=kwargs.pop('end_wd', 0.01),
            wd_incr_steps=lr_decay_steps,
            wd_incr_style=kwargs.pop('wd_incr_style', 'constant'),
            **kwargs,
        )

    def _get_trainable_parameters(self, adapter_name: str = _default_adapter_name) -> Dict[str, nn.Parameter]:
        """Get trainable parameters.

        Args:
            adapter_name: Name of adapter.

        Returns:
            Dict mapping parameter names to parameters.
        """
        is_default = adapter_name == _default_adapter_name
        # Same pattern family as the transformers backend: LoRA deltas plus the fully trained
        # modules_to_save copies -- missing it here would freeze those copies out of the optimizer.
        pattern = re.compile(rf'\.(?:lora_\w+|modules_to_save)\.{re.escape(adapter_name)}')

        params = {}
        model = self.strategy.unwrap_model(self.model)
        for _model in model:
            for name, param in _model.named_parameters():
                if param.requires_grad and (pattern.search(name) or is_default):
                    params[name] = param
        return params

    @remote_function(dispatch='all')
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str], **kwargs):
        """Set learning rate scheduler.

        Args:
            scheduler_cls: Scheduler class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer = optimizer_config.optimizer
        if not scheduler_cls or scheduler_cls in ('OptimizerParamScheduler', 'default'):
            optimizer_config.lr_scheduler = self._create_megatron_scheduler(optimizer, **kwargs)  # noqa
        else:
            raise NotImplementedError(
                f'Unsupported scheduler: {scheduler_cls}, only support OptimizerParamScheduler currently.')

    @remote_function(dispatch='all')
    @timed
    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)

    @remote_function(dispatch='all', collect='first', sync=True, timeout=3600)
    @timed
    def save(self,
             name: Optional[str] = None,
             output_dir: Optional[str] = None,
             interval: int = 1,
             save_optimizer: bool = False,
             merge_lora: bool = False,
             **kwargs):
        """Save model checkpoint.

        Always saves HF-format model weights. When ``save_optimizer`` is True,
        additionally saves optimizer / lr_scheduler / RNG state in mcore
        distributed-checkpoint format so that training can be resumed later.

        Args:
            name: Checkpoint name. Defaults to ``'checkpoint-step-{cur_step}'``.
            output_dir: Output directory. Defaults to ``'output'``.
            interval: Save each *interval* steps.
            save_optimizer: If True, save optimizer + lr_scheduler + RNG state
                alongside the HF weights for checkpoint resumption.
            merge_lora: If True, merge LoRA adapters into base weights and save
                the full merged model instead of PEFT adapter format. The merge
                is reversed after saving so training can continue.
            **kwargs: Additional arguments forwarded to the underlying save
                methods (e.g. ``adapter_name``, ``safe_serialization``, and ``max_shard_size``).
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        safe_serialization = kwargs.pop('safe_serialization', True)
        max_shard_size = kwargs.pop('max_shard_size', '5GB')
        save_total_limit = kwargs.pop('save_total_limit', None)
        async_save = kwargs.pop('async_save', False)
        if async_save and save_total_limit is not None:
            # The rolling delete cannot tell whether an in-flight async write has finished, so
            # the two would race on the same directory.
            raise ValueError('async_save is incompatible with save_total_limit: the rotation cannot tell '
                             'whether a background write has completed. Disable one of the two.')
        if not safe_serialization:
            raise NotImplementedError('Megatron HF-format checkpoints require safe_serialization=True.')
        optimizer_config = self.optimizer_group[adapter_name]
        if optimizer_config.cur_step % interval != 0:
            return

        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)

        is_lora = (optimizer_config.adapter_name != _default_adapter_name)

        if merge_lora and is_lora:
            self._merge_lora_adapters(optimizer_config.adapter_name)
            self._save_hf_format(checkpoint_dir, _default_adapter_name, max_shard_size=max_shard_size)
            self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)
            self._unmerge_lora_adapters()
        else:
            self._save_hf_format(checkpoint_dir, optimizer_config.adapter_name, max_shard_size=max_shard_size)
            self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)

        # Optionally save mcore optimizer state (for training resumption).
        async_request = None
        if save_optimizer:
            async_request = self._save_mcore_optimizer(
                checkpoint_dir,
                optimizer_config=optimizer_config,
                async_save=async_save,
                **kwargs,
            )
        trainer_state = {
            'checkpoint_version': 1,
            'cur_step': optimizer_config.cur_step,
            'consumed_train_samples': kwargs.get('consumed_train_samples', 0),
            'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
            'component_states': kwargs.get('component_states', {}),
        }
        state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        if async_request is not None:
            # trainer_state.json + the iteration tracker are what a resume reads, so they ride the
            # async finalize and are written only after every rank's shard write landed.
            def _finalize_async_checkpoint() -> None:
                rank = dist.get_rank()
                if rank == 0:
                    with open(state_path, 'w') as f:
                        json.dump(trainer_state, f, indent=2)
                    tracker_path = os.path.join(checkpoint_dir, 'latest_checkpointed_iteration.txt')
                    with open(tracker_path, 'w') as f:
                        f.write(str(optimizer_config.cur_step))
                    copy_checkpoint_args(output_dir, checkpoint_dir)

            async_request.finalize_fns.append(_finalize_async_checkpoint)
            self._mcore_async_calls_queue().schedule_async_request(async_request)
            # Join any earlier write that has since completed (mirrors v4's per-step finalize probe).
            self._maybe_finalize_mcore_async_saves()
            return checkpoint_dir
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and save_optimizer:
            with open(state_path, 'w') as f:
                json.dump(trainer_state, f, indent=2)

        # Copy the run metadata and rotate only after every checkpoint component has completed. Rank 0
        # removes old directories; the final barrier keeps workers from continuing while rotation is in flight.
        copy_checkpoint_args(output_dir, checkpoint_dir)
        rotate_checkpoints(output_dir, checkpoint_dir, save_total_limit)
        if dist.is_initialized():
            dist.barrier()

        return checkpoint_dir

    @remote_function(dispatch='all')
    @timed
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        """Load model weights, and optionally optimizer / scheduler / RNG state.

        Args:
            name: Checkpoint name or HuggingFace Hub model id.
            output_dir: Parent directory that contains the checkpoint folder.
                If None **and** ``load_optimizer`` is False, downloads from Hub.
            load_optimizer: If True, restore optimizer, lr_scheduler and RNG state
                from the mcore sub-checkpoint for training resumption.
            **kwargs: Additional arguments (``adapter_name``, ``no_load_optim``,
                ``no_load_rng``, etc.).
        """
        resume = kwargs.pop('load_optimizer', False)
        if output_dir is not None:
            checkpoint_dir = os.path.join(output_dir, name)
        elif os.path.exists(name):
            checkpoint_dir = name
        elif not resume:
            # load from hub
            token = kwargs.pop('token', None)
            checkpoint_dir = HubOperation.download_model(name, token=token)
        else:
            checkpoint_dir = os.path.join('output', name)

        adapter_name = kwargs.pop('adapter_name', self._get_default_group())

        if resume:
            self._load_mcore_optimizer(
                checkpoint_dir,
                adapter_name=adapter_name,
                **kwargs,
            )
        else:
            bridge = self.strategy.bridge
            bridge.load_weights(
                self.strategy.unwrap_model(self.model),
                checkpoint_dir,
                peft_format=(adapter_name != _default_adapter_name),
            )
            optimizer_config = self.optimizer_group.get(adapter_name)
            if optimizer_config is not None and optimizer_config.optimizer is not None:
                optimizer_config.optimizer.reload_model_params()

        if dist.is_initialized():
            dist.barrier()

    @remote_function(dispatch='all')
    def reload_initial_weights(self, **kwargs):
        """Reload the base model weights from the original model path, used by full-parameter
        server deployments after a tenant releases the (exclusive) model."""
        bridge = self.strategy.bridge
        bridge.load_weights(
            self.strategy.unwrap_model(self.model),
            self._model_path,
            peft_format=False,
        )
        # Drop any leftover gradients from the previous tenant so they cannot
        # leak into the next tenant's first optimizer step.
        if self._is_model_ddp_wrapped():
            for model_chunk in self.model:
                if hasattr(model_chunk, 'zero_grad_buffer'):
                    model_chunk.zero_grad_buffer()
        for _model in self.strategy.unwrap_model(self.model):
            for param in _model.parameters():
                param.grad = None
        if dist.is_initialized():
            dist.barrier()

    # lazy_collect=False: the driver consumes the returned training-state dict immediately
    # (loop.resume subscripts state['cur_step']); a lazy callable default would break it.
    @remote_function(dispatch='all', collect='first', sync=True, lazy_collect=False, timeout=3600)
    @timed
    def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]

        self.load(checkpoint_dir, load_optimizer=not resume_only_model, adapter_name=adapter_name, **kwargs)
        if resume_only_model:
            return {
                'cur_step': 0,
                'consumed_train_samples': 0,
                'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
                'component_states': {},
            }

        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)
        result = {
            'cur_step': trainer_state['cur_step'],
            'consumed_train_samples': trainer_state['consumed_train_samples'],
            'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
        }
        if 'component_states' in trainer_state:
            result['component_states'] = trainer_state['component_states']
        return result

    @staticmethod
    def _get_rng_state() -> 'ShardedObject':
        from megatron.core import parallel_state as mpu
        from megatron.core import tensor_parallel
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        rng_state = {
            'random_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            # Backend-agnostic device RNG (CUDA / NPU / MPS); key kept as
            # 'cuda_rng_state' for backward compatibility with existing checkpoints.
            'cuda_rng_state': Platform.get_device_rng_state(),
            'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        rng_state_list = [rng_state]

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        return ShardedObject(
            'rng_state',
            rng_state_list,
            (pp_size, tp_size),
            (pp_rank, tp_rank),
            replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
        )

    @staticmethod
    def _generate_state_dict(
        model: list,
        optimizer=None,
        opt_param_scheduler=None,
        rng_state=None,
        iteration: Optional[int] = None,
        model_sd_kwargs: Optional[dict] = None,
        optim_sd_kwargs: Optional[dict] = None,
        save_optim: bool = True,
        save_rng: bool = True,
    ) -> dict:
        model_sd_kwargs = model_sd_kwargs or {}
        optim_sd_kwargs = optim_sd_kwargs or {}

        state_dict: dict = {
            'checkpoint_version': 3.0,
        }
        if iteration is not None:
            state_dict['iteration'] = iteration

        # Model sharded state dict
        for i, m in enumerate(model):
            key = 'model' if len(model) == 1 else f'model{i}'
            state_dict[key] = m.sharded_state_dict(**model_sd_kwargs)

        # Optimizer + scheduler
        if save_optim and optimizer is not None:
            state_dict['optimizer'] = optimizer.sharded_state_dict(
                state_dict,
                **optim_sd_kwargs,
            )
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

        # RNG
        if save_rng and rng_state is not None:
            state_dict['rng_state'] = rng_state

        return state_dict

    def _mcore_async_calls_queue(self):
        """The mcore-native async write queue, owned per model instance (no megatron global args)."""
        if self._mcore_async_queue is None:
            from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
            self._mcore_async_queue = AsyncCallsQueue(persistent=False)
        return self._mcore_async_queue

    def _maybe_finalize_mcore_async_saves(self, blocking: bool = False) -> None:
        """Join completed async shard writes; runs on all ranks (the finalize holds a barrier)."""
        if self._mcore_async_queue is not None:
            self._mcore_async_queue.maybe_finalize_async_calls(
                blocking, no_dist=not dist.is_initialized())

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def wait_async_saves(self) -> None:
        """Join every in-flight save on both async channels (thread-pool tail + mcore queue)."""
        super().wait_async_saves()
        self._maybe_finalize_mcore_async_saves(blocking=True)

    def _save_mcore_optimizer(
        self,
        checkpoint_dir: str,
        optimizer_config: 'MegatronOptimizerGroup',
        **kwargs,
    ):
        """Write the optimizer/scheduler/RNG state in mcore dist-ckpt format.

        With ``async_save=True`` the tensors are staged to host memory here and the shard writes
        move to an mcore async worker; the caller receives the AsyncRequest. Returns None on the
        synchronous path."""
        from megatron.core import dist_checkpointing
        from megatron.core import parallel_state as mpu
        from megatron.core.dist_checkpointing.serialization import TorchDistSaveShardedStrategy
        from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper

        iteration = optimizer_config.cur_step
        iter_dir = os.path.join(checkpoint_dir, f'iter_{iteration:07d}')
        os.makedirs(iter_dir, exist_ok=True)

        # The sharding type must match the data-parallel wrapper the strategy chose at wrap time
        # (Megatron-FSDP stores optimizer state as DTensors, plain DDP does not).
        sharded_sd_metadata = self.strategy.get_sharded_sd_metadata()

        no_save_rng = kwargs.pop('no_save_rng', False)
        rng_state = None if no_save_rng else self._get_rng_state()
        model = self.model

        state_dict = self._generate_state_dict(
            model=model,
            optimizer=optimizer_config.optimizer,
            opt_param_scheduler=optimizer_config.lr_scheduler,
            rng_state=rng_state,
            iteration=iteration,
            model_sd_kwargs={'metadata': sharded_sd_metadata},
            optim_sd_kwargs={'metadata': sharded_sd_metadata},
        )

        # thread_count is read at save time, so setting it post-construction is safe;
        # non-torch_dist backends do not expose it and are skipped.
        thread_count = kwargs.pop('thread_count', 2)
        # megatron-core < 0.19 exposed get_default_save_sharded_strategy(); 0.19 made
        # save(sharded_strategy=None) default to TorchDistSaveShardedStrategy itself -- build that
        # explicitly, the thread_count tweak and the wrapper below need a concrete instance.
        save_strategy = TorchDistSaveShardedStrategy()
        if hasattr(save_strategy, 'thread_count'):
            save_strategy.thread_count = thread_count
        if mpu.get_data_parallel_world_size(with_context_parallel=True) > 1:
            save_strategy = FullyParallelSaveStrategyWrapper(
                save_strategy,
                mpu.get_data_parallel_group(with_context_parallel=True),
            )

        # Async staging is only sound under an initialized process group (the metadata finalize
        # holds a barrier); a single-process run has no skew to hide anyway.
        async_save = kwargs.pop('async_save', False) and dist.is_initialized()
        async_request = dist_checkpointing.save(
            state_dict,
            iter_dir,
            save_strategy,
            async_sharded_save=async_save,
            validate_access_integrity=True,
            content_metadata=sharded_sd_metadata,
        )
        if async_request is not None:
            # The strategy already staged the tensors to host memory; the queue's worker performs
            # the shard writes and the metadata finalize once every rank landed.
            return async_request

        if dist.is_initialized():
            dist.barrier()

        # Write tracker file (rank 0 only).
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            tracker_path = os.path.join(
                checkpoint_dir,
                'latest_checkpointed_iteration.txt',
            )
            with open(tracker_path, 'w') as f:
                f.write(str(iteration))

        logger.info(f'Saved mcore optimizer state at iteration {iteration} '
                    f'to {checkpoint_dir}')

    def _load_mcore_optimizer(
        self,
        checkpoint_dir: str,
        adapter_name: str = '',
        **kwargs,
    ):
        from megatron.core import dist_checkpointing
        from megatron.core import parallel_state as mpu
        from megatron.core import tensor_parallel
        from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
        from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper

        no_load_optim = kwargs.pop('no_load_optim', False)
        no_load_rng = kwargs.pop('no_load_rng', False)
        data_parallel_random_init = kwargs.pop('data_parallel_random_init', False)

        optimizer_config = self.optimizer_group.get(adapter_name or self._get_default_group(), )

        # Read iteration from tracker file.
        tracker_path = os.path.join(
            checkpoint_dir,
            'latest_checkpointed_iteration.txt',
        )
        iteration = self._read_iteration(tracker_path)
        if iteration == 0:
            logger.warning(f'No checkpoint found in {checkpoint_dir}')
            return

        iter_dir = os.path.join(checkpoint_dir, f'iter_{iteration:07d}')

        # Load common (non-sharded) state to inspect content metadata.
        common_state = dist_checkpointing.load_common_state_dict(iter_dir)
        sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=common_state, )

        # Build optimizer / scheduler references for the sharded state dict.
        optimizer = optimizer_config.optimizer if not no_load_optim else None
        opt_param_scheduler = (optimizer_config.lr_scheduler if not no_load_optim else None)
        rng_state = self._get_rng_state() if not no_load_rng else None

        optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
        model_sd_kwargs = dict(metadata=sharded_sd_metadata)

        sharded_state_dict = self._generate_state_dict(
            model=self.model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            rng_state=rng_state,
            iteration=iteration,
            model_sd_kwargs=model_sd_kwargs,
            optim_sd_kwargs=optim_sd_kwargs,
        )

        # Load using fully-parallel strategy for speed.
        load_strategy = get_default_load_sharded_strategy(iter_dir)
        if mpu.get_data_parallel_world_size(with_context_parallel=True) > 1:
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy,
                mpu.get_data_parallel_group(with_context_parallel=True),
            )
        state_dict = dist_checkpointing.load(
            sharded_state_dict,
            iter_dir,
            load_strategy,
        )

        # Restore model weights.
        if len(self.model) == 1:
            self.model[0].load_state_dict(state_dict['model'], strict=False)
        else:
            for i, m in enumerate(self.model):
                key = f'model{i}'
                if key in state_dict:
                    m.load_state_dict(state_dict[key], strict=False)

        # Restore optimizer + LR scheduler.
        if not no_load_optim and optimizer is not None and 'optimizer' in state_dict:
            with torch.no_grad():
                optimizer.load_state_dict(state_dict['optimizer'])
            if (opt_param_scheduler is not None and 'opt_param_scheduler' in state_dict):
                opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'], )

        if not no_load_rng and 'rng_state' in state_dict:
            rng = state_dict['rng_state']
            if isinstance(rng, (list, tuple)):
                rng_index = mpu.get_data_parallel_rank() if data_parallel_random_init else 0
                if rng_index >= len(rng):
                    raise ValueError(
                        f'Checkpoint contains {len(rng)} RNG states, but data-parallel rank {rng_index} was requested.')
                rng = rng[rng_index]
            random.setstate(rng['random_rng_state'])
            np.random.set_state(rng['np_rng_state'])
            torch.set_rng_state(rng['torch_rng_state'])
            # Backend-agnostic restore: tolerates ckpt produced on different backend
            # (returns None) and avoids hard-coded torch.cuda which crashes on NPU.
            Platform.set_device_rng_state(rng.get('cuda_rng_state'))
            tensor_parallel.get_cuda_rng_tracker().set_states(rng['rng_tracker_states'], )

        # Restore iteration counter.
        if optimizer_config is not None and 'iteration' in state_dict:
            optimizer_config.cur_step = state_dict['iteration']

        if dist.is_initialized():
            dist.barrier()

        logger.info(f'Resumed from mcore checkpoint at iteration {iteration} '
                    f'from {checkpoint_dir}')

    @staticmethod
    def _read_iteration(tracker_path: str) -> int:
        # All ranks must enter the all_reduce together; a missing tracker on some ranks
        # (NFS lag, partial mount) is treated as 0 and the MAX reduction recovers the
        # canonical iteration from any rank that read the file.
        iteration = 0
        if os.path.exists(tracker_path):
            with open(tracker_path) as f:
                iteration = int(f.read().strip())
        if torch.distributed.is_initialized():
            # Use Platform.get_local_device() to stay backend-agnostic
            # (CUDA / NPU / MPS); 'cuda' would crash on NPU.
            iters_dev = torch.tensor([iteration], dtype=torch.long, device=Platform.get_local_device())
            torch.distributed.all_reduce(iters_dev, op=torch.distributed.ReduceOp.MAX)
            iteration = int(iters_dev[0].item())
        return iteration

    def _merge_lora_adapters(self, adapter_name: str = 'default'):
        """Merge LoRA adapters into base model weights."""
        from mcore_bridge import LoraParallelLinear
        with torch.no_grad():
            for model in self.strategy.unwrap_model(self.model):
                for module in model.modules():
                    if isinstance(module, (LoraParallelLinear, LoraLinear)):
                        module.merge(adapter_names=[adapter_name])

    def _unmerge_lora_adapters(self):
        """Unmerge LoRA adapters to restore training state."""
        from mcore_bridge import LoraParallelLinear
        with torch.no_grad():
            for model in self.strategy.unwrap_model(self.model):
                for module in model.modules():
                    if isinstance(module, (LoraParallelLinear, LoraLinear)):
                        module.unmerge()

    def _export_hf_config(self):
        """A copy of ``hf_config`` finalized for HF export: sync the MTP layer count from the
        mcore model config, and drop a stale ``quantization_config`` (exported safetensors are
        plain bf16/fp16 weights). The live ``self.hf_config`` stays untouched for training."""
        from copy import deepcopy
        hf_config = deepcopy(self.hf_config)
        mtp_num_layers = getattr(self.strategy.config, 'mtp_num_layers', None)
        if mtp_num_layers:
            text_config = hf_config.get_text_config()
            for key in ('num_nextn_predict_layers', 'mtp_num_hidden_layers'):
                if hasattr(text_config, key):
                    setattr(text_config, key, mtp_num_layers)
                    break
            else:
                text_config.num_nextn_predict_layers = mtp_num_layers
        if getattr(hf_config, 'quantization_config', None) is not None:
            try:
                delattr(hf_config, 'quantization_config')
            except AttributeError:
                hf_config.quantization_config = None
        return hf_config

    def _save_hf_format(self,
                        output_dir: str,
                        adapter_name: str,
                        lora_converter=None,
                        max_shard_size: str = '5GB'):
        """Save in HuggingFace format using bridge adapter.

        For distributed training:
        - All PP ranks participate in export (each has different layers)
        - Only global rank 0 actually writes shared config files
        - Uses barrier for synchronization

        For LoRA training:
        - Saves in PEFT format (adapter_model.safetensors + adapter_config.json)
        """
        # Check if this is LoRA training
        is_peft_format = (adapter_name != _default_adapter_name)
        is_global_zero = (not dist.is_initialized()) or dist.get_rank() == 0

        if is_global_zero:
            os.makedirs(output_dir, exist_ok=True)

        # Synchronize before saving
        if dist.is_initialized():
            dist.barrier()

        # Get the model (unwrap if DDP wrapped)
        model = self.strategy.unwrap_model(self.model)
        save_weights = self.strategy.bridge.save_weights
        save_kwargs = {
            'peft_format': is_peft_format,
            'adapter_name': adapter_name,
            'converter': lora_converter,
            'max_shard_size': max_shard_size,
        }
        # Newer mcore-bridge releases can preserve source-checkpoint tensors for modules Megatron
        # does not materialize; adapter checkpoints contain deltas only and ignore the flag.
        try:
            supports_missing_weights = 'save_missing_weights' in inspect.signature(save_weights).parameters
        except (TypeError, ValueError):
            supports_missing_weights = False
        if supports_missing_weights:
            save_kwargs['save_missing_weights'] = True
        save_weights(model, output_dir, **save_kwargs)

        # Save config on global rank 0 only (avoid concurrent writers).
        if is_global_zero:
            self._export_hf_config().save_pretrained(output_dir)
            # Only write an adapter_config when actually saving a PEFT adapter: merge_lora saves merged
            # full weights with adapter_name='' while the model is still PeftModel-wrapped, and indexing
            # peft_config[''] there raises KeyError.
            if is_peft_format and isinstance(model[0], PeftModel):
                config = model[0].peft_config[adapter_name]
                target_modules = config.target_modules
                config.target_modules = 'all-linear'
                model[0].peft_config[adapter_name].save_pretrained(output_dir)
                config.target_modules = target_modules

        if dist.is_initialized():
            dist.barrier()

    def _save_megatron_format(self, output_dir: str, adapter_name: str, lora_converter=None):
        """Save in Megatron checkpoint format."""
        is_global_zero = (not dist.is_initialized()) or dist.get_rank() == 0
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self._get_trainable_parameters(adapter_name)
        cpu_state_dict = {}
        for k, v in state_dict.items():
            if lora_converter is not None:
                kv = lora_converter(k, v)
                if kv is None:
                    continue
                k, v = kv
            if k is not None and v is not None:
                cpu_state_dict[k] = v.cpu()

        # Save with rank info for distributed checkpointing
        rank = dist.get_rank() if dist.is_initialized() else 0
        checkpoint_path = os.path.join(output_dir, f'model_rank{rank}.pt')
        torch.save(cpu_state_dict, checkpoint_path)
        # Save shared config on global rank 0 only (avoid concurrent writers).
        model = self.strategy.unwrap_model(self.model)
        if is_global_zero:
            self.hf_config.save_pretrained(output_dir)
            if isinstance(model[0], PeftModel):
                model[0].peft_config[adapter_name].save_pretrained(output_dir)

        # Finalize barrier: ensure all ranks finish writing model_rank*.pt
        # before the caller proceeds (e.g. uploading / loading the ckpt).
        if dist.is_initialized():
            dist.barrier()

    def _save_tokenizer(self, output_dir: str, **kwargs):
        from twinkle.utils import is_last_rank
        if not is_last_rank():
            return

        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        template_ins = optimizer_config.template
        if template_ins is not None:
            template_ins.processor.save_pretrained(output_dir)
        else:
            self._default_tokenizer.save_pretrained(output_dir)

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        """Get trainable state dict.

        Args:
            **kwargs: Additional arguments.

        Returns:
            State dict of trainable parameters.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        return self._get_trainable_parameters(adapter_name)

    def get_hf_state_dict(self, adapter_name: str = '') -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Yield model weights in HuggingFace format, one by one (generator to avoid OOM).

        Preferred for weight synchronization to vLLM: converts Megatron->HF on the fly, avoids
        loading all weights into memory, and works with IPCWeightLoader's bucket transfer.
        """
        model = self.strategy.unwrap_model(self.model)
        yield from self.strategy.bridge.export_weights(
            model,
            target_device=None,  # Keep on current device for IPC transfer
            only_master_rank=False,  # All ranks participate in weight sync
            peft_format=bool(adapter_name),
            adapter_name=adapter_name if adapter_name else None,
            tqdm_desc='Weight sync: ',
        )

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str, Dict[str, Any]], **kwargs):
        assert adapter_name, 'Use a non-empty adapter_name'
        # freeze_* drive the multimodal 'all-linear' target expansion in get_target_modules; popped so they
        # do not leak into get_peft_model. Defaults match dev's TunerConfig defaults.
        freeze_llm = kwargs.pop('freeze_llm', False)
        freeze_vit = kwargs.pop('freeze_vit', True)
        freeze_aligner = kwargs.pop('freeze_aligner', True)
        model = self.strategy.unwrap_model(self.model)
        original_target_modules: List[str] = []
        original_modules_to_save: List[str] = []
        if isinstance(config_or_dir, str):
            config_or_dir = HubOperation.download_model(config_or_dir)
        else:
            if isinstance(config_or_dir, dict):
                config_or_dir = LoraConfig(**config_or_dir)
            # Capture the caller's target spec BEFORE the loop: config.target_modules is reassigned
            # to each chunk's expansion inside it, and a virtual-PP chunk has its own module names,
            # so re-expanding a previous chunk's result would target the wrong layers and skip this
            # chunk's freeze_* filtering. modules_to_save is captured for the same reason.
            if isinstance(config_or_dir.target_modules, str):
                original_target_modules = [config_or_dir.target_modules]
            else:
                original_target_modules = list(config_or_dir.target_modules or [])
            original_modules_to_save = list(config_or_dir.modules_to_save or [])

        _models = []
        for _model in model:
            if isinstance(config_or_dir, str):
                _model = PeftModel.from_pretrained(
                    _model, config_or_dir, adapter_name=adapter_name, is_trainable=kwargs.get('is_trainable', True))
                config = _model.peft_config
            else:
                config = config_or_dir

                # Expand target_modules (e.g., 'all-linear' -> this chunk's actual module names) per chunk.
                if original_target_modules:
                    config.target_modules = self.get_target_modules(
                        _model,
                        original_target_modules,
                        freeze_llm=freeze_llm,
                        freeze_vit=freeze_vit,
                        freeze_aligner=freeze_aligner)
                # Expand the modules_to_save shorthands per chunk too ('all-embedding'; 'all-norm' is
                # rejected inside get_modules_to_save -- mcore fuses norm into its linear layers).
                if original_modules_to_save:
                    config.modules_to_save = self.get_modules_to_save(_model, original_modules_to_save)

                _model = get_peft_model(_model, config, adapter_name=adapter_name)  # noqa
            _models.append(_model)
        self.model = _models

        # Create optimizer group for adapter
        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config  # noqa
        self.optimizer_group[adapter_name].gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        # Fix: use .processor instead of .tokenizer - Template class uses self.processor
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        self.active_group = adapter_name

    @remote_function(dispatch='all', sync=True, timeout=3600)
    def add_adapter_to_model(
        self,
        adapter_name: str,
        config_or_dir: Union[Dict[str, Any], LoraConfig, str],
        **kwargs,
    ):
        """Add LoRA adapter to model.

        Args:
            adapter_name: Name of the adapter.
            config_or_dir: LoRA config or path to saved adapter.
            **kwargs: Additional arguments.
        """
        self._patch_adapter(adapter_name, config_or_dir, **kwargs)

    @remote_function()
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs):
        apply_patch(self.model, patch_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """Set template for input encoding.

        Args:
            template_cls: Template class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if 'template_spec' not in kwargs:
            kwargs['model_id'] = self.tokenizer_id
        optimizer_config.template = construct_template(template_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str, Callable], **kwargs):
        """Set input processor.

        Args:
            processor_cls: Processor class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['framework'] = 'megatron'
        # processor/base.py: self.device_mesh.cp_world_size
        kwargs['device_mesh'] = kwargs.get('device_mesh', self.device_mesh)
        processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)
        if processor.padding_free and not self.variable_seq_lengths:
            raise ValueError('padding_free=True requires variable_seq_lengths=True in MegatronModel. '
                             'Padding-free packing merges sequences into batch=1, making fixed-length '
                             'microbatch slicing impossible.')
        optimizer_config.processor = processor

    @remote_function(execute='first', lazy_collect=False)
    def get_train_configs(self, **kwargs):
        """Get training configuration summary.

        Args:
            **kwargs: Additional arguments.

        Returns:
            Configuration summary string.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]

        expr = 'Backend: Megatron-Core\n'
        expr += f'DP size: {self.device_mesh.dp_world_size}\n'
        expr += f'TP size: {self.device_mesh.tp_world_size}\n'
        expr += f'  - VPP size: {self.device_mesh.vpp_size}\n'
        expr += f'PP size: {self.device_mesh.pp_world_size}\n'
        expr += f'CP size: {self.device_mesh.cp_world_size}\n'
        expr += f'EP size: {self.device_mesh.ep_size}\n'
        expr += f'Sequence Parallel: {self.strategy.sequence_parallel}\n'

        if optimizer_config.adapter_config is not None:
            config = optimizer_config.adapter_config.__dict__
            config = {key: str(value) for key, value in config.items() if value is not None}
            expr += f'Adapter config:\n{json.dumps(config, indent=2, ensure_ascii=False)}\n'

        if optimizer_config.optimizer:
            expr += f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
            expr += f'Learning rate: {optimizer_config.optimizer.chained_optimizers[0].config.lr}\n'
        if optimizer_config.lr_scheduler:
            expr += f'LR scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
        expr += f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n'

        return expr

    # ── Checkpoint Engine (from CheckpointEngineMixin) ──────────────────
    # The weight generator is shared by direct and checkpoint-engine sync. Megatron uses
    # TP/PP, so get_hf_state_dict() internally performs TP allgather and handles PP layer
    # distribution; all model ranks MUST execute it concurrently for the collectives to complete.

    def _get_weight_generator(
        self,
        adapter_name: str = None,
        base_sync_done: bool = False,
        merge_and_sync: bool = False,
        model_keys: List[str] = None,
        **kwargs,
    ):
        if adapter_name is None:
            adapter_name = self._get_default_group()

        @contextmanager
        def merge_lora():
            for _model in self.strategy.unwrap_model(self.model):
                if isinstance(_model, PeftModel):
                    _model.merge_adapter()
            yield
            for _model in self.strategy.unwrap_model(self.model):
                if isinstance(_model, PeftModel):
                    _model.unmerge_adapter()

        if base_sync_done and adapter_name:
            # The first base model synchronization finished, and is lora training
            if merge_and_sync:

                def weight_generator():
                    with merge_lora():
                        names = []
                        for name, tensor in self.get_hf_state_dict(adapter_name=''):
                            if name is None or tensor is None:
                                continue
                            # Skip LoRA-specific weights for base model sync
                            if is_lora_key(name):
                                continue
                            name = normalize_param_name(name, keep_base_layer=False)
                            names.append(name)
                            yield name, tensor
                        print_weight_example(names)

            else:

                def weight_generator():
                    names = []
                    for name, tensor in self.get_hf_state_dict(adapter_name=adapter_name):
                        if name is None or tensor is None:
                            continue
                        if 'lora' not in name:
                            continue
                        name = normalize_param_name(name, keep_base_layer=True)
                        names.append(name)
                        yield name, tensor
                    print_weight_example(names)
        else:
            # Need to synchronize the base model
            # First full base-model sync.
            def _raw_weights(with_base_layer=False):
                names = []
                for name, tensor in self.get_hf_state_dict(adapter_name=''):
                    if name is None or tensor is None:
                        continue
                    # Skip LoRA-specific weights for base model sync
                    if is_lora_key(name):
                        continue
                    name = normalize_param_name(name, keep_base_layer=False)
                    if with_base_layer:
                        name = add_base_layer_suffix(name, model_keys)
                    names.append(name)
                    yield name, tensor
                print_weight_example(names)

            def weight_generator():
                # Add the ``.base_layer.`` suffix whenever the sampler runs with
                # ``enable_lora`` (``merge_and_sync=False``).  The rename
                # self-guards via ``model_keys``, so it only renames params vLLM
                # actually exposes as ``*WithLoRA`` and is a no-op for full-param.
                if not merge_and_sync:
                    yield from _raw_weights(True)
                else:
                    yield from _raw_weights(False)

        return weight_generator()

    # ── Paired EMA support (policy→reference via checkpoint-engine) ─────────
    # send_weights / prepare_ema_sender / receive_weights_ema are inherited from
    # CheckpointEngineMixin (transport protocol); the weight-source hooks below feed them.

    def _get_local_weight_generator(self):
        """Yield (name, local_param_data) for this rank's Megatron shard.

        Iterates each model chunk's named_parameters directly — no HF conversion,
        no TP allgather.  Names are prefixed with ``chunk{i}.`` to disambiguate VPP
        chunks.  The paired reference (same parallel_spec) uses identical naming.
        """
        models = self.strategy.unwrap_model(self.model)
        for chunk_idx, chunk in enumerate(models):
            for name, param in chunk.named_parameters():
                yield f'chunk{chunk_idx}.{name}', param.data

    def _get_local_state_dict(self):
        """Return {name: param_data} for this rank's local Megatron shard."""
        models = self.strategy.unwrap_model(self.model)
        state = {}
        for chunk_idx, chunk in enumerate(models):
            for name, param in chunk.named_parameters():
                state[f'chunk{chunk_idx}.{name}'] = param.data
        return state

    @remote_function(collect='first', lazy_collect=False)
    def get_peft_config_dict(self, adapter_name: str = None) -> Optional[Dict[str, Any]]:
        """Return the PEFT config as a dict for vLLM's PEFTHelper.

        Used by CheckpointEngineManager for LoRA-only weight sync.

        Returns:
            PEFT config dict, or None if no LoRA adapter is present.
        """
        if adapter_name is None:
            adapter_name = self._get_default_group()
        optimizer_config = self.optimizer_group.get(adapter_name)
        if optimizer_config is None or optimizer_config.adapter_config is None:
            return None
        config = optimizer_config.adapter_config
        if isinstance(config, dict):
            config = config.get(adapter_name, next(iter(config.values())))
        target_modules = config.target_modules
        config.target_modules = 'all-linear'
        _peft_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        _peft_config['target_modules'] = target_modules
        return _peft_config

    @staticmethod
    def get_target_modules(model: 'torch.nn.Module',
                           target_modules: List[str],
                           *,
                           freeze_llm: bool = False,
                           freeze_vit: bool = True,
                           freeze_aligner: bool = True) -> List[str]:
        """Expand the ``all-*`` keywords into concrete module names, honouring the freeze_* partition.

        On a multimodal model an ``all-*`` keyword must respect freeze_llm / freeze_vit /
        freeze_aligner (by default only the language model is adapted). The tower partition is
        read from the live model, never hardcoded; explicit names are kept verbatim.
        """
        import torch

        def find_all_linears(model: torch.nn.Module) -> List[str]:
            from megatron.core.extensions.transformer_engine import (TEGroupedLinear, TELayerNormColumnParallelLinear,
                                                                     TELinear)

            def _cond(name: str, module: torch.nn.Module) -> bool:
                if name == 'output_layer' or 'lora' in name:
                    return False
                if isinstance(module, (TELinear, TELayerNormColumnParallelLinear, TEGroupedLinear, torch.nn.Linear)):
                    return True
                return False

            return _find_layers(model, _cond)

        def find_router(model: torch.nn.Module) -> List[str]:
            from megatron.core.transformer.moe.router import TopKRouter
            return _find_layers(model, lambda name, module: isinstance(module, TopKRouter) and 'lora' not in name)

        keywords = ('all-linear', 'all-embedding', 'all-router')
        # Explicit names are kept verbatim; only the all-* keywords are expanded (and, on a
        # multimodal model, filtered by tower).
        explicit = [t for t in target_modules if t not in keywords]
        expanded: List[str] = []
        if 'all-linear' in target_modules:
            expanded += find_all_linears(model)
        if 'all-embedding' in target_modules:
            expanded += _find_embedding(model)
        if 'all-router' in target_modules:
            expanded += find_router(model)

        from twinkle.module.optimizer import resolve_megatron_tower_prefixes
        tower_prefixes = resolve_megatron_tower_prefixes([model])
        is_multimodal = bool(tower_prefixes['vision_tower'] or tower_prefixes['aligner'])
        if is_multimodal and any(keyword in target_modules for keyword in keywords):
            if freeze_llm and freeze_vit and freeze_aligner:
                raise ValueError('freeze_llm / freeze_vit / freeze_aligner are all True, so a multimodal '
                                 'LoRA has no tower left to adapt. Unfreeze at least one of them.')
            frozen = {'language_model': freeze_llm, 'vision_tower': freeze_vit, 'aligner': freeze_aligner}
            # Longest prefix first, so the aligner (nested under the vision tower) is classified
            # as the aligner rather than the tower; a module in no declared tower is dropped.
            ranked = sorted(((prefix, role) for role, prefixes in tower_prefixes.items() for prefix in prefixes),
                            key=lambda item: -len(item[0]))
            kept = []
            for name in expanded:
                role = next((r for prefix, r in ranked if name.startswith(prefix)), None)
                if role is not None and not frozen[role]:
                    kept.append(name)
            expanded = kept
        return list(set(explicit + expanded))

    @staticmethod
    def get_modules_to_save(model: 'torch.nn.Module', modules_to_save: List[str]) -> List[str]:
        """Expand the ``modules_to_save`` shorthands into concrete mcore module names.

        Only ``'all-embedding'`` is expanded; ``'all-norm'`` is rejected (mcore fuses the
        normalization into its linear layers, so there is no standalone norm module).
        """
        if 'all-norm' in modules_to_save:
            raise ValueError(
                "modules_to_save 'all-norm' is not supported on the megatron backend: mcore fuses the "
                'normalization into its linear layers, so there is no standalone norm module to save. '
                'List the concrete module names to save instead.')
        if 'all-embedding' not in modules_to_save:
            return list(modules_to_save)
        explicit = [m for m in modules_to_save if m != 'all-embedding']
        found = _find_embedding(model)
        if not found:
            logger.warning("modules_to_save 'all-embedding' matched no mcore embedding module; ignored.")
        return list(dict.fromkeys(explicit + found))
