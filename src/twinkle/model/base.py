# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re
import shutil
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, Type, Union

from twinkle import Platform, remote_function, torch_util
from twinkle.data_format import InputFeature, ModelOutput
from twinkle.hub import HubOperation
from twinkle.loss.base import Loss
from twinkle.metric import Metric
from twinkle.patch import Patch
from twinkle.processor import InputProcessor
from twinkle.template import Template

if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from transformers import PreTrainedModel, PretrainedConfig


def copy_checkpoint_args(output_dir: str, checkpoint_dir: str) -> None:
    """Copy the run-level CLI metadata into a completed checkpoint on the master rank."""
    if not Platform.is_master():
        return
    source = os.path.join(output_dir, 'args.json')
    target = os.path.join(checkpoint_dir, 'args.json')
    if os.path.isfile(source) and os.path.realpath(source) != os.path.realpath(target):
        shutil.copy2(source, target)


def rotate_checkpoints(output_dir: str, current_checkpoint_dir: str, save_total_limit: Optional[int]) -> None:
    """Retain the newest completed checkpoint directories, always protecting the current save."""
    if save_total_limit is None:
        return
    if save_total_limit < 1:
        raise ValueError('save_total_limit must be >= 1.')
    if not Platform.is_master() or not os.path.isdir(output_dir):
        return
    current = os.path.realpath(current_checkpoint_dir)
    checkpoints = []
    for entry in os.scandir(output_dir):
        if not entry.is_dir(follow_symlinks=False) or re.fullmatch(r'checkpoint-(?:\d+|final)', entry.name) is None:
            continue
        path = os.path.realpath(entry.path)
        checkpoints.append((path == current, entry.stat(follow_symlinks=False).st_mtime_ns, entry.name, path))
    checkpoints.sort()
    for _, _, _, checkpoint_path in checkpoints[:-save_total_limit]:
        shutil.rmtree(checkpoint_path)


def _freeze_then_activate(named_params,
                          *,
                          freeze_ratio: float = 0.0,
                          freeze_names=(),
                          freeze_regex: Optional[str] = None,
                          trainable_names=(),
                          trainable_regex: Optional[str] = None) -> None:
    """Mutate ``requires_grad`` on a list of ``(name, param)`` pairs: freeze_* then trainable_*.

    The pure core of :meth:`TrainableModel.freeze_parameters`, split out so the ordering / matching
    rules live in one testable place independent of how a backend gathers its parameters. Mirrors
    legacy swift's ``freeze_parameters`` + ``activate_parameters`` (cumulative-element-count ratio via
    bisect, name-prefix ``startswith``, regex ``search``), except an invalid regex raises here rather
    than being warned-and-skipped -- a bad pattern should fail the run, not silently freeze nothing.
    """
    from bisect import bisect_right

    if freeze_ratio > 0:
        cumulative = []
        running = 0
        for _, param in named_params:
            running += param.numel()
            cumulative.append(running)
        n_freeze = int(running * freeze_ratio)
        idx = bisect_right(cumulative, n_freeze)
        for _, param in named_params[:idx]:
            param.requires_grad = False

    if freeze_names:
        for name, param in named_params:
            if any(name.startswith(prefix) for prefix in freeze_names):
                param.requires_grad = False

    if freeze_regex is not None:
        pattern = re.compile(freeze_regex)
        for name, param in named_params:
            if pattern.search(name):
                param.requires_grad = False

    if trainable_names:
        for name, param in named_params:
            if any(name.startswith(prefix) for prefix in trainable_names):
                param.requires_grad = True

    if trainable_regex is not None:
        pattern = re.compile(trainable_regex)
        for name, param in named_params:
            if pattern.search(name):
                param.requires_grad = True


class ModelLoaderProtocol(Protocol):
    """The construction hooks twinkle's model backends call on a caller-supplied ``model_loader``.

    A backend hands checkpoint construction back to the caller by accepting a ``model_loader``. The
    transformers backend calls all six hooks (build config -> processor -> model, each paired with a
    ``process_*`` post-hook); the megatron backend calls only ``build_config`` / ``process_config``
    (mcore builds the module itself and the bridge loads the weights). This Protocol names exactly that
    consumer-defined surface -- and nothing more -- so the seam is type-checked instead of duck-typed
    ``Any``, and a misspelled hook is caught statically rather than at load time.

    It is a structural contract, deliberately NOT a base class: twinkle never imports the concrete
    loader and never ``isinstance``-checks it, so the dependency stays one-directional (caller ->
    twinkle). swift/dev's ``ModelLoader`` -- the family-registry base, which additionally carries
    ``model_arch`` / ``model_info`` / registration metadata that twinkle does not consume -- satisfies
    this Protocol structurally without inheriting it.
    """

    def build_config(self, model_dir: str, **kwargs) -> 'PretrainedConfig':
        ...

    def process_config(self, config: 'PretrainedConfig') -> 'PretrainedConfig':
        ...

    def build_processor(self, model_dir: str, config: 'PretrainedConfig', **kwargs) -> Any:
        ...

    def process_tokenizer(self, tokenizer: Any) -> Any:
        ...

    def build_model(self, model_dir: str, config: 'PretrainedConfig', processor: Any,
                    **kwargs) -> 'PreTrainedModel':
        ...

    def process_model(self, model: 'PreTrainedModel') -> 'PreTrainedModel':
        ...


class TrainableModel(ABC):

    _checkpoint_engine = None

    @abstractmethod
    def forward(self, *, inputs: Dict[str, Any], **kwargs) -> ModelOutput:
        ...

    @abstractmethod
    def forward_only(self, *, inputs: Dict[str, Any], **kwargs) -> ModelOutput:
        ...

    @abstractmethod
    def calculate_loss(self, **kwargs) -> float:
        ...

    @abstractmethod
    def backward(self, **kwargs) -> None:
        ...

    @abstractmethod
    def forward_backward(self, *, inputs: Dict[str, Any], **kwargs) -> ModelOutput:
        ...

    @abstractmethod
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs) -> float:
        ...

    @abstractmethod
    def step(self, **kwargs) -> None:
        ...

    @abstractmethod
    def zero_grad(self, **kwargs) -> None:
        ...

    @abstractmethod
    def lr_step(self, **kwargs) -> None:
        ...

    @abstractmethod
    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs) -> None:
        ...

    @abstractmethod
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...],
                                                                       'torch.Tensor']], **kwargs) -> None:
        ...

    @abstractmethod
    def set_optimizer(self, optimizer_cls: Union['Optimizer', Type['Optimizer'], str], **kwargs) -> None:
        ...

    @abstractmethod
    def set_lr_scheduler(self, scheduler_cls: Union['LRScheduler', Type['LRScheduler'], str], **kwargs) -> None:
        ...

    @abstractmethod
    def save(self, name: str, output_dir: Optional[str] = None, **kwargs) -> str:
        ...

    @abstractmethod
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs) -> None:
        ...

    @abstractmethod
    def get_state_dict(self, **kwargs) -> Dict[str, Any]:
        ...

    @abstractmethod
    def resume_from_checkpoint(self,
                               checkpoint_dir: str,
                               *,
                               resume_only_model: bool = False,
                               **kwargs) -> Dict[str, Any]:
        ...

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        ...

    @abstractmethod
    def add_metric(self, metric_cls: Union[Metric, str], is_training: Optional[bool] = None, **kwargs) -> None:
        ...

    @abstractmethod
    def calculate_metric(self, is_training: bool, **kwargs) -> Dict[str, Any]:
        ...

    @abstractmethod
    def empty_cache(self, **kwargs) -> None:
        ...

    @abstractmethod
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs) -> None:
        ...

    @abstractmethod
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs) -> None:
        ...

    @abstractmethod
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs) -> None:
        ...

    @abstractmethod
    def get_train_configs(self, **kwargs) -> str:
        ...

    def upload_to_hub(self,
                      checkpoint_dir: str,
                      hub_model_id: str,
                      hub_token: Optional[str] = None,
                      async_upload: bool = True) -> None:
        """Upload model checkpoint to hub.

        Args:
            checkpoint_dir: The directory path of the checkpoint to upload.
            hub_model_id: The hub model id.
            hub_token: The hub token (optional).
            async_upload: Whether to use async upload (default: True).
        """
        if async_upload:
            HubOperation.async_push_to_hub(
                repo_id=hub_model_id, folder_path=checkpoint_dir, token=hub_token, private=True)
        else:
            HubOperation.push_to_hub(repo_id=hub_model_id, folder_path=checkpoint_dir, token=hub_token, private=True)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def offload_to_cpu(self) -> None:
        """Hand this rank's training memory back so a colocated process can use the device.

        Colocation -- an online-RL rollout engine, or an in-training generative-eval sampler, sharing
        the trainer's GPU -- cannot fit both at once, so the trainer steps aside between steps and
        :meth:`reload_to_gpu` brings it back. The device work is the strategy's: only it knows where
        this backend keeps the parameters (Megatron pools them into flat per-bucket buffers; the
        transformers backends hold them on the wrapped module), and an offload that moved the wrong
        object would report success while freeing nothing. The optimizer state travels with them -- it
        dwarfs the weight bytes, so offloading the weights alone would reclaim almost nothing.

        This is the driver-facing handle over ``strategy.offload_to_cpu(model, optimizer)``; both
        concrete models carry the ``strategy`` / ``model`` / ``optimizer_group`` / ``_get_default_group``
        shape it reads. A strategy with no offload of its own (deepspeed, native FSDP -- whose sharded
        parameters need backend-specific handling, not a plain move) leaves nothing to delegate to.

        ``@remote_function`` is what makes the handle reach the workers: under Ray the model on the
        driver is a proxy with no ``strategy`` of its own, so an undecorated method would run against
        that empty stub. ``dispatch='all'`` runs it on every rank (each offloads its own shard) and
        ``lazy_collect=False`` blocks until they finish -- the colocated sampler is a SEPARATE actor,
        so Ray's per-actor ordering cannot guarantee the GPU is actually free before the caller wakes
        the engine and lets it allocate its KV cache.
        """
        self.strategy.offload_to_cpu(self.model, self._colocation_optimizer())

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def reload_to_gpu(self) -> None:
        """Bring back what :meth:`offload_to_cpu` released, to the device it was moved off.

        Decorated for the same reason as :meth:`offload_to_cpu`, and blocking so the trainer's next
        step cannot start on weights the workers have not finished moving back.
        """
        self.strategy.reload_to_gpu(self.model, self._colocation_optimizer())

    def _colocation_optimizer(self) -> Optional['Optimizer']:
        """The default group's optimizer, or None before one exists.

        Resolved through the group on every call rather than cached: a model offloaded before its
        optimizer is built -- a frozen reference model, or the rollout phase of a colocated step --
        offloads its weights alone and returns None here instead of raising on a missing optimizer.
        """
        group = self.optimizer_group.get(self._get_default_group())
        return group.optimizer if group is not None else None

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def freeze_parameters(self,
                          *,
                          freeze_ratio: float = 0.0,
                          freeze_names=(),
                          freeze_regex: Optional[str] = None,
                          trainable_names=(),
                          trainable_regex: Optional[str] = None) -> None:
        """Set ``requires_grad`` on this rank's parameters for full-parameter freeze / activate.

        This is the driver-facing handle for a full-parameter run's ``freeze_parameters`` /
        ``trainable_parameters`` knobs. It runs on every rank (``dispatch='all'``) because under Ray
        the model on the driver is a proxy with no parameters of its own, and it blocks
        (``lazy_collect=False``) so the caller's next step -- building the optimizer, which reads
        ``requires_grad`` to choose its param groups -- cannot run before the freeze has landed on the
        workers. It must therefore be called BEFORE the optimizer is constructed.

        The order is legacy's and is load-bearing: freeze first (by ratio, then name-prefix, then
        regex), then activate (name-prefix, then regex), so ``trainable_*`` WINS over ``freeze_*``.

        One deliberate divergence from legacy's full branch: it calls ``requires_grad_(True)`` on the
        whole model first, which would undo a loader's intentional freeze (e.g. the Qwen3-TTS
        ``speaker_encoder``). This does not reset -- a freshly built model is already all-trainable
        except what its loader froze, so the common case is identical while loader-level freezes
        survive.

        ``freeze_ratio`` counts by cumulative element count in ``named_parameters()`` order and is
        only meaningful when every rank holds the same slice of the model; a pipeline-parallel run
        (each rank holding different layers) must reject it at the config layer, not here.
        """
        model = self.strategy.unwrap_model(self.model)
        modules = model if isinstance(model, (list, tuple)) else [model]
        named = []
        for module in modules:
            named.extend(module.named_parameters())
        _freeze_then_activate(
            named,
            freeze_ratio=freeze_ratio,
            freeze_names=freeze_names,
            freeze_regex=freeze_regex,
            trainable_names=trainable_names,
            trainable_regex=trainable_regex)

    def _should_bind_device_id_for_process_group(self, backend: str) -> bool:
        return backend in ('nccl', 'hccl')

    def _try_init_process_group(self):
        import torch
        import torch.distributed as dist
        if not dist.is_initialized() and Platform.get_world_size() > 1:
            torch_util.set_device()
            backend = Platform.device_backend()
            if backend == 'hccl':
                # fix: In multi-job NPU runs, HCCL default ports may collide (bind/listen failures).
                # fix: Inject deterministic per-job port ranges before PG init to reduce cross-job conflicts.
                # Keep training-side HCCL sockets on a per-job port layout to
                # avoid collisions with other jobs on the same host.
                from twinkle.utils.platforms import ensure_hccl_socket_env
                master_port = int(os.environ.get('MASTER_PORT', '29500'))
                ensure_hccl_socket_env(master_port)
            init_kwargs = {
                'backend': backend,
                'init_method': 'env://',
                'rank': Platform.get_rank(),
                'world_size': Platform.get_world_size(),
            }
            if self._should_bind_device_id_for_process_group(backend):
                init_kwargs['device_id'] = torch.device(Platform.get_local_device())
            dist.init_process_group(**init_kwargs)
            if backend == 'hccl':
                default_pg = dist.distributed_c10d._get_default_group()
                if getattr(default_pg, 'bound_device_id', None) is not None:
                    # If the default HCCL PG keeps a bound device id, PyTorch may
                    # propagate that binding into later Gloo subgroup creation. That
                    # breaks the metrics/object-gather path on NPU, so clear it
                    # before Megatron creates its Gloo DP groups.
                    default_pg.bound_device_id = None
