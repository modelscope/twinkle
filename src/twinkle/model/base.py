# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, Type, Union

from twinkle import Platform, remote_function, torch_util
from twinkle.data_format import InputFeature, ModelOutput
from twinkle.hub import HubOperation
from twinkle.loss.base import Loss
from twinkle.metric import Metric
from twinkle.model.tasks import get_task_spec_obj
from twinkle.patch import Patch
from twinkle.perf import timed
from twinkle.processor import InputProcessor
from twinkle.processor.tasks import get_task_spec
from twinkle.template import Template

if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from transformers import PreTrainedModel, PretrainedConfig


from twinkle.utils.storage import LocalDiskStorage, Storage, WriteLease


def copy_checkpoint_args(output_dir: str, checkpoint_dir: str) -> None:
    """Copy the run-level CLI metadata into a completed local checkpoint (master rank).

    The local-only form the server-side model classes use: their checkpoints always live
    on the server's local disk, so they never take a write lease (see
    :func:`stage_checkpoint_args` for the lease-based form a training run's save uses).
    """
    if not Platform.is_master():
        return
    LocalDiskStorage().copy_into_local(
        os.path.join(output_dir, 'args.json'), os.path.join(checkpoint_dir, 'args.json'))


def stage_checkpoint_args(output_dir: str, lease: WriteLease) -> None:
    """Stage the run-level CLI metadata into the checkpoint's write lease on the master rank.

    The lease carries the write: the real checkpoint dir for a local destination,
    the staging copy a remote flush uploads together with the rest of the save.
    """
    if not Platform.is_master():
        return
    lease.storage.copy_into_local(os.path.join(output_dir, 'args.json'), os.path.join(lease.path, 'args.json'))


#: A checkpoint is resumable when any of these lands (full save: trainer_state.json;
#: weights-only: the HF-format config an adapter/full save always writes; megatron
#: resume: the iteration tracker). Remote rotation counts only these and cleans
#: the marker-less partials a crashed upload leaves behind.
_ROTATION_MARKERS = (
    'trainer_state.json',
    'adapter_config.json',
    'config.json',
    'latest_checkpointed_iteration.txt',
)


def _rotation_order(name: str) -> float:
    """Numeric fallback ordering for media whose directory listings carry no mtime.

    Object stores have no directory objects: a prefix lists with mtime 0, so the
    checkpoint name's step number orders retention there instead (``checkpoint-final``
    highest, mirroring the local convention that the final save is the last one)."""
    match = re.fullmatch(r'checkpoint-(\d+)', name)
    return int(match.group(1)) if match else float('inf')


def write_task_head_checkpoint(checkpoint_dir: str, state: Dict[str, Any]) -> None:
    """Write an already collected task head using slot-independent checkpoint keys."""
    if not state:
        return
    from safetensors.torch import save_file
    save_file({key: value.contiguous() for key, value in state.items()},
              os.path.join(checkpoint_dir, 'task_head.safetensors'))


def rotate_checkpoints(output_dir: str, current_checkpoint_dir: str, save_total_limit: Optional[int]) -> None:
    """Retain the newest completed checkpoint directories, always protecting the current save."""
    if save_total_limit is None:
        return
    if save_total_limit < 1:
        raise ValueError('save_total_limit must be >= 1.')
    if not Platform.is_master():
        return
    storage = Storage.from_url(output_dir)
    if not storage.is_dir(output_dir):
        return
    checkpoints = []
    for entry in storage.iterdir(output_dir):
        if not entry.is_dir or re.fullmatch(r'checkpoint-(?:\d+|final)', entry.name) is None:
            continue
        if storage.remote:
            # Only a marker-complete directory counts remotely: a marker-less partial
            # is a crashed upload, cleaned here instead of occupying a retention slot.
            if not any(storage.exists(f'{entry.path}/{marker}') for marker in _ROTATION_MARKERS):
                if not storage.same_dir(entry.path, current_checkpoint_dir):
                    storage.delete_tree(entry.path)
                continue
        checkpoints.append(
            (storage.same_dir(entry.path, current_checkpoint_dir),
             entry.mtime_ns or _rotation_order(entry.name), entry.name, entry.path))
    checkpoints.sort()
    for _, _, _, checkpoint_path in checkpoints[:-save_total_limit]:
        storage.delete_tree(checkpoint_path)


# Trajectory fields ``TrainableModel._rollout_features`` lifts into loss kwargs on training calls.
# The names match the loss kwargs exactly; token-level fields are response-framed on both sides.
_LIFTED_TRAJECTORY_FIELDS = (
    'rollout_logps', 'sampling_masks', 'advantages', 'returns',
    'ref_logps', 'teacher_logps', 'old_logps', 'old_values', 'truncated',
)


def _record_rollout_identity(rows) -> None:
    """Append each handed-over row's rollout ids beside the trainer's own columns, if the probe is on.

    The contract being measured is that the response tokens a rollout GENERATED arrive bit-identical
    in what the trainer consumes. This is the last seam where both are in scope: the caller next
    filters to ``InputProcessor.FEATURE_KEYS``, which drops ``response_token_ids`` and the row
    identity, and the processor then pads/collates, after which a row can no longer be singled out.
    """
    import os
    root = os.environ.get('TWINKLE_ROLLOUT_IDENTITY_PROBE')
    if not root:
        return
    import json
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'identity.{os.getpid()}.jsonl')

    def flat(value):
        return value.tolist() if hasattr(value, 'tolist') else list(value)

    with open(path, 'a', encoding='utf-8') as handle:
        for row in rows:
            record = {key: row.get(key) for key in ('episode_id', 'group_key', 'policy_version', 'role')}
            for key in ('response_token_ids', 'input_ids', 'labels', 'completion_mask', 'response_positions'):
                value = row.get(key)
                if value is not None:
                    record[key] = flat(value)
            handle.write(json.dumps(record) + '\n')


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
    transformers backend calls config -> processor -> model, each paired with a ``process_*`` post-hook.
    Non-source native FSDP ranks use ``build_model_from_config`` instead of loading the checkpoint
    when memory-efficient initialization is enabled. The megatron backend calls only
    ``build_config`` / ``process_config``
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

    def build_model_from_config(self, config: 'PretrainedConfig', processor: Any,
                                **kwargs) -> 'PreTrainedModel':
        """Construct without checkpoint I/O, inside the backend's empty-weight context."""
        ...

    def process_model(self, model: 'PreTrainedModel') -> 'PreTrainedModel':
        ...


def _collect_temperatures(results, device_mesh):
    values = [value for value in results if value is not None]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError('Expected consistent task-head temperatures on all head ranks.')
    return values[0]


class TrainableModel(ABC):
    """The backend-agnostic training surface.

    The abstract methods below are grouped by lifecycle stage -- read this class
    as the table of contents for a model's life:

    - forward / forward_only / forward_backward / calculate_loss / backward:  compute
    - step / zero_grad / lr_step / clip_grad_norm / clip_grad_and_step:  optimization
    - save / load / get_state_dict / resume_from_checkpoint:  persistence
    - set_template / set_processor / add_adapter_to_model / apply_patch / set_loss
      / set_optimizer / set_lr_scheduler / add_metric:  assembly
    - offload_to_cpu / reload_to_gpu / empty_cache / upload_to_hub:  resource lifecycle

    Weight streaming into a sampler/reference (send_weights and friends) lives on
    CheckpointEngineMixin, not here: it is transport protocol, not model surface.
    """

    _checkpoint_engine = None
    _async_write_executor: Optional[ThreadPoolExecutor] = None
    _async_write_future: Optional[Future] = None
    #: write leases whose save returned but whose seal (rotation locally; barrier,
    #: markers, cleanup, rotation remotely) still waits on every rank's write
    #: landing -- run at the next join, which every participating rank enters;
    #: None until an async save runs
    _pending_write_leases: Optional[list] = None

    # -- async checkpoint persistence (staging/write separation) --------------------
    #
    # A save splits into a staging half (collecting state off the training device: gathers, D2H copies,
    # optimizer/rng/scheduler state_dicts -- everything that touches the model or a process group) and a
    # write half (serializing staged CPU payloads to disk -- pure file I/O, safe off-thread). Callers
    # stage synchronously, then hand the write closure to :meth:`submit_async_write`; the checkpoint
    # bytes are frozen at stage time, so training steps taken while the write is in flight never leak
    # into the saved artifact.
    #
    # One write in flight per instance: a new submission first joins the previous write, so a slow disk
    # can at most delay the next save, never pile up snapshots. Exceptions raised by the background
    # write are re-raised to the next join (or :meth:`wait_async_saves`), never swallowed.

    def _stage(self) -> None:
        """Hook for subclasses needing extra synchronous staging work before a write is submitted."""

    def submit_async_write(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` (pure file I/O over already-staged CPU payloads) on the background writer."""
        executor = self._async_save_executor()
        self._join_async_save()
        self._async_write_future = executor.submit(fn)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def wait_async_saves(self) -> None:
        """Join and re-raise any failure from the in-flight background write."""
        self._join_async_save()

    def _async_save_executor(self) -> ThreadPoolExecutor:
        # Lazily created so instances that never save asynchronously pay nothing.
        if self._async_write_executor is None:
            self._async_write_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='async-save')
        return self._async_write_executor

    def _join_async_save(self) -> None:
        # The deferred seal of async saves: the barrier (and marker upload
        # behind it) can only run here, on the main thread -- never in the tail.
        leases, self._pending_write_leases = self._pending_write_leases, None
        future = self._async_write_future
        if future is not None:
            self._async_write_future = None
            try:
                future.result()
            except BaseException:
                # A failed write must never be sealed: its markers would vouch for files
                # that never landed. Drop the staged/partial copies, then re-raise.
                for lease in leases or ():
                    lease.abort()
                raise
        for lease in leases or ():
            lease.seal()

    def _defer_lease_seal(self, lease: WriteLease) -> None:
        """Queue an async save's lease for the next join's seal."""
        if self._pending_write_leases is None:
            self._pending_write_leases = []
        self._pending_write_leases.append(lease)

    def shutdown_async_saves(self) -> None:
        """Join any in-flight write and retire the writer thread (idempotent)."""
        self._join_async_save()
        executor = self._async_write_executor
        if executor is not None:
            self._async_write_executor = None
            executor.shutdown(wait=True)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def close(self) -> None:
        """Finish outstanding checkpoint writes before releasing this model's replicas."""
        self.shutdown_async_saves()

    def __del__(self):
        # Best-effort: an unjoined in-flight write at teardown loses the checkpoint silently otherwise.
        executor = getattr(self, '_async_write_executor', None)
        if executor is not None:
            self.shutdown_async_saves()

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

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def remove_metric(self, metric_type: type, *, adapter_name: Optional[str] = None) -> None:
        """Strip metrics that are instances of ``metric_type`` from a group's train/eval statuses.

        Worker-side handle for the decision readout's metric swap: the default optimizer group
        seeds a token ``Accuracy`` that a scoring head replaces with its own metric. Under Ray the
        driver proxy has no ``optimizer_group`` (it is built worker-side, like the parameters
        :meth:`freeze_parameters` reads), so the strip must run on the workers as :meth:`add_metric`
        does -- ``dispatch='all'`` cleans every rank's group and blocking ensures the next metric
        read sees the cleaned statuses.
        """
        name = adapter_name if adapter_name is not None else self._get_default_group()
        optimizer_config = self.optimizer_group[name]
        for status in (optimizer_config.train_status, optimizer_config.eval_status):
            status.metrics = [m for m in status.metrics if not isinstance(m, metric_type)]

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def perf_on(self, *, profiler_spec=None):
        """Enable profiling on this model's unwrapped modules."""
        from twinkle.perf import Perf
        model = self.strategy.unwrap_model(self.model)
        self._profiler = Perf.enable_process(model=model, profiler_spec=profiler_spec)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def memory_stats(self, *, synchronize: bool = False) -> dict[str, int]:
        """Raw per-rank allocator peaks and host RAM in bytes; no cross-rank reduction."""
        from twinkle import Platform
        from twinkle.perf.stats import memory_stats

        return memory_stats(synchronize=synchronize, device_prefix=Platform.device_prefix())

    @abstractmethod
    def calculate_metric(self, is_training: bool, **kwargs) -> Dict[str, Any]:
        ...

    @abstractmethod
    def empty_cache(self, **kwargs) -> None:
        ...

    @abstractmethod
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs) -> None:
        ...

    # -- attached readout heads: one lifecycle, backend hooks -----------------------
    _head_task: Optional[str] = None
    task_head_slots: Optional[Any] = None  # MultiLora hosts only

    def _head_model_path(self) -> str:
        """Where the decision families' loader replays its build from."""
        raise NotImplementedError

    def _task_head_host(self):
        """The module the head hangs off; the single-model state hooks read it."""
        raise NotImplementedError

    def _task_head_is_master(self) -> bool:
        """Which rank writes the head file."""
        raise NotImplementedError

    def _build_task_head(self, task, task_head_loader=None, num_labels=None):
        """Build the readout task's head, the two flavours at twinkle.module.head.

        decision heads are family artifacts -- the loader's (module, class) NAME ships
        from the driver and replays the build here; the seq_cls family builds its
        standard head from the config (+ num_labels). Placement (device / dtype /
        attachment names) stays with the caller -- backend-specific by design."""
        import importlib
        spec_obj = get_task_spec_obj(task)
        if not spec_obj.row().readout:
            raise ValueError(f'task={task!r} carries no readout head; nothing to attach.')
        if task == 'decision':
            if task_head_loader is None:
                raise ValueError(
                    "task='decision' needs task_head_loader=(module, class) to rebuild its family head.")
            module_name, loader_name = task_head_loader
            loader_cls = getattr(importlib.import_module(module_name), loader_name)
            processor = self._default_tokenizer
            if processor is None:
                from transformers import AutoTokenizer
                processor = AutoTokenizer.from_pretrained(self.tokenizer_id, trust_remote_code=True)
                self._default_tokenizer = processor
            return loader_cls.build_task_head(self._head_model_path(), self.hf_config, processor)
        if task_head_loader is not None:
            raise ValueError(
                'task_head_loader is the decision families\' head replay; '
                f'task={task!r} builds its standard head from the config instead.')
        return spec_obj.build_head(self.hf_config, num_labels)

    def _attached_head(self, adapter_name):
        """Resolve an adapter head, or either registered alias on a single-head host."""
        host = self._task_head_host()
        heads = getattr(host, 'task_heads', None)
        if heads is None:
            head = getattr(host, 'task_head', None)
            return head if head is not None else getattr(host, 'scoring_head', None)
        return heads.get(adapter_name)

    def _task_head_state(self, adapter_name: Optional[str]) -> Dict[str, Any]:
        """The head's parameters and calibration buffers under slot-independent keys.

        Default (single-model) form reads the head hanging on the host; MultiLora
        hosts override this with their slot-pool export."""
        head = self._attached_head(adapter_name)
        if head is None:
            return {}
        return {f'task_head.{name}': torch_util.to_local_tensor(tensor).detach().clone().cpu()
                for name, tensor in chain(head.named_parameters(), head.named_buffers())}

    def _task_head_carries_on_this_rank(self) -> bool:
        """Whether THIS rank is expected to hold the head at all. True by default;
        the megatron backend overrides it -- only the last pipeline stage carries
        heads, so their absence on earlier stages is the layout, not a mismatch."""
        return True

    def _restore_task_head_state(self, adapter_name: Optional[str], state: Dict[str, Any]) -> None:
        """Write a ``_task_head_state`` snapshot back in place; MultiLora hosts override
        this with their slot-pool import. An empty snapshot passes -- every host checks
        the OTHER side's emptiness, never its own (a rank without slots skips, a
        checkpoint written head-less stays head-less)."""
        head = self._attached_head(adapter_name)
        if head is None:
            if state and self._task_head_carries_on_this_rank():
                raise ValueError(
                    f'task_head checkpoint has keys {sorted(state)} but adapter {adapter_name!r} '
                    'carries no head to restore them into.')
            return
        params = dict(head.named_parameters())
        tensors = {**params, **dict(head.named_buffers())}
        expected = {f'task_head.{name}' for name in tensors}
        legacy = {f'task_head.{name}' for name in params}
        if set(state) not in (expected, legacy):
            raise ValueError(
                f'task_head checkpoint keys {sorted(state)} do not match the head it '
                f'restores: expected {sorted(expected)}.')
        for key, value in state.items():
            tensor = tensors[key[len('task_head.'):]]
            if value.shape != tensor.shape:
                raise ValueError(f'task-head shape mismatch for {key}: {value.shape} != {tensor.shape}')
        for key, value in state.items():
            torch_util.copy_tensor_(tensors[key[len('task_head.'):]], value)

    @remote_function(dispatch='all', collect=_collect_temperatures, lazy_collect=False, sync=True)
    def task_head_temperatures(self, *, adapter_name: str = 'default', update: Optional[dict] = None):
        """Collectively read or replace head temperatures; every rank calls with the same update.

        Local torchrun bypasses the Ray collector, so the model process group also collects
        here. This gives head-less PP stages the same return dict as the head-owning stage
        and checks every rank's validation before any buffer is changed.
        """
        import torch.distributed as dist

        error, values, writes = None, None, []
        try:
            values, writes = self._prepare_task_head_temperatures(adapter_name, update)
        except Exception as exc:
            error = str(exc)
        states = [(error, values)]
        if dist.is_available() and dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, (error, values))
        errors = [message for message, _ in states if message is not None]
        if errors:
            raise ValueError(f'Task-head temperature validation failed: {errors[0]}')
        result = _collect_temperatures([value for _, value in states], self.device_mesh)
        if update is not None:
            for buffer, value in writes:
                torch_util.copy_tensor_(buffer, value)
        return result

    def _prepare_task_head_temperatures(self, adapter_name, update):
        """Validate local buffers and prepare writes without mutating any rank's head."""
        import torch

        if not self._task_head_carries_on_this_rank():
            return None, []
        head = self._attached_head(adapter_name)
        if head is None:
            if update is not None:
                raise ValueError('No task head is attached to this training handle.')
            return {'temperatures': None}, []
        buffers = dict(head.named_buffers(recurse=False))
        kind = buffers.get('temperatures', buffers.get('temps'))
        global_t = buffers.get('head_temperature')
        if kind is None and global_t is None and update is None:
            return {'temperatures': None}, []
        if (kind is None or kind.ndim != 1 or not kind.numel() or not kind.is_floating_point()
                or (global_t is not None and (global_t.ndim != 0 or not global_t.is_floating_point()))):
            raise ValueError('Expected a task-head temperature vector and an optional scalar buffer.')
        names = ('global_temperature', 'kind_temperatures')
        if update is not None and (not isinstance(update, dict) or set(update) != set(names)):
            raise ValueError(f'Temperature updates must contain exactly {names}.')
        values, writes = {}, []
        for name, buffer in zip(names, (global_t, kind)):
            if update is not None and (buffer is None) != (update[name] is None):
                raise ValueError(f'{name} must match the attached head buffer layout.')
            if buffer is None:
                values[name] = None
                continue
            value = (torch_util.to_local_tensor(buffer).detach().cpu() if update is None
                     else torch.as_tensor(update[name], dtype=buffer.dtype, device='cpu'))
            if value.shape != buffer.shape or not (value.isfinite() & (value > 0)).all():
                raise ValueError(f'{name} must have shape {tuple(buffer.shape)} and finite positive values.')
            values[name] = value.tolist()
            writes.append((buffer, value))
        return {'temperatures': values}, writes

    def _save_task_head_checkpoint(self, checkpoint_dir: str, adapter_name: Optional[str]) -> None:
        """Gather the attached head on participating ranks and write it on the master."""
        state = self._task_head_state(adapter_name)
        if state and self._task_head_is_master():
            write_task_head_checkpoint(checkpoint_dir, state)

    def _load_task_head_checkpoint(self, checkpoint_dir: str, adapter_name: Optional[str]) -> None:
        """Restore the attached head from task_head.safetensors; head-less legacy
        checkpoints (no file) load unchanged, a head checkpoint on a host without the
        head fails loudly rather than training a random-init one."""
        path = os.path.join(checkpoint_dir, 'task_head.safetensors')
        if not os.path.exists(path):
            return
        from safetensors.torch import load_file
        self._restore_task_head_state(adapter_name, load_file(path))

    @abstractmethod
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs) -> None:
        ...

    @abstractmethod
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs) -> None:
        ...

    @abstractmethod
    def get_train_configs(self, **kwargs) -> str:
        ...

    @staticmethod
    def _not_encoded(inputs):
        """True when ``inputs`` has not been run through the input processor yet.

        A packed batch arrives as list[list[InputFeature]]: PackingDataset yields a
        list of encoded rows per item and the processor flattens it only later
        (prepare_inputs), after this probe has run.  Descend to the first leaf
        dict so the probe reads a packed batch's encoding state instead of
        asserting on the intermediate list; for a flat dict / list[dict] the
        loop is a no-op.
        """
        while isinstance(inputs, (list, tuple)):
            if not inputs:
                return False
            inputs = inputs[0]
        assert isinstance(inputs, dict)
        # A row is encoded once it carries ANY tokenized field, not only a bare `input_ids`. Embedding /
        # reranker rows are GROUP-shaped: one row holds an anchor plus its candidates under prefixed keys
        # (`anchor_input_ids`, `positive_input_ids`, `negative0_input_ids`, ...) and the processor splits
        # it into flat per-sequence rows later (prepare_inputs). A raw Trajectory never carries a
        # `*input_ids` key (it has `messages` + media), so keying on the suffix distinguishes the two
        # without a bare-`input_ids`-only probe misreading a group row as raw and re-encoding it.
        return not any(
            k in ('input_ids', 'input_embedding') or k.endswith('_input_ids') or k.endswith('_input_embedding')
            for k in inputs)

    @staticmethod
    def _rollout_features(inputs, kwargs: Dict[str, Any], *, lift: bool) -> tuple:
        """Split trajectories handed over by a rollout into model features and the loss's columns.

        A rollout's merged trajectory (it carries ``response_token_ids``) holds its messages, episode
        facts, the sampler's per-token columns, and the training targets a write-back produced --
        all next to the features. Only the features go on to the processor; with ``lift`` the
        trajectory columns fill the kwargs the loss reads (``rollout_logps`` / ``sampling_masks``
        and the standard training-target names), unless the caller passed the same kwarg -- in
        which case both sides must agree or the call is rejected. Other inputs pass through.
        """
        rows = inputs if isinstance(inputs, list) else [inputs]
        if not rows or not isinstance(rows[0], dict) or 'response_token_ids' not in rows[0]:
            return inputs, kwargs
        _record_rollout_identity(rows)
        if lift:
            from twinkle.utils.torch_utils import values_agree
            kwargs = dict(kwargs)
            for field in _LIFTED_TRAJECTORY_FIELDS:
                row_values = [row.get(field) for row in rows]
                if field == 'truncated':
                    # an untruncated episode carries False and a row without the fact trains
                    # normally too -- only a True flag masks the loss (with overlong_filter on).
                    row_values = [bool(value) for value in row_values]
                if not any(value is not None for value in row_values):
                    continue
                existing = kwargs.get(field)
                if existing is None:
                    kwargs[field] = row_values
                elif not values_agree(existing, row_values):
                    # Both sides present: the explicit kwarg wins only when it repeats the same
                    # values; a silent override would mask a stale parallel list.
                    raise ValueError(
                        f'kwarg {field!r} is also set on the trajectories and disagrees; '
                        'pass one of them, not two different values')
        keys = InputProcessor.FEATURE_KEYS
        features = [{key: value for key, value in row.items() if key in keys} for row in rows]
        return (features if isinstance(inputs, list) else features[0]), kwargs

    # -- response-framed forward_only output (shared by both backends) ------------------

    ALLOWED_OUTPUT_FIELDS = frozenset({'logps', 'values', 'entropies', 'logits', 'embeddings', 'routed_experts'})

    @staticmethod
    def resolve_output_frame_request(kwargs: Dict[str, Any]) -> tuple:
        """Validate the ``output_frame`` / ``output_fields`` request riding a forward kwargs dict.

        Returns ``(output_frame, output_fields_or_None, task)`` without popping: the kwargs flow on
        to the single-batch / micro-batch callees, which pop the controls themselves.
        """
        output_frame = kwargs.get('output_frame', 'sequence')
        if output_frame not in ('sequence', 'response'):
            raise ValueError(f'output_frame must be "sequence" or "response", got {output_frame!r}')
        output_fields = kwargs.get('output_fields')
        if output_fields is not None:
            if isinstance(output_fields, str) or not isinstance(output_fields, (list, tuple, set, frozenset)):
                raise TypeError('output_fields must be a sequence of field names, not a bare string')
            unknown = set(output_fields) - TrainableModel.ALLOWED_OUTPUT_FIELDS
            if unknown:
                raise ValueError(f'output_fields names unknown fields {sorted(unknown)}; '
                                 f'allowed: {sorted(TrainableModel.ALLOWED_OUTPUT_FIELDS)}')
        task = kwargs.get('task', 'causal_lm')
        if output_frame == 'response' and not get_task_spec(task).token_level_labels:
            raise ValueError(f'task {task!r} has no token-level labels; output_frame="response" does not apply')
        if output_fields and 'embeddings' in output_fields and task != 'embedding':
            raise ValueError(f'output_fields requested "embeddings" but task {task!r} does not produce them')
        return output_frame, (frozenset(output_fields) if output_fields else None), task

    @staticmethod
    def tokenizer_fingerprint(tokenizer) -> Optional[str]:
        """A comparable identity for the tokenizer whose labels indexed a forward."""
        if tokenizer is None:
            return None
        name = getattr(tokenizer, 'name_or_path', None) or type(tokenizer).__name__
        return f'{name}#{getattr(tokenizer, "vocab_size", "")}'

    @staticmethod
    def _capture_response_rows(inputs) -> Optional[list]:
        """Row identity (episode_id / response_token_ids) captured before feature splitting.

        Returns ``None`` when no row carries identity (plain encoded features); the list stays
        aligned with the input rows so micro-batch slicing and DP dispatch keep it in step.
        """
        rows = inputs if isinstance(inputs, list) else [inputs]
        if not rows or not isinstance(rows[0], dict):
            return None
        if not any('episode_id' in row or 'response_token_ids' in row for row in rows):
            return None
        return [{'episode_id': row.get('episode_id'), 'response_token_ids': row.get('response_token_ids')}
                for row in rows]

    @staticmethod
    def project_response_frame(outputs: Dict[str, Any], *, labels, completion_mask, rows, tokenizer) -> Dict[str, Any]:
        """Project a forward_only public return copy onto per-sample response rows.

        Runs at the public boundary only -- after the processor's gather/unpack and the internal
        status recording, on the copied return value: the training chain's full-length tensors are
        untouched. Token-level channels (logps/values/entropies) become per-sample Python numeric
        lists over the response positions (``labels != -100`` intersected with ``completion_mask``
        when present); logits/embeddings/routed_experts keep their own frames, declared in
        ``field_frames``. Each row's derived response ids are cross-checked against the
        ``response_token_ids`` its input row declared.
        """
        if labels is None:
            raise ValueError('output_frame="response" requires labels in the forward batch')
        # Labels may be the batched [B, L] tensor (transformers single-batch path) or the per-row
        # list the megatron micro-batch adapter flattens; both index row-wise below.
        batch = len(labels) if isinstance(labels, (list, tuple)) else labels.shape[0]
        if rows is not None and len(rows) != batch:
            raise ValueError(f'response identity rows ({len(rows)}) do not match the forward batch ({batch})')
        positions_rows: list = []
        token_rows: list = []
        for i in range(batch):
            label_row = labels[i]
            mask = label_row != -100
            if completion_mask is not None:
                mask = mask & (completion_mask[i] != 0)
            positions = mask.nonzero(as_tuple=False).squeeze(-1)
            token_ids = label_row[positions].tolist()
            if rows is not None and rows[i].get('response_token_ids') is not None:
                declared = list(rows[i]['response_token_ids'])
                if declared != token_ids:
                    raise ValueError(
                        f'row {i} declared response_token_ids of {len(declared)} entries but the forward labels '
                        f'carry {token_ids}: the rows and this forward are not the same sequences')
            positions_rows.append(positions.tolist())
            token_rows.append(token_ids)
        projected = dict(outputs)
        frames: Dict[str, str] = {}
        for key in ('logps', 'values', 'entropies'):
            channel = outputs.get(key)
            if channel is None:
                continue
            rows_out = []
            for i, positions in enumerate(positions_rows):
                values = channel[i][positions]
                rows_out.append([float(v) for v in (values.tolist() if hasattr(values, 'tolist') else values)])
            projected[key] = rows_out
            frames[key] = 'response'
        if outputs.get('logits') is not None:
            frames['logits'] = 'sequence'
        if outputs.get('embeddings') is not None:
            frames['embeddings'] = 'row'
        if outputs.get('routed_experts') is not None:
            frames['routed_experts'] = 'sequence'
        projected['field_frames'] = frames
        projected['response_token_ids'] = token_rows
        if rows is not None:
            episode_ids = [row.get('episode_id') for row in rows]
            if any(episode_id is not None for episode_id in episode_ids):
                if any(episode_id is None for episode_id in episode_ids):
                    raise ValueError('inconsistent episode_id presence across input rows')
                projected['episode_ids'] = episode_ids
        projected['tokenizer_fingerprint'] = TrainableModel.tokenizer_fingerprint(
            getattr(tokenizer, 'tokenizer', tokenizer) if tokenizer is not None else None)
        return projected

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
    @timed
    def offload_to_cpu(self, *, offload_model: bool = True, offload_optimizer: bool = True,
                       gc_min_free_fraction: float | None = None) -> None:
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
        shape it reads. ``offload_model`` and ``offload_optimizer`` select which piece moves; both
        default on so existing callers still step the whole trainer aside. Passing both false leaves
        the parameters and optimizer resident and records nothing for :meth:`reload_to_gpu`, but still
        releases this rank's caching allocator: the colocated sampler is another process and cannot
        use blocks this process is only reserving. Megatron-FSDP still refuses when a piece is requested,
        because its parameter storage is not the DDP buckets this path knows how to move.

        ``@remote_function`` is what makes the handle reach the workers: under Ray the model on the
        driver is a proxy with no ``strategy`` of its own, so an undecorated method would run against
        that empty stub. ``dispatch='all'`` runs it on every rank (each offloads its own shard) and
        ``lazy_collect=False`` blocks until they finish -- the colocated sampler is a SEPARATE actor,
        so Ray's per-actor ordering cannot guarantee the GPU is actually free before the caller wakes
        the engine and lets it allocate its KV cache.
        """
        if not offload_model and not offload_optimizer:
            from twinkle.model.offload import record_offload, release_caching_allocator
            record_offload(self.strategy, model=False, optimizer=False)
            if gc_min_free_fraction is None:
                release_caching_allocator()
            else:
                release_caching_allocator(min_free_fraction=gc_min_free_fraction)
            return
        optimizer = self._colocation_optimizer() if offload_optimizer else None
        if offload_model:
            # Keep the historical positional call so strategies and tests that only accept
            # (model, optimizer) keep working when both pieces move.
            self.strategy.offload_to_cpu(self.model, optimizer)
        else:
            self.strategy.offload_to_cpu(self.model, optimizer, offload_model=False)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    @timed
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
        # A just-frozen tower must stop building a graph (legacy #10214); backends without the helper skip it.
        patch_frozen_towers = getattr(self, '_patch_frozen_towers', None)
        if patch_frozen_towers is not None:
            for module in modules:
                patch_frozen_towers(module)

    def _should_bind_device_id_for_process_group(self, backend: str) -> bool:
        # Platform hook: CUDA/NPU bind lazily-built communicators to the device;
        # XPU's BKCL crashes on a bound id in later subgroup splits.
        return Platform.get_platform().binds_device_id_to_process_group()

    def _try_init_process_group(self, force: bool = False):
        """Initialise the default process group if nothing has yet.

        ``force`` serves the Megatron backends: mcore's
        ``initialize_model_parallel`` asserts an initialised group even at
        world_size==1, so a single-rank MegatronModel must init unconditionally
        (the ray worker env carries MASTER_ADDR/PORT/RANK/WORLD_SIZE for every
        group size; torchrun does the same locally).
        """
        from twinkle.utils.torch_utils import ensure_default_process_group

        ensure_default_process_group(
            force=force, should_bind_device_id=self._should_bind_device_id_for_process_group)
