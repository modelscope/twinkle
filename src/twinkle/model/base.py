# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, Type, Union

from twinkle import Platform, remote_function
from twinkle.data_format import InputFeature, ModelOutput
from twinkle.hub import HubOperation
from twinkle.loss.base import Loss
from twinkle.metric import Metric
from twinkle.model.tasks import get_task_spec, get_task_spec_obj
from twinkle.patch import Patch
from twinkle.perf import timed
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


# Trajectory fields ``TrainableModel._rollout_features`` lifts into loss kwargs on training calls.
# The names match the loss kwargs exactly; token-level fields are response-framed on both sides.
_LIFTED_TRAJECTORY_FIELDS = (
    'rollout_logps', 'sampling_masks', 'advantages', 'returns',
    'ref_logps', 'teacher_logps', 'old_logps', 'old_values', 'truncated',
)


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
        future = self._async_write_future
        if future is not None:
            self._async_write_future = None
            future.result()

    def shutdown_async_saves(self) -> None:
        """Join any in-flight write and retire the writer thread (idempotent)."""
        self._join_async_save()
        executor = self._async_write_executor
        if executor is not None:
            self._async_write_executor = None
            executor.shutdown(wait=True)

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
    def memory_stats(self, *, synchronize: bool = False) -> dict[str, int]:
        """Raw per-rank allocator peaks and host RAM in bytes; no cross-rank reduction."""
        from twinkle.perf.stats import memory_stats

        return memory_stats(synchronize=synchronize)

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
        """The adapter's head on the host: the task_heads dict when the host keeps one,
        else the single task_head attribute (megatron single-model form)."""
        host = self._task_head_host()
        heads = getattr(host, 'task_heads', None)
        if heads is None:
            return getattr(host, 'task_head', None)
        return heads.get(adapter_name)

    def _task_head_state(self, adapter_name: Optional[str]) -> Dict[str, Any]:
        """The adapter's head weights under slot-independent 'task_head.' keys.

        Default (single-model) form reads the head hanging on the host; MultiLora
        hosts override this with their slot-pool export."""
        head = self._attached_head(adapter_name)
        if head is None:
            return {}
        return {f'task_head.{name}': p.detach().clone().cpu() for name, p in head.named_parameters()}

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
        import torch
        head = self._attached_head(adapter_name)
        if head is None:
            if state and self._task_head_carries_on_this_rank():
                raise ValueError(
                    f'task_head checkpoint has keys {sorted(state)} but adapter {adapter_name!r} '
                    'carries no head to restore them into.')
            return
        params = dict(head.named_parameters())
        expected = {f'task_head.{name}' for name in params}
        if set(state) != expected:
            raise ValueError(
                f'task_head checkpoint keys {sorted(state)} do not match the head it '
                f'restores: expected {sorted(expected)}.')
        with torch.no_grad():
            for name, param in params.items():
                value = state[f'task_head.{name}'].to(device=param.device, dtype=param.dtype)
                param.data.copy_(value)

    def _save_task_head_checkpoint(self, checkpoint_dir: str, adapter_name: Optional[str]) -> None:
        """Write the adapter's attached head to its own task_head.safetensors.

        Keyed 'task_head.<param>' -- slot-independent, so a re-added adapter may land
        on a different slot and still restore. Runs on the master rank only; the head
        file is tiny, so it writes synchronously alongside the (async) weights."""
        state = self._task_head_state(adapter_name)
        if state and self._task_head_is_master():
            from safetensors.torch import save_file
            save_file({key: value.contiguous() for key, value in state.items()},
                      os.path.join(checkpoint_dir, 'task_head.safetensors'))

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
        if lift:
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
                elif existing != row_values:
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
