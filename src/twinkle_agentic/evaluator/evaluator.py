"""The lightweight, single-use EvalScope facade."""

from copy import deepcopy
from enum import Enum
from threading import Lock
from typing import Any, Mapping, Sequence

from twinkle_agentic.protocol.base import API

from ._contracts import EvaluatorConfigError


_OWNED_TASK_KEYS = {
    'model', 'model_id', 'datasets', 'eval_type', 'eval_backend', 'model_task', 'api_url', 'api_key', 'model_args',
}


class _State(Enum):
    NEW = 'new'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'


def _copy_mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluatorConfigError(f'{name} must be a mapping')
    return deepcopy(dict(value))


def _explicit_generation_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set(value)
    return set(getattr(value, 'model_fields_set', set()))


class Evaluator:
    """Evaluate one Twinkle API or sampler through EvalScope's native runner."""

    def __init__(
        self,
        *,
        datasets: Sequence[str],
        sampler: object | None = None,
        api: API | None = None,
        model_id: str | None = None,
        template: object | None = None,
        sampler_kwargs: Mapping[str, Any] | None = None,
        task_config: Mapping[str, Any] | None = None,
        sampler_batch_size: int | None = None,
        sampler_batch_wait_ms: float = 5.0,
    ) -> None:
        if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence) or not datasets:
            raise EvaluatorConfigError('datasets must be a non-empty sequence of non-empty strings')
        self._datasets = list(datasets)
        if any(not isinstance(item, str) or not item.strip() for item in self._datasets):
            raise EvaluatorConfigError('datasets must contain only non-empty strings')
        if (sampler is None) == (api is None):
            raise EvaluatorConfigError('provide exactly one of sampler or api')
        if api is not None and not isinstance(api, API):
            raise EvaluatorConfigError('api must implement twinkle_agentic.protocol.base.API')
        self._sampler = sampler
        self._api = api
        self._task_config = _copy_mapping(task_config, 'task_config')
        conflicts = sorted(_OWNED_TASK_KEYS.intersection(self._task_config))
        if conflicts:
            names = ', '.join(conflicts)
            raise EvaluatorConfigError(f'task_config cannot set Twinkle-owned field(s): {names}')
        self._sampler_kwargs = _copy_mapping(sampler_kwargs, 'sampler_kwargs')
        if not isinstance(sampler_batch_wait_ms, (int, float)) or sampler_batch_wait_ms < 0:
            raise EvaluatorConfigError('sampler_batch_wait_ms must be >= 0')
        if sampler_batch_size is not None and (not isinstance(sampler_batch_size, int) or sampler_batch_size < 1):
            raise EvaluatorConfigError('sampler_batch_size must be an integer >= 1')
        if api is not None and (template is not None or self._sampler_kwargs or sampler_batch_size is not None
                                or sampler_batch_wait_ms != 5.0):
            raise EvaluatorConfigError('template and sampler batching options are only valid with sampler')
        self._template = template if sampler is not None else None
        if self._template is None and sampler is not None:
            self._template = getattr(sampler, 'template', None)
        inferred = model_id or getattr(sampler, 'model_id', None) or getattr(api, 'model', None) or getattr(api, 'model_name', None)
        if not isinstance(inferred, str) or not inferred.strip():
            raise EvaluatorConfigError('model_id is required when it cannot be inferred from sampler.model_id or api.model')
        self._model_id = inferred
        self._sampler_batch_size = sampler_batch_size or self._task_config.get('eval_batch_size', 8)
        if not isinstance(self._sampler_batch_size, int) or self._sampler_batch_size < 1:
            raise EvaluatorConfigError('task_config.eval_batch_size must be an integer >= 1')
        self._sampler_batch_wait_ms = float(sampler_batch_wait_ms)
        self._generation_keys = _explicit_generation_keys(self._task_config.get('generation_config', {}))
        self._state = _State.NEW
        self._state_lock = Lock()
        self._resolved_task_config: Any = None
        self._output_dir: str | None = None

    @property
    def resolved_task_config(self) -> Any:
        return self._resolved_task_config

    @property
    def output_dir(self) -> str | None:
        return self._output_dir

    def run(self) -> Any:
        with self._state_lock:
            if self._state is not _State.NEW:
                raise RuntimeError('Evaluator instances are single-use; create a new Evaluator to run again')
            self._state = _State.RUNNING
        batcher = None
        try:
            try:
                from ._evalscope_adapter import ProtocolModelAPI, SamplerModelAPI
                from evalscope.config import TaskConfig
                from evalscope.constants import EvalBackend, EvalType
                from evalscope.run import run_task
            except ImportError as exc:
                raise ImportError("Evaluator requires EvalScope. Install it with:\n  pip install 'twinkle-kit[eval]'") from exc
            if self._api is not None:
                adapter = ProtocolModelAPI(self._api, self._model_id, self._generation_keys)
            else:
                adapter = SamplerModelAPI(
                    self._sampler,
                    self._model_id,
                    self._template,
                    self._generation_keys,
                    batch_size=self._sampler_batch_size,
                    batch_wait_ms=self._sampler_batch_wait_ms,
                    sampler_kwargs=self._sampler_kwargs,
                )
                batcher = adapter.batcher
            config = dict(self._task_config)
            config.setdefault('eval_batch_size', 8)
            config.update({
                'model': adapter,
                'model_id': self._model_id,
                'datasets': list(self._datasets),
                'eval_type': EvalType.CUSTOM,
                'eval_backend': EvalBackend.NATIVE,
                'model_task': 'text_generation',
            })
            self._resolved_task_config = TaskConfig(**config)
            adapter.validate_generation_config(self._resolved_task_config.generation_config)
            result = run_task(self._resolved_task_config)
            self._output_dir = self._resolved_task_config.work_dir
        except Exception:
            with self._state_lock:
                self._state = _State.FAILED
            raise
        else:
            with self._state_lock:
                self._state = _State.SUCCEEDED
            return result
        finally:
            if batcher is not None:
                batcher.close()
