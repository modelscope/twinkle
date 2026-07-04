import functools
import hashlib
import inspect
import json
import os
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class EvalRecord:
    """A single evaluation record comparing student and teacher outputs."""
    student_result: Any
    teacher_result: Any
    match: bool


@dataclass
class DistillationState:
    """Per-key state tracking confidence, dataset, and call count."""
    confidence: Optional[float] = None
    dataset: List[EvalRecord] = field(default_factory=list)
    call_count: int = 0


class DistillationRegistry:
    """Thread-safe registry maintaining distillation state per unique key."""

    def __init__(self):
        self._states: Dict[str, DistillationState] = defaultdict(DistillationState)
        self._lock = threading.Lock()

    def get_confidence(self, key: str) -> float:
        """Get confidence for a key. Returns 0.0 if no data available."""
        with self._lock:
            state = self._states[key]
            if state.confidence is not None:
                return state.confidence
            if state.dataset:
                state.confidence = self._compute_confidence(state.dataset)
                return state.confidence
            return 0.0

    def increment_call(self, key: str) -> int:
        with self._lock:
            state = self._states[key]
            state.call_count += 1
            return state.call_count

    def add_record(self, key: str, student_result: Any, teacher_result: Any, match: bool):
        with self._lock:
            state = self._states[key]
            state.dataset.append(EvalRecord(
                student_result=student_result,
                teacher_result=teacher_result,
                match=match,
            ))

    def refresh_confidence(self, key: str) -> float:
        with self._lock:
            state = self._states[key]
            if state.dataset:
                state.confidence = self._compute_confidence(state.dataset)
            else:
                state.confidence = 0.0
            return state.confidence

    @staticmethod
    def _compute_confidence(dataset: List[EvalRecord]) -> float:
        if not dataset:
            return 0.0
        matches = sum(1 for r in dataset if r.match)
        return matches / len(dataset)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_registry = DistillationRegistry()
_teacher_api = None
_teacher_lock = threading.Lock()


def _get_teacher_api():
    """Lazy-init global teacher API from environment variables.

    Env vars:
        LLM_BACKUP_MODEL: Model name (default: "gpt-4o")
        LLM_BACKUP_API_KEY: API key
        LLM_BACKUP_BASE_URL: Base URL for OpenAI-compatible endpoint
    """
    global _teacher_api
    if _teacher_api is not None:
        return _teacher_api
    with _teacher_lock:
        if _teacher_api is not None:
            return _teacher_api
        from twinkle_agentic.protocol.openai import OpenAI
        _teacher_api = OpenAI(
            model=os.environ.get('LLM_BACKUP_MODEL', 'gpt-4o'),
            api_key=os.environ.get('LLM_BACKUP_API_KEY'),
            base_url=os.environ.get('LLM_BACKUP_BASE_URL'),
        )
        return _teacher_api


def _call_teacher(trajectory, sampling_params) -> str:
    """Call teacher API and extract raw content string."""
    api = _get_teacher_api()
    message = api(trajectory, sampling_params)
    if isinstance(message, list):
        message = message[0]
    return message.get('content', '') if isinstance(message, dict) else ''


# ---------------------------------------------------------------------------
# Key building
# ---------------------------------------------------------------------------
def _build_key(func_name: str, args: tuple, kwargs: dict,
               param_names: List[str], key_params: Sequence[str]) -> str:
    """Build a unique key from specified parameter values."""
    key_parts = [func_name]
    for i, name in enumerate(param_names):
        if name in key_params:
            if i < len(args):
                key_parts.append(f"{name}={_serialize_value(args[i])}")
            elif name in kwargs:
                key_parts.append(f"{name}={_serialize_value(kwargs[name])}")
    for name in key_params:
        if name not in param_names[:len(args)] and name in kwargs:
            if f"{name}={_serialize_value(kwargs[name])}" not in key_parts:
                key_parts.append(f"{name}={_serialize_value(kwargs[name])}")
    raw_key = "|".join(key_parts)
    return hashlib.md5(raw_key.encode()).hexdigest()


def _serialize_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _extract_param(args: tuple, kwargs: dict, param_names: List[str], name: str) -> Any:
    """Extract a named parameter from args/kwargs given the signature's param_names."""
    if name in kwargs:
        return kwargs[name]
    for i, pname in enumerate(param_names):
        if pname == name and i < len(args):
            return args[i]
    return None


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------
def llm_backup(
    key_params: Sequence[str],
    comparator: Optional[Callable[[Any, Any], bool]] = None,
    sample_rate: float = 0.2,
    refresh_env_var: str = "LLM_BACKUP_REFRESH_INTERVAL",
    default_refresh_interval: int = 50,
):
    """Decorator for progressive distillation from teacher API to student model.

    The decorated function is the STUDENT (local model sampling). The TEACHER
    is a global OpenAI-compatible API constructed from environment variables.

    The decorated function MUST accept ``trajectory`` and ``sampling_params``
    as parameters (by name) and return a raw string. This ensures:
    - Teacher and student receive identical inputs
    - The dataset contains raw (trajectory, student_output, teacher_output) tuples
    - No pre/post processing is included, making data directly trainable

    Routing logic:
        - confidence% -> use student (decorated fn)
            - Of those, sample_rate% also call teacher for comparison
        - (1 - confidence)% -> use teacher API
            - Always also call student for comparison

    Every N calls the confidence is recalculated from the comparison dataset.

    Environment variables:
        LLM_BACKUP_MODEL: Teacher model name (default "gpt-4o")
        LLM_BACKUP_API_KEY: Teacher API key
        LLM_BACKUP_BASE_URL: Teacher API base URL
        LLM_BACKUP_REFRESH_INTERVAL: Confidence refresh interval N (default 50)

    Args:
        key_params: Parameter names for unique confidence key (e.g. ["query"]).
        comparator: function(student, teacher) -> bool. Default: equality.
        sample_rate: Probability of teacher verification when student is used.
        refresh_env_var: Env var name for refresh interval.
        default_refresh_interval: Default refresh interval.

    Example:
        >>> @llm_backup(key_params=["query"])
        ... def _sample(self, trajectory, sampling_params, query=None) -> str:
        ...     responses = self.sampler.sample([trajectory], ...)
        ...     return decode(responses[0])
    """
    if comparator is None:
        comparator = lambda a, b: a == b  # noqa: E731

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = _build_key(fn.__qualname__, args, kwargs, param_names, key_params)
            confidence = _registry.get_confidence(key)

            try:
                refresh_interval = int(os.environ.get(refresh_env_var, default_refresh_interval))
            except (ValueError, TypeError):
                refresh_interval = default_refresh_interval

            # Extract trajectory and sampling_params for teacher call
            trajectory = _extract_param(args, kwargs, param_names, 'trajectory')
            sampling_params = _extract_param(args, kwargs, param_names, 'sampling_params')

            roll = random.random()
            use_student = roll < confidence

            if use_student:
                # High confidence: trust student
                result = fn(*args, **kwargs)
                # Occasionally verify against teacher
                if random.random() < sample_rate:
                    teacher_result = _call_teacher(trajectory, sampling_params)
                    match = comparator(result, teacher_result)
                    _registry.add_record(key, result, teacher_result, match)
                    if not match:
                        result = teacher_result
            else:
                # Low confidence: use teacher
                teacher_result = _call_teacher(trajectory, sampling_params)
                student_result = fn(*args, **kwargs)
                match = comparator(student_result, teacher_result)
                _registry.add_record(key, student_result, teacher_result, match)
                result = teacher_result

            call_count = _registry.increment_call(key)
            if refresh_interval > 0 and call_count % refresh_interval == 0:
                _registry.refresh_confidence(key)

            return result

        wrapper._registry = _registry
        return wrapper

    return decorator


def llm_backup_async(
    key_params: Sequence[str],
    comparator: Optional[Callable[[Any, Any], bool]] = None,
    sample_rate: float = 0.2,
    refresh_env_var: str = "LLM_BACKUP_REFRESH_INTERVAL",
    default_refresh_interval: int = 50,
):
    """Async version of llm_backup. Same semantics."""
    if comparator is None:
        comparator = lambda a, b: a == b  # noqa: E731

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            key = _build_key(fn.__qualname__, args, kwargs, param_names, key_params)
            confidence = _registry.get_confidence(key)

            try:
                refresh_interval = int(os.environ.get(refresh_env_var, default_refresh_interval))
            except (ValueError, TypeError):
                refresh_interval = default_refresh_interval

            trajectory = _extract_param(args, kwargs, param_names, 'trajectory')
            sampling_params = _extract_param(args, kwargs, param_names, 'sampling_params')

            roll = random.random()
            use_student = roll < confidence

            if use_student:
                result = await fn(*args, **kwargs)
                if random.random() < sample_rate:
                    teacher_result = _call_teacher(trajectory, sampling_params)
                    match = comparator(result, teacher_result)
                    _registry.add_record(key, result, teacher_result, match)
                    if not match:
                        result = teacher_result
            else:
                teacher_result = _call_teacher(trajectory, sampling_params)
                student_result = await fn(*args, **kwargs)
                match = comparator(student_result, teacher_result)
                _registry.add_record(key, student_result, teacher_result, match)
                result = teacher_result

            call_count = _registry.increment_call(key)
            if refresh_interval > 0 and call_count % refresh_interval == 0:
                _registry.refresh_confidence(key)

            return result

        wrapper._registry = _registry
        return wrapper

    return decorator
