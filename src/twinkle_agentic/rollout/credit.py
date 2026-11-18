# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional state and tool provenance for multi-turn credit assignment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

STATE_KEY_INFO = 'twinkle.state_key'
TERMINAL_SUCCESS_INFO = 'twinkle.terminal_success'
TOOL_EVENT_ID_INFO = 'twinkle.tool_event_id'
MAX_STATE_KEY_CHARS = 256

__all__ = [
    'ObservationSpan',
    'STATE_KEY_INFO',
    'TERMINAL_SUCCESS_INFO',
    'TOOL_EVENT_ID_INFO',
    'ToolEvent',
    'TurnActionSpan',
    'TurnCreditRecord',
    'TurnCreditState',
    'make_tool_events',
    'normalize_state_key',
    'qualified_state_key',
    'state_from_tool_infos',
]


@dataclass(frozen=True)
class TurnActionSpan:
    """Trainable action positions in the final labels index space."""

    start: int
    end: int


@dataclass(frozen=True)
class ObservationSpan:
    """Masked positions appended between two generation turns."""

    start: int
    end: int


@dataclass(frozen=True)
class ToolEvent:
    tool_call_id: str
    tool_name: str
    arguments_digest: str
    result_digest: str
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnCreditRecord:
    turn_id: int
    action_token_start: int
    action_token_end: int
    observation_token_start: int | None
    observation_token_end: int | None
    pre_state_key: str
    post_state_key: str
    tool_events: tuple[ToolEvent, ...]
    policy_version: int | str | None
    terminal_success: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TurnCreditState:
    available: bool
    fallback_reason: str | None = None


def normalize_state_key(value: Any) -> str | None:
    """Validate one environment-provided state key."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f'state_key must be a string or None, got {type(value).__name__}')
    key = value.strip()
    if not key:
        raise ValueError('state_key must not be empty')
    if len(key) > MAX_STATE_KEY_CHARS:
        raise ValueError(f'state_key exceeds {MAX_STATE_KEY_CHARS} characters')
    if any(ord(char) < 32 for char in key):
        raise ValueError('state_key must not contain control characters')
    return key


def qualified_state_key(
    environment_id: str | None,
    task_id: str | None,
    local_state_key: Any,
) -> str | None:
    """Build the rollout-scoped key expected by the state graph."""
    if not environment_id or not task_id:
        return None
    local = normalize_state_key(local_state_key)
    if local is None:
        return None
    return f'{environment_id}:{task_id}:{local}'


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _tool_call_parts(tool_call: Any, index: int) -> tuple[str, str, Any, str]:
    call = tool_call if isinstance(tool_call, Mapping) else {}
    call_id = str(call.get('id') or call.get('tool_call_id') or f'tool-{index}')
    function = call.get('function') if isinstance(call.get('function'), Mapping) else {}
    name = str(function.get('name') or '')
    arguments = function.get('arguments', {})
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            decoded = arguments
    else:
        decoded = arguments
    return call_id, name, decoded, _digest(decoded)


def make_tool_events(
    tool_calls: Sequence[Any],
    results: Sequence[Any],
    *,
    turn_id: int,
) -> tuple[ToolEvent, ...]:
    """Pair tool calls with observations or named error results."""
    if len(tool_calls) != len(results):
        raise ValueError('tool_calls and results must have equal length')
    events = []
    for index, (call, result) in enumerate(zip(tool_calls, results)):
        call_id, name, _, arguments_digest = _tool_call_parts(call, index)
        if isinstance(result, Mapping):
            observation = result.get('observation', '')
            info = dict(result.get('info') or {})
        else:
            observation = getattr(result, 'observation', result)
            info = dict(getattr(result, 'info', {}) or {})
        info.setdefault(TOOL_EVENT_ID_INFO, f'turn-{turn_id}/call-{call_id}')
        events.append(
            ToolEvent(
                tool_call_id=call_id,
                tool_name=name,
                arguments_digest=arguments_digest,
                result_digest=_digest(str(observation)),
                info=info,
            ))
    return tuple(events)


def state_from_tool_infos(infos: Iterable[Mapping[str, Any]], ) -> tuple[str | None, bool | None, str | None]:
    """Return one state and terminal flag, or a conflict reason."""
    state_keys = []
    successes = []
    for info in infos:
        raw_state = info.get(STATE_KEY_INFO) if isinstance(info, Mapping) else None
        if raw_state is not None:
            state_keys.append(normalize_state_key(raw_state))
        raw_success = info.get(TERMINAL_SUCCESS_INFO) if isinstance(info, Mapping) else None
        if raw_success is not None:
            if not isinstance(raw_success, bool):
                raise TypeError(f'{TERMINAL_SUCCESS_INFO} must be bool when present')
            successes.append(raw_success)
    unique_states = set(state_keys)
    unique_successes = set(successes)
    if len(unique_states) > 1:
        return None, None, 'state_conflict'
    if len(unique_successes) > 1:
        return None, None, 'terminal_success_conflict'
    return (
        next(iter(unique_states)) if unique_states else None,
        next(iter(unique_successes)) if unique_successes else None,
        None,
    )
