# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

from .input_feature import InputFeature, InputType
from .message import Message, Tool


class Trajectory(InputFeature, total=False):
    """The single public sample payload: conversation, encoding, and RL numeric fields.

    Token-level numeric fields are response-framed: their length is always the
    trajectory's response width (``len(response_token_ids)``). Row-level facts
    (rewards outcome scalars, identity) never masquerade as token frames.
    """
    messages: List[Message]
    tools: List[Tool]
    # PyArrow-stable encoding: each entry is (key, json.dumps(value)). Use the helpers below.
    user_data: List[Tuple[str, str]]
    images: Optional[List[Union[str, Any]]]
    videos: Optional[List[Union[str, Any]]]
    audios: Optional[List[Union[str, Any]]]
    prompt: Optional[str]
    # Identity: episode_id unique per episode; batch_id carries source identity + hand-out index.
    episode_id: str
    group_key: Any
    batch_id: str
    policy_version: int
    role: str
    # Response frame: response_positions address shifted labels; labels[positions] == token ids.
    response_token_ids: List[int]
    response_positions: List[int]
    tokenizer_fingerprint: Optional[str]
    # Sampling: rollout_logps is response-framed; engines without logps leave it absent.
    rollout_logps: List[float]
    sampling_masks: InputType
    # Rewards: orm/env/baseline row-level named entries; process_rewards response-framed per scorer.
    rewards: Dict[str, Any]
    process_rewards: Dict[str, List[float]]
    prm_segments: Any
    # Training targets: response-framed; GRPO scalar advantages are broadcast by their producer.
    advantages: List[float]
    returns: List[float]
    old_logps: List[float]
    ref_logps: List[float]
    old_values: List[float]
    # Teacher signals: single-teacher logps response-framed; multi-teacher is K x T_i per trajectory.
    teacher_logps: List[float]
    multi_teacher_logps: List[List[float]]
    teacher_ids: List[str]
    teacher_weights: List[float]
    # Episode facts.
    truncated: bool
    stop_reason: Any
    turns: int


# The conversational half of a trajectory: what another model re-encodes with its own template.
CONVERSATION_FIELDS = ('messages', 'tools', 'user_data', 'images', 'videos', 'audios', 'prompt')

# Token-level fields that always share the response frame; length is asserted, never inferred.
RESPONSE_FRAME_FIELDS = frozenset({
    'response_token_ids', 'rollout_logps', 'ref_logps', 'old_logps', 'teacher_logps',
    'advantages', 'returns', 'old_values', 'sampling_masks',
})


def pack_value(value: Any) -> str:
    """Encode a single user_data value to a JSON string."""
    return json.dumps(value, ensure_ascii=False, default=str)


def pack_user_data(values: Any) -> List[Tuple[str, str]]:
    """Build a canonical ``user_data`` payload from a plain mapping.

    ``user_data`` must be a list of ``(key, json_string)`` pairs: a dict cannot be written to
    PyArrow (its struct schema would differ from shard to shard), and readers go through
    :func:`user_data_get`, which only understands the packed form — a dict is silently ignored
    there, so e.g. ``key_rounds`` would be dropped without any error. Always pack before
    attaching to a trajectory.
    """
    if values is None:
        return []
    if isinstance(values, Mapping):
        return [(k, v if isinstance(v, str) else pack_value(v)) for k, v in values.items()]
    return [(k, v if isinstance(v, str) else pack_value(v)) for k, v in values]


def attach_user_data(trajectory: Trajectory, **values: Any) -> Trajectory:
    """Return ``trajectory`` with ``values`` merged into its packed ``user_data``.

    ``user_data`` is a list of ``(key, json_string)`` pairs rather than a dict,
    so it cannot be updated in place with ``update()``; going through
    :func:`pack_user_data` keeps it in the one shape readers understand.
    """
    merged: Dict[str, Any] = {}
    for entry in trajectory.get('user_data') or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            merged[entry[0]] = entry[1]
    merged.update(values)
    out = dict(trajectory)
    out['user_data'] = pack_user_data(merged)
    return out


def ground_truth(trajectory: Trajectory, key: str = 'ground_truth') -> Any:
    """A trajectory's gold label: the packed ``user_data`` entry first, the flat column second.

    Both spellings carry the same label -- a twinkle pipeline packs ``user_data``, a swift scoring
    row keeps dataset columns flat -- so a Reward reads one helper and works in either world.
    """
    value = user_data_get(trajectory.get('user_data'), key)
    if value is None:
        value = trajectory.get(key)
    return value


def msg_content_text(msg: Dict[str, Any]) -> str:
    """Extract plain text from a message's content (str | list | dict)."""
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(p.get('text', '') for p in c if isinstance(p, dict) and p.get('type') == 'text')
    if isinstance(c, dict) and c.get('type') == 'text':
        return c.get('text', '')
    return ''


def last_assistant_text(trajectory: Trajectory) -> str:
    """The newest assistant turn's text -- the completion a reward scores.

    Multimodal messages carry content as a list of parts, so text goes through
    :func:`msg_content_text` instead of being read as a plain string. The *last* assistant
    turn: a conversation that went through tools has several, and the model's answer is
    the turn it finished on.
    """
    for message in reversed(trajectory.get('messages') or []):
        if isinstance(message, dict) and message.get('role') == 'assistant':
            return msg_content_text(message)
    return ''


def conversation(trajectory: Trajectory) -> Trajectory:
    """Only the conversational fields of ``trajectory``, without any model's encoding of it.

    A rollout's trajectory also carries the policy's token fields; a different model (a reward
    model, a judge) has to re-encode the conversation with its own template instead.
    """
    return {key: trajectory[key] for key in CONVERSATION_FIELDS if key in trajectory}


def assert_response_frame(trajectory: Trajectory, field: str) -> None:
    """Raise unless ``field`` (a response-frame numeric) matches this trajectory's response width.

    Length is asserted against ``response_token_ids`` only -- never against the full input, and
    never used to decide what a field means: the field name declares its frame.
    """
    values = trajectory.get(field)
    if values is None:
        raise ValueError(f'{field} is missing on trajectory {trajectory.get("episode_id")!r}')
    expected = len(trajectory.get('response_token_ids') or ())
    if len(values) != expected:
        raise ValueError(f'{field} has {len(values)} entries but trajectory '
                         f'{trajectory.get("episode_id")!r} has {expected} response tokens')


def validate_trajectory(trajectory: Trajectory) -> None:
    """Assert the response contract of a completed (encoded, rolled-out) trajectory.

    A trajectory without response fields passes untouched: dataset rows before rollout never
    enter this validation face.
    """
    token_ids = trajectory.get('response_token_ids')
    if token_ids is None:
        return
    positions = trajectory.get('response_positions')
    if positions is None:
        raise ValueError(f'trajectory {trajectory.get("episode_id")!r} has response tokens '
                         'without response_positions')
    if len(token_ids) != len(positions):
        raise ValueError(f'response_token_ids ({len(token_ids)}) and response_positions '
                         f'({len(positions)}) disagree on trajectory {trajectory.get("episode_id")!r}')
    labels = trajectory.get('labels')
    if labels is not None:
        # The shifted-labels contract: every trainable position's label IS the response token.
        for pos, token in zip(positions, token_ids):
            if labels[pos] != token:
                raise ValueError(f'labels[{pos}] != response_token_ids entry on trajectory '
                                 f'{trajectory.get("episode_id")!r}')
    for field in RESPONSE_FRAME_FIELDS:
        if field in trajectory and trajectory[field] is not None:
            assert_response_frame(trajectory, field)


def _input_order_labels(labels: List[int]) -> List[int]:
    """Undo twinkle's output-order roll to get labels aligned with ``input_ids``.

    ``append_ids`` stores labels shifted, so ``labels[i]`` is the target for position
    ``i + 1``; shifting right by one recovers input order, where a trainable position
    ``j`` holds ``in_labels[j] == input_ids[j]``.
    """
    return (labels[-1:] + labels[:-1]) if labels else []


def _trainable_positions(input_ids: List[int], labels: List[int], completion_mask: Any) -> List[int]:
    """Input-order positions of the policy-produced tokens, in order.

    ``labels != -100`` intersected with ``completion_mask`` when one is present. Labels AND
    the mask both arrive in output/shifted order, so both are unrolled before the
    position-wise intersection.
    """
    in_labels = _input_order_labels(list(labels or []))
    mask = (
        _input_order_labels(list(completion_mask))
        if completion_mask is not None and len(completion_mask) == len(in_labels)
        else None
    )
    positions: List[int] = []
    for j, lab in enumerate(in_labels):
        if lab == -100:
            continue
        if mask is not None and not mask[j]:
            continue
        positions.append(j)
    return positions


def trainable_runs(encoded: Dict[str, Any]) -> List[List[int]]:
    """Contiguous runs of trainable input_ids positions, one run per assistant turn.

    The ledger records a whole generation as trainable and every observation as
    masked, so the runs alternate one-to-one with the assistant turns.
    """
    input_ids = list(encoded.get('input_ids') or [])
    labels = list(encoded.get('labels') or [])
    completion_mask = encoded.get('completion_mask')
    positions = _trainable_positions(input_ids, labels, completion_mask)
    runs: List[List[int]] = []
    for j in positions:
        if runs and j == runs[-1][-1] + 1:
            runs[-1].append(j)
        else:
            runs.append([j])
    return runs


def turn_token_spans(encoded: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Per-assistant-turn ``[start, end)`` spans within the flat response tokens.

    The k-th contiguous trainable run occupies a contiguous slice of the flat
    response array, so this returns those slices as half-open spans -- the turn
    boundaries multi-turn PRM step segmentation needs.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for run in trainable_runs(encoded):
        spans.append((cursor, cursor + len(run)))
        cursor += len(run)
    return spans


def user_data_get(items: Any, key: str, default: Any = None) -> Any:
    """Look up the first value matching ``key`` in packed user_data, decoded."""
    if isinstance(items, Mapping):
        raise TypeError('user_data must be a list of (key, json_string) pairs, got a mapping. '
                        'Wrap it with twinkle.data_format.pack_user_data().')
    if not isinstance(items, list):
        return default
    for entry in items:
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == key:
            v = entry[1]
            if not isinstance(v, str):
                return v
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return v
    return default
