# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import re
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, cast

from .types import EncodedRow, InputOrderRow, MaxLengthError, MediaBundle
from twinkle.data_format import Trajectory

SHIFTED_FIELDS: Mapping[str, Any] = MappingProxyType(
    {
        "labels": -100,
        "loss_scale": 0.0,
        "completion_mask": 0,
    }
)
TOKEN_FILL: Mapping[str, int] = MappingProxyType(
    {
        "token_type_ids": 0,
        "mm_token_type_ids": 0,
    }
)
_MEDIA_TAG = re.compile(r"<(?:image|video|audio)>")
_MEDIA_PAYLOAD_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "input_features",
        "audio_values",
        "audio_features",
        "feature_attention_mask",
    }
)
_ANCHOR = [{"role": "user", "content": "x"}]


def feature_has_media(feature: Mapping[str, Any]) -> bool:
    return bool(MediaBundle.from_sources(feature)) or any(feature.get(key) is not None for key in _MEDIA_PAYLOAD_KEYS)


def preserve_trajectory(source: Trajectory, encoded: Mapping[str, Any]) -> Trajectory:
    """Keep raw conversation columns separate from the model's existing numerical fields."""
    from twinkle.data_format import InputFeature, copy_trajectory

    result = {**source, **encoded}
    previous = set(source.get('_trajectory_columns', ()))
    model_keys = set(InputFeature.__annotations__)
    result['_trajectory_columns'] = sorted((previous | source.keys()) - encoded.keys() - model_keys)
    for key in ('messages', 'tools', 'user_data', 'images', 'videos', 'audios'):
        if key in source:
            result[key] = source[key]
    return copy_trajectory(result)


def check_generation_prompt(sample, add_generation_prompt):
    messages = sample.get('messages')
    if not messages:
        return
    last = messages[-1]
    open_turn = last.get('role') not in {'assistant', 'tool_call'} or (
        last.get('role') == 'assistant' and last.get('content') is None and not last.get('tool_calls'))
    if add_generation_prompt and not open_turn:
        raise ValueError(
            'add_generation_prompt=True but the conversation already ends with an assistant turn; '
            'drop the trailing response or encode it as a complete conversation.')
    if not add_generation_prompt and open_turn:
        raise ValueError(
            f"add_generation_prompt=False but the conversation ends with a {last.get('role')!r} turn, "
            'which the encoder would open a generation prompt after.')


def model_payload_keys(feature):
    from twinkle.data_format import InputFeature, Trajectory
    metadata = set(Trajectory.__annotations__) - set(InputFeature.__annotations__)
    text = {'input_ids', 'labels', 'loss_scale', 'completion_mask', 'attention_mask', 'attention_mask_2d',
            'position_ids', 'text_position_ids', 'token_type_ids', 'mm_token_type_ids', 'routed_experts',
            'replay_mask', 'length', 'lengths', 'channel', 'margin', 'label', 'objects', 'chat_template_kwargs',
            'template_inputs', 'packed_seq_params', 'cu_seq_lens_q', 'cu_seq_lens_k', 'cu_seqlens_q',
            'cu_seqlens_kv', 'max_length_q', 'max_length_k', 'system', 'extra_kwargs', 'inputs_embeds',
            'logprobs', 'stuck_stop', 'tool_stop', 'followups', 'error', 'rollout_infos', 'decoded'}
    return frozenset(key for key, value in feature.items() if value is not None and key not in metadata
                     and key not in text and key not in ('images', 'videos', 'audios', 'tools', 'mm_processor_kwargs')
                     and not key.startswith('_') and key not in feature.get('_trajectory_columns', ()))


def normalize_token_fields(row):
    for key in ('input_ids', 'engine_input_ids', 'labels', 'loss_scale', 'completion_mask',
                'attention_mask', 'token_type_ids', 'mm_token_type_ids', 'position_ids'):
        value = row.get(key)
        if value is not None and hasattr(value, 'ndim') and value.ndim == 1:
            row[key] = value.tolist()
        elif key in SHIFTED_FIELDS and value is None:
            row.pop(key, None)
    return row


def slice_row(row, indices, *, supervised=True):
    result = dict(row)
    length = len(row['input_ids'])
    result['_original_input_ids'] = list(row['input_ids'])
    result['_kept_token_indices'] = list(indices)
    for key in ('input_ids', 'labels', 'loss_scale', 'completion_mask', 'token_type_ids', 'mm_token_type_ids'):
        value = row.get(key)
        if value is None:
            continue
        if _is_torch(value):
            import torch
            selected = value.index_select(-1, torch.tensor(indices, dtype=torch.long, device=value.device))
        elif _is_numpy(value):
            selected = value[..., indices].copy()
        else:
            selected = [value[index] for index in indices]
            if isinstance(value, tuple):
                selected = tuple(selected)
        if supervised and key in SHIFTED_FIELDS and indices:
            fill = SHIFTED_FIELDS[key]
            if _is_torch(selected) or _is_numpy(selected):
                selected[..., 0] = fill
            else:
                selected = [fill, *selected[1:]]
                if isinstance(value, tuple):
                    selected = tuple(selected)
        result[key] = selected
    if len(indices) != length:
        for key in ('position_ids', 'text_position_ids', 'attention_mask', 'attention_mask_2d'):
            result.pop(key, None)
    result['length'] = len(indices)
    return result


def truncate_rows(row, max_length, strategy, *, placeholder_tokens=(), multimodal=False, supervised=True):
    if strategy not in ('raise', 'left', 'right', 'split', 'delete'):
        raise ValueError(f'Invalid truncation_strategy: {strategy}')
    ids = row.get('input_ids')
    if ids is None or max_length is None or len(ids) <= max_length:
        return [row]
    length = len(ids)
    if strategy in ('raise', 'delete'):
        raise MaxLengthError(f'Current length of row({length}) is larger than the max_length({max_length}).')
    if strategy == 'split':
        if multimodal:
            raise ValueError('multimodal token splitting cannot partition its media payload safely')
        return [slice_row(row, list(range(start, min(start + max_length, length))), supervised=supervised)
                for start in range(0, length, max_length)]
    protected_tokens = set(placeholder_tokens)
    protected = [index for index, token in enumerate(ids) if int(token) in protected_tokens]
    protected_set = set(protected)
    remaining = [index for index in range(length) if index not in protected_set]
    budget = max(0, max_length - len(protected))
    selected = remaining[-budget:] if strategy == 'left' and budget else remaining[:budget]
    return [slice_row(row, sorted(protected + selected), supervised=supervised)]


def _is_numpy(value):
    return type(value).__module__.startswith("numpy")


def _is_torch(value):
    return type(value).__module__.startswith("torch")


def _boundary_is(value, index, fill):
    if _is_numpy(value) or _is_torch(value):
        result = value[..., index] == fill
        return bool(result.all().item() if hasattr(result.all(), "item") else result.all())
    return value[index] == fill


def shift_values(values, fill):
    if len(values) == 0:
        return values.copy() if hasattr(values, "copy") else type(values)(values)
    if not _boundary_is(values, 0, fill):
        raise ValueError(f"shift expects the first position to be {fill!r}.")
    if _is_numpy(values):
        import numpy as np

        result = np.roll(values, -1, axis=-1)
        result[..., -1] = fill
        return result
    if _is_torch(values):
        result = values.roll(-1, dims=-1)
        result[..., -1] = fill
        return result
    shifted = list(values[1:]) + [fill]
    return tuple(shifted) if isinstance(values, tuple) else shifted


def unshift_values(values, fill):
    if len(values) == 0:
        return values.copy() if hasattr(values, "copy") else type(values)(values)
    if not _boundary_is(values, -1, fill):
        raise ValueError(f"unshift expects the last position to be {fill!r}.")
    if _is_numpy(values):
        import numpy as np

        result = np.roll(values, 1, axis=-1)
        result[..., 0] = fill
        return result
    if _is_torch(values):
        result = values.roll(1, dims=-1)
        result[..., 0] = fill
        return result
    unshifted = [fill] + list(values[:-1])
    return tuple(unshifted) if isinstance(values, tuple) else unshifted


def shift(row: InputOrderRow) -> EncodedRow:
    result: Dict[str, Any] = dict(row)
    for key, fill in SHIFTED_FIELDS.items():
        if result.get(key) is not None:
            result[key] = shift_values(result[key], fill)
    return cast(EncodedRow, result)


def unshift(row: EncodedRow) -> InputOrderRow:
    result: Dict[str, Any] = dict(row)
    for key, fill in SHIFTED_FIELDS.items():
        if result.get(key) is not None:
            result[key] = unshift_values(result[key], fill)
    return cast(InputOrderRow, result)


def _append_fill(value, count, fill):
    if _is_numpy(value):
        import numpy as np

        padding = np.full((*value.shape[:-1], count), fill, dtype=value.dtype)
        return np.concatenate((value, padding), axis=-1)
    if _is_torch(value):
        import torch

        padding = torch.full((*value.shape[:-1], count), fill, dtype=value.dtype, device=value.device)
        return torch.cat((value, padding), dim=-1)
    values = list(value) + [fill] * count
    return tuple(values) if isinstance(value, tuple) else values


def append_input_order(row: EncodedRow, tokens: Sequence[int], *, scored: bool, sampled: bool) -> InputOrderRow:
    if row.get("loss_scale") is not None:
        raise ValueError("a rollout prefix must not carry loss_scale.")
    result = cast(Dict[str, Any], unshift(row))
    input_ids = list(result["input_ids"])
    new_tokens = list(tokens)
    labels = result.get("labels")
    labels = [-100] * len(input_ids) if labels is None else list(labels)
    completion_mask = result.get("completion_mask")
    if completion_mask is None:
        completion_mask = [0 if label == -100 else 1 for label in labels]
    else:
        completion_mask = list(completion_mask)
    if len(labels) != len(input_ids) or len(completion_mask) != len(input_ids):
        raise ValueError("input_ids, labels and completion_mask must be aligned before splice.")
    result["input_ids"] = input_ids + new_tokens
    if new_tokens:
        for key in ('position_ids', 'text_position_ids', 'attention_mask_2d'):
            result.pop(key, None)
    result["labels"] = labels + (new_tokens if scored else [-100] * len(new_tokens))
    result["completion_mask"] = completion_mask + [int(sampled)] * len(new_tokens)
    for key, fill in TOKEN_FILL.items():
        if result.get(key) is not None:
            result[key] = _append_fill(result[key], len(new_tokens), fill)
    return cast(InputOrderRow, result)


def append_and_finalize(
    row: EncodedRow,
    tokens: Sequence[int],
    *,
    scored: bool,
    sampled: bool,
    finalize: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> Optional[EncodedRow]:
    input_order = append_input_order(_batched_token_fields_as_arrays(row), tokens, scored=scored, sampled=sampled)
    finalized = finalize([input_order])
    if not finalized:
        return None
    if len(finalized) != 1:
        raise ValueError("appending one turn must produce exactly one feature.")
    result = cast(EncodedRow, finalized[0])
    if list(result['input_ids']) != list(row['input_ids']) + list(tokens):
        raise ValueError('appending sampled tokens must not truncate or modify the recorded sequence')
    return result


def splice(row: EncodedRow, tokens: Sequence[int], *, scored: bool, sampled: bool) -> EncodedRow:
    return refresh(append_input_order(row, tokens, scored=scored, sampled=sampled))


def refresh(row: InputOrderRow) -> EncodedRow:
    result: Dict[str, Any] = dict(row)
    input_ids = result.get("input_ids")
    if input_ids is None:
        return shift(cast(InputOrderRow, result))
    result["attention_mask"] = [1] * len(input_ids)
    result["length"] = len(input_ids)
    return shift(cast(InputOrderRow, result))


def _has_media(message: Dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(_MEDIA_TAG.search(content))
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type", "text") != "text" for item in content
    )


def _without_media(message: Dict[str, Any]) -> Dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        return {**message, "content": _MEDIA_TAG.sub("", content)}
    if isinstance(content, list) and content and isinstance(content[0], dict) and "type" in content[0]:
        return {**message, "content": "".join(item.get("text") or "" for item in content if item.get("type") == "text")}
    return message


def render_delta(
    messages_before: List[Dict[str, Any]],
    appended: List[Dict[str, Any]],
    *,
    gen_prompt_before: bool,
    gen_prompt_after: bool,
    render: Callable[[List[Dict[str, Any]], bool], Sequence[int]],
) -> List[int]:
    if any(_has_media(message) for message in appended):
        raise NotImplementedError("render_delta does not render media in appended turns.")
    history = [_without_media(message) for message in messages_before]
    before = list(render(history, gen_prompt_before))
    after = list(render(history + list(appended), gen_prompt_after))
    if after[: len(before)] == before:
        return after[len(before) :]
    anchors = [_ANCHOR if gen_prompt_before else _ANCHOR + [{"role": "assistant", "content": "x"}]]
    if not gen_prompt_before:
        anchors.append(_ANCHOR)
    for anchor in anchors:
        anchor_before = list(render(anchor, gen_prompt_before))
        anchor_after = list(render(anchor + list(appended), gen_prompt_after))
        if anchor_after[: len(anchor_before)] != anchor_before:
            continue
        bridge = anchor_after[len(anchor_before) :]
        # The anchor stands in for a history whose own render is not monotonic, so the bridge it
        # yields is trustworthy only if the appended turns render the same way after the real
        # history. A tool turn does not: its block depends on the assistant tool_calls turn it
        # answers, which the anchor lacks. Reject and fail loudly rather than return a bridge that
        # would silently splice the wrong tokens into a training sequence.
        if bridge and after[-len(bridge) :] == bridge:
            return bridge
    raise RuntimeError("non-monotonic render: appended turns are not a prefix-extension of the history or anchor.")


def _batched_token_fields_as_arrays(row):
    result = row
    for key in TOKEN_FILL:
        value = row.get(key)
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            import numpy as np
            if result is row:
                result = dict(row)
            result[key] = np.asarray(value)
    return result


def engine_input_ids(trajectory: Trajectory) -> list[int]:
    """Read engine IDs without conflating collapsed media placeholders with training IDs."""
    ids = trajectory.get('engine_input_ids')
    if feature_has_media(trajectory):
        if ids is None or not MediaBundle.from_sources(trajectory):
            raise ValueError('encoded multimodal trajectory requires engine_input_ids and original media')
    if ids is None:
        ids = trajectory['input_ids']
    return [int(value) for value in (ids.tolist() if hasattr(ids, 'tolist') else ids)]


def validate_sampling_prefix(trajectory: Trajectory, actual_ids=None) -> None:
    """Validate alignment without replacing model IDs with the engine's media projection."""
    ids = trajectory['input_ids']
    for key in ('labels', 'completion_mask'):
        value = trajectory.get(key)
        if value is not None and len(value) != len(ids):
            raise ValueError(f'trajectory {key} must align with input_ids')
    if actual_ids is not None and not feature_has_media(trajectory) and list(actual_ids) != list(ids):
        raise RuntimeError('engine text prompt differs from the recorded trajectory')


def project_engine_ids(original_ids, engine_ids, kept_indices):
    """Apply a token slice to the matching engine sequence without retokenizing retained text."""
    from difflib import SequenceMatcher

    kept = set(kept_indices)
    result = []
    matcher = SequenceMatcher(a=list(original_ids), b=list(engine_ids), autojunk=False)
    for tag, start, stop, engine_start, engine_stop in matcher.get_opcodes():
        selected = [index for index in range(start, stop) if index in kept]
        if tag == 'equal':
            result.extend(engine_ids[engine_start + index - start] for index in selected)
        elif tag == 'insert':
            if start in kept or (start == len(original_ids) and start - 1 in kept):
                result.extend(engine_ids[engine_start:engine_stop])
        elif selected:
            if len(selected) != stop - start:
                raise ValueError('truncation cuts an engine-specific media span; its token projection is ambiguous')
            result.extend(engine_ids[engine_start:engine_stop])
    return result


def project_sliced_messages(messages, original_ids, kept_indices, *, render_prefix, decode, span_cache=None):
    """Project tokens onto structured messages; cache only span offsets across slices of one encoding."""
    if not messages:
        return []
    boundaries = [0]
    for stop in range(1, len(messages)):
        width = span_cache.get(stop) if span_cache is not None else None
        if width is None:
            prefix = render_prefix(messages[:stop])
            width = 0
            for left, right in zip(original_ids, prefix):
                if int(left) != int(right):
                    break
                width += 1
            if span_cache is not None:
                span_cache[stop] = width
        if width < boundaries[-1]:
            raise ValueError('cannot recover message spans from a non-monotonic truncated render')
        boundaries.append(width)
    boundaries.append(len(original_ids))
    kept = set(kept_indices)
    result = []
    for message, start, stop in zip(messages, boundaries, boundaries[1:]):
        selected = [index for index in range(start, stop) if index in kept]
        if not selected:
            continue
        if len(selected) == stop - start:
            from twinkle.data_format import copy_trajectory
            result.append(copy_trajectory(message))
        else:
            partial = {'role': message.get('role', 'user'),
                       'content': decode([original_ids[index] for index in selected])}
            if message.get('tool_call_id') is not None:
                partial['tool_call_id'] = message['tool_call_id']
            result.append(partial)
    return result
