# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared template encoding and candidate finalization for samplers and model.generate.

Transformers batching stays here; all backends use the same complete-trajectory output contract.
"""

from __future__ import annotations
from typing import Any, Optional

import torch

from twinkle import get_logger
from twinkle.data_format import InputFeature, SampledSequence, SampleResponse, Trajectory, copy_trajectory, is_encoded
from twinkle.template.contract import engine_input_ids
from twinkle.template.contract.encoding import validate_sampling_prefix
from twinkle.protocol.json_utils import json_safe

logger = get_logger()


def not_encoded(inputs: Any) -> bool:
    """Whether ``inputs`` is still a raw ``Trajectory`` rather than an encoded ``InputFeature``.

    A trajectory carries ``messages`` and no token ids yet; an encoded feature carries ``input_ids``
    (or a precomputed ``input_embedding``). Mirrors ``Sampler._not_encoded`` /
    ``TransformersModel._not_encoded`` so all three agree on the boundary.
    """
    assert isinstance(inputs, dict), f"Expected dict, got {type(inputs)}"
    return not is_encoded(inputs)


def encode_trajectory(template, trajectory, add_generation_prompt: bool = True):
    """Encode a complete prompt while retaining conversation and numerical fields."""
    if template is None:
        raise ValueError('Template not set; call set_template() before encoding a Trajectory')
    feature = template.encode_prompt(trajectory, backend='transformers', add_generation_prompt=add_generation_prompt)
    if not isinstance(feature, dict) or feature.get('input_ids') is None:
        raise ValueError("encode_prompt must return 'input_ids' in a complete trajectory")
    return feature


def error_response(feat: dict[str, Any]) -> SampleResponse:
    """The placeholder an input dropped under ``strict=False`` gets.

    Empty tokens and ``stop_reason='error'`` rather than an exception, so the caller's positional zip
    against its inputs stays aligned and one bad row cannot end a long offline run.
    """
    return SampleResponse(
        sequences=[SampledSequence(stop_reason="error", tokens=[], decoded="", new_input_feature=copy_trajectory(feat))],
        prompt_token_ids=list(feat.get("input_ids", [])),
    )


def encode_all(template, inputs_list, *, logprobs_only: bool, strict: bool):
    encoded, failures = {}, {}
    for index, item in enumerate(inputs_list):
        try:
            feature = copy_trajectory(item)
            if not_encoded(feature):
                feature = encode_trajectory(template, feature, add_generation_prompt=not logprobs_only)
            engine_input_ids(feature)
            encoded[index] = feature
        except Exception as exc:
            if strict:
                raise
            logger.warning(f'Dropping input {index}, encode failed: {exc}')
            encoded[index] = copy_trajectory(item)
            failures[index] = error_response(encoded[index])
    return encoded, failures


def prompt_ids(feat):
    ids = feat['input_ids']
    return [int(value) for value in (ids.tolist() if hasattr(ids, 'tolist') else ids)]


def collate_extra_model_inputs(chunk: list[int], encoded: dict[int, dict[str, Any]], template=None) -> dict[str, Any]:
    """Collate the multimodal tensors the template encoded for ``chunk`` into one batched dict.

    A vision-language row carries ``pixel_values`` / ``image_grid_thw``; HF's processor lays those out
    flat over a row's images, so batching concatenates along dim 0 -- exactly how one multi-image prompt
    is shaped for a single ``generate``. ``input_ids`` / ``attention_mask`` / ``labels`` are excluded:
    the engine rebuilds the first two from the left-padded prompt ids (and a passed ``attention_mask``
    would clobber the padding mask), and labels are not a model input. Text-only rows contribute no
    tensor keys, so a plain batch returns ``{}``.
    """
    engine_fields = {'input_ids', 'attention_mask', 'labels', 'completion_mask', 'loss_scale',
                     'routed_experts', 'replay_mask', 'engine_input_ids'}
    blocked = engine_fields | (Trajectory.__annotations__.keys() - InputFeature.__annotations__.keys())
    result, consumed = {}, set()
    if template is not None and hasattr(template, 'pop_media') and hasattr(template, 'collate_media'):
        media = []
        for index in chunk:
            row = copy_trajectory(encoded[index])
            if '_trajectory_columns' not in row:
                row["_trajectory_columns"] = [
                    key
                    for key, value in row.items()
                    if key not in InputFeature.__annotations__ and isinstance(value, (str, int, float, bool))
                ]
            media.append(template.pop_media(row))
        consumed = {key for row in media for key in row}
        result = {key: value for key, value in template.collate_media(media).items() if key not in engine_fields}
    collected: dict[str, list[torch.Tensor]] = {}
    companions = set()
    for index in chunk:
        for key, value in encoded[index].items():
            if (value is None or key in result or key in consumed or key in blocked
                    or key in encoded[index].get('_trajectory_columns', ())):
                continue
            if key in ('token_type_ids', 'position_ids') or key.endswith('_token_type_ids'):
                companions.add(key)
            elif isinstance(value, torch.Tensor):
                collected.setdefault(key, []).append(value)
            elif type(value).__module__.startswith('numpy'):
                collected.setdefault(key, []).append(torch.as_tensor(value))
    result.update({key: torch.cat(values, dim=0) for key, values in collected.items()})
    width = max(len(encoded[index]['input_ids']) for index in chunk)
    for key in companions:
        prototype = torch.as_tensor(next(encoded[index][key] for index in chunk if encoded[index].get(key) is not None))
        if key == 'position_ids' and prototype.ndim == 3 and prototype.shape[1] == 1:
            prototype = prototype.squeeze(1)
        elif key == 'position_ids' and prototype.ndim == 2 and prototype.shape[0] == 1:
            prototype = prototype.squeeze(0)
        leading = prototype.shape[:-1] if key == 'position_ids' else ()
        rows = []
        for index in chunk:
            length = len(encoded[index]['input_ids'])
            value = encoded[index].get(key)
            if value is None:
                value = (torch.arange(length, device=prototype.device).expand(*leading, length)
                         if key == 'position_ids' else prototype.new_zeros(length))
            else:
                value = torch.as_tensor(value, device=prototype.device)
                if key == 'position_ids' and value.ndim == 3 and value.shape[1] == 1:
                    value = value.squeeze(1)
                elif key == 'position_ids' and value.ndim == 2 and value.shape[0] == 1:
                    value = value.squeeze(0)
                elif key != 'position_ids':
                    value = value.reshape(-1)
            if value.shape != (*leading, length):
                raise ValueError(f'{key} must align with the prompt input_ids')
            rows.append(torch.nn.functional.pad(value, (width - length, 0)))
        result[key] = torch.stack(rows, dim=1 if leading else 0)
    return result


def attach_features(template, results, encoded, *, logprobs_only=False, token_only=False):
    """Finalize all backends' candidates without mixing token-only and complete-trajectory outputs."""
    for index, response in enumerate(results):
        feat = encoded.get(index)
        if response is None or feat is None:
            continue
        if not token_only and feat.get('input_ids') is not None:
            validate_sampling_prefix(feat, response.prompt_token_ids)
        for seq in response.sequences:
            seq.routed_experts = json_safe(seq.routed_experts)
            if logprobs_only:
                seq.tokens = []
                seq.logprobs = None
                seq.decoded = None
                seq.sampling_masks = None
            if token_only:
                seq.decoded = None
                seq.new_input_feature = None
                continue
            if logprobs_only or (seq.stop_reason == 'error' and not seq.tokens):
                seq.new_input_feature = copy_trajectory(feat)
            else:
                if template is None:
                    raise ValueError('normal sampling requires a template; use token_only for engine primitives')
                seq.new_input_feature = template.append_completion(feat, list(seq.tokens or ()))
                if seq.decoded is None:
                    seq.decoded = template.decode(seq.tokens)
            seq.new_input_feature.pop('routed_experts', None)
            if seq.routed_experts is not None:
                seq.new_input_feature['routed_experts'] = seq.routed_experts


def group_by_adapter(
    pending: list[int], adapter_paths: list[str | None] | None, adapter_path: str | None
) -> dict[str | None, list[int]]:
    """Bucket the pending input indices by which adapter they need.

    Insertion order is preserved per bucket, so results still land at their original positions. peft
    activates one adapter at a time, so a batch cannot mix adapters; when every input shares one adapter
    (the common case) this is a single bucket and costs nothing.
    """
    if adapter_paths is None:
        return {adapter_path: pending}
    groups: dict[str | None, list[int]] = {}
    for index in pending:
        groups.setdefault(adapter_paths[index], []).append(index)
    return groups


def chunks(items: list[int], size: int):
    """Yield consecutive sublists of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_chunk(
    chunk: list[int],
    encoded: dict[int, dict[str, Any]],
    call_batch,
    *,
    template=None,
    strict: bool = True,
) -> list[SampleResponse]:
    """Generate one padded batch, falling back to one-at-a-time to isolate a bad input when not strict.

    ``call_batch(prompts, extra_model_inputs)`` runs the engine and returns one response per prompt. The
    caller supplies it because how the engine is driven differs: a sampler awaits an async engine on a
    background loop, while a training model calls the engine's synchronous core inline. Everything around
    that call -- prompt fixup, multimodal collation, the isolating retry and the error placeholder -- is
    shared so the two entry points cannot drift.
    """
    prompts = [prompt_ids(encoded[i]) for i in chunk]
    # A vision-language model needs the image tensors the template encoded (pixel_values and friends)
    # alongside the prompt ids; a text-only batch collates to nothing and passes an empty dict.
    try:
        extra = collate_extra_model_inputs(chunk, encoded, template)
        return call_batch(prompts, extra)
    except Exception as exc:
        if strict:
            raise
        if len(chunk) == 1:
            return [error_response(encoded[chunk[0]])]
        logger.warning(f"Batch of {len(chunk)} failed ({exc}); retrying individually to isolate the input")
        out: list[SampleResponse] = []
        for position, index in enumerate(chunk):
            try:
                out.append(call_batch([prompts[position]], collate_extra_model_inputs([index], encoded, template))[0])
            except Exception as inner:
                logger.warning(f"Input dropped, generate failed: {inner}")
                out.append(error_response(encoded[index]))
        return out
