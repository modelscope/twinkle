# Copyright (c) ModelScope Contributors. All rights reserved.
"""The template-facing half of the transformers generation pipeline, shared by both entry points.

Two callers generate through a :class:`~twinkle.sampler.transformers_sampler.TransformersEngine`:

- :class:`~twinkle.sampler.TransformersSampler`, which loads its own weights from ``model_id``;
- :meth:`TransformersModel.generate`, a facade that generates on the weights a *training* model already
  holds, inside that model's own data-parallel workers.

Around the engine's ``batch_sample`` both run the same steps that talk to a
:class:`~twinkle.template.Template`: encode each ``Trajectory`` into a prompt ``InputFeature``, collate
the multimodal tensors a batch needs, and stitch the generated tokens back onto the prompt feature.
Those steps live here so the two callers cannot drift -- which matters because the facade forwards its
inputs whole and relies on this pipeline matching the sampler's exactly.

What is deliberately *not* here: left-padding, logprobs, ``num_return_sequences`` and decoding. Those
are the engine's business (see ``transformers_engine``); this module is engine-agnostic and owns only
the encoding contract with ``Template``.
"""

from __future__ import annotations
from typing import Any, Optional

import torch

from twinkle import get_logger
from twinkle.data_format import InputFeature, SampledSequence, SampleResponse
from twinkle.template.contract import DatasetTemplate, PromptEncoder, RolloutState, feature_has_media

logger = get_logger()


def not_encoded(inputs: Any) -> bool:
    """Whether ``inputs`` is still a raw ``Trajectory`` rather than an encoded ``InputFeature``.

    A trajectory carries ``messages`` and no token ids yet; an encoded feature carries ``input_ids``
    (or a precomputed ``input_embedding``). Mirrors ``Sampler._not_encoded`` /
    ``TransformersModel._not_encoded`` so all three agree on the boundary.
    """
    assert isinstance(inputs, dict), f"Expected dict, got {type(inputs)}"
    return "input_ids" not in inputs and "input_embedding" not in inputs


def encode_trajectory(template: DatasetTemplate, trajectory: Any, add_generation_prompt: bool = True) -> InputFeature:
    """Encode one ``Trajectory`` into an ``InputFeature`` prompt, keeping every template key but labels.

    ``labels`` is dropped because a prompt has no targets yet; they are rebuilt by
    ``template.append_completion`` once the completion tokens exist. Every other key the template
    produced (``pixel_values`` / ``image_grid_thw`` for a vision-language row, ``attention_mask``, ...)
    is carried through so generation and the later stitch see the full feature.
    """
    if template is None:
        raise ValueError("Template not set; call set_template() before encoding a Trajectory.")
    encoded = template.encode(trajectory, add_generation_prompt=add_generation_prompt)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("Template.encode() must return 'input_ids'")
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    result = InputFeature(input_ids=input_ids)
    for key, value in encoded.items():
        if key not in ("input_ids", "labels"):
            result[key] = value
    return result


def error_response(feat: dict[str, Any]) -> SampleResponse:
    """The placeholder an input dropped under ``strict=False`` gets.

    Empty tokens and ``stop_reason='error'`` rather than an exception, so the caller's positional zip
    against its inputs stays aligned and one bad row cannot end a long offline run.
    """
    return SampleResponse(
        sequences=[SampledSequence(stop_reason="error", tokens=[], decoded="")],
        prompt_token_ids=list(feat.get("input_ids") or []),
    )


def encode_all(
    template: PromptEncoder | None,
    inputs_list: list[dict[str, Any]],
    *,
    logprobs_only: bool,
    strict: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, RolloutState | None], dict[int, SampleResponse]]:
    encoded: dict[int, dict[str, Any]] = {}
    states: dict[int, RolloutState | None] = {}
    failures: dict[int, SampleResponse] = {}
    for index, item in enumerate(inputs_list):
        try:
            if isinstance(item, RolloutState):
                encoded[index] = item.feature
                states[index] = item
            elif not not_encoded(item):
                if feature_has_media(item):
                    raise NotImplementedError("multimodal sampling requires a Trajectory or RolloutState.")
                encoded[index] = item
                states[index] = None
            else:
                if template is None:
                    raise ValueError("Template not set; call set_template() before encoding a Trajectory.")
                prompt = template.encode_prompt(item, backend="transformers", add_generation_prompt=not logprobs_only)
                encoded[index] = prompt.state.feature
                states[index] = prompt.state
        except Exception as exc:
            if strict:
                raise
            logger.warning(f"Dropping input {index}, encode failed: {exc}")
            encoded[index] = {}
            states[index] = None
            failures[index] = error_response({})
    return encoded, states, failures


def prompt_ids(feat: dict[str, Any], state: RolloutState | None) -> list[int]:
    if state is None and feature_has_media(feat):
        raise NotImplementedError("multimodal sampling requires a Trajectory or RolloutState.")
    input_ids = state.engine_input_ids if state is not None else feat["input_ids"]
    return input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)


def collate_extra_model_inputs(chunk: list[int], encoded: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Collate the multimodal tensors the template encoded for ``chunk`` into one batched dict.

    A vision-language row carries ``pixel_values`` / ``image_grid_thw``; HF's processor lays those out
    flat over a row's images, so batching concatenates along dim 0 -- exactly how one multi-image prompt
    is shaped for a single ``generate``. ``input_ids`` / ``attention_mask`` / ``labels`` are excluded:
    the engine rebuilds the first two from the left-padded prompt ids (and a passed ``attention_mask``
    would clobber the padding mask), and labels are not a model input. Text-only rows contribute no
    tensor keys, so a plain batch returns ``{}``.
    """
    collected: dict[str, list[torch.Tensor]] = {}
    per_token: dict[str, list[torch.Tensor]] = {}
    for index in chunk:
        for key, value in encoded[index].items():
            if key in ("input_ids", "attention_mask", "labels") or not isinstance(value, torch.Tensor):
                continue
            if key.endswith("_token_type_ids"):
                # A per-token companion (mm_token_type_ids and friends): the model indexes it as
                # [batch, position] against the LEFT-PADDED input_ids the engine builds, so it is
                # stacked to a [B, width] batch here, never concatenated flat like pixel_values.
                per_token.setdefault(key, []).append(value)
            else:
                collected.setdefault(key, []).append(value)
    result = {key: torch.cat(values, dim=0) for key, values in collected.items()}
    for key, values in per_token.items():
        width = max(value.shape[-1] for value in values)
        rows = [torch.nn.functional.pad(value, (width - value.shape[-1], 0)) for value in values]
        result[key] = torch.stack(rows, dim=0)
    return result


def attach_features(
    template: Any,
    results: list[SampleResponse | None],
    encoded: dict[int, dict[str, Any]],
    states: dict[int, RolloutState | None],
) -> None:
    """Fill in ``new_input_feature`` so downstream training code can consume the samples directly.

    The prompt feature plus the generated tokens, run through the template so labels, ``completion_mask``
    and any post-pipeline stay consistent -- matching what the training path expects from a rollout.
    """
    if template is None:
        return
    for index, response in enumerate(results):
        feat = encoded.get(index)
        if response is None or not feat:
            continue
        for seq in response.sequences:
            if seq.tokens:
                completion = template.append_completion(states.get(index) or feat, seq.tokens)
                seq.new_input_feature = completion.state.feature
                seq.new_state = completion.state


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
    states: dict[int, RolloutState | None],
    call_batch,
    *,
    strict: bool = True,
) -> list[SampleResponse]:
    """Generate one padded batch, falling back to one-at-a-time to isolate a bad input when not strict.

    ``call_batch(prompts, extra_model_inputs)`` runs the engine and returns one response per prompt. The
    caller supplies it because how the engine is driven differs: a sampler awaits an async engine on a
    background loop, while a training model calls the engine's synchronous core inline. Everything around
    that call -- prompt fixup, multimodal collation, the isolating retry and the error placeholder -- is
    shared so the two entry points cannot drift.
    """
    prompts = [prompt_ids(encoded[i], states.get(i)) for i in chunk]
    # A vision-language model needs the image tensors the template encoded (pixel_values and friends)
    # alongside the prompt ids; a text-only batch collates to nothing and passes an empty dict.
    extra = collate_extra_model_inputs(chunk, encoded)
    try:
        return call_batch(prompts, extra)
    except Exception as exc:
        if strict or len(chunk) == 1:
            raise
        logger.warning(f"Batch of {len(chunk)} failed ({exc}); retrying individually to isolate the input")
        out: list[SampleResponse] = []
        for position, index in enumerate(chunk):
            try:
                out.append(call_batch([prompts[position]], collate_extra_model_inputs([index], encoded))[0])
            except Exception as inner:
                logger.warning(f"Input dropped, generate failed: {inner}")
                out.append(error_response({}))
        return out
