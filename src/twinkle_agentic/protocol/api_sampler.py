# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler-shaped adapter for external generation APIs."""

from typing import Any, Literal, Optional, Union

from twinkle.data_format import Trajectory, copy_trajectory, last_assistant_text
from twinkle.template.contract import append_and_finalize, engine_input_ids
from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams, StopReason
from twinkle.template import Template
from twinkle_agentic.utils.token_utils import encode_appended_turn
from .base import API

_FINISH_TO_STOP: dict[str | None, StopReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "stop",
    "function_call": "stop",
    "content_filter": "abort",
}


class APIGenerationError(RuntimeError):
    """The endpoint failed before returning a response to validate."""


def _normalise_assistant(reply: Any, turn: int) -> dict[str, Any]:
    """Make an API reply safe to render and feed into the next turn."""
    if not isinstance(reply, dict):
        raise TypeError(f"API must return an assistant message dict, got {type(reply).__name__}")
    message = copy_trajectory(reply)
    message["role"] = "assistant"
    message["content"] = reply.get("content") or ""
    tool_calls = reply.get("tool_calls") or []
    if tool_calls:
        normalised = []
        for i, tool_call in enumerate(tool_calls):
            tool_call = dict(tool_call)
            tool_call.setdefault("id", f"call_{turn}_{i}")
            tool_call.setdefault("type", "function")
            normalised.append(tool_call)
        message["tool_calls"] = normalised
    finish_reason = reply.get("finish_reason")
    if finish_reason is not None:
        message["finish_reason"] = finish_reason
    return message


class APISampler:
    """Normalize one :class:`API` turn into a :class:`SampledSequence`.

    Holds the local ``template`` (an API turn's text must be tokenised the way
    the trainer reads it back, not by the endpoint) and the tool schema the
    endpoint should see; both live on the complete trajectory.

    Also sampler-shaped (:meth:`sample`), so a rollout takes it in place of a sampler.
    """

    def __init__(
        self,
        api: API,
        template: Template,
        *,
        tools: list[dict[str, Any]] | None = None,
        appended_as: Literal["demonstration", "context"] = "demonstration",
        api_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            appended_as: how the turn enters training -- ``'demonstration'``
                (scored by SFT, skipped by RL) or ``'context'`` (no loss).
                ``'completion'`` is refused: it would claim a per-token log-prob
                the API never returns.
            api_kwargs: request fields forwarded to every API call.
        """
        if appended_as not in ("demonstration", "context"):
            raise ValueError(
                "APISampler appended_as must be 'demonstration' or 'context', "
                f"got {appended_as!r}; an API turn has no log-prob to be a completion."
            )
        self.api = api
        self.template = template
        self.tools = list(tools) if tools else None
        self.appended_as = appended_as
        self.api_kwargs = dict(api_kwargs or {})

    def __call__(
        self, prefix: Trajectory, sampling_params: SamplingParams | None = None, **adapter_kwargs
    ) -> SampledSequence:
        """Generate one external turn in the callback's normalized shape.

        ``adapter_kwargs`` (``adapter_path`` / ``use_base_model``) name a weight
        set the API does not have; they are accepted and ignored so callback code
        can forward the same values to either backend.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        if sampling_params.num_samples != 1:
            raise ValueError(f"APISampler draws one turn per input; got num_samples={sampling_params.num_samples}.")
        feature = copy_trajectory(prefix)
        messages = feature.get("messages") or []
        if not messages:
            raise ValueError("APISampler needs an encoded prefix carrying its messages.")
        tools = feature.get("tools", self.tools)
        if tools is not None:
            feature['tools'] = copy_trajectory({'tools': tools})['tools']
        request = copy_trajectory(feature)
        if tools:
            request["tools"] = list(tools)

        try:
            reply = self.api(request, sampling_params, **self.api_kwargs)
        except Exception as exc:
            raise APIGenerationError(f"{type(exc).__name__}: {exc}") from exc
        if isinstance(reply, list):
            raise TypeError(
                "APISampler expects one message per turn but the API returned a "
                "list; num_samples > 1 is rejected above, so this is an API bug."
            )
        turn = sum(message.get("role") == "assistant" for message in messages) + 1
        reply = _normalise_assistant(reply, turn)

        if "input_ids" not in feature:
            feature = self.template.encode_prompt(feature, backend="transformers")
        new_tokens = encode_appended_turn(messages, reply, self.template, tools)
        new_input_feature = append_and_finalize(
            feature, new_tokens, scored=self.appended_as == 'demonstration', sampled=False,
            finalize=self.template.finalize_appended_features)
        if new_input_feature is None:
            raise ValueError('appended API turn exceeds max_length under delete truncation strategy')
        new_input_feature['messages'] = copy_trajectory({'messages': messages + [reply]})['messages']
        serving_ids = engine_input_ids(feature)
        if 'engine_input_ids' in feature:
            new_input_feature['engine_input_ids'] = serving_ids + new_tokens
        new_input_feature['reasoning_started'] = self.template._reasoning_started(new_input_feature['input_ids'])

        return SampledSequence(
            stop_reason=_FINISH_TO_STOP.get(reply.get("finish_reason"), "stop"),
            tokens=new_tokens,
            logprobs=None,
            decoded=last_assistant_text(new_input_feature),
            new_input_feature=new_input_feature,
        )

    def sample(self, inputs: list[Any], sampling_params: SamplingParams | None = None, **kwargs) -> list[SampleResponse]:
        """One API turn per input, in the sampler's return shape.

        ``kwargs`` (``adapter_path`` / ``use_base_model`` / ``session_id``) name things the API
        does not have, and are ignored.
        """
        return [SampleResponse(sequences=[self(item, sampling_params)]) for item in inputs]
