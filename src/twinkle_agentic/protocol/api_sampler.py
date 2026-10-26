# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler-shaped adapter for external generation APIs."""

from dataclasses import replace
from typing import Any, Literal, Optional, Union

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampledSequence, SamplingParams, StopReason
from twinkle.template import Template
from twinkle.template.contract import RolloutState
from twinkle_agentic.utils.token_utils import _to_plain, encode_appended_turn
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
    message: dict[str, Any] = {
        "role": "assistant",
        "content": reply.get("content") or "",
    }
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
    endpoint should see (a rollout's ``pif`` no longer carries it after encode).
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
        self, prefix: dict[str, Any] | RolloutState, sampling_params: SamplingParams | None = None, **adapter_kwargs
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
        state = prefix if isinstance(prefix, RolloutState) else None
        feature = state.feature if state is not None else prefix
        messages = list(state.messages) if state is not None else list(feature.get("messages") or [])
        if not messages:
            raise ValueError("APISampler needs an encoded prefix carrying its messages.")
        tools = list(state.tools) if state is not None else feature.get("tools", self.tools)
        request: Trajectory = {"messages": messages}
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

        new_tokens = encode_appended_turn(messages, reply, self.template, tools)
        completion = self.template.append_completion(
            state or feature, new_tokens, appended_as=self.appended_as, tool_calls=reply.get("tool_calls")
        )
        new_input_feature = _to_plain(completion.state.feature)
        # append_completion reconstructs content by decoding ``new_tokens``;
        # those include the template's rendered tool-call block. Keep the API's
        # original content beside its structured calls instead of duplicating it.
        assistant_message = {key: reply[key] for key in ("role", "content", "tool_calls") if key in reply}
        if new_input_feature.get("messages"):
            new_input_feature["messages"][-1] = assistant_message
        new_state = replace(
            completion.state,
            feature=new_input_feature,
            messages=completion.state.messages[:-1] + (assistant_message,),
        )

        return SampledSequence(
            stop_reason=_FINISH_TO_STOP.get(reply.get("finish_reason"), "stop"),
            tokens=new_tokens,
            logprobs=None,
            decoded=reply.get("content") or completion.response.content,
            new_input_feature=new_input_feature,
            new_state=new_state,
        )
