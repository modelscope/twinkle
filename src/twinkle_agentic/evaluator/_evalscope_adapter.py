"""EvalScope ``ModelAPI`` adapters and deliberately narrow converters."""

import json
from copy import deepcopy
from threading import Lock
from time import monotonic
from typing import Any, Mapping, Sequence

from evalscope.api.messages import (ChatMessageAssistant, ContentReasoning, ContentText)
from evalscope.api.model import (ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput)
from evalscope.api.model.model_output import Logprob, Logprobs, ModelUsage, TopLogprob, as_stop_reason
from evalscope.api.tool import ToolCall, ToolFunction

from twinkle.data_format import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI

from ._batcher import SamplerBatcher
from ._contracts import BackendContractError, UnsupportedCapabilityError, read_value


_COMMON_FIELDS = {
    'max_tokens', 'seed', 'stop_seqs', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'n', 'logprobs',
    'top_logprobs',
}
_OPENAI_FIELDS = {
    'timeout', 'frequency_penalty', 'presence_penalty', 'logit_bias',
    'parallel_tool_calls', 'reasoning_effort', 'reasoning_summary', 'extra_body', 'extra_query', 'extra_headers',
}


def _content_parts(content: Any) -> tuple[str, str | None]:
    if isinstance(content, str):
        return content, None
    if not isinstance(content, Sequence):
        raise UnsupportedCapabilityError(f'Unsupported message content type {type(content).__name__}')
    texts: list[str] = []
    reasoning: list[str] = []
    for part in content:
        kind = read_value(part, 'type')
        if kind == 'text':
            texts.append(read_value(part, 'text', ''))
        elif kind == 'reasoning':
            reasoning.append(read_value(part, 'reasoning', ''))
        else:
            raise UnsupportedCapabilityError(f'Multimodal content type {kind!r} is unsupported by Twinkle Evaluator')
    return '\n'.join(texts), '\n'.join(reasoning) if reasoning else None


def _tool_call_to_twinkle(tool_call: Any) -> dict[str, Any]:
    function = read_value(tool_call, 'function', {})
    arguments = read_value(function, 'arguments', {})
    return {
        'id': read_value(tool_call, 'id'),
        'type': read_value(tool_call, 'type', 'function') or 'function',
        'function': {'name': read_value(function, 'name'), 'arguments': deepcopy(arguments)},
    }


def to_twinkle_trajectory(input: Sequence[Any], tools: Sequence[Any], *, include_tools: bool = True) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for source in input:
        role = read_value(source, 'role')
        content, reasoning = _content_parts(read_value(source, 'content'))
        if role not in {'system', 'user', 'assistant', 'tool'}:
            raise BackendContractError(f'Unsupported EvalScope message role {role!r}')
        message: dict[str, Any] = {'role': role, 'content': content}
        if reasoning:
            message['reasoning_content'] = reasoning
        if role == 'assistant':
            calls = read_value(source, 'tool_calls')
            if calls:
                message['tool_calls'] = [_tool_call_to_twinkle(call) for call in calls]
        if role == 'tool':
            call_id = read_value(source, 'tool_call_id')
            if call_id:
                message['tool_call_id'] = call_id
        messages.append(message)
    trajectory: dict[str, Any] = {'messages': messages}
    if include_tools and tools:
        trajectory['tools'] = [{
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters.model_dump(exclude_none=True),
            },
        } for tool in tools]
    return trajectory


def _normalize_tool_calls(raw_calls: Any, request_id: int, choice_index: int) -> list[ToolCall]:
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise BackendContractError('tool_calls must be a sequence')
    calls: list[ToolCall] = []
    for call_index, raw in enumerate(raw_calls):
        function = read_value(raw, 'function')
        name = read_value(function, 'name')
        arguments = read_value(function, 'arguments')
        if not isinstance(name, str) or not name:
            raise BackendContractError(f'Tool call {call_index} has no function name')
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise BackendContractError(f'Tool call {call_index} has invalid JSON arguments') from exc
        if not isinstance(arguments, Mapping):
            raise BackendContractError(f'Tool call {call_index} arguments must be an object')
        call_id = read_value(raw, 'id') or f'call_{request_id}_{choice_index}_{call_index}'
        calls.append(ToolCall(id=call_id, function=ToolFunction(name=name, arguments=dict(arguments))))
    return calls


def _assistant_choice(raw: Any, *, model: str, request_id: int, choice_index: int, decoded: str | None = None,
                      stop_reason: str | None = None, logprobs: Logprobs | None = None) -> ChatCompletionChoice:
    content = read_value(raw, 'content', decoded if decoded is not None else '')
    reasoning = read_value(raw, 'reasoning_content')
    calls = read_value(raw, 'tool_calls')
    content_parts: list[Any] = []
    if reasoning:
        content_parts.append(ContentReasoning(reasoning=reasoning))
    if content is not None:
        content_parts.append(ContentText(text=content))
    message_content: str | list[Any] = content_parts if reasoning else (content if content is not None else '')
    tool_calls = _normalize_tool_calls(calls, request_id, choice_index) if calls else None
    return ChatCompletionChoice(
        message=ChatMessageAssistant(content=message_content, tool_calls=tool_calls, model=model),
        stop_reason='tool_calls' if tool_calls else as_stop_reason(stop_reason),
        logprobs=logprobs,
    )


def _sequence_logprobs(sequence: Any, template: Any) -> Logprobs | None:
    values = read_value(sequence, 'logprobs')
    if values is None:
        return None
    tokens = read_value(sequence, 'tokens')
    if not isinstance(tokens, Sequence) or len(values) != len(tokens) or template is None:
        raise UnsupportedCapabilityError('Exact token logprobs require token ids, aligned logprobs, and a template decoder')
    result: list[Logprob] = []
    for token_id, position in zip(tokens, values):
        if not position:
            raise BackendContractError('Sampler returned an empty logprob position')
        token = template.decode([token_id])
        top = [TopLogprob(token=template.decode([candidate]), logprob=score) for candidate, score in position]
        own = next((score for candidate, score in position if candidate == token_id), position[0][1])
        result.append(Logprob(token=token, logprob=own, top_logprobs=top))
    return Logprobs(content=result)


class _BaseModelAPI(ModelAPI):
    def __init__(self, model_id: str, explicit_generation_keys: set[str]) -> None:
        super().__init__(model_name=model_id)
        self._explicit_generation_keys = set(explicit_generation_keys)
        self._request_id = 0
        self._request_lock = Lock()

    def _next_request_id(self) -> int:
        with self._request_lock:
            value = self._request_id
            self._request_id += 1
            return value

    def _unsupported(self, supported: set[str]) -> list[str]:
        ignored = {'batch_size'}
        unsupported = self._explicit_generation_keys - supported - ignored
        if 'stream' in unsupported:
            unsupported.remove('stream')
        return sorted(unsupported)

    def _sampling_params(self, config: GenerateConfig, *, is_openai: bool = False) -> SamplingParams:
        repetition = config.repetition_penalty if config.repetition_penalty is not None else 1.0
        if is_openai and 'repetition_penalty' in self._explicit_generation_keys:
            extra = config.extra_body or {}
            if extra.get('repetition_penalty') != config.repetition_penalty:
                raise UnsupportedCapabilityError(
                    'protocol.OpenAI cannot map repetition_penalty exactly; pass the same value in extra_body or omit it')
            repetition = 1.0
        return SamplingParams(
            max_tokens=config.max_tokens,
            seed=config.seed,
            stop=config.stop_seqs,
            temperature=config.temperature if config.temperature is not None else 1.0,
            top_k=config.top_k if config.top_k is not None else -1,
            top_p=config.top_p if config.top_p is not None else 1.0,
            repetition_penalty=repetition,
            logprobs=config.top_logprobs if config.logprobs else None,
            num_samples=config.n if config.n is not None else 1,
        )

    def _validate_common(self, config: GenerateConfig, supported: set[str]) -> None:
        if 'stream' in self._explicit_generation_keys and config.stream:
            raise UnsupportedCapabilityError('Streaming is unsupported by Twinkle Evaluator')
        unsupported = self._unsupported(supported)
        if unsupported:
            raise UnsupportedCapabilityError(
                f'{type(self).__name__} for {self.model_name} cannot represent explicit generation fields: '
                f"{', '.join(unsupported)}")
        if 'top_logprobs' in self._explicit_generation_keys and not config.logprobs:
            raise UnsupportedCapabilityError('top_logprobs requires logprobs=True')


class ProtocolModelAPI(_BaseModelAPI):
    def __init__(self, api: Any, model_id: str, explicit_generation_keys: set[str]) -> None:
        super().__init__(model_id, explicit_generation_keys)
        self.api = api

    def validate_generation_config(self, config: GenerateConfig) -> None:
        supported = _COMMON_FIELDS | (_OPENAI_FIELDS if isinstance(self.api, OpenAI) else set())
        self._validate_common(config, supported)
        self._sampling_params(config, is_openai=isinstance(self.api, OpenAI))

    def _api_overrides(self, config: GenerateConfig, tool_choice: Any) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if isinstance(self.api, OpenAI):
            for name in _OPENAI_FIELDS:
                if name in self._explicit_generation_keys:
                    overrides[name] = getattr(config, name)
            if tool_choice == 'any':
                overrides['tool_choice'] = 'required'
            elif tool_choice == 'none':
                overrides['tool_choice'] = 'none'
            elif tool_choice != 'auto':
                overrides['tool_choice'] = {'type': 'function', 'function': {'name': tool_choice.name}}
        elif tool_choice not in ('auto', None):
            overrides['tool_choice'] = tool_choice
        return overrides

    def generate(self, input: list[Any], tools: list[Any], tool_choice: Any, config: GenerateConfig) -> ModelOutput:
        self.validate_generation_config(config)
        if tools and tool_choice != 'none' and config.n not in (None, 1):
            raise UnsupportedCapabilityError('Agent/tool evaluation requires generation_config.n == 1')
        request_id = self._next_request_id()
        trajectory = to_twinkle_trajectory(input, tools, include_tools=tool_choice != 'none')
        started = monotonic()
        response = self.api(
            trajectory,
            self._sampling_params(config, is_openai=isinstance(self.api, OpenAI)),
            **self._api_overrides(config, tool_choice),
        )
        elapsed = monotonic() - started
        choices_raw = response if isinstance(response, list) else [response]
        if not choices_raw:
            raise BackendContractError(f'API {type(self.api).__name__} returned no choices for {self.model_name}')
        if any(not isinstance(item, Mapping) for item in choices_raw):
            raise BackendContractError(f'API {type(self.api).__name__} returned a non-message choice for {self.model_name}')
        choices = [_assistant_choice(
            item, model=self.model_name, request_id=request_id, choice_index=index,
            stop_reason=read_value(item, 'finish_reason'),
        ) for index, item in enumerate(choices_raw)]
        return ModelOutput(model=self.model_name, choices=choices, time=elapsed)


class SamplerModelAPI(_BaseModelAPI):
    def __init__(self, sampler: Any, model_id: str, template: Any, explicit_generation_keys: set[str], *, batch_size: int,
                 batch_wait_ms: float, sampler_kwargs: Mapping[str, Any]) -> None:
        super().__init__(model_id, explicit_generation_keys)
        self.sampler = sampler
        self.template = template
        self.batcher = SamplerBatcher(
            sampler, batch_size=batch_size, batch_wait_ms=batch_wait_ms, sampler_kwargs=sampler_kwargs)

    def validate_generation_config(self, config: GenerateConfig) -> None:
        self._validate_common(config, _COMMON_FIELDS)
        self._sampling_params(config)

    def generate(self, input: list[Any], tools: list[Any], tool_choice: Any, config: GenerateConfig) -> ModelOutput:
        self.validate_generation_config(config)
        if tool_choice not in ('auto', 'none', None):
            raise UnsupportedCapabilityError('Sampler backends support only tool_choice="auto" or "none"')
        if tools and tool_choice != 'none' and config.n not in (None, 1):
            raise UnsupportedCapabilityError('Agent/tool evaluation requires generation_config.n == 1')
        request_id = self._next_request_id()
        trajectory = to_twinkle_trajectory(input, tools, include_tools=tool_choice != 'none')
        started = monotonic()
        response = self.batcher.submit(trajectory, self._sampling_params(config))
        elapsed = monotonic() - started
        sequences = read_value(response, 'sequences')
        if not isinstance(sequences, Sequence) or not sequences:
            raise BackendContractError(f'Sampler response for {self.model_name} has no sequences')
        choices: list[ChatCompletionChoice] = []
        for index, sequence in enumerate(sequences):
            stop_reason = read_value(sequence, 'stop_reason')
            if stop_reason in ('abort', 'error'):
                raise BackendContractError(f'Sampler response sequence {index} ended with {stop_reason}')
            decoded = read_value(sequence, 'decoded')
            feature = read_value(sequence, 'new_input_feature')
            messages = read_value(feature, 'messages', []) if feature is not None else []
            structured = messages[-1] if messages and read_value(messages[-1], 'role') == 'assistant' else None
            if decoded is None and structured is None:
                raise BackendContractError(f'Sampler response sequence {index} has neither decoded text nor assistant message')
            raw = dict(structured) if isinstance(structured, Mapping) else {}
            if not raw:
                raw['content'] = decoded
            elif raw.get('content') is None:
                raw['content'] = decoded or ''
            if tools and tool_choice != 'none' and not raw.get('tool_calls'):
                if self.template is None or not hasattr(self.template, 'parse_tool_call'):
                    raise UnsupportedCapabilityError('Sampler tool calls require structured output or template.parse_tool_call()')
                parsed = self.template.parse_tool_call(decoded or '')
                if parsed:
                    raw['tool_calls'] = parsed
            choices.append(_assistant_choice(
                raw,
                model=self.model_name,
                request_id=request_id,
                choice_index=index,
                decoded=decoded,
                stop_reason=stop_reason,
                logprobs=_sequence_logprobs(sequence, self.template)
                if 'top_logprobs' in self._explicit_generation_keys else None,
            ))
        prompt_ids = read_value(response, 'prompt_token_ids')
        output_tokens = sum(len(read_value(sequence, 'tokens', []) or []) for sequence in sequences)
        input_tokens = len(prompt_ids) if prompt_ids is not None else 0
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            usage=ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens),
            time=elapsed,
        )
