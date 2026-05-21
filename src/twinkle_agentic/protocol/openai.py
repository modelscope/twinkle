from typing import Any, Dict, List, Optional, Union

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message
from twinkle.data_format.sampling import SamplingParams
from .base import API


class OpenAI(API):
    """OpenAI-compatible chat-completions client.

    Works with any endpoint speaking the ``/v1/chat/completions`` protocol
    (OpenAI, Azure OpenAI, vLLM, SGLang, Ollama, ...).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI as _OpenAIClient

        self.model = model
        self._client = _OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            **(client_kwargs or {}),
        )

    def __call__(
        self,
        trajectory: Trajectory,
        sampling_params: SamplingParams,
        **kwargs,
    ) -> Union[Message, List[Message]]:
        request = self._build_request(trajectory, sampling_params, kwargs)
        response = self._client.chat.completions.create(**request)
        messages = [self._choice_to_message(c) for c in response.choices]
        return messages[0] if sampling_params.num_samples == 1 else messages

    def _build_request(
        self,
        trajectory: Trajectory,
        sampling_params: SamplingParams,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Trajectory.messages / .tools are already OpenAI-shaped TypedDicts,
        # so they pass through verbatim — no field renaming needed.
        body: Dict[str, Any] = {
            'model': self.model,
            'messages': list(trajectory.get('messages', [])),
            'n': sampling_params.num_samples,
            'temperature': sampling_params.temperature,
            'top_p': sampling_params.top_p,
        }
        tools = trajectory.get('tools')
        if tools:
            body['tools'] = list(tools)
        if sampling_params.max_tokens is not None:
            body['max_tokens'] = sampling_params.max_tokens
        if sampling_params.seed is not None:
            body['seed'] = sampling_params.seed
        if sampling_params.stop:
            stop = sampling_params.stop
            if isinstance(stop, str):
                body['stop'] = [stop]
            elif stop and not isinstance(stop[0], int):
                # OpenAI spec only accepts string stops; silently drop
                # stop_token_ids (vLLM-only concept).
                body['stop'] = list(stop)
        if sampling_params.logprobs is not None:
            body['logprobs'] = True
            body['top_logprobs'] = sampling_params.logprobs
        if sampling_params.repetition_penalty != 1.0:
            # OpenAI has no repetition_penalty; frequency_penalty is the
            # closest knob (range -2..2, where 0 == no penalty).
            body['frequency_penalty'] = sampling_params.repetition_penalty - 1.0
        body.update(overrides)
        return body

    @staticmethod
    def _choice_to_message(choice) -> Message:
        m = choice.message
        msg: Message = {'role': 'assistant'}
        if m.content is not None:
            msg['content'] = m.content
        reasoning = getattr(m, 'reasoning_content', None)
        if reasoning:
            msg['reasoning_content'] = reasoning
        tool_calls = getattr(m, 'tool_calls', None)
        if tool_calls:
            msg['tool_calls'] = [{
                'id': tc.id,
                'type': 'function',
                'function': {
                    'name': tc.function.name,
                    'arguments': tc.function.arguments,
                },
            } for tc in tool_calls]
        # Surface finish_reason so multi-turn drivers can detect length-cap truncation.
        finish = getattr(choice, 'finish_reason', None)
        if finish is not None:
            msg['finish_reason'] = finish
        return msg
