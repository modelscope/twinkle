import threading
from contextlib import nullcontext
from typing import Any, ContextManager, Dict, List, Optional, Union

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message
from twinkle.data_format.sampling import SamplingParams
from .base import API


class OpenAI(API):
    """OpenAI-compatible chat-completions client.

    Works with any endpoint speaking the ``/v1/chat/completions`` protocol
    (OpenAI, Azure OpenAI, vLLM, SGLang, Ollama, ...).

    Requests in flight are capped here rather than by whatever thread pool calls
    in. A caller's thread count sizes local parallelism and wants to be large; a
    provider's quota belongs to the endpoint and wants to be small. One number
    cannot serve both, and only this object knows which endpoint it is talking to.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        concurrency: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            concurrency: most requests allowed in flight at once, or None for no
                cap. The limit is per instance and shared by every thread holding
                it, so a module-level client caps the whole process.
            timeout: per-request timeout in seconds. Left at the SDK's default
                when None.
            max_retries: how many times the SDK retries a request it deems
                transient -- 429, 5xx, timeouts, dropped connections -- using its
                own exponential backoff. Left at the SDK's default when None.
            client_kwargs: anything else the ``openai`` constructor accepts.
        """
        from openai import OpenAI as _OpenAIClient

        if concurrency is not None and concurrency < 1:
            raise ValueError(f'concurrency must be >= 1 or None, got {concurrency}')
        kwargs = dict(client_kwargs or {})
        for name, value in (('timeout', timeout), ('max_retries', max_retries)):
            if value is None:
                continue
            if name in kwargs:
                raise ValueError(f'{name} was passed both directly and in client_kwargs; '
                                 'drop one so that which value wins is not a matter of ordering')
            kwargs[name] = value

        self.model = model
        self.concurrency = concurrency
        # Held across the SDK's own retries too: a request that is backing off
        # still occupies the endpoint's attention, so it keeps its slot.
        self._slots: ContextManager[Any] = (
            threading.BoundedSemaphore(concurrency) if concurrency is not None else nullcontext())
        self._client = _OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def __call__(
        self,
        trajectory: Trajectory,
        sampling_params: SamplingParams,
        **kwargs,
    ) -> Union[Message, List[Message]]:
        request = self._build_request(trajectory, sampling_params, kwargs)
        with self._slots:
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
