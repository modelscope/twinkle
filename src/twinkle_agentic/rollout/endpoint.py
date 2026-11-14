# Copyright (c) ModelScope Contributors. All rights reserved.
"""An OpenAI-shaped door onto the rollout, for agents that drive their own loop.

A coding agent that ships as its own program cannot be called turn by turn: it
owns its loop, its tools and its context, and the only thing it will accept from
us is a base URL. This puts one in front of the rollout that holds the episode --
which is the whole point. Point such an agent at a model served somewhere else
and it trains on a policy that is not the one being updated; every generation is
off-policy by however far the two have drifted, and nothing reports it.

The endpoint never samples. A request is parsed into an :class:`EndpointRequest`
and put on the queue of the episode its API key names; the rollout's next step
takes it, samples (token-in: only the part of the conversation it has not seen
is encoded), records the round, and completes ``reply``. The endpoint then writes
that reply out in its protocol's wire format. So one step of the episode is one
request of the agent, and everything about accounting lives in the rollout.

Which episode a request belongs to is decided by the API key, never by matching
the conversation. Two episodes on the same task open with the same messages, so a
prefix match would hand one episode's tokens to the other.

Two protocols, one subclass each: :class:`ChatEndpoint` (``/v1/chat/completions``)
and :class:`ResponsesEndpoint` (``/v1/responses``).
"""
import asyncio
import json
import queue
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from twinkle.utils import find_node_ip, get_logger

logger = get_logger()

# How long to wait for the server thread to report itself up, and to wind down.
_STARTUP_TIMEOUT = 60.0
_SHUTDOWN_TIMEOUT = 10.0


@dataclass
class EndpointRequest:
    """One request of an agent, waiting for the rollout step that answers it.

    ``reply`` resolves to ``{'message': <assistant message, chat shape>,
    'finish_reason': 'stop' | 'tool_calls' | 'length', 'usage': {'prompt_tokens',
    'completion_tokens'}}``, or raises when the episode ended without answering.
    """

    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    params: Dict[str, Any]
    reply: Future = field(default_factory=Future)


def _tool_calls_for_wire(parsed: List[Dict[str, Any]], turn_id: str) -> List[Dict[str, Any]]:
    """Make parsed calls satisfy the wire contract clients validate against.

    The template's parser returns ``arguments`` as a dict, which is what a jinja
    chat template wants. The OpenAI protocol says it is a *string* of JSON, and
    clients unconditionally ``json.loads`` it -- handing them a dict raises inside
    the client, before the agent's own error handling can see it. An ``id`` is
    likewise assumed present, and tool results are addressed by it.
    """
    calls = []
    for index, call in enumerate(parsed):
        function = dict(call.get('function') or {})
        arguments = function.get('arguments', call.get('arguments'))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        calls.append({
            'id': call.get('id') or f'call_{turn_id}_{index}',
            'type': 'function',
            'index': index,
            'function': {
                'name': function.get('name') or call.get('name') or '',
                'arguments': arguments,
            },
        })
    return calls


def assistant_message(text: str, parsed_calls: List[Dict[str, Any]], cleaned: str) -> Dict[str, Any]:
    """The assistant message a reply carries: ``cleaned`` text plus wire-shaped tool calls when there are any."""
    message: Dict[str, Any] = {'role': 'assistant', 'content': cleaned if parsed_calls else text}
    if parsed_calls:
        message['tool_calls'] = _tool_calls_for_wire(parsed_calls, uuid.uuid4().hex[:12])
    return message


def _bearer(authorization: Optional[str]) -> str:
    if not authorization:
        return ''
    return authorization.split(' ', 1)[-1].strip() if ' ' in authorization else authorization.strip()


class PolicyEndpoint(ABC):
    """An OpenAI-protocol server whose requests are answered by rollout steps.

    One per rollout shard, started on first use. :meth:`open` gives an episode
    its queue under its key, :meth:`close` takes it away (failing whatever was
    still waiting); a request under an unknown key is refused with 401.

    Args:
        host: where to bind; defaults to this node's address, which an agent in
            a sandbox on another machine can route back to.
        port: 0 picks a free one, which is what lets many endpoints coexist.
    """

    #: The route this protocol serves.
    path: str = ''

    def __init__(self, *, host: Optional[str] = None, port: int = 0) -> None:
        self.host = host or find_node_ip()
        self.port = port
        self._queues: Dict[str, 'queue.Queue[EndpointRequest]'] = {}
        self._lock = threading.Lock()
        self._socket: Optional[socket.socket] = None
        self._server: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- episodes

    def open(self, key: str) -> 'queue.Queue':
        """Start taking requests under ``key``; returns the queue they arrive on.

        The episode may put its own markers on the same queue (an agent that exited, say), so that
        one ``get`` waits for whichever comes first.
        """
        with self._lock:
            return self._queues.setdefault(key, queue.Queue())

    def close(self, key: str) -> None:
        """Stop taking requests under ``key``; requests still queued fail."""
        with self._lock:
            pending = self._queues.pop(key, None)
        while pending is not None:
            try:
                request = pending.get_nowait()
            except queue.Empty:
                break
            if isinstance(request, EndpointRequest) and not request.reply.done():
                request.reply.set_exception(RuntimeError(f'episode {key!r} ended before this request was answered'))

    # ---------------------------------------------------------------- lifetime

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def base_url(self) -> str:
        """What to hand the agent as its endpoint."""
        if self._server is None:
            raise RuntimeError('the endpoint is not running: call start() first')
        return f'http://{self.host}:{self.port}/v1'

    def start(self) -> str:
        """Bring the server up and return :attr:`base_url`.

        The listening socket is bound here, before the server thread starts, so
        ``port=0`` can be resolved to a real port without racing: bind, read the
        port, then hand the already-bound socket to uvicorn.
        """
        if self._server is not None:
            raise RuntimeError('the endpoint is already running')
        # Imported here, not at module scope, so that using the rest of this
        # package does not require a web stack to be installed.
        import uvicorn

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.port = sock.getsockname()[1]
        self._socket = sock
        self._server = uvicorn.Server(uvicorn.Config(self._build_app(), log_level='warning'))
        self._thread = threading.Thread(
            target=self._server.run, kwargs={'sockets': [sock]}, name=f'policy-endpoint-{self.port}', daemon=True)
        self._thread.start()
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while not self._server.started:
            if not self._thread.is_alive():
                self._server = self._thread = None
                sock.close()
                self._socket = None
                raise RuntimeError('the endpoint thread exited before the server came up')
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f'the endpoint did not come up within {_STARTUP_TIMEOUT}s')
            time.sleep(0.01)
        return self.base_url

    def stop(self) -> None:
        """Take the server down and fail every waiting request. Safe to call when it is not running."""
        with self._lock:
            keys = list(self._queues)
        for key in keys:
            self.close(key)
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=_SHUTDOWN_TIMEOUT)
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._server = None
        self._thread = None

    # ----------------------------------------------------------------- serving

    def _build_app(self) -> Any:
        from fastapi import Body, FastAPI, Header
        from fastapi.responses import JSONResponse, StreamingResponse

        app = FastAPI()

        # Async: the handler only waits for the step that answers it, so a
        # thousand agents waiting on their episodes cost no threads.
        @app.post(self.path)
        async def serve(payload: Dict[str, Any] = Body(...), authorization: Optional[str] = Header(default=None)) -> Any:
            key = _bearer(authorization)
            with self._lock:
                pending = self._queues.get(key)
            if pending is None:
                return JSONResponse({'error': {'message': f'unknown api key {key!r}'}}, status_code=401)
            try:
                messages, tools, params = self.parse(payload)
            except ValueError as e:
                return JSONResponse({'error': {'message': str(e)}}, status_code=400)
            request = EndpointRequest(messages=messages, tools=tools, params=params)
            pending.put(request)
            try:
                reply = await asyncio.wrap_future(request.reply)
            except Exception as e:  # noqa
                return JSONResponse({'error': {'message': f'{type(e).__name__}: {e}'}}, status_code=500)
            body = self.render(payload, reply)
            if payload.get('stream'):
                return StreamingResponse(self.stream(body), media_type='text/event-stream')
            return body

        return app

    @abstractmethod
    def parse(self, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """``(messages, tools, params)`` in chat shape, from one request body; ValueError when malformed.

        ``params`` holds only the decoding knobs the request may adjust (``max_tokens``,
        ``temperature``, ``top_p``, ``stop``) -- never ``logprobs``, whose collection is a
        training decision an agent must not change.
        """

    @abstractmethod
    def render(self, payload: Dict[str, Any], reply: Dict[str, Any]) -> Dict[str, Any]:
        """The response body for ``reply`` (see :class:`EndpointRequest`)."""

    @abstractmethod
    def stream(self, body: Dict[str, Any]) -> Iterator[str]:
        """The rendered body re-served as the protocol's event stream.

        There is nothing to stream incrementally: a step returns a generation
        whole. So the reply goes out as one complete stream of events, which is a
        valid stream, not a partial implementation of a different protocol.
        """


def _decoding_params(payload: Dict[str, Any], max_tokens_keys: Tuple[str, ...]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in max_tokens_keys:
        if payload.get(key) is not None:
            params['max_tokens'] = int(payload[key])
            break
    if payload.get('temperature') is not None:
        params['temperature'] = float(payload['temperature'])
    if payload.get('top_p') is not None:
        params['top_p'] = float(payload['top_p'])
    if payload.get('stop'):
        params['stop'] = payload['stop']
    return params


class ChatEndpoint(PolicyEndpoint):
    """``/v1/chat/completions``, plain and streamed."""

    path = '/v1/chat/completions'

    def parse(self, payload):
        messages = list(payload.get('messages') or [])
        if not messages:
            raise ValueError('a chat-completions request must carry at least one message')
        if payload.get('n', 1) != 1:
            raise ValueError(f"this endpoint serves n=1 only, got {payload.get('n')}")
        return messages, list(payload.get('tools') or []), _decoding_params(payload,
                                                                            ('max_completion_tokens', 'max_tokens'))

    def render(self, payload, reply):
        usage = reply.get('usage') or {}
        prompt_tokens = int(usage.get('prompt_tokens', 0))
        completion_tokens = int(usage.get('completion_tokens', 0))
        return {
            'id': f'chatcmpl-{uuid.uuid4().hex[:12]}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': payload.get('model') or 'twinkle-policy',
            'choices': [{
                'index': 0,
                'message': reply['message'],
                'finish_reason': reply['finish_reason'],
            }],
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            },
        }

    def stream(self, body):
        choice = body['choices'][0]
        chunk = {
            'id': body['id'],
            'object': 'chat.completion.chunk',
            'created': body['created'],
            'model': body['model'],
            'choices': [{
                'index': 0,
                'delta': dict(choice['message']),
                'finish_reason': choice['finish_reason'],
            }],
            'usage': body['usage'],
        }
        yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
        yield 'data: [DONE]\n\n'


def _content_text(content: Any) -> str:
    """The text of a Responses content field: a string, or a list of ``*_text`` parts."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if isinstance(part, dict) and isinstance(part.get('text'), str):
            texts.append(part['text'])
    return ''.join(texts)


class ResponsesEndpoint(PolicyEndpoint):
    """``/v1/responses``, plain and as SSE events.

    Stateless: every request must carry the whole conversation in ``input``
    (``store=false``, which is what agents like codex send), since the episode's
    history lives in the rollout, not here. ``function_call`` items are produced;
    ``reasoning`` items in the input (encrypted or not) are dropped; tools other
    than ``type: function`` are not advertised to the policy.
    """

    path = '/v1/responses'

    def parse(self, payload):
        if payload.get('previous_response_id'):
            raise ValueError('this endpoint keeps no responses: send the whole conversation in input '
                             '(store=false) instead of previous_response_id')
        messages: List[Dict[str, Any]] = []
        if payload.get('instructions'):
            messages.append({'role': 'system', 'content': str(payload['instructions'])})
        items = payload.get('input')
        if isinstance(items, str):
            items = [{'role': 'user', 'content': items}]
        for item in items or []:
            kind = item.get('type', 'message')
            if kind == 'message':
                role = item.get('role', 'user')
                messages.append({'role': 'system' if role == 'developer' else role,
                                 'content': _content_text(item.get('content'))})
            elif kind == 'function_call':
                call = {
                    'id': item.get('call_id') or item.get('id') or f'call_{uuid.uuid4().hex[:12]}',
                    'type': 'function',
                    'function': {
                        'name': item.get('name', ''),
                        'arguments': item.get('arguments') or '{}'
                    },
                }
                if messages and messages[-1]['role'] == 'assistant':
                    messages[-1].setdefault('tool_calls', []).append(call)
                else:
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
            elif kind == 'function_call_output':
                output = item.get('output')
                messages.append({
                    'role': 'tool',
                    'tool_call_id': item.get('call_id', ''),
                    'content': output if isinstance(output, str) else _content_text(output),
                })
            # 'reasoning' and anything else carries nothing the policy is to be prompted with.
        if not any(m['role'] != 'system' for m in messages):
            raise ValueError('a responses request must carry input')
        tools = []
        for tool in payload.get('tools') or []:
            if tool.get('type') != 'function':
                continue
            tools.append({
                'type': 'function',
                'function': {
                    'name': tool.get('name', ''),
                    'description': tool.get('description', ''),
                    'parameters': tool.get('parameters') or {
                        'type': 'object',
                        'properties': {}
                    },
                },
            })
        return messages, tools, _decoding_params(payload, ('max_output_tokens', ))

    def render(self, payload, reply):
        message = reply['message']
        output: List[Dict[str, Any]] = []
        if message.get('content'):
            output.append({
                'type': 'message',
                'id': f'msg_{uuid.uuid4().hex[:12]}',
                'status': 'completed',
                'role': 'assistant',
                'content': [{
                    'type': 'output_text',
                    'text': message['content'],
                    'annotations': []
                }],
            })
        for call in message.get('tool_calls') or []:
            output.append({
                'type': 'function_call',
                'id': f'fc_{uuid.uuid4().hex[:12]}',
                'status': 'completed',
                'call_id': call['id'],
                'name': call['function']['name'],
                'arguments': call['function']['arguments'],
            })
        usage = reply.get('usage') or {}
        input_tokens = int(usage.get('prompt_tokens', 0))
        output_tokens = int(usage.get('completion_tokens', 0))
        truncated = reply['finish_reason'] == 'length'
        return {
            'id': f'resp_{uuid.uuid4().hex[:12]}',
            'object': 'response',
            'created_at': int(time.time()),
            'status': 'incomplete' if truncated else 'completed',
            'incomplete_details': {
                'reason': 'max_output_tokens'
            } if truncated else None,
            'model': payload.get('model') or 'twinkle-policy',
            'output': output,
            'parallel_tool_calls': True,
            'tool_choice': payload.get('tool_choice', 'auto'),
            'tools': payload.get('tools') or [],
            'store': False,
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
            },
        }

    def stream(self, body):
        sequence = iter(range(1 << 30))

        def event(kind: str, **data: Any) -> str:
            data = {'type': kind, 'sequence_number': next(sequence), **data}
            return f'event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'

        yield event('response.created', response={**body, 'status': 'in_progress', 'output': []})
        for index, item in enumerate(body['output']):
            if item['type'] == 'message':
                yield event('response.output_item.added', output_index=index, item={**item, 'content': []})
                part = item['content'][0]
                yield event('response.content_part.added', item_id=item['id'], output_index=index, content_index=0,
                            part={**part, 'text': ''})
                yield event('response.output_text.delta', item_id=item['id'], output_index=index, content_index=0,
                            delta=part['text'])
                yield event('response.output_text.done', item_id=item['id'], output_index=index, content_index=0,
                            text=part['text'])
                yield event('response.content_part.done', item_id=item['id'], output_index=index, content_index=0,
                            part=part)
            else:
                yield event('response.output_item.added', output_index=index, item={**item, 'arguments': ''})
                yield event('response.function_call_arguments.delta', item_id=item['id'], output_index=index,
                            delta=item['arguments'])
                yield event('response.function_call_arguments.done', item_id=item['id'], output_index=index,
                            arguments=item['arguments'])
            yield event('response.output_item.done', output_index=index, item=item)
        yield event('response.incomplete' if body['status'] == 'incomplete' else 'response.completed', response=body)


ENDPOINTS = {'chat': ChatEndpoint, 'responses': ResponsesEndpoint}
