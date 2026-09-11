# Copyright (c) ModelScope Contributors. All rights reserved.
"""An OpenAI-shaped door onto a sampler, for agents that drive their own loop.

A coding agent that ships as its own program cannot be called turn by turn: it
owns its loop, its tools and its context, and the only thing it will accept from
us is a base URL. This puts one in front of a sampler, in the same process that
holds the weights -- which is the whole point. Point such an agent at a model
served somewhere else and it trains on a policy that is not the one being
updated; every generation is off-policy by however far the two have drifted, and
nothing reports it. Here there is no copy to drift: the sampler answering the
request is the sampler the trainer just synced.

Two things this deliberately does not do.

It does not keep an account. What makes an externally driven episode trainable is
recorded through ``on_round``, and the endpoint's only obligation is to report
each round honestly -- which key it came in under, which prompt ids the model
actually ran on, which tokens came back. Whether that is banked into a
:class:`~.ledger.TurnLedger`, written to disk, or ignored is the caller's
business. So this is equally usable as a plain inference server, and equally
usable by code that has its own idea of what a trajectory is.

It does not require a twinkle sampler. Anything with ``sample(inputs, params) ->
[SampleResponse]`` will do, so an outside generation backend can be dropped in
without inheriting anything.

Which episode a request belongs to is decided by the API key, never by matching
the conversation. Two episodes on the same task open with the same messages, so a
prefix match would hand one episode's tokens to the other's account -- and the
tokens would fit, which is why nothing would catch it. The key names the account;
the prefix check inside the ledger then asks the different question of whether
*this* account is still being extended, or whether the agent rewrote its history
behind our back.
"""
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterator, List, Optional

from twinkle.data_format.sampling import SampledSequence, SamplingParams
from twinkle.sampler.base import Sampler
from twinkle.template.base import Template

# How long to wait for the server thread to report itself up, and to wind down.
_STARTUP_TIMEOUT = 60.0
_SHUTDOWN_TIMEOUT = 10.0


@dataclass
class Round:
    """One request answered, described in the terms an account needs.

    A dataclass rather than four positional arguments because this is the seam
    between the endpoint and whoever is keeping score: adding a field later must
    not break every existing callback.
    """

    key: str
    """The API key the request arrived under -- the identity of the episode."""

    prompt_token_ids: List[int]
    """The ids the model was actually run on, straight from the sampler.

    Not the request's messages re-encoded afterwards. Re-encoding is where a
    trajectory quietly stops matching what was sampled; see
    :mod:`twinkle_agentic.utils.token_utils`.
    """

    sequence: SampledSequence
    """What came back: tokens, logprobs, stop reason."""

    messages: List[Dict[str, Any]]
    """The conversation as the agent sent it, plus the reply we returned.

    For traces and reward functions. The ids above are what gets trained.
    """


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


class PolicyEndpoint:
    """An OpenAI ``/v1/chat/completions`` server over one sampler.

    Usage is two lines: ``start()`` returns the base URL to hand the agent, and
    ``stop()`` takes it down. Also a context manager, which is the form to prefer
    when an episode owns the endpoint's lifetime.

    Args:
        sampler: Anything exposing ``sample(inputs, sampling_params)`` and
            returning ``SampleResponse`` objects carrying ``prompt_token_ids``.
        template: Used to read tool calls out of the reply text. Defaults to the
            sampler's own, which is normally the one that encoded the prompt --
            pass it explicitly only if the sampler has none.
        host: Bound loopback by default. This serves the training policy; there
            is no authentication here beyond the key naming an account.
        port: 0 picks a free one, which is what lets many endpoints coexist.
        sampling_params: The defaults every request starts from. Requests may
            override the usual decoding knobs, but not ``logprobs``: whether
            logprobs are collected is a training decision, and an agent that
            asked for a different number would silently change what is trainable.
        on_round: Called with a :class:`Round` after each reply, on the thread
            that served the request. Raising from it fails the request, so a
            callback that keeps an account should handle its own disagreements
            rather than let them reach the agent.
        max_concurrent_requests: How many requests may be generating at once.
            The handler is a blocking ``def``, so this is the size of the thread
            pool starlette runs it in; anyio's default is 40, and request 41
            waits for a thread rather than for the sampler. Raise it above the
            number of agents that can be running at the same time. It is a
            ceiling on requests in flight, not a promise about throughput: what
            they are all waiting on is one sampler, which does its own batching.
            None leaves the default alone.
    """

    def __init__(
        self,
        sampler: Any,
        *,
        template: Optional[Template] = None,
        host: str = '127.0.0.1',
        port: int = 0,
        sampling_params: Optional[SamplingParams] = None,
        on_round: Optional[Callable[[Round], None]] = None,
        max_concurrent_requests: Optional[int] = None,
    ) -> None:
        self.sampler = sampler
        # A twinkle sampler is asked whether it tolerates one request at a time:
        # this serves them as they arrive, and a slice_dp sampler spreads a batch
        # of one over every worker and raises on the ranks that get nothing.
        # Anything else is taken at its word -- the attribute is twinkle's, and
        # its absence means the sampler is not one rather than that it is broken.
        sample = getattr(type(sampler), 'sample', None)
        if isinstance(sampler, Sampler) and not getattr(sample, '_enable_continous_work', False):
            raise ValueError(f'{type(sampler).__name__}.sample must be declared with '
                             'enable_continous_work=True to serve an endpoint: requests arrive '
                             'one at a time, and a slice_dp sampler raises when a worker gets '
                             'nothing from a batch of one.')
        self.template = template if template is not None else getattr(sampler, 'template', None)
        if self.template is None:
            raise ValueError('PolicyEndpoint needs a template to read tool calls out of replies, '
                             'and the sampler does not carry one: pass template=...')
        params = sampling_params or SamplingParams()
        if params.num_samples != 1:
            # Each request is one turn of one conversation; a second sequence
            # would have nowhere to go and no account to be banked into.
            raise ValueError(f'PolicyEndpoint serves num_samples=1 only, got {params.num_samples}')
        self.sampling_params = params
        self.on_round = on_round
        self.host = host
        self.port = port
        if max_concurrent_requests is not None and max_concurrent_requests < 1:
            raise ValueError(f'max_concurrent_requests must be >= 1 or None, '
                             f'got {max_concurrent_requests}')
        self.max_concurrent_requests = max_concurrent_requests
        self._socket: Optional[socket.socket] = None
        self._server: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- lifetime

    @property
    def running(self) -> bool:
        """Is the server up? Lets a caller start it once without racing to check."""
        return self._server is not None

    @property
    def base_url(self) -> str:
        """What to put in the agent's ``OPENAI_BASE_URL``."""
        if self._server is None:
            raise RuntimeError('the endpoint is not running: call start() first')
        return f'http://{self.host}:{self.port}/v1'

    def start(self) -> str:
        """Bring the server up and return :attr:`base_url`.

        The listening socket is bound here, before the server thread starts, so
        ``port=0`` can be resolved to a real port without racing: bind, read the
        port, then hand the already-bound socket to uvicorn. Asking uvicorn to
        bind and then reading the port back leaves a window in which the agent
        has a URL that nothing is listening on yet.
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
        self._thread = threading.Thread(target=self._server.run, kwargs={'sockets': [sock]},
                                        name=f'policy-endpoint-{self.port}', daemon=True)
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
        """Take the server down. Safe to call when it is not running."""
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

    def __enter__(self) -> 'PolicyEndpoint':
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ----------------------------------------------------------------- serving

    def _build_app(self) -> Any:
        from contextlib import asynccontextmanager

        from fastapi import Body, FastAPI, Header
        from fastapi.responses import StreamingResponse

        @asynccontextmanager
        async def lifespan(_app: Any) -> Any:
            # Set here rather than in __init__ because the limiter lives in a
            # RunVar: it belongs to whichever event loop is running, and the one
            # that matters is the loop uvicorn started on the server thread.
            if self.max_concurrent_requests is not None:
                import anyio.to_thread
                anyio.to_thread.current_default_thread_limiter().total_tokens = self.max_concurrent_requests
            yield

        app = FastAPI(lifespan=lifespan)

        # A plain ``def``: the sampler call blocks, and FastAPI runs sync handlers
        # in a thread pool. Declared async it would hold the event loop for the
        # length of a generation and serialise every concurrent episode.
        @app.post('/v1/chat/completions')
        def chat_completions(payload: Dict[str, Any] = Body(...),
                             authorization: Optional[str] = Header(default=None)) -> Any:
            key = ''
            if authorization:
                key = authorization.split(' ', 1)[-1].strip() if ' ' in authorization else authorization.strip()
            completion = self.complete(payload, key=key)
            if payload.get('stream'):
                return StreamingResponse(_stream(completion), media_type='text/event-stream')
            return completion

        return app

    def complete(self, payload: Dict[str, Any], *, key: str = '') -> Dict[str, Any]:
        """Answer one chat-completions request and report the round.

        Public because it is the whole endpoint minus HTTP: a caller that already
        has a transport, or a test that wants no sockets, drives this directly.
        """
        messages = list(payload.get('messages') or [])
        if not messages:
            raise ValueError('a chat-completions request must carry at least one message')
        if payload.get('n', 1) != 1:
            raise ValueError(f"PolicyEndpoint serves n=1 only, got {payload.get('n')}")

        request: Dict[str, Any] = {'messages': messages}
        tools = payload.get('tools')
        if tools:
            request['tools'] = list(tools)
        response = self.sampler.sample([request], self._params_for(payload))[0]
        seq = response.sequences[0]

        turn_id = uuid.uuid4().hex[:12]
        # Decoded off the sampled ids rather than taken from ``seq.decoded``,
        # which keeps the template's end-of-turn marker. That marker reaching an
        # agent's message content is not cosmetic: it ends up in the files and
        # answers the agent writes from it.
        tokenizer = getattr(self.template, 'tokenizer', None)
        if tokenizer is not None and seq.tokens:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        else:
            text = seq.decoded or ''
        parsed = self.template.parse_tool_call(text)
        message: Dict[str, Any] = {
            'role': 'assistant',
            'content': self.template.clean_tool_call(text) if parsed else text,
        }
        if parsed:
            message['tool_calls'] = _tool_calls_for_wire(parsed, turn_id)
        finish_reason = 'tool_calls' if parsed else ('length' if seq.stop_reason == 'length' else 'stop')

        if self.on_round is not None:
            self.on_round(
                Round(
                    key=key,
                    prompt_token_ids=list(response.prompt_token_ids or []),
                    sequence=seq,
                    messages=messages + [message],
                ))

        prompt_tokens = len(response.prompt_token_ids or [])
        completion_tokens = len(seq.tokens or [])
        return {
            'id': f'chatcmpl-{turn_id}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': payload.get('model') or 'twinkle-policy',
            'choices': [{
                'index': 0,
                'message': message,
                'finish_reason': finish_reason,
            }],
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            },
        }

    def _params_for(self, payload: Dict[str, Any]) -> SamplingParams:
        """Let the request adjust decoding, but not what is trainable."""
        overrides: Dict[str, Any] = {}
        if payload.get('max_tokens') is not None:
            overrides['max_tokens'] = int(payload['max_tokens'])
        if payload.get('temperature') is not None:
            overrides['temperature'] = float(payload['temperature'])
        if payload.get('top_p') is not None:
            overrides['top_p'] = float(payload['top_p'])
        if payload.get('stop'):
            overrides['stop'] = payload['stop']
        return replace(self.sampling_params, **overrides) if overrides else self.sampling_params


def _stream(completion: Dict[str, Any]) -> Iterator[str]:
    """Re-serve a finished completion as the event stream clients ask for.

    Streaming is a client-side default -- ms-agent ships with it on -- and a
    client that asked for a stream will not read a plain body. There is nothing
    to stream incrementally: the sampler returns a generation whole. So the reply
    goes out as one chunk in the streaming shape, which is a valid stream of one
    event, not a partial implementation of a different protocol.
    """
    choice = completion['choices'][0]
    delta = dict(choice['message'])
    chunk = {
        'id': completion['id'],
        'object': 'chat.completion.chunk',
        'created': completion['created'],
        'model': completion['model'],
        'choices': [{
            'index': 0,
            'delta': delta,
            'finish_reason': choice['finish_reason'],
        }],
        'usage': completion['usage'],
    }
    yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
    yield 'data: [DONE]\n\n'
