# Copyright (c) ModelScope Contributors. All rights reserved.
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.base import API
from twinkle_agentic.tools.tool_manager import ToolManager
from .base import Rollout

# Termination reasons surfaced via ``trajectory['stop_reason']``.
_STOP_NO_TOOL = 'stop'
_STOP_LENGTH = 'length'
_STOP_MAX_TURNS = 'max_turns'
_STOP_API_ERROR = 'api_error'


class APIMultiTurnRollout(Rollout):
    """Multi-turn rollout over an OpenAI-compatible chat-completions API.

    Per-trajectory loop:
      1. POST ``messages + tools`` to the API; receive an assistant message
         (``content`` and/or structured ``tool_calls``).
      2. Append the assistant message to ``messages``.
      3. If the assistant emitted ``tool_calls``, dispatch each through the
         trajectory-bound :class:`ToolManager`, append one
         ``{role:'tool', tool_call_id, content}`` per call, then loop.
      4. Else terminate with ``stop_reason='stop'``.
      5. ``finish_reason='length'`` => terminate with ``stop_reason='length'``.
      6. ``turn >= max_turns`` => terminate with ``stop_reason='max_turns'``
         (and ``truncated=True``).

    Constructor and per-call override semantics intentionally mirror
    :class:`MultiTurnRollout`: ``tool_manager`` may be a single instance
    (broadcast) or a list aligned 1:1 with trajectories, and it is optional --
    a challenger inventing tasks has nothing to execute.

    Tool schema source: ``trajectory['tools']`` if present, else
    ``tool_manager.tool_infos()`` of the trajectory's manager. Caller is
    free to set neither — the API will simply be told there are no tools.

    Output trajectory shape (keys added to the input dict):
      * ``messages``: the full conversation including tool turns.
      * ``turns``: number of API round-trips actually performed.
      * ``stop_reason``: one of ``'stop' | 'length' | 'max_turns' | 'api_error'``.
      * ``truncated``: True iff terminated by ``max_turns`` or ``length``.
      * ``error``: error string when ``stop_reason == 'api_error'``.
    """

    def __init__(
        self,
        api: API,
        tool_manager: Optional[ToolManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        concurrency: int = 8,
        extra_body: Optional[Dict[str, Any]] = None,
        trace_dir: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        super().__init__()
        if api is None:
            raise ValueError('APIMultiTurnRollout requires an API client')
        if concurrency < 1:
            raise ValueError(f'concurrency must be >= 1, got {concurrency}')
        self._init_common(
            max_turns=max_turns,
            sampling_params=sampling_params,
            trace_dir=trace_dir,
            trace_callback=trace_callback,
            success_callback=success_callback)
        self.api = api
        self.tool_manager = tool_manager
        self.concurrency = concurrency
        self.extra_body = dict(extra_body or {})

    def __call__(
        self,
        trajectories: List[Trajectory],
        **kwargs,
    ) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError('APIMultiTurnRollout.__call__ expects a List[Trajectory]; '
                            'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        n = len(trajectories)
        if n == 0:
            return []

        sampling_params: SamplingParams = kwargs.get('sampling_params', self.sampling_params)
        tool_managers = self._broadcast(kwargs.get('tool_manager', self.tool_manager), n, name='tool_manager')
        extra_body = dict(self.extra_body)
        if 'extra_body' in kwargs and kwargs['extra_body']:
            extra_body.update(kwargs['extra_body'])

        # Per-trajectory thread pool. OpenAI ``/chat/completions`` is
        # one-conversation-per-call; concurrency only buys us network
        # parallelism, never batched compute.
        outs: List[Optional[Trajectory]] = [None] * n
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._run_one, trajectories[i], tool_managers[i], sampling_params, extra_body): i
                for i in range(n)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                outs[i] = fut.result()

        result_outs: List[Trajectory] = [o if o is not None else dict(trajectories[i]) for i, o in enumerate(outs)]
        if self.trace_dir:
            self._write_rollout_traces(result_outs, global_step=kwargs.get('global_step'))
        return result_outs

    # ------------------------------------------------------------------ private

    def _run_one(
        self,
        trajectory: Trajectory,
        tool_manager: Optional[ToolManager],
        sampling_params: SamplingParams,
        extra_body: Dict[str, Any],
    ) -> Trajectory:
        """Drive the API turn loop for a single trajectory.

        Never raises; API failures are encoded in ``stop_reason='api_error'``
        with the exception text in ``error``. This keeps one bad row from
        poisoning a whole rollout batch.
        """
        messages: List[Dict[str, Any]] = list(trajectory.get('messages') or [])
        tools = trajectory.get('tools')
        if tools is None and tool_manager is not None:
            tools = tool_manager.tool_infos() or None

        turn = 0
        stop_reason = _STOP_MAX_TURNS
        truncated = False
        error: Optional[str] = None

        while turn < self.max_turns:
            turn += 1
            req_traj = {'messages': messages}
            if tools:
                req_traj['tools'] = list(tools)
            try:
                reply = self.api(
                    req_traj, sampling_params, extra_body=extra_body) if extra_body else self.api(
                        req_traj, sampling_params)
            except Exception as exc:
                stop_reason = _STOP_API_ERROR
                error = f'{type(exc).__name__}: {exc}'
                truncated = True
                break

            assistant_msg = self._normalise_assistant(reply, turn)
            messages.append(assistant_msg)
            finish = assistant_msg.get('finish_reason')
            tool_calls = assistant_msg.get('tool_calls') or []

            if finish == 'length':
                stop_reason = _STOP_LENGTH
                truncated = True
                break
            if not tool_calls:
                stop_reason = _STOP_NO_TOOL
                break

            # Skip tool execution at the last turn — results would never be
            # consumed by a subsequent API call (consistent with multi_turn.py).
            if turn >= self.max_turns:
                truncated = True
                stop_reason = _STOP_MAX_TURNS
                break

            if tool_manager is None:
                # Nothing can run the call, so the conversation cannot continue:
                # say why rather than looping on an unanswered tool turn.
                stop_reason = _STOP_API_ERROR
                error = ('model emitted tool_calls but this rollout has no ToolManager; '
                         'pass one at construction time or as a per-call kwarg')
                truncated = True
                break

            try:
                for tc in tool_calls:
                    response = tool_manager(tc)
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.get('id'),
                        'content': str(response),
                    })
            except Exception as exc:
                stop_reason = _STOP_API_ERROR
                error = f'ToolExecution {type(exc).__name__}: {exc}'
                truncated = True
                break
        else:
            # Loop exited normally => max_turns reached.
            truncated = True
            stop_reason = _STOP_MAX_TURNS

        out = dict(trajectory)
        out['messages'] = messages
        out['turns'] = turn
        out['stop_reason'] = stop_reason
        out['truncated'] = truncated
        if error is not None:
            out['error'] = error
        return out

    @staticmethod
    def _normalise_assistant(reply: Any, turn: int) -> Dict[str, Any]:
        """Ensure tool_calls have stable ``id``/``type`` fields and strip
        message-internal noise that would confuse the next API turn.

        Some OpenAI-compatible servers (vLLM, SGLang) occasionally omit
        ``tool_call.id``; the assistant->tool round-trip needs a stable
        id to wire ``role:'tool'.tool_call_id`` back to the call site.
        """
        if not isinstance(reply, dict):
            return {'role': 'assistant', 'content': str(reply)}
        msg: Dict[str, Any] = {'role': 'assistant'}
        content = reply.get('content')
        msg['content'] = content if content is not None else ''
        finish = reply.get('finish_reason')
        if finish is not None:
            msg['finish_reason'] = finish
        tool_calls = reply.get('tool_calls') or []
        if tool_calls:
            normalised: List[Dict[str, Any]] = []
            for i, tc in enumerate(tool_calls):
                tc = dict(tc)
                tc.setdefault('id', f'call_{turn}_{i}')
                tc.setdefault('type', 'function')
                normalised.append(tc)
            msg['tool_calls'] = normalised
        # Reasoning content is informational only; keep it for trace
        # forensics but it is never re-fed to the API.
        reasoning = reply.get('reasoning_content')
        if reasoning:
            msg['reasoning_content'] = reasoning
        return msg

    def _build_trace_record(
        self,
        traj: Dict[str, Any],
        *,
        idx: int,
        success: bool,
    ) -> Dict[str, Any]:
        """The shared record, plus the two fields only this loop produces.

        ``turns`` counts API round-trips and ``error`` carries the exception
        text behind ``stop_reason='api_error'`` -- without it a trace of a
        failed batch shows an empty conversation and no reason.
        """
        record = super()._build_trace_record(traj, idx=idx, success=success)
        record['turns'] = traj.get('turns')
        if traj.get('error'):
            record['error'] = traj['error']
        return record
