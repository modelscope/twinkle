# Copyright (c) ModelScope Contributors. All rights reserved.
"""Agent harness: framework-owned message/tool lifecycle, minus generate/execute.

``MultiTurnRollout`` owns batched sampling and ``new_input_feature`` extension.
``Env`` owns tool execution. A harness mutates the same :class:`Trajectory`
the rest of the stack already uses (``messages`` / ``tools`` / ``user_data``).

Only *append-only* mutations of ``messages`` are safe after the first encode:
rewriting earlier turns would break the token-id chain MultiTurn keeps in
``new_input_feature``. Implementations that compact/rewrite history must do
it in :meth:`start` / the first :meth:`before_generate` (before encode), or
opt in explicitly.

Who drives
----------
This interface assumes twinkle drives the loop: ``MultiTurnRollout`` decides when
to generate, and calls the harness in between. That is the direction a harness in
the wrapped framework's own process can be called in -- a python object whose
methods we invoke.

An agent that ships as its own program (a Node or Rust CLI) inverts that: it runs
the loop, and reaches the policy over HTTP. Nothing here fits that, and nothing
here should be bent to -- the object to write for those is not a harness but an
endpoint they can be pointed at, and the token ledger the run is trained on comes
from the sampler behind it (see ``rollout/ledger.py``, whose ``graft`` exists for
exactly that caller). What must hold either way is that the tokens trained on are
the tokens sampled, never text re-encoded after the fact.
"""
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle_agentic.utils.leases import Leases


class AgentHarness(ABC):
    """Per-episode agent-framework hooks.

    Default implementations are no-ops so MultiTurn can take ``harness=None``
    or a subclass that only overrides some phases. Subclasses that wrap a
    specific framework (ms-agent, …) live next to this file, not in
    ``rollout/`` or ``rsi/``. Harness-private runtime (LLMAgent, session)
    lives on the harness instance, not on the trajectory.

    That private runtime is why a harness belongs to *one* episode: memory,
    accumulated context and skill state are the framework's, not the
    trajectory's, so two episodes sharing an instance read each other's. See
    :class:`HarnessLeases`.
    """

    #: Whether :meth:`reset` really empties this harness, so one instance can
    #: serve one episode after another. Left False, :class:`HarnessLeases`
    #: throws the instance away between episodes and builds a fresh one --
    #: slower, but a harness that forgets to clean itself cannot leak state into
    #: the next episode. Set it True only alongside a ``reset`` that has been
    #: checked against every place the wrapped framework keeps state.
    reusable: bool = False

    def reset(self) -> None:
        """Forget the episode just finished. Only called when :attr:`reusable`."""
        raise NotImplementedError(f'{type(self).__name__} declares reusable=True but has no reset()')

    def close(self) -> None:
        """Release whatever the wrapped framework holds. Called once, at teardown."""

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-shaped tool list this harness puts in the prompt.

        The harness owns the tool *names and schemas* so training and serving
        advertise the identical set; the Env owns the *implementation*. Build
        the executing side from the same list::

            tm = env.tool_manager(harness.tool_schemas())

        Skipping that step lets the prompt advertise tools the Env cannot run,
        and every call comes back as an unknown-tool error.
        """
        return []

    def start(self, query: str, **kwargs) -> Trajectory:
        """Open an episode: system + user (+ tool schema).

        Called by the training driver *before* MultiTurn encodes. Not invoked
        by MultiTurn itself. Extra kwargs are merged onto the trajectory
        (``user_data``, ``tools``, …).
        """
        traj: Trajectory = {'messages': [{'role': 'user', 'content': query}]}
        traj.update(kwargs)
        return traj

    def before_generate(self, trajectory: Trajectory) -> Trajectory:
        """Mutate ``trajectory`` immediately before a generate turn.

        First call happens before the initial ``template.encode``. Later calls
        must be append-only relative to ``messages`` already in the pif,
        or MultiTurn will ignore the rewrite to protect token alignment.
        """
        return trajectory

    def after_generate(
        self,
        trajectory: Trajectory,
        decoded: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        """Normalize the assistant turn (content / tool_calls / reasoning).

        ``decoded`` and ``tool_calls`` come from the sampler; the pif already
        contains the generated tokens. This hook only updates message metadata
        so the next encode-bridge and the serving agent see the same shape.
        """
        return trajectory

    def after_tools(
        self,
        trajectory: Trajectory,
        observations: List[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        """Turn raw Env observations into ``role=tool`` messages (append).

        Default: one tool message per observation, copying ``id`` / ``name``
        from the corresponding tool call when present.
        """
        msgs = trajectory.setdefault('messages', [])
        calls = list(tool_calls or [])
        for i, obs in enumerate(observations):
            msg: Dict[str, Any] = {'role': 'tool', 'content': obs if obs is not None else ''}
            if i < len(calls):
                tc = calls[i] if isinstance(calls[i], dict) else {}
                fn = tc.get('function') if isinstance(tc.get('function'), dict) else {}
                tid = tc.get('id') or tc.get('tool_call_id')
                name = fn.get('name') or tc.get('name') or tc.get('tool_name')
                if tid:
                    msg['tool_call_id'] = tid
                if name:
                    msg['name'] = name
            msgs.append(msg)
        return trajectory


class HarnessLeases(Leases[AgentHarness]):
    """Lend one harness to one episode, built from a factory rather than given.

    A pool of harnesses cannot simply be handed in the way a pool of
    environments is, because the way to empty a harness is framework-specific
    and most frameworks offer no way at all: ms-agent's ``LLMAgent`` keeps
    memory, an assembled context and skill state with no ``reset`` between
    them. So the pool is told *how to build one* instead, and an instance that
    does not claim :attr:`AgentHarness.reusable` is replaced between episodes
    rather than cleaned.

    That default is the point of taking a factory: a harness for a framework
    nobody has audited yet is correct without its author doing anything, and
    ``reusable = True`` is an optimisation to opt into once the state is known.
    Building one is cheap for the shapes used in training -- a harness on the
    training host advertises tools it does not execute, so no tool runtime is
    stood up.

    Args:
        factory: builds a prepared harness, ready for :meth:`AgentHarness.start`.
        size: how many episodes may run at once. Match the worker count and a
            lease never blocks.
    """

    def __init__(self, factory: Callable[[], AgentHarness], size: int):
        if size < 1:
            raise ValueError(f'size must be >= 1, got {size}')
        self._factory = factory
        super().__init__([factory() for _ in range(size)])
        # The pool starts out clean, and a slot's first lease is the one that
        # gets the instance built above -- there are as many first leases as
        # slots, because a slot is only in the queue once. Without this the
        # first round would build every harness twice.
        self._fresh = size

    def _prepare(self, harness: AgentHarness) -> AgentHarness:
        with self._lock:
            if self._fresh:
                self._fresh -= 1
                return harness
        if harness.reusable:
            harness.reset()
            return harness
        harness.close()
        return self._factory()
