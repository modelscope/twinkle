# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout for agents that are programs, not loops we call.

:class:`~.multi_turn.MultiTurnRollout` owns the conversation: it generates, reads
the tool calls, runs them, appends the results, generates again. That is the right
shape when the turn structure is ours to decide -- and the wrong shape for a
coding agent that already exists as a command. Such an agent owns its loop, its
tools, its context management and its notion of when a task is done; there is no
seam to call it turn by turn, and reimplementing its loop here to get one means
training a policy on a loop nobody will run at inference time.

So the control is inverted. We start it, it works, it exits. What it produces
comes back not through a return value but through the requests it made on the way:
the endpoint serves them from the training policy and reports each round, and the
accounts assemble into trajectories.

Three pieces, and each is usable without the others. The endpoint
(:class:`~.endpoint.PolicyEndpoint`) is an inference server that happens to report
rounds. The accounts (:class:`~.ledger.LedgerBook`) turn reported rounds into
trainable trajectories. :class:`~twinkle_agentic.agents.base.CliAgent` describes
how to invoke one particular program and knows nothing about training. This module
is the small amount of wiring between them.
"""
import subprocess
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template

from .base import STOP_GENERATION_ERROR, STOP_NO_TOOL, Rollout
from .endpoint import PolicyEndpoint, Round
from .ledger import LedgerBook
from .trace import TraceWriter

# What a timed-out command reports, following the convention of ``timeout(1)``.
# Spelled out here rather than imported so that this module does not depend on the
# environment package: any object with ``run_script`` can drive an episode.
TIMEOUT_EXIT_CODE = 124

# How much of a failing agent's output to carry on the trajectory.
_OUTPUT_TAIL = 4000


def default_task_text(trajectory: Trajectory) -> str:
    """The prompt, flattened into the single string a CLI agent accepts.

    Everything the caller wrote -- system framing and question both -- in order,
    because an agent program brings its own system prompt and would drop ours if
    only the last user message were passed.
    """
    parts = [
        str(message['content']) for message in (trajectory.get('messages') or [])
        if message.get('role') in ('system', 'user') and message.get('content')
    ]
    if not parts:
        raise ValueError('cannot build a task for an external agent: the trajectory has no '
                         'system or user message to send. Pass task_fn=... for a prompt that '
                         'lives somewhere else.')
    return '\n\n'.join(parts)


class ExternalRollout(Rollout):
    """One agent process per trajectory, trained on the requests it made.

    Args:
        sampler: what answers the agent's requests. Must be the sampler the
            trainer syncs, which is the reason this endpoint is local. A ready
            :class:`~.endpoint.PolicyEndpoint` is accepted in its place, and so is
            anything with the same ``base_url`` / ``start`` / ``stop`` /
            ``on_round`` surface -- a translation in front of an agent that does
            not speak the OpenAI protocol goes here, and needs no change to this
            class. The endpoint's ``on_round`` is taken over either way, since
            reported rounds are what there is to train on.
        agent: the program to run: a
            :class:`~twinkle_agentic.agents.base.CliAgent`, or any object with a
            ``command`` of the same signature, or that function on its own.
        template: for encoding the accounts. Defaults to the sampler's.
        sampling_params: the endpoint's decoding defaults. Set ``logprobs`` here
            if the loss needs rollout logprobs; an agent cannot ask for them.
        max_tokens: length limit per trajectory. Rounds past it are not recorded
            and the agent is left to finish on its own.
        timeout: seconds before the agent's command is killed. There is no other
            bound on an episode: the agent decides when it is done.
        endpoint_host: where to bind. Loopback serves an agent running on this
            machine or in a container sharing its network; a sandbox reached over
            a network needs an address it can route back to. Ignored when an
            endpoint was passed in, which is already bound or will bind itself.
        endpoint_port: 0 picks a free one.
        endpoint_concurrency: how many of the agents' requests may be generating
            at once. Defaults to anyio's 40, which is a ceiling a batch of more
            than 40 agents reaches -- see
            :class:`~.endpoint.PolicyEndpoint`. Ignored when an endpoint was
            passed in.
        task_fn: trajectory -> the text handed to the agent. Defaults to
            :func:`default_task_text`.

    Pass ``env=`` per call to run the agent somewhere other than this machine:
    anything with ``run_script(command, interpreter='shell', timeout=...) ->
    (exit_code, output)`` will do, and its ``workspace`` attribute, if it has one,
    becomes the working directory. Without one the command runs here, which is
    fine for a demo and a poor idea for a batch of agents writing files.

    One trajectory comes back per input, as every rollout must. An agent that
    rewrote its own history mid-episode produced more than one account (see
    :class:`~.ledger.LedgerBook`); the last is returned, being the one the reward
    is about, and ``segments`` on the trajectory says how many there were.
    """

    def __init__(
        self,
        sampler: Any,
        agent: Any,
        *,
        template: Optional[Template] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 1800.0,
        endpoint_host: str = '127.0.0.1',
        endpoint_port: int = 0,
        endpoint_concurrency: Optional[int] = None,
        concurrency: Optional[int] = None,
        task_fn: Optional[Callable[[Trajectory], str]] = None,
        tracer: Optional[TraceWriter] = None,
    ) -> None:
        self._init_common(
            sampling_params=sampling_params,
            concurrency=concurrency,
            tracer=tracer,
        )
        # An agent is whatever can name a command. Insisting on the base class
        # would mean a caller with a two-line command has to declare a class to
        # pass it, and a function is what such a caller has.
        self.agent = getattr(agent, 'command', agent)
        if not callable(self.agent):
            raise TypeError(f'agent must be a CliAgent, an object with a command(...) method, or '
                            f'that function itself; got {type(agent).__name__}')
        self.timeout = timeout
        self.task_fn = task_fn or default_task_text
        resolved = template if template is not None else getattr(sampler, 'template', None)
        if resolved is None:
            raise ValueError('ExternalRollout needs a template to encode the accounts, and the '
                             'sampler does not carry one: pass template=...')
        self.template = resolved
        self.book = LedgerBook(resolved, max_tokens=max_tokens)
        if hasattr(sampler, 'base_url'):
            # An endpoint, not a sampler. Its reports are redirected here rather
            # than merged with whatever it had: two books filing the same rounds
            # under keys only one of them ever closes is a leak, and a shared
            # endpoint saves one thread.
            self.endpoint = sampler
            self.endpoint.on_round = self._on_round
        else:
            self.endpoint = PolicyEndpoint(
                sampler,
                template=resolved,
                host=endpoint_host,
                port=endpoint_port,
                sampling_params=self.sampling_params,
                on_round=self._on_round,
                max_concurrent_requests=endpoint_concurrency,
            )

    def close(self) -> None:
        """Take the endpoint down. Idempotent, and the rollout runs again after it.

        Both properties matter to a caller that is a loop: the next call brings the
        endpoint back up, and nothing from the last round survives into it.
        """
        self.endpoint.stop()

    def _on_round(self, round_: Round) -> None:
        """File one served request. Never raises at the agent."""
        self.book.bank(round_.key, round_.prompt_token_ids, round_.sequence, messages=round_.messages)

    def _resolve_call(self, kwargs: Dict[str, Any], n: int) -> Dict[str, Any]:
        # Started here rather than in _run_one: this runs once, before the pool,
        # so there is no window in which two episodes both find it down and race
        # to bind the port.
        if not self.endpoint.running:
            self.endpoint.start()
        # per_trajectory: an environment is a working directory with state in it.
        # Sharing one between agents running at the same time is not interleaving,
        # it is two agents editing each other's files.
        return {'envs': self._broadcast(kwargs.get('env'), n, name='env', per_trajectory=True)}

    def _run_one(self, trajectory: Trajectory, index: int, ctx: Dict[str, Any]) -> Trajectory:
        env = ctx['envs'][index]
        # Unique per episode, and readable: this is the label the ledger's
        # alignment errors are raised under.
        key = f'ep{index}-{uuid.uuid4().hex[:8]}'
        command = self.agent(
            task=self.task_fn(trajectory),
            base_url=self.endpoint.base_url,
            api_key=key,
            workspace=str(getattr(env, 'workspace', '') or ''),
        )
        exit_code, output = self._spawn(env, command)
        ledgers = self.book.close(key)

        if not ledgers:
            # The agent never reached the endpoint under its key: wrong config,
            # a crash before the first call, or a config that overrode the key.
            # There is nothing to train on, and saying why beats an empty result.
            out = dict(trajectory)
            out['stop_reason'] = STOP_GENERATION_ERROR
            out['agent_exit_code'] = exit_code
            out['agent_output'] = output[-_OUTPUT_TAIL:] if output else ''
            return out

        fields: Dict[str, Any] = {
            'stop_reason': STOP_NO_TOOL if exit_code == 0 else STOP_GENERATION_ERROR,
            'agent_exit_code': exit_code,
            'segments': len(ledgers),
        }
        if exit_code != 0:
            # Only on failure: the tail is for diagnosing one, and carrying every
            # agent's stdout through a training batch is pure weight.
            fields['agent_output'] = output[-_OUTPUT_TAIL:] if output else ''
        return ledgers[-1].merge(trajectory, **fields)

    def _spawn(self, env: Any, command: str) -> Tuple[int, str]:
        """Run the agent to completion, in ``env`` if there is one."""
        if env is not None:
            return env.run_script(command, interpreter='shell', timeout=self.timeout)
        try:
            done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as expired:
            partial = expired.output or ''
            if isinstance(partial, bytes):
                partial = partial.decode(errors='replace')
            return TIMEOUT_EXIT_CODE, partial
        return done.returncode, (done.stdout or '') + (done.stderr or '')
