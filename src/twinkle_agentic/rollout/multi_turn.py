# Copyright (c) ModelScope Contributors. All rights reserved.
"""The rollout: one episode advanced one step at a time.

``step(traj)`` takes an episode one generation further -- with its tool round, or with one
request of an agent program -- and returns ``(traj, RolloutStatus)``. The first step of an
episode opens it (its env session, its tools, its agent); the step that ends it returns the
merged, trainable trajectory and releases everything. A caller that drives many episodes (the
streaming driver) interleaves their steps; a caller that wants whole episodes uses ``__call__``.
"""
import json
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from twinkle.data_format import Trajectory, user_data_get
from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams
from twinkle.infra import remote_class, remote_function
from twinkle.protocol.types.sampler import SampleResponseModel
from twinkle.template.base import Template
from twinkle.utils import get_logger
from twinkle_agentic.envs.base import NoSlot
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.protocol.api_sampler import APIGenerationError
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.verifier.result_check import CheckContext, checks_from_dicts, run_checks
from .endpoint import ENDPOINTS, EndpointRequest, PolicyEndpoint, assistant_message
from .ledger import MessageLedger, TurnLedger

logger = get_logger()


def _final_answer(traj: Trajectory) -> str:
    """The last assistant turn's text -- what the ``answer_*`` checks read; '' when there is none.

    Multimodal content is a part list; the text parts concatenate in order.
    """
    content = None
    for message in reversed(traj.get('messages') or []):
        if message.get('role') == 'assistant':
            content = message.get('content')
            break
    if content is None:
        return ''
    if isinstance(content, list):
        return ''.join(str(part.get('text') or '') for part in content if isinstance(part, dict))
    return str(content or '')


def _dump_rollout_rows(episode_id: str, rows: list) -> None:
    """Write each delivered training row's contract arrays when ``TWINKLE_ROLLOUT_TRAJ_DUMP`` names a dir.

    An independent depth oracle for the rollout trajectory contract (G28). The producer already
    asserts ``validate_trajectory`` at ``ledger.merge``, but a self-consistent encoding can still
    pass its own check; dumping the arrays lets an offline judge re-derive the shifted-labels,
    logps-alignment and mask-count contracts from the numbers themselves, and compare an episode's
    segments for prefix-extension. Write-only and env-gated: a no-op (zero cost) when unset.
    """
    import os
    root = os.environ.get('TWINKLE_ROLLOUT_TRAJ_DUMP')
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    rank = int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0)
    path = os.path.join(root, f'rows.rank{rank}.jsonl')

    def _grid(value: Any) -> Any:
        # image_grid_thw / video_grid_thw arrive as a tensor (one row per media item);
        # the judge re-derives the placeholder-token count from the grid, so dump it as
        # plain nested ints. Absent on text rows -- leave it None, not an empty grid.
        if value is None:
            return None
        tolist = getattr(value, 'tolist', None)
        return tolist() if callable(tolist) else value

    def _media_count(value: Any) -> Optional[int]:
        return len(value) if isinstance(value, (list, tuple)) else None

    with open(path, 'a', encoding='utf-8') as handle:
        for seg, row in enumerate(rows):
            messages = row.get('messages') or []
            rec = {
                'episode_id': episode_id,
                'segment_index': row.get('segment_index', seg),
                'rank': rank,
                'input_ids': row.get('input_ids') or [],
                'labels': row.get('labels') or [],
                'completion_mask': row.get('completion_mask'),
                'response_token_ids': row.get('response_token_ids') or [],
                'response_positions': row.get('response_positions') or [],
                'rollout_logps': row.get('rollout_logps'),
                'messages': messages,
                'stop_reason': row.get('stop_reason'),
                # D6 media frame: raw media counts + the processor grid. The judge checks
                # image-count consistency and the placeholder-token expansion against the grid.
                'images_count': _media_count(row.get('images')),
                'videos_count': _media_count(row.get('videos')),
                'audios_count': _media_count(row.get('audios')),
                'image_grid_thw': _grid(row.get('image_grid_thw')),
                'video_grid_thw': _grid(row.get('video_grid_thw')),
            }
            handle.write(json.dumps(rec, default=str) + '\n')


# Termination reasons surfaced via ``trajectory['stop_reason']``. The sampler
# path takes the first three from the sampler itself; the API path has to name
# them, and one vocabulary for both is what lets a consumer read either.
STOP_NO_TOOL = 'stop'
STOP_LENGTH = 'length'
STOP_MAX_TURNS = 'max_turns'
STOP_GENERATION_ERROR = 'generation_error'

# Runaway guard: a ``followup_fn`` is expected to return None eventually. This
# only bounds a callback that never does, so one bad hook cannot spin forever.
MAX_FOLLOWUPS = 20

#: How many steps one rollout replica may run at once (Ray actor concurrency). A performance
#: ceiling only: how many episodes are in flight is the driver's admission, not this.
ROLLOUT_MAX_CONCURRENCY = 1024

def _record_dispatch_probe(event: str, episode_id: str, **fields: Any) -> None:
    """Append one dispatch event per call when ``TWINKLE_DISPATCH_PROBE`` names a directory.

    A deadlock can show eight env sessions held by live agents while no thread is inside
    ``_harness_step`` answering them. Stacks say where everything is blocked, not the dispatch
    order that got it there, so this records each step entry, each session open, and each harness
    wait/item/serve per episode. Env-gated: unset, it does nothing.
    """
    import json
    import os

    root = os.environ.get('TWINKLE_DISPATCH_PROBE')
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'dispatch.{os.getpid()}.jsonl')
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'t': time.monotonic(), 'event': event, 'episode_id': episode_id,
                                 **fields}, default=str) + '\n')


#: Put on an episode's request queue when its agent program exits.
_AGENT_EXITED = object()
#: Seconds past the agent's own timeout before a step stops waiting for it.
_AGENT_GRACE = 60.0
#: How much of a failing agent's output to carry in the error.
_OUTPUT_TAIL = 4000

ResponseCallback = Callable[[Trajectory, Any, SamplingParams], SampledSequence]


class EpisodeError(RuntimeError):
    """An episode that cannot produce a sample. The driver skips it and counts it.

    Raised for a prompt that leaves no room for one generated token, an agent that timed out,
    exited non-zero or never called the endpoint, tool markup still malformed after the retries,
    a failed API generation before any turn, and a step for an episode this rollout no longer
    holds. Any other exception is a bug and stops the run.
    """


@dataclass(frozen=True)
class StreamDelta:
    """One token delta of the current generation, for live display."""

    text: str


@dataclass(frozen=True)
class StreamToolCall:
    """One turn's dispatched tool calls: the function names and what each returned."""

    names: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StreamDone:
    """The finished episode: a message-level transcript trajectory (NOT trainable -- no token fields)."""

    trajectory: Trajectory


@dataclass
class RolloutStatus:
    """The light half of one step: what the driver and the recipe read, never the tokens.

    The trajectory itself travels apart (by reference under ray). The rollout fills the episode
    fields; the driver stamps the group, the version and the scores.
    """

    # Group identity and role, stamped by the driver.
    prompt_id: str = ''
    episode_id: str = ''
    group_key: str | None = None
    #: '' for a policy sample, 'baseline' for a baseline episode a group asked for.
    role: str = ''
    # The policy version the episode was admitted under, and how far the policy moved by handover.
    policy_version: int = 0
    version_span: int = 0
    # Episode state, filled by the rollout.
    turns: int = 0
    done: bool = False
    stop_reason: str | None = None
    truncated: bool = False
    #: The opening ``env.reset`` found no free session permit and was asked not to wait (wait=0).
    #: The episode produced no sample and is not failed: the driver retries this member later.
    no_slot: bool = False
    #: (prompt_tokens, response_tokens) of the final trajectory; None before the episode ends.
    token_lengths: tuple[int, int] | None = None
    #: Whether the final trajectory carries the sampler-exported MoE routing.
    routing_exported: bool = False
    #: The episode's reward from its env session (the sum of its step rewards).
    env_reward: float = 0.0
    # Scores, filled by the driver: one value per ORM; per PRM, one value per segment.
    orm_reward: list[float] | None = None
    prm_rewards: list[list[float]] | None = None
    #: The ORM row of the group's baseline episode, when it has one.
    baseline: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _single_sequence(responses: Any) -> SampledSequence:
    if not isinstance(responses, list):
        raise TypeError(f'expected List[SampleResponse] from sampler.sample, got {type(responses).__name__}')
    if len(responses) != 1:
        raise RuntimeError(f'sampler returned {len(responses)} responses for a single request; expected exactly one.')
    response = responses[0]
    if not isinstance(response, (SampleResponse, SampleResponseModel)):
        raise TypeError(f'expected SampleResponse or SampleResponseModel from sampler.sample, '
                        f'got {type(response).__name__}')
    if len(response.sequences) != 1:
        raise RuntimeError(f'SampleResponse contains {len(response.sequences)} sequences; expected exactly one.')
    sequence = response.sequences[0]
    if isinstance(response, SampleResponseModel):
        return SampledSequence(**sequence.model_dump())
    return sequence


def is_error_observation(observation: str) -> bool:
    """Did a tool come back with a failure rather than a result?

    Only the two shapes tools actually produce are matched, taken from a dump of
    239 real calls: ms-agent wraps a failure as ``{"success": false, ...}``, and
    a dispatch that never reached a tool (unknown name, a file the tool refuses
    to touch) comes back as a bare line starting with ``Error:``. Plus the two
    messages an unreachable sandbox produces.

    Deliberately narrow. Matching on words like ``failed`` or ``not found``
    anywhere in the text also matches a *successful* read of a file that happens
    to contain them, and this decides whether an episode is cut short.
    """
    text = (observation or '').strip()
    if not text:
        return False
    if text.startswith('Error:'):
        return True
    if text.startswith(('Tool runtime unreachable:', 'Tool runtime returned no result')):
        return True
    return bool(re.search(r'"success"\s*:\s*false', text))


def _call_key(tool_call: dict[str, Any]) -> str:
    """A stable identity for a tool call: its name plus its arguments verbatim.

    Byte-identical is the point. A model that changes one path and tries again is
    making progress; one that reissues the same call with the same arguments is
    not, whatever the tool answered.
    """
    fn = tool_call.get('function') if isinstance(tool_call.get('function'), dict) else {}
    name = fn.get('name') or tool_call.get('name') or tool_call.get('tool_name') or ''
    args = fn.get('arguments', tool_call.get('arguments'))
    if not isinstance(args, str):
        try:
            args = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args = repr(args)
    return f'{name}\x00{args}'


def _tool_call_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    """The function names of this turn's calls, in order, for display.

    Reads the same three shapes :func:`_call_key` does (``function.name`` / ``name`` / ``tool_name``);
    a call whose name cannot be read is dropped rather than shown as an empty string.
    """
    names: list[str] = []
    for tool_call in tool_calls:
        fn = tool_call.get('function') if isinstance(tool_call.get('function'), dict) else {}
        name = fn.get('name') or tool_call.get('name') or tool_call.get('tool_name') or ''
        if name:
            names.append(name)
    return names


def _default_tool_messages(
    tool_calls: list[dict[str, Any]],
    observations: list[str],
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for i, obs in enumerate(observations):
        msg: dict[str, Any] = {'role': 'tool', 'content': '' if obs is None else str(obs)}
        if i < len(tool_calls) and isinstance(tool_calls[i], dict):
            tc = tool_calls[i]
            fn = tc.get('function') if isinstance(tc.get('function'), dict) else {}
            tid = tc.get('id') or tc.get('tool_call_id')
            name = fn.get('name') or tc.get('name') or tc.get('tool_name')
            if tid:
                msg['tool_call_id'] = tid
            if name:
                msg['name'] = name
        msgs.append(msg)
    return msgs


def _malformed_tool_message(errors: list[str]) -> dict[str, Any]:
    """What goes back to the model when its tool-call markup did not parse.

    ``role='tool'`` because it is the outcome of the call the model just tried to
    make. There is no ``tool_call_id`` to pair it with -- the call never became a
    call -- which ``_default_tool_messages`` above already treats as optional.
    """
    reason = '; '.join(e for e in errors if e) or 'the markup could not be parsed'
    return {
        'role': 'tool',
        'content': ('Your tool call was not run: ' + reason + '. Send the call again. Inside '
                    'a JSON string a backslash has to be written as \\\\ and a line break as '
                    '\\n; a single quote needs no backslash at all.'),
    }


def _episode_id_of(trajectory: Any) -> Optional[str]:
    if not isinstance(trajectory, dict):
        return None
    episode_id = trajectory.get('episode_id')
    if episode_id is None:
        episode_id = user_data_get(trajectory.get('user_data'), 'episode_id')
    return str(episode_id) if episode_id else None


def _episode_key(args: tuple, kwargs: dict) -> Optional[str]:
    """The routing key of ``step``/``stop``: the episode id, so every step of an episode meets its state.

    Read off ``episode_id`` when the caller passes it (the trajectory may be a reference the caller
    cannot open), else off the trajectory's ``user_data``.
    """
    episode_id = kwargs.get('episode_id') or (args[1] if len(args) > 1 else None)
    if episode_id:
        return str(episode_id)
    return _episode_id_of(kwargs.get('traj', args[0] if args else None))


def _canonical(message: dict[str, Any]) -> tuple:
    """A message reduced to what an agent cannot reformat without changing it.

    The agent echoes our replies back in its own serialisation: tool-call arguments re-dumped,
    content ``None`` for ``''``, whitespace trimmed. Comparing these forms is what tells a
    conversation that grew from one that was rewritten.
    """
    content = message.get('content')
    if isinstance(content, list):
        content = ''.join(part.get('text') or '' for part in content if isinstance(part, dict))
    calls = []
    for call in message.get('tool_calls') or []:
        fn = call.get('function') if isinstance(call.get('function'), dict) else {}
        arguments = fn.get('arguments')
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass
        calls.append((fn.get('name') or '', json.dumps(arguments, sort_keys=True, ensure_ascii=False)))
    return (message.get('role'), str(content or '').strip(), tuple(calls), message.get('tool_call_id') or '')


class _Episode:
    """One episode's state on the rollout replica that runs it."""

    def __init__(self, episode_id: str, trajectory: Trajectory, params: SamplingParams):
        self.episode_id = episode_id
        self.trajectory = trajectory
        self.params = params
        self.ledger = None
        self.tool_manager: Optional[ToolManager] = None
        self.env_open = False
        #: Set by ``_discard``. ``_open`` blocks in ``env.reset`` while every session slot is taken, and
        #: the episode can be discarded meanwhile, so the call that returns afterwards has to be able to
        #: tell that nobody owns its session any more. See ``MultiTurnRollout._open``.
        self.discarded = False
        self.touched = time.monotonic()
        self.busy = False
        # The tool loop: see ``MultiTurnRollout._tool_step``.
        self.budget = 0
        self.spent = 0
        self.followups = 0
        self.malformed = 0
        self.stuck = 0
        self.seen_calls: set = set()
        self.stuck_stop = False
        self.tool_stop: Optional[str] = None
        self.truncated = False
        self.stop_reason: Optional[str] = None
        self.error: Optional[str] = None
        # The agent program: see ``MultiTurnRollout._harness_step``.
        self.requests: Optional[queue.Queue] = None
        self.exit: Optional[tuple] = None
        self.deadline = 0.0
        self.segments = 0
        self.sent: list[tuple] = []
        self.tools_key: Optional[str] = None
        #: The last served reply, for resubmitted requests (an agent HTTP client that timed out
        #: and retried sends the same payload verbatim: it is answered verbatim, not re-served).
        self.last_reply: Optional[dict] = None
        #: Continue-on-length (defect110): while an assistant turn is being continued, its reply
        #: reaches the agent as several wire messages the agent merges into ONE. ``merged_content``
        #: accumulates their content (None = no open split); ``split_start`` is where the split
        #: pieces begin in ``sent``, so the agent's merged view can be reconstructed for detection.
        self.merged_content: str | None = None
        self.split_start = 0
        #: Under segment_policy='collect': the rows of segments a compaction boundary already
        #: froze, oldest first; the final segment's row joins them when the episode ends.
        self.closed_rows: list = []


@remote_class(max_concurrency=ROLLOUT_MAX_CONCURRENCY)
class MultiTurnRollout:
    """Agentic multi-turn rollout, one step per call.

    An episode is driven one of two ways:

    * **tool loop** (no ``harness``): each step is one generation and, when it called tools,
      one round of them::

          sampler.sample (or response_callback)  -> SampledSequence
          ToolManager.call_many                  # this turn's calls, one Env round trip
          ledger.observe                         # labels=-100; never decode-reencode history

      Tools are the shared ``tool_manager`` (calculator, search API, ... -- anything not tied
      to a session) plus, with an ``env``, the env's tools bound to the episode's session.
      Every part of the loop except the generation is optional: no tools and no
      ``followup_fn`` is a single-turn sampling pass, a supported way to use this.

    * **agent program** (``harness``, which needs an ``env``): the first step starts the
      program in the episode's env session, pointed at this replica's
      :class:`~.endpoint.PolicyEndpoint`; each step answers one of its requests. The prompt
      it sends is token-in: when it extends what was recorded only the new part is encoded;
      when the agent rewrote its history a fresh segment starts (the last one is returned,
      ``segments`` says how many there were). The step after the program exits returns the
      trajectory.

    Steps of one episode route to the replica holding it (by ``episode_id``, read from the
    ``episode_id`` argument or ``user_data``). Intermediate steps return a message view;
    the last returns the trajectory merged with its token account (``input_ids`` / ``labels``
    / ``completion_mask`` / ``logprobs`` / multimodal features).

    Args:
        sampler: answers each generation. With a ``template`` it must return the sampled ids
            (``new_input_feature``); a twinkle sampler must declare ``sample`` with
            ``enable_continous_work``. An :class:`~..protocol.api_sampler.APISampler` works
            here as well. Without a template the episode is accounted in messages (a
            text-only backend), and the result is a transcript, not a trainable sample.
        template: encodes the opening and every bridge after it.
        sampling_params: per-generation params; ``user_data.sampling_params`` overrides them
            for one episode. ``logprobs`` is forced with a template (they are old_logps).
        max_turns: generations of the tool loop (each granted follow-up buys one more).
        max_trajectory_tokens: length past which the episode ends truncated.
        env: an :class:`~..envs.Env` component; one session per episode.
        tool_manager: tools shared by every episode, copied per episode.
        harness: a :class:`~..harness.CliAgent`; the episode is run by that program.
        response_callback: ``(view, state_or_feature, params) -> SampledSequence``, replaces
            the sampler call (e.g. to route some turns to another backend).
        followup_fn: ``(trajectory, n_followups) -> text | (text, SamplingParams) | None``.
        on_episode_end: ``(trajectory, env, episode_id) -> value``, run before the env session
            closes (a check that reads the workspace); a dict lands in ``RolloutStatus.extra``.
        stop_after_stuck_turns: stuck tool turns in a row that end the episode; 0 never.
        max_malformed_retries: replies with unparsable tool markup told so and retried.
        adapter_path / use_base_model: which weights every generation uses.
        concurrency: threads ``__call__`` runs episodes on; None is one per trajectory.
    """

    #: Seconds an episode may go without a step before another step releases it (a driver that
    #: died or never called ``stop`` would otherwise hold its session and agent forever).
    episode_idle_timeout = 1800.0

    def __init__(
        self,
        sampler=None,
        template: Template | None = None,
        sampling_params: SamplingParams | None = None,
        max_turns: int = 6,
        max_trajectory_tokens: int | None = None,
        env=None,
        tool_manager: ToolManager | None = None,
        harness=None,
        response_callback: ResponseCallback | None = None,
        followup_fn: Callable[[Trajectory, int], Any] | None = None,
        on_episode_end: Callable[[Trajectory, Any, str], Any] | None = None,
        stop_after_stuck_turns: int = 0,
        max_malformed_retries: int = 2,
        adapter_path: str | None = None,
        use_base_model: bool = False,
        concurrency: int | None = None,
        segment_policy: str = 'drop',
        seed: int | None = None,
    ):
        if sampler is None and response_callback is None:
            raise ValueError('MultiTurnRollout requires a sampler or a response_callback.')
        if sampler is not None and template is not None and getattr(
                getattr(type(sampler), 'sample', None), '_enable_continous_work', None) is False:
            # A remote sampler whose sample is not continuous-work spreads each one-request
            # call over every worker and raises on the ranks that receive nothing.
            raise ValueError(f'{type(sampler).__name__}.sample must be declared with enable_continous_work=True: '
                             'this rollout samples one trajectory per call.')
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f'max_trajectory_tokens must be >= 1 or None, got {max_trajectory_tokens}')
        if concurrency is not None and concurrency < 1:
            raise ValueError(f'concurrency must be >= 1 or None, got {concurrency}')
        if adapter_path and use_base_model:
            raise ValueError('adapter_path and use_base_model=True ask for opposite '
                             'weights; the sampler would drop the adapter silently.')
        if harness is not None:
            if env is None:
                raise ValueError('a harness runs its program in an env session: pass env= '
                                 '(LocalEnv for this machine).')
            if template is None:
                raise ValueError('a harness needs a template to account the requests it makes.')
            if tool_manager is not None or followup_fn is not None:
                raise ValueError('a harness owns its tools and its loop; tool_manager= and followup_fn= '
                                 'apply to the tool loop only.')
            if harness.protocol not in ENDPOINTS:
                raise ValueError(f'harness protocol {harness.protocol!r} is not one of {sorted(ENDPOINTS)}')
        # How many stuck turns in a row end the episode; 0 runs to ``max_turns``
        # regardless. A turn is stuck when it made no progress at all: every call in
        # it came back an error, or every call in it was byte-identical to one already
        # made in this episode. Both halves are needed, measured by replaying 12
        # recorded episodes: errors alone stop 1 of 12 and save 9 of 239 calls; adding
        # the repeat rule stops 3 of 12 and saves 63 calls.
        if stop_after_stuck_turns < 0:
            raise ValueError(f'stop_after_stuck_turns must be >= 0, got {stop_after_stuck_turns}')
        # How a compaction boundary treats the segment it writes out: 'drop' (the default) keeps
        # only the final segment trainable -- the episode trains as one row; 'collect' freezes
        # each written-out segment into its own row, and the episode is delivered as several
        # independently-coded rows that share one identity and one outcome reward.
        if segment_policy not in ('drop', 'collect'):
            raise ValueError(f"segment_policy must be 'drop' or 'collect', got {segment_policy!r}.")
        # How many replies in a row may carry tool-call markup that does not parse
        # before the episode is given up. Such a reply asked for a tool and had the
        # markup rejected, so it gets the parser's reason back and another turn.
        if max_malformed_retries < 0:
            raise ValueError(f'max_malformed_retries must be >= 0, got {max_malformed_retries}')
        if template is not None:
            assert template.truncation_strategy != 'split', (
                "MultiTurnRollout does not support truncation_strategy='split'; "
                'use left/right/delete/raise on the template.')
        self.sampler = sampler
        self.template = template
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.env = env
        self.tool_manager = tool_manager
        self.harness = harness
        self.response_callback = response_callback
        # Called with (trajectory, follow-ups so far) at the moment an episode would end:
        # because the model stopped calling tools, used up ``max_turns``, or was stuck.
        # Text appends a user message and the episode goes on (tools withdrawn); None
        # ends it. Appending keeps a multi-stage episode in ONE trajectory, so the
        # labels and logprobs of the earlier turns survive.
        self.followup_fn = followup_fn
        self.on_episode_end = on_episode_end
        self.stop_after_stuck_turns = stop_after_stuck_turns
        self.max_malformed_retries = max_malformed_retries
        # Left out entirely when unset, so a sampler without LoRA sees the call it always did.
        self._adapter_kwargs: dict[str, Any] = {'adapter_path': adapter_path} if adapter_path else {}
        if use_base_model:
            self._adapter_kwargs['use_base_model'] = True
        self.adapter_path = adapter_path
        self.concurrency = concurrency
        self.segment_policy = segment_policy
        self._base_seed = seed
        self._template_backend = getattr(type(sampler), 'template_backend', 'vllm') if sampler is not None else 'vllm'
        self.sampling_params = self._params(sampling_params or SamplingParams())
        self._episodes: dict[str, _Episode] = {}
        self._lock = threading.Lock()
        self._endpoint_lock = threading.Lock()
        self._policy_endpoint: Optional[PolicyEndpoint] = None
        self._env_tool_infos: Optional[list] = None

    # ------------------------------------------------------------------ entries

    @remote_function(enable_continous_work=True, sticky_key=_episode_key, stash=(0,), lazy_collect=True)
    def step(self, traj: Trajectory, episode_id: str | None = None,
             session_wait: Optional[float] = None) -> tuple[list, RolloutStatus]:
        """Advance the episode one generation; ``(rows, sample)`` with ``sample.done`` on the last.

        The first call of an episode opens it from ``traj``; later calls pass the first ``row``
        of what the previous one returned (only its episode id is read). An unfinished episode
        returns one row, its message view; the last step returns the episode's training rows:
        one under the default ``segment_policy='drop'`` (or an episode that never compacted),
        one per collected segment otherwise -- independently coded, sharing one identity and
        one outcome. Raises :class:`EpisodeError` for an episode that cannot produce a sample;
        it is released either way.

        ``session_wait`` is forwarded to the opening ``env.reset`` only: the driver's admission
        loop passes ``0`` so a full session pool returns a ``no_slot`` status (retry later) instead
        of blocking; the batch ``__call__`` and ``stream`` leave it ``None`` (inherit the env default).
        """
        return self._step(traj, episode_id, session_wait)

    @remote_function(enable_continous_work=True, sticky_key=_episode_key, lazy_collect=True)
    def stop(self, traj: Trajectory, episode_id: str | None = None) -> None:
        """Drop the episode: end its agent, close its env session, forget it."""
        self._discard(episode_id or _episode_id_of(traj))

    @remote_function(dispatch='slice', collect='flatten')
    def __call__(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Run every trajectory to its end; the results in the input order.

        An episode that raised :class:`EpisodeError` comes back as its input with
        ``stop_reason='generation_error'`` and the reason in ``error``. Under
        ``segment_policy='collect'`` an episode that compacted yields one row per
        collected segment, so the flattened output may hold more rows than inputs.
        """
        if isinstance(trajectories, dict):
            raise TypeError(f'{type(self).__name__}.__call__ expects a List[Trajectory]; '
                            'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        if not trajectories:
            return []

        def run(trajectory: Trajectory) -> list[Trajectory]:
            episode_id = _episode_id_of(trajectory) or uuid.uuid4().hex
            current = trajectory
            try:
                while True:
                    rows, sample = self._step(current, episode_id)
                    current = rows[0]
                    if sample.done:
                        return rows
            except EpisodeError as error:
                out = dict(trajectory)
                out['stop_reason'] = STOP_GENERATION_ERROR
                out['error'] = str(error)
                return [out]

        workers = min(len(trajectories), self.concurrency or len(trajectories))
        if workers == 1:
            return [row for trajectory in trajectories for row in run(trajectory)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return [row for rows in pool.map(run, trajectories) for row in rows]

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def close(self) -> None:
        """Drop every episode and take the endpoint down; the sampler and env are not owned here."""
        with self._lock:
            episode_ids = list(self._episodes)
        for episode_id in episode_ids:
            self._discard(episode_id)
        with self._endpoint_lock:
            endpoint, self._policy_endpoint = self._policy_endpoint, None
        if endpoint is not None:
            endpoint.stop()

    def stream(self, trajectory: Trajectory, *, sampling_params: SamplingParams | None = None):
        """Display-only streaming variant of ONE episode: yield token deltas live, then the transcript.

        Runs the tool loop at the MESSAGE level (a template-aware :class:`MessageLedger`, so
        tool-call markup in the reply still parses) and streams each generation's text as it
        arrives, for an interactive REPL. The result is a transcript, NOT a trainable trajectory.

        Local only: a generator does not cross the Ray boundary, so this needs an in-process
        sampler with ``sample_stream``. Termination is simpler than the tool loop's: the episode
        ends when the model stops calling tools, when a generation hits its length cap, or at
        ``max_turns``.

        Yields:
            :class:`StreamDelta` for each token delta of each turn's generation;
            :class:`StreamToolCall` after each turn that dispatched tools;
            :class:`StreamDone` once at the end, carrying the transcript trajectory.
        """
        if self.harness is not None or self.sampler is None or not hasattr(self.sampler, 'sample_stream'):
            raise ValueError('MultiTurnRollout.stream requires a local sampler exposing sample_stream and no harness.')
        params = sampling_params if sampling_params is not None else self.sampling_params
        episode_id = uuid.uuid4().hex
        tool_manager = self.tool_manager
        if self.env is not None:
            result = self.env.reset(episode_id, trajectory)
            trajectory = self._with_observation(trajectory, result)
            tool_manager = self._episode_tools(episode_id)
        try:
            yield from self._stream_episode(trajectory, params, tool_manager)
        finally:
            if self.env is not None:
                self.env.close(episode_id)

    # ------------------------------------------------------------------ episode

    def _step(self, traj: Trajectory, episode_id: str | None,
              session_wait: Optional[float] = None) -> tuple[list, RolloutStatus]:
        episode_id = episode_id or _episode_id_of(traj)
        if not episode_id:
            raise ValueError('step needs an episode id: pass episode_id= or set user_data.episode_id.')
        self._release_idle()
        with self._lock:
            ep = self._episodes.get(episode_id)
            opening = ep is None
            if opening:
                if 'turns' in traj:
                    # A view an earlier step returned: the episode it names is gone.
                    raise EpisodeError(f'episode {episode_id!r} is not held by this rollout (stopped, idle '
                                       'past episode_idle_timeout, or its replica restarted).')
                ep = _Episode(episode_id, dict(traj), self._episode_params(traj))
                self._episodes[episode_id] = ep
            elif 'turns' not in traj:
                raise ValueError(f'episode {episode_id!r} is already running; every episode needs its own id.')
            ep.busy = True
        _record_dispatch_probe('step_entry', episode_id, opening=opening)
        try:
            if opening:
                if self._open(ep, session_wait):
                    # No free session permit and we were asked not to wait: the episode is already
                    # discarded by _open. Report it along the normal return channel (not an exception,
                    # which the driver would raise) so the admission loop can retry this member later.
                    return [traj], RolloutStatus(episode_id=episode_id, no_slot=True)
            if self.harness is not None:
                return self._harness_step(ep)
            return self._tool_step(ep)
        except BaseException:
            self._discard(episode_id)
            raise
        finally:
            ep.busy = False
            ep.touched = time.monotonic()

    def _open(self, ep: _Episode, session_wait: Optional[float] = None) -> bool:
        """Open the episode's env session and first turn. Returns True only on a no-slot sentinel."""
        if self.env is not None:
            _record_dispatch_probe('reset_start', ep.episode_id)
            try:
                result = self.env.reset(ep.episode_id, ep.trajectory, wait=session_wait)
            except TimeoutError as exc:
                # A waiting reset that exhausts its slot deadline raises a bare TimeoutError. The batch
                # driver's __call__ catches only EpisodeError, so without this wrap a full pool crashes
                # the whole batch instead of recording one generation_error row. Discard the registration
                # (no session opened) and surface it as an episode failure, preserving the real cause.
                self._discard(ep.episode_id)
                raise EpisodeError(f'episode {ep.episode_id!r} could not open an env session: {exc}') from exc
            _record_dispatch_probe('reset_returned', ep.episode_id, discarded=ep.discarded)
            if isinstance(result, NoSlot):
                # The pool is full and this call was asked not to wait. Drop the registration we just
                # made (no session was opened, so there is nothing to close) and let _step report it.
                self._discard(ep.episode_id)
                return True
            if ep.discarded:
                # ``env.reset`` waits for a free session slot, and the episode can be discarded while it
                # waits -- an idle release, or another step's failure calling ``_discard`` for this id.
                # The session this call has just created then belongs to nobody: ``_discard`` already ran
                # and closed an env session that did not exist yet, and the episode is out of
                # ``_episodes``, so no later ``_finish`` will ever close it. Close it here instead. A
                # permit that is not returned is not reused, so leaking one per discarded opening
                # exhausts the pool even when the pool is larger than the admission bound.
                try:
                    self.env.close(ep.episode_id)
                except Exception:  # noqa
                    logger.warning(f'closing the env session opened for discarded episode '
                                   f'{ep.episode_id!r} failed.', exc_info=True)
                raise EpisodeError(f'episode {ep.episode_id!r} was discarded while its env session was '
                                   'waiting to open.')
            ep.env_open = True
            ep.trajectory = self._with_observation(ep.trajectory, result)
        if self.harness is not None:
            self._start_agent(ep)
            _record_dispatch_probe('agent_started', ep.episode_id)
            return False
        ep.tool_manager = self._episode_tools(ep.episode_id)
        ep.ledger = self._new_ledger(ep.episode_id)
        # A trajectory that named no tools advertises the manager's, so the prompt
        # lists what can actually be dispatched.
        opening_tools = None
        if 'tools' not in ep.trajectory and ep.tool_manager is not None:
            opening_tools = list(ep.tool_manager.tool_infos() or [])
        ep.ledger.open(ep.trajectory, tools=opening_tools)
        ep.budget = self.max_turns
        return False

    def _finish(self, ep: _Episode, **fields: Any) -> tuple[list, RolloutStatus]:
        """Merge the trajectory, run ``on_episode_end``, close the session, release the episode.

        Under ``segment_policy='collect'`` the rows the compaction boundaries froze join the
        final segment's row, in the order they happened: one episode, several independently-
        coded training rows sharing one identity and one outcome.
        """
        traj = ep.ledger.merge(
            ep.trajectory, turns=ep.ledger.turns, stop_reason=ep.stop_reason, truncated=ep.truncated,
            segment_index=len(ep.closed_rows), **fields)
        if ep.error is not None:
            traj['error'] = ep.error
        episode_end = None
        if self.on_episode_end is not None:
            episode_end = self.on_episode_end(traj, self.env, ep.episode_id)
        # Terminal checks run inside the episode's env session, BEFORE ``close`` ends it: they
        # must see exactly the state the agent left behind (the session -- and for LocalEnv its
        # workspace -- dies with close). The score lands as the rollout's own ``rewards.check``
        # key beside ``rewards.env``; a row that declares no checks carries no key. Checks that
        # need to run something (shell / python) ride ``ctx.env``, so episodes run and score in
        # the same sandbox; ``env=None`` (a tools-only local episode) falls back to a LocalEnv
        # over the row's own ``workspace`` column.
        check_score: Optional[float] = None
        if traj.get('checks') is not None:
            if self.env is not None and not ep.env_open:
                raise EpisodeError(
                    f'episode {ep.episode_id!r}: its env session is already closed, so its terminal '
                    'checks cannot see the state the agent left behind.')
            try:
                checks = checks_from_dicts(traj.get('checks'))
            except Exception as exc:
                raise EpisodeError(
                    f'episode {ep.episode_id!r}: malformed checks column: {exc}') from exc
            workspace = (self.env.workspace(ep.episode_id) if self.env is not None
                         else str(traj.get('workspace') or ''))
            report = run_checks(
                checks,
                CheckContext(workspace=workspace, final_answer=_final_answer(traj),
                             env=self.env, episode_id=ep.episode_id))
            check_score = float(report.score)
        env_reward: Optional[float] = None
        if self.env is not None and ep.env_open:
            env_reward = self.env.close(ep.episode_id)
            ep.env_open = False
        self._discard(ep.episode_id)
        # ``rewards.env`` / ``rewards.check`` are the reward keys the rollout itself owns; the
        # scorer columns (orm / baseline) are the driver's to name when it hands the batch out.
        # ``rewards.env`` lands only when the env session actually scored (``close`` returns None
        # for a sandbox used only for its tools: its constant zero is "no reward", not a score),
        # so an absent env key never fabricates one.
        rewards = dict(traj.get('rewards') or {})
        if env_reward is not None:
            rewards['env'] = float(env_reward)
        if check_score is not None:
            rewards['check'] = check_score
        traj['rewards'] = rewards
        if episode_end is not None:
            traj['episode_end'] = episode_end
        for row in ep.closed_rows:
            # The frozen rows carry the episode's env reward too: the reward keys belong to
            # the episode, and every row of it carries the same ones.
            row_rewards = dict(row.get('rewards') or {})
            if env_reward is not None:
                row_rewards['env'] = float(env_reward)
            if check_score is not None:
                row_rewards['check'] = check_score
            row['rewards'] = row_rewards
        rows = list(ep.closed_rows) + [traj]
        _dump_rollout_rows(ep.episode_id, rows)
        extra = {
            key: traj[key]
            for key in ('stuck_stop', 'tool_stop', 'followups', 'segments', 'agent_exit_code', 'error') if key in traj
        }
        if isinstance(episode_end, dict):
            extra.update(episode_end)
        elif episode_end is not None:
            extra['episode_end'] = episode_end
        prompt_tokens = response_tokens = 0
        coded = False
        row_lengths: list = []
        for row in rows:
            input_ids = row.get('input_ids') or []
            response = row.get('response_token_ids') or []
            coded = coded or bool(input_ids)
            prompt_tokens += len(input_ids) - len(response)
            response_tokens += len(response)
            row_lengths.append((len(input_ids) - len(response), len(response)))
        if coded:
            # The per-row lengths ride the status so the driver can hand each delivered row its
            # own facts: the episode totals would repeat on every row and weigh the batch's
            # metrics by its row count.
            extra['row_token_lengths'] = row_lengths
        sample = RolloutStatus(
            episode_id=ep.episode_id,
            # The episode's turns, not the last segment's: every row carries its own segment's
            # banked turns, and the episode is the unit that ran.
            turns=sum(int(row.get('turns') or 0) for row in rows),
            done=True,
            stop_reason=ep.stop_reason,
            truncated=bool(ep.truncated),
            token_lengths=(prompt_tokens, response_tokens) if coded else None,
            routing_exported=traj.get('routed_experts') is not None,
            env_reward=float(env_reward) if env_reward is not None else 0.0,
            extra=extra,
        )
        return rows, sample

    def _continue(self, ep: _Episode) -> tuple[list, RolloutStatus]:
        """The episode goes on: return its message view as the one row of an unfinished step."""
        return [self._view(ep)], RolloutStatus(
            episode_id=ep.episode_id, turns=ep.ledger.turns, stop_reason=ep.stop_reason)

    def _view(self, ep: _Episode) -> Trajectory:
        view = dict(ep.trajectory)
        if ep.ledger is not None:
            view['messages'] = ep.ledger.messages
            view['tools'] = ep.ledger.tools
        view['turns'] = ep.ledger.turns if ep.ledger is not None else 0
        view['followups'] = ep.followups
        return view

    def _discard(self, episode_id: Optional[str]) -> None:
        """Forget the episode: fail its queued requests, end its agent and env session."""
        if not episode_id:
            return
        with self._lock:
            ep = self._episodes.pop(episode_id, None)
        if ep is None:
            return
        # Set before anything that can block: ``_open`` may be inside ``env.reset`` at this moment and has
        # to see the flag the instant that call returns.
        ep.discarded = True
        if ep.requests is not None and self._policy_endpoint is not None:
            self._policy_endpoint.close(episode_id)
        if self.env is not None:
            ep.env_open = False
            try:
                # Kills whatever the session still runs, the agent program included. Not gated on
                # ``env_open``: that flag is only set once ``env.reset`` returns, so gating it skipped the
                # close for exactly the episodes discarded while their reset was still waiting for a slot.
                # ``Env.close`` is idempotent -- no session under this id, no effect -- so an unconditional
                # close costs nothing and covers the window the flag cannot.
                self.env.close(episode_id)
            except Exception:  # noqa
                logger.warning(f'closing the env session of episode {episode_id!r} failed.', exc_info=True)

    def _release_idle(self) -> None:
        now = time.monotonic()
        with self._lock:
            idle = [
                episode_id for episode_id, ep in self._episodes.items()
                if not ep.busy and now - ep.touched > self.episode_idle_timeout
            ]
        for episode_id in idle:
            logger.warning(f'releasing episode {episode_id!r}: no step for over {self.episode_idle_timeout:.0f}s.')
            self._discard(episode_id)

    # ------------------------------------------------------------------ helpers

    def _params(self, params: SamplingParams) -> SamplingParams:
        """One episode's per-generation params: one sample, and old_logps whenever there are tokens."""
        if params.num_samples != 1:
            # n>1 would fork the conversation at turn 1, and a forked trajectory has no
            # defined continuation: run the trajectory several times instead.
            raise ValueError(f'MultiTurnRollout supports num_samples=1 only, got {params.num_samples}')
        if self.template is not None and not params.logprobs:
            params = replace(params, logprobs=1)
        return params

    def _episode_params(self, traj: Trajectory) -> SamplingParams:
        override = user_data_get(traj.get('user_data'), 'sampling_params')
        params = (self.sampling_params if not override
                  else self._params(replace(self.sampling_params, **dict(override))))
        # defect52b: --seed reaches the rollout half too. Derive a per-request seed from the episode's
        # stable (group_key, sample_index) identity so a group's num_generations samples differ -- a
        # constant seed would collapse them to one identical trajectory and zero the group's advantage.
        # An explicit user_data seed override wins; an episode without a stable identity (the display
        # stream, a bare __call__) stays unseeded.
        if self._base_seed is not None and not (override and 'seed' in dict(override)):
            derived = self._derive_seed(traj)
            if derived is not None:
                params = replace(params, seed=derived)
        return params

    def _derive_seed(self, traj: Trajectory) -> int | None:
        """A per-(group, sample) seed: crc32 of the identity disperses without linear collisions."""
        import zlib
        group_key = traj.get('group_key')
        sample_index = traj.get('sample_index')
        if group_key is None or sample_index is None:
            return None
        try:
            identity = f'{int(group_key)}:{int(sample_index)}'.encode()
        except (TypeError, ValueError):
            return None
        return (self._base_seed + zlib.crc32(identity)) & 0x7FFFFFFF

    def _new_ledger(self, label: str):
        """The episode account: token-level with a template, message-level without one."""
        if self.template is None:
            return MessageLedger(label=label, max_tokens=self.max_trajectory_tokens)
        return TurnLedger(
            self.template, label=label, max_tokens=self.max_trajectory_tokens, backend=self._template_backend)

    @staticmethod
    def _with_observation(trajectory: Trajectory, result: Any) -> Trajectory:
        """The opening, with the env's first observation appended as a user message."""
        observation = getattr(result, 'observation', None)
        if not observation:
            return trajectory
        trajectory = dict(trajectory)
        trajectory['messages'] = list(trajectory.get('messages') or []) + [{'role': 'user', 'content': observation}]
        return trajectory

    def _episode_tools(self, episode_id: str) -> Optional[ToolManager]:
        """The shared tools plus, with an env, the env's tools bound to this episode's session."""
        manager = self.tool_manager.copy() if self.tool_manager is not None else None
        if self.env is None:
            return manager
        infos = self._env_tool_infos
        if not infos:
            # Read after the first reset: an env may only know its tools once a session is up.
            infos = self._env_tool_infos = list(self.env.tools() or [])
        if not infos:
            return manager
        manager = manager if manager is not None else ToolManager()
        for tool in EnvTool.from_env(self.env, episode_id, infos):
            manager.register(tool)
        return manager

    def _generate(self, ep: _Episode, ledger, params: SamplingParams) -> SampledSequence:
        """One generation for the episode, from the sampler or the ``response_callback``."""
        if self.response_callback is not None:
            state = ledger.state if ledger.state is not None else ledger.input_feature
            seq = self.response_callback(self._view(ep), state, params)
        else:
            use_state = ledger.state is not None and getattr(type(self.sampler), 'accepts_rollout_state', False)
            responses = self.sampler.sample([ledger.state if use_state else ledger.input_feature],
                                            sampling_params=params,
                                            session_id=ep.episode_id,
                                            **self._adapter_kwargs)
            seq = _single_sequence(responses)
        if not isinstance(seq, SampledSequence):
            raise TypeError(f'expected SampledSequence, got {type(seq).__name__}')
        return seq

    @staticmethod
    def _run_tools(tool_manager: ToolManager, tool_calls: list[dict[str, Any]]) -> list[str]:
        """Run one turn's calls together (one Env round trip per session).

        A manager that answers with fewer results than calls leaves the rest
        empty rather than shifting them onto the wrong call.
        """
        if hasattr(tool_manager, 'call_many'):
            contents = tool_manager.call_many(tool_calls)
        else:
            contents = [tool_manager(tc) for tc in tool_calls]
        obs = [''] * len(tool_calls)
        for i, content in enumerate(contents[:len(tool_calls)]):
            obs[i] = '' if content is None else str(content)
        return obs

    # ---------------------------------------------------------------- tool loop

    def _tool_step(self, ep: _Episode) -> tuple[list, RolloutStatus]:
        """One generation and, when it called tools, their round."""
        ledger = ep.ledger
        ep.spent += 1

        # The turn's completion budget must still fit under the template's max_length.
        # With turns already banked, ending here is a truncated episode keeping them. On
        # the FIRST turn nothing is banked, so cap the budget to the room left instead
        # (a truncated partial response); not even one token fitting means there is no
        # sample to make.
        params = ep.params
        if ledger.unfit(params):
            if ledger.turns:
                ep.truncated = True
                return self._finish_tool_episode(ep)
            params = ledger.capped_turn_params(params)
            if params is None:
                raise EpisodeError(f'episode {ep.episode_id!r}: the prompt leaves no room for a generated token.')

        try:
            seq = self._generate(ep, ledger, params)
        except APIGenerationError as error:
            if not ledger.turns:
                raise EpisodeError(f'episode {ep.episode_id!r}: the first generation failed: {error}') from error
            ep.stop_reason = STOP_GENERATION_ERROR
            ep.error = str(error)
            ep.truncated = True
            return self._finish_tool_episode(ep)

        ledger.record(seq)
        ep.stop_reason = seq.stop_reason

        msgs = ledger.messages
        last_msg = msgs[-1] if msgs else None
        tool_calls = last_msg.get('tool_calls') if isinstance(last_msg, dict) else None
        if not tool_calls:
            tool_calls = ledger.parse_tool_call(seq.decoded or '')
        if ep.followups:
            # Tools are withdrawn once a follow-up was appended: those stages write text
            # about the state as it is, and a code block in a reply parses as a call.
            tool_calls = None
            if isinstance(last_msg, dict):
                message = dict(last_msg)
                message['content'] = ledger.decode_tokens(seq)
                message.pop('tool_calls', None)
                ledger.replace_last_message(message)

        # A reply cut off at max_tokens is truncated in exactly the sense the flag names;
        # tool calls it happens to contain are not dispatched.
        if seq.stop_reason == 'length' or ledger.full():
            ep.truncated = True
            return self._finish_tool_episode(ep)

        if not tool_calls:
            # Markup that did not parse is the model asking for a tool, not declining
            # one: hand back the parser's reason and let it write the call again. Not
            # after a follow-up, where a reply that looks like a call is text.
            parse_errors = [] if ep.followups else ledger.tool_call_errors(seq.decoded or '')
            if parse_errors:
                if ep.malformed >= self.max_malformed_retries:
                    raise EpisodeError(f'episode {ep.episode_id!r}: tool-call markup still malformed after '
                                       f'{ep.malformed} retries: {"; ".join(parse_errors)}')
                ep.malformed += 1
                if not ledger.observe([_malformed_tool_message(parse_errors)]):
                    ep.truncated = True
                    return self._finish_tool_episode(ep)
                return self._next_turn(ep)
            if self._grant_followup(ep):
                return self._next_turn(ep)
            return self._finish_tool_episode(ep)

        if ledger.turns >= self.max_turns:
            # Out of tool turns, not out of episode: the stages that read the end
            # state can still run on what was built.
            ep.tool_stop = 'max_turns'
            if self._grant_followup(ep):
                return self._next_turn(ep)
            ep.truncated = True
            return self._finish_tool_episode(ep)

        ep.malformed = 0
        if ep.tool_manager is None:
            raise ValueError('the model emitted tool_calls but this rollout has no tools (tool_manager= or env=).')
        observations = self._run_tools(ep.tool_manager, tool_calls)
        if self.stop_after_stuck_turns:
            keys = [_call_key(tc) for tc in tool_calls]
            all_repeats = bool(keys) and all(k in ep.seen_calls for k in keys)
            ep.seen_calls.update(keys)
            all_errors = bool(observations) and all(is_error_observation(o) for o in observations)
            ep.stuck = ep.stuck + 1 if all_errors or all_repeats else 0

        overflowed = not ledger.observe(_default_tool_messages(tool_calls, observations))
        if overflowed:
            ep.truncated = True
        # Checked after the messages are appended, so the turns that ended the episode
        # are in the trajectory.
        if self.stop_after_stuck_turns and ep.stuck >= self.stop_after_stuck_turns:
            ep.stuck_stop = True
            ep.tool_stop = 'stuck'
            if not overflowed and self._grant_followup(ep):
                return self._next_turn(ep)
            ep.truncated = True
            return self._finish_tool_episode(ep)
        if overflowed:
            return self._finish_tool_episode(ep)
        return self._next_turn(ep)

    def _next_turn(self, ep: _Episode) -> tuple[list, RolloutStatus]:
        """Another generation is due -- unless the budget is spent, which ends the episode here."""
        if ep.spent >= ep.budget:
            return self._finish_tool_episode(ep)
        return self._continue(ep)

    def _finish_tool_episode(self, ep: _Episode) -> tuple[list, RolloutStatus]:
        # ``truncated`` says something was cut off; these say what ended the tool-calling
        # part -- an episode can run out of turns, be handed a follow-up, and finish it.
        return self._finish(ep, stuck_stop=ep.stuck_stop, tool_stop=ep.tool_stop, followups=ep.followups)

    def _grant_followup(self, ep: _Episode) -> bool:
        """Ask for one more stage; True when the episode carries on.

        Each granted follow-up buys the one extra generation it asked for, so an episode
        that spent its whole tool budget still reaches the stages that read what it built.
        """
        if self.followup_fn is None or ep.followups >= MAX_FOLLOWUPS:
            return False
        ledger = ep.ledger
        followup = self.followup_fn(
            ledger.merge(ep.trajectory, turns=ledger.turns, stop_reason=ep.stop_reason, truncated=ep.truncated),
            ep.followups)
        if followup is None:
            return False
        text, next_params = followup if isinstance(followup, tuple) else (followup, None)
        if not ledger.observe([{'role': 'user', 'content': text}]):
            ep.truncated = True
            return False
        # Follow-up stages are answers, so a backend must not see tool schemas.
        ledger.clear_tools()
        ep.followups += 1
        ep.budget += 1
        if next_params is not None:
            ep.params = self._params(next_params)
        return True

    # ------------------------------------------------------------ agent program

    def _endpoint(self) -> PolicyEndpoint:
        # Under a lock: two episodes finding the endpoint down at once would race to bind it.
        with self._endpoint_lock:
            if self._policy_endpoint is None:
                endpoint = ENDPOINTS[self.harness.protocol]()
                endpoint.start()
                self._policy_endpoint = endpoint
            return self._policy_endpoint

    def _start_agent(self, ep: _Episode) -> None:
        """Start the program in the episode's session; its requests arrive on ``ep.requests``."""
        endpoint = self._endpoint()
        # The API key IS the episode id: it is how a request finds its episode.
        ep.requests = endpoint.open(ep.episode_id)
        command = self.harness.command(
            task=self.harness.task(ep.trajectory),
            base_url=endpoint.base_url,
            api_key=ep.episode_id,
            workspace=self.env.workspace(ep.episode_id) or '',
        )
        ep.deadline = time.monotonic() + self.harness.timeout + _AGENT_GRACE
        requests = ep.requests

        def run() -> None:
            try:
                # The agent program is a full third-party runtime, not a model-written tool
                # script: the tool-script address-space cap would strangle it before its first
                # request (defect108). Its containment is the harness timeout passed here.
                ep.exit = self.env.run_script(ep.episode_id, command, interpreter='shell',
                                              timeout=self.harness.timeout, cap_memory=False)
            except BaseException as error:  # noqa # reported by the step that reads the exit
                ep.exit = (None, f'{type(error).__name__}: {error}')
            finally:
                requests.put(_AGENT_EXITED)

        threading.Thread(target=run, name=f'agent-{ep.episode_id[:8]}', daemon=True).start()

    def _harness_step(self, ep: _Episode) -> tuple[list, RolloutStatus]:
        """Answer the agent's next request, or end the episode when the program exited."""
        wait = max(ep.deadline - time.monotonic(), 0.0)
        _record_dispatch_probe('harness_wait', ep.episode_id, wait=wait)
        try:
            item = ep.requests.get(timeout=wait)
        except queue.Empty:
            _record_dispatch_probe('harness_timeout', ep.episode_id)
            raise EpisodeError(f'episode {ep.episode_id!r}: the agent did not finish within '
                               f'{self.harness.timeout:.0f}s.') from None
        _record_dispatch_probe('harness_item', ep.episode_id, exited=item is _AGENT_EXITED)
        if item is _AGENT_EXITED:
            return self._agent_exited(ep)
        try:
            result = self._serve(ep, item)
            _record_dispatch_probe('harness_served', ep.episode_id)
            return result
        except BaseException as error:
            _record_dispatch_probe('harness_serve_error', ep.episode_id,
                                   error=f'{type(error).__name__}: {error}')
            if not item.reply.done():
                item.reply.set_exception(error)
            raise

    def _agent_exited(self, ep: _Episode) -> tuple[list, RolloutStatus]:
        code, output = ep.exit or (None, '')
        tail = (output or '')[-_OUTPUT_TAIL:]
        if ep.ledger is None:
            # Wrong config, a crash before the first call, or a config that overrode the key.
            raise EpisodeError(f'episode {ep.episode_id!r}: the agent exited ({code}) without calling the '
                               f'endpoint:\n{tail}')
        if code != 0:
            raise EpisodeError(f'episode {ep.episode_id!r}: the agent exited with {code}:\n{tail}')
        ep.stop_reason = STOP_NO_TOOL
        return self._finish(ep, segments=ep.segments, agent_exit_code=code)

    def _open_segment(self, ep: _Episode, request: EndpointRequest):
        """Start a fresh segment from what the agent sent: freeze the old one under collect.

        A history the agent rewrote (compacted, dropped a turn) or a tool change admits no
        append, so the recorded segment is over whatever the rewrite left it at.
        """
        if self.segment_policy == 'collect' and ep.ledger is not None:
            # The written-out segment freezes here: its account never changes again, so it
            # merges into its own row now (identity on it already, from admission); the
            # episode's other rows join it when the episode ends.
            ep.closed_rows.append(ep.ledger.merge(
                ep.trajectory, turns=ep.ledger.turns, truncated=False,
                segment_index=len(ep.closed_rows)))
        ledger = self._new_ledger(f'{ep.episode_id}#{ep.segments}')
        ledger.open({'messages': list(request.messages)}, tools=list(request.tools))
        return ledger

    def _serve(self, ep: _Episode, request: EndpointRequest) -> tuple[list, RolloutStatus]:
        """Sample one reply to the agent's request, record it, and answer the request."""
        canonical = [_canonical(message) for message in request.messages]
        tools_key = json.dumps(request.tools, sort_keys=True, ensure_ascii=False)
        ledger = ep.ledger
        if (ep.last_reply is not None and tools_key == ep.tools_key and canonical
                and canonical == ep.sent[:-1]):
            # The same request a retry re-sends: answer with the reply it never saw, without
            # sampling, recording or starting a segment (the no-new-content history would
            # otherwise read as a rewrite and needlessly swap the ledger).
            request.reply.set_result(dict(ep.last_reply))
            return self._continue(ep)
        if (ledger is not None and ep.last_reply is not None
                and ep.last_reply.get('finish_reason') == 'length'
                and tools_key == ep.tools_key and canonical and canonical == ep.sent):
            # Continue-on-length (chat protocol): the agent echoed the conversation ending with
            # our truncated reply and asks for more of THAT turn. Detection is against the WIRE
            # view (ep.sent), not ledger.messages: the ledger banks the decoded text with the
            # template's end-of-turn marker, the agent echoes the marker-stripped wire content.
            # Generating from the ledger's current state extends the open assistant message;
            # opening a segment instead would re-encode an assistant-ending prompt and crash
            # (defect110).
            return self._continue_generation(ep, request, canonical, tools_key)
        merged_extends = False
        if ep.merged_content is not None and ledger is not None and tools_key == ep.tools_key and canonical:
            # A continued turn reaches the agent as several wire messages it merges into ONE
            # assistant message; reconstruct that merged view to recognise what comes back.
            merged_view = ep.sent[:ep.split_start] + [
                ('assistant', ep.merged_content.strip(), ep.sent[-1][2], '')]
            if (ep.last_reply is not None and ep.last_reply.get('finish_reason') == 'length'
                    and canonical == merged_view):
                # Still truncated after the merge: continue the same turn again.
                return self._continue_generation(ep, request, canonical, tools_key)
            if len(canonical) > len(merged_view) and canonical[:len(merged_view)] == merged_view:
                # The merged turn plus what the environment added (tool results, a new
                # question): an append onto the SAME ledger, so the banked assistant turn
                # keeps its sampled tokens instead of being dropped with a fresh segment.
                if not ledger.observe(request.messages[len(merged_view):]):
                    return self._end_agent_truncated(ep, request)
                ep.merged_content = None
                merged_extends = True
        noise = None
        if self.harness is not None and not merged_extends:
            noise = self.harness.noise_reply(request.messages, request.tools, request.params)
        extends = merged_extends or (
            ledger is not None and tools_key == ep.tools_key and len(canonical) > len(ep.sent)
            and canonical[:len(ep.sent)] == ep.sent)
        if noise is not None:
            # Housekeeping traffic the agent program does not expect a generation for: the
            # harness's reply answers it verbatim, costs no sampling, and rides the account
            # masked (the pair keeps the conversation prefix aligned, the reply is not trained).
            message = noise.get('message') or {'role': 'assistant', 'content': ''}
            if extends:
                if not ledger.observe(request.messages[len(ep.sent):] + [message]):
                    return self._end_agent_truncated(ep, request)
            else:
                # A compaction or tool change the harness's own filter caught: swap the segment
                # exactly as a generation request would and record the reply onto the fresh
                # segment, so the account never drifts from the agent's real history.
                ledger = self._open_segment(ep, request)
                ep.ledger = ledger
                ep.segments += 1
                if not ledger.observe([message]):
                    return self._end_agent_truncated(ep, request)
            ep.sent = canonical + [_canonical(message)]
            ep.tools_key = tools_key
            ep.last_reply = noise
            request.reply.set_result(dict(noise))
            return self._continue(ep)
        if extends:
            # Token-in: what the agent added since our last reply (tool results, its own
            # notes) is the environment's, appended masked. A merged continuation already
            # observed its suffix above.
            if not merged_extends and not ledger.observe(request.messages[len(ep.sent):]):
                return self._end_agent_truncated(ep, request)
        else:
            # The first request, or a history the agent rewrote (compacted, dropped a
            # turn): no append expresses it, so a fresh segment starts from what it sent.
            ledger = self._open_segment(ep, request)

        # Same length rule as the tool loop: a segment with turns banked ends truncated, a
        # fresh one gets its first budget capped to the room left.
        params = self._request_params(ep, request.params)
        if ledger.unfit(params):
            capped = None if ledger.turns else ledger.capped_turn_params(params)
            if capped is None:
                if ep.ledger is None:
                    raise EpisodeError(f'episode {ep.episode_id!r}: the prompt leaves no room for a generated token.')
                # Ends on the last recorded segment (a rewritten history with no room never became one).
                return self._end_agent_truncated(ep, request)
            params = capped

        prompt_tokens = len(ledger.input_feature.get('input_ids') or [])
        try:
            seq = self._generate(ep, ledger, params)
        except APIGenerationError as error:
            raise EpisodeError(f'episode {ep.episode_id!r}: generation failed: {error}') from error
        ledger.record(seq)
        if ledger is not ep.ledger:
            ep.ledger = ledger
            ep.segments += 1
        ep.stop_reason = seq.stop_reason

        # Decoded off the sampled ids rather than taken from ``seq.decoded``, which keeps
        # the template's end-of-turn marker -- a marker the agent would write into its files.
        text = ledger.decode_tokens(seq)
        parsed = ledger.parse_tool_call(text)
        message = assistant_message(text, parsed, self.template.clean_tool_call(text) if parsed else text)
        ep.sent = canonical + [_canonical(message)]
        ep.tools_key = tools_key
        ep.last_reply = {
            'message': message,
            'finish_reason': 'tool_calls' if parsed else ('length' if seq.stop_reason == 'length' else 'stop'),
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': len(seq.tokens or []),
            },
        }
        request.reply.set_result(dict(ep.last_reply))
        if ledger.full():
            ep.truncated = True
            return self._finish(ep, segments=ep.segments)
        return self._continue(ep)

    def _continue_generation(self, ep: _Episode, request: EndpointRequest, canonical: list,
                             tools_key: str) -> tuple[list, RolloutStatus]:
        """Extend the open assistant turn: the agent asked for more of a length-truncated reply.

        The ledger already holds the conversation including the truncated reply, so the sampler
        generates straight from its current state -- no re-encode (an assistant-ending prompt
        refuses ``add_generation_prompt=True``), no new segment, no new turn. The reply carries
        ONLY the continuation: chat-protocol agents concatenate it onto their partial message.
        Tool calls are parsed on the MERGED wire text, since a call may complete across the
        boundary; the merged content is tracked on the episode so the agent's folded view of
        the split replies stays recognisable (defect110).
        """
        ledger = ep.ledger
        params = self._request_params(ep, request.params)
        if ledger.unfit(params):
            # The turn is already banked: no room ends the episode truncated, as in _serve.
            return self._end_agent_truncated(ep, request)
        prompt_tokens = len(ledger.input_feature.get('input_ids') or [])
        prev_wire = ep.last_reply['message']
        # Parse on the ledger's RAW contiguous text: the model generated it as one token
        # stream, and a call completing across the boundary must read as it was sampled.
        prev_raw = ledger.messages[-1].get('content') or ''
        try:
            seq = self._generate(ep, ledger, params)
        except APIGenerationError as error:
            raise EpisodeError(f'episode {ep.episode_id!r}: generation failed: {error}') from error
        ledger.record_continuation(seq)
        ep.stop_reason = seq.stop_reason
        text = ledger.decode_tokens(seq)
        parsed = ledger.parse_tool_call(prev_raw + text)
        message = assistant_message(text, parsed, self.template.clean_tool_call(text) if parsed else text)
        if ep.merged_content is None:
            ep.split_start = len(ep.sent) - 1
            ep.merged_content = (prev_wire.get('content') or '').strip()
        # The agent's merged view is NOT the raw concatenation: ms-agent strips each stored
        # message at send time (its _format_input_message mutates content.strip()), so the
        # boundary whitespace of earlier pieces is gone before the next piece appends.
        # Track its exact recurrence echo_k = strip(echo_{k-1} + piece_k), or the second
        # continuation's merged echo reads as a rewrite and re-encodes assistant-ending
        # (the attempt4 crash, defect110 follow-up).
        ep.merged_content = (ep.merged_content + (message.get('content') or '')).strip()
        ep.sent = canonical + [_canonical(message)]
        ep.tools_key = tools_key
        ep.last_reply = {
            'message': message,
            'finish_reason': 'tool_calls' if parsed else ('length' if seq.stop_reason == 'length' else 'stop'),
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': len(seq.tokens or []),
            },
        }
        request.reply.set_result(dict(ep.last_reply))
        if ledger.full():
            ep.truncated = True
            return self._finish(ep, segments=ep.segments)
        return self._continue(ep)

    def _end_agent_truncated(self, ep: _Episode, request: EndpointRequest) -> tuple[list, RolloutStatus]:
        """No room for the request: end on what is recorded (closing the session ends the agent)."""
        request.reply.set_exception(RuntimeError(f'episode {ep.episode_id!r} reached its length limit'))
        ep.truncated = True
        return self._finish(ep, segments=ep.segments)

    def _request_params(self, ep: _Episode, overrides: dict[str, Any]) -> SamplingParams:
        """The agent may adjust decoding, but not what is trainable (logprobs stay forced)."""
        if not overrides:
            return ep.params
        return self._params(replace(ep.params, **overrides))

    # ------------------------------------------------------------------ stream

    def _stream_episode(self, trajectory: Trajectory, params: SamplingParams, tool_manager: ToolManager | None):
        """The streaming loop proper; the env session above is held for all of it."""
        # Message-level accounting, but template-aware so a reply's tool-call markup still
        # parses. The feature stays a {'messages','tools'} dict with no input_ids, so
        # sample_stream re-encodes the whole trajectory each turn.
        ledger = MessageLedger(label='stream', max_tokens=self.max_trajectory_tokens, template=self.template)
        opening_tools = None
        if 'tools' not in trajectory and tool_manager is not None:
            opening_tools = list(tool_manager.tool_infos() or [])
        ledger.open(trajectory, tools=opening_tools)

        stop_reason: str | None = None
        truncated = False
        turns = 0
        while turns < self.max_turns:
            turns += 1
            pieces: list[str] = []
            finish: str | None = None
            for delta, finish_reason in self.sampler.sample_stream(
                    ledger.input_feature, params, adapter_path=self.adapter_path):
                if delta:
                    pieces.append(delta)
                    yield StreamDelta(delta)
                if finish_reason is not None:
                    finish = finish_reason
            text = ''.join(pieces)
            stop_reason = finish or STOP_NO_TOOL
            ledger.record(SampledSequence(stop_reason=stop_reason, tokens=[], decoded=text))
            if finish == 'length':
                # Cut off mid-reply: no tool call it was still emitting gets dispatched.
                truncated = True
                break
            tool_calls = ledger.parse_tool_call(text)
            if not tool_calls:
                break
            if turns >= self.max_turns:
                # Out of turns: there is no further generation to read the results.
                truncated = True
                break
            if tool_manager is None:
                raise ValueError('the model emitted tool_calls but stream() has no tools (tool_manager= or env=).')
            observations = self._run_tools(tool_manager, tool_calls)
            yield StreamToolCall(names=_tool_call_names(tool_calls), observations=observations)
            ledger.observe(_default_tool_messages(tool_calls, observations))
        out = ledger.merge(trajectory, turns=ledger.turns, stop_reason=stop_reason, truncated=truncated)
        yield StreamDone(out)


__all__ = [
    'EpisodeError', 'MAX_FOLLOWUPS', 'MultiTurnRollout', 'RolloutStatus', 'STOP_GENERATION_ERROR', 'STOP_LENGTH',
    'STOP_MAX_TURNS', 'STOP_NO_TOOL', 'StreamDelta', 'StreamDone', 'StreamToolCall', 'is_error_observation'
]
