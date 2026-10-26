# Copyright (c) ModelScope Contributors. All rights reserved.
import contextlib
import copy
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional

import json

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams
from twinkle.infra import remote_class, remote_function
from twinkle.template.base import Template
from twinkle_agentic.protocol.api_sampler import APIGenerationError, APISampler
from twinkle_agentic.protocol.base import API
from twinkle_agentic.tools.tool_manager import ToolManager
from .base import MAX_FOLLOWUPS, STOP_GENERATION_ERROR, STOP_NO_TOOL, Rollout, SampleHandle
from .ledger import MessageLedger, TurnLedger, last_assistant_text
from .trace import TraceWriter

ResponseCallback = Callable[..., SampledSequence]


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
class RolloutSample:
    """One on-policy rollout trajectory, pre-collation (the RL training-sample layer).

    ``encoded`` carries the full prompt+completion feature with the ledger's already-shifted
    labels; ``rollout_logprobs`` are the sampling-policy old_logps, one per response token.
    ``response_loss_mask`` mirrors ``response_token_ids``.
    """

    # Defaults let a consumer synthesise a sample-shaped row (a scored ground truth, a
    # cache-loaded candidate) without faking engine state; the engine factory always fills them.
    encoded: dict = field(default_factory=dict)
    response_token_ids: list[int] = field(default_factory=list)
    rollout_logprobs: list[float] = field(default_factory=list)
    prompt_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    decoded: str = ""
    truncated: bool = False
    response_loss_mask: list[int] = field(default_factory=list)
    messages: list[dict] | None = None
    rollout_infos: dict[str, Any] = field(default_factory=dict)
    # The sampler's per-token sampling support set (twinkle SamplingMask); assembled into the
    # top-level sampling_masks forward kwarg by the consumer, never into encoded.
    sampling_mask: Any | None = None
    # The policy version this trajectory was generated under (the admission pin); 0 when the
    # producing path does not track versions.
    policy_version: int = 0
    # How many publishes the policy advanced between admission and collection; in_place only
    # (adapter_snapshot pins one frozen path, so the span is always 0).
    version_span: int = 0
    # The ORIGINAL prompt messages, stamped on the streaming collect path for PRM scoring and
    # the teacher privileged view; None on the blocking generate path.
    prompt: list[dict] | None = None
    # Per-sample outcome-reward row (one value per scorer), scored by the streaming layer before
    # the sample enters the ready buffer; None until then.
    orm_reward: list[float] | None = None
    # Per-token process reward in the response-token frame, scored by the same seam; None when
    # no PRM channel is wired.
    prm_process: list[float] | None = None
    # The kernel's group identity (the admission expansion key), stamped when the trajectory enters
    # the ready buffer; positions are the training contract, this is provenance/debugging.
    group_key: str | None = None
    # A group-level baseline (e.g. ReMax's greedy reward), written by the group-completion hook;
    # one value shared by every sibling of the group.
    baseline: float | None = None

    @property
    def input_feature(self) -> dict:
        """Back-compat alias: the training feature is ``encoded`` (was ``input_feature``)."""
        return self.encoded

    @property
    def old_logps(self) -> list[float]:
        """The sampling-policy logprobs, already flat (one per response token)."""
        return list(self.rollout_logprobs)

    @property
    def rollout_logps(self) -> list[float]:
        return list(self.rollout_logprobs)

    @property
    def prompt_index(self) -> str:
        """Back-compat alias for the group id."""
        return self.prompt_id


def _build_sampling_params(num_samples: int, sampling_params: dict | None, force_logprobs: bool) -> SamplingParams:
    """Assemble the twinkle SamplingParams shared by every rollout entry point.

    One contract (temperature default, forced old_logps, per-submission num_samples) so no
    path can silently sample under different params than another; the engine drives one
    trajectory per turn, hence num_samples is always 1 here.
    """
    sp = dict(sampling_params or {})
    sp.setdefault("temperature", 1.0)
    if force_logprobs:
        # Contract 15: these logprobs ARE old_logps, so requesting them is forced, not defaulted.
        sp["logprobs"] = max(int(sp.get("logprobs") or 0), 1)
    sp["num_samples"] = num_samples
    return SamplingParams(**sp)


#: Prompt-row media columns threaded onto the Trajectory so the sampler encodes them.
_MULTIMODAL_PROMPT_KEYS = ("images", "videos", "audios", "objects", "tools")

#: Keys of the flat trajectory a training forward consumes: the encoded feature proper
#: plus the MoE replay pair the sampler reported.
_TRAIN_FEATURE_KEYS = ("input_ids", "labels", "completion_mask", "routed_experts", "replay_mask")


def _vision_feature_keys() -> frozenset:
    """The multimodal feature keys to lift from the engine's feature into ``encoded``."""
    from twinkle.processor.base import InputProcessor

    return frozenset(InputProcessor.VLM_CONCAT_FIELDS) | frozenset(("mm_token_type_ids", "second_per_grid_ts"))


def trajectory_to_rollout_sample(traj: dict[str, Any], prompt_id: str, extra: dict[str, Any]) -> RolloutSample:
    """Translate one ledger-merged Trajectory into a RolloutSample.

    The token fields arrive ready-made from the ledger's ``merge`` (labels already
    next-token shifted; ``response_token_ids`` / ``rollout_logprobs`` the flat training
    view); message-only mode (a text-only backend) has no token fields, so only
    ``messages`` / ``decoded`` / ``rollout_infos`` carry anything.
    """
    input_ids = list(traj.get("input_ids") or [])
    labels = list(traj.get("labels") or [])
    messages = copy.deepcopy(traj.get("messages"))
    encoded: dict[str, Any] = {}
    response_token_ids: list[int] = []
    rollout_logprobs: list[float] = []
    if input_ids and labels:
        keys = set(_TRAIN_FEATURE_KEYS) | _vision_feature_keys()
        encoded = {key: traj[key] for key in keys if key in traj}
        response_token_ids = list(traj.get("response_token_ids") or [])
        rollout_logprobs = list(traj.get("rollout_logprobs") or [])
    rollout_infos = {
        key: traj[key]
        for key in ("turns", "stop_reason", "truncated", "stuck_stop", "tool_stop", "followups")
        if key in traj
    }
    if traj.get("error") is not None:
        rollout_infos["error"] = traj["error"]
    return RolloutSample(
        encoded=encoded,
        response_token_ids=response_token_ids,
        rollout_logprobs=rollout_logprobs,
        response_loss_mask=[1] * len(response_token_ids),
        prompt_id=prompt_id,
        extra=extra,
        decoded=last_assistant_text(messages),
        truncated=bool(traj.get("truncated")),
        messages=messages,
        rollout_infos=rollout_infos,
        sampling_mask=traj.get("sampling_mask"),
    )


#: Ray runs a decorated actor's methods one at a time unless max_concurrency is raised; each
#: in-flight episode is a separate generate call. A PERFORMANCE ceiling only, never a second
#: admission cap: the sole admission truth is the driver's backpressure (buffer over handles).
_MAX_CONCURRENT_EPISODES = 1024


def default_response_callback(
    sampler, api, sampling_params, *, input_feature, adapter_kwargs, rollout_state=None, **kwargs
) -> SampledSequence:
    """Use the sampler when present, otherwise the API adapter. Public so a caller-side
    ``response_callback`` can adjust ``sampling_params`` and delegate here."""
    if sampler is None:
        if api is None:
            raise ValueError("response_callback was omitted, but no sampler or API was provided")
        return api(rollout_state or input_feature, sampling_params, **adapter_kwargs)
    use_state = rollout_state is not None and getattr(type(sampler), "accepts_rollout_state", False)
    sample_input = rollout_state if use_state else input_feature
    responses = sampler.sample([sample_input], sampling_params=sampling_params, **adapter_kwargs)
    if not isinstance(responses, list):
        raise TypeError(f"expected List[SampleResponse] from sampler.sample, got {type(responses).__name__}")
    if len(responses) != 1:
        raise RuntimeError(f"sampler returned {len(responses)} responses for a single request; expected exactly one.")
    response = responses[0]
    if not isinstance(response, SampleResponse):
        raise TypeError(f"expected SampleResponse from sampler.sample, got {type(response).__name__}")
    if len(response.sequences) != 1:
        raise RuntimeError(f"SampleResponse contains {len(response.sequences)} sequences; expected exactly one.")
    sequence = response.sequences[0]
    if not isinstance(sequence, SampledSequence):
        raise TypeError(f"expected SampledSequence, got {type(sequence).__name__}")
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
    text = (observation or "").strip()
    if not text:
        return False
    if text.startswith("Error:"):
        return True
    if text.startswith(("Tool runtime unreachable:", "Tool runtime returned no result")):
        return True
    return bool(re.search(r'"success"\s*:\s*false', text))


def _call_key(tool_call: dict[str, Any]) -> str:
    """A stable identity for a tool call: its name plus its arguments verbatim.

    Byte-identical is the point. A model that changes one path and tries again is
    making progress; one that reissues the same call with the same arguments is
    not, whatever the tool answered.
    """
    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = fn.get("name") or tool_call.get("name") or tool_call.get("tool_name") or ""
    args = fn.get("arguments", tool_call.get("arguments"))
    if not isinstance(args, str):
        try:
            args = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args = repr(args)
    return f"{name}\x00{args}"


def _tool_call_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    """The function names of this turn's calls, in order, for display.

    Reads the same three shapes :func:`_call_key` does (``function.name`` / ``name`` / ``tool_name``);
    a call whose name cannot be read is dropped rather than shown as an empty string.
    """
    names: list[str] = []
    for tool_call in tool_calls:
        fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = fn.get("name") or tool_call.get("name") or tool_call.get("tool_name") or ""
        if name:
            names.append(name)
    return names


def _default_tool_messages(
    tool_calls: list[dict[str, Any]],
    observations: list[str],
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for i, obs in enumerate(observations):
        msg: dict[str, Any] = {"role": "tool", "content": "" if obs is None else str(obs)}
        if i < len(tool_calls) and isinstance(tool_calls[i], dict):
            tc = tool_calls[i]
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            tid = tc.get("id") or tc.get("tool_call_id")
            name = fn.get("name") or tc.get("name") or tc.get("tool_name")
            if tid:
                msg["tool_call_id"] = tid
            if name:
                msg["name"] = name
        msgs.append(msg)
    return msgs


def _malformed_tool_message(errors: list[str]) -> dict[str, Any]:
    """What goes back to the model when its tool-call markup did not parse.

    ``role='tool'`` because it is the outcome of the call the model just tried to
    make. There is no ``tool_call_id`` to pair it with -- the call never became a
    call -- which ``_default_tool_messages`` above already treats as optional.
    """
    reason = "; ".join(e for e in errors if e) or "the markup could not be parsed"
    return {
        "role": "tool",
        "content": (
            "Your tool call was not run: " + reason + ". Send the call again. Inside "
            "a JSON string a backslash has to be written as \\\\ and a line break as "
            "\\n; a single quote needs no backslash at all."
        ),
    }


# execute='first': a deployed rollout is ONE payload actor running every episode thread-side;
# max_concurrency lifts Ray's one-method-at-a-time default so in-flight episodes overlap (a
# PERFORMANCE ceiling only -- the admission truth stays with the driver's backpressure).
@remote_class(execute="first", max_concurrency=_MAX_CONCURRENT_EPISODES)
class MultiTurnRollout(Rollout):
    """Agentic multi-turn rollout with tool use, one episode per thread.

    Contract (matches :class:`Rollout`): accepts a ``List[Trajectory]`` and
    returns a ``List[Trajectory]`` of the same length, in the same order.

    Per-trajectory loop::

        response_callback(...)      # sampler or API -> SampledSequence
        ToolManager.call_many       # this turn's calls, one Env round trip
        extend_with_bridge          # labels=-100; never decode-reencode history

    Each trajectory runs its whole loop in its own thread. The callback may route
    each turn to the sampler or the API adapter; either can overlap with other
    trajectories while its thread waits on a GPU worker, endpoint, or sandbox.

    Every part of the loop except the generation is optional. Without a
    ``tool_manager`` nothing is dispatched and a reply that calls a tool simply
    ends the episode; without a ``followup_fn`` nothing is asked afterwards. All
    of them absent is a single-turn sampling pass, and that is a supported way to
    use this rather than a degenerate one.

    This drives the conversation itself. An agent that ships as its own program
    cannot be driven this way and belongs in
    :class:`~.external.ExternalRollout`.

    A supplied sampler must declare ``sample`` with ``enable_continous_work``.
    Without it, ``slice_dp`` spreads each single-request call over every worker
    and raises on ranks that receive nothing.

    Shared state: ``sampler``, API client and ``template`` are read-only during a
    rollout and safe to share.

    Per-call overrides via ``**kwargs``:
        * ``sampling_params``: :class:`SamplingParams` for every episode.
        * ``response_callback``: chooses a backend for each assistant turn and
          returns one :class:`SampledSequence`.
        * ``tool_manager``: a single :class:`ToolManager` or a 1:1 list.
        * ``adapter_path`` / ``use_base_model``: see ``__init__``.
        * ``followup_fn``: see ``__init__``.
        * ``sticky_session``: keep every turn of an episode on the same sampler
          worker. Generate one session ID per episode when enabled.
        * ``session_ids``: explicit per-trajectory session strings (or ``None``
          for load balancing), aligned with the trajectory batch. Supplying them
          also enables sticky routing. Requires a continuous-work sampler.
    """

    def __init__(
        self,
        sampler=None,
        template: Template | None = None,
        tool_manager: ToolManager | None = None,
        harness=None,
        sampling_params: SamplingParams | None = None,
        max_turns: int = 6,
        max_trajectory_tokens: int | None = None,
        concurrency: int | None = None,
        tracer: TraceWriter | None = None,
        adapter_path: str | None = None,
        use_base_model: bool = False,
        stop_after_stuck_turns: int = 0,
        max_malformed_retries: int = 2,
        followup_fn: Callable[[Trajectory, int], Any] | None = None,
        api: API | None = None,
        response_callback: ResponseCallback | None = None,
        api_appended_as: Literal["demonstration", "context"] = "demonstration",
        api_kwargs: dict[str, Any] | None = None,
        allow_partial_rollout: bool = False,
        tools_factory: Callable[[Any], tuple[Any, list[Any], Any]] | None = None,
        tools_config: Any = None,
        sticky_session: bool = False,
    ):
        super().__init__()
        if isinstance(sampler, (API, APISampler)):
            if api is not None:
                raise ValueError("the positional backend and api= both specify an API")
            api, sampler = sampler, None
        # A local Template is what makes a trajectory trainable: it encodes the
        # opening turn, reads tool-call markup, and owns the tokenizer. Without one
        # the rollout still drives a conversation, but accounts in messages rather
        # than tokens (see MessageLedger) -- the mode a text-only backend such as a
        # remote teacher runs in. ``api=`` cannot work without a template, since
        # APISampler formats the call with it.
        message_only = template is None
        if message_only and api is not None:
            raise ValueError(
                "api=/APISampler requires a local Template to format the call; "
                "message-only mode (template=None) has none."
            )
        if response_callback is None and sampler is None and api is None:
            raise ValueError("MultiTurnRollout requires a sampler or API when response_callback is omitted")
        if sampler is not None:
            sample = getattr(type(sampler), "sample", None)
            if sample is None:
                raise TypeError(f"backend must be an API or sampler, got {type(sampler).__name__}")
            # A message-only backend is a local text sampler, not a slice_dp GPU
            # worker, so the one-request-per-call spreading this guards against does
            # not apply; only the token-level path needs the declaration.
            if not message_only and not getattr(sample, "_enable_continous_work", False):
                raise ValueError(
                    f"{type(sampler).__name__}.sample must be declared with "
                    "enable_continous_work=True: this rollout samples one trajectory per "
                    "call, and a slice_dp sampler raises when a worker gets nothing from "
                    "a batch of one."
                )
        if adapter_path and use_base_model:
            raise ValueError(
                "adapter_path and use_base_model=True ask for opposite "
                "weights; the sampler would drop the adapter silently."
            )
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f"max_trajectory_tokens must be >= 1 or None, got {max_trajectory_tokens}")
        if sticky_session and not getattr(getattr(type(sampler), "sample", None), "_enable_continous_work", False):
            raise ValueError("sticky_session requires a sampler with enable_continous_work=True.")
        self._init_common(max_turns=max_turns, sampling_params=sampling_params, concurrency=concurrency, tracer=tracer)
        self.sampler = sampler
        self.sticky_session = sticky_session
        self.template = template
        self._template_backend = getattr(type(sampler), "template_backend", "vllm") if sampler is not None else "vllm"
        # No template -> account in messages, not tokens (see ``_new_ledger``).
        self._message_only = message_only
        if isinstance(api, APISampler):
            if api_kwargs:
                raise ValueError("api_kwargs belongs on the APISampler when api= is already adapted")
            if api.template is not template:
                raise ValueError("MultiTurnRollout and APISampler must share the same template instance")
            self.api = api
        elif api is not None:
            self.api = APISampler(api, template, appended_as=api_appended_as, api_kwargs=api_kwargs)
        else:
            if api_kwargs:
                raise ValueError("api_kwargs requires an API backend")
            self.api = None
        self.response_callback = response_callback or default_response_callback
        if tools_factory is not None and tool_manager is not None:
            raise ValueError(
                "tools_factory builds a per-episode sandbox manager; a shared "
                "tool_manager cannot also be set. Pass one or the other."
            )
        # A sandbox env pool holds live locks/queues and cannot cross the Ray boundary, so it is
        # built HERE (engine-side) via the injected factory, never handed in pre-built. A sandbox
        # binds tools per episode (each trajectory leases its own env), so the shared tool_manager
        # is dropped then and every episode passes its own leased manager.
        self.env_pool, self.tool_plugins, self._tool_manager_for = (
            (None, [], None) if tools_factory is None else tools_factory(tools_config)
        )
        self._per_episode_tools = self.env_pool is not None and bool(self.tool_plugins)
        # Every RL entry (generate/admit) threads this into each per-turn sample call: it makes
        # a turn resumable across an in-place weight republish instead of returning truncated.
        self._allow_partial_rollout = allow_partial_rollout
        self.tool_manager = tool_manager
        # An optional AgentHarness (a pool exposing ``.lease()``, or a 1:1 list)
        # that shapes messages each turn. The ledger still owns every token id;
        # the harness only reshapes the message view and this turn's tool
        # framing, so a framework agent (ms-agent, ...) can drive this loop
        # locally without this class knowing which framework it is. This is the
        # forward-tunnel alternative to ``ExternalRollout``'s reverse endpoint;
        # both stay selectable and general.
        self.harness = harness
        # A LoRA directory on disk, forwarded to every sample call. Training syncs
        # its adapter into the sampler directly, but evaluating a saved one has no
        # such channel: without this, an eval script would silently measure the
        # base model and report it as the trained one.
        self.adapter_path = adapter_path
        # The other direction: force the base weights. Needed because a sampler
        # mid-training falls back to the LoRA synced into it whenever a call names
        # no adapter, so a utility rollout (summarizing, judging) that wants the
        # untrained model has to say so rather than stay silent.
        self.use_base_model = use_base_model
        self.max_trajectory_tokens = max_trajectory_tokens
        # How many stuck turns in a row end the episode; 0 runs to ``max_turns``
        # regardless. A turn is stuck when it made no progress at all, which is
        # either of:
        #   * every call in it came back an error, or
        #   * every call in it was byte-identical to one already made in this
        #     episode, whatever it answered.
        # One useful call in a turn resets the count, so probing for something
        # and then creating it is untouched.
        #
        # Both halves are needed, measured by replaying 12 recorded episodes:
        # errors alone stop 1 of 12 and save 9 of 239 calls, because the worst
        # offenders interleave a failing call with a glob that succeeds. Adding
        # the repeat rule stops 3 of 12 and saves 63 calls, and the three are
        # exactly the ones that spent 54, 84 and 17 calls to leave behind a
        # script that could not run. Nothing an episode kept was written after
        # its stop point except those broken scripts.
        if stop_after_stuck_turns < 0:
            raise ValueError(f"stop_after_stuck_turns must be >= 0, got {stop_after_stuck_turns}")
        self.stop_after_stuck_turns = stop_after_stuck_turns
        # How many replies in a row may carry tool-call markup that does not
        # parse before the episode ends anyway. Such a reply is not the model
        # declining to call a tool -- it asked for one and the markup was
        # rejected -- so it gets the parser's reason back as a tool message and
        # another turn. Measured on one challenger run: 6 of 59 episodes ended
        # here, each having written a whole ``<tool_call>`` block whose JSON held
        # a Python-style ``\'`` escape or a raw newline, and each was told
        # nothing. The cap exists because a model that cannot produce valid JSON
        # would otherwise spend all of ``max_turns`` failing to; 0 restores the
        # old behaviour of ending the episode on the first one.
        if max_malformed_retries < 0:
            raise ValueError(f"max_malformed_retries must be >= 0, got {max_malformed_retries}")
        self.max_malformed_retries = max_malformed_retries
        # Called with (trajectory, how many follow-ups it has had already) at the
        # moment an episode would end: because the model stopped calling tools,
        # because it used up ``max_turns``, or because it was stopped for being
        # stuck. Returning a string appends it as a user message and the episode
        # keeps going; returning None ends it. May also return
        # ``(text, SamplingParams)`` to give that stage its own budget.
        #
        # It is asked in the ran-out-of-budget cases too, not only when the model
        # says it is done, because what those stages read is the state the episode
        # left behind -- which exists either way. An episode dropped for hitting
        # the turn limit costs its whole sandbox run and produces nothing.
        #
        # This is what keeps a multi-stage episode in ONE trajectory. The
        # alternative -- ending here and starting a second rollout whose prompt is
        # this conversation -- re-encodes the history as prompt, so every earlier
        # assistant turn comes back with labels == -100 and only the last stage is
        # trainable. Appending goes through the same append-only bridge the tool
        # observations use, so labels and logprobs of the earlier turns survive and
        # the whole chain can be trained as one sample.
        #
        # Tool calls are no longer dispatched once a follow-up has been appended:
        # the stages that come after the tool-using one are meant to produce text
        # about the state as it is, and a python block in a reply parses as a call
        # list -- 41 of 146 such replies dispatched something in a measured run --
        # which would rewrite the very state the text is about.
        self.followup_fn = followup_fn
        if self.template is not None:
            assert self.template.truncation_strategy != "split", (
                "MultiTurnRollout does not support truncation_strategy='split'; "
                "use left/right/delete/raise on the template."
            )

    @remote_function()
    def __call__(self, trajectories: list[Trajectory], **kwargs) -> list[Trajectory]:
        """The base implementation; the decorator is what a deployed handle needs."""
        return super().__call__(trajectories, **kwargs)

    def stream(
        self,
        trajectory: Trajectory,
        *,
        sampling_params: SamplingParams | None = None,
        tool_manager: ToolManager | None = None,
        adapter_path: str | None = None,
    ):
        """Display-only streaming variant of ONE episode: yield token deltas live, then the transcript.

        Runs the same tool loop as :meth:`_run_episode` but accounts at the MESSAGE level (a
        :class:`MessageLedger`, template-aware so tool-call markup in the reply text still parses) and
        streams each generation's text deltas as they arrive, so an interactive REPL can echo tokens
        live. The result is a transcript, NOT a trainable trajectory: ``sample_stream`` yields text
        only, so no token ids / labels / logprobs are banked. Training rollouts keep using the blocking
        :meth:`__call__` path, which produces the trainable per-token account.

        Single-trajectory and local: a generator does not cross the Ray boundary, so this needs an
        in-process sampler with ``sample_stream`` (the TUI), not a deployed handle or an API backend.

        Termination is deliberately simpler than ``_run_episode`` (no follow-up stages, no stuck-turn
        or malformed-markup retries -- those are training-rollout concerns): the episode ends when the
        model stops calling tools, when a generation hits its length cap, or at ``max_turns``.

        Yields:
            :class:`StreamDelta` for each token delta of each turn's generation;
            :class:`StreamToolCall` after each turn that dispatched tools;
            :class:`StreamDone` once at the end, carrying the transcript trajectory.
        """
        if self.sampler is None or not hasattr(self.sampler, "sample_stream"):
            raise ValueError(
                "MultiTurnRollout.stream requires a local sampler exposing sample_stream; "
                "an API / response_callback backend has no token stream to show."
            )
        params = sampling_params if sampling_params is not None else self.sampling_params
        if params is None:
            params = SamplingParams()
        # Fall back to the construction-time tool_manager, as __call__/_resolve_call does, so a caller
        # that wired tools at build time need not re-pass them per stream. A tools_factory sandbox
        # carries no shared manager (it was rejected at construction); the single streamed episode
        # leases one env for the whole conversation turn and builds its manager from it, exactly as
        # _run_one does per training episode.
        tool_manager = tool_manager if tool_manager is not None else self.tool_manager
        lease = contextlib.ExitStack()
        try:
            if tool_manager is None and self._per_episode_tools:
                env = lease.enter_context(self.env_pool.lease())
                tool_manager = self._tool_manager_for(env, self.tool_plugins)
            yield from self._stream_episode(trajectory, params, adapter_path, tool_manager)
        finally:
            lease.close()

    def _stream_episode(
        self,
        trajectory: Trajectory,
        params: SamplingParams,
        adapter_path: str | None,
        tool_manager: ToolManager | None,
    ):
        """The streaming loop proper; the lease above is held for all of it."""
        # Message-level accounting, but template-aware so a reply's tool-call markup still parses. The
        # feature stays a {'messages','tools'} dict with no input_ids, so sample_stream re-encodes the
        # whole trajectory each turn and the tool schema reaches the prompt tokens.
        ledger = MessageLedger(label="stream", max_tokens=self.max_trajectory_tokens, template=self.template)
        opening_tools = None
        if "tools" not in trajectory and tool_manager is not None:
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
                ledger.input_feature, params, adapter_path=adapter_path
            ):
                if delta:
                    pieces.append(delta)
                    yield StreamDelta(delta)
                if finish_reason is not None:
                    finish = finish_reason
            text = "".join(pieces)
            stop_reason = finish or STOP_NO_TOOL
            ledger.record(SampledSequence(stop_reason=stop_reason, tokens=[], decoded=text))
            if finish == "length":
                # Cut off mid-reply: the tokens streamed are all there is, and no tool call it was
                # still emitting gets dispatched (the turn never decided it was done).
                truncated = True
                break
            tool_calls = ledger.parse_tool_call(text)
            if not tool_calls:
                break
            if turns >= self.max_turns:
                # Out of turns: the model asked for tools but there is no further generation to read
                # their results, so do not dispatch (mirrors _run_episode) -- mark the episode truncated.
                truncated = True
                break
            if tool_manager is None:
                raise ValueError("the model emitted tool_calls but stream() was given no tool_manager")
            observations = self._run_tools(tool_manager, tool_calls)
            yield StreamToolCall(names=_tool_call_names(tool_calls), observations=observations)
            ledger.observe(_default_tool_messages(tool_calls, observations))
        out = ledger.merge(trajectory, turns=ledger.turns, stop_reason=stop_reason, truncated=truncated)
        yield StreamDone(out)

    # ---------------------------------------------------------- RL entries
    #
    # The batch and streaming faces over the same engine: ``generate`` is the blocking
    # (locally) / lazy (over Ray) batch entry, and admit/poll/collect/cancel is the
    # four-call protocol the streaming driver's engine seams default-bind to. Both
    # translate each episode's ledger-merged trajectory into a RolloutSample.

    @remote_function(dispatch="all", collect="none", lazy_collect=True)
    def generate(
        self,
        prompts: list[list[dict]],
        num_samples: int = 1,
        sampling_params: dict | None = None,
        prompt_extras: list[dict[str, Any]] | None = None,
        force_logprobs: bool = True,
        adapter_path: str | None = None,
        allow_partial_rollout: bool | None = None,
    ) -> list[RolloutSample]:
        """Run one episode per (prompt, sample) pair and translate each output.

        ``num_samples`` completions per prompt, grouped by prompt -- a GRPO group is one
        prompt's slice of the returned list. ``lazy_collect=True`` makes a REMOTE call
        return a LazyCollect handle immediately (what :meth:`admit` parks on its
        payload); a local call still blocks.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if prompt_extras is not None and len(prompt_extras) != len(prompts):
            raise ValueError("prompt_extras must contain exactly one mapping per prompt.")
        extras = prompt_extras or [{} for _ in prompts]
        trajectories, metadata = [], []
        for prompt_index, prompt in enumerate(prompts):
            for _ in range(num_samples):
                extra = copy.deepcopy(extras[prompt_index])
                trajectory: dict[str, Any] = {"messages": copy.deepcopy(prompt)}
                # Thread the prompt row's media columns so a multimodal episode samples its media.
                for key in _MULTIMODAL_PROMPT_KEYS:
                    if extra.get(key):
                        trajectory[key] = copy.deepcopy(extra[key])
                trajectories.append(trajectory)
                metadata.append((str(prompt_index), extra))
        sp = self._rl_sampling_params(sampling_params, force_logprobs)
        # adapter_path threads into every per-turn sampler.sample (a LoRA run must not sample
        # from the base); allow_partial_rollout rides the same kwargs channel (False reads as
        # omitted).
        kwargs: dict[str, Any] = {
            "sampling_params": sp,
            "allow_partial_rollout": (
                self._allow_partial_rollout if allow_partial_rollout is None else allow_partial_rollout
            ),
        }
        if adapter_path:
            kwargs["adapter_path"] = adapter_path
        outputs = self(trajectories, **kwargs)
        if len(outputs) != len(metadata):
            raise RuntimeError(f"multi-turn rollout returned {len(outputs)} trajectories for {len(metadata)} inputs.")
        return [
            trajectory_to_rollout_sample(output, prompt_id, extra)
            for output, (prompt_id, extra) in zip(outputs, metadata, strict=False)
        ]

    def _rl_sampling_params(self, sampling_params: Any, force_logprobs: bool) -> SamplingParams:
        """The one SamplingParams every RL entry samples under (contract 15).

        A per-call dict replaces the construction-time params wholesale; either way
        old_logps are forced and num_samples is 1 -- the engine drives one trajectory
        per call, so no path can silently sample under different params than another.
        """
        if isinstance(sampling_params, SamplingParams):
            sp = sampling_params
        elif sampling_params is None:
            sp = self.sampling_params or SamplingParams()
        else:
            sp = _build_sampling_params(1, sampling_params, force_logprobs)
        if force_logprobs and not sp.logprobs:
            sp = replace(sp, logprobs=1)
        if sp.num_samples != 1:
            sp = replace(sp, num_samples=1)
        return sp

    def admit(
        self,
        prompt: list[dict],
        trajectory_idx: int = 0,
        *,
        prompt_idx: Any = 0,
        sampling_params: dict | None = None,
        prompt_extras: dict[str, Any] | None = None,
        adapter_path: str | None = None,
        policy_version: int = 0,
    ) -> SampleHandle:
        """Admit ONE trajectory without blocking; poll/collect/cancel complete the protocol.

        ``prompt_idx`` / ``trajectory_idx`` are carried for the collect-side group
        stamping; ``policy_version`` is the admission pin, stamped onto the sample. A
        REMOTE deployment parks the lazy generate on the handle's payload; a local one
        runs the episode on a daemon-thread Future.
        """
        handle = self._new_handle(
            prompt, trajectory_idx, prompt_idx=prompt_idx, prompt_extras=prompt_extras,
            adapter_path=adapter_path, policy_version=policy_version)
        generate_kwargs = dict(
            num_samples=1,
            sampling_params=sampling_params,
            prompt_extras=[prompt_extras] if prompt_extras else None,
            force_logprobs=True,
            adapter_path=adapter_path,
        )
        if hasattr(self, "_actors"):
            # The remote episode runs on the actor: the generate call itself returns a
            # LazyCollect immediately, so admission never blocks the driver thread.
            handle.payload = self.generate([prompt], **generate_kwargs)
            return handle
        return self._admit_local(handle, lambda: self.generate([prompt], **generate_kwargs))

    def poll(self, handles: list[SampleHandle]) -> list[SampleHandle]:
        """Non-blocking as-completed query; a failed local episode raises here, fail-loudly."""
        completed: list[SampleHandle] = []
        for handle in handles:
            if handle.payload is not None:
                # ready() is a ray.wait(timeout=0) status probe, never a fetch; a FAILED episode
                # still reports ready and its exception surfaces at collect.
                if handle.payload.ready():
                    completed.append(handle)
                continue
            if self._episode_done(handle):
                completed.append(handle)
        return completed

    def collect(self, handle: SampleHandle) -> RolloutSample:
        """Collect ONE completed trajectory, stamped with its group id / pin version / prompt."""
        if handle.payload is not None:
            samples = handle.payload()
        else:
            samples = self._pop_episode_future(handle).result()
        if len(samples) != 1:
            raise RuntimeError(
                f"episode collect expected exactly 1 sample (submission {handle.submission_id}), got {len(samples)}."
            )
        return self._stamp_admission(samples[0], handle)

    def cancel(self, handle: SampleHandle | None) -> None:
        """Drop an admitted-but-uncollected episode: remote drops the ref, local abandons.

        A concurrent.futures.Future silently discards an unretrieved exception, so an
        abandoned episode that later fails needs no drain here.
        """
        if handle is None:
            return
        if handle.payload is not None:
            # The in-actor episode finishes on its own; its result is discarded with the ref.
            return
        self._drop_episode_future(handle.submission_id)

    def stop(self, submission_id: str) -> None:
        """End an in-flight submission at driver teardown (abort + settle).

        The local episode's future is dropped -- its daemon thread runs to completion
        discarded, a thread being the one thing Python will not end early. A remote
        episode's payload ref is dropped by the caller together with the handle, so
        there is nothing further to end here.
        """
        self._drop_episode_future(submission_id)

    # `dispatch` has no 'first' mode (only all / slice / slice_dp); the single-worker
    # placement already comes from @remote_class(execute='first'), so 'all' reaches it.
    @remote_function(dispatch="all", collect="none", lazy_collect=False)
    def abort_all_inflight(self) -> None:
        """Abort every in-flight sampler generation so an in-place republish can resume them.

        Distinct from cancel (which DROPS a submission): abort keeps it alive and
        resumable on the fresh weights. Runs where the sampler handle is reachable --
        on a Ray proxy the driver instance has no sampler attribute, so the decorated
        dispatch is what carries the call to the worker that holds one.
        """
        abort = getattr(getattr(self, "sampler", None), "abort_all_inflight", None)
        if not callable(abort):
            raise RuntimeError(
                "abort-on-publish needs a sampler with abort_all_inflight (the twinkle "
                "partial-rollout capability): the core vLLM/SGLang samplers provide it."
            )
        return abort()

    @remote_function(dispatch="all", collect="none", lazy_collect=False)
    def close(self) -> None:
        """Release the sandbox env pool; the sampler's lifecycle is not owned here."""
        getattr(self, "_episode_futures", {}).clear()
        if self.env_pool is not None:
            self.env_pool.close()

    def _resolve_call(self, kwargs: dict[str, Any], n: int) -> dict[str, Any]:
        adapter_path = kwargs.get("adapter_path", self.adapter_path)
        # Left out entirely when unset, so a sampler without LoRA enabled sees the
        # same call it always did.
        adapter_kwargs = {"adapter_path": adapter_path} if adapter_path else {}
        if kwargs.get("use_base_model", self.use_base_model):
            adapter_kwargs["use_base_model"] = True
        # ``allow_partial_rollout`` makes each per-turn ``sample()`` resumable across an in-place weight
        # republish: a publish aborts the turn in flight and, with this set, it continues from its own
        # tokens on the fresh weights instead of returning truncated -- what lets a MULTI-TURN episode
        # span versions soundly under the streaming driver's ``in_place`` publication. It is a
        # local-sampler capability, so it is threaded only when a sampler drives the turns; an API/teacher
        # backend has no in-place weight sync to resume across (and its callback would not take the kwarg).
        # Left out entirely when unset, so the default per-turn call is unchanged.
        if self.sampler is not None and kwargs.get("allow_partial_rollout", False):
            adapter_kwargs["allow_partial_rollout"] = True
        sampling_params = kwargs.get("sampling_params", self.sampling_params)
        if sampling_params.num_samples != 1:
            raise ValueError(f"MultiTurnRollout supports num_samples=1 only, got {sampling_params.num_samples}")
        response_callback = kwargs.get("response_callback", self.response_callback)
        if not callable(response_callback):
            raise TypeError("response_callback must be callable")
        session_ids = kwargs.get("session_ids")
        if kwargs.get("sticky_session", self.sticky_session) or session_ids is not None:
            sample = getattr(type(self.sampler), "sample", None)
            if not getattr(sample, "_enable_continous_work", False):
                raise ValueError("sticky_session requires a sampler with enable_continous_work=True.")
            if session_ids is None:
                # Resolve once before launching episode threads, never once per turn.
                session_ids = [uuid.uuid4().hex for _ in range(n)]
            elif not isinstance(session_ids, (list, tuple)):
                raise TypeError("session_ids must be a list or tuple aligned with trajectories.")
            elif len(session_ids) != n:
                raise ValueError(f"session_ids must contain {n} entries, got {len(session_ids)}.")
            if any(sid is not None and (not isinstance(sid, str) or not sid) for sid in session_ids):
                raise ValueError("session_ids entries must be non-empty strings or None.")
            session_ids = tuple(session_ids)
        return {
            "sampling_params": sampling_params,
            "adapter_kwargs": adapter_kwargs,
            "session_ids": session_ids,
            "response_callback": response_callback,
            "tool_managers": self._broadcast(kwargs.get("tool_manager", self.tool_manager), n, name="tool_manager"),
            "followup_fn": kwargs.get("followup_fn", self.followup_fn),
            "harnesses": self._resolve_harness(kwargs.get("harness", self.harness), n),
        }

    def _resolve_harness(self, harness, n: int) -> list[Any]:
        """One harness per episode. A pool is shared and leased per thread; a
        bare harness carries per-episode state and cannot be shared across
        parallel threads, so n>1 needs a pool or a 1:1 list."""
        if harness is None:
            return [None] * n
        if hasattr(harness, "lease"):
            return [harness] * n
        return self._broadcast(harness, n, name="harness", per_trajectory=True)

    def _new_ledger(self, index: int):
        """The episode account for one trajectory: token-level or message-level.

        With a template the trajectory is trainable and every id passes through a
        :class:`TurnLedger`; without one (a text-only backend) there is nothing to
        encode, so a :class:`MessageLedger` keeps the same call surface over a
        message list. The loop below is identical either way -- it talks only to
        the ledger, never to ``self.template`` directly.
        """
        label = f"trajectory {index}"
        if self._message_only:
            return MessageLedger(label=label, max_tokens=self.max_trajectory_tokens)
        return TurnLedger(
            self.template,
            label=label,
            max_tokens=self.max_trajectory_tokens,
            backend=self._template_backend,
        )

    def _run_one(self, trajectory: Trajectory, index: int, ctx: dict[str, Any]) -> Trajectory:
        if self._per_episode_tools:
            # A sandbox env holds one workspace: the trajectory drives alone with its
            # tools bound to the env it leased; the lease blocks when the pool empties
            # (the sandbox's own concurrency bound) and is held for the whole episode.
            with self.env_pool.lease() as env:
                scoped = dict(ctx)
                scoped["tool_managers"] = [self._tool_manager_for(env, self.tool_plugins)] * len(ctx["tool_managers"])
                return self._run_with_harness(trajectory, index, scoped)
        return self._run_with_harness(trajectory, index, ctx)

    def _run_with_harness(self, trajectory: Trajectory, index: int, ctx: dict[str, Any]) -> Trajectory:
        # A harness from a pool is leased for the episode and returned after, so
        # its per-episode state (an agent's memory, say) never leaks into the
        # next one. Anything else is used as passed.
        harness = ctx["harnesses"][index]
        if harness is not None and hasattr(harness, "lease"):
            with harness.lease() as leased:
                return self._run_episode(trajectory, index, ctx, leased)
        return self._run_episode(trajectory, index, ctx, harness)

    def _run_episode(self, trajectory: Trajectory, index: int, ctx: dict[str, Any], harness=None) -> Trajectory:
        tool_manager: ToolManager = ctx["tool_managers"][index]
        followup_fn = ctx["followup_fn"]
        # Each episode owns its metadata; mutating the shared dict would let
        # concurrent episodes overwrite one another's session ID.
        adapter_kwargs: dict[str, Any] = dict(ctx["adapter_kwargs"])
        if ctx.get("session_ids") is not None:
            adapter_kwargs["_session_ids"] = [ctx["session_ids"][index]]
        response_callback: ResponseCallback = ctx["response_callback"]

        # The token account for this episode. Every id the trajectory ends up
        # trained on passes through it; what stays in this function is the policy
        # that decides when to add one. See ``ledger.py``.
        ledger = self._new_ledger(index)
        # A trajectory that named no tools advertises the manager's, so the prompt
        # lists what can actually be dispatched.
        # A harness may shape the opening (system prompt, tool schema) before
        # the one encode of the episode; after that it is append-only (below).
        if harness is not None:
            trajectory = harness.before_generate(trajectory)
        opening_tools = None
        if "tools" not in trajectory and tool_manager is not None:
            opening_tools = list(tool_manager.tool_infos() or [])
        ledger.open(trajectory, tools=opening_tools)

        stop_reason: str | None = None
        generation_error: str | None = None
        truncated = False
        params = ctx["sampling_params"]
        # Consecutive turns that made no progress, the calls already issued, and
        # whether being stuck is what ended the episode. All three stay at their
        # initial value when ``stop_after_stuck_turns`` is 0.
        stuck_turns = 0
        seen_calls: set = set()
        stuck_stop = False
        # Replies in a row whose tool-call markup did not parse. Reset by any
        # reply that produced a call, so one bad escape in the middle of a
        # working episode does not count against a later one.
        malformed_turns = 0
        followups = 0
        # Why the tool-calling part ended, when it was not the model's own
        # choice: 'max_turns' or 'stuck'. Reported separately from ``truncated``
        # because an episode can hit the turn limit and still go on to answer the
        # follow-up stages, in which case nothing was cut off.
        tool_stop: str | None = None
        # The loop counts generations, and each granted follow-up buys the one
        # extra generation it asked for. Paying for the follow-up stages out of
        # ``max_turns`` would mean an episode that spent its whole tool budget
        # never reaches the stages that read what it built, and a short one
        # silently gets more tool turns than a long one.
        budget = self.max_turns
        spent = 0

        def grant_followup() -> bool:
            """Ask for one more stage; True when the episode carries on.

            Sets ``truncated`` itself in the one case where the answer is "there
            is no room for another stage", which is a cut trajectory rather than
            a caller that had nothing more to ask.
            """
            nonlocal followups, budget, params, truncated
            if followup_fn is None or followups >= MAX_FOLLOWUPS:
                return False
            followup = followup_fn(
                ledger.merge(trajectory, turns=ledger.turns, stop_reason=stop_reason, truncated=truncated), followups
            )
            if followup is None:
                return False
            text, next_params = self._unpack_followup(followup)
            if not ledger.observe([{"role": "user", "content": text}]):
                truncated = True
                return False
            # Follow-up stages are answers, so an API must not see tool schemas.
            ledger.clear_tools()
            followups += 1
            budget += 1
            if next_params is not None:
                params = next_params
            return True

        while spent < budget:
            spent += 1

            # 2. One request. The callback chooses the local sampler or the API
            # adapter, but both paths return exactly one SampledSequence.
            try:
                seq = response_callback(
                    self.sampler,
                    self.api,
                    params,
                    input_feature=ledger.input_feature,
                    rollout_state=ledger.state,
                    adapter_kwargs=adapter_kwargs,
                    trajectory=trajectory,
                    trajectory_index=index,
                    turn=ledger.turns + 1,
                    followups=followups,
                )
            except APIGenerationError as exc:
                stop_reason = STOP_GENERATION_ERROR
                generation_error = str(exc)
                truncated = True
                break
            if not isinstance(seq, SampledSequence):
                raise TypeError(f"response_callback must return SampledSequence, got {type(seq).__name__}")

            ledger.record(seq)
            stop_reason = seq.stop_reason

            msgs = ledger.messages
            last_msg = msgs[-1] if msgs else None
            tool_calls = last_msg.get("tool_calls") if isinstance(last_msg, dict) else None
            if not tool_calls:
                tool_calls = ledger.parse_tool_call(seq.decoded or "")
            if followups:
                tool_calls = None
                if isinstance(last_msg, dict):
                    message = dict(last_msg)
                    message["content"] = ledger.decode_tokens(seq)
                    message.pop("tool_calls", None)
                    ledger.replace_last_message(message)

            if harness is not None:
                self._harness_after_generate(harness, ledger, seq.decoded or "", tool_calls)

            # 3. Termination conditions
            # A reply cut off at ``max_tokens`` is truncated in exactly the sense
            # the flag names, and consumers read the flag to tell a trajectory
            # that finished from one that ran out of room: a difficulty
            # measurement counting such an attempt as a genuine failure blames
            # the task for the token budget. Tool calls the cut reply happens to
            # contain are still not dispatched -- the turn never got to decide it
            # was done emitting them.
            if seq.stop_reason == "length":
                truncated = True
                break

            # 3a. Sequence-length cap.
            if ledger.full():
                truncated = True
                break

            if not tool_calls:
                # Markup that did not parse is the model asking for a tool, not
                # declining one -- ending here tells it nothing and throws the
                # turn away. Hand back the parser's own reason and let it write
                # the call again. Not after a follow-up: tools are withdrawn
                # there on purpose (see ``followup_fn``), so a reply that looks
                # like a call is meant to be read as text.
                parse_errors = [] if followups else ledger.tool_call_errors(seq.decoded or "")
                if parse_errors and malformed_turns < self.max_malformed_retries:
                    malformed_turns += 1
                    if not ledger.observe([_malformed_tool_message(parse_errors)]):
                        truncated = True
                        break
                    continue
                # The episode is over as far as the model is concerned. Give the
                # caller one chance to say otherwise -- see ``followup_fn`` for
                # why this is not a second rollout.
                if grant_followup():
                    continue
                break

            if ledger.turns >= self.max_turns:
                # Out of tool turns, not out of episode: the stages that read the
                # end state can still run on what was built.
                tool_stop = "max_turns"
                if grant_followup():
                    continue
                truncated = True
                break

            malformed_turns = 0

            # 4. This turn's calls, appended as an append-only bridge of tool
            #    messages the model did not write.
            if tool_manager is None:
                raise ValueError("the model emitted tool_calls but this trajectory has no ToolManager")
            observations = self._run_tools(tool_manager, tool_calls)
            if self.stop_after_stuck_turns:
                keys = [_call_key(tc) for tc in tool_calls]
                all_repeats = bool(keys) and all(k in seen_calls for k in keys)
                seen_calls.update(keys)
                all_errors = bool(observations) and all(is_error_observation(o) for o in observations)
                if all_errors or all_repeats:
                    stuck_turns += 1
                else:
                    stuck_turns = 0

            if harness is not None:
                tool_messages = self._harness_tool_messages(harness, ledger, observations, tool_calls)
            else:
                tool_messages = _default_tool_messages(tool_calls, observations)
            overflowed = not ledger.observe(tool_messages)
            if overflowed:
                truncated = True
            # Checked after the messages are appended, so the turns that ended
            # the episode are in the trajectory the caller reads.
            if self.stop_after_stuck_turns and stuck_turns >= self.stop_after_stuck_turns:
                stuck_stop = True
                tool_stop = "stuck"
                # Same as the turn limit: the tool phase is over, the state it
                # left is not, so the stages still get their turn.
                if not overflowed and grant_followup():
                    continue
                truncated = True
                break
            if overflowed:
                break

        # 5. Merge pif fields into the trajectory dict at TOP LEVEL so downstream
        #    consumers (VLLMSampler with ``'input_ids' in inputs``) see an encoded
        #    InputFeature and skip re-encoding. The ledger audits its own account
        #    on the way out -- one logprob per trainable token, or it raises.
        out = ledger.merge(
            trajectory,
            turns=ledger.turns,
            stop_reason=stop_reason,
            truncated=truncated,
            # ``truncated`` says something was cut off; these two say what ended
            # the tool-calling part, which is a different question -- an episode
            # can run out of turns, be handed a follow-up stage, and finish it.
            stuck_stop=stuck_stop,
            tool_stop=tool_stop,
            followups=followups,
        )
        if generation_error is not None:
            out["error"] = generation_error
        return out

    # ------------------------------------------------------------------ private

    @staticmethod
    def _harness_after_generate(harness, ledger, decoded: str, tool_calls) -> None:
        messages = ledger.messages
        if not messages:
            return
        prior = messages[:-1]
        shaped = harness.after_generate({"messages": prior, "tools": ledger.tools}, decoded, tool_calls)
        shaped_messages = (shaped or {}).get("messages") or []
        if len(shaped_messages) > len(prior):
            ledger.replace_last_message(shaped_messages[len(prior)])

    @staticmethod
    def _harness_tool_messages(harness, ledger, observations, tool_calls):
        prior = ledger.messages
        shaped = harness.after_tools({"messages": prior, "tools": ledger.tools}, observations, tool_calls)
        shaped_messages = (shaped or {}).get("messages") or []
        return shaped_messages[len(prior) :] or _default_tool_messages(tool_calls, observations)

    @staticmethod
    def _run_tools(tool_manager: ToolManager, tool_calls: list[dict[str, Any]]) -> list[str]:
        """Run one turn's calls, through ``call_many`` when the manager has it.

        A turn's calls go together because they share one Env round trip
        (``Env.step_batch``). Calls from *different* trajectories no longer meet
        here -- each episode has its own thread and, in the sandbox case, its own
        Env -- so there is nothing left to group across.

        A manager that answers with fewer results than calls leaves the rest
        empty rather than shifting them onto the wrong call.
        """
        if hasattr(tool_manager, "call_many"):
            contents = tool_manager.call_many(tool_calls)
        else:
            contents = [tool_manager(tc) for tc in tool_calls]
        obs = [""] * len(tool_calls)
        for i, content in enumerate(contents[: len(tool_calls)]):
            obs[i] = "" if content is None else str(content)
        return obs
