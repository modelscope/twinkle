# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, Callable, Dict, List, Literal, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams
from twinkle.infra import remote_class, remote_function
from twinkle.template.base import Template
from twinkle_agentic.protocol.api_sampler import APIGenerationError, APISampler
from twinkle_agentic.protocol.base import API
from twinkle_agentic.tools.tool_manager import ToolManager
from .base import MAX_FOLLOWUPS, STOP_GENERATION_ERROR, Rollout
from .ledger import TurnLedger
from .trace import TraceWriter


ResponseCallback = Callable[..., SampledSequence]


def _default_response_callback(sampler, api, sampling_params, *, input_feature, adapter_kwargs,
                               **kwargs) -> SampledSequence:
    """Use the sampler when present, otherwise the API adapter."""
    if sampler is None:
        if api is None:
            raise ValueError('response_callback was omitted, but no sampler or API was provided')
        return api(input_feature, sampling_params, **adapter_kwargs)
    responses = sampler.sample([input_feature], sampling_params=sampling_params, **adapter_kwargs)
    if not isinstance(responses, list):
        raise TypeError(f'expected List[SampleResponse] from sampler.sample, got '
                        f'{type(responses).__name__}')
    if len(responses) != 1:
        raise RuntimeError(f'sampler returned {len(responses)} responses for a single request; '
                           'expected exactly one.')
    response = responses[0]
    if not isinstance(response, SampleResponse):
        raise TypeError(f'expected SampleResponse from sampler.sample, got '
                        f'{type(response).__name__}')
    if len(response.sequences) != 1:
        raise RuntimeError(f'SampleResponse contains {len(response.sequences)} sequences; expected exactly one.')
    sequence = response.sequences[0]
    if not isinstance(sequence, SampledSequence):
        raise TypeError(f'expected SampledSequence, got {type(sequence).__name__}')
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


def _call_key(tool_call: Dict[str, Any]) -> str:
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


def _default_tool_messages(
    tool_calls: List[Dict[str, Any]],
    observations: List[str],
) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for i, obs in enumerate(observations):
        msg: Dict[str, Any] = {'role': 'tool', 'content': '' if obs is None else str(obs)}
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


def _malformed_tool_message(errors: List[str]) -> Dict[str, Any]:
    """What goes back to the model when its tool-call markup did not parse.

    ``role='tool'`` because it is the outcome of the call the model just tried to
    make. There is no ``tool_call_id`` to pair it with -- the call never became a
    call -- which ``_default_tool_messages`` above already treats as optional.
    """
    reason = '; '.join(e for e in errors if e) or 'the markup could not be parsed'
    return {
        'role':
        'tool',
        'content':
        ('Your tool call was not run: ' + reason + '. Send the call again. Inside '
         'a JSON string a backslash has to be written as \\\\ and a line break as '
         '\\n; a single quote needs no backslash at all.'),
    }


@remote_class()
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
    """

    def __init__(
        self,
        sampler=None,
        template: Optional[Template] = None,
        tool_manager: Optional[ToolManager] = None,
        harness=None,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        max_trajectory_tokens: Optional[int] = None,
        concurrency: Optional[int] = None,
        tracer: Optional[TraceWriter] = None,
        adapter_path: Optional[str] = None,
        use_base_model: bool = False,
        stop_after_stuck_turns: int = 0,
        max_malformed_retries: int = 2,
        followup_fn: Optional[Callable[[Trajectory, int], Any]] = None,
        api: Optional[API] = None,
        response_callback: Optional[ResponseCallback] = None,
        api_appended_as: Literal['demonstration', 'context'] = 'demonstration',
        api_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if isinstance(sampler, (API, APISampler)):
            if api is not None:
                raise ValueError('the positional backend and api= both specify an API')
            api, sampler = sampler, None
        if template is None:
            raise ValueError('MultiTurnRollout requires a local Template instance')
        if response_callback is None and sampler is None and api is None:
            raise ValueError('MultiTurnRollout requires a sampler or API when response_callback is omitted')
        if sampler is not None:
            sample = getattr(type(sampler), 'sample', None)
            if sample is None:
                raise TypeError(f'backend must be an API or sampler, got {type(sampler).__name__}')
            if not getattr(sample, '_enable_continous_work', False):
                raise ValueError(
                    f'{type(sampler).__name__}.sample must be declared with '
                    'enable_continous_work=True: this rollout samples one trajectory per '
                    'call, and a slice_dp sampler raises when a worker gets nothing from '
                    'a batch of one.')
        if adapter_path and use_base_model:
            raise ValueError('adapter_path and use_base_model=True ask for opposite '
                             'weights; the sampler would drop the adapter silently.')
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f'max_trajectory_tokens must be >= 1 or None, got '
                             f'{max_trajectory_tokens}')
        self._init_common(
            max_turns=max_turns,
            sampling_params=sampling_params,
            concurrency=concurrency,
            tracer=tracer)
        self.sampler = sampler
        self.template = template
        if isinstance(api, APISampler):
            if api_kwargs:
                raise ValueError('api_kwargs belongs on the APISampler when api= is already adapted')
            if api.template is not template:
                raise ValueError('MultiTurnRollout and APISampler must share the same template instance')
            self.api = api
        elif api is not None:
            self.api = APISampler(
                api, template, appended_as=api_appended_as, api_kwargs=api_kwargs)
        else:
            if api_kwargs:
                raise ValueError('api_kwargs requires an API backend')
            self.api = None
        self.response_callback = response_callback or _default_response_callback
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
            raise ValueError(f'stop_after_stuck_turns must be >= 0, got '
                             f'{stop_after_stuck_turns}')
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
            raise ValueError(f'max_malformed_retries must be >= 0, got '
                             f'{max_malformed_retries}')
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
        assert self.template.truncation_strategy != 'split', (
            "MultiTurnRollout does not support truncation_strategy='split'; "
            'use left/right/delete/raise on the template.')

    @remote_function()
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        """The base implementation; the decorator is what a deployed handle needs."""
        return super().__call__(trajectories, **kwargs)

    def _resolve_call(self, kwargs: Dict[str, Any], n: int) -> Dict[str, Any]:
        adapter_path = kwargs.get('adapter_path', self.adapter_path)
        # Left out entirely when unset, so a sampler without LoRA enabled sees the
        # same call it always did.
        adapter_kwargs = {'adapter_path': adapter_path} if adapter_path else {}
        if kwargs.get('use_base_model', self.use_base_model):
            adapter_kwargs['use_base_model'] = True
        sampling_params = kwargs.get('sampling_params', self.sampling_params)
        if sampling_params.num_samples != 1:
            raise ValueError(f'MultiTurnRollout supports num_samples=1 only, got '
                             f'{sampling_params.num_samples}')
        response_callback = kwargs.get('response_callback', self.response_callback)
        if not callable(response_callback):
            raise TypeError('response_callback must be callable')
        return {
            'sampling_params': sampling_params,
            'adapter_kwargs': adapter_kwargs,
            'response_callback': response_callback,
            'tool_managers': self._broadcast(
                kwargs.get('tool_manager', self.tool_manager), n, name='tool_manager'),
            'followup_fn': kwargs.get('followup_fn', self.followup_fn),
            'harnesses': self._resolve_harness(kwargs.get('harness', self.harness), n),
        }

    def _resolve_harness(self, harness, n: int) -> List[Any]:
        """One harness per episode. A pool is shared and leased per thread; a
        bare harness carries per-episode state and cannot be shared across
        parallel threads, so n>1 needs a pool or a 1:1 list."""
        if harness is None:
            return [None] * n
        if hasattr(harness, 'lease'):
            return [harness] * n
        return self._broadcast(harness, n, name='harness', per_trajectory=True)

    def _run_one(self, trajectory: Trajectory, index: int, ctx: Dict[str, Any]) -> Trajectory:
        # A harness from a pool is leased for the episode and returned after, so
        # its per-episode state (an agent's memory, say) never leaks into the
        # next one. Anything else is used as passed.
        harness = ctx['harnesses'][index]
        if harness is not None and hasattr(harness, 'lease'):
            with harness.lease() as leased:
                return self._run_episode(trajectory, index, ctx, leased)
        return self._run_episode(trajectory, index, ctx, harness)

    def _run_episode(self, trajectory: Trajectory, index: int, ctx: Dict[str, Any],
                     harness=None) -> Trajectory:
        tool_manager: ToolManager = ctx['tool_managers'][index]
        followup_fn = ctx['followup_fn']
        adapter_kwargs: Dict[str, Any] = ctx['adapter_kwargs']
        response_callback: ResponseCallback = ctx['response_callback']

        # The token account for this episode. Every id the trajectory ends up
        # trained on passes through it; what stays in this function is the policy
        # that decides when to add one. See ``ledger.py``.
        ledger = TurnLedger(self.template, label=f'trajectory {index}',
                            max_tokens=self.max_trajectory_tokens)
        # A trajectory that named no tools advertises the manager's, so the prompt
        # lists what can actually be dispatched.
        # A harness may shape the opening (system prompt, tool schema) before
        # the one encode of the episode; after that it is append-only (below).
        if harness is not None:
            trajectory = harness.before_generate(trajectory)
        opening_tools = None
        if 'tools' not in trajectory and tool_manager is not None:
            opening_tools = list(tool_manager.tool_infos() or [])
        ledger.open(trajectory, tools=opening_tools)

        stop_reason: Optional[str] = None
        generation_error: Optional[str] = None
        truncated = False
        params = ctx['sampling_params']
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
        tool_stop: Optional[str] = None
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
                ledger.merge(trajectory, turns=ledger.turns, stop_reason=stop_reason,
                             truncated=truncated), followups)
            if followup is None:
                return False
            text, next_params = self._unpack_followup(followup)
            if not ledger.observe([{'role': 'user', 'content': text}]):
                truncated = True
                return False
            # Follow-up stages are answers, so an API must not see tool schemas.
            ledger.input_feature['tools'] = []
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
                raise TypeError(f'response_callback must return SampledSequence, got '
                                f'{type(seq).__name__}')

            ledger.record(seq)
            pif = ledger.input_feature
            stop_reason = seq.stop_reason

            msgs = pif.get('messages') or []
            last_msg = msgs[-1] if msgs else None
            tool_calls = (last_msg.get('tool_calls') if isinstance(last_msg, dict) else None)
            if not tool_calls:
                tool_calls = self.template.parse_tool_call(seq.decoded or '')
            # After a follow-up, a parsed call is not a call: the tools were
            # withdrawn for these stages on purpose (see ``followup_fn``), and
            # dispatching python that the model wrote as *an answer* would edit
            # the state the answer is about.
            if followups:
                tool_calls = None
                # The parse also *rewrote* the message: when a reply parses as
                # a call, the template stores it with the call text removed, so
                # a caller reading the message gets less than the model wrote.
                # For these stages the reply is the deliverable, and one of the
                # tool-call formats is XML-shaped, so a check script asserting
                # the content of an .xml file matches it: 5 of ex12's 72 check
                # scripts came back with the XML cut out of them -- three then
                # ran with `content == ''` where the model had written the file's
                # real text, and two no longer held a code block at all.
                if msgs and isinstance(last_msg, dict):
                    # Decoded without the special tokens, the way the template
                    # writes a message: ``seq.decoded`` keeps the closing
                    # ``<|im_end|>``, and putting that in the content put it in
                    # the problem statements ex13 handed to solvers -- 7 of 7 of
                    # them ended in a literal '<|im_end|>'.
                    tok = getattr(self.template, 'tokenizer', None)
                    if tok is not None and seq.tokens:
                        last_msg['content'] = tok.decode(seq.tokens, skip_special_tokens=True)
                    else:
                        last_msg['content'] = seq.decoded or ''
                    last_msg.pop('tool_calls', None)

            # Let the harness normalize the assistant turn's message metadata
            # (tool-call ids, content shape) without touching the tokens the
            # ledger just banked.
            if harness is not None:
                self._harness_after_generate(harness, pif, seq.decoded or '', tool_calls)

            # 3. Termination conditions
            # A reply cut off at ``max_tokens`` is truncated in exactly the sense
            # the flag names, and consumers read the flag to tell a trajectory
            # that finished from one that ran out of room: a difficulty
            # measurement counting such an attempt as a genuine failure blames
            # the task for the token budget. Tool calls the cut reply happens to
            # contain are still not dispatched -- the turn never got to decide it
            # was done emitting them.
            if seq.stop_reason == 'length':
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
                parse_errors = ([] if followups else self.template.tool_call_errors(seq.decoded or ''))
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
                tool_stop = 'max_turns'
                if grant_followup():
                    continue
                truncated = True
                break

            malformed_turns = 0

            # 4. This turn's calls, appended as an append-only bridge of tool
            #    messages the model did not write.
            if tool_manager is None:
                raise ValueError('the model emitted tool_calls but this trajectory has no ToolManager')
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
                tool_messages = self._harness_tool_messages(
                    harness, ledger.input_feature, observations, tool_calls)
            else:
                tool_messages = _default_tool_messages(tool_calls, observations)
            overflowed = not ledger.observe(tool_messages)
            if overflowed:
                # Trajectory exceeded max_length.
                truncated = True
            else:
                pif = ledger.input_feature
            # Checked after the messages are appended, so the turns that ended
            # the episode are in the trajectory the caller reads.
            if self.stop_after_stuck_turns and stuck_turns >= self.stop_after_stuck_turns:
                stuck_stop = True
                tool_stop = 'stuck'
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
            out['error'] = generation_error
        return out

    # ------------------------------------------------------------------ private

    @staticmethod
    def _harness_after_generate(harness, pif: Dict[str, Any], decoded: str, tool_calls) -> None:
        """Swap the harness-shaped assistant message onto the banked turn.

        The ledger owns the ids; only the human-readable last message is
        replaced. The harness is handed the turns *before* this reply and
        appends its own normalized assistant, whose shaped form we take back.
        """
        msgs = list(pif.get('messages') or [])
        if not msgs:
            return
        prior = msgs[:-1]
        shaped = harness.after_generate(
            {'messages': list(prior), 'tools': pif.get('tools')}, decoded, tool_calls)
        shaped_msgs = (shaped or {}).get('messages') or []
        if len(shaped_msgs) > len(prior):
            pif['messages'][-1] = shaped_msgs[len(prior)]

    @staticmethod
    def _harness_tool_messages(harness, pif: Dict[str, Any], observations, tool_calls):
        """This turn's tool messages, framed by the harness (append-only tail).

        Falls back to the default framing when the harness appended nothing.
        """
        prior = list(pif.get('messages') or [])
        shaped = harness.after_tools(
            {'messages': list(prior), 'tools': pif.get('tools')}, observations, tool_calls)
        shaped_msgs = (shaped or {}).get('messages') or []
        tail = shaped_msgs[len(prior):]
        return tail or _default_tool_messages(tool_calls, observations)

    @staticmethod
    def _run_tools(tool_manager: ToolManager, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """Run one turn's calls, through ``call_many`` when the manager has it.

        A turn's calls go together because they share one Env round trip
        (``Env.step_batch``). Calls from *different* trajectories no longer meet
        here -- each episode has its own thread and, in the sandbox case, its own
        Env -- so there is nothing left to group across.

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
