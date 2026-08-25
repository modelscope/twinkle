# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampleResponse, SamplingParams
from twinkle.infra import remote_class, remote_function
from twinkle.template.base import Template
from twinkle_agentic.harness.base import AgentHarness
from twinkle_agentic.tools.tool_manager import ToolManager
from .base import Rollout
from .bridge import _to_plain, extend_with_bridge


def _append_only_delta(
    old_messages: List[Dict[str, Any]],
    new_messages: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Return newly appended messages, or None if ``new`` rewrote history."""
    old = list(old_messages or [])
    new = list(new_messages or [])
    if len(new) < len(old):
        return None
    for a, b in zip(old, new):
        if a != b:
            return None
    return new[len(old):]


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


@remote_class()
class MultiTurnRollout(Rollout):
    """Agentic multi-turn rollout with tool use (batched).

    Contract (matches :class:`Rollout`): accepts a ``List[Trajectory]`` and
    returns a ``List[Trajectory]`` of the same length, in the same order.
    Every turn issues a SINGLE batched ``sampler.sample(active_pifs)`` call
    so vLLM can run all live trajectories in parallel; finished trajectories
    are parked and excluded from subsequent batches.

    Per-trajectory loop::

        harness.before_generate     # append-only after the first encode
        sampler.sample(batch)       # keep seq.new_input_feature
        harness.after_generate
        ToolManager.call_many       # Env.step_batch when tools share an Env
        harness.after_tools         # format observations as tool messages
        extend_with_bridge          # labels=-100; never decode-reencode history

    Per-call overrides via ``**kwargs``:
        * ``sampling_params``: shared :class:`SamplingParams` for the batch.
        * ``tool_manager``: a single :class:`ToolManager` or a 1:1 list.
        * ``harness``: a single :class:`AgentHarness` or a 1:1 list. Framework
          specifics (ms-agent system/memory/tool-message shape) live in the
          harness subclass, not here.
        * ``followup_fn``: see ``__init__``.
    """

    def __init__(
        self,
        sampler,
        template: Template,
        tool_manager: Optional[ToolManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        max_trajectory_tokens: Optional[int] = None,
        trace_dir: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        harness: Optional[AgentHarness] = None,
        adapter_path: Optional[str] = None,
        stop_after_stuck_turns: int = 0,
        followup_fn: Optional[Callable[[Trajectory, int], Any]] = None,
    ):
        super().__init__()
        if template is None:
            raise ValueError('MultiTurnRollout requires a local Template instance')
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f'max_trajectory_tokens must be >= 1 or None, got '
                             f'{max_trajectory_tokens}')
        self._init_common(
            max_turns=max_turns,
            sampling_params=sampling_params,
            trace_dir=trace_dir,
            trace_callback=trace_callback,
            success_callback=success_callback)
        self.sampler = sampler
        self.template = template
        self.tool_manager = tool_manager
        self.harness = harness
        # A LoRA directory on disk, forwarded to every sample call. Training syncs
        # its adapter into the sampler directly, but evaluating a saved one has no
        # such channel: without this, an eval script would silently measure the
        # base model and report it as the trained one.
        self.adapter_path = adapter_path
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
        if isinstance(trajectories, dict):
            raise TypeError('MultiTurnRollout.__call__ expects a List[Trajectory]; '
                            'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        n = len(trajectories)
        if n == 0:
            return []

        sampling_params = kwargs.get('sampling_params', self.sampling_params)
        adapter_path = kwargs.get('adapter_path', self.adapter_path)
        # Left out entirely when unset, so a sampler without LoRA enabled sees the
        # same call it always did.
        adapter_kwargs = {'adapter_path': adapter_path} if adapter_path else {}
        tool_managers = self._broadcast(
            kwargs.get('tool_manager', self.tool_manager), n, name='tool_manager', required=True)
        harnesses = self._broadcast(kwargs.get('harness', self.harness), n, name='harness')
        lives: List[Optional[Trajectory]] = [
            dict(trajectories[i]) if harnesses[i] is not None else None for i in range(n)
        ]
        for live in lives:
            if live is not None:
                live['messages'] = list(live.get('messages') or [])

        # 1. First before_generate happens *before* encode so memory/system
        #    injection is in the initial prefix (not a later rewrite).
        encode_trajs: List[Trajectory] = []
        for i, traj in enumerate(trajectories):
            h, live = harnesses[i], lives[i]
            if h is not None and live is not None:
                lives[i] = h.before_generate(live)
                live = lives[i]
                traj = dict(traj)
                traj['messages'] = list(live.get('messages') or [])
                if live.get('tools'):
                    traj['tools'] = list(live['tools'])
            encode_trajs.append(traj)

        pifs: List[Dict[str, Any]] = []
        for i, traj in enumerate(encode_trajs):
            pif = self.template.encode(traj, add_generation_prompt=True)
            pif = _to_plain(pif)
            pif.setdefault('messages', list(traj.get('messages', [])))
            pifs.append(pif)
            if lives[i] is not None:
                lives[i]['messages'] = list(pifs[i].get('messages') or [])

        all_logprobs: List[List[Any]] = [[] for _ in range(n)]
        stop_reasons: List[Optional[str]] = [None] * n
        turns: List[int] = [0] * n
        truncated: List[bool] = [False] * n
        done: List[bool] = [False] * n
        # Consecutive turns that made no progress, the calls already issued in
        # each episode, and whether being stuck is what ended it. All three stay
        # at their initial value when ``stop_after_stuck_turns`` is 0.
        stuck_turns: List[int] = [0] * n
        seen_calls: List[set] = [set() for _ in range(n)]
        stuck_stop: List[bool] = [False] * n
        # Follow-up bookkeeping (all no-ops when ``followup_fn`` is None):
        # how many follow-ups each trajectory has had, and the params its next
        # turn should use. A trajectory that has had one stops dispatching tools.
        followups: List[int] = [0] * n
        params_for: List[Any] = [sampling_params] * n
        followup_fn = kwargs.get('followup_fn', self.followup_fn)
        # Why the tool-calling part of each episode ended, when it was not the
        # model's own choice: 'max_turns' or 'stuck'. Reported separately from
        # ``truncated`` because an episode can hit the turn limit and still go on
        # to answer the follow-up stages, in which case nothing was cut off.
        tool_stop: List[Optional[str]] = [None] * n

        def append_followup(global_idx: int) -> bool:
            """Ask for one more stage; True when the episode carries on.

            Sets ``truncated`` itself in the one case where the answer is "there
            is no room for another stage", which is a cut trajectory rather than
            a caller that had nothing more to ask.
            """
            nonlocal iterations
            if followup_fn is None:
                return False
            followup = followup_fn(
                self._as_trajectory(trajectories[global_idx], pifs[global_idx],
                                    all_logprobs[global_idx], turns[global_idx],
                                    stop_reasons[global_idx], truncated[global_idx]),
                followups[global_idx])
            if followup is None:
                return False
            text, next_params = followup if isinstance(followup, tuple) else (followup, None)
            extended = extend_with_bridge(
                pifs[global_idx], [{'role': 'user', 'content': text}], self.template)
            if extended is None:
                truncated[global_idx] = True
                return False
            pifs[global_idx] = extended
            if lives[global_idx] is not None:
                lives[global_idx]['messages'] = list(extended.get('messages') or [])
            followups[global_idx] += 1
            iterations += 1
            if next_params is not None:
                params_for[global_idx] = next_params
            return True

        # The loop counts generations, and each granted follow-up buys the one
        # extra generation it asked for. Paying for the follow-up stages out of
        # ``max_turns`` would mean an episode that spent its whole tool budget
        # never reaches the stages that read what it built, and a short one
        # silently gets more tool turns than a long one.
        iterations = self.max_turns
        done_iterations = 0
        first_turn = True
        while done_iterations < iterations:
            done_iterations += 1
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            if not first_turn:
                for global_idx in active:
                    pifs[global_idx], lives[global_idx], dropped = self._harness_before_generate(
                        pifs[global_idx], lives[global_idx], harnesses[global_idx])
                    if dropped:
                        truncated[global_idx] = True
                        done[global_idx] = True
                active = [i for i in range(n) if not done[i]]
                if not active:
                    break
            first_turn = False

            # 2. One batched sample call per distinct SamplingParams among the
            #    live trajectories -- normally exactly one, since only a
            #    follow-up stage asks for its own budget. Grouping rather than
            #    taking the first is what keeps a mixed batch honest: sampling one
            #    trajectory under another's token limit would silently truncate or
            #    over-spend, and the two are indistinguishable afterwards.
            groups: List[List[int]] = []
            group_params: List[Any] = []
            for global_idx in active:
                for slot, params in enumerate(group_params):
                    if params is params_for[global_idx]:
                        groups[slot].append(global_idx)
                        break
                else:
                    group_params.append(params_for[global_idx])
                    groups.append([global_idx])

            resps_by_idx: Dict[int, Any] = {}
            device_mesh = getattr(self.sampler, 'device_mesh', None)
            min_batch_size = (device_mesh.data_world_size if device_mesh is not None else 1)
            for slot, group in enumerate(groups):
                batch_pifs = [pifs[i] for i in group]
                actual = len(batch_pifs)
                if actual < min_batch_size:
                    batch_pifs = batch_pifs + ([batch_pifs[-1]] * (min_batch_size - actual))
                group_resps = self.sampler.sample(batch_pifs,
                                                  sampling_params=group_params[slot],
                                                  **adapter_kwargs)
                group_resps = self._unwrap_response_list(group_resps, len(batch_pifs))[:actual]
                for local_idx, global_idx in enumerate(group):
                    resps_by_idx[global_idx] = group_resps[local_idx]

            pending_tools: List[tuple] = []  # (global_idx, tool_calls)
            for global_idx in active:
                turns[global_idx] += 1
                seq = resps_by_idx[global_idx].sequences[0]

                if seq.new_input_feature is None or 'input_ids' not in seq.new_input_feature:
                    raise RuntimeError(f'Sampler returned a SampledSequence without '
                                       f'new_input_feature.input_ids for trajectory '
                                       f'{global_idx}; cannot continue multi-turn.')

                pifs[global_idx] = _to_plain(dict(seq.new_input_feature))
                if seq.logprobs is not None:
                    if len(seq.logprobs) != len(seq.tokens):
                        raise RuntimeError(f'logprobs length ({len(seq.logprobs)}) does not '
                                           f'match sampled token count ({len(seq.tokens)}) '
                                           f'at turn {turns[global_idx]} '
                                           f'(trajectory {global_idx})')
                    all_logprobs[global_idx].extend(seq.logprobs)
                stop_reasons[global_idx] = seq.stop_reason

                _msgs = pifs[global_idx].get('messages') or []
                _last_msg = _msgs[-1] if _msgs else None
                tool_calls = (_last_msg.get('tool_calls') if isinstance(_last_msg, dict) else None)
                if not tool_calls:
                    tool_calls = self.template.parse_tool_call(seq.decoded or '')
                # After a follow-up, a parsed call is not a call: the tools were
                # withdrawn for these stages on purpose (see ``followup_fn``), and
                # dispatching python that the model wrote as *an answer* would edit
                # the state the answer is about.
                if followups[global_idx]:
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
                    if _msgs and isinstance(_last_msg, dict):
                        # Decoded without the special tokens, the way the
                        # template writes a message: ``seq.decoded`` keeps the
                        # closing ``<|im_end|>``, and putting that in the content
                        # put it in the problem statements ex13 handed to solvers
                        # -- 7 of 7 of them ended in a literal '<|im_end|>'.
                        tok = getattr(self.template, 'tokenizer', None)
                        if tok is not None and seq.tokens:
                            _last_msg['content'] = tok.decode(
                                seq.tokens, skip_special_tokens=True)
                        else:
                            _last_msg['content'] = seq.decoded or ''
                        _last_msg.pop('tool_calls', None)

                if lives[global_idx] is not None:
                    lives[global_idx]['messages'] = list(_msgs)
                if harnesses[global_idx] is not None and lives[global_idx] is not None:
                    lives[global_idx] = harnesses[global_idx].after_generate(
                        lives[global_idx], seq.decoded or '', tool_calls or [])
                    self._merge_assistant_metadata(pifs[global_idx], lives[global_idx])

                # 3. Termination conditions
                # A reply cut off at ``max_tokens`` is truncated in exactly the
                # sense the flag names, and consumers read the flag to tell a
                # trajectory that finished from one that ran out of room: a
                # difficulty measurement counting such an attempt as a genuine
                # failure blames the task for the token budget. Tool calls the
                # cut reply happens to contain are still not dispatched -- the
                # turn never got to decide it was done emitting them.
                if seq.stop_reason == 'length':
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                # 3a. Sequence-length cap.
                if (self.max_trajectory_tokens is not None
                        and len(pifs[global_idx].get('input_ids') or []) >= self.max_trajectory_tokens):
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                if not tool_calls:
                    # The episode is over as far as the model is concerned. Give
                    # the caller one chance to say otherwise -- see
                    # ``followup_fn`` for why this is not a second rollout.
                    if append_followup(global_idx):
                        continue
                    done[global_idx] = True
                    continue

                if turns[global_idx] >= self.max_turns:
                    # Out of tool turns, not out of episode: the stages that read
                    # the end state can still run on what was built.
                    tool_stop[global_idx] = 'max_turns'
                    if append_followup(global_idx):
                        continue
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                pending_tools.append((global_idx, list(tool_calls)))

            # 4. Parallel tool dispatch across the live batch, then harness
            #    formats observations into tool messages (append-only bridge).
            #    The bridge itself is computed serially: it is a cheap
            #    decode-diff-encode on python strings / token lists.
            if pending_tools:
                obs_by_traj = self._dispatch_tools(tool_managers, pending_tools)
                for global_idx, tool_calls in pending_tools:
                    observations = obs_by_traj.get(global_idx) or [''] * len(tool_calls)
                    if self.stop_after_stuck_turns:
                        keys = [_call_key(tc) for tc in tool_calls]
                        all_repeats = bool(keys) and all(k in seen_calls[global_idx]
                                                        for k in keys)
                        seen_calls[global_idx].update(keys)
                        all_errors = bool(observations) and all(
                            is_error_observation(o) for o in observations)
                        if all_errors or all_repeats:
                            stuck_turns[global_idx] += 1
                        else:
                            stuck_turns[global_idx] = 0
                    tool_messages, lives[global_idx] = self._tool_messages_after(
                        pifs[global_idx], lives[global_idx], harnesses[global_idx],
                        observations, tool_calls)
                    extended = extend_with_bridge(pifs[global_idx], tool_messages, self.template)
                    if extended is None:
                        # Trajectory exceeded max_length, mark as done (deleted)
                        truncated[global_idx] = True
                        done[global_idx] = True
                    else:
                        pifs[global_idx] = extended
                        if lives[global_idx] is not None:
                            lives[global_idx]['messages'] = list(extended.get('messages') or [])
                    # Checked after the messages are appended, so the turns that
                    # ended the episode are in the trajectory the caller reads.
                    if (self.stop_after_stuck_turns
                            and stuck_turns[global_idx] >= self.stop_after_stuck_turns):
                        stuck_stop[global_idx] = True
                        tool_stop[global_idx] = 'stuck'
                        # Same as the turn limit: the tool phase is over, the
                        # state it left is not, so the stages still get their turn.
                        if not done[global_idx] and append_followup(global_idx):
                            continue
                        truncated[global_idx] = True
                        done[global_idx] = True

        for i in range(n):
            if not all_logprobs[i]:
                continue
            labels_i = pifs[i].get('labels') or []
            trainable_i = sum(1 for label in labels_i if label != -100)
            if len(all_logprobs[i]) != trainable_i:
                raise RuntimeError(f'logprobs/labels misaligned for trajectory {i}: '
                                   f'{len(all_logprobs[i])} logprobs vs {trainable_i} '
                                   f'trainable labels (labels != -100). This invariant is '
                                   f'required by grpo._pad_and_align_to_batch; a mismatch '
                                   f'would silently corrupt GRPO old_logps alignment.')

        # 5. Merge pif fields into each trajectory dict at TOP LEVEL so
        #    downstream consumers (VLLMSampler with ``'input_ids' in inputs``)
        #    see an encoded InputFeature and skip re-encoding.
        outs: List[Trajectory] = []
        for i, traj in enumerate(trajectories):
            out = dict(traj)
            out.update(pifs[i])
            out['messages'] = list(pifs[i].get('messages') or out.get('messages', []))
            out['logprobs'] = all_logprobs[i] if all_logprobs[i] else None
            out['turns'] = turns[i]
            out['stop_reason'] = stop_reasons[i]
            out['truncated'] = truncated[i]
            # ``truncated`` says something was cut off; these two say what ended
            # the tool-calling part, which is a different question -- an episode
            # can run out of turns, be handed a follow-up stage, and finish it.
            out['stuck_stop'] = stuck_stop[i]
            out['tool_stop'] = tool_stop[i]
            out['followups'] = followups[i]
            outs.append(out)

        # Per-rollout trace dump: one JSON file per selected trajectory.
        # ``trace_callback`` decides whether to store; ``success_callback``
        # decides the filename prefix. Observability only -- any failure
        # is swallowed inside ``_write_rollout_traces``.
        if self.trace_dir:
            self._write_rollout_traces(outs, global_step=kwargs.get('global_step'))
        return outs

    # ------------------------------------------------------------------ private

    @staticmethod
    def _as_trajectory(traj: Trajectory, pif: Dict[str, Any], logprobs: List[Any],
                      turns: int, stop_reason: Optional[str],
                      truncated: bool) -> Trajectory:
        """The episode so far, shaped like the value ``__call__`` returns.

        Handed to ``followup_fn`` so the callback reads an episode the same way
        every other consumer does -- ``messages`` complete, token fields present --
        rather than having to know this loop's local variables.
        """
        out = dict(traj)
        out.update(pif)
        out['messages'] = list(pif.get('messages') or traj.get('messages') or [])
        out['logprobs'] = logprobs if logprobs else None
        out['turns'] = turns
        out['stop_reason'] = stop_reason
        out['truncated'] = truncated
        return out

    def _harness_before_generate(
        self,
        pif: Dict[str, Any],
        live: Optional[Trajectory],
        harness: Optional[AgentHarness],
    ) -> Tuple[Dict[str, Any], Optional[Trajectory], bool]:
        """Run before_generate; bridge append-only deltas. ``dropped`` if encode fails."""
        if harness is None or live is None:
            return pif, live, False
        live['messages'] = list(pif.get('messages') or [])
        live = harness.before_generate(live)
        delta = _append_only_delta(pif.get('messages') or [], live.get('messages') or [])
        if not delta:
            return pif, live, False
        extended = extend_with_bridge(pif, delta, self.template)
        if extended is None:
            return pif, live, True
        live['messages'] = list(extended.get('messages') or [])
        return extended, live, False

    @staticmethod
    def _merge_assistant_metadata(pif: Dict[str, Any], live: Trajectory) -> None:
        """Copy tool_calls / reasoning onto the sampled assistant message.

        Content is left untouched so the token-id chain stays valid.
        """
        pif_msgs = pif.get('messages') or []
        if not pif_msgs or pif_msgs[-1].get('role') != 'assistant':
            return
        last_asst = None
        for m in reversed(live.get('messages') or []):
            if m.get('role') == 'assistant':
                last_asst = m
                break
        if last_asst is None:
            return
        dst = pif_msgs[-1]
        for key in ('tool_calls', 'reasoning_content', 'name'):
            if last_asst.get(key) and not dst.get(key):
                dst[key] = last_asst[key]

    def _dispatch_tools(
        self,
        tool_managers: List[ToolManager],
        pending: List[Tuple[int, List[Dict[str, Any]]]],
    ) -> Dict[int, List[str]]:
        """Run tool calls for the live batch, grouped by ToolManager.

        Trajectories that share a manager (and therefore often one Env) go
        through ``call_many`` / ``Env.step_batch``. Distinct managers run
        concurrently so remote sandboxes are not serialized on generate.
        """
        obs: Dict[int, List[str]] = {
            gi: [''] * len(tcs) for gi, tcs in pending
        }
        groups: Dict[int, List[Tuple[int, int, Dict[str, Any]]]] = defaultdict(list)
        mgr_by_id: Dict[int, ToolManager] = {}
        for gi, tcs in pending:
            mid = id(tool_managers[gi])
            mgr_by_id[mid] = tool_managers[gi]
            for ci, tc in enumerate(tcs):
                groups[mid].append((gi, ci, tc))

        def _run_group(items: List[Tuple[int, int, Dict[str, Any]]], mgr: ToolManager):
            tcs = [tc for _, _, tc in items]
            if hasattr(mgr, 'call_many'):
                contents = mgr.call_many(tcs)
            else:
                contents = [mgr(tc) for tc in tcs]
            return list(zip(items, contents))

        group_items = list(groups.items())
        if len(group_items) == 1:
            mid, items = group_items[0]
            finished = [_run_group(items, mgr_by_id[mid])]
        else:
            finished = []
            with ThreadPoolExecutor(max_workers=min(32, len(group_items))) as pool:
                futs = [
                    pool.submit(_run_group, items, mgr_by_id[mid])
                    for mid, items in group_items
                ]
                for fut in as_completed(futs):
                    finished.append(fut.result())

        for group_result in finished:
            for (gi, ci, _tc), content in group_result:
                obs[gi][ci] = '' if content is None else str(content)
        return obs

    def _tool_messages_after(
        self,
        pif: Dict[str, Any],
        live: Optional[Trajectory],
        harness: Optional[AgentHarness],
        observations: List[str],
        tool_calls: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[Trajectory]]:
        fallback = _default_tool_messages(tool_calls, observations)
        if harness is None or live is None:
            return fallback, live
        old = list(pif.get('messages') or [])
        live['messages'] = list(old)
        live = harness.after_tools(live, observations, tool_calls)
        delta = _append_only_delta(old, live.get('messages') or [])
        if not delta:
            return fallback, live
        return delta, live

    @staticmethod
    def _unwrap_response_list(resps, expected: int) -> List[SampleResponse]:
        """Validate that the sampler returned ``expected`` ``SampleResponse``s,
        one per input in the batch.
        """
        if not isinstance(resps, list):
            raise TypeError(f'expected List[SampleResponse] from sampler.sample (batched '
                            f'call), got {type(resps).__name__}')
        if len(resps) != expected:
            raise RuntimeError(f'sampler returned {len(resps)} responses for a batch of '
                               f'{expected} trajectories; expected one per input.')
        for i, r in enumerate(resps):
            if not isinstance(r, SampleResponse):
                raise TypeError(f'expected SampleResponse at batch index {i}, got '
                                f'{type(r).__name__}')
            if not r.sequences:
                raise RuntimeError(f'SampleResponse at batch index {i} has no sequences')
        return resps
