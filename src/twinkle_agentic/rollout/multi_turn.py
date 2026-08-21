# Copyright (c) ModelScope Contributors. All rights reserved.
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
        self.max_trajectory_tokens = max_trajectory_tokens
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

        first_turn = True
        for _ in range(self.max_turns):
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

            # 2. One batched sample call for all currently-live trajectories.
            batch_pifs = [pifs[i] for i in active]
            actual = len(batch_pifs)
            device_mesh = getattr(self.sampler, 'device_mesh', None)
            min_batch_size = (device_mesh.data_world_size if device_mesh is not None else 1)
            if actual < min_batch_size:
                batch_pifs = batch_pifs + ([batch_pifs[-1]] * (min_batch_size - actual))
            resps = self.sampler.sample(batch_pifs, sampling_params=sampling_params)
            resps = self._unwrap_response_list(resps, len(batch_pifs))[:actual]

            pending_tools: List[tuple] = []  # (global_idx, tool_calls)
            for local_idx, global_idx in enumerate(active):
                turns[global_idx] += 1
                seq = resps[local_idx].sequences[0]

                if seq.new_input_feature is None or 'input_ids' not in seq.new_input_feature:
                    raise RuntimeError(f'Sampler returned a SampledSequence without '
                                       f'new_input_feature.input_ids at batch index '
                                       f'{local_idx} (trajectory {global_idx}); '
                                       f'cannot continue multi-turn.')

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

                if lives[global_idx] is not None:
                    lives[global_idx]['messages'] = list(_msgs)
                if harnesses[global_idx] is not None and lives[global_idx] is not None:
                    lives[global_idx] = harnesses[global_idx].after_generate(
                        lives[global_idx], seq.decoded or '', tool_calls or [])
                    self._merge_assistant_metadata(pifs[global_idx], lives[global_idx])

                # 3. Termination conditions
                if seq.stop_reason == 'length':
                    done[global_idx] = True
                    continue

                # 3a. Sequence-length cap.
                if (self.max_trajectory_tokens is not None
                        and len(pifs[global_idx].get('input_ids') or []) >= self.max_trajectory_tokens):
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                if not tool_calls:
                    done[global_idx] = True
                    continue

                if turns[global_idx] >= self.max_turns:
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
            outs.append(out)

        # Per-rollout trace dump: one JSON file per selected trajectory.
        # ``trace_callback`` decides whether to store; ``success_callback``
        # decides the filename prefix. Observability only -- any failure
        # is swallowed inside ``_write_rollout_traces``.
        if self.trace_dir:
            self._write_rollout_traces(outs, global_step=kwargs.get('global_step'))
        return outs

    # ------------------------------------------------------------------ private

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
