# Copyright (c) ModelScope Contributors. All rights reserved.
import dataclasses
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

import json

from twinkle.data_format import Trajectory, user_data_get
from twinkle.data_format.sampling import SampleResponse, SamplingParams
from twinkle.infra import remote_class, remote_function
from twinkle.template.base import Template
from twinkle_agentic.tools.tool_manager import ToolManager
from .base import Rollout
from .bridge import _to_plain, extend_with_bridge


class TurnController(Protocol):
    """Boundary contract for framework-specific multi-turn scheduling.

    Twinkle owns batching, token budgets, termination and response aggregation. A controller only translates
    between a framework's mutable request state and Twinkle sampler inputs/turn updates.
    """

    def create_state(self, trajectory: Trajectory) -> Any: ...

    def sampler_input(self, state: Any) -> Any: ...

    def start(self, states: List[Any]) -> None: ...

    def on_turn(self, states: List[Any], sequences: List[Any], current_turn: int,
                forced_stops: List[bool]) -> List[Dict[str, Any]]: ...

    def finalize(self, state: Any, response_token_ids: List[List[int]],
                 response_loss_mask: List[List[int]]) -> Dict[str, Any]: ...

    def close(self, states: List[Any]) -> None: ...


@remote_class()
class MultiTurnRollout(Rollout):
    """Agentic multi-turn rollout with tool use (batched).

    Contract (matches :class:`Rollout`): accepts a ``List[Trajectory]`` and
    returns a ``List[Trajectory]`` of the same length, in the same order.
    Every turn issues a SINGLE batched ``sampler.sample(active_pifs)`` call
    so vLLM can run all live trajectories in parallel; finished trajectories
    are parked and excluded from subsequent batches.

    Per-trajectory loop:
        1. Encode the initial trajectory into an ``InputFeature`` with a
           generation prompt at the tail.
        2. Call ``sampler.sample(pifs)`` (batched). The sampler internally
           invokes ``template.concat_input_feature`` to append the freshly
           sampled assistant tokens; we pick up ``seq.new_input_feature`` as
           the new running ``pif``.
        3. If ``stop_reason == 'length'`` or the decoded assistant output has
           no tool calls, mark the trajectory as done.
        4. Otherwise, invoke the tools via ``ToolManager`` and append each
           tool response as a ``{'role':'tool', 'content': ...}`` message.
           Compute "bridge" tokens (tool turns + next ``<|im_start|>assistant``
           header) with ``labels = -100`` and extend the pif.
        5. Repeat until all trajectories are done or ``max_turns`` is hit.

    Per-call overrides via ``**kwargs``:
        * ``sampling_params``: shared :class:`SamplingParams` for the batch.
        * ``tool_manager``: either a single :class:`ToolManager` (applied to
          every trajectory) or a list of ``ToolManager`` aligned 1:1 with
          ``trajectories`` (used by :class:`MultiTurnCondenseRollout` to
          attach a trajectory-bound ``ExtractCondensed``).

    The class intentionally has no knowledge of condensers/chunkers; they are
    applied upstream (on the trajectory before rollout) or downstream
    (on the returned messages).
    """

    def __init__(
        self,
        sampler,
        template: Template,
        tool_manager: Optional[ToolManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: Optional[int] = 6,
        max_trajectory_tokens: Optional[int] = None,
        trace_dir: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        turn_controller: Optional[TurnController] = None,
        completion_length_limit_scope: str = 'per_round',
    ):
        super().__init__()
        if template is None:
            raise ValueError('MultiTurnRollout requires a local Template instance')
        if max_turns is not None and max_turns < 1:
            raise ValueError(f'max_turns must be >= 1 or None, got {max_turns}')
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f'max_trajectory_tokens must be >= 1 or None, got '
                             f'{max_trajectory_tokens}')
        self.sampler = sampler
        self.template = template
        self.tool_manager = tool_manager
        self.sampling_params = sampling_params or SamplingParams()
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.trace_dir = trace_dir
        self.trace_callback = trace_callback
        self.success_callback = success_callback
        self.turn_controller = turn_controller
        if completion_length_limit_scope not in ('total', 'per_round'):
            raise ValueError("completion_length_limit_scope must be 'total' or 'per_round'.")
        self.completion_length_limit_scope = completion_length_limit_scope
        if self.trace_dir:
            os.makedirs(self.trace_dir, exist_ok=True)

        if self.sampling_params.num_samples != 1:
            raise ValueError(f'MultiTurnRollout currently supports num_samples=1 only, '
                             f'got {self.sampling_params.num_samples}')
        if self.turn_controller is None and self.max_turns is None:
            raise ValueError('max_turns cannot be None for tool-managed multi-turn rollout.')
        if self.turn_controller is None:
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
        if self.turn_controller is not None:
            return self._run_controlled(trajectories, sampling_params)
        tool_managers = self._resolve_tool_managers(kwargs.get('tool_manager', self.tool_manager), n)

        # 1. Encode each trajectory once; ``pifs[i]`` is the live per-turn
        #    state for trajectory ``i``.
        pifs: List[Dict[str, Any]] = []
        for traj in trajectories:
            pif = self.template.encode(traj, add_generation_prompt=True)
            pif = _to_plain(pif)
            pif.setdefault('messages', list(traj.get('messages', [])))
            pifs.append(pif)

        all_logprobs: List[List[Any]] = [[] for _ in range(n)]
        stop_reasons: List[Optional[str]] = [None] * n
        turns: List[int] = [0] * n
        truncated: List[bool] = [False] * n
        done: List[bool] = [False] * n

        for _ in range(self.max_turns):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            # 2. One batched sample call for all currently-live trajectories.
            batch_pifs = [pifs[i] for i in active]
            actual = len(batch_pifs)
            device_mesh = getattr(self.sampler, 'device_mesh', None)
            min_batch_size = (device_mesh.data_world_size if device_mesh is not None else 1)
            if actual < min_batch_size:
                batch_pifs = batch_pifs + ([batch_pifs[-1]] * (min_batch_size - actual))
            resps = self.sampler.sample(batch_pifs, sampling_params=sampling_params)
            resps = self._unwrap_response_list(resps, len(batch_pifs))[:actual]

            pending_bridges: List[tuple] = []  # (global_idx, tool_messages)
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

                _msgs = pifs[global_idx].get('messages') or []
                _last_msg = _msgs[-1] if _msgs else None
                tool_calls = (_last_msg.get('tool_calls') if isinstance(_last_msg, dict) else None)
                if not tool_calls:
                    tool_calls = self.template.parse_tool_call(seq.decoded or '')
                if not tool_calls:
                    done[global_idx] = True
                    continue

                if turns[global_idx] >= self.max_turns:
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                # 4. Dispatch tools per trajectory (uses this trajectory's
                #    tool_manager, which may be a trajectory-bound clone).
                tool_messages = [{
                    'role': 'tool',
                    'content': tool_managers[global_idx](tc),
                } for tc in tool_calls]
                pending_bridges.append((global_idx, tool_messages))

            # Extend pif with bridge tokens for every trajectory that has
            # outstanding tool turns. Done serially: bridge computation is
            # a cheap decode-diff-encode on python strings / token lists.
            for global_idx, tool_messages in pending_bridges:
                extended = extend_with_bridge(pifs[global_idx], tool_messages, self.template)
                if extended is None:
                    # Trajectory exceeded max_length, mark as done (deleted)
                    truncated[global_idx] = True
                    done[global_idx] = True
                else:
                    pifs[global_idx] = extended

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

    @staticmethod
    def _controlled_params(sampling_params, max_tokens: Optional[int]) -> SamplingParams:
        """Clone sampling parameters for one controller-driven batch."""
        if isinstance(sampling_params, dict):
            values = dict(sampling_params)
            values.update(max_tokens=max_tokens, num_samples=1)
            return SamplingParams.from_dict(values)
        return dataclasses.replace(sampling_params, max_tokens=max_tokens, num_samples=1)

    def _sample_controlled(self, states: List[Any], active: List[int], sampling_params) -> Dict[int, Any]:
        """Sample every live controller state, batching equal remaining token budgets."""
        base_limit = (sampling_params.get('max_tokens')
                      if isinstance(sampling_params, dict) else sampling_params.max_tokens)
        grouped: Dict[Optional[int], List[int]] = {}
        for index in active:
            max_tokens = base_limit
            if self.completion_length_limit_scope == 'total' and base_limit is not None:
                max_tokens = max(0, int(base_limit) - states[index]['generated_tokens'])
            if max_tokens == 0:
                raise RuntimeError('multi-turn trajectory remained active after exhausting its total token budget.')
            grouped.setdefault(max_tokens, []).append(index)

        sampled: Dict[int, Any] = {}
        for max_tokens, indices in grouped.items():
            params = self._controlled_params(sampling_params, max_tokens)
            inputs = [self.turn_controller.sampler_input(states[index]['value']) for index in indices]
            responses = self.sampler.sample(inputs, sampling_params=params)
            if len(responses) != len(indices):
                raise RuntimeError(f'multi-turn sampler returned {len(responses)} responses for '
                                   f'{len(indices)} inputs.')
            for index, response in zip(indices, responses, strict=True):
                if len(response.sequences) != 1:
                    raise RuntimeError('multi-turn sampling requires exactly one sequence per active trajectory.')
                sampled[index] = response.sequences[0]
        return sampled

    @staticmethod
    def _record_controlled_turn(state: Dict[str, Any], update: Dict[str, Any]) -> None:
        token_ids = list(update['response_token_ids'])
        loss_mask = list(update.get('response_loss_mask', [1] * len(token_ids)))
        logprobs = list(update.get('rollout_logprobs') or [])
        if len(token_ids) != len(loss_mask):
            raise RuntimeError(f'multi-turn response has {len(token_ids)} token IDs but '
                               f'{len(loss_mask)} loss masks.')
        trainable = sum(int(bool(value)) for value in loss_mask)
        if len(logprobs) != trainable:
            raise RuntimeError(f'multi-turn rollout logprobs misaligned: {len(logprobs)} logprobs for '
                               f'{trainable} trainable tokens.')
        if update.get('continuation') and state['response_token_ids']:
            state['response_token_ids'][-1].extend(token_ids)
            state['response_loss_mask'][-1].extend(loss_mask)
            state['rollout_logprobs'][-1].extend(logprobs)
        else:
            state['response_token_ids'].append(token_ids)
            state['response_loss_mask'].append(loss_mask)
            state['rollout_logprobs'].append(logprobs)

    def _controlled_stops(self, states: List[Dict[str, Any]], active: List[int], sampled: Dict[int, Any],
                          current_turn: int, sampling_params) -> List[bool]:
        base_limit = (sampling_params.get('max_tokens')
                      if isinstance(sampling_params, dict) else sampling_params.max_tokens)
        stops = []
        for index in active:
            sequence = sampled[index]
            states[index]['generated_tokens'] += len(sequence.tokens or [])
            total_stop = (self.completion_length_limit_scope == 'total' and base_limit is not None
                          and states[index]['generated_tokens'] >= int(base_limit))
            turn_stop = self.max_turns is not None and current_turn >= self.max_turns
            states[index]['truncated'] |= sequence.stop_reason == 'length' or total_stop or turn_stop
            stops.append(total_stop or turn_stop)
        return stops

    def _finalize_controlled(self, trajectories: List[Trajectory], states: List[Dict[str, Any]]) -> List[Trajectory]:
        outs: List[Trajectory] = []
        for trajectory, state in zip(trajectories, states, strict=True):
            finalized = self.turn_controller.finalize(
                state['value'], state['response_token_ids'], state['response_loss_mask'])
            encoded = finalized.get('encoded')
            labels = encoded.get('labels') if isinstance(encoded, dict) else None
            flat_logprobs = [value for turn in state['rollout_logprobs'] for value in turn]
            if labels is not None and len(flat_logprobs) != sum(int(label) != -100 for label in labels):
                raise RuntimeError(f'multi-turn training feature has '
                                   f'{sum(int(label) != -100 for label in labels)} trainable labels but '
                                   f'{len(flat_logprobs)} rollout logprobs.')
            out = dict(trajectory)
            out.update(finalized)
            out.update({
                'response_token_ids': state['response_token_ids'],
                'response_loss_mask': state['response_loss_mask'],
                'rollout_logprobs': state['rollout_logprobs'],
                'logprobs': flat_logprobs,
                'rollout_infos': state['rollout_infos'],
                'turns': state['turns'],
                'stop_reason': state['stop_reason'],
                'truncated': state['truncated'],
            })
            outs.append(out)
        return outs

    def _run_controlled(self, trajectories: List[Trajectory], sampling_params) -> List[Trajectory]:
        """Run the shared batched state machine with a framework-owned turn controller.

        The controller translates framework-specific scheduler requests at the boundary; Twinkle owns
        batching, token budgets, per-turn alignment and output aggregation.
        """
        states = [{
            'value': self.turn_controller.create_state(trajectory),
            'response_token_ids': [],
            'response_loss_mask': [],
            'rollout_logprobs': [],
            'rollout_infos': {},
            'generated_tokens': 0,
            'turns': 0,
            'stop_reason': None,
            'truncated': False,
        } for trajectory in trajectories]
        active = list(range(len(states)))
        current_turn = 1
        try:
            self.turn_controller.start([state['value'] for state in states])
            while active:
                sampled = self._sample_controlled(states, active, sampling_params)
                forced_stops = self._controlled_stops(states, active, sampled, current_turn, sampling_params)
                updates = self.turn_controller.on_turn(
                    [states[index]['value'] for index in active],
                    [sampled[index] for index in active],
                    current_turn,
                    forced_stops,
                )
                if len(updates) != len(active):
                    raise RuntimeError(f'multi-turn controller returned {len(updates)} updates for '
                                       f'{len(active)} active trajectories.')
                next_active = []
                for index, sequence, forced_stop, update in zip(
                        active, (sampled[i] for i in active), forced_stops, updates, strict=True):
                    self._record_controlled_turn(states[index], update)
                    states[index]['rollout_infos'].update(update.get('rollout_infos') or {})
                    states[index]['turns'] = current_turn
                    states[index]['stop_reason'] = update.get('stop_reason', sequence.stop_reason)
                    states[index]['truncated'] |= bool(update.get('truncated'))
                    done = forced_stop or bool(update.get('done'))
                    if not done:
                        if not sequence.tokens and self.max_turns is None:
                            raise RuntimeError('multi-turn controller kept an empty generation active without '
                                               'max_turns; the trajectory cannot make progress.')
                        next_active.append(index)
                    else:
                        states[index]['rollout_infos']['num_turns'] = current_turn
                active = next_active
                current_turn += 1
            outs = self._finalize_controlled(trajectories, states)
            if self.trace_dir:
                self._write_rollout_traces(outs)
            return outs
        finally:
            self.turn_controller.close([state['value'] for state in states])

    # ------------------------------------------------------------------ private

    @staticmethod
    def _resolve_tool_managers(arg, n: int) -> List[ToolManager]:
        """Broadcast a single ``ToolManager`` or validate a per-trajectory list."""
        if arg is None:
            raise ValueError('tool_manager is required but was not provided. '
                             'Pass it at construction time or as a per-call kwarg.')
        if isinstance(arg, list):
            if len(arg) != n:
                raise ValueError(f'per-call tool_manager list length ({len(arg)}) does '
                                 f'not match number of trajectories ({n})')
            return list(arg)
        return [arg] * n

    _TRACE_SKIP_KEYS = (
        'input_ids',
        'labels',
        'attention_mask',
        'position_ids',
        'logprobs',
        'pixel_values',
        'image_grid_thw',
        'mm_token_type_ids',
    )

    @classmethod
    def _serialize_for_trace(cls, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Drop tensor-like / oversized fields; keep messages + metadata.

        Trace files are for human forensics; raw token ids, labels and
        image buffers would bloat the file by orders of magnitude without
        adding diagnostic value (the chat-template rendering of
        ``messages`` already captures the textual content).
        """
        slim = {k: v for k, v in traj.items() if k not in cls._TRACE_SKIP_KEYS}
        return _to_plain(slim)

    @staticmethod
    def _extract_ground_truth(traj: Dict[str, Any]) -> str:
        """Pull ``ground_truth`` out of packed ``user_data``."""
        return user_data_get(traj.get('user_data'), 'ground_truth', '') or ''

    @staticmethod
    def _resolve_traj_id(traj: Dict[str, Any], fallback_idx: int) -> str:
        """Stable-ish trajectory id for filenames.

        Prefers an explicit ``id`` / ``prompt_id`` key in ``user_data``
        (sanitised for filesystem safety); else falls back to
        ``{timestamp_ms}-{fallback_idx}`` so concurrent rollouts do not
        overwrite each other's files.
        """
        for key in ('id', 'prompt_id'):
            val = user_data_get(traj.get('user_data'), key)
            if val not in (None, ''):
                safe = re.sub(r'[^A-Za-z0-9_\-.]+', '_', str(val))[:64]
                if safe:
                    return safe
        return f'{int(time.time() * 1000)}-{fallback_idx}'

    def _build_trace_record(
        self,
        traj: Dict[str, Any],
        *,
        idx: int,
        success: bool,
    ) -> Dict[str, Any]:
        """Assemble one trace record. Subclasses override to add fields.

        ``idx`` is the trajectory's position in the rollout output list,
        so subclasses can correlate the record with any per-call state
        they stashed on ``self`` during ``__call__``.
        """
        return {
            'trajectory': self._serialize_for_trace(traj),
            'ground_truth': self._extract_ground_truth(traj),
            'stop_reason': traj.get('stop_reason'),
            'truncated': bool(traj.get('truncated')),
            'success': success,
        }

    def _write_rollout_traces(
        self,
        outs: List[Dict[str, Any]],
        *,
        global_step: Optional[int] = None,
    ) -> None:
        """Dump one pretty-printed JSON file per selected trajectory.

        ``trace_callback`` (if set) decides WHETHER to store;
        ``success_callback`` (if set) decides the filename prefix
        (``ok-`` vs ``fail-``). Defaults: store-all / mark-fail.

        Observability must never break training -- any I/O or encoding
        problem on a single trajectory is swallowed so the remaining
        dumps and the optimisation loop continue unaffected.
        """
        if not self.trace_dir:
            return
        for idx, traj in enumerate(outs):
            try:
                should_store = True
                if self.trace_callback is not None:
                    try:
                        should_store = bool(self.trace_callback(traj))
                    except Exception:
                        should_store = False
                if not should_store:
                    continue

                success = False
                if self.success_callback is not None:
                    try:
                        success = bool(self.success_callback(traj))
                    except Exception:
                        success = False

                record = self._build_trace_record(traj, idx=idx, success=success)
                prefix = 'ok' if success else 'fail'
                # global_step prefix lets file listings sort by training step.
                step_tag = f'step{int(global_step):06d}-' if global_step is not None else ''
                fname = f'{step_tag}{prefix}-{self._resolve_traj_id(traj, idx)}.json'
                path = os.path.join(self.trace_dir, fname)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                # Per-trajectory failure never aborts the loop.
                pass

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
