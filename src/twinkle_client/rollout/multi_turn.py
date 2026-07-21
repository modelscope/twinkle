# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side multi-turn agentic rollout orchestration.

This module hosts :class:`ClientMultiTurnRollout`, a hand-maintained multi-turn
rollout orchestrator whose algorithmic structure mirrors
``twinkle_agentic.rollout.multi_turn.MultiTurnRollout`` but issues sampling over
HTTP via ``twinkle_client.sampler.vLLMSampler.sample()`` instead of holding a
Ray actor handle.

Design notes:
    * It deliberately does NOT subclass ``MultiTurnRollout``. That class is
      decorated with ``@remote_class`` / ``@remote_function`` for Ray remote
      dispatch and assumes ``self.sampler`` is a Ray actor handle, which does
      not match the HTTP-client semantics here.
    * The ``tool_manager`` type is reused directly from
      ``twinkle_agentic.tools.tool_manager.ToolManager`` (imported, not copied).
    * Bridge-token stitching is reused from
      ``twinkle_agentic.rollout.bridge.extend_with_bridge``.
"""
import dataclasses
from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template
from twinkle_agentic.rollout.bridge import extend_with_bridge
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.sampler import vLLMSampler


class ClientMultiTurnRollout:
    """Agentic multi-turn rollout with tool use, driven over HTTP.

    Mirrors the per-trajectory state machine of
    ``twinkle_agentic.rollout.multi_turn.MultiTurnRollout`` but issues sampling
    via ``vLLMSampler.sample()`` (an HTTP call to ``/twinkle/sample``) rather
    than a Ray actor call.
    """

    def __init__(
        self,
        sampler: vLLMSampler,
        template: Template,
        tool_manager: Optional[ToolManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        max_trajectory_tokens: Optional[int] = None,
    ):
        # Validation aligned with MultiTurnRollout.__init__.
        if template is None:
            raise ValueError('ClientMultiTurnRollout requires a local Template instance')
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f'max_trajectory_tokens must be >= 1 or None, got '
                             f'{max_trajectory_tokens}')

        self.sampler = sampler
        self.template = template
        self.tool_manager = tool_manager
        self.sampling_params = sampling_params or SamplingParams()
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens

        if self.sampling_params.num_samples != 1:
            raise ValueError(f'ClientMultiTurnRollout currently supports num_samples=1 only, '
                             f'got {self.sampling_params.num_samples}')

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        """Run the batched multi-turn rollout state machine over HTTP.

        Structurally mirrors ``MultiTurnRollout.__call__`` but issues each
        round's sampling through ``vLLMSampler.sample()`` (an HTTP POST to
        ``/twinkle/sample``) rather than a Ray actor call. Every round makes a
        SINGLE batched HTTP call for all currently-live trajectories so the
        sampler can run them in parallel; finished trajectories are parked and
        excluded from later batches.

        Returns a ``List[Trajectory]`` of the same length and order as the
        input, each augmented with ``messages`` / ``logprobs`` / ``turns`` /
        ``stop_reason`` / ``truncated`` fields.

        Exception handling and boundary truncation contract:
            * ``new_input_feature`` missing / lacking ``input_ids`` -> RuntimeError
              carrying both the batch index and the trajectory index.
            * per-round ``len(seq.logprobs) != len(seq.tokens)`` -> RuntimeError
              carrying the specific counts.
            * final per-trajectory ``len(all_logprobs[i]) != count(labels != -100)``
              -> RuntimeError (protects downstream GRPO old_logps alignment).
            * ``vLLMSampler.sample()`` network/timeout errors propagate unchanged
              (never wrapped or swallowed).
            * tool_calls produced with no ``tool_manager`` -> ValueError.
            * ``max_turns == 1`` with a first-round tool call -> the trajectory is
              marked ``truncated=True, stop_reason='max_turns'`` and sampling stops.
        """
        if isinstance(trajectories, dict):
            raise TypeError('ClientMultiTurnRollout.__call__ expects a List[Trajectory]; '
                            'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        n = len(trajectories)
        if n == 0:
            return []

        sampling_params = self._as_sampling_params_dict(
            kwargs.get('sampling_params', self.sampling_params))
        tool_managers = self._resolve_tool_managers(
            kwargs.get('tool_manager', self.tool_manager), n)

        # 1. Encode each trajectory once; ``pifs[i]`` is the live per-turn
        #    state for trajectory ``i``. ``vLLMSampler.sample`` is responsible for
        #    JSON-serialising the feature (ndarray / tensor -> list) before the
        #    HTTP POST, so no conversion is needed here.
        pifs: List[Dict[str, Any]] = []
        for traj in trajectories:
            pif = self.template.encode(traj, add_generation_prompt=True)
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

            # 2. One batched HTTP sample call for all currently-live
            #    trajectories. No device_mesh / min_batch_size padding: an HTTP
            #    client has no Ray DP ranks to align against.
            #
            #    Passthrough contract: ``vLLMSampler.sample()`` may raise
            #    network / timeout / HTTP errors (e.g. requests exceptions). We
            #    deliberately do NOT wrap this call in try/except -- such errors
            #    propagate unchanged to the caller so ret/backoff policy stays an
            #    upstream concern (retry/backoff) and failures are never
            #    silently swallowed.
            batch_pifs = [pifs[i] for i in active]
            resps = self.sampler.sample(batch_pifs, sampling_params=sampling_params)

            pending_bridges: List[tuple] = []  # (global_idx, tool_messages)
            for local_idx, global_idx in enumerate(active):
                turns[global_idx] += 1
                seq = resps[local_idx].sequences[0]

                # ``new_input_feature`` is the running pif for the next round;
                # the /twinkle/sample response contract guarantees it is set and
                # carries ``input_ids``. A missing feature makes the next round
                # impossible, so raise a batch/trajectory-indexed RuntimeError.
                if seq.new_input_feature is None or 'input_ids' not in seq.new_input_feature:
                    raise RuntimeError(
                        f'Sampler returned a sequence without new_input_feature.input_ids at '
                        f'batch index {local_idx} (trajectory {global_idx}); '
                        f'cannot continue multi-turn.')

                pifs[global_idx] = dict(seq.new_input_feature)
                # Per-round logprobs/token alignment guard: each sampled token
                # must carry exactly one logprob entry. Mirrors the core-lib
                # ``len(seq.logprobs) != len(seq.tokens)`` semantic so client and
                # Ray paths cannot drift on this invariant.
                if seq.logprobs is not None:
                    if len(seq.logprobs) != len(seq.tokens):
                        raise RuntimeError(
                            f'logprobs length ({len(seq.logprobs)}) does not match sampled '
                            f'token count ({len(seq.tokens)}) at turn {turns[global_idx]} '
                            f'(trajectory {global_idx})')
                    all_logprobs[global_idx].extend(seq.logprobs)
                stop_reasons[global_idx] = seq.stop_reason

                # 3. Termination conditions.
                if seq.stop_reason == 'length':
                    done[global_idx] = True
                    continue

                # 3a. Sequence-length cap.
                if (self.max_trajectory_tokens is not None and len(
                        pifs[global_idx].get('input_ids') or []) >= self.max_trajectory_tokens):
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                # 3b. Parse tool calls from the freshly sampled assistant turn.
                _msgs = pifs[global_idx].get('messages') or []
                _last_msg = _msgs[-1] if _msgs else None
                tool_calls = (_last_msg.get('tool_calls') if isinstance(_last_msg, dict) else None)
                if not tool_calls:
                    tool_calls = self.template.parse_tool_call(seq.decoded or '')
                if not tool_calls:
                    done[global_idx] = True
                    continue

                # 3c. Hit the turn cap while still wanting to call a tool: force
                #     truncation. Also covers the ``max_turns == 1`` edge, where
                #     the very first sampled turn trips this branch.
                if turns[global_idx] >= self.max_turns:
                    truncated[global_idx] = True
                    stop_reasons[global_idx] = 'max_turns'
                    done[global_idx] = True
                    continue

                # 4. Dispatch tools for this trajectory via its ToolManager.
                tool_manager = tool_managers[global_idx]
                if tool_manager is None:
                    raise ValueError(
                        f'trajectory {global_idx} produced tool_calls but no tool_manager '
                        f'was provided (at construction time or as a per-call kwarg).')
                tool_messages = [{
                    'role': 'tool',
                    'content': tool_manager(tc),
                } for tc in tool_calls]
                pending_bridges.append((global_idx, tool_messages))

            # Stitch bridge tokens (tool turns + next generation prompt) for
            # every trajectory with outstanding tool turns. Reuses the shared
            # pure function so client and core-lib paths cannot drift.
            for global_idx, tool_messages in pending_bridges:
                extended = extend_with_bridge(pifs[global_idx], tool_messages, self.template)
                if extended is None:
                    # Trajectory exceeded max_length (truncation strategy 'delete').
                    truncated[global_idx] = True
                    done[global_idx] = True
                else:
                    pifs[global_idx] = extended

        # 4b. Final logprobs/labels alignment guard. For every trajectory that
        #     collected logprobs, the total logprob count must equal the number
        #     of trainable positions (labels != -100) in the final pif. This is
        #     the same invariant grpo._pad_and_align_to_batch relies on; a
        #     mismatch would silently corrupt GRPO old_logps alignment, so we
        #     fail loudly with the specific numbers.
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

        # 5. Merge pif fields into each trajectory dict at TOP LEVEL, preserving
        #    input length and order.
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
        return outs

    # ------------------------------------------------------------------ private

    @staticmethod
    def _as_sampling_params_dict(sampling_params) -> Optional[Dict[str, Any]]:
        """Coerce ``sampling_params`` into the ``Optional[Dict]`` that
        ``vLLMSampler.sample()`` expects.

        ``self.sampling_params`` is a core-lib :class:`SamplingParams` dataclass,
        while the HTTP sampler wants a plain dict. A per-call kwarg override may
        be either a dataclass or an already-built dict.
        """
        if sampling_params is None:
            return None
        if isinstance(sampling_params, dict):
            return sampling_params
        if dataclasses.is_dataclass(sampling_params):
            return dataclasses.asdict(sampling_params)
        return sampling_params

    @staticmethod
    def _resolve_tool_managers(arg, n: int) -> List[Optional[ToolManager]]:
        """Broadcast a single ``ToolManager`` or validate a per-trajectory list.

        Unlike the core-lib rollout, ``None`` is tolerated here and broadcast as
        ``[None] * n``; the ValueError is raised lazily at the tool-dispatch site
        only when a trajectory actually produces tool_calls.
        """
        if isinstance(arg, list):
            if len(arg) != n:
                raise ValueError(f'per-call tool_manager list length ({len(arg)}) does '
                                 f'not match number of trajectories ({n})')
            return list(arg)
        return [arg] * n
