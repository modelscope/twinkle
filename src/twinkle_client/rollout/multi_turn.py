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
      ``twinkle_agentic.utils.token_utils.extend_with_bridge``.
"""
import dataclasses
from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.utils.token_utils import extend_with_bridge
from twinkle_client.sampler import vLLMSampler


@dataclasses.dataclass
class _RolloutState:
    pifs: list[dict[str, Any]]
    all_logprobs: list[list[Any]]
    stop_reasons: list[str | None]
    turns: list[int]
    truncated: list[bool]
    done: list[bool]


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
            * ``stop_reason == 'length'`` -> the trajectory is marked
              ``truncated=True`` and sampling stops without dispatching any tool
              call the cut reply contains.
        """
        if isinstance(trajectories, dict):
            raise TypeError('ClientMultiTurnRollout.__call__ expects a List[Trajectory]; '
                            'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        n = len(trajectories)
        if n == 0:
            return []

        sampling_params = self._as_sampling_params_dict(kwargs.get('sampling_params', self.sampling_params))
        tool_managers = self._resolve_tool_managers(kwargs.get('tool_manager', self.tool_manager), n)

        state = self._initialize_state(trajectories)
        for _ in range(self.max_turns):
            active = [index for index in range(n) if not state.done[index]]
            if not active:
                break

            # One batched HTTP call for all live trajectories. Network and timeout
            # errors intentionally propagate unchanged so retry policy stays upstream.
            responses = self.sampler.sample(
                [state.pifs[index] for index in active],
                sampling_params=sampling_params,
            )
            pending_bridges = self._process_responses(active, responses, state, tool_managers)
            self._apply_bridges(state, pending_bridges)

        self._validate_logprob_alignment(state.pifs, state.all_logprobs)
        return self._build_outputs(
            trajectories,
            state.pifs,
            state.all_logprobs,
            state.turns,
            state.stop_reasons,
            state.truncated,
        )

    # ------------------------------------------------------------------ private

    def _initialize_state(self, trajectories: List[Trajectory]) -> _RolloutState:
        pifs: list[dict[str, Any]] = []
        for trajectory in trajectories:
            pif = self.template.encode(trajectory, add_generation_prompt=True)
            pif.setdefault('messages', list(trajectory.get('messages', [])))
            pifs.append(pif)
        size = len(trajectories)
        return _RolloutState(
            pifs=pifs,
            all_logprobs=[[] for _ in range(size)],
            stop_reasons=[None] * size,
            turns=[0] * size,
            truncated=[False] * size,
            done=[False] * size,
        )

    def _process_responses(self, active, responses, state: _RolloutState,
                           tool_managers) -> list[tuple[int, list[dict]]]:
        pending_bridges = []
        for local_index, global_index in enumerate(active):
            sequence = responses[local_index].sequences[0]
            tool_messages = self._process_sequence(local_index, global_index, sequence, state,
                                                   tool_managers[global_index])
            if tool_messages is not None:
                pending_bridges.append((global_index, tool_messages))
        return pending_bridges

    def _process_sequence(self, local_index, global_index, sequence, state: _RolloutState,
                          tool_manager: ToolManager | None) -> list[dict] | None:
        state.turns[global_index] += 1
        if sequence.new_input_feature is None or 'input_ids' not in sequence.new_input_feature:
            raise RuntimeError(f'Sampler returned a sequence without new_input_feature.input_ids at '
                               f'batch index {local_index} (trajectory {global_index}); '
                               f'cannot continue multi-turn.')

        state.pifs[global_index] = dict(sequence.new_input_feature)
        if sequence.logprobs is not None:
            if len(sequence.logprobs) != len(sequence.tokens):
                raise RuntimeError(f'logprobs length ({len(sequence.logprobs)}) does not match sampled '
                                   f'token count ({len(sequence.tokens)}) at turn {state.turns[global_index]} '
                                   f'(trajectory {global_index})')
            state.all_logprobs[global_index].extend(sequence.logprobs)
        state.stop_reasons[global_index] = sequence.stop_reason

        if sequence.stop_reason == 'length' or self._at_token_limit(state.pifs[global_index]):
            state.truncated[global_index] = True
            state.done[global_index] = True
            return None

        messages = state.pifs[global_index].get('messages') or []
        last_message = messages[-1] if messages else None
        tool_calls = last_message.get('tool_calls') if isinstance(last_message, dict) else None
        tool_calls = tool_calls or self.template.parse_tool_call(sequence.decoded or '')
        if not tool_calls:
            state.done[global_index] = True
            return None
        if state.turns[global_index] >= self.max_turns:
            state.truncated[global_index] = True
            state.stop_reasons[global_index] = 'max_turns'
            state.done[global_index] = True
            return None
        if tool_manager is None:
            raise ValueError(f'trajectory {global_index} produced tool_calls but no tool_manager '
                             f'was provided (at construction time or as a per-call kwarg).')
        return [{'role': 'tool', 'content': tool_manager(tool_call)} for tool_call in tool_calls]

    def _at_token_limit(self, pif: dict[str, Any]) -> bool:
        return self.max_trajectory_tokens is not None and len(pif.get('input_ids') or []) >= self.max_trajectory_tokens

    def _apply_bridges(self, state: _RolloutState, pending_bridges: list[tuple[int, list[dict]]]) -> None:
        for global_index, tool_messages in pending_bridges:
            extended = extend_with_bridge(state.pifs[global_index], tool_messages, self.template)
            if extended is None:
                state.truncated[global_index] = True
                state.done[global_index] = True
            else:
                state.pifs[global_index] = extended

    @staticmethod
    def _validate_logprob_alignment(pifs: List[Dict[str, Any]], all_logprobs: List[List[Any]]) -> None:
        """Reject output that would corrupt downstream GRPO old-logprob alignment."""
        for index, logprobs in enumerate(all_logprobs):
            if not logprobs:
                continue
            labels = pifs[index].get('labels') or []
            trainable = sum(1 for label in labels if label != -100)
            if len(logprobs) != trainable:
                raise RuntimeError(f'logprobs/labels misaligned for trajectory {index}: '
                                   f'{len(logprobs)} logprobs vs {trainable} '
                                   f'trainable labels (labels != -100). This invariant is '
                                   f'required by grpo._pad_and_align_to_batch; a mismatch '
                                   f'would silently corrupt GRPO old_logps alignment.')

    @staticmethod
    def _build_outputs(
        trajectories: List[Trajectory],
        pifs: List[Dict[str, Any]],
        all_logprobs: List[List[Any]],
        turns: List[int],
        stop_reasons: List[Optional[str]],
        truncated: List[bool],
    ) -> List[Trajectory]:
        """Merge final per-trajectory state while preserving input order."""
        outputs: List[Trajectory] = []
        for index, trajectory in enumerate(trajectories):
            output = dict(trajectory)
            output.update(pifs[index])
            output['messages'] = list(pifs[index].get('messages') or output.get('messages', []))
            output['logprobs'] = all_logprobs[index] or None
            output['turns'] = turns[index]
            output['stop_reason'] = stop_reasons[index]
            output['truncated'] = truncated[index]
            outputs.append(output)
        return outputs

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
