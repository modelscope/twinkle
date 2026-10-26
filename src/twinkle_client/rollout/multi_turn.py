# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side trainable multi-turn rollout over HTTP."""
from __future__ import annotations
import dataclasses
from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template
from twinkle_agentic.rollout.ledger import TurnLedger
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.sampler import vLLMSampler


class ClientMultiTurnRollout:
    """Drive safe text-only multi-turn sampling through ``/twinkle/sample``."""

    def __init__(
        self,
        sampler: vLLMSampler,
        template: Template,
        tool_manager: Optional[ToolManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        max_trajectory_tokens: Optional[int] = None,
    ):
        if template is None:
            raise ValueError('ClientMultiTurnRollout requires a local Template instance')
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f'max_trajectory_tokens must be >= 1 or None, got {max_trajectory_tokens}')
        self.sampler = sampler
        self.template = template
        self.tool_manager = tool_manager
        self.sampling_params = sampling_params or SamplingParams()
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        if self.sampling_params.num_samples != 1:
            raise ValueError('ClientMultiTurnRollout supports num_samples=1 only, '
                             f'got {self.sampling_params.num_samples}')

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError('ClientMultiTurnRollout expects List[Trajectory]')
        trajectories = list(trajectories)
        if not trajectories:
            return []
        sampling_params = self._as_sampling_params_dict(kwargs.get('sampling_params', self.sampling_params))
        tool_managers = self._resolve_tool_managers(kwargs.get('tool_manager', self.tool_manager), len(trajectories))
        ledgers = [
            self._open_ledger(trajectory, tool_managers[index], index)
            for index, trajectory in enumerate(trajectories)
        ]
        stop_reasons: List[Optional[str]] = [None] * len(trajectories)
        truncated = [False] * len(trajectories)
        done = [False] * len(trajectories)

        for _ in range(self.max_turns):
            active = [index for index, finished in enumerate(done) if not finished]
            if not active:
                break
            responses = self.sampler.sample(
                [ledgers[index].input_feature for index in active], sampling_params=sampling_params)
            if len(responses) != len(active):
                raise RuntimeError(f'sampler returned {len(responses)} responses for {len(active)} trajectories')
            pending: List[tuple[int, List[Dict[str, Any]]]] = []
            for batch_index, trajectory_index in enumerate(active):
                response = responses[batch_index]
                if len(response.sequences) != 1:
                    raise RuntimeError('ClientMultiTurnRollout requires exactly one sequence per trajectory')
                sequence = response.sequences[0]
                ledger = ledgers[trajectory_index]
                try:
                    ledger.record(sequence)
                except RuntimeError as error:
                    raise RuntimeError(f'batch index {batch_index} (trajectory {trajectory_index}): {error}') from error
                stop_reasons[trajectory_index] = sequence.stop_reason
                if sequence.stop_reason == 'length':
                    truncated[trajectory_index] = True
                    done[trajectory_index] = True
                    continue
                if ledger.full():
                    truncated[trajectory_index] = True
                    done[trajectory_index] = True
                    continue
                message = ledger.messages[-1] if ledger.messages else {}
                tool_calls = list(message.get('tool_calls') or ledger.parse_tool_call(sequence.decoded or ''))
                if not tool_calls:
                    done[trajectory_index] = True
                    continue
                if ledger.turns >= self.max_turns:
                    truncated[trajectory_index] = True
                    stop_reasons[trajectory_index] = 'max_turns'
                    done[trajectory_index] = True
                    continue
                manager = tool_managers[trajectory_index]
                if manager is None:
                    raise ValueError(f'trajectory {trajectory_index} produced tool_calls but no tool_manager was provided')
                observations = manager.call_many(tool_calls)
                pending.append((trajectory_index, self._tool_messages(tool_calls, observations)))
            for trajectory_index, messages in pending:
                ledger = ledgers[trajectory_index]
                if not ledger.observe(messages):
                    truncated[trajectory_index] = True
                    done[trajectory_index] = True
                    continue
                self._require_resubmittable(ledger, trajectory_index)

        return [
            ledger.merge(
                trajectory,
                turns=ledger.turns,
                stop_reason=stop_reasons[index],
                truncated=truncated[index],
            )
            for index, (trajectory, ledger) in enumerate(zip(trajectories, ledgers))
        ]

    def _open_ledger(self, trajectory: Trajectory, manager: Optional[ToolManager], index: int) -> TurnLedger:
        ledger = TurnLedger(
            self.template,
            label=f'trajectory {index}',
            max_tokens=self.max_trajectory_tokens,
            backend='vllm',
        )
        tools = manager.tool_infos() if manager is not None else list(trajectory.get('tools') or [])
        ledger.open(trajectory, tools=tools)
        self._require_resubmittable(ledger, index)
        return ledger

    def _require_resubmittable(self, ledger: TurnLedger, index: int) -> None:
        state = ledger.state
        feature_ids = list(ledger.input_feature.get('input_ids') or [])
        if state is None or state.media:
            raise NotImplementedError(
                f'trajectory {index} requires server-side RolloutState state_id for HTTP multi-turn media')
        if feature_ids != list(state.engine_input_ids):
            raise NotImplementedError(
                f'trajectory {index} requires server-side RolloutState state_id because feature and engine ids differ')

    @staticmethod
    def _tool_messages(tool_calls: List[Dict[str, Any]], observations: List[str]) -> List[Dict[str, Any]]:
        messages = []
        for call, observation in zip(tool_calls, observations):
            function = call.get('function') if isinstance(call.get('function'), dict) else {}
            message: Dict[str, Any] = {'role': 'tool', 'content': str(observation)}
            if call.get('id'):
                message['tool_call_id'] = call['id']
            if function.get('name'):
                message['name'] = function['name']
            messages.append(message)
        return messages

    @staticmethod
    def _as_sampling_params_dict(sampling_params) -> Optional[Dict[str, Any]]:
        if sampling_params is None:
            return None
        if isinstance(sampling_params, dict):
            return dict(sampling_params)
        if dataclasses.is_dataclass(sampling_params):
            return dataclasses.asdict(sampling_params)
        return sampling_params

    @staticmethod
    def _resolve_tool_managers(arg, count: int) -> List[Optional[ToolManager]]:
        if isinstance(arg, list):
            if len(arg) != count:
                raise ValueError(f'tool_manager list length ({len(arg)}) does not match trajectories ({count})')
            return list(arg)
        return [arg] * count
