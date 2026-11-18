# Copyright (c) ModelScope Contributors. All rights reserved.
"""State-graph turn credit with episode-level GRPO as the safe fallback."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .base import Advantage
from .grpo import GRPOAdvantage

__all__ = ['StateGraphAdvantage', 'StateGraphAdvantageDiagnostics']


@dataclass
class StateGraphAdvantageDiagnostics:
    transition_count: int = 0
    graph_node_count: int = 0
    graph_edge_count: int = 0
    comparable_state_group_count: int = 0
    fallback_counts: dict[str, int] = field(default_factory=dict)
    output_shape: tuple[int, int] = (0, 0)

    def add_fallback(self, reason: str) -> None:
        self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1


class StateGraphAdvantage(Advantage):
    """Combine GRPO episode advantages with state-conditioned graph returns.

    ``turn_credits`` are emitted by ``MultiTurnRollout`` and use labels-space,
    half-open action spans. A group is graph-scored only when every participating
    policy version matches, at least one explicit success state exists, and the
    graph stays within the configured budget. Otherwise its rows remain exactly
    equal to the episode-level GRPO advantage.
    """

    def __init__(
        self,
        *,
        beta_episode: float = 1.0,
        beta_graph: float = 1.0,
        gamma: float = 0.1,
        min_state_support: int = 2,
        max_graph_nodes: int = 4096,
        max_graph_edges: int = 16384,
        epsilon: float = 1e-8,
    ):
        if not math.isfinite(beta_episode) or beta_episode < 0:
            raise ValueError('beta_episode must be finite and non-negative')
        if not math.isfinite(beta_graph) or beta_graph < 0:
            raise ValueError('beta_graph must be finite and non-negative')
        if not 0 < gamma <= 1:
            raise ValueError('gamma must be in (0, 1]')
        if min_state_support < 2:
            raise ValueError('min_state_support must be at least two')
        if max_graph_nodes < 1 or max_graph_edges < 1:
            raise ValueError('graph budgets must be positive')
        if epsilon <= 0:
            raise ValueError('epsilon must be positive')
        self.beta_episode = beta_episode
        self.beta_graph = beta_graph
        self.gamma = gamma
        self.min_state_support = min_state_support
        self.max_graph_nodes = max_graph_nodes
        self.max_graph_edges = max_graph_edges
        self.epsilon = epsilon
        self.last_diagnostics: StateGraphAdvantageDiagnostics | None = None

    def compute(
        self,
        rewards: Sequence[float],
        trajectories: Sequence[Mapping[str, Any]],
        *,
        num_generations: int,
        sequence_lengths: Sequence[int] | None = None,
    ) -> tuple[Any, StateGraphAdvantageDiagnostics]:
        import torch

        if num_generations <= 1:
            raise ValueError('StateGraphAdvantage requires at least two generations per prompt')
        if len(trajectories) != len(rewards) or len(rewards) % num_generations:
            raise ValueError('rewards and trajectories must contain complete groups')
        if sequence_lengths is None:
            lengths = [self._sequence_length(trajectory) for trajectory in trajectories]
        else:
            lengths = [int(length) for length in sequence_lengths]
            if len(lengths) != len(trajectories) or any(length <= 0 for length in lengths):
                raise ValueError('sequence_lengths must be positive and match trajectories')

        episode = GRPOAdvantage()(torch.as_tensor(rewards, dtype=torch.float32), num_generations=num_generations)
        advantages = (self.beta_episode * episode).unsqueeze(1).repeat(1, max(lengths))
        diagnostics = StateGraphAdvantageDiagnostics(output_shape=tuple(advantages.shape))

        for group_start in range(0, len(trajectories), num_generations):
            group_rows = range(group_start, group_start + num_generations)
            records = []
            fallback_rows: set[int] = set()
            versions: set[int | str] = set()
            versioned_records = 0
            for row in group_rows:
                trajectory = trajectories[row]
                status = trajectory.get('turn_credit_status') or {}
                if status.get('available') is not True:
                    fallback_rows.add(row)
                    diagnostics.add_fallback(str(status.get('fallback_reason') or 'turn_credit_unavailable'))
                    continue
                credits = list(trajectory.get('turn_credits') or [])
                if not credits:
                    fallback_rows.add(row)
                    diagnostics.add_fallback('turn_credit_empty')
                    continue
                for record in credits:
                    try:
                        parsed = self._validate_record(record, lengths[row])
                    except (TypeError, ValueError) as exc:
                        fallback_rows.add(row)
                        diagnostics.add_fallback(str(exc) or 'turn_credit_invalid')
                        break
                    records.append((row, parsed))
                    if parsed.get('policy_version') is not None:
                        versions.add(parsed['policy_version'])
                        versioned_records += 1

            if len(versions) > 1:
                diagnostics.add_fallback('policy_version_mismatch')
                continue
            if not records:
                continue
            if versioned_records != len(records):
                diagnostics.add_fallback('policy_version_missing')
                continue

            valid_records = [(row, record) for row, record in records if row not in fallback_rows]
            if not valid_records:
                continue
            try:
                distances, node_count, edge_count = self._distances(valid_records)
            except ValueError as exc:
                diagnostics.add_fallback(str(exc))
                continue
            diagnostics.graph_node_count += node_count
            diagnostics.graph_edge_count += edge_count
            diagnostics.transition_count += len(valid_records)

            state_groups: dict[str, list[tuple[int, Mapping[str, Any], float]]] = defaultdict(list)
            for row, record in valid_records:
                distance = distances.get(record['post_state_key'], math.inf)
                graph_return = 0.0 if math.isinf(distance) else 10.0 * self.gamma**distance
                state_groups[record['pre_state_key']].append((row, record, graph_return))

            for group in state_groups.values():
                if len(group) < self.min_state_support:
                    continue
                values = [value for _, _, value in group]
                mean = statistics.fmean(values)
                std = statistics.stdev(values)
                if std <= self.epsilon:
                    continue
                diagnostics.comparable_state_group_count += 1
                for row, record, value in group:
                    local = (value - mean) / std
                    start, end = record['action_token_start'], record['action_token_end']
                    advantages[row, start:end] += self.beta_graph * local

        return advantages, diagnostics

    @staticmethod
    def _sequence_length(trajectory: Mapping[str, Any]) -> int:
        for key in ('labels', 'input_ids', 'completion_mask'):
            value = trajectory.get(key)
            if value is not None:
                return len(value)
        raise ValueError('trajectory needs labels, input_ids, or completion_mask to size advantages')

    @staticmethod
    def _validate_record(record: Any, sequence_length: int) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise TypeError('turn_credit_invalid')
        raw_pre_state = record.get('pre_state_key')
        raw_post_state = record.get('post_state_key')
        if raw_pre_state is None or raw_post_state is None:
            raise ValueError('state_key_missing')
        if not isinstance(raw_pre_state, str) or not isinstance(raw_post_state, str):
            raise ValueError('state_key_invalid')
        pre_state = raw_pre_state.strip()
        post_state = raw_post_state.strip()
        if not pre_state or not post_state:
            raise ValueError('state_key_missing')
        try:
            start = int(record.get('action_token_start', -1))
            end = int(record.get('action_token_end', -1))
        except (TypeError, ValueError) as exc:
            raise ValueError('action_span_invalid') from exc
        if start < 0 or end <= start or end > sequence_length:
            raise ValueError('action_span_invalid')
        observation_start = record.get('observation_token_start')
        observation_end = record.get('observation_token_end')
        if (observation_start is None) != (observation_end is None):
            raise ValueError('observation_span_invalid')
        if observation_start is not None:
            try:
                observation_start = int(observation_start)
                observation_end = int(observation_end)
            except (TypeError, ValueError) as exc:
                raise ValueError('observation_span_invalid') from exc
            if (observation_start < end or observation_end <= observation_start or observation_end > sequence_length):
                raise ValueError('observation_span_invalid')
        policy_version = record.get('policy_version')
        if policy_version is not None and (isinstance(policy_version, bool)
                                           or not isinstance(policy_version, (int, str))):
            raise ValueError('policy_version_invalid')
        terminal_success = record.get('terminal_success')
        if terminal_success is not None and not isinstance(terminal_success, bool):
            raise ValueError('terminal_success_invalid')
        parsed = dict(record)
        parsed.update(
            action_token_start=start,
            action_token_end=end,
            observation_token_start=observation_start,
            observation_token_end=observation_end,
            pre_state_key=pre_state,
            post_state_key=post_state,
            policy_version=policy_version,
            terminal_success=terminal_success,
        )
        return parsed

    def _distances(
        self,
        records: Sequence[tuple[int, Mapping[str, Any]]],
    ) -> tuple[dict[str, int], int, int]:
        reverse: dict[str, set[str]] = defaultdict(set)
        nodes: set[str] = set()
        edges: set[tuple[str, str]] = set()
        success: set[str] = set()
        for _, record in records:
            pre_state, post_state = record['pre_state_key'], record['post_state_key']
            nodes.update((pre_state, post_state))
            edges.add((pre_state, post_state))
            reverse[post_state].add(pre_state)
            if record.get('terminal_success') is True:
                success.add(post_state)
        if not success:
            raise ValueError('terminal_success_missing')
        if len(nodes) > self.max_graph_nodes:
            raise ValueError('graph_budget_exceeded')
        if len(edges) > self.max_graph_edges:
            raise ValueError('graph_budget_exceeded')
        distances = {node: math.inf for node in nodes}
        queue: deque[str] = deque()
        for node in success:
            distances[node] = 0
            queue.append(node)
        while queue:
            child = queue.popleft()
            for parent in reverse.get(child, ()):
                candidate = distances[child] + 1
                if candidate < distances[parent]:
                    distances[parent] = candidate
                    queue.append(parent)
        return distances, len(nodes), len(edges)

    def __call__(
        self,
        rewards: Sequence[float],
        trajectories: Sequence[Mapping[str, Any]],
        *,
        num_generations: int,
        sequence_lengths: Sequence[int] | None = None,
        **kwargs,
    ):
        del kwargs
        advantages, diagnostics = self.compute(
            rewards,
            trajectories,
            num_generations=num_generations,
            sequence_lengths=sequence_lengths,
        )
        self.last_diagnostics = diagnostics
        return advantages
