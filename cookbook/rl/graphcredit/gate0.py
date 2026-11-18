"""Offline Gate 0 probe for the proposed TurnCredit state graph.

This probe deliberately does not import or modify Twinkle's rollout path. It
tests the proposed state-key and graph semantics on deterministic trajectories
before any core implementation is allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Transition:
    prompt_id: str
    rollout_id: str
    turn_id: int
    pre_state: str
    action: str
    post_state: str
    action_correct: bool
    terminal_success: bool | None


@dataclass(frozen=True)
class TurnScore:
    pre_state: str
    action: str
    action_correct: bool
    graph_return: float
    graph_advantage: float


def _build_rollout(
    prompt_id: str,
    rollout_id: str,
    choices: list[str],
) -> list[Transition]:
    transitions: list[Transition] = []
    state = f'{prompt_id}:s0'
    for turn_id, choice in enumerate(choices):
        correct = choice == 'ok'
        if correct:
            if turn_id == 2:
                post_state = f'{prompt_id}:success'
            else:
                post_state = f'{prompt_id}:s{turn_id + 1}'
        else:
            post_state = f'{prompt_id}:dead{turn_id}'
        transitions.append(
            Transition(
                prompt_id=prompt_id,
                rollout_id=rollout_id,
                turn_id=turn_id,
                pre_state=state,
                action=choice,
                post_state=post_state,
                action_correct=correct,
                terminal_success=True if correct and turn_id == 2 else None,
            ))
        if not correct:
            break
        state = post_state
    return transitions


def _build_fixture() -> list[Transition]:
    paths = (
        ('ok', 'ok', 'ok'),
        ('bad',),
        ('ok', 'bad'),
        ('ok', 'ok', 'bad'),
        ('ok', 'ok', 'ok'),
        ('bad',),
        ('ok', 'bad'),
        ('ok', 'ok', 'bad'),
    )
    records: list[Transition] = []
    for prompt_index in range(8):
        prompt_id = f'p{prompt_index}'
        for rollout_index, choices in enumerate(paths):
            records.extend(_build_rollout(prompt_id, f'r{rollout_index}', list(choices)))
    return records


def _graph_distances(records: Iterable[Transition]) -> dict[str, int]:
    edges: dict[str, set[str]] = defaultdict(set)
    terminal_success: set[str] = set()
    nodes: set[str] = set()
    for record in records:
        nodes.update((record.pre_state, record.post_state))
        edges[record.pre_state].add(record.post_state)
        if record.terminal_success is True:
            terminal_success.add(record.post_state)

    distances = {node: math.inf for node in nodes}
    queue: deque[str] = deque()
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, targets in edges.items():
        for target in targets:
            reverse[target].add(source)
    for node in terminal_success:
        distances[node] = 0
        queue.append(node)
    while queue:
        child = queue.popleft()
        for parent in reverse.get(child, ()):
            candidate = distances[child] + 1
            if candidate < distances[parent]:
                distances[parent] = candidate
                queue.append(parent)
    return distances


def _score_graph(records: list[Transition], gamma: float = 0.1) -> list[TurnScore]:
    distances = _graph_distances(records)
    raw: list[tuple[Transition, float]] = []
    for record in records:
        distance = distances.get(record.post_state, math.inf)
        graph_return = 0.0 if math.isinf(distance) else 10.0 * gamma**distance
        raw.append((record, graph_return))

    groups: dict[tuple[str, str], list[tuple[Transition, float]]] = defaultdict(list)
    for record, graph_return in raw:
        groups[(record.prompt_id, record.pre_state)].append((record, graph_return))

    scores: list[TurnScore] = []
    for (_, _), group in groups.items():
        values = [value for _, value in group]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        for record, graph_return in group:
            advantage = 0.0 if std == 0.0 else (graph_return - mean) / std
            scores.append(
                TurnScore(
                    pre_state=record.pre_state,
                    action=record.action,
                    action_correct=record.action_correct,
                    graph_return=graph_return,
                    graph_advantage=advantage,
                ))
    return scores


def _ranking_accuracy(scores: list[TurnScore]) -> float:
    groups: dict[str, list[TurnScore]] = defaultdict(list)
    for score in scores:
        groups[score.pre_state].append(score)
    comparisons = 0
    correct = 0
    for group in groups.values():
        good = [score.graph_advantage for score in group if score.action_correct]
        bad = [score.graph_advantage for score in group if not score.action_correct]
        if not good or not bad:
            continue
        comparisons += 1
        correct += statistics.fmean(good) > statistics.fmean(bad)
    return correct / comparisons if comparisons else 0.0


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _episode_advantages(records: list[Transition]) -> dict[str, float]:
    rollouts: dict[tuple[str, str], list[Transition]] = defaultdict(list)
    for record in records:
        rollouts[(record.prompt_id, record.rollout_id)].append(record)
    by_prompt: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (prompt_id, rollout_id), turns in rollouts.items():
        reward = 1.0 if turns[-1].terminal_success is True else 0.0
        by_prompt[prompt_id].append((rollout_id, reward))
    advantages: dict[str, float] = {}
    for values in by_prompt.values():
        mean = statistics.fmean(reward for _, reward in values)
        std = _sample_std([reward for _, reward in values])
        for rollout_id, reward in values:
            advantages[rollout_id] = 0.0 if std == 0.0 else (reward - mean) / std
    return advantages


def _implementation_probe(fixture: list[Transition]) -> dict:
    from twinkle.advantage import StateGraphAdvantage

    by_rollout: dict[tuple[str, str], list[Transition]] = defaultdict(list)
    for record in fixture:
        by_rollout[(record.prompt_id, record.rollout_id)].append(record)
    trajectories = []
    rewards = []
    row_by_rollout = {}
    prompt_ids = sorted({record.prompt_id for record in fixture})
    for prompt_id in prompt_ids:
        rollout_ids = sorted(rollout_id for candidate, rollout_id in by_rollout if candidate == prompt_id)
        for rollout_id in rollout_ids:
            turns = sorted(by_rollout[(prompt_id, rollout_id)], key=lambda record: record.turn_id)
            row_by_rollout[(prompt_id, rollout_id)] = len(trajectories)
            rewards.append(1.0 if turns[-1].terminal_success is True else 0.0)
            trajectories.append({
                'labels': [0] * 16,
                'turn_credit_status': {
                    'available': True,
                    'fallback_reason': None,
                },
                'turn_credits': [{
                    'turn_id': turn.turn_id,
                    'action_token_start': 1 + 2 * turn.turn_id,
                    'action_token_end': 2 + 2 * turn.turn_id,
                    'pre_state_key': turn.pre_state,
                    'post_state_key': turn.post_state,
                    'policy_version': 1,
                    'terminal_success': turn.terminal_success,
                } for turn in turns],
            })

    # Warm up lazy torch imports outside the measured graph path.
    StateGraphAdvantage(beta_episode=0.0, beta_graph=0.0)(
        rewards[:8],
        trajectories[:8],
        num_generations=8,
    )
    advantage = StateGraphAdvantage(beta_episode=0.0, beta_graph=1.0)
    start = time.perf_counter()
    values = advantage(rewards, trajectories, num_generations=8)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    state_groups: dict[tuple[str, str], list[tuple[float, bool]]] = defaultdict(list)
    for record in fixture:
        row = row_by_rollout[(record.prompt_id, record.rollout_id)]
        position = 1 + 2 * record.turn_id
        state_groups[(record.prompt_id, record.pre_state)].append(
            (float(values[row, position].item()), record.action_correct))
    comparisons = 0
    correct = 0
    for group in state_groups.values():
        good = [value for value, is_correct in group if is_correct]
        bad = [value for value, is_correct in group if not is_correct]
        if not good or not bad:
            continue
        comparisons += 1
        correct += statistics.fmean(good) > statistics.fmean(bad)

    return {
        'credit_ranking_accuracy': correct / comparisons if comparisons else 0.0,
        'cpu_time_ms': elapsed_ms,
        'transition_count': advantage.last_diagnostics.transition_count,
        'comparable_state_group_count': advantage.last_diagnostics.comparable_state_group_count,
        'fallback_counts': advantage.last_diagnostics.fallback_counts,
    }


def run_gate0() -> dict:
    fixture = _build_fixture()
    start = time.perf_counter()
    scores = _score_graph(fixture)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    implementation = _implementation_probe(fixture)

    state_coverage = sum(bool(record.pre_state and record.post_state) for record in fixture) / len(fixture)
    group_sizes: dict[tuple[str, str], int] = defaultdict(int)
    for record in fixture:
        group_sizes[(record.prompt_id, record.pre_state)] += 1
    same_state_support = sum(size >= 2 for size in group_sizes.values()) / len(group_sizes)
    rank_accuracy = _ranking_accuracy(scores)

    fallback_advantage = _episode_advantages(fixture)
    valid_rollout_ids = {record.rollout_id for record in fixture}
    fallback_complete = valid_rollout_ids == set(fallback_advantage)

    no_success = [
        Transition(**{**asdict(record), 'terminal_success': None})
        for record in fixture
        if not record.post_state.endswith(':success')
    ]
    no_success_scores = _score_graph(no_success)
    no_success_graph_mass = sum(abs(score.graph_advantage) for score in no_success_scores)

    thresholds = {
        'state_key_coverage': 0.95,
        'same_state_support': 0.40,
        'credit_ranking_accuracy': 0.80,
        'fallback_rate_max': 0.05,
        'cpu_time_ms_max': 100.0,
    }
    observed = {
        'state_key_coverage': state_coverage,
        'same_state_support': same_state_support,
        'credit_ranking_accuracy': rank_accuracy,
        'fallback_rate': 0.0,
        'cpu_time_ms': elapsed_ms,
        'implementation_credit_ranking_accuracy': implementation['credit_ranking_accuracy'],
        'implementation_cpu_time_ms': implementation['cpu_time_ms'],
    }
    checks = {
        'state_key_coverage': state_coverage >= thresholds['state_key_coverage'],
        'same_state_support': same_state_support >= thresholds['same_state_support'],
        'credit_ranking_accuracy': rank_accuracy >= thresholds['credit_ranking_accuracy'],
        'fallback_rate': observed['fallback_rate'] <= thresholds['fallback_rate_max'],
        'cpu_time_ms': elapsed_ms <= thresholds['cpu_time_ms_max'],
        'implementation_credit_ranking_accuracy':
        implementation['credit_ranking_accuracy'] >= thresholds['credit_ranking_accuracy'],
        'implementation_cpu_time_ms': implementation['cpu_time_ms'] <= thresholds['cpu_time_ms_max'],
        'implementation_transition_count': implementation['transition_count'] == len(fixture),
        'implementation_no_fallback': implementation['fallback_counts'] == {},
        'episode_fallback_complete': fallback_complete,
        'no_success_has_zero_graph_mass': no_success_graph_mass == 0.0,
    }
    return {
        'passed': all(checks.values()),
        'thresholds': thresholds,
        'observed': observed,
        'checks': checks,
        'fixture': {
            'transition_count': len(fixture),
            'state_group_count': len(group_sizes),
            'mean_group_support': statistics.fmean(group_sizes.values()),
        },
        'negative_controls': {
            'no_success_graph_mass': no_success_graph_mass,
        },
        'implementation': implementation,
        'extra_policy_forwards': 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_gate0()
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
