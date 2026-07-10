"""check_no_repeated_calls: single-tool spin loops are penalized, batches aren't.

Regression for a dead-loop that exact-duplicate detection missed: the same tool
fired ~20 times with *different* arguments (empty repeated spins) used to score
1.0 because no two (name, args) pairs were identical.
"""
import json

from twinkle_agentic.verifier.hard_scorer import (TrajectoryView,
                                                  check_no_repeated_calls)


def _call(name, args):
    return {'role': 'assistant',
            'tool_calls': [{'function': {'name': name, 'arguments': json.dumps(args)}}]}


def _score(msgs):
    return check_no_repeated_calls(TrajectoryView({'messages': msgs})).score


def test_single_tool_spin_is_penalized():
    loop = [_call('LatexFixResponse', {'part': str(i)}) for i in range(18)]
    assert _score(loop) <= 0.4


def test_exact_duplicate_calls_penalized():
    dupes = [_call('read', {'p': 'same'}) for _ in range(4)]
    # 3 of 4 are exact duplicates -> 1 - 3/4 = 0.25
    assert _score(dupes) <= 0.3


def test_mixed_tools_not_penalized():
    mixed = [_call('read', {'p': str(i)}) for i in range(6)]
    mixed += [_call('grep', {'q': 'a'}), _call('edit', {'f': 'b'}),
              _call('run', {'c': 'c'}), _call('read', {'p': 'z'})]
    assert _score(mixed) == 1.0


def test_short_same_tool_loop_ok():
    # A legitimate 4-step same-tool loop (below the spin floor of 8 calls).
    small = [_call('read', {'p': str(i)}) for i in range(4)]
    assert _score(small) == 1.0


def test_fewer_than_two_calls_ok():
    assert _score([_call('read', {'p': 'a'})]) == 1.0
