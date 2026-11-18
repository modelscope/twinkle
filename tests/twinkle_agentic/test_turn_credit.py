# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import types

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType('resource')
    stub.RLIMIT_AS = 0
    stub.setrlimit = lambda *_args, **_kwargs: None
    sys.modules['resource'] = stub

from twinkle.data_format.sampling import SampledSequence
from twinkle_agentic.envs.base import Env, StepResult
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.rollout.credit import (
    STATE_KEY_INFO,
    TERMINAL_SUCCESS_INFO,
    make_tool_events,
    qualified_state_key,
    state_from_tool_infos,
)
from twinkle_agentic.rollout.ledger import TurnLedger
from twinkle_agentic.tools.tool_manager import ToolManager


class StatefulEnv(Env):

    def __init__(self):
        self.state = 's0'

    def state_key(self):
        return self.state

    def step(self, tool_name, arguments):
        return self.step_batch([(tool_name, arguments)])[0]

    def step_batch(self, calls):
        results = []
        for _, arguments in calls:
            self.state = arguments.get('next_state', self.state)
            results.append(
                StepResult(
                    observation=f'reached {self.state}',
                    info={
                        STATE_KEY_INFO: self.state,
                        TERMINAL_SUCCESS_INFO: arguments.get('success'),
                    },
                ))
        return results


class FakeTemplate:

    def encode(self, trajectory, add_generation_prompt=False):
        del add_generation_prompt
        messages = trajectory.get('messages') or []
        ids = list(range(len(messages) + 2))
        return {
            'messages': messages,
            'input_ids': ids,
            'labels': list(ids),
            'completion_mask': [0] * len(ids),
        }


def test_tool_manager_preserves_info_without_changing_call_many():
    env = StatefulEnv()
    manager = ToolManager(EnvTool.from_schemas(env, [{
        'type': 'function',
        'function': {
            'name': 'act'
        }
    }]))
    calls = [{
        'id': 'c1',
        'type': 'function',
        'function': {
            'name': 'act',
            'arguments': {
                'next_state': 's1'
            }
        },
    }]

    rich = manager.call_many_results(calls)
    assert len(rich) == 1
    assert rich[0].observation == 'reached s1'
    assert rich[0].info[STATE_KEY_INFO] == 's1'
    assert manager.call_many(calls) == ['reached s1']
    assert manager.current_state_key() == 's1'


def test_tool_manager_marks_unknown_calls_without_losing_metadata():
    manager = ToolManager()
    result = manager.call_result({
        'id': 'missing',
        'function': {
            'name': 'not_registered',
            'arguments': {}
        }
    })
    assert 'unknown tool' in result.observation
    assert 'error' in result.info


def test_state_and_tool_provenance_helpers():
    assert qualified_state_key('env', 'task', ' s1 ') == 'env:task:s1'
    assert qualified_state_key(None, 'task', 's1') is None
    assert state_from_tool_infos([
        {STATE_KEY_INFO: 's1'},
        {STATE_KEY_INFO: 's1', TERMINAL_SUCCESS_INFO: True},
    ]) == ('s1', True, None)
    assert state_from_tool_infos([
        {STATE_KEY_INFO: 's1'},
        {STATE_KEY_INFO: 's2'},
    ]) == (None, None, 'state_conflict')

    events = make_tool_events(
        [{
            'id': 'a',
            'function': {
                'name': 'act',
                'arguments': {'next_state': 's1'}
            }
        }],
        [StepResult(observation='ok', info={STATE_KEY_INFO: 's1'})],
        turn_id=1,
    )
    assert events[0].tool_call_id == 'a'
    assert events[0].tool_name == 'act'
    assert len(events[0].arguments_digest) == 64
    assert len(events[0].result_digest) == 64


def test_turn_ledger_records_action_span_and_state_edge():
    ledger = TurnLedger(
        FakeTemplate(),
        track_turn_credits=True,
        initial_state_key='env:task:s0',
        policy_version=3,
    )
    ledger.open({'messages': [{'role': 'user', 'content': 'go'}]})
    old_length = len(ledger.input_feature['input_ids'])
    new_feature = {
        **ledger.input_feature,
        'input_ids': [0, 1, 2, 10, 11, 12],
        'labels': [0, 1, 10, 11, 12, -100],
        'completion_mask': [0, 0, 1, 1, 1, 0],
    }
    span = ledger.record(SampledSequence(
        stop_reason='stop',
        tokens=[10, 11, 12],
        logprobs=None,
        new_input_feature=new_feature,
    ))
    assert span.start == old_length - 1
    assert span.end == len(new_feature['input_ids']) - 1
    ledger.complete_credit_turn(
        post_state_key='env:task:s1',
        tool_events=(),
        terminal_success=True,
    )
    trajectory = ledger.merge({'messages': []})
    assert trajectory['turn_credit_status']['available'] is True
    record = trajectory['turn_credits'][0]
    assert record['pre_state_key'] == 'env:task:s0'
    assert record['post_state_key'] == 'env:task:s1'
    assert record['action_token_start'] == span.start
    assert record['action_token_end'] == span.end
    assert record['policy_version'] == 3


def test_turn_ledger_marks_missing_initial_state_as_fallback():
    ledger = TurnLedger(FakeTemplate(), track_turn_credits=True)
    trajectory = ledger.merge({'messages': []})
    assert trajectory['turn_credit_status'] == {
        'available': False,
        'fallback_reason': 'state_key_missing',
    }


def test_turn_ledger_requires_policy_version_when_tracking():
    ledger = TurnLedger(FakeTemplate(), track_turn_credits=True, initial_state_key='env:task:s0')
    trajectory = ledger.merge({'messages': []})
    assert trajectory['turn_credit_status'] == {
        'available': False,
        'fallback_reason': 'policy_version_missing',
    }


def test_turn_ledger_rejects_non_sampled_tail():
    ledger = TurnLedger(
        FakeTemplate(),
        track_turn_credits=True,
        initial_state_key='env:task:s0',
        policy_version=0,
    )
    ledger.open({'messages': [{'role': 'user', 'content': 'go'}]})
    old_feature = ledger.input_feature
    new_feature = {
        **old_feature,
        'input_ids': [0, 1, 2, 10, 11, 99],
        'labels': [0, 1, 10, 11, 99, -100],
        'completion_mask': [0, 0, 1, 1, 1, 0],
    }
    span = ledger.record(SampledSequence(
        stop_reason='stop',
        tokens=[10, 11, 12],
        logprobs=None,
        new_input_feature=new_feature,
    ))
    trajectory = ledger.merge({'messages': []})

    assert span is None
    assert trajectory['turn_credit_status'] == {
        'available': False,
        'fallback_reason': 'action_span_invalid',
    }
