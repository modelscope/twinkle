# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import sys
import types

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType('resource')
    stub.RLIMIT_AS = 0
    stub.setrlimit = lambda *_args, **_kwargs: None
    sys.modules['resource'] = stub

from twinkle_agentic.envs.base import Env, StepResult
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.rollout.credit import STATE_KEY_INFO, TERMINAL_SUCCESS_INFO
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager
from tests.twinkle_agentic.test_multi_turn_rollout import FakeSampler, FakeTemplate, FakeTokenizer


class StatefulEnv(Env):

    def __init__(self):
        self.state = 's0'

    def state_key(self):
        return self.state

    def step(self, tool_name, arguments):
        self.state = arguments.get('next_state', self.state)
        return StepResult(
            observation=f'state={self.state}',
            info={
                STATE_KEY_INFO: self.state,
                TERMINAL_SUCCESS_INFO: arguments.get('success'),
            },
        )


def _rollout(sampler, template, tool_manager, *, track):
    return MultiTurnRollout(
        sampler=sampler,
        template=template,
        tool_manager=tool_manager,
        max_turns=3,
        track_turn_credits=track,
    )


def test_multi_turn_emits_state_key_and_action_span():
    tokenizer = FakeTokenizer()
    template = FakeTemplate(tokenizer)
    sampler = FakeSampler(template)
    env = StatefulEnv()
    tool_manager = ToolManager(EnvTool.from_schemas(env, [{
        'type': 'function',
        'function': {
            'name': 'act'
        }
    }]))
    sampler.queue(
        '<tool_call>'
        + json.dumps({
            'name': 'act',
            'arguments': {
                'next_state': 's1',
                'success': True,
            },
        })
        + '</tool_call>')
    sampler.queue('done')
    rollout = _rollout(sampler, template, tool_manager, track=True)
    trajectory = rollout(
        [{
            'messages': [{
                'role': 'user',
                'content': 'go'
            }],
        }],
        environment_id='env',
        task_id='task',
        policy_version=7,
    )[0]

    assert trajectory['turn_credit_status']['available'] is True
    assert len(trajectory['turn_credits']) == 1
    credit = trajectory['turn_credits'][0]
    assert credit['pre_state_key'] == 'env:task:s0'
    assert credit['post_state_key'] == 'env:task:s1'
    assert credit['action_token_start'] < credit['action_token_end']
    assert credit['policy_version'] == 7
    assert credit['terminal_success'] is True
    assert credit['tool_events'][0]['tool_name'] == 'act'


def test_multi_turn_keeps_turn_credit_fields_absent_by_default():
    tokenizer = FakeTokenizer()
    template = FakeTemplate(tokenizer)
    sampler = FakeSampler(template)
    sampler.queue('done')
    trajectory = _rollout(sampler, template, None, track=False)(
        [{
            'messages': [{
                'role': 'user',
                'content': 'go'
            }],
        }])[0]
    assert 'turn_credits' not in trajectory
    assert 'turn_credit_status' not in trajectory
