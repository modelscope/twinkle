# Copyright (c) ModelScope Contributors. All rights reserved.
"""Harness + Env.step_batch + ToolManager.call_many."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.envs.base import Env, StepResult
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.harness.base import AgentHarness
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager

from test_multi_turn_rollout import (
    FakeSampler,
    FakeTemplate,
    FakeTokenizer,
    _tool_call_text,
    _user_traj,
)


class PrefixHarness(AgentHarness):
    """Inject a system message before the first encode; later turns are no-ops."""

    def before_generate(self, trajectory: Trajectory) -> Trajectory:
        msgs = list(trajectory.get('messages') or [])
        if not msgs or msgs[0].get('role') != 'system':
            trajectory['messages'] = [{'role': 'system', 'content': 'SYS'}] + msgs
        return trajectory


class TagToolHarness(AgentHarness):
    """Prefix every Env observation so we can see after_tools ran."""

    def after_tools(
        self,
        trajectory: Trajectory,
        observations: List[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        tagged = [f'H:{o}' for o in observations]
        return super().after_tools(trajectory, tagged, tool_calls)


class BatchEnv(Env):
    def __init__(self) -> None:
        self.step_calls = 0
        self.batch_calls = 0

    def step(self, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        self.step_calls += 1
        return StepResult(observation=f'{tool_name}:{json.dumps(arguments, sort_keys=True)}')

    def step_batch(self, calls):
        self.batch_calls += 1
        return super().step_batch(calls)

    def tools(self):
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'search',
                    'description': 'search',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'lookup',
                    'description': 'lookup',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            },
        ]


def _rollout(sampler, template, tool_manager, harness=None, max_turns=4):
    return MultiTurnRollout(
        sampler=sampler,
        template=template,
        tool_manager=tool_manager,
        sampling_params=SamplingParams(),
        max_turns=max_turns,
        harness=harness,
    )


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def template(tokenizer):
    return FakeTemplate(tokenizer)


@pytest.fixture
def sampler(template):
    return FakeSampler(template)


def test_harness_start_default():
    h = AgentHarness()
    traj = h.start('hello', user_data=[('id', '"t1"')])
    assert traj['messages'] == [{'role': 'user', 'content': 'hello'}]
    assert traj['user_data'] == [('id', '"t1"')]


def test_multiturn_harness_injects_system_before_encode(sampler, template):
    env = BatchEnv()
    mgr = ToolManager(EnvTool.from_env(env))
    sampler.queue('done.', stop_reason='stop')
    out = _rollout(sampler, template, mgr, harness=PrefixHarness())([_user_traj('hi')])[0]
    roles = [m['role'] for m in out['messages']]
    assert roles[0] == 'system'
    assert out['messages'][0]['content'] == 'SYS'
    assert 'user' in roles
    assert 'assistant' in roles


def test_multiturn_harness_after_tools_tags_observation(sampler, template):
    env = BatchEnv()
    mgr = ToolManager(EnvTool.from_env(env))
    sampler.queue(_tool_call_text('search', {'q': 'a'}), stop_reason='stop')
    sampler.queue('final', stop_reason='stop')
    out = _rollout(sampler, template, mgr, harness=TagToolHarness())([_user_traj('hi')])[0]
    tool_msgs = [m for m in out['messages'] if m['role'] == 'tool']
    assert len(tool_msgs) == 1
    assert tool_msgs[0]['content'].startswith('H:')
    assert tool_msgs[0].get('name') == 'search'


def test_tool_manager_call_many_uses_env_step_batch():
    env = BatchEnv()
    mgr = ToolManager(EnvTool.from_env(env))
    calls = [
        {'type': 'function', 'function': {'name': 'search', 'arguments': {'q': 'a'}}},
        {'type': 'function', 'function': {'name': 'lookup', 'arguments': {'k': 'b'}}},
    ]
    out = mgr.call_many(calls)
    assert env.batch_calls == 1
    assert env.step_calls == 2
    assert out[0].startswith('search:')
    assert out[1].startswith('lookup:')


def _run_wrapped(source: str):
    """exec the wrapper the way ms-agent's python_executor does: split dicts."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from twinkle_agentic.harness.ms_agent import single_namespace_source

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        exec(single_namespace_source(source), {'__builtins__': __builtins__}, {})
    return out.getvalue(), err.getvalue()


@pytest.mark.parametrize(
    'source, expect_err',
    [
        # A comprehension seeing a top-level name: broken under split dicts.
        ('xs = [1, 2]\nlim = 3\nassert all(x <= lim for x in xs)\nprint("ok")', False),
        ('import os\npaths = []\nassert all(os.path.exists(p) for p in paths)\nprint("ok")', False),
        # sys.exit must fail this call only -- never reach the caller's loop.
        ('print("ok")\nimport sys\nsys.exit(3)', True),
        ('print("ok")\nimport sys\nsys.exit(0)', False),
        ('print("ok")\nimport sys\nsys.exit()', False),
    ],
)
def test_single_namespace_source(source, expect_err):
    stdout, stderr = _run_wrapped(source)
    assert 'ok' in stdout
    assert bool(stderr) is expect_err
    if expect_err:
        assert 'SystemExit: 3' in stderr


def test_single_namespace_source_keeps_real_errors():
    """The patch must not turn a failing check into a passing one."""
    with pytest.raises(AssertionError):
        _run_wrapped('assert 1 == 2, "counts differ"')


def test_ms_agent_harness_start_system_and_user():
    import sys
    from pathlib import Path
    ms_root = Path(__file__).resolve().parents[2] / 'ms-agent'
    if ms_root.is_dir() and str(ms_root) not in sys.path:
        sys.path.insert(0, str(ms_root))
    pytest.importorskip('ms_agent')
    from twinkle_agentic.harness.ms_agent import MsAgentHarness

    try:
        harness = MsAgentHarness(auto_prepare=False)
        traj = harness.start('what is 1+1')
    except Exception as e:
        pytest.skip(f'ms-agent LLMAgent could not start: {e}')
    msgs = traj['messages']
    roles = [m['role'] for m in msgs]
    assert roles[0] == 'system'
    assert roles[-1] == 'user'
    assert '1+1' in msgs[-1]['content']
    assert isinstance(msgs[0]['content'], str)
    assert len(msgs[0]['content']) > 0
