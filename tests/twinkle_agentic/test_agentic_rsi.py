# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the agentic building blocks: program checks and the ms-agent Env.

No GPU and no ms-agent runtime: the Env is driven with a fake ToolManager that
records what it was asked to run, which is enough to pin the two behaviours the
trainer depends on -- calls arrive batched, and a check's exit status survives
the round trip through a text-only tool.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from twinkle_agentic.envs.env_tool import EnvTool  # noqa: E402
from twinkle_agentic.envs.ms_agent_tool_env import MsAgentToolEnv  # noqa: E402
from twinkle_agentic.tools.tool_manager import ToolManager  # noqa: E402
from twinkle_agentic.verifier.result_check import (Check, CheckContext,  # noqa: E402
                                                   checks_from_dicts, run_checks)


class FakeMsToolManager:
    """Stands in for ms-agent's ToolManager, recording dispatch shape."""

    # ms-agent namespaces tools as ``{server}---{tool}``; keep that here so the
    # tests exercise the same name resolution production hits.
    TOOLS = [
        {'tool_name': 'code_executor---shell_executor'},
        {'tool_name': 'code_executor---python_executor'},
        {'tool_name': 'file_system---write_file'},
    ]

    def __init__(self, handler=None):
        self.single_calls = []
        self.batch_sizes = []
        self._handler = handler or (lambda call: f'ran {call["tool_name"]}')

    async def get_tools(self):
        return list(self.TOOLS)

    async def single_call_tool(self, tool_info):
        self.single_calls.append(tool_info)
        return self._handler(tool_info)

    async def parallel_call_tool(self, tool_list, on_result=None):
        self.batch_sizes.append(len(tool_list))
        return [self._handler(call) for call in tool_list]

    async def cleanup(self):
        pass


class ResultCheckFileTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='rescheck_test_')
        with open(os.path.join(self.tmp, 'report.md'), 'w', encoding='utf-8') as f:
            f.write('# Sales Report\n- Q1\n- Q2\n- Q3\n- Q4\n')
        with open(os.path.join(self.tmp, 'data.json'), 'w', encoding='utf-8') as f:
            json.dump({'result': {'items': [{'n': 7}]}}, f)

    def ctx(self, answer=''):
        return CheckContext(workspace=self.tmp, final_answer=answer)

    def test_file_exists_and_absent(self):
        report = run_checks([
            Check(kind='file_exists', path='report.md'),
            Check(kind='file_absent', path='nope.txt'),
        ], self.ctx())
        self.assertTrue(report.all_passed)
        self.assertEqual(report.score, 1.0)

    def test_file_contains_value_and_pattern(self):
        report = run_checks([
            Check(kind='file_contains', path='report.md', value='# Sales Report'),
            Check(kind='file_contains', path='report.md', pattern=r'(?s)Q1.*Q4'),
        ], self.ctx())
        self.assertTrue(report.all_passed)

    def test_missing_file_fails_with_reason(self):
        report = run_checks([Check(kind='file_contains', path='gone.md', value='x')], self.ctx())
        self.assertFalse(report.all_passed)
        self.assertIn('does not exist', report.failures()[0])

    def test_file_json_dotted_key_including_list_index(self):
        report = run_checks(
            [Check(kind='file_json', path='data.json', key='result.items.0.n', value=7)],
            self.ctx())
        self.assertTrue(report.all_passed)

    def test_path_escaping_workspace_is_rejected(self):
        report = run_checks([Check(kind='file_exists', path='../../etc/passwd')], self.ctx())
        self.assertFalse(report.all_passed)
        self.assertIn('escapes the workspace', report.failures()[0])

    def test_empty_checks_score_zero_not_one(self):
        # A task with no checks must not look solved.
        report = run_checks([], self.ctx())
        self.assertEqual(report.score, 0.0)
        self.assertEqual(report.n_total, 0)

    def test_fraction_vs_all_or_nothing(self):
        checks = [Check(kind='file_exists', path='report.md'),
                  Check(kind='file_exists', path='missing.md')]
        self.assertEqual(run_checks(checks, self.ctx(), mode='fraction').score, 0.5)
        self.assertEqual(run_checks(checks, self.ctx(), mode='all_or_nothing').score, 0.0)

    def test_weight_shifts_partial_credit(self):
        checks = [Check(kind='file_exists', path='report.md', weight=3.0),
                  Check(kind='file_exists', path='missing.md', weight=1.0)]
        self.assertAlmostEqual(run_checks(checks, self.ctx()).score, 0.75)

    def test_answer_kinds(self):
        report = run_checks([
            Check(kind='answer_contains', value='Alibaba'),
            Check(kind='answer_regex', pattern=r'(?i)qwen\d'),
        ], self.ctx(answer='Qwen3 was published by Alibaba.'))
        self.assertTrue(report.all_passed)

    def test_local_shell_and_python_run_in_workspace(self):
        report = run_checks([
            Check(kind='shell', code='test -f report.md'),
            Check(kind='python', code='open("report.md").read()'),
        ], self.ctx())
        self.assertTrue(report.all_passed, report.failures())

    def test_failing_python_check_reports_nonzero(self):
        report = run_checks([Check(kind='python', code='assert 1 == 2')], self.ctx())
        self.assertFalse(report.all_passed)

    def test_bad_kind_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Check(kind='definitely_not_a_kind')

    def test_checks_from_dicts(self):
        checks = checks_from_dicts([{'kind': 'file_exists', 'path': 'a'}])
        self.assertEqual(checks[0].kind, 'file_exists')


class MsAgentToolEnvTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='envtest_')
        self.tm = FakeMsToolManager()
        self.env = MsAgentToolEnv(tool_manager=self.tm, workspace=self.tmp)

    def test_step_forwards_name_and_arguments(self):
        result = self.env.step('read_file', {'path': 'a.txt'})
        self.assertEqual(self.tm.single_calls[0],
                         {'tool_name': 'read_file', 'arguments': {'path': 'a.txt'}})
        self.assertEqual(result.observation, 'ran read_file')

    def test_step_batch_uses_one_parallel_call(self):
        results = self.env.step_batch([('read_file', {'p': 1}), ('grep', {'q': 'x'})])
        self.assertEqual(self.tm.batch_sizes, [2])
        self.assertEqual([r.observation for r in results], ['ran read_file', 'ran grep'])

    def test_single_call_batch_does_not_go_through_parallel(self):
        self.env.step_batch([('glob', {})])
        self.assertEqual(self.tm.batch_sizes, [])
        self.assertEqual(len(self.tm.single_calls), 1)

    def test_observation_is_truncated(self):
        tm = FakeMsToolManager(handler=lambda call: 'x' * 50)
        env = MsAgentToolEnv(tool_manager=tm, workspace=self.tmp, max_observation_chars=10)
        obs = env.step('grep', {}).observation
        self.assertTrue(obs.startswith('x' * 10))
        self.assertIn('truncated 40 chars', obs)

    def test_non_string_result_is_json_encoded(self):
        tm = FakeMsToolManager(handler=lambda call: {'ok': True})
        env = MsAgentToolEnv(tool_manager=tm, workspace=self.tmp)
        self.assertEqual(env.step('t', {}).observation, '{"ok": true}')

    def test_requires_a_tool_manager(self):
        with self.assertRaises(ValueError):
            MsAgentToolEnv()

    def test_runner_recovers_exit_code_from_text_output(self):
        # The sandbox tools return prose; the marker is how the exit status
        # survives. Emulate a shell that echoes the marker. Matching on the
        # namespaced name also proves the plain name was resolved.
        def handler(call):
            if call['tool_name'] == 'code_executor---shell_executor':
                return 'some output\n__TWINKLE_RC__:0'
            return '__TWINKLE_RC__:3'

        env = MsAgentToolEnv(tool_manager=FakeMsToolManager(handler), workspace=self.tmp)
        runner = env.runner()
        self.assertEqual(runner('ls', 'shell'), (0, 'some output'))
        self.assertEqual(runner('boom()', 'python')[0], 3)

    def test_resolve_tool_maps_plain_name_onto_namespaced_one(self):
        env = MsAgentToolEnv(tool_manager=FakeMsToolManager(), workspace=self.tmp)
        self.assertEqual(env.resolve_tool('shell_executor'),
                         'code_executor---shell_executor')
        # An already-qualified name is left alone.
        self.assertEqual(env.resolve_tool('file_system---write_file'),
                         'file_system---write_file')

    def test_resolve_tool_raises_on_unknown_name(self):
        # Silently passing a bad name through would surface as a failed check,
        # which is indistinguishable from the task genuinely not being solved.
        env = MsAgentToolEnv(tool_manager=FakeMsToolManager(), workspace=self.tmp)
        with self.assertRaises(ValueError):
            env.resolve_tool('no_such_tool')

    def test_runner_missing_marker_is_a_failure_not_a_pass(self):
        env = MsAgentToolEnv(tool_manager=FakeMsToolManager(lambda c: 'sandbox died'),
                             workspace=self.tmp)
        code, out = env.runner()('ls', 'shell')
        self.assertNotEqual(code, 0)
        self.assertIn('sandbox died', out)

    def test_checks_run_through_the_env_runner(self):
        env = MsAgentToolEnv(
            tool_manager=FakeMsToolManager(lambda c: '__TWINKLE_RC__:0'),
            workspace=self.tmp)
        report = run_checks([Check(kind='shell', code='true')],
                            CheckContext(workspace=self.tmp, runner=env.runner()))
        self.assertTrue(report.all_passed)


class ToolBridgeTest(unittest.TestCase):
    """The prompt's tool list and the executing tool list must be one list."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='bridge_')
        self.tm = FakeMsToolManager()
        self.env = MsAgentToolEnv(tool_manager=self.tm, workspace=self.tmp)
        self.schemas = [
            {'type': 'function', 'function': {'name': 'read_file', 'parameters': {}}},
            {'type': 'function', 'function': {'name': 'shell_executor', 'parameters': {}}},
        ]

    def test_from_schemas_binds_every_declared_tool(self):
        manager = ToolManager(EnvTool.from_schemas(self.env, self.schemas))
        self.assertEqual(sorted(manager.names()), ['read_file', 'shell_executor'])

    def test_declared_tools_collapse_into_one_step_batch(self):
        manager = ToolManager(EnvTool.from_schemas(self.env, self.schemas))
        calls = [
            {'id': '1', 'type': 'function',
             'function': {'name': 'read_file', 'arguments': '{"path": "a"}'}},
            {'id': '2', 'type': 'function',
             'function': {'name': 'shell_executor', 'arguments': '{"command": "ls"}'}},
        ]
        out = manager.call_many(calls)
        self.assertEqual(self.tm.batch_sizes, [2])
        self.assertEqual(out, ['ran read_file', 'ran shell_executor'])

    def test_nameless_schema_is_refused(self):
        with self.assertRaises(ValueError):
            EnvTool.from_schemas(self.env, [{'type': 'function', 'function': {}}])


if __name__ == '__main__':
    unittest.main()
