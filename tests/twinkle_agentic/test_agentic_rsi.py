# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the agentic building blocks: program checks, the Env tool bridge and
the challenger's statement parser.

No GPU, no microVM and no agent runtime. What each block has to get right is
narrow and testable on its own: a check's exit status decides whether a task was
solved, a turn's calls have to reach the environment as one batch under the names
the prompt advertised, and a statement that describes file content has to keep
that content.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

_REPO = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(_REPO, 'src'))

from twinkle_agentic.envs.base import Env, StepResult  # noqa: E402
from twinkle_agentic.envs.env_tool import EnvTool  # noqa: E402
from twinkle_agentic.tools.tool_manager import ToolManager  # noqa: E402
from twinkle_agentic.verifier.result_check import (Check, CheckContext,  # noqa: E402
                                                   checks_from_dicts, run_checks)


class RecordingEnv(Env):
    """An Env that records what reached it, one entry per dispatched batch.

    Whether a turn's calls left as one batch is invisible from the observations
    -- a batch and a serial loop return the same list -- so the dispatch itself
    is what gets recorded.
    """

    def __init__(self):
        self.batches = []

    def step(self, tool_name, arguments):
        return self.step_batch([(tool_name, arguments)])[0]

    def step_batch(self, calls):
        calls = [(name, args or {}) for name, args in calls]
        self.batches.append(calls)
        return [StepResult(observation=f'ran {name}') for name, _ in calls]


class ResultCheckFileTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='rescheck_test_')
        with open(os.path.join(self.tmp, 'report.md'), 'w', encoding='utf-8') as f:
            f.write('# Sales Report\n- Q1\n- Q2\n- Q3\n- Q4\n')
        with open(os.path.join(self.tmp, 'data.json'), 'w', encoding='utf-8') as f:
            json.dump({'result': {'items': [{'n': 7}]}}, f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

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


class ToolBridgeTest(unittest.TestCase):
    """The prompt's tool list and the executing tool list must be one list."""

    def setUp(self):
        self.env = RecordingEnv()
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
        # One dispatch for the turn, not one per call: the tools share an Env.
        self.assertEqual(len(self.env.batches), 1)
        # And each name reaches the Env as declared -- the list the model was
        # shown is the list the Env is asked to answer to.
        self.assertEqual(self.env.batches[0],
                         [('read_file', {'path': 'a'}), ('shell_executor', {'command': 'ls'})])
        self.assertEqual(out, ['ran read_file', 'ran shell_executor'])

    def test_nameless_schema_is_refused(self):
        with self.assertRaises(ValueError):
            EnvTool.from_schemas(self.env, [{'type': 'function', 'function': {}}])


class ProblemStatementParseTest(unittest.TestCase):
    """What a statement is allowed to carry.

    A statement that says what a file must contain has to be able to show the
    content, and the model shows it in a fence. Stripping every fence -- which is
    what "the statement is prose, not code" had been implemented as -- turned
    "1. `data.json` containing:" into a sentence that ends there. 7 of ex11's 16
    measured statements had a fence and 5 of those 7 were solved 0 times out of
    8, against 1 of the 9 that had none: those tasks were unanswerable, not hard.
    """

    def setUp(self):
        from twinkle_agentic.challenger.agentic import parse_problem_statement
        self.parse = parse_problem_statement

    def test_fenced_file_content_stays_in_the_statement(self):
        reply = ('<think>planning</think>\n'
                 'Create `data.json` containing:\n\n'
                 '```json\n{"a": 1}\n```\n\n'
                 'No other files may exist.')
        statement = self.parse(reply)
        self.assertIn('{"a": 1}', statement)
        self.assertIn('No other files may exist.', statement)

    def test_a_fence_around_the_whole_reply_is_unwrapped_not_deleted(self):
        reply = '<think>planning</think>\n```\nCreate data.json holding {}.\n```'
        self.assertEqual(self.parse(reply), 'Create data.json holding {}.')

    def test_thinking_is_never_part_of_the_statement(self):
        reply = '<think>Create secret.txt</think>\nCreate visible.txt.'
        self.assertEqual(self.parse(reply), 'Create visible.txt.')

    def test_an_empty_reply_is_no_statement(self):
        self.assertIsNone(self.parse('<think>only thought</think>\n   \n'))


if __name__ == '__main__':
    unittest.main()
