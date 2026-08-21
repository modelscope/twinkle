# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the agentic building blocks: program checks and the sandboxed Env.

No GPU, no microVM and no ms-agent runtime. The Env is driven against a fake
sandbox that implements the two operations the real transport uses -- write a
file, run a command -- which is enough to pin what the trainer depends on: a
turn's calls leave as one request, a check's exit status survives the round trip
through a text-only tool, and a mistyped tool name is refused rather than
quietly scored as a failed check.
"""
import json
import os
import sys
import tempfile
import unittest

_REPO = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(_REPO, 'src'))
# The RSI wiring lives in cookbook, not in the framework: it is one deployment's
# choice of sandbox backend, and the tests follow it there.
_COOKBOOK = os.path.join(_REPO, 'cookbook', 'rl', 'rsi_agentic')
sys.path.insert(0, _COOKBOOK)
sys.path.insert(0, os.path.join(_COOKBOOK, 'sandbox_server'))

from remote_tool_env import RemoteMsAgentToolEnv  # noqa: E402
from tool_server import _usable_llm, _without_llm_args  # noqa: E402
from twinkle_agentic.envs.env_tool import EnvTool  # noqa: E402
from twinkle_agentic.tools.tool_manager import ToolManager  # noqa: E402
from twinkle_agentic.verifier.result_check import (Check, CheckContext,  # noqa: E402
                                                   checks_from_dicts, run_checks)

AGENT_CONFIG = os.path.join(_COOKBOOK, 'rsi_agent.yaml')

# ms-agent namespaces tools as ``{server}---{tool}``; keep that here so the
# tests exercise the same name resolution production hits.
DEFAULT_TOOLS = [
    {'type': 'function', 'function': {'name': 'code_executor---shell_executor', 'parameters': {}}},
    {'type': 'function', 'function': {'name': 'code_executor---python_executor', 'parameters': {}}},
    {'type': 'function', 'function': {'name': 'file_system---write_file', 'parameters': {}}},
]


class _Result:

    def __init__(self, stdout='', stderr='', exit_code=0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _FakeFiles:

    def __init__(self):
        self.store = {}

    def write(self, path, content):
        self.store[path] = content

    def read(self, path):
        return self.store[path]


class _FakeCommands:

    def __init__(self, sandbox):
        self._sandbox = sandbox

    def run(self, command, timeout=None, background=False, cwd=None):
        return self._sandbox.handle(command, background)


class FakeSandbox:
    """Stands in for an e2b sandbox: a filesystem plus a command channel.

    The Env reaches its in-sandbox server by writing a request file and then
    running curl, so a fake that understands those two operations exercises the
    real transport -- request shape included -- without booting a microVM.
    """

    def __init__(self, responder=None, tools=None):
        self.files = _FakeFiles()
        self.commands = _FakeCommands(self)
        self.requests = []
        self.killed = False
        self.tools = DEFAULT_TOOLS if tools is None else tools
        self._responder = responder or (lambda call: f'ran {call["tool_name"]}')

    def kill(self):
        self.killed = True

    def handle(self, command, background=False):
        if background or 'tool_server.py' in command or command.startswith('tail '):
            return _Result()
        if command.startswith('find '):
            prefix = '/workspace/'
            return _Result('\n'.join(p[len(prefix):] for p in self.files.store if p.startswith(prefix)))
        if '/health' in command:
            return _Result(json.dumps({'status': 'ok'}))
        if '/tools' in command:
            return _Result(json.dumps({'tools': self.tools}))
        if '/call' in command:
            payload = json.loads(self.files.store['/opt/rsi/request.json'])
            self.requests.append(payload)
            results = [{'observation': self._responder(call)} for call in payload['calls']]
            return _Result(json.dumps({'results': results}))
        raise AssertionError(f'unexpected sandbox command: {command}')


def make_env(responder=None, tools=None, **kwargs):
    """An Env already attached to a fake sandbox.

    ``reset`` would create a real one, so the sandbox is injected instead and
    everything above the e2b SDK boundary still runs for real.
    """
    env = RemoteMsAgentToolEnv(template='fake', config_path=AGENT_CONFIG, **kwargs)
    env._sandbox = FakeSandbox(responder, tools)
    return env


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


class RemoteMsAgentToolEnvTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='envtest_')
        self.env = make_env()
        self.sandbox = self.env._sandbox

    def test_step_forwards_name_and_arguments(self):
        result = self.env.step('read_file', {'path': 'a.txt'})
        self.assertEqual(self.sandbox.requests[0]['calls'],
                         [{'tool_name': 'read_file', 'arguments': {'path': 'a.txt'}}])
        self.assertEqual(result.observation, 'ran read_file')

    def test_a_turn_leaves_as_one_request(self):
        # Two calls, one sandbox round trip: the server runs them through
        # ms-agent's own parallel dispatch, as production would.
        results = self.env.step_batch([('read_file', {'p': 1}), ('grep', {'q': 'x'})])
        self.assertEqual(len(self.sandbox.requests), 1)
        self.assertEqual(len(self.sandbox.requests[0]['calls']), 2)
        self.assertEqual([r.observation for r in results], ['ran read_file', 'ran grep'])

    def test_observation_is_truncated(self):
        env = make_env(responder=lambda call: 'x' * 50, max_observation_chars=10)
        obs = env.step('grep', {}).observation
        self.assertTrue(obs.startswith('x' * 10))
        self.assertIn('truncated 40 chars', obs)

    def test_unreachable_runtime_becomes_an_observation(self):
        # A dead sandbox must not take down the training step: the episode plays
        # out and scores zero, which is what a broken run deserves anyway.
        def explode(command, background=False):
            raise RuntimeError('connection refused')

        self.sandbox.handle = explode
        obs = self.env.step('read_file', {}).observation
        self.assertIn('unreachable', obs)

    def test_tool_schemas_come_from_the_sandbox(self):
        self.assertEqual(self.env.tool_names(),
                         [t['function']['name'] for t in DEFAULT_TOOLS])

    def test_resolve_tool_maps_plain_name_onto_namespaced_one(self):
        self.assertEqual(self.env.resolve_tool('shell_executor'),
                         'code_executor---shell_executor')
        # An already-qualified name is left alone.
        self.assertEqual(self.env.resolve_tool('file_system---write_file'),
                         'file_system---write_file')

    def test_resolve_tool_raises_on_unknown_name(self):
        # Silently passing a bad name through would surface as a failed check,
        # which is indistinguishable from the task genuinely not being solved.
        with self.assertRaises(ValueError):
            self.env.resolve_tool('no_such_tool')

    def test_runner_recovers_exit_code_from_text_output(self):
        # The sandbox tools return prose; the marker is how the exit status
        # survives. Emulate a shell that echoes the marker. Matching on the
        # namespaced name also proves the plain name was resolved.
        def responder(call):
            if call['tool_name'] == 'code_executor---shell_executor':
                return 'some output\n__TWINKLE_RC__:0'
            return '__TWINKLE_RC__:3'

        runner = make_env(responder).runner()
        self.assertEqual(runner('ls', 'shell'), (0, 'some output'))
        self.assertEqual(runner('boom()', 'python')[0], 3)

    def test_runner_missing_marker_is_a_failure_not_a_pass(self):
        code, out = make_env(lambda c: 'sandbox died').runner()('ls', 'shell')
        self.assertNotEqual(code, 0)
        self.assertIn('sandbox died', out)

    def test_checks_run_through_the_env_runner(self):
        env = make_env(lambda c: '__TWINKLE_RC__:0')
        report = run_checks([Check(kind='shell', code='true')],
                            CheckContext(workspace=self.tmp, runner=env.runner()))
        self.assertTrue(report.all_passed)

    def test_download_workspace_brings_files_back_for_file_checks(self):
        # file_* checks read an ordinary local directory and cannot see into a
        # microVM, so the episode's output has to be copied out first.
        self.sandbox.files.store['/workspace/report.md'] = '# done\n'
        self.sandbox.files.store['/workspace/src/main.py'] = 'print(1)\n'
        dest = self.env.download_workspace(os.path.join(self.tmp, 'snap'))
        with open(os.path.join(dest, 'report.md'), encoding='utf-8') as f:
            self.assertEqual(f.read(), '# done\n')
        self.assertTrue(os.path.exists(os.path.join(dest, 'src', 'main.py')))

    def test_close_kills_the_sandbox(self):
        self.env.close()
        self.assertTrue(self.sandbox.killed)


class ToolServerSchemaTest(unittest.TestCase):
    """What /tools advertises must be what the runtime can actually honour."""

    READ_FILE = {
        'type': 'function',
        'function': {
            'name': 'file_system---read_file',
            'parameters': {'properties': {'paths': {}, 'abbreviate': {}}},
        },
    }

    def test_abbreviate_is_withdrawn_when_no_llm_is_configured(self):
        # abbreviate asks an LLM to summarise a file. With no key in the sandbox
        # it can only fail, and a model that learns "abbreviate is broken" would
        # carry that to a deployment where it works.
        stripped = _without_llm_args(self.READ_FILE)
        self.assertEqual(sorted(stripped['function']['parameters']['properties']), ['paths'])
        # The input is left alone: ms-agent owns that dict.
        self.assertIn('abbreviate', self.READ_FILE['function']['parameters']['properties'])

    def test_other_tools_pass_through_untouched(self):
        schema = {'type': 'function', 'function': {'name': 'file_system---glob', 'parameters': {}}}
        self.assertIs(_without_llm_args(schema), schema)

    def test_missing_llm_section_is_not_a_usable_llm(self):
        from omegaconf import OmegaConf
        self.assertFalse(_usable_llm(OmegaConf.create({})))
        # ms-agent's default agent.yaml declares a service but no credentials;
        # treating that as "configured" is what makes FileSystemTool assert.
        self.assertFalse(_usable_llm(OmegaConf.create({'llm': {'service': 'modelscope'}})))
        self.assertTrue(_usable_llm(
            OmegaConf.create({'llm': {'service': 'modelscope', 'modelscope_api_key': 'k'}})))


class ToolBridgeTest(unittest.TestCase):
    """The prompt's tool list and the executing tool list must be one list."""

    def setUp(self):
        self.env = make_env()
        self.sandbox = self.env._sandbox
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
        self.assertEqual(len(self.sandbox.requests), 1)
        self.assertEqual(out, ['ran read_file', 'ran shell_executor'])

    def test_nameless_schema_is_refused(self):
        with self.assertRaises(ValueError):
            EnvTool.from_schemas(self.env, [{'type': 'function', 'function': {}}])


if __name__ == '__main__':
    unittest.main()
