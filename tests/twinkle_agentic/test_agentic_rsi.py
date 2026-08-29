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
import re
import shutil
import sys
import tempfile
import threading
import unittest

_REPO = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(_REPO, 'src'))
# The RSI wiring lives in cookbook, not in the framework: it is one deployment's
# choice of sandbox backend, and the tests follow it there.
_COOKBOOK = os.path.join(_REPO, 'cookbook', 'rsi', 'agentic')
sys.path.insert(0, _COOKBOOK)
sys.path.insert(0, os.path.join(_COOKBOOK, 'sandbox_server'))

from remote_tool_env import RemoteMsAgentToolEnv  # noqa: E402
from tool_server import (ToolRuntime, _usable_llm,  # noqa: E402
                         _without_internal_args, _without_llm_args)
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
            # The request file name carries a uuid, so that concurrent calls cannot
            # overwrite each other's payload. Read the one this command names
            # rather than a fixed path: reading a fixed path is what would keep
            # passing after the Env went back to a shared file.
            match = re.search(r'--data-binary @(\S+)', command)
            if not match:
                raise AssertionError(f'call command names no request file: {command}')
            payload = json.loads(self.files.store[match.group(1)])
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
        # Advertised without ms-agent's `{server}---` prefix: a 4B policy wrote a
        # bare `shell_executor` 7 times across three arms and lost the turn to
        # "unknown tool". The prefix carries nothing it can act on.
        self.assertEqual(
            self.env.tool_names(),
            [t['function']['name'].rsplit('---', 1)[-1] for t in DEFAULT_TOOLS])

    def test_short_name_is_expanded_before_dispatch(self):
        # The runtime only answers to its own spelling, so the prefix has to come
        # back on the way out. Shortening the advertised name without this would
        # make every call fail.
        self.env.step('shell_executor', {'command': 'ls'})
        sent = [c['tool_name'] for c in self.sandbox.requests[-1]['calls']]
        self.assertEqual(sent, ['code_executor---shell_executor'])

    def test_resolve_tool_accepts_either_spelling(self):
        self.assertEqual(self.env.resolve_tool('shell_executor'), 'shell_executor')
        # A caller written before the names were shortened still resolves.
        self.assertEqual(self.env.resolve_tool('code_executor---shell_executor'),
                         'shell_executor')

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

    def test_host_owned_call_id_is_never_advertised(self):
        # ms-agent declares __call_id on shell_executor as "injected by host when
        # supported". In the prompt it reads as an argument the model may choose.
        schema = {
            'type': 'function',
            'function': {
                'name': 'code_executor---shell_executor',
                'parameters': {
                    'properties': {'command': {}, '__call_id': {}},
                    'required': ['command', '__call_id'],
                },
            },
        }
        stripped = _without_internal_args(schema)
        self.assertEqual(sorted(stripped['function']['parameters']['properties']), ['command'])
        self.assertEqual(stripped['function']['parameters']['required'], ['command'])
        self.assertIn('__call_id', schema['function']['parameters']['properties'])


class ToolCallReconcileTest(unittest.TestCase):
    """Two failures arrive as one TypeError; only one of them is the model's.

    ms-agent asking for an argument its own tool cannot take is a bug and is
    repaired silently. The model reaching for another tool's arguments is not,
    and is refused -- with the name of the tool it should have called, because a
    call that is quietly rewritten teaches a shape that fails outside this
    sandbox.
    """

    # The real line-up, reduced to the two arguments each case turns on.
    CONTRACTS = {
        'file_system---write_file': ({'path', 'content'}, {'path', 'content'}),
        'file_system---edit_file': ({'path', 'old_string', 'new_string', 'replace_all'},
                                    {'path', 'old_string', 'new_string', 'replace_all'}),
        'file_system---glob': ({'pattern', 'path'}, {'pattern', 'path'}),
        'file_system---read_file': ({'path'}, {'path', 'abbreviate'}),
        'code_executor---shell_executor': ({'command', 'run_in_background', 'timeout'},
                                           {'command', 'run_in_background', 'timeout', 'call_id'}),
        'code_executor---python_executor': ({'code', 'description', 'timeout'},
                                            {'code', 'description', 'timeout'}),
    }

    def setUp(self):
        # No ms-agent, no kernel: _reconcile only reads the contract table, and
        # building a real runtime here would need a microVM's worth of setup.
        self.runtime = ToolRuntime.__new__(ToolRuntime)
        self.runtime._contracts = dict(self.CONTRACTS)

    def test_framework_timeout_is_dropped_for_tools_without_one(self):
        # ms-agent's own timeout message tells the model to pass `timeout` in the
        # tool arguments; write_file has no such parameter and raises TypeError.
        args, error = self.runtime._reconcile('file_system---write_file',
                                             {'path': 'a.txt', 'content': 'x', 'timeout': 30})
        self.assertIsNone(error)
        self.assertEqual(args, {'path': 'a.txt', 'content': 'x'})

    def test_description_is_dropped_for_the_sibling_that_lacks_it(self):
        args, error = self.runtime._reconcile('code_executor---shell_executor',
                                             {'command': 'ls', 'description': 'list'})
        self.assertIsNone(error)
        self.assertEqual(args, {'command': 'ls'})

    def test_declared_arguments_are_left_alone(self):
        call = {'code': 'print(1)', 'description': 'demo', 'timeout': 20}
        args, error = self.runtime._reconcile('code_executor---python_executor', dict(call))
        self.assertIsNone(error)
        self.assertEqual(args, call)

    def test_empty_glob_path_becomes_the_workspace_root(self):
        # '' is glob's own default, but ms-agent's safety guard rejects it as an
        # empty file path before the tool is reached.
        args, error = self.runtime._reconcile('file_system---glob', {'pattern': '*', 'path': ''})
        self.assertIsNone(error)
        self.assertEqual(args, {'pattern': '*', 'path': '.'})

    def test_edit_file_arguments_on_write_file_are_refused_by_name(self):
        args, error = self.runtime._reconcile(
            'file_system---write_file',
            {'path': 'a.py', 'old_string': '', 'new_string': 'print(1)'})
        self.assertIsNotNone(error)
        # The message has to carry three things: what was rejected, what this
        # tool takes, and who owns the arguments that were passed.
        self.assertIn("'new_string', 'old_string'", error)
        self.assertIn('It accepts: content, path.', error)
        self.assertIn('file_system---edit_file', error)
        # Not repaired into a content= write: that is the mistake being reported.
        self.assertNotIn('content', args)

    def test_withdrawn_argument_is_refused_rather_than_attempted(self):
        # abbreviate exists on the method but was withdrawn from the schema
        # because this sandbox has no LLM to serve it.
        _args, error = self.runtime._reconcile('file_system---read_file',
                                              {'path': 'a.txt', 'abbreviate': True})
        self.assertIn("has no argument 'abbreviate'", error)

    def test_unknown_tool_is_left_for_ms_agent_to_report(self):
        call = {'anything': 1}
        args, error = self.runtime._reconcile('file_system---nope', dict(call))
        self.assertIsNone(error)
        self.assertEqual(args, call)


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
        # read_file has no prefix to restore; shell_executor does, and the
        # FakeSandbox echoes whatever name reached it.
        self.assertEqual(out, ['ran read_file', 'ran code_executor---shell_executor'])

    def test_nameless_schema_is_refused(self):
        with self.assertRaises(ValueError):
            EnvTool.from_schemas(self.env, [{'type': 'function', 'function': {}}])


class EmptyWorkspaceTest(unittest.TestCase):
    """An episode that left nothing behind must not become a task.

    When the explorer writes no files, the only assertion true of the end state is
    that the directory is empty -- and every solver satisfies that by doing
    nothing, so the task scores 4 of 4 and teaches nothing. Five of the ten
    verified tasks in one generation run were exactly this.
    """

    def _challenger(self, snapshot):
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts

        self.checks_run = []

        def explorer(trajectories, **kwargs):
            return [{'messages': list(t['messages'])
                     + [{'role': 'assistant', 'content': '```python\nassert True\n```'}]}
                    for t in trajectories]

        def run_check_fn(script, slot=0):
            self.checks_run.append(script)
            return 0, ''

        prompts = AgenticPrompts(
            system='s', from_scratch='u',
            check_followup='write checks for {final_state}',
            check_retry_followup='{error} / {final_state}',
            problem_followup='write the statement')
        return AgenticChallenger(
            prompts, explorer,
            reset_fn=lambda slot=0: None,
            run_check_fn=run_check_fn,
            workspace_snapshot_fn=lambda slot=0: snapshot,
            solver_rollouts=0,
        )

    def _explored(self):
        return {'messages': [{'role': 'user', 'content': 'do something'},
                             {'role': 'assistant', 'content': 'I made three files.'}]}

    def test_empty_snapshot_ends_the_episode_before_any_check_is_written(self):
        ch = self._challenger('')
        state = {}
        # None means "nothing more to say": the rollout ends the episode here.
        self.assertIsNone(ch._followup(state, self._explored(), 0))
        self.assertEqual(ch.stats['empty_workspace'], 1)
        self.assertEqual(state['reject'][0], 'empty_workspace')
        # No check script was even run: there was nothing to check.
        self.assertEqual(self.checks_run, [])
        # And the episode is not turned into a task afterwards.
        self.assertIsNone(ch._finish_episode(state, self._explored()))

    def test_whitespace_only_snapshot_counts_as_empty(self):
        ch = self._challenger('   \n  ')
        state = {}
        self.assertIsNone(ch._followup(state, self._explored(), 0))
        self.assertEqual(ch.stats['empty_workspace'], 1)

    def test_a_real_snapshot_asks_for_checks_and_then_runs_them(self):
        ch = self._challenger('data.csv 15\n\n--- data.csv ---\nA,B\n1,2')
        state = {}
        text, params = ch._followup(state, self._explored(), 0)
        self.assertEqual(ch.stats['empty_workspace'], 0)
        # The listing reaches the model verbatim -- it is the ground truth the
        # checks are written against.
        self.assertIn('--- data.csv ---', text)
        self.assertIsNone(params)
        self.assertEqual(self.checks_run, [])

        wrote_script = {'messages': [
            {'role': 'assistant', 'content': '```python\nassert True\n```'}]}
        self.assertEqual(ch._followup(state, wrote_script, 1),
                         ('write the statement', None))
        self.assertEqual(self.checks_run, ['assert True'])


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


class EpisodeStagesTest(unittest.TestCase):
    """One conversation carries the work, the checks and the statement.

    The three used to be three separate calls, which meant only the last one's
    tokens were trainable. The fake explorer here plays the part
    ``MultiTurnRollout`` plays for real: it appends whatever ``followup_fn``
    returns and keeps generating until it returns None.
    """

    def _challenger(self, replies, snapshot='a.txt 1\n\n--- a.txt ---\nx',
                    check_exit=0, check_exits=None, **kwargs):
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts

        self.emitted = []
        self.rejected = []
        self.appended = []
        # One exit code per check run, so a test can make the first fail and the
        # rewrite pass.
        exits = list(check_exits) if check_exits is not None else None

        def run_check(script, slot=0):
            code = exits.pop(0) if exits else check_exit
            return (code, 'AssertionError' if code else '')

        def explorer(trajectories, **kw):
            followup_fn = kw.get('followup_fn')
            traj = {'messages': list(trajectories[0]['messages']), 'input_ids': [1, 2, 3]}
            for reply in replies:
                traj['messages'].append({'role': 'assistant', 'content': reply})
                if followup_fn is None:
                    break
                out = followup_fn(traj, len(self.appended))
                if out is None:
                    break
                text, _params = out
                self.appended.append(text)
                traj['messages'].append({'role': 'user', 'content': text})
            return [traj]

        prompts = AgenticPrompts(
            system='s', from_scratch='u',
            check_followup='checks please: {final_state}',
            check_retry_followup='it failed: {error} / state: {final_state}',
            problem_followup='statement please')
        return AgenticChallenger(
            prompts, explorer,
            reset_fn=lambda slot=0: None,
            run_check_fn=run_check,
            workspace_snapshot_fn=lambda slot=0: snapshot,
            reject_sink=self.rejected.append,
            propose_sink=self.emitted.append,
            solver_rollouts=0,
            **kwargs)

    def test_one_episode_yields_the_script_and_the_statement(self):
        from twinkle.data_format import user_data_get
        ch = self._challenger(['Done.', '```python\nassert True\n```',
                               'Create a.txt holding x.'])
        kept = ch._round(1)

        self.assertEqual(len(kept), 1)
        self.assertEqual(user_data_get(kept[0].get('user_data'), 'check_script'),
                         'assert True')
        self.assertEqual(kept[0]['messages'][-1]['content'], 'Create a.txt holding x.')
        # Both stages were asked for, in order, in the same conversation.
        self.assertEqual(len(self.appended), 2)
        self.assertIn('--- a.txt ---', self.appended[0])
        self.assertEqual(self.appended[1], 'statement please')
        # One record, not three: one conversation has one set of token ids.
        self.assertEqual([r['stage'] for r in self.emitted[0]['rounds']], ['episode'])
        self.assertEqual(self.emitted[0]['outcome'], 'kept')

    def test_a_check_that_fails_on_its_own_workspace_stops_before_the_statement(self):
        ch = self._challenger(['Done.', '```python\nassert False\n```',
                               'never asked for'], check_exit=1, check_retries=0)
        kept = ch._round(1)

        self.assertEqual(kept, [])
        self.assertEqual(ch.stats['check_run_fail'], 1)
        self.assertEqual(len(self.appended), 1)
        self.assertEqual(self.rejected[0]['reason'], 'check_run_fail')
        # The record has to say what the workspace held when the check ran.
        self.assertIn('--- state before check ---', self.rejected[0]['detail'])
        # Rejected attempts are dumped too: they are the zero-reward half of a
        # GRPO group.
        self.assertEqual(self.emitted[0]['outcome'], 'check_run_fail')

    def test_a_failing_check_gets_one_rewrite_and_the_episode_carries_on(self):
        """29 of ex12's 36 check failures were one assertion, on a state that was
        fine; the rewrite reads the traceback and the listing."""
        from twinkle.data_format import user_data_get
        ch = self._challenger(['Done.',
                               '```python\nassert len(rows) == 5\n```',
                               '```python\nassert len(rows) == 3\n```',
                               'Create a.txt holding x.'],
                              check_exits=[1, 0])
        kept = ch._round(1)

        self.assertEqual(len(kept), 1)
        # The task ships the script that passed, not the first one.
        self.assertEqual(user_data_get(kept[0].get('user_data'), 'check_script'),
                         'assert len(rows) == 3')
        self.assertEqual(ch.stats['check_retry'], 1)
        self.assertEqual(ch.stats['check_retry_pass'], 1)
        self.assertEqual(ch.stats['check_run_fail'], 0)
        # checks -> rewrite -> statement, and the rewrite was told what broke.
        self.assertEqual(len(self.appended), 3)
        self.assertIn('AssertionError', self.appended[1])
        self.assertEqual(self.appended[2], 'statement please')

    def test_a_rewrite_that_fails_too_is_rejected_with_both_attempts(self):
        ch = self._challenger(['Done.',
                               '```python\nassert False\n```',
                               '```python\nassert False\n```',
                               'never asked for'],
                              check_exits=[1, 1])
        kept = ch._round(1)

        self.assertEqual(kept, [])
        self.assertEqual(ch.stats['check_run_fail'], 1)
        self.assertEqual(ch.stats['check_retry_pass'], 0)
        detail = self.rejected[0]['detail']
        self.assertIn('--- attempt 1:', detail)
        self.assertIn('--- attempt 2:', detail)

    def test_a_rewrite_that_never_arrives_is_not_shipped_as_a_task(self):
        """The failed script is still in the scratchpad when the episode dies."""
        ch = self._challenger(['Done.', '```python\nassert False\n```'],
                              check_exits=[1])
        kept = ch._round(1)

        self.assertEqual(kept, [])
        self.assertEqual(ch.stats['episode_cut_short'], 1)
        self.assertEqual(self.rejected[0]['reason'], 'episode_cut_short')

    def test_an_episode_that_never_reached_the_stages_is_recorded_as_cut_short(self):
        # The explorer returns after its single reply without consulting the
        # callback, which is what a rollout does when the episode ran out of turns.
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts

        self.emitted, self.rejected = [], []

        def explorer(trajectories, **kw):
            return [{'messages': list(trajectories[0]['messages'])
                     + [{'role': 'assistant', 'content': 'half a thought'}],
                     'truncated': True, 'stop_reason': 'length'}]

        ch = AgenticChallenger(
            AgenticPrompts(system='s', from_scratch='u',
                           check_followup='c {final_state}',
                           check_retry_followup='{error} / {final_state}',
                           problem_followup='p'),
            explorer,
            reset_fn=lambda slot=0: None,
            run_check_fn=lambda script, slot=0: (0, ''),
            workspace_snapshot_fn=lambda slot=0: 'a.txt 1\n',
            reject_sink=self.rejected.append,
            propose_sink=self.emitted.append,
            solver_rollouts=0)
        kept = ch._round(1)

        self.assertEqual(kept, [])
        self.assertEqual(ch.stats['episode_cut_short'], 1)
        self.assertEqual(self.rejected[0]['reason'], 'episode_cut_short')
        self.assertIn('truncated=True', self.rejected[0]['detail'])


class ConcurrentEpisodeSlotsTest(unittest.TestCase):
    """Concurrent episodes must each drive their own sandbox slot.

    A rack of one sandbox per slot is the whole point of running episodes in
    parallel; if the slot the challenger passes for episode i is not the slot
    reset_fn / run_check_fn / workspace_snapshot_fn / tool_manager see for that
    episode, then two episodes end up sharing a workspace and the check written
    against one runs against the other. That is the failure mode this test
    exists to catch.
    """

    def test_each_episode_uses_its_own_slot_end_to_end(self):
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts

        n_slots = 4
        resets = []  # (slot,) per call
        checks = []  # (slot, script) per call
        snaps = []   # (slot,) per call
        tm_calls = []  # (slot,) per tool_manager use
        lock = threading.Lock()

        class FakeTM:
            def __init__(self, slot): self.slot = slot
            def tool_infos(self): return []
            def __call__(self, tc):
                with lock:
                    tm_calls.append(self.slot)
                return 'ok'

        tool_managers = [FakeTM(i) for i in range(n_slots)]

        def reset_fn(slot):
            with lock:
                resets.append(slot)

        def run_check_fn(script, slot):
            with lock:
                checks.append((slot, script))
            return 0, ''

        def workspace_snapshot_fn(slot):
            with lock:
                snaps.append(slot)
            # Encode the slot in the snapshot so an episode reading the wrong
            # slot's workspace would produce a mismatched check statement.
            return f'slot_{slot}.txt 1\n\n--- slot_{slot}.txt ---\nx'

        def explorer(trajectories, **kw):
            # The two follow-ups (check script, then statement) are threaded
            # through the callback so the slot-aware handlers actually run.
            tm = kw.get('tool_manager')
            if tm is not None:
                tm({'id': 'x', 'type': 'function',
                    'function': {'name': 'noop', 'arguments': '{}'}})
            traj = {'messages': list(trajectories[0]['messages']),
                    'input_ids': [1, 2, 3]}
            followup = kw.get('followup_fn')
            replies = ['```python\nassert True\n```',
                       'Statement:\n\n```\ndo the thing\n```']
            for i, reply in enumerate(replies):
                traj['messages'].append({'role': 'assistant', 'content': reply})
                if followup is None:
                    break
                out = followup(traj, i)
                if out is None:
                    break
                text, _params = out
                traj['messages'].append({'role': 'user', 'content': text})
            return [traj]

        emitted = []
        prompts = AgenticPrompts(
            system='s', from_scratch='u {keywords}',
            check_followup='c {final_state}',
            check_retry_followup='{error} / {final_state}',
            problem_followup='p')
        ch = AgenticChallenger(
            prompts, explorer,
            reset_fn=reset_fn,
            run_check_fn=run_check_fn,
            workspace_snapshot_fn=workspace_snapshot_fn,
            episode_concurrency=n_slots,
            episode_tool_managers=tool_managers,
            propose_sink=emitted.append,
            solver_rollouts=0,
            max_proposals_per_round=8,
        )
        kept = ch._round(8)

        self.assertEqual(len(kept), 8)
        # 8 episodes across 4 slots, evenly split -> each slot reset twice, ran
        # its own check twice, and every check saw the slot's own snapshot text.
        from collections import Counter
        self.assertEqual(Counter(resets), Counter({0: 2, 1: 2, 2: 2, 3: 2}))
        self.assertEqual(Counter(s for s, _ in checks), Counter({0: 2, 1: 2, 2: 2, 3: 2}))
        # The tool_manager slot used matches the check slot for each episode.
        self.assertEqual(Counter(tm_calls), Counter({0: 2, 1: 2, 2: 2, 3: 2}))

    def test_wrong_tool_manager_count_is_refused(self):
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts
        prompts = AgenticPrompts(system='s', from_scratch='u',
                                 check_followup='c {final_state}',
                                 check_retry_followup='{error} / {final_state}',
                                 problem_followup='p')
        with self.assertRaises(ValueError):
            AgenticChallenger(
                prompts, explorer=lambda t, **k: t,
                reset_fn=lambda slot=0: None,
                run_check_fn=lambda s, slot=0: (0, ''),
                workspace_snapshot_fn=lambda slot=0: '',
                episode_concurrency=4,
                episode_tool_managers=[object(), object()],  # wrong count
                solver_rollouts=0)


class PreseedInputsTest(unittest.TestCase):
    """A task carrying a setup script has it replayed before every attempt.

    Order is the whole point: clear, then write the inputs back, then let the
    solver run. Replaying before the clear would delete the files it just wrote,
    and skipping the replay would measure the task against a workspace missing the
    data its statement says is there -- which reads as 'too hard' and is not.
    """

    def _challenger(self, run_check_fn, **kwargs):
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts

        prompts = AgenticPrompts(
            system='s', from_scratch='u',
            check_followup='c {final_state}',
            check_retry_followup='{error} / {final_state}',
            problem_followup='p')
        return AgenticChallenger(
            prompts,
            lambda trajs, **kw: [{'messages': list(t['messages']), 'stop_reason': 'stop'}
                                 for t in trajs],
            run_check_fn=run_check_fn,
            workspace_snapshot_fn=lambda slot=0: 'input/a.csv 3\n',
            solver_rollouts=2,
            keep_min_pass=1,
            keep_max_pass_margin=0,
            propose_sink=[].append,
            **kwargs)

    def _task(self, setup):
        from twinkle_agentic.challenger.base import attach_user_data
        return attach_user_data({'messages': [{'role': 'user', 'content': 'q'}]},
                                check_script='assert True', setup_script=setup,
                                keywords=[])

    def test_setup_runs_after_the_clear_and_before_the_check(self):
        events = []
        ch = self._challenger(
            run_check_fn=lambda script, slot=0: (events.append(
                'setup' if script.startswith('#SETUP') else 'check'), (0, ''))[1],
            reset_fn=lambda slot=0: events.append('clear'))
        kept = ch._filter_difficulty([self._task('#SETUP\nopen("a","w")')])
        self.assertEqual(events, ['clear', 'setup', 'check'] * 2)
        self.assertEqual(len(kept), 1)

    def test_failed_setup_skips_the_attempt_instead_of_scoring_it_zero(self):
        """An attempt that never ran must not be counted as an attempt that failed."""
        checks = []

        def run_check(script, slot=0):
            if script.startswith('#SETUP'):
                return 1, 'no space left on device'
            checks.append(script)
            return 0, ''

        ch = self._challenger(run_check_fn=run_check, reset_fn=lambda slot=0: None)
        kept = ch._filter_difficulty([self._task('#SETUP\nboom')])
        # The solver was never asked, so nothing was checked and nothing is kept.
        self.assertEqual(checks, [])
        self.assertEqual(kept, [])
        self.assertEqual(ch.stats['setup_replay_fail'], 2)


class ParallelDifficultyTest(unittest.TestCase):
    """Solver attempts run in waves, each attempt isolated in its own sandbox.

    The measurement is only a measurement if attempt A cannot pass on files
    attempt B wrote, so what this pins down is that within one wave the clear,
    the tool dispatch and the check all reach the *same* slot for a given
    attempt, and that a wave is one batched explorer call rather than one call
    per attempt (which is what left the GPUs idle).
    """

    def test_attempts_are_batched_per_wave_and_stay_in_their_slot(self):
        from twinkle.data_format import user_data_get
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts
        from twinkle_agentic.challenger.base import attach_user_data

        n_slots = 4
        lock = threading.Lock()
        batch_sizes = []      # trajectories per explorer call
        reset_slots = []      # slot per reset
        pairs = []            # (tool_manager slot, check slot) per attempt

        class FakeTM:
            def __init__(self, slot): self.slot = slot
            def tool_infos(self): return []
            def __call__(self, tc): return 'ok'

        tool_managers = [FakeTM(i) for i in range(n_slots)]
        # Which slot's manager each trajectory of the current wave was handed.
        wave_slots = []

        def explorer(trajectories, **kw):
            tms = kw.get('tool_manager')
            with lock:
                batch_sizes.append(len(trajectories))
                wave_slots.clear()
                wave_slots.extend([tm.slot for tm in (tms or [])])
            return [{'messages': list(t['messages']), 'stop_reason': 'stop'}
                    for t in trajectories]

        # One attempt in flight per slot, so the check for the attempt that used
        # slot k must itself run in slot k. Recorded as a pair to compare.
        seq = iter(range(10_000))

        def run_check_fn(script, slot=0):
            with lock:
                pairs.append(slot)
                next(seq)
            return 0, ''

        def reset_fn(slot=0):
            with lock:
                reset_slots.append(slot)

        prompts = AgenticPrompts(
            system='s', from_scratch='u',
            check_followup='c {final_state}',
            check_retry_followup='{error} / {final_state}',
            problem_followup='p')
        ch = AgenticChallenger(
            prompts, explorer,
            reset_fn=reset_fn,
            run_check_fn=run_check_fn,
            workspace_snapshot_fn=lambda slot=0: 'a.txt 1\n',
            episode_concurrency=n_slots,
            episode_tool_managers=tool_managers,
            solver_rollouts=4,
            keep_min_pass=1,
            keep_max_pass_margin=0,
            propose_sink=[].append,
        )
        tasks = [attach_user_data({'messages': [{'role': 'user', 'content': f'task {i}'}]},
                                  check_script='assert True', keywords=[])
                 for i in range(2)]
        kept = ch._filter_difficulty(tasks)

        # 2 tasks x 4 attempts = 8 attempts, 4 slots -> two waves of 4, each a
        # single explorer call. Serial code would have made 8 calls of 1.
        self.assertEqual(batch_sizes, [4, 4])
        # Every slot cleared once per wave, and the managers handed out are the
        # slots that were cleared.
        self.assertEqual(sorted(reset_slots), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(sorted(wave_slots), [0, 1, 2, 3])
        # One check per attempt, one per slot per wave.
        self.assertEqual(sorted(pairs), [0, 0, 1, 1, 2, 2, 3, 3])
        # All checks passed -> both tasks scored 4 of 4.
        self.assertEqual([user_data_get(t.get('user_data'), 'n_pass', -1) for t in kept],
                         [4, 4])


class TruncatedSolverTest(unittest.TestCase):
    """A solver attempt cut off at its token budget has to be countable.

    It is still scored as a failure -- whether to discount it decides which tasks
    are kept, which is not this code's call -- but the count is what says whether
    ``n_pass`` measured difficulty or the token budget. On one run 15 of 50
    attempts ended that way, all 15 with an untouched workspace.
    """

    def _challenger(self, attempt_flags, **kwargs):
        """``attempt_flags``: one (truncated, passes) pair per solver attempt."""
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts

        self.flags = list(attempt_flags)
        self.emitted = []

        def explorer(trajectories, **kw):
            truncated, _ = self.flags[0]
            return [{'messages': list(t['messages']),
                     'truncated': truncated,
                     'stop_reason': 'length' if truncated else 'stop'}
                    for t in trajectories]

        def run_check_fn(script, slot=0):
            _, passes = self.flags.pop(0)
            return (0 if passes else 1), ''

        prompts = AgenticPrompts(
            system='s', from_scratch='u',
            check_followup='cs {final_state}',
            check_retry_followup='{error} / {final_state}', problem_followup='ps')
        return AgenticChallenger(
            prompts, explorer,
            reset_fn=lambda slot=0: None,
            run_check_fn=run_check_fn,
            workspace_snapshot_fn=lambda slot=0: 'a.txt 1\n',
            propose_sink=self.emitted.append,
            solver_rollouts=4,
            **kwargs)

    def _task(self):
        from twinkle.data_format import pack_user_data
        return {'messages': [{'role': 'user', 'content': 'make a.txt'}],
                'user_data': pack_user_data({'check_script': 'assert True'}),
                'propose_rounds': [{'input_ids': [1]}]}

    def test_truncated_attempts_are_counted_and_still_scored_as_failures(self):
        # 1 pass, 1 honest failure, 2 truncated failures -> 1 of 4, and the two
        # truncations visible in stats so the 1-of-4 can be read for what it is.
        ch = self._challenger([(False, True), (False, False),
                               (True, False), (True, False)],
                              keep_min_pass=1, keep_max_pass_margin=1)
        kept = ch._filter_difficulty([self._task()])

        self.assertEqual(ch.stats['solver_truncated'], 2)
        self.assertEqual(self.emitted[0]['n_pass'], 1)
        self.assertEqual(self.emitted[0]['n_rollouts'], 4)
        self.assertEqual(self.emitted[0]['outcome'], 'kept')
        self.assertEqual(len(kept), 1)

    def test_all_four_truncated_reads_as_nobody_solved_it(self):
        # Pinned as the known cost of scoring them as failures: this task is
        # discarded for being too hard and stats['solver_truncated'] == 4 is the
        # only thing that says no solver ever acted.
        ch = self._challenger([(True, False)] * 4,
                              keep_min_pass=1, keep_max_pass_margin=1)
        kept = ch._filter_difficulty([self._task()])

        self.assertEqual(ch.stats['solver_truncated'], 4)
        self.assertEqual(self.emitted[0]['n_pass'], 0)
        self.assertEqual(self.emitted[0]['outcome'], 'outside_band')
        self.assertEqual(kept, [])


class KeywordBankTest(unittest.TestCase):
    """The bank has to actually fill, and say so when it does not.

    It went empty for whole runs: the agentic prompt asked for "one keyword per
    line" while ``parse_keyword_list`` reads a JSON array, so every generation
    call parsed to nothing, ``_refill`` returned silently, and all 17 proposals in
    one run ran the no-keyword prompt with no log line to say so.
    """

    def _challenger(self, reply_text):
        from twinkle_agentic.challenger import KeywordStore
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts
        from prompts import KEYWORD_EXPAND_USER, KEYWORD_SYSTEM, KEYWORD_USER

        self.gen_records = []
        self.tool_explorer_calls = 0

        def tool_explorer(trajectories, **kwargs):
            self.tool_explorer_calls += 1
            return [{'messages': list(t['messages'])
                     + [{'role': 'assistant', 'content': reply_text}]}
                    for t in trajectories]

        def text_explorer(trajectories, **kwargs):
            return [{'messages': list(t['messages'])
                     + [{'role': 'assistant', 'content': reply_text}],
                     'stop_reason': 'stop'}
                    for t in trajectories]

        self.store = KeywordStore(os.path.join(self.tmp, 'kw.jsonl'), ('filesystem',))
        prompts = AgenticPrompts(
            system='s', from_scratch='u', from_keywords='dir:\n{keywords}',
            check_followup='cs {final_state}',
            check_retry_followup='{error} / {final_state}', problem_followup='ps',
            keyword_system=KEYWORD_SYSTEM, keyword_user=KEYWORD_USER,
            keyword_expand_user=KEYWORD_EXPAND_USER)
        return AgenticChallenger(
            prompts, tool_explorer,
            keyword_store=self.store,
            category_desc={'filesystem': 'files and directories'},
            reset_fn=lambda slot=0: None,
            run_check_fn=lambda script, slot=0: (0, ''),
            workspace_snapshot_fn=lambda slot=0: 'a.txt 1\n',
            keyword_explorer=text_explorer,
            keyword_sink=self.gen_records.append,
            keyword_gen_calls=1,
            keyword_refill_target=4,
            solver_rollouts=0)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='kwbank_test_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_shipped_prompt_asks_for_what_the_parser_reads(self):
        """The real prompt string, not a stand-in: this is the contract that broke."""
        from twinkle_agentic.challenger.code import parse_keyword_list
        from prompts import KEYWORD_EXPAND_USER, KEYWORD_USER

        for text in (KEYWORD_USER, KEYWORD_EXPAND_USER):
            self.assertIn('JSON array', text)
        # And what a model following that instruction returns must parse.
        self.assertEqual(
            parse_keyword_list('["csv deduplication", "log rotation"]'),
            ['csv deduplication', 'log rotation'])
        # While the format the prompt used to ask for does not -- so a future
        # rewording back to one-per-line fails here rather than in a night's run.
        self.assertEqual(parse_keyword_list('csv deduplication\nlog rotation'), [])

    def test_a_json_reply_fills_the_bank_and_reaches_the_proposal(self):
        from twinkle.data_format import user_data_get
        ch = self._challenger('["csv deduplication", "log rotation"]')
        proposals = ch.propose(1)

        self.assertTrue(self.store.texts('filesystem'))
        picks = user_data_get(proposals[0].get('user_data'), 'keywords', [])
        self.assertTrue(picks, 'the drawn keywords must reach the proposal')
        self.assertIn(picks[0][1], proposals[0]['messages'][-1]['content'])

    def test_keyword_generation_does_not_use_the_tool_explorer(self):
        ch = self._challenger('["csv deduplication"]')
        ch.propose(1)
        # Brainstorming a list needs no sandbox, and a bracketed list in the reply
        # is exactly what the tool explorer would try to dispatch.
        self.assertEqual(self.tool_explorer_calls, 0)

    def test_an_unparseable_reply_is_recorded_rather_than_swallowed(self):
        from twinkle.data_format import user_data_get
        ch = self._challenger('csv deduplication\nlog rotation')
        proposals = ch.propose(1)

        self.assertEqual(user_data_get(proposals[0].get('user_data'), 'keywords', []), [])
        self.assertTrue(self.gen_records, 'the keyword sink must see the failing call')
        rec = self.gen_records[0]
        self.assertEqual(rec['n_parsed'], 0)
        self.assertEqual(rec['reply'], 'csv deduplication\nlog rotation')
        self.assertIn('JSON array', rec['prompt'])


class SerialKeywordRefillTest(unittest.TestCase):
    """A refill's calls go out one at a time, each told what the earlier ones said.

    Batched, the calls were identical but for a trailing index, and the 'do not
    repeat these' list could only name what the bank already held -- which on a
    first refill is nothing. Measured on armD: all eight parallel calls answered
    with the same three phrases ('aggregating data', 'processing data',
    'generating a single output file'), and 22 of that run's 24 drawn phrases came
    from that one batch. So what is pinned here is not that the code is serial but
    the reason it is: call k+1 must be able to see call k's output.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='kwserial_test_')
        self.seen = []          # the user message of every call, in order
        self.batch_sizes = []   # trajectories per call

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _challenger(self, prompts, category, refill_concurrency=1):
        from twinkle_agentic.challenger import KeywordStore
        from twinkle_agentic.challenger.agentic import AgenticChallenger

        n_kw = [0]

        def explorer(trajectories, **kwargs):
            self.batch_sizes.append(len(trajectories))
            out = []
            for t in trajectories:
                user = t['messages'][-1]['content']
                self.seen.append(user)
                if 'KIND of work' in user:
                    reply = f'["kind {n_kw[0]}"]'
                    n_kw[0] += 1
                elif 'JSON array' in user:
                    reply = f'["topic {n_kw[0]}"]'
                    n_kw[0] += 1
                else:
                    reply = 'A draft task: read some files and compute something.'
                out.append({'messages': list(t['messages'])
                            + [{'role': 'assistant', 'content': reply}],
                            'stop_reason': 'stop'})
            return out

        self.store = KeywordStore(os.path.join(self.tmp, 'kw.jsonl'), (category,))
        return AgenticChallenger(
            prompts, explorer,
            keyword_store=self.store,
            category_desc={category: 'some kind of work'},
            reset_fn=lambda slot=0: None,
            run_check_fn=lambda script, slot=0: (0, ''),
            workspace_snapshot_fn=lambda slot=0: 'a.txt 1\n',
            keyword_explorer=explorer,
            keyword_gen_calls=3,
            keyword_refill_concurrency=refill_concurrency,
            min_batch=1,
            solver_rollouts=0)

    def _three_axis_prompts(self):
        from twinkle_agentic.challenger.agentic import AgenticPrompts
        from prompts import KEYWORD_EXPAND_USER, KEYWORD_SYSTEM, KEYWORD_USER

        return AgenticPrompts(
            system='s', from_scratch='u', from_keywords='dir:\n{keywords}',
            check_followup='cs {final_state}',
            check_retry_followup='{error} / {final_state}', problem_followup='ps',
            keyword_system=KEYWORD_SYSTEM, keyword_user=KEYWORD_USER,
            keyword_expand_user=KEYWORD_EXPAND_USER)

    def test_three_axis_refill_shows_each_call_the_previous_output(self):
        ch = self._challenger(self._three_axis_prompts(), 'transform')

        got = ch._generate_keywords('transform', 9)

        self.assertEqual(sorted(got), ['topic 0', 'topic 1', 'topic 2'])
        self.assertEqual(self.batch_sizes, [1, 1, 1],
                         'one call at a time, or the calls cannot see each other')
        self.assertNotIn('topic 0', self.seen[0], 'nothing exists yet for the first call')
        self.assertIn('topic 0', self.seen[1])
        for kw in ('topic 0', 'topic 1'):
            self.assertIn(kw, self.seen[2])


    def test_raising_the_concurrency_restores_the_batched_behaviour(self):
        """At n_calls in flight, no call can see any other -- the first round's setup.

        Kept measurable on one build: the arms were first compared with a whole
        refill going out at once, and telling that apart from what serial produces
        means being able to run both without checking out an older file.
        """
        ch = self._challenger(self._three_axis_prompts(), 'transform',
                             refill_concurrency=3)

        got = ch._generate_keywords('transform', 9)

        self.assertEqual(sorted(got), ['topic 0', 'topic 1', 'topic 2'])
        self.assertEqual(self.batch_sizes, [3], 'all three go out as one batch')
        for user in self.seen:
            for kw in ('topic 0', 'topic 1', 'topic 2'):
                self.assertNotIn(kw, user, 'a batched call cannot see its siblings')

    def test_the_avoid_list_is_capped_and_drops_older_entries_first(self):
        """The cap evicts banked phrases before this refill's own, and holds a ceiling.

        Both halves matter. Capping by sampling the whole list would start dropping
        exactly what this refill just produced, and the serial ordering would buy
        nothing. Not capping at all is what made the eighth call of armA2ser's
        edge_case refill invent 'îRAPIÓN holistic replace' and nine other
        non-phrases: 150 quoted phrases left it no room to answer.
        """
        from twinkle_agentic.challenger.agentic import AgenticPrompts
        from prompts import KEYWORD_EXPAND_USER, KEYWORD_SYSTEM, KEYWORD_USER

        prompts = AgenticPrompts(
            system='s', from_scratch='u', from_keywords='dir:\n{keywords}',
            check_followup='cs {final_state}',
            check_retry_followup='{error} / {final_state}', problem_followup='ps',
            keyword_system=KEYWORD_SYSTEM, keyword_user=KEYWORD_USER,
            keyword_expand_user=KEYWORD_EXPAND_USER)
        ch = self._challenger(prompts, 'transform')
        cap = ch._AVOID_TOTAL
        older = [f'old {i}' for i in range(200)]
        fresh = [f'new {i}' for i in range(5)]
        note = ch._avoid_note(older, fresh, 'avoid: ')
        for kw in fresh:
            self.assertIn(kw, note)
        self.assertEqual(note.count('old '), cap - len(fresh))

        # Once this refill alone fills the cap, no banked phrase is quoted and the
        # line stops growing -- it is the growth that broke the eighth call.
        many = [f'new {i}' for i in range(cap + 30)]
        note = ch._avoid_note(older, many, 'avoid: ')
        self.assertEqual(note.count('old '), 0)
        self.assertEqual(note.count('new '), cap)
        self.assertNotIn('new 0', note, 'the oldest of this refill falls off first')
        self.assertIn(f'new {cap + 29}', note, 'the newest is always kept')


class ProposeTrajIndexTest(unittest.TestCase):
    """index.jsonl has to carry what the proposing side trains on.

    The challenger emits a proposal record and challenge.py copies it into
    index.jsonl field by field. Two of those fields are the reason the dump
    exists at all: train_offline.py groups proposals by ``group_id`` to get a
    GRPO advantage out of them, and skips a dump without it as a 'pre-grouping
    run'. While the copy dropped both, SIDES=both trained 384 solver and 0
    proposer trajectories, and said so only in a line nobody read.
    """

    def test_group_id_and_reward_survive_the_copy(self):
        from challenge import ProposeTrajWriter

        out = tempfile.mkdtemp(prefix='proposetraj_test_')
        try:
            writer = ProposeTrajWriter(out)
            writer.write({
                'outcome': 'kept',
                'group_id': 0,
                'challenger_reward': 0.75,
                'n_pass': 4,
                'n_rollouts': 8,
                'pass_rate': 0.5,
                'keywords': [['transform', 'parse a binary log']],
                'seeded': False,
                'rounds': [{'stage': 'episode', 'messages': [],
                            'input_ids': [1, 2], 'labels': [-100, 2],
                            'logprobs': []}],
            })
            writer.close()
            with open(os.path.join(out, 'index.jsonl'), encoding='utf-8') as f:
                rec = json.loads(f.readline())
        finally:
            shutil.rmtree(out, ignore_errors=True)

        # Group 0 is a real group, so this also pins that the copy reads the key
        # rather than testing it for truth.
        self.assertEqual(rec['group_id'], 0)
        self.assertEqual(rec['challenger_reward'], 0.75)


class TaskCarriesGroupIdTest(unittest.TestCase):
    """A built task has to remember which group proposed it.

    The difficulty stage emits kept and outside_band proposals off the *task*,
    so a task built without its group_id reaches the dump ungrouped and the
    proposing side gets no advantage from it. The reject path reads group_id off
    the episode instead, so while only the successful path dropped it, a run
    showed 4 grouped proposals -- all of them early failures -- against 92
    ungrouped kept/outside_band ones, and the copy downstream looked correct.
    """

    def _challenger(self):
        from twinkle_agentic.challenger.agentic import AgenticChallenger, AgenticPrompts

        prompts = AgenticPrompts(
            system='s', from_scratch='u',
            check_followup='write checks for {final_state}',
            check_retry_followup='{error} / {final_state}',
            problem_followup='write the statement')
        return AgenticChallenger(
            prompts,
            lambda trajectories, **kwargs: list(trajectories),
            reset_fn=lambda slot=0: None,
            run_check_fn=lambda script, slot=0: (0, ''),
            workspace_snapshot_fn=lambda slot=0: 'data.csv 3',
            solver_rollouts=0,
        )

    def test_group_id_reaches_the_built_task(self):
        from twinkle.data_format import user_data_get
        from twinkle_agentic.challenger.base import attach_user_data

        ch = self._challenger()
        explored = attach_user_data(
            {'messages': [{'role': 'user', 'content': 'explore'},
                          {'role': 'assistant', 'content': 'done'}]},
            keywords=[['transform', 'parse a binary log']], seeded=False, group_id=7)
        state = {'checked': True, 'script': 'assert True',
                 'statement': 'PROBLEM: build a parser\nEND'}

        task = ch._finish_episode(state, explored)

        self.assertIsNotNone(task, 'a checked episode with a statement is a task')
        # 7, not None: the emit sites downstream read exactly this key, and a None
        # here is what silently turned SIDES=both into solver-only training.
        self.assertEqual(user_data_get(task.get('user_data'), 'group_id', None), 7)


if __name__ == '__main__':
    unittest.main()
