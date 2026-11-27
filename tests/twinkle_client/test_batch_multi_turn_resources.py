# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run the three Blackjack entry points without model weights or external services."""

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cookbook.client.async_rl import e2b_multi_turn_grpo as e2b
from cookbook.client.async_rl import streaming_multi_turn_grpo as streaming
from cookbook.client.twinkle import multi_turn_rollout as batch
from tests.twinkle_agentic.test_token_utils import FakeTemplate, FakeTokenizer
from twinkle.data_format import validate_trajectory
from twinkle.data_format.sampling import SampleResponse, SampledSequence
from twinkle.template.contract import shift
from twinkle_agentic.utils.token_utils import append_ids
from twinkle_client.http import ClientContext, ClientTransport
from twinkle_client.manager import TwinkleClient


class OfflineTemplate(FakeTemplate):
    """Token-only wire contract (75e969ab/012aff34): the server receives only ``input_ids``,
    so the tool bridge must carry the real observation text (inherited ``Template.render_delta``
    over the char-level FakeTokenizer) and the client rebuilds the extended feature through
    ``append_completion`` / ``decode`` instead of a server-returned ``new_input_feature``."""

    def finalize_appended_features(self, rows):
        return [shift(row) for row in rows]

    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(list(tokens), **kwargs)

    def append_completion(self, prompt, tokens, **kwargs):
        feature = append_ids(prompt, list(tokens), self, trainable=True)
        feature['messages'] = list(prompt.get('messages') or []) + [
            {'role': 'assistant', 'content': self.decode(tokens)}]
        return feature

    def parse_tool_call(self, text):
        if text not in ('hit', 'stand'):
            return []
        return [{'id': 'play', 'type': 'function', 'function': {'name': 'play', 'arguments': {'action': text}}}]

    def tool_call_errors(self, text):
        return []


@pytest.fixture
def local_sandbox(monkeypatch, tmp_path):
    """Run the uploaded game in an isolated temporary directory, replacing only the E2B transport."""
    pytest.importorskip('pyspiel')
    monkeypatch.setenv('E2B_API_KEY', 'offline-test')
    sandboxes = []

    def create(_env):
        workspace = Path(mkdtemp(dir=tmp_path))
        killed = Mock()
        sandbox = SimpleNamespace(
            files=SimpleNamespace(write=lambda path, text: Path(path).write_text(text)),
            kill=killed,
        )

        def execute(command, _cwd, _timeout):
            if command.startswith('python3 -m pip install '):
                return 0, ''  # Use the locally installed OpenSpiel wheel.
            result = subprocess.run([sys.executable, *shlex.split(command)[1:]], cwd=workspace,
                                    capture_output=True, text=True, check=False, timeout=20)
            return result.returncode, result.stdout + result.stderr

        state = SimpleNamespace(sandbox=sandbox, sandbox_id=workspace.name, workspace=str(workspace), execute=execute)
        sandboxes.append(state)
        return state

    monkeypatch.setattr(e2b.E2BBlackjackEnv, '_create', create)
    return sandboxes


@pytest.mark.parametrize('entry', ['batch', 'streaming', 'e2b'])
@pytest.mark.parametrize('outcome', ['trained', 'zero-updates', 'interrupted'])
def test_blackjack_cookbooks_offline(entry, outcome, monkeypatch, request):
    sandboxes = []
    if entry == 'e2b':
        sandboxes = request.getfixturevalue('local_sandbox')
        env = e2b.E2BBlackjackEnv()
    else:
        pytest.importorskip('openspiel_env')
        env = streaming.BlackjackEnv()

    tokenizer = FakeTokenizer()
    tokenizer.encode(''.join(chr(i) for i in range(32, 127)) + '\n')
    template = OfflineTemplate(tokenizer)
    lock = Lock()
    opening_ids, requests = {}, []
    env.shutdown = Mock(wraps=env.shutdown)
    calls, policy_names, batches, logs = [], [], [], []

    class Session:
        live = set()

        def post(self, url, *, data=None, json=None, **kwargs):
            endpoint = url.rsplit('/', 1)[-1]
            body = json if data is None else decode_json(data)
            calls.append(endpoint)
            if endpoint == 'create':
                response = {'training_handles': True}
            else:
                result = None
                if endpoint == 'add_adapter_to_model':
                    self.live.add('training-1')
                    result = {'training_id': 'training-1', 'adapter_name': body['adapter_name']}
                elif endpoint == 'save':
                    policy_names.append(body['name'])
                    result = {'twinkle_path': f'twinkle://{body["name"]}'}
                elif endpoint == 'forward_backward':
                    batches.append(body['inputs'])
                    result = {'result': {}}
                elif endpoint == 'calculate_metric':
                    result = {'result': {}}
                elif endpoint == 'remove_adapter':
                    self.live.remove(body['training_id'])
                response = {'request_id': endpoint, 'status': 'completed', 'result': result}
            return SimpleNamespace(ok=True, status_code=200, json=lambda: response)

        def close(self):
            assert not self.live
            calls.append('close')

    decode_json = json.loads
    session = Session()
    client = TwinkleClient(transport=ClientTransport(
        ClientContext(base_url='http://offline-test', api_key='offline-test-key'), session=session))

    def sample(inputs, params, *, adapter_uri, **kwargs):
        # Token-only contract: the wire entry carries just the encoded prompt ids. The scripted
        # server recovers the tool observations by decoding the prompt (a real engine sees the
        # same text), and alternates hit/stand per member of the same opening deal: members of
        # one group share the identical opening prompt, so the n-th first-turn call for an
        # opening is that group's n-th member (the old per-episode index parity).
        (entry,) = inputs
        assert kwargs.get('token_only') is True
        prompt_ids = list(entry['input_ids'])
        text = tokenizer.decode(prompt_ids)
        observations = re.findall(r'<\|im_start\|>tool\n(.*?)<\|im_end\|>', text, flags=re.S)
        index = 0
        with lock:
            if not observations:
                key = tuple(prompt_ids)
                index = opening_ids.get(key, 0)
                opening_ids[key] = index + 1
            requests.append(adapter_uri)
        if outcome == 'zero-updates' or observations and observations[-1].endswith('Game over: True.'):
            decoded = 'finished'
        else:
            decoded = 'hit' if not observations and index % 2 else 'stand'
        ids = tokenizer.encode(decoded)
        return [SampleResponse(prompt_token_ids=prompt_ids, sequences=[SampledSequence(
            tokens=ids, logprobs=[[(token, -0.25)] for token in ids], stop_reason='stop',
            decoded=decoded,
        )])]

    sampler = SimpleNamespace(sample=sample, set_template=lambda *args, **kwargs: None)
    monkeypatch.setattr(client, 'sampler', lambda _: sampler)
    monkeypatch.setattr(streaming, 'Qwen3_5Template', lambda **_: template)
    monkeypatch.setattr(streaming.logger, 'info', logs.append)
    monkeypatch.setattr(streaming, 'init_twinkle_client', lambda: client)
    monkeypatch.setattr(batch, 'init_twinkle_client', lambda: client)
    monkeypatch.setattr(batch, 'BlackjackEnv', lambda: env)
    for recipe in (streaming, batch):
        monkeypatch.setattr(recipe, 'BASE_SEED', 20261110)
        monkeypatch.setattr(recipe, 'MAX_GROUPS', 2)
        monkeypatch.setattr(recipe, 'MAX_STEPS', 2)

    def run():
        if entry == 'batch':
            batch.train()
        else:
            (e2b.train if entry == 'e2b' else streaming.train)(lambda: env)

    if outcome == 'interrupted':
        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(streaming, 'train_batch', interrupt)
        monkeypatch.setattr(batch, 'train_batch', interrupt)
        with pytest.raises(KeyboardInterrupt):
            run()
    else:
        run()

    updates = int(outcome == 'trained')
    assert calls.count('forward_backward') == calls.count('clip_grad_and_step') == updates
    assert len(policy_names) == updates + 1
    assert set(requests) == {f'twinkle://{name}' for name in policy_names}
    assert any(f'optimizer_updates={updates} latest_lora=twinkle://{policy_names[-1]}' in row for row in logs)
    if not updates:
        assert any('No training updates' in row for row in logs)
    for rows in batches:
        assert {row['rewards']['env'] for row in rows} == {-1.0, 1.0}
        assert {row['turns'] for row in rows} == {2, 3}
        assert len({row['episode_id'] for row in rows}) == len(rows)
        for row in rows:
            validate_trajectory(row)
            assert len(row['rollout_logps']) == len(row['advantages']) == len(row['response_token_ids'])
            assert all(value * row['rewards']['env'] > 0 for value in row['advantages'])
    assert not env._sessions
    env.shutdown.assert_called_once()
    assert calls[-2:] == ['remove_adapter', 'close']
    for sandbox in sandboxes:
        sandbox.sandbox.kill.assert_called_once()
