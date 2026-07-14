# Copyright (c) ModelScope Contributors. All rights reserved.
"""Local CPU-only multi-turn control-flow E2E for ``ClientMultiTurnRollout``.

This test boots a REAL CPU-only Twinkle server (Ray Serve, in-process) whose
sampler backend is the enhanced :class:`MockSampler` (task 8.6), then drives
:class:`twinkle_client.rollout.multi_turn.ClientMultiTurnRollout` over ACTUAL
HTTP through :class:`twinkle_client.sampler.vLLMSampler`. No GPU is required.

What is "real" vs "test double" here
------------------------------------
  * REAL: the HTTP server (gateway + sampler apps on Ray Serve), the
    ``/twinkle/sample`` protocol hop, ``vLLMSampler.sample()`` and the
    enhanced ``MockSampler`` producing ``new_input_feature`` / configurable
    ``stop_reason`` / configurable tool-call text over the wire.
  * TEST DOUBLE (client-local only): a lightweight char-level Template used by
    the client to ``encode`` trajectories and ``parse_tool_call`` the sampled
    text, plus a small ``ToolManager`` with an echo tool. The Template never
    crosses the network; encoding, tool parsing and bridge stitching are
    client-side concerns. A real HF Template would require a downloaded
    tokenizer (network/model weights) which is unavailable in a local offline
    CPU environment, so a deterministic char-level double is used instead —
    exactly the established pattern from
    ``tests/twinkle_client/test_client_multi_turn_rollout.py`` (task 8.4).

Why the sampler knobs are set at CONSTRUCTION time
--------------------------------------------------
The multi-turn knobs (``stop_reason`` / ``tool_call_text`` / ``tool_call_turns``)
CANNOT be injected per-call through the HTTP layer: the ``/twinkle/sample``
handler rebuilds ``sampling_params`` via ``SamplingParams.from_dict`` which
filters the payload down to the known dataclass fields, dropping any extra
knobs before they reach ``MockSampler.sample``. Setting them per call would
require server-side protocol changes beyond this task's scope. Therefore each
behaviour is realised as a SEPARATE sampler app whose ``MockSampler`` is
constructed with the desired knobs:

  * ``mock-tool``   — ``stop_reason='stop'`` + a tool call injected on EVERY
    round (``tool_call_turns`` covers a wide range). This deterministically
    exercises the "sample -> tool -> bridge -> sample -> ... -> max_turns
    truncation" control flow regardless of the sampler's monotonic per-call
    round counter (so the module-scoped server can be reused across cases).
  * ``mock-stop``   — ``stop_reason='stop'`` with NO tool call, i.e. natural
    single-turn termination with ``stop_reason == 'stop'``.
  * ``mock-length`` — ``stop_reason='length'``, i.e. immediate termination with
    ``stop_reason == 'length'``.

==============================================================================
CONDA ENVIRONMENT REQUIREMENT (MANDATORY) — NO GPU REQUIRED
==============================================================================
ALL cases in this file are CPU-only and MUST run inside the ``twinkle`` conda env:

    conda run -n twinkle pytest \
        tests/server/integration/test_client_multi_turn_rollout_mock_e2e.py -v
==============================================================================
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is importable for both pytest and direct execution.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.rollout.multi_turn import ClientMultiTurnRollout

# ═══════════════════════════════════════════════════════════════════════════
# Sampler model ids (one app per multi-turn behaviour — see module docstring).
# ═══════════════════════════════════════════════════════════════════════════
MODEL_TOOL = 'mock-tool'      # stop_reason='stop' + tool call on every round
MODEL_STOP = 'mock-stop'      # stop_reason='stop', no tool call
MODEL_LENGTH = 'mock-length'  # stop_reason='length'

# Tool call text the mock emits as SampledSequence.decoded on injected rounds.
# Must match the client Template's parse_tool_call (Qwen <tool_call>{...}</tool_call>).
TOOL_NAME = 'echo'
TOOL_CALL_TEXT = '<tool_call>' + json.dumps({'name': TOOL_NAME, 'arguments': {'text': 'hi'}}) + '</tool_call>'

# Wide range so "inject a tool call on every round" holds even as the mock's
# monotonic per-call round counter advances across reused test cases.
_ALWAYS_INJECT_TURNS = list(range(1, 201))

# Sampling params carried on every round. ``max_tokens`` MUST be set: the mock
# rejects max_tokens < 1, and the rollout's default SamplingParams leaves it None.
SAMPLE_MAX_TOKENS = 4


# ═══════════════════════════════════════════════════════════════════════════
# Client-local test doubles: char-level tokenizer + Template + echo tool.
# (Mirrors tests/twinkle_client/test_client_multi_turn_rollout.py — task 8.4.)
# ═══════════════════════════════════════════════════════════════════════════
class _FakeTokenizer:
    """Char-level tokenizer with atomic special tokens.

    Guarantees ``decode(encode(s)) == s`` so ``extend_with_bridge``'s
    template-space delta computation is deterministic.
    """
    SPECIALS = ('<|im_start|>', '<|im_end|>')

    def __init__(self) -> None:
        self._s2i: Dict[str, int] = {}
        self._i2s: Dict[int, str] = {}
        for s in self.SPECIALS:
            self._add(s)

    def _add(self, tok: str) -> int:
        if tok not in self._s2i:
            i = len(self._s2i)
            self._s2i[tok] = i
            self._i2s[i] = tok
        return self._s2i[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids: List[int] = []
        i = 0
        while i < len(text):
            matched = False
            for sp in self.SPECIALS:
                if text.startswith(sp, i):
                    ids.append(self._add(sp))
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                ids.append(self._add(text[i]))
                i += 1
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        specials = set(self.SPECIALS)
        toks = [self._i2s[int(i)] for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in specials]
        return ''.join(toks)

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_,
    ):
        s = ''
        for m in messages:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += '<|im_start|>assistant\n'
        return self.encode(s) if tokenize else s


class _FakeTemplate:
    """Minimal Template mirroring the parts ClientMultiTurnRollout touches.

    A client-local test double (never crosses the network). Implements exactly
    ``encode`` / ``parse_tool_call`` / ``_invoke_post_pipeline`` /
    ``enable_thinking`` / ``tokenizer`` — everything ``extend_with_bridge`` and
    the rollout need.
    """
    model_id = 'qwen-fake'
    truncation_strategy = 'right'
    enable_thinking = False

    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, trajectory: Dict[str, Any], add_generation_prompt: bool = False) -> Dict[str, Any]:
        messages = trajectory.get('messages', [])
        s = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        input_ids = self.tokenizer.encode(s, add_special_tokens=False)
        pif: Dict[str, Any] = dict(trajectory)  # preserve top-level fields (incl. _tid)
        pif['input_ids'] = input_ids
        pif['labels'] = [-100] * len(input_ids)  # inference mode
        return self._invoke_post_pipeline([pif])[0]

    def _invoke_post_pipeline(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for pif in inputs:
            pif = dict(pif)
            input_ids = list(pif['input_ids'])
            labels = list(pif.get('labels') or [])
            if labels:
                if len(labels) != len(input_ids):
                    raise RuntimeError(f'FakeTemplate post_pipeline: labels({len(labels)}) '
                                       f'!= input_ids({len(input_ids)})')
                labels = labels[1:] + labels[:1]  # np.roll(labels, -1): shift LEFT by 1
            pif['input_ids'] = input_ids
            pif['labels'] = labels
            pif['attention_mask'] = [1] * len(input_ids)
            pif['position_ids'] = list(range(len(input_ids)))
            pif['length'] = len(input_ids)
            out.append(pif)
        return out

    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        matches = re.findall(r'<tool_call>\s*([\s\S]*?)\s*</tool_call>', decoded or '')
        results: List[Dict[str, Any]] = []
        for m in matches:
            try:
                d = json.loads(m)
            except json.JSONDecodeError:
                continue
            name = d.get('name') or d.get('tool_name')
            if not name:
                continue
            results.append({
                'type': 'function',
                'function': {'name': name, 'arguments': d.get('arguments', {})},
            })
        return results


class EchoTool(Tool):
    """Echoes its arguments as a JSON string."""

    def __init__(self, name: str = TOOL_NAME):
        self._name = name

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        return f'echo[{tool_name}]:{json.dumps(arguments, sort_keys=True)}'

    def tool_info(self):
        return {
            'type': 'function',
            'function': {'name': self._name, 'description': 'echo test tool', 'parameters': {}},
        }


def _make_tool_manager() -> ToolManager:
    mgr = ToolManager({})
    mgr.register(EchoTool(TOOL_NAME))
    return mgr


def _make_template() -> _FakeTemplate:
    return _FakeTemplate(_FakeTokenizer())


def _make_trajectories(n: int) -> List[Dict[str, Any]]:
    """Build ``n`` single-user-message trajectories tagged with a hidden _tid."""
    return [{'messages': [{'role': 'user', 'content': f'q{tid}'}], '_tid': tid} for tid in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# CPU-only mock server config.
#
# Built as plain dicts and fed DIRECTLY to the app builders (build_gateway_app /
# build_sampler_app) instead of through ``ServerConfig.model_validate``. The
# typed ``SamplerArgs`` schema uses ``extra='forbid'`` and does NOT declare the
# mock multi-turn knobs (stop_reason / tool_call_text / tool_call_turns), so
# routing them through the validated config would be rejected. The builders
# themselves accept ``**kwargs`` and forward them to the MockSampler ctor, so we
# skip validation and call them directly (this is test-only wiring).
# ═══════════════════════════════════════════════════════════════════════════
def _sampler_app(name: str, model_id: str, *, extra_args: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: ``queue_config`` is intentionally omitted. When calling the builder
    # directly (bypassing ServerConfig validation) a raw dict would NOT be
    # coerced into a TaskQueueConfig; leaving it unset lets the sampler app
    # construct a default TaskQueueConfig, which is fine for this CPU test.
    args: Dict[str, Any] = {
        'sampler_type': 'mock',
        'model_id': model_id,
        'nproc_per_node': 1,
        'device_group': {'name': f'sampler_{model_id}', 'ranks': 1, 'device_type': 'CPU'},
        'device_mesh': {'device_type': 'CPU', 'dp_size': 1},
    }
    args.update(extra_args)
    return {
        'name': name,
        'route_prefix': f'/api/v1/sampler/{model_id}',
        'import_path': 'sampler',
        'args': args,
    }


def _build_applications() -> List[Dict[str, Any]]:
    """Return the plain-dict application specs (gateway + 3 sampler apps)."""
    return [
        {
            'name': 'server',
            'route_prefix': '/api/v1',
            'import_path': 'server',
            'args': {
                'server_config': {'per_token_model_limit': 3},
                'supported_models': [MODEL_TOOL, MODEL_STOP, MODEL_LENGTH],
            },
        },
        _sampler_app('sampler-mock-tool', MODEL_TOOL, extra_args={
            'stop_reason': 'stop',
            'tool_call_text': TOOL_CALL_TEXT,
            'tool_call_turns': _ALWAYS_INJECT_TURNS,
        }),
        _sampler_app('sampler-mock-stop', MODEL_STOP, extra_args={'stop_reason': 'stop'}),
        _sampler_app('sampler-mock-length', MODEL_LENGTH, extra_args={'stop_reason': 'length'}),
    ]


# File-backed persistence path (gateway + samplers run as separate replicas, so
# ``memory`` mode is insufficient — cross-process visibility requires a file).
_PERSISTENCE_FILE = '/tmp/twinkle_state_multi_turn_mock.json'


# ═══════════════════════════════════════════════════════════════════════════
# In-process CPU-only Ray Serve harness (mirrors MockEmbeddingServerHarness).
# ═══════════════════════════════════════════════════════════════════════════
class MultiTurnMockServerHarness:
    """Boots the CPU-only mock multi-turn server in-process via Ray Serve.

    Manages its own local Ray head node (independent of the session-scoped Ray
    fixture), starts Ray Serve on a randomized port, and runs the gateway plus
    all three sampler apps declared by ``_build_server_config_dict``. No GPU.
    """

    READY_BUDGET_SECONDS = 90.0
    RAY_NODE_CPUS = 8

    def __init__(self) -> None:
        self.host = '127.0.0.1'
        self.port = 18900 + (os.getpid() % 700)
        self.base_url = f'http://{self.host}:{self.port}'
        self._started = False

    @staticmethod
    def _run_ray_command(*args: str) -> None:
        ray_bin = os.path.join(os.path.dirname(sys.executable), 'ray')
        result = subprocess.run(
            [ray_bin, *args], check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'ray {" ".join(args)} failed ({result.returncode}):\n{result.stdout}')

    def start(self) -> str:
        import ray
        from ray import serve

        from twinkle.server.config.persistence import PersistenceConfig
        from twinkle.server.gateway import build_gateway_app
        from twinkle.server.sampler import build_sampler_app

        # File-backed persistence so gateway + sampler replicas share state.
        persistence_env = PersistenceConfig(mode='file', file_path=_PERSISTENCE_FILE).to_env_vars()
        for k, v in persistence_env.items():
            os.environ[k] = v

        if ray.is_initialized():
            ray.shutdown()
        self._run_ray_command('stop', '--force')
        self._run_ray_command(
            'start', '--head', '--port=0', f'--num-cpus={self.RAY_NODE_CPUS}',
            '--num-gpus=0', '--include-dashboard=false', '--disable-usage-stats')
        ray.init(
            address='auto',
            runtime_env={'env_vars': persistence_env} if persistence_env else None,
        )
        self._started = True

        serve.start(http_options={'host': self.host, 'port': self.port})
        # Call the builders DIRECTLY with plain-dict args (see _build_applications):
        # this bypasses the typed SamplerArgs schema (extra='forbid') so the mock
        # multi-turn knobs reach the MockSampler ctor.
        builders = {'server': build_gateway_app, 'sampler': build_sampler_app}
        deploy_options: Dict[str, Any] = {'ray_actor_options': {'num_cpus': 0.1}}
        for app_spec in _build_applications():
            builder = builders[app_spec['import_path']]
            args = dict(app_spec['args'])
            if app_spec['import_path'] == 'server':
                args.setdefault('http_options', {'host': self.host, 'port': self.port})
            bound = builder(deploy_options=deploy_options, **args)
            serve.run(bound, name=app_spec['name'], route_prefix=app_spec['route_prefix'])

        self._wait_until_healthy(serve, self.READY_BUDGET_SECONDS)
        return self.base_url

    def _wait_until_healthy(self, serve_module: Any, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = serve_module.status()
            last = {name: app.status for name, app in status.applications.items()}
            if last and all(s == 'RUNNING' for s in last.values()):
                return
            time.sleep(0.5)
        raise TimeoutError(f'Mock multi-turn server not RUNNING within {timeout}s: {last}')

    def stop(self) -> None:
        if not self._started:
            return
        try:
            import ray
            from ray import serve
            try:
                serve.shutdown()
            except Exception:
                pass
            try:
                ray.shutdown()
            except Exception:
                pass
        finally:
            try:
                self._run_ray_command('stop', '--force')
            except Exception:
                pass
            self._started = False


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope='module')
def mock_multi_turn_server():
    """Boot the CPU-only mock multi-turn server ONCE for the module."""
    harness = MultiTurnMockServerHarness()
    base_url = harness.start()
    try:
        from twinkle_client import init_twinkle_client
        init_twinkle_client(base_url=base_url, api_key='EMPTY_TOKEN')
        yield base_url
    finally:
        harness.stop()


def _make_sampler(model_id: str):
    """Create a real vLLMSampler bound to the given mock sampler app."""
    from twinkle_client.sampler import vLLMSampler
    return vLLMSampler(model_id=model_id)


def _make_rollout(model_id: str, *, tool_manager: Optional[ToolManager], max_turns: int) -> ClientMultiTurnRollout:
    return ClientMultiTurnRollout(
        sampler=_make_sampler(model_id),
        template=_make_template(),
        tool_manager=tool_manager,
        sampling_params=SamplingParams(max_tokens=SAMPLE_MAX_TOKENS, num_samples=1),
        max_turns=max_turns,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-turn control flow over real HTTP (tool app: tool call on every round).
#
# With ``stop_reason='stop'`` and a tool call injected every round, the loop
# runs "sample -> tool -> bridge -> sample -> ..." until it hits ``max_turns``
# and force-truncates. Exercises Property 3 (len/order), Property 4
# ('max_turns' in the allowed set), and Property 6 (turns <= max_turns).
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize('n_traj,max_turns', [(1, 3), (3, 2), (2, 4)])
def test_multi_turn_tool_loop_over_http(mock_multi_turn_server, n_traj, max_turns):
    """Full multi-turn control flow over real HTTP terminates at max_turns.

    Boots a real CPU server, drives ClientMultiTurnRollout via vLLMSampler HTTP
    calls through several "sample -> tool -> bridge -> sample" rounds, and
    checks the batch invariants.

    Validates: Requirements 3.2 (Property 3), 3.3 (Property 4), 3.5 (Property 6), 7.1, 7.2
    """
    rollout = _make_rollout(MODEL_TOOL, tool_manager=_make_tool_manager(), max_turns=max_turns)
    trajectories = _make_trajectories(n_traj)

    outs = rollout(copy.deepcopy(trajectories))

    # Property 3: output list is same length and order as the input.
    assert len(outs) == n_traj
    for i, out in enumerate(outs):
        assert out['_tid'] == i, 'output order must match input order'

    for out in outs:
        # Property 4: stop_reason is within the allowed set.
        assert out['stop_reason'] in {'length', 'stop', 'max_turns'}, out['stop_reason']
        # Property 6: actual turns never exceed the configured max_turns.
        assert out['turns'] <= max_turns, f"turns({out['turns']}) > max_turns({max_turns})"
        # A tool call on every round means the loop must hit the turn cap.
        assert out['stop_reason'] == 'max_turns'
        assert out['truncated'] is True
        assert out['turns'] == max_turns
        # Multi-turn actually stitched tool turns via the shared bridge helper:
        # the running context grew past the initial prompt encoding.
        assert out['messages'][0]['role'] == 'user'
        if max_turns >= 2:
            # At least one tool message was appended by extend_with_bridge.
            assert any(m['role'] == 'tool' for m in out['messages'])


# ═══════════════════════════════════════════════════════════════════════════
# Property 7: max_turns == 1 with a first-round tool call forces truncation.
# ═══════════════════════════════════════════════════════════════════════════
def test_property7_max_turns_one_forces_truncation(mock_multi_turn_server):
    """max_turns==1 + first-round tool call -> truncated=True, stop_reason='max_turns'.

    Validates: Requirements 3.6 (Property 7), 7.1, 7.2
    """
    rollout = _make_rollout(MODEL_TOOL, tool_manager=_make_tool_manager(), max_turns=1)
    trajectories = _make_trajectories(3)

    outs = rollout(copy.deepcopy(trajectories))

    assert len(outs) == 3
    for i, out in enumerate(outs):
        assert out['_tid'] == i
        assert out['truncated'] is True
        assert out['stop_reason'] == 'max_turns'
        assert out['turns'] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Property 4: natural termination reasons ('stop' and 'length') over real HTTP.
# ═══════════════════════════════════════════════════════════════════════════
def test_natural_stop_termination_over_http(mock_multi_turn_server):
    """No tool call + stop_reason='stop' -> single-turn natural termination.

    Validates: Requirements 3.2 (Property 3), 3.3 (Property 4), 3.5 (Property 6), 7.1, 7.2
    """
    rollout = _make_rollout(MODEL_STOP, tool_manager=_make_tool_manager(), max_turns=4)
    trajectories = _make_trajectories(2)

    outs = rollout(copy.deepcopy(trajectories))

    assert len(outs) == 2
    for i, out in enumerate(outs):
        assert out['_tid'] == i
        assert out['stop_reason'] == 'stop'
        assert out['truncated'] is False
        assert out['turns'] == 1
        assert out['turns'] <= 4


def test_length_termination_over_http(mock_multi_turn_server):
    """stop_reason='length' -> immediate termination on the first round.

    Validates: Requirements 3.3 (Property 4), 3.5 (Property 6), 7.1, 7.2
    """
    rollout = _make_rollout(MODEL_LENGTH, tool_manager=_make_tool_manager(), max_turns=4)
    trajectories = _make_trajectories(2)

    outs = rollout(copy.deepcopy(trajectories))

    assert len(outs) == 2
    for out in outs:
        assert out['stop_reason'] == 'length'
        assert out['truncated'] is False
        assert out['turns'] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Exception path (task 8.3): tool_calls produced but no tool_manager -> ValueError.
#
# Note on ``new_input_feature=None``: the enhanced MockSampler ALWAYS populates
# new_input_feature, so that specific 8.3 error path is not reproducible against
# a real mock server and is covered by the unit tests in
# tests/twinkle_client/test_client_multi_turn_rollout.py instead. The
# tool_manager-missing path IS reachable over real HTTP and is asserted here.
# ═══════════════════════════════════════════════════════════════════════════
def test_tool_calls_without_tool_manager_raises_value_error_over_http(mock_multi_turn_server):
    """A tool call with no tool_manager raises ValueError over the real HTTP path.

    Validates: Requirements 3.10, 7.1, 7.2
    """
    rollout = _make_rollout(MODEL_TOOL, tool_manager=None, max_turns=3)
    trajectories = _make_trajectories(1)

    with pytest.raises(ValueError) as excinfo:
        rollout(copy.deepcopy(trajectories))

    msg = str(excinfo.value)
    assert 'tool_manager' in msg, msg
    assert 'trajectory 0' in msg, msg
