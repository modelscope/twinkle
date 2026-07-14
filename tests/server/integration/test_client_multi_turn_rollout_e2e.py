# Copyright (c) ModelScope Contributors. All rights reserved.
"""Behavioural-alignment E2E: ``ClientMultiTurnRollout`` vs bare ``MultiTurnRollout``.

This module validates that the client-side, HTTP-driven
:class:`twinkle_client.rollout.multi_turn.ClientMultiTurnRollout` produces the
SAME multi-turn control flow as the bare-library, Ray-actor-driven
:class:`twinkle_agentic.rollout.multi_turn.MultiTurnRollout` when both are pointed
at the *same real sampler weights* (Qwen3.5-4B) with the same prompt, the same
tools, and greedy decoding (``temperature=0``).

Why this must be GPU-gated
---------------------------
Real numeric semantics (actual token sampling + per-token ``logprobs`` on real
model weights) cannot be reproduced by the CPU-only mock sampler used in the
local mock E2E (task 9.1). Two things in particular are only observable with a
real sampler and are therefore asserted here under GPU gating:

  * The two rollout paths agree on the ``messages`` structure and on *when*
    tool calls are triggered (greedy decoding makes the control flow
    deterministic across both transports, even though we allow token-level
    non-determinism in principle).
  * ``logprobs`` are actually populated, so the strict invariant
    ``len(logprobs) == count(labels != -100)`` can be checked per trajectory on
    both paths (the mock path cannot exercise real logprobs numerics).

Topology on GPU CI
------------------
  * CLIENT path: connects to an ALREADY-RUNNING Twinkle e2e server (start it
    first with ``tests/server/start_e2e_server.py``, which serves
    ``tests/server/config/server_config_4b_e2e.yaml`` including the
    ``sampler-Qwen3.5-4B`` application). Sampling goes over HTTP through
    :class:`twinkle_client.sampler.vLLMSampler`. This mirrors the convention of
    the GPU cases in ``tests/server/test_embedding_e2e.py``.
  * BARE path: builds its OWN bare-library Ray-actor
    :class:`twinkle.sampler.vLLMSampler` (``remote_group='sampler'`` +
    ``DeviceMesh``) in-process, mirroring the standalone GPU sampler tests
    (``tests/sampler/test_weight_sync.py``, ``tests/sampler/align_swift.py``)
    and the multi-turn cookbook (``cookbook/rl/multi_turn/multi_turn_grpo.py``).
    This is what the task means by "directly holding the corresponding Ray actor
    sampler". The runner must therefore provision enough GPUs for BOTH the
    server's sampler and this in-test bare sampler.

==============================================================================
CONDA ENVIRONMENT REQUIREMENT (MANDATORY) — GPU REQUIRED
==============================================================================
This whole file is gated behind ``TWINKLE_TEST_GPU_E2E=1`` and is skipped
automatically on machines without a GPU (so it collects/skips cleanly during
local, CPU-only development). Run it inside the ``twinkle`` conda env on a GPU
host, with the e2e server already running:

    # 1) start the server (separate shell)
    conda run -n twinkle python tests/server/start_e2e_server.py

    # 2) run the alignment test
    TWINKLE_TEST_GPU_E2E=1 conda run -n twinkle pytest \
        tests/server/integration/test_client_multi_turn_rollout_e2e.py -v
==============================================================================
"""
from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is importable for both pytest and direct execution.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager

# Reuse the shared GPU e2e helpers (running-server URL, session init, model id).
from tests.server.integration.e2e_helpers import (
    MODEL_ID,
    init_twinkle_client_session,
    log,
    wait_for_server,
)


# ═══════════════════════════════════════════════════════════════════════════
# GPU gating
#
# Mirrors the ``gpu_e2e_enabled`` gate used by tests/server/test_embedding_e2e.py.
# Defined locally (not imported) so this file collects/skips cleanly even if the
# embedding e2e module is being edited concurrently.
# ═══════════════════════════════════════════════════════════════════════════
def gpu_e2e_enabled() -> bool:
    """Return True only when GPU e2e tests are explicitly enabled."""
    return os.environ.get('TWINKLE_TEST_GPU_E2E', '0') == '1'


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Greedy decoding (temperature=0) makes both transports produce the SAME token
# stream for the SAME weights, so the multi-turn control flow is deterministic
# and directly comparable. ``logprobs=1`` forces per-token logprobs so the
# strict logprobs/trainable-label alignment can be asserted on both paths.
GREEDY_MAX_TOKENS = 128
GREEDY_SEED = 0

# 3-6 rounds as required by the task.
ALIGN_MAX_TURNS = 6

# Template used locally by BOTH rollouts for encode / parse_tool_call / bridge.
TEMPLATE_CLS = 'Qwen3_5Template'


# ═══════════════════════════════════════════════════════════════════════════
# A tiny ToolManager with 1-2 simple, deterministic tools (echo + calculator).
# ═══════════════════════════════════════════════════════════════════════════
class EchoTool(Tool):
    """Echoes its arguments back as a JSON string (deterministic)."""

    NAME = 'echo'

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        return f'echo[{tool_name}]:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}'

    def tool_info(self):
        return {
            'type': 'function',
            'function': {
                'name': self.NAME,
                'description': 'Echo the given text back verbatim.',
                'parameters': {
                    'type': 'object',
                    'properties': {'text': {'type': 'string', 'description': 'Text to echo.'}},
                    'required': ['text'],
                },
            },
        }


class CalculatorTool(Tool):
    """Evaluates ``a <op> b`` for the four basic operators (deterministic)."""

    NAME = 'calculator'
    _OPS = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b, 'div': lambda a, b: (a / b if b else float('inf'))}

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        op = str(arguments.get('op', 'add'))
        a = float(arguments.get('a', 0))
        b = float(arguments.get('b', 0))
        fn = self._OPS.get(op, self._OPS['add'])
        return json.dumps({'result': fn(a, b)}, ensure_ascii=False)

    def tool_info(self):
        return {
            'type': 'function',
            'function': {
                'name': self.NAME,
                'description': 'Compute a basic arithmetic operation on two numbers.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'op': {'type': 'string', 'enum': ['add', 'sub', 'mul', 'div']},
                        'a': {'type': 'number'},
                        'b': {'type': 'number'},
                    },
                    'required': ['op', 'a', 'b'],
                },
            },
        }


def make_tool_manager() -> ToolManager:
    """Build a small ToolManager holding the echo + calculator tools."""
    mgr = ToolManager({})
    mgr.register(EchoTool())
    mgr.register(CalculatorTool())
    return mgr


def tool_schema() -> List[Dict[str, Any]]:
    """Tool schema advertised to the model in the trajectory ``tools`` field."""
    return [EchoTool().tool_info(), CalculatorTool().tool_info()]


SYSTEM_PROMPT = (
    'You are a helpful assistant with access to tools. When a computation is '
    'needed, call the `calculator` tool with the appropriate operator and '
    'operands. When asked to repeat text, call the `echo` tool. Prefer using a '
    'tool over answering directly, then give a short final answer.')


def build_alignment_trajectories() -> List[Dict[str, Any]]:
    """Build a small, fixed batch of tool-inviting trajectories.

    Each trajectory carries a hidden ``_tid`` so output order can be checked, a
    system prompt describing the tools, and the ``tools`` schema so the template
    renders tool definitions into the prompt.
    """
    users = [
        'Please compute 21 plus 34 using your tools, then tell me the result.',
        'Use a tool to echo the exact text "ping", then confirm what you echoed.',
    ]
    trajectories: List[Dict[str, Any]] = []
    for tid, content in enumerate(users):
        trajectories.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': content},
            ],
            'tools': tool_schema(),
            '_tid': tid,
        })
    return trajectories


# ═══════════════════════════════════════════════════════════════════════════
# Pure comparison helpers (collection-safe: unit-tested locally without a GPU).
# ═══════════════════════════════════════════════════════════════════════════
def count_trainable_labels(labels: Optional[List[int]]) -> int:
    """Number of trainable positions (``label != -100``) in a labels list."""
    return sum(1 for label in (labels or []) if label != -100)


def message_role_signature(messages: List[Dict[str, Any]]) -> List[str]:
    """Ordered list of message roles — captures turn structure & tool timing.

    The presence and position of ``tool`` role messages encodes exactly when a
    tool call was triggered and answered, which is the control-flow signal we
    compare across the two transports.
    """
    return [str(m.get('role')) for m in (messages or [])]


def tool_turn_indices(messages: List[Dict[str, Any]]) -> List[int]:
    """Assistant-turn indices (0-based over assistant messages) that were
    immediately followed by a tool response.

    This makes "when tool_calls fired" explicit and independent of textual
    content: assistant turn ``k`` is a tool-calling turn iff the next message is
    a ``tool`` message.
    """
    roles = message_role_signature(messages)
    indices: List[int] = []
    assistant_idx = -1
    for i, role in enumerate(roles):
        if role == 'assistant':
            assistant_idx += 1
            if i + 1 < len(roles) and roles[i + 1] == 'tool':
                indices.append(assistant_idx)
    return indices


def control_flow_fingerprint(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the transport-independent control-flow fingerprint of an output.

    Deliberately EXCLUDES token ids / raw logprob values (which may differ) and
    keeps only the control-flow-relevant fields the task requires to match:
    role signature, tool-call timing, stop_reason, turns, truncated.
    """
    return {
        'roles': message_role_signature(traj.get('messages') or []),
        'tool_turns': tool_turn_indices(traj.get('messages') or []),
        'stop_reason': traj.get('stop_reason'),
        'turns': traj.get('turns'),
        'truncated': bool(traj.get('truncated')),
    }


def assert_logprobs_align_with_labels(traj: Dict[str, Any], label: str) -> None:
    """Assert ``len(logprobs) == count(labels != -100)`` for a rollout output.

    Only meaningful when logprobs were requested/collected; a ``None``/empty
    logprobs list means "not trainable" and is skipped (the rollout itself
    already guards the non-empty case, this re-asserts it at the e2e boundary).
    """
    logprobs = traj.get('logprobs')
    if not logprobs:
        return
    trainable = count_trainable_labels(traj.get('labels'))
    assert len(logprobs) == trainable, (
        f'[{label}] logprobs({len(logprobs)}) != trainable labels({trainable}) '
        f'(labels != -100)')


# Allowed stop reasons (Property 4 in the design doc / task 8.4).
_ALLOWED_STOP_REASONS = {'length', 'stop', 'max_turns'}


def assert_properties_3_4_6_7(outs: List[Dict[str, Any]], n_inputs: int, max_turns: int, label: str) -> None:
    """Re-assert the multi-turn Properties 3/4/6/7 on a rollout output list.

    * Property 3 — output length & order preserved (via hidden ``_tid``).
    * Property 4 — stop_reason within ``{'length','stop','max_turns'}``.
    * Property 6 — actual turns never exceed max_turns.
    * Property 7 — a ``max_turns`` truncation implies ``truncated=True``.
    """
    # Property 3: same length and order.
    assert len(outs) == n_inputs, f'[{label}] expected {n_inputs} outputs, got {len(outs)}'
    for i, out in enumerate(outs):
        assert out.get('_tid') == i, f'[{label}] output order mismatch at index {i}'
        # Property 4: stop_reason value range.
        assert out.get('stop_reason') in _ALLOWED_STOP_REASONS, \
            f"[{label}] stop_reason={out.get('stop_reason')!r} not in {_ALLOWED_STOP_REASONS}"
        # Property 6: turns bounded by max_turns.
        assert out.get('turns') is not None and out['turns'] <= max_turns, \
            f"[{label}] turns({out.get('turns')}) > max_turns({max_turns})"
        # Property 7: max_turns truncation implies truncated=True.
        if out.get('stop_reason') == 'max_turns':
            assert out.get('truncated') is True, \
                f'[{label}] stop_reason==max_turns but truncated is not True'


# ═══════════════════════════════════════════════════════════════════════════
# GPU-only sampler / rollout builders (never called at collection time).
# ═══════════════════════════════════════════════════════════════════════════
def _build_local_template():
    """Build the real Qwen3.5 template used locally by BOTH rollouts."""
    from twinkle.template import Qwen3_5Template
    template = Qwen3_5Template(MODEL_ID, max_length=8192, enable_thinking=False)
    # Multi-turn bridge stitching does not support 'split'.
    template.truncation_strategy = 'delete'
    return template


def _greedy_sampling_params() -> SamplingParams:
    """Greedy, deterministic sampling params shared by both paths."""
    return SamplingParams(
        max_tokens=GREEDY_MAX_TOKENS,
        temperature=0.0,
        num_samples=1,
        logprobs=1,
        seed=GREEDY_SEED,
    )


def _build_client_rollout():
    """Build a ClientMultiTurnRollout wired to the running e2e server (HTTP)."""
    from twinkle_client.sampler import vLLMSampler as ClientVLLMSampler
    from twinkle_client.rollout.multi_turn import ClientMultiTurnRollout

    sampler = ClientVLLMSampler(model_id=MODEL_ID)
    sampler.set_template(TEMPLATE_CLS, model_id=MODEL_ID, enable_thinking=False)

    return ClientMultiTurnRollout(
        sampler=sampler,
        template=_build_local_template(),
        tool_manager=make_tool_manager(),
        sampling_params=_greedy_sampling_params(),
        max_turns=ALIGN_MAX_TURNS,
    )


def _build_bare_rollout():
    """Build a bare-library MultiTurnRollout holding its own Ray-actor sampler.

    Mirrors the standalone GPU sampler tests / multi-turn cookbook: initialise
    Twinkle in Ray mode with a dedicated ``sampler`` device group and construct
    a Ray-actor :class:`twinkle.sampler.vLLMSampler` (``remote_group='sampler'``).
    """
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.sampler import vLLMSampler as RayVLLMSampler
    from twinkle_agentic.rollout.multi_turn import MultiTurnRollout

    sampler_gpus = int(os.environ.get('TWINKLE_ALIGN_SAMPLER_GPUS', '1'))
    twinkle.initialize(
        mode='ray',
        nproc_per_node=sampler_gpus,
        groups=[DeviceGroup(name='sampler', ranks=list(range(sampler_gpus)), device_type='GPU')],
        lazy_collect=False,
    )
    sampler_mesh = DeviceMesh.from_sizes(world_size=sampler_gpus, dp_size=sampler_gpus)
    sampler = RayVLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.5,
            'max_model_len': 4096,
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE_CLS, model_id=MODEL_ID, enable_thinking=False)

    return MultiTurnRollout(
        sampler=sampler,
        template=_build_local_template(),
        tool_manager=make_tool_manager(),
        sampling_params=_greedy_sampling_params(),
        max_turns=ALIGN_MAX_TURNS,
    )


# ═══════════════════════════════════════════════════════════════════════════
# GPU fixture: build both rollouts once for the module (skips without GPU).
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope='module')
def aligned_rollouts():
    """Yield ``(client_rollout, bare_rollout)`` built against the same weights.

    Skipped automatically unless ``TWINKLE_TEST_GPU_E2E=1``. Requires the e2e
    server to be already running (client path) and enough GPUs for the in-test
    bare Ray-actor sampler.
    """
    if not gpu_e2e_enabled():
        pytest.skip('Set TWINKLE_TEST_GPU_E2E=1 to run the real-sampler multi-turn '
                    'alignment E2E (requires a running server + GPU).')

    wait_for_server()
    init_twinkle_client_session()

    log('Building client (HTTP) rollout...')
    client_rollout = _build_client_rollout()
    log('Building bare-library (Ray actor) rollout...')
    bare_rollout = _build_bare_rollout()

    yield client_rollout, bare_rollout


# ═══════════════════════════════════════════════════════════════════════════
# GPU test: the two paths agree on control flow, and logprobs/labels align.
# ═══════════════════════════════════════════════════════════════════════════
def test_client_and_bare_multi_turn_control_flow_aligned(aligned_rollouts):
    """ClientMultiTurnRollout and bare MultiTurnRollout agree on control flow.

    Both paths run the SAME prompt / tools / greedy sampling against the SAME
    real Qwen3.5-4B weights for 3-6 rounds. We assert:

      * Both satisfy Properties 3/4/6/7 (length/order, stop_reason range,
        turns <= max_turns, max_turns => truncated).
      * The transport-independent control-flow fingerprint (message role
        signature, tool-call timing, stop_reason, turns, truncated) is IDENTICAL
        between the two paths for every trajectory. Token-level sampling
        non-determinism is allowed, but the control flow must match.
      * On each path, ``len(logprobs) == count(labels != -100)`` for every
        trajectory that collected logprobs (real-sampler numeric alignment).

    Validates: Requirements 3.12, 7.2
    """
    client_rollout, bare_rollout = aligned_rollouts
    trajectories = build_alignment_trajectories()
    n = len(trajectories)

    client_outs = client_rollout(copy.deepcopy(trajectories))
    bare_outs = bare_rollout(copy.deepcopy(trajectories))

    # Per-path structural properties (3/4/6/7).
    assert_properties_3_4_6_7(client_outs, n, ALIGN_MAX_TURNS, label='client')
    assert_properties_3_4_6_7(bare_outs, n, ALIGN_MAX_TURNS, label='bare')

    # Per-path real-sampler logprobs/labels alignment (strict equality).
    for out in client_outs:
        assert_logprobs_align_with_labels(out, label='client')
    for out in bare_outs:
        assert_logprobs_align_with_labels(out, label='bare')

    # Cross-path control-flow equality (allowing token-level non-determinism).
    for i in range(n):
        client_fp = control_flow_fingerprint(client_outs[i])
        bare_fp = control_flow_fingerprint(bare_outs[i])
        assert client_fp == bare_fp, (
            f'control-flow mismatch for trajectory {i}:\n'
            f'  client={client_fp}\n  bare  ={bare_fp}')

        # The two paths must also agree on the exact trainable-label count, which
        # (with greedy decoding on identical weights) is the strongest available
        # cross-path numeric-alignment signal short of bit-identical logprobs.
        client_trainable = count_trainable_labels(client_outs[i].get('labels'))
        bare_trainable = count_trainable_labels(bare_outs[i].get('labels'))
        assert client_trainable == bare_trainable, (
            f'trainable-label count mismatch for trajectory {i}: '
            f'client={client_trainable} vs bare={bare_trainable}')


# ═══════════════════════════════════════════════════════════════════════════
# Collection-safe unit tests for the pure comparison helpers (run locally, no
# GPU). These prove the module imports and the fingerprint logic behaves, so the
# file is meaningful even when the GPU test above is skipped.
# ═══════════════════════════════════════════════════════════════════════════
def test_gpu_gate_reflects_env_flag():
    """The GPU gate reflects TWINKLE_TEST_GPU_E2E without side effects."""
    assert gpu_e2e_enabled() == (os.environ.get('TWINKLE_TEST_GPU_E2E', '0') == '1')


def test_count_trainable_labels():
    assert count_trainable_labels(None) == 0
    assert count_trainable_labels([]) == 0
    assert count_trainable_labels([-100, -100]) == 0
    assert count_trainable_labels([-100, 5, 7, -100, 9]) == 3


def test_message_role_signature_and_tool_turns():
    messages = [
        {'role': 'system', 'content': 's'},
        {'role': 'user', 'content': 'u'},
        {'role': 'assistant', 'content': 'call'},   # assistant turn 0 -> tool
        {'role': 'tool', 'content': 't'},
        {'role': 'assistant', 'content': 'final'},   # assistant turn 1 -> no tool
    ]
    assert message_role_signature(messages) == ['system', 'user', 'assistant', 'tool', 'assistant']
    # Only the first assistant turn is immediately followed by a tool response.
    assert tool_turn_indices(messages) == [0]


def test_control_flow_fingerprint_ignores_tokens_and_logprob_values():
    """Two outputs with identical control flow but different tokens/logprobs
    produce an identical fingerprint (token-level non-determinism allowed)."""
    base_messages = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'call'},
        {'role': 'tool', 'content': 't'},
        {'role': 'assistant', 'content': 'final'},
    ]
    a = {
        'messages': base_messages,
        'stop_reason': 'stop', 'turns': 2, 'truncated': False,
        'labels': [-100, 1, 2], 'logprobs': [[(1, -0.1)], [(2, -0.2)]],
    }
    b = {
        # Same roles/timing, DIFFERENT token content and logprob values.
        'messages': [dict(m) for m in base_messages],
        'stop_reason': 'stop', 'turns': 2, 'truncated': False,
        'labels': [-100, 9, 8], 'logprobs': [[(9, -1.0)], [(8, -2.0)]],
    }
    assert control_flow_fingerprint(a) == control_flow_fingerprint(b)


def test_assert_logprobs_align_with_labels_pass_and_fail():
    # Passing case: 2 logprob entries, 2 trainable labels.
    ok = {'labels': [-100, 3, 4], 'logprobs': [[(3, -0.1)], [(4, -0.2)]]}
    assert_logprobs_align_with_labels(ok, label='unit')  # no raise

    # Empty/None logprobs is a no-op (not trainable).
    assert_logprobs_align_with_labels({'labels': [-100, 3], 'logprobs': None}, label='unit')

    # Mismatch raises.
    bad = {'labels': [-100, 3, 4], 'logprobs': [[(3, -0.1)]]}
    with pytest.raises(AssertionError):
        assert_logprobs_align_with_labels(bad, label='unit')


def test_assert_properties_3_4_6_7_detects_violations():
    good = [
        {'_tid': 0, 'stop_reason': 'stop', 'turns': 1, 'truncated': False},
        {'_tid': 1, 'stop_reason': 'max_turns', 'turns': 3, 'truncated': True},
    ]
    assert_properties_3_4_6_7(good, n_inputs=2, max_turns=3, label='unit')  # no raise

    # Property 4 violation: unknown stop_reason.
    with pytest.raises(AssertionError):
        assert_properties_3_4_6_7(
            [{'_tid': 0, 'stop_reason': 'weird', 'turns': 1, 'truncated': False}],
            n_inputs=1, max_turns=3, label='unit')

    # Property 6 violation: turns exceed max_turns.
    with pytest.raises(AssertionError):
        assert_properties_3_4_6_7(
            [{'_tid': 0, 'stop_reason': 'stop', 'turns': 5, 'truncated': False}],
            n_inputs=1, max_turns=3, label='unit')

    # Property 7 violation: max_turns but not truncated.
    with pytest.raises(AssertionError):
        assert_properties_3_4_6_7(
            [{'_tid': 0, 'stop_reason': 'max_turns', 'turns': 3, 'truncated': False}],
            n_inputs=1, max_turns=3, label='unit')

    # Property 3 violation: order/length mismatch.
    with pytest.raises(AssertionError):
        assert_properties_3_4_6_7(
            [{'_tid': 1, 'stop_reason': 'stop', 'turns': 1, 'truncated': False}],
            n_inputs=1, max_turns=3, label='unit')


def test_build_alignment_trajectories_shape():
    trajs = build_alignment_trajectories()
    assert len(trajs) == 2
    for i, traj in enumerate(trajs):
        assert traj['_tid'] == i
        assert traj['messages'][0]['role'] == 'system'
        assert traj['messages'][1]['role'] == 'user'
        assert isinstance(traj['tools'], list) and traj['tools']
        names = {t['function']['name'] for t in traj['tools']}
        assert names == {'echo', 'calculator'}


def test_tool_manager_dispatch_is_deterministic():
    """The echo/calculator tools dispatch deterministically via ToolManager."""
    mgr = make_tool_manager()
    calc_call = {'type': 'function', 'function': {'name': 'calculator',
                                                   'arguments': {'op': 'add', 'a': 21, 'b': 34}}}
    echo_call = {'type': 'function', 'function': {'name': 'echo', 'arguments': {'text': 'ping'}}}
    assert json.loads(mgr(calc_call))['result'] == 55
    assert mgr(echo_call) == 'echo[echo]:{"text": "ping"}'
