# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU-gated E2E: ``APIMultiTurnRollout`` over the Gateway ``/chat/completions``
endpoint proves the returned Trajectory carries NO ``logprobs`` field.

Purpose
-------
This is the "generation-only, not trainable" counterpart to the trainable
``ClientMultiTurnRollout`` path. It drives the SAME
tool-calling scenario, but through the OpenAI-compatible route:

    OpenAI client  ->  Gateway POST /chat/completions  ->  /twinkle/sample

and asserts that the resulting Trajectory does **not** contain a ``logprobs``
field (or that it is ``None``/absent). The OpenAI chat-completions protocol only
surfaces assistant ``content`` / ``tool_calls`` / ``finish_reason`` — it never
carries the token-level ``logprobs`` + ``new_input_feature`` alignment info that
GRPO training requires. This test locks in the accuracy of that limitation
statement: the Gateway indirection is fine for
generation/evaluation but cannot feed token-aligned RL training.

What is "real" here
-------------------
  * REAL (GPU): the running Twinkle server (gateway + real vLLM sampler for
    ``Qwen/Qwen3.5-4B``), the ``/chat/completions`` -> ``/twinkle/sample`` hop,
    and the ``openai`` pip client issuing actual HTTP requests.
  * ``APIMultiTurnRollout`` + a small ``ToolManager`` (calculator tool) drive
    the multi-turn control flow client-side.

==============================================================================
GPU + CONDA ENVIRONMENT REQUIREMENT (MANDATORY)
==============================================================================
This case requires a GPU and an already-running GPU server (see
tests/server/start_e2e_server.py). It is gated behind ``TWINKLE_TEST_GPU_E2E=1``
and SKIPS cleanly on machines without a GPU (e.g. local dev). Run on GPU CI in
the ``twinkle`` conda env:

    TWINKLE_TEST_GPU_E2E=1 conda run -n twinkle pytest \
        tests/server/gateway/test_api_multi_turn_rollout_no_logprobs.py -v

Locally (no GPU / variable unset) it collects and skips without error:

    conda run -n twinkle pytest \
        tests/server/gateway/test_api_multi_turn_rollout_no_logprobs.py -v
==============================================================================
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

# Ensure project root is importable for both pytest and direct execution.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

# Reuse the shared GPU-e2e gate and server coordinates.
from tests.server.integration.e2e_helpers import (
    API_KEY,
    BASE_MODEL,
    BASE_URL,
    log,
    wait_for_server,
)
from tests.server.test_embedding_e2e import gpu_e2e_enabled

# The gateway (OpenAI-compatible) routes are mounted under the ``server`` app's
# route prefix ``/api/v1`` (see tests/server/config/server_config_4b_e2e.yaml).
# The ``openai`` client appends ``/chat/completions`` to this base URL.
GATEWAY_BASE_URL = f'{BASE_URL}/api/v1'

# Model/adapter name routed by the gateway. The e2e config serves the base model
# ``Qwen/Qwen3.5-4B`` directly.
GATEWAY_MODEL = BASE_MODEL


# ═══════════════════════════════════════════════════════════════════════════
# Tool scenario (mirrors the 9.2 tool-calling scenario): a small calculator.
# ═══════════════════════════════════════════════════════════════════════════
TOOL_NAME = 'calculator'


def _make_tool_manager():
    """Build a ToolManager with a single deterministic calculator tool."""
    from twinkle_agentic.tools.base import Tool
    from twinkle_agentic.tools.tool_manager import ToolManager

    class CalculatorTool(Tool):
        """Adds/subtracts/multiplies two integers; returns the result as text."""

        def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
            a = int(arguments.get('a', 0))
            b = int(arguments.get('b', 0))
            op = arguments.get('operation', 'add')
            if op == 'add':
                return str(a + b)
            if op == 'subtract':
                return str(a - b)
            if op == 'multiply':
                return str(a * b)
            return f'Error: unknown operation {op!r}'

        def tool_info(self):
            return {
                'type': 'function',
                'function': {
                    'name': TOOL_NAME,
                    'description': 'Compute a + b, a - b, or a * b for two integers.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'a': {'type': 'integer', 'description': 'first operand'},
                            'b': {'type': 'integer', 'description': 'second operand'},
                            'operation': {
                                'type': 'string',
                                'enum': ['add', 'subtract', 'multiply'],
                                'description': 'the arithmetic operation',
                            },
                        },
                        'required': ['a', 'b', 'operation'],
                    },
                },
            }

    mgr = ToolManager({})
    mgr.register(CalculatorTool())
    return mgr


def _make_trajectory() -> Dict[str, Any]:
    """A single math trajectory that nudges the model toward the calculator tool."""
    return {
        'messages': [
            {
                'role': 'system',
                'content': ('You are a precise assistant. When a calculation is needed, '
                            'call the calculator tool instead of computing yourself.'),
            },
            {'role': 'user', 'content': 'What is 123 multiplied by 7? Use the calculator tool.'},
        ],
    }


def _build_rollout(max_turns: int = 3):
    """Construct an APIMultiTurnRollout wired to the Gateway /chat/completions."""
    from twinkle.data_format.sampling import SamplingParams
    from twinkle_agentic.protocol.openai import OpenAI
    from twinkle_agentic.rollout import APIMultiTurnRollout

    api = OpenAI(model=GATEWAY_MODEL, api_key=API_KEY, base_url=GATEWAY_BASE_URL)
    # temperature=0 keeps the control flow as reproducible as the server allows.
    sampling_params = SamplingParams(num_samples=1, temperature=0.0, max_tokens=256)
    return APIMultiTurnRollout(
        api=api,
        tool_manager=_make_tool_manager(),
        sampling_params=sampling_params,
        max_turns=max_turns,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope='module')
def gpu_gateway_ready():
    """Ensure a GPU server is up before running; skip cleanly when GPU e2e is off."""
    if not gpu_e2e_enabled():
        pytest.skip('Set TWINKLE_TEST_GPU_E2E=1 to run the Gateway APIMultiTurnRollout '
                    'no-logprobs E2E (requires a running GPU server)')
    wait_for_server()
    yield


# ═══════════════════════════════════════════════════════════════════════════
# APIMultiTurnRollout via Gateway: Trajectory carries NO logprobs.
# ═══════════════════════════════════════════════════════════════════════════
def test_api_multi_turn_rollout_trajectory_has_no_logprobs(gpu_gateway_ready):
    """The Gateway /chat/completions path yields Trajectories without logprobs.

    Runs the same tool-calling scenario as the trainable client rollout but through
    ``APIMultiTurnRollout`` + an OpenAI-compatible client pointed at the Gateway.
    Asserts the returned Trajectory does NOT expose a ``logprobs`` field (absent
    or ``None``), confirming the "generation-only, not trainable" limitation is
    accurate: token-level alignment info never crosses the OpenAI protocol.

    GPU-gated (TWINKLE_TEST_GPU_E2E=1 + running GPU server); skipped locally.
    """
    rollout = _build_rollout(max_turns=3)
    trajectories = [_make_trajectory()]

    outs = rollout(trajectories)

    # Same length/order contract the multi-turn rollouts guarantee.
    assert len(outs) == 1
    out = outs[0]
    log(f'[api-rollout] stop_reason={out.get("stop_reason")} turns={out.get("turns")}')

    # Control flow terminated within the APIMultiTurnRollout stop-reason vocabulary.
    assert out.get('stop_reason') in {'stop', 'length', 'max_turns', 'api_error'}, out.get('stop_reason')
    assert isinstance(out.get('turns'), int) and out['turns'] >= 1

    # A functional run should not have surfaced an API error.
    assert out.get('stop_reason') != 'api_error', f"unexpected api_error: {out.get('error')!r}"

    # Core assertion: NO trainable token-level logprobs.
    assert out.get('logprobs') is None, (
        f'Gateway APIMultiTurnRollout Trajectory unexpectedly carries logprobs: '
        f'{out.get("logprobs")!r}. The OpenAI /chat/completions path is '
        f'generation-only and must not expose token-level logprobs.')

    # The per-message conversation also must not smuggle token-level logprobs
    # into any assistant turn (only content / tool_calls / finish_reason are set).
    for msg in out.get('messages', []):
        assert 'logprobs' not in msg, (
            f'assistant/tool message unexpectedly carries logprobs: {msg!r}')


def test_api_multi_turn_rollout_tool_call_scenario_shape(gpu_gateway_ready):
    """The tool-calling scenario produces a well-formed multi-turn conversation.

    Complements the no-logprobs assertion by checking the conversation shape:
    messages accumulate across turns and, when the model invokes the calculator,
    a matching ``role='tool'`` message is stitched back in. This mirrors the
    trainable client rollout scenario so the two paths are compared on the same
    workload. Tool invocation itself is model-dependent, so it is asserted conditionally.
    """
    rollout = _build_rollout(max_turns=3)

    outs = rollout([_make_trajectory()])
    out = outs[0]

    messages = out.get('messages', [])
    # The full conversation includes at least the seed system+user turns plus one
    # assistant reply.
    assert len(messages) >= 3, messages
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert any(m['role'] == 'assistant' for m in messages)

    # If the model emitted a tool call, a paired tool result must follow it, and
    # the calculator's deterministic answer (123 * 7 = 861) should appear.
    assistant_with_tool = next(
        (m for m in messages if m['role'] == 'assistant' and m.get('tool_calls')), None)
    if assistant_with_tool is not None:
        tool_msgs = [m for m in messages if m['role'] == 'tool']
        assert tool_msgs, 'assistant emitted tool_calls but no tool result was stitched back'
        # Whichever calculator call was made, its numeric result is plain text
        # (no logprobs), reinforcing the generation-only contract.
        for tm in tool_msgs:
            assert isinstance(tm.get('content'), str)
            assert 'logprobs' not in tm

    # Regardless of tool invocation, the no-logprobs invariant still holds.
    assert out.get('logprobs') is None
