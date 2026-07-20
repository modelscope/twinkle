# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property-based tests for
:class:`twinkle_client.rollout.multi_turn.ClientMultiTurnRollout`.

These tests are 100% CPU-only and do NOT require a GPU or a running server.
They reuse the char-level Fake Tokenizer / Template infrastructure style from
``tests/twinkle_agentic/test_multi_turn_rollout.py`` but adapt the fake sampler
to the ``twinkle_client`` HTTP contract: ``FakeClientSampler.sample()`` mirrors
``vLLMSampler.sample()`` and returns ``List[SampleResponseModel]`` (pydantic,
from ``twinkle_client.types.sampler``) whose ``sequences[0]`` carries a populated
``new_input_feature`` so the multi-turn loop can proceed round after round.

Properties covered:
    * Output length & order preservation
    * stop_reason value range
    * logprobs / trainable-label alignment
    * Actual turns never exceed max_turns
    * Forced truncation at the max_turns edge
"""
from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.rollout.multi_turn import ClientMultiTurnRollout
from twinkle_client.types.sampler import SampledSequenceModel, SampleResponseModel


# =============================================================================
# Fakes (tokenizer / template mirror the twinkle_agentic test infra)
# =============================================================================
class FakeTokenizer:
    """Char-level tokenizer with atomic special tokens.

    Guarantees ``decode(encode(s)) == s`` for any mix of raw chars and
    registered specials, which is what makes ``extend_with_bridge``'s
    template-space delta computation deterministic in the test.
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
            role = m['role']
            content = m['content']
            s += f'<|im_start|>{role}\n{content}<|im_end|>\n'
        if add_generation_prompt:
            s += '<|im_start|>assistant\n'
        if tokenize:
            return self.encode(s)
        return s


class FakeTemplate:
    """Minimal Template mirroring the parts ClientMultiTurnRollout touches."""
    model_id = 'qwen-fake'
    truncation_strategy = 'right'
    enable_thinking = False

    def __init__(self, tokenizer: FakeTokenizer) -> None:
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
                # np.roll(labels, -1): shift LEFT by 1 (output/shifted order)
                labels = labels[1:] + labels[:1]
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
                'function': {
                    'name': name,
                    'arguments': d.get('arguments', {}),
                },
            })
        return results

    def concat_input_feature(self, pif: Dict[str, Any], new_tokens: List[int]) -> Dict[str, Any]:
        result = copy.deepcopy(pif)
        prompt_ids = list(result['input_ids'])
        labels = list(result.get('labels') or [])
        if labels:
            # Unroll (shift RIGHT by 1): reverse the post_pipeline roll
            labels = labels[-1:] + labels[:-1]
        else:
            labels = [-100] * len(prompt_ids)
        input_ids = prompt_ids + list(new_tokens)
        labels = labels + list(new_tokens)  # assistant tokens trainable
        result['input_ids'] = input_ids
        result['labels'] = labels
        result = self._invoke_post_pipeline([result])[0]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        messages = list(result.get('messages') or [])
        messages.append({'role': 'assistant', 'content': response_text})
        result['messages'] = messages
        return result


class FakeClientSampler:
    """Script-driven sampler mirroring ``vLLMSampler.sample()``.

    Adapts the ``twinkle_agentic`` fake sampler to the client HTTP contract:
    ``sample()`` accepts ``(inputs, sampling_params=<dict>, ...)`` and returns
    ``List[SampleResponseModel]`` (pydantic), each sequence carrying a populated
    ``new_input_feature`` so the multi-turn loop can proceed.

    Each trajectory is identified by a hidden ``_tid`` field that survives
    ``encode`` / ``concat_input_feature`` / ``extend_with_bridge`` (all preserve
    top-level keys), so we can look up its scripted turns regardless of how the
    active set shrinks across rounds.
    """

    def __init__(self, template: FakeTemplate, scripts: Dict[int, List[Dict[str, Any]]]) -> None:
        self.template = template
        self.scripts = scripts
        self.turn_counters: Dict[int, int] = defaultdict(int)
        self.sample_calls = 0

    def sample(self, inputs, sampling_params: Optional[Dict[str, Any]] = None, **kwargs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        assert isinstance(inputs, list), f'expects a list, got {type(inputs).__name__}'
        # Contract check: the rollout coerces core-lib SamplingParams into a
        # plain dict before calling the HTTP sampler.
        assert sampling_params is None or isinstance(sampling_params, dict)

        responses: List[SampleResponseModel] = []
        for pif in inputs:
            tid = pif['_tid']
            script = self.scripts[tid]
            idx = self.turn_counters[tid]
            self.turn_counters[tid] += 1
            self.sample_calls += 1

            if idx < len(script):
                turn = script[idx]
            else:
                # Defensive fallback: terminate cleanly if over-sampled.
                turn = {'kind': 'terminal', 'stop_reason': 'stop', 'logprobs': False}

            if turn['kind'] == 'tool':
                decoded = _tool_call_text('search', {'q': f't{idx}'})
                stop_reason = 'stop'
            else:
                decoded = f'final-{idx}'
                stop_reason = turn['stop_reason']

            raw = decoded + '<|im_end|>'
            tokens = self.template.tokenizer.encode(raw, add_special_tokens=False)
            logprobs = ([[(int(t), -0.1)] for t in tokens] if turn['logprobs'] else None)
            new_pif = self.template.concat_input_feature(pif, tokens)

            seq = SampledSequenceModel(
                stop_reason=stop_reason,
                tokens=tokens,
                logprobs=logprobs,
                decoded=decoded,
                new_input_feature=new_pif,
            )
            responses.append(SampleResponseModel(sequences=[seq]))
        return responses


class EchoTool(Tool):
    """Echoes its arguments as a JSON string."""

    def __init__(self, name: str = 'search'):
        self._name = name

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        return f'echo[{tool_name}]:{json.dumps(arguments, sort_keys=True)}'

    def tool_info(self):
        return {
            'type': 'function',
            'function': {
                'name': self._name,
                'description': 'echo test tool',
                'parameters': {},
            },
        }


# =============================================================================
# Helpers
# =============================================================================
def _tool_call_text(name: str, arguments: Dict[str, Any]) -> str:
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


def _count_trainable(labels: List[int]) -> int:
    return sum(1 for label in labels if label != -100)


def _make_tool_manager() -> ToolManager:
    mgr = ToolManager({})
    mgr.register(EchoTool('search'))
    return mgr


def _build_from_scripts(scripts_spec: List[Dict[str, Any]]):
    """Turn a list of per-trajectory specs into (rollout inputs, sampler).

    ``scripts_spec[k]`` = {'num_tools': int, 'terminal': 'stop'|'length',
                           'logprobs': bool}. Each trajectory's script is
    ``num_tools`` tool-call turns followed by one terminal turn.
    """
    tokenizer = FakeTokenizer()
    template = FakeTemplate(tokenizer)

    scripts: Dict[int, List[Dict[str, Any]]] = {}
    trajectories: List[Dict[str, Any]] = []
    for tid, spec in enumerate(scripts_spec):
        turns: List[Dict[str, Any]] = []
        for _ in range(spec['num_tools']):
            turns.append({'kind': 'tool', 'stop_reason': 'stop', 'logprobs': spec['logprobs']})
        turns.append({'kind': 'terminal', 'stop_reason': spec['terminal'], 'logprobs': spec['logprobs']})
        scripts[tid] = turns
        trajectories.append({'messages': [{'role': 'user', 'content': f'q{tid}'}], '_tid': tid})

    sampler = FakeClientSampler(template, scripts)
    return trajectories, sampler, template


# =============================================================================
# Hypothesis strategies
# =============================================================================
_MAX_TOOLS = 5


def _traj_spec():
    return st.fixed_dictionaries({
        'num_tools': st.integers(min_value=0, max_value=_MAX_TOOLS),
        'terminal': st.sampled_from(['stop', 'length']),
        'logprobs': st.booleans(),
    })


def _batch_specs(min_size: int = 1, max_size: int = 4):
    return st.lists(_traj_spec(), min_size=min_size, max_size=max_size)


# =============================================================================
# Multi-turn output length & order preservation
# =============================================================================
@settings(deadline=None, max_examples=60)
@given(scripts_spec=_batch_specs(min_size=0, max_size=5), max_turns=st.integers(min_value=1, max_value=6))
def test_output_length_and_order_preserved(scripts_spec, max_turns):
    """``__call__`` returns a list of the same length as the input, in the exact
    same order (verified via the hidden per-trajectory ``_tid`` tag)."""
    trajectories, sampler, template = _build_from_scripts(scripts_spec)
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=max_turns)

    outs = rollout(copy.deepcopy(trajectories))

    assert len(outs) == len(trajectories)
    for i, out in enumerate(outs):
        assert out['_tid'] == i, 'output order must match input order'


# =============================================================================
# stop_reason value range
# =============================================================================
@settings(deadline=None, max_examples=60)
@given(scripts_spec=_batch_specs(), max_turns=st.integers(min_value=1, max_value=6))
def test_stop_reason_value_range(scripts_spec, max_turns):
    """Every returned trajectory's ``stop_reason`` is one of
    ``{'length', 'stop', 'max_turns'}``."""
    trajectories, sampler, template = _build_from_scripts(scripts_spec)
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=max_turns)

    outs = rollout(copy.deepcopy(trajectories))

    for out in outs:
        assert out['stop_reason'] in {'length', 'stop', 'max_turns'}, out['stop_reason']


# =============================================================================
# logprobs / trainable-label alignment
# =============================================================================
@settings(deadline=None, max_examples=60)
@given(scripts_spec=_batch_specs(), max_turns=st.integers(min_value=1, max_value=6))
def test_logprobs_align_with_trainable_labels(scripts_spec, max_turns):
    """For every returned trajectory with a non-empty ``logprobs`` list, its length
    equals the number of trainable labels (``label != -100``)."""
    trajectories, sampler, template = _build_from_scripts(scripts_spec)
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=max_turns)

    outs = rollout(copy.deepcopy(trajectories))

    for out in outs:
        logprobs = out.get('logprobs')
        if logprobs:
            trainable = _count_trainable(out.get('labels') or [])
            assert len(logprobs) == trainable, (
                f'logprobs({len(logprobs)}) != trainable labels({trainable})')


# =============================================================================
# Actual turns never exceed max_turns
# =============================================================================
@settings(deadline=None, max_examples=60)
@given(scripts_spec=_batch_specs(), max_turns=st.integers(min_value=1, max_value=6))
def test_turns_do_not_exceed_max_turns(scripts_spec, max_turns):
    """Every returned trajectory's ``turns`` count is <= the configured
    ``max_turns``."""
    trajectories, sampler, template = _build_from_scripts(scripts_spec)
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=max_turns)

    outs = rollout(copy.deepcopy(trajectories))

    for out in outs:
        assert out['turns'] <= max_turns, f"turns({out['turns']}) > max_turns({max_turns})"


# =============================================================================
# Forced truncation at the max_turns edge
# =============================================================================
@settings(deadline=None, max_examples=60)
@given(logprobs_flags=st.lists(st.booleans(), min_size=1, max_size=5))
def test_max_turns_one_forces_truncation(logprobs_flags):
    """With ``max_turns == 1`` and a first-round tool_call, every trajectory is
    marked ``truncated=True`` and ``stop_reason='max_turns'`` and stops after
    exactly one turn."""
    # Every trajectory emits a tool_call on its first (and only allowed) turn.
    scripts_spec = [{'num_tools': 3, 'terminal': 'stop', 'logprobs': lp} for lp in logprobs_flags]
    trajectories, sampler, template = _build_from_scripts(scripts_spec)
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=1)

    outs = rollout(copy.deepcopy(trajectories))

    assert len(outs) == len(trajectories)
    for out in outs:
        assert out['truncated'] is True
        assert out['stop_reason'] == 'max_turns'
        assert out['turns'] == 1


# =============================================================================
# Deterministic unit tests: exception paths & dependency reuse (non-hypothesis)
#
# These cover the failure/edge contract described in the module docstring of
# ``twinkle_client.rollout.multi_turn``.
# All are CPU-only and reuse the Fake infra above.
# =============================================================================
class NetworkError(Exception):
    """Stand-in for a requests-style transport error (connection reset/timeout)."""


class _NullFeatureSampler:
    """Sampler that violates the contract by returning ``new_input_feature=None``.

    Mirrors ``vLLMSampler.sample()`` shape (returns ``List[SampleResponseModel]``)
    but every sequence lacks ``new_input_feature``, which must make the multi-turn
    loop raise a batch/trajectory-indexed ``RuntimeError``.
    """

    def __init__(self, template: FakeTemplate) -> None:
        self.template = template
        self.sample_calls = 0

    def sample(self, inputs, sampling_params=None, **kwargs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        self.sample_calls += 1
        responses: List[SampleResponseModel] = []
        for _ in inputs:
            seq = SampledSequenceModel(
                stop_reason='stop',
                tokens=[0, 1],
                logprobs=None,
                decoded='final',
                new_input_feature=None,  # contract violation under test
            )
            responses.append(SampleResponseModel(sequences=[seq]))
        return responses


class _NetworkFailingSampler:
    """Sampler whose ``sample()`` always raises a network-like exception.

    Used to assert the rollout NEVER wraps or swallows transport errors coming
    from ``vLLMSampler.sample()``; they must propagate unchanged.
    """

    def __init__(self, template: FakeTemplate, exc: Exception) -> None:
        self.template = template
        self._exc = exc
        self.sample_calls = 0

    def sample(self, inputs, sampling_params=None, **kwargs):
        self.sample_calls += 1
        raise self._exc


def test_missing_new_input_feature_raises_indexed_runtime_error():
    """new_input_feature=None -> RuntimeError naming batch AND trajectory index.

    _Requirements: 3.7_
    """
    trajectories, _script_sampler, template = _build_from_scripts(
        [{'num_tools': 0, 'terminal': 'stop', 'logprobs': False}])
    sampler = _NullFeatureSampler(template)
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=3)

    with pytest.raises(RuntimeError) as excinfo:
        rollout(copy.deepcopy(trajectories))

    msg = str(excinfo.value)
    # Message must carry both the batch index and the trajectory index so the
    # failure is localizable in a batched HTTP round.
    assert 'batch index 0' in msg, msg
    assert 'trajectory 0' in msg, msg
    assert 'new_input_feature' in msg, msg


def test_tool_calls_without_tool_manager_raises_value_error():
    """tool_calls produced but tool_manager missing -> ValueError.

    _Requirements: 3.10, 4.5_
    """
    # One tool-call turn then a terminal turn; max_turns=2 so the tool-dispatch
    # site (not the max_turns truncation edge) is what fails.
    trajectories, sampler, template = _build_from_scripts(
        [{'num_tools': 1, 'terminal': 'stop', 'logprobs': False}])
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=None, max_turns=2)

    with pytest.raises(ValueError) as excinfo:
        rollout(copy.deepcopy(trajectories))

    msg = str(excinfo.value)
    assert 'tool_manager' in msg, msg
    assert 'trajectory 0' in msg, msg


def test_tool_calls_without_tool_manager_via_per_call_kwarg_raises_value_error():
    """Passing tool_manager=None as a per-call kwarg also raises at dispatch.

    _Requirements: 3.10, 4.5_
    """
    trajectories, sampler, template = _build_from_scripts(
        [{'num_tools': 1, 'terminal': 'stop', 'logprobs': False}])
    # Constructed WITH a manager, but the per-call override nulls it out.
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=2)

    with pytest.raises(ValueError):
        rollout(copy.deepcopy(trajectories), tool_manager=None)


def test_sampler_network_error_propagates_unchanged():
    """vLLMSampler.sample() network error propagates unchanged (not swallowed/wrapped).

    _Requirements: 3.9_
    """
    trajectories, _script_sampler, template = _build_from_scripts(
        [{'num_tools': 0, 'terminal': 'stop', 'logprobs': False}])
    sentinel = NetworkError('simulated connection reset by peer')
    sampler = _NetworkFailingSampler(template, sentinel)
    rollout = ClientMultiTurnRollout(
        sampler=sampler, template=template, tool_manager=_make_tool_manager(), max_turns=3)

    with pytest.raises(NetworkError) as excinfo:
        rollout(copy.deepcopy(trajectories))

    # Same exact exception object, neither re-wrapped nor replaced.
    assert excinfo.value is sentinel
    assert str(excinfo.value) == 'simulated connection reset by peer'
    assert sampler.sample_calls == 1


def test_dependencies_are_reused_not_reimplemented():
    """ClientMultiTurnRollout imports (does not copy) ToolManager & extend_with_bridge.

    _Requirements: 4.3, 4.4, 7.1_
    """
    import twinkle_agentic.rollout.bridge as bridge_mod
    import twinkle_agentic.tools.tool_manager as tool_manager_mod
    import twinkle_client.rollout.multi_turn as m

    # Same object identity => the symbols are imported from the shared core-lib
    # modules rather than re-defined locally.
    assert m.ToolManager is tool_manager_mod.ToolManager
    assert m.extend_with_bridge is bridge_mod.extend_with_bridge
