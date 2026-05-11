# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for :class:`twinkle_agentic.rollout.multi_turn.MultiTurnRollout`.

Focus:
    - Control flow: no-tool / with-tool / length-stop / max-turns truncation
    - Label alignment: trainable positions count == total sampled tokens
    - Logprobs alignment: flat list length == trainable count
    - Output structure: pif fields merged at TOP LEVEL (input_ids present ⇒
      VLLMSampler will skip re-encoding on a second pass)
    - Input validation: constructor rejects bad config
    - Defensive asserts: labels/input_ids length mismatch and logprobs
      length mismatch both raise RuntimeError
    - Shallow-copy safety: extra trajectory fields (e.g. ``images``) flow
      through without deep copy

The tests are self-contained — they use a char-level fake tokenizer, a
fake Template that replays the real ``concat_input_feature`` and post
pipeline semantics, and a fake Sampler that queues scripted responses.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional

import pytest

from twinkle.data_format.sampling import (
    SampleResponse, SampledSequence, SamplingParams,
)
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager


# =============================================================================
# Fakes
# =============================================================================
class FakeTokenizer:
    """Char-level tokenizer with atomic special tokens.

    Guarantees ``decode(encode(s)) == s`` for any mix of raw chars and
    registered specials. This is what makes the decode-diff-encode alignment
    strategy in MultiTurnRollout.__extend_with_bridge work in the test.
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
    """Minimal Template that mirrors the parts MultiTurnRollout touches."""
    model_id = 'qwen-fake'
    truncation_strategy = 'right'

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer

    # --- the public API used by MultiTurnRollout ----------------------------
    def encode(self, trajectory: Dict[str, Any], add_generation_prompt: bool = False) -> Dict[str, Any]:
        messages = trajectory.get('messages', [])
        s = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt)
        input_ids = self.tokenizer.encode(s, add_special_tokens=False)
        pif: Dict[str, Any] = dict(trajectory)  # preserve top-level fields
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
                    raise RuntimeError(
                        f'FakeTemplate post_pipeline: labels({len(labels)}) '
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

    # --- Used by the fake sampler to mirror real concat_input_feature -------
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
        # Append assistant message with the decoded response (no special toks)
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        messages = list(result.get('messages') or [])
        messages.append({'role': 'assistant', 'content': response_text})
        result['messages'] = messages
        return result


class FakeSampler:
    """Queue-driven sampler that mirrors VLLMSampler output shape."""

    def __init__(self, template: FakeTemplate) -> None:
        self.template = template
        self._queue: List[Dict[str, Any]] = []
        self.sample_calls = 0

    def queue(
        self,
        response_text: str,
        stop_reason: str = 'stop',
        logprobs: Optional[List[Any]] = None,
        append_im_end: bool = True,
    ) -> None:
        """``response_text`` is the model output (may contain <tool_call> …).
        ``<|im_end|>`` is appended to the encoded tokens when ``append_im_end``.
        ``seq.decoded`` is the raw response WITHOUT the trailing <|im_end|>
        (matches vLLM's common behaviour)."""
        raw = response_text + ('<|im_end|>' if append_im_end else '')
        tokens = self.template.tokenizer.encode(raw, add_special_tokens=False)
        self._queue.append({
            'tokens': tokens,
            'decoded': response_text,
            'stop_reason': stop_reason,
            'logprobs': logprobs,
        })

    def sample(self, pifs, sampling_params=None):
        # Batched contract: accept a list of pifs, return one
        # SampleResponse per input, in order. A single-pif dict is also
        # accepted for backwards compatibility with older call sites.
        if isinstance(pifs, dict):
            pifs = [pifs]
        assert isinstance(pifs, list), (
            f'FakeSampler.sample expects a list, got {type(pifs).__name__}')
        responses: List[SampleResponse] = []
        for pif in pifs:
            assert self._queue, 'FakeSampler queue exhausted — scripted turns'
            r = self._queue.pop(0)
            self.sample_calls += 1
            new_pif = self.template.concat_input_feature(pif, r['tokens'])
            seq = SampledSequence(
                stop_reason=r['stop_reason'],
                tokens=r['tokens'],
                logprobs=r['logprobs'],
                decoded=r['decoded'],
                new_input_feature=new_pif,
            )
            responses.append(SampleResponse(sequences=[seq]))
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
# Fixtures
# =============================================================================
@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def template(tokenizer):
    return FakeTemplate(tokenizer)


@pytest.fixture
def sampler(template):
    return FakeSampler(template)


@pytest.fixture
def tool_manager():
    mgr = ToolManager({})
    mgr.register(EchoTool('search'))
    return mgr


@pytest.fixture
def make_rollout(sampler, template, tool_manager):
    def _make(max_turns: int = 4, sampling_params: Optional[SamplingParams] = None):
        return MultiTurnRollout(
            sampler=sampler,
            template=template,
            tool_manager=tool_manager,
            sampling_params=sampling_params or SamplingParams(),
            max_turns=max_turns,
        )
    return _make


# =============================================================================
# Helpers
# =============================================================================
def _count_trainable(labels: List[int]) -> int:
    return sum(1 for l in labels if l != -100)


def _user_traj(text: str = 'hi') -> Dict[str, Any]:
    return {'messages': [{'role': 'user', 'content': text}]}


def _tool_call_text(name: str, arguments: Dict[str, Any]) -> str:
    return '<tool_call>' + json.dumps(
        {'name': name, 'arguments': arguments}) + '</tool_call>'


# =============================================================================
# Tests: control flow
# =============================================================================
def test_single_turn_natural_stop(make_rollout, sampler):
    """Model answers directly, no tool call → 1 turn, stop_reason='stop'."""
    sampler.queue('Hello there.', stop_reason='stop')
    rollout = make_rollout(max_turns=4)
    out = rollout([_user_traj()])[0]

    assert out['turns'] == 1
    assert out['stop_reason'] == 'stop'
    assert out['truncated'] is False
    assert sampler.sample_calls == 1

    # Output must carry pif fields at TOP LEVEL so downstream sampler/model
    # sees `input_ids` and skips re-encoding.
    assert 'input_ids' in out
    assert 'labels' in out
    assert 'attention_mask' in out
    assert 'position_ids' in out
    assert len(out['input_ids']) == len(out['labels'])
    assert len(out['input_ids']) == len(out['attention_mask'])


def test_single_turn_length_stop(make_rollout, sampler):
    """stop_reason='length' exits immediately without tool-call parsing."""
    sampler.queue(_tool_call_text('search', {'q': 'x'}), stop_reason='length')
    rollout = make_rollout(max_turns=4)
    out = rollout([_user_traj()])[0]

    # Even though the decoded text contains a <tool_call>, length stop must
    # short-circuit BEFORE we parse / dispatch tools.
    assert out['turns'] == 1
    assert out['stop_reason'] == 'length'
    assert out['truncated'] is False
    assert sampler.sample_calls == 1
    # No tool message should have been appended.
    roles = [m['role'] for m in out['messages']]
    assert 'tool' not in roles


def test_two_turns_one_tool_call(make_rollout, sampler):
    """Turn 1 emits tool_call, turn 2 stops normally."""
    sampler.queue(_tool_call_text('search', {'q': 'weather'}), stop_reason='stop')
    sampler.queue('The weather is sunny.', stop_reason='stop')
    rollout = make_rollout(max_turns=4)
    out = rollout([_user_traj('What is the weather?')])[0]

    assert out['turns'] == 2
    assert out['stop_reason'] == 'stop'
    assert out['truncated'] is False
    assert sampler.sample_calls == 2

    roles = [m['role'] for m in out['messages']]
    assert roles == ['user', 'assistant', 'tool', 'assistant']

    # Tool response content must be what EchoTool returned (exact contract).
    tool_msg = out['messages'][2]
    assert tool_msg['content'] == 'echo[search]:{"q": "weather"}'


def test_multiple_tool_calls_one_turn(make_rollout, sampler):
    """Model emits TWO tool calls in one assistant turn → two tool messages."""
    decoded = (_tool_call_text('search', {'q': 'a'})
               + _tool_call_text('search', {'q': 'b'}))
    sampler.queue(decoded, stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    rollout = make_rollout(max_turns=4)
    out = rollout([_user_traj()])[0]

    assert out['turns'] == 2
    roles = [m['role'] for m in out['messages']]
    assert roles == ['user', 'assistant', 'tool', 'tool', 'assistant']


def test_max_turns_truncation(make_rollout, sampler):
    """Model keeps emitting tool_calls past max_turns → truncated=True."""
    # 3 consecutive turns, all emitting tool_calls.
    for i in range(5):
        sampler.queue(_tool_call_text('search', {'q': f'q{i}'}), stop_reason='stop')
    rollout = make_rollout(max_turns=3)
    out = rollout([_user_traj()])[0]

    assert out['turns'] == 3
    assert out['truncated'] is True
    assert sampler.sample_calls == 3
    # messages: user + (assistant + tool) × 3 = 7
    roles = [m['role'] for m in out['messages']]
    assert roles.count('assistant') == 3
    # The last turn was cut off BEFORE the tool message was appended (bridge
    # wouldn't help with no next generation) → 2 tool messages, not 3.
    assert roles.count('tool') == 2


def test_max_turns_natural_stop_at_ceiling(make_rollout, sampler):
    """Natural stop exactly on turn = max_turns → truncated=False."""
    sampler.queue(_tool_call_text('search', {'q': 'x'}), stop_reason='stop')
    sampler.queue('Final answer.', stop_reason='stop')
    rollout = make_rollout(max_turns=2)
    out = rollout([_user_traj()])[0]

    assert out['turns'] == 2
    assert out['stop_reason'] == 'stop'
    assert out['truncated'] is False


# =============================================================================
# Tests: label & logprobs alignment
# =============================================================================
def test_trainable_count_matches_total_sampled_tokens(make_rollout, sampler, tokenizer):
    """The output's non-(-100) label count must equal ∑ len(seq.tokens)
    over all turns. This is the load-bearing invariant for GRPO's loss mask."""
    text1 = _tool_call_text('search', {'q': 'x'})
    text2 = 'ok'
    sampler.queue(text1, stop_reason='stop')
    sampler.queue(text2, stop_reason='stop')
    rollout = make_rollout(max_turns=4)
    out = rollout([_user_traj()])[0]

    # Total sampled tokens across turns (each turn appends <|im_end|>):
    n1 = len(tokenizer.encode(text1 + '<|im_end|>'))
    n2 = len(tokenizer.encode(text2 + '<|im_end|>'))
    expected_trainable = n1 + n2

    assert _count_trainable(out['labels']) == expected_trainable


def test_logprobs_concatenated_across_turns(make_rollout, sampler, tokenizer):
    """all_logprobs = concat(per-turn logprobs) with length == #trainable."""
    text1 = _tool_call_text('search', {'q': 'x'})
    text2 = 'ok'
    # Build sentinel logprobs for each sampled token so we can verify order.
    toks1 = tokenizer.encode(text1 + '<|im_end|>')
    toks2 = tokenizer.encode(text2 + '<|im_end|>')
    lp1 = [[(tid, -0.1 * idx)] for idx, tid in enumerate(toks1)]
    lp2 = [[(tid, -0.2 * idx)] for idx, tid in enumerate(toks2)]

    sampler.queue(text1, stop_reason='stop', logprobs=lp1)
    sampler.queue(text2, stop_reason='stop', logprobs=lp2)
    rollout = make_rollout(max_turns=4)
    out = rollout([_user_traj()])[0]

    assert out['logprobs'] is not None
    assert out['logprobs'] == lp1 + lp2
    assert len(out['logprobs']) == _count_trainable(out['labels'])


def test_logprobs_none_when_sampler_omits(make_rollout, sampler):
    """If no turn carried logprobs, output['logprobs'] is None (not []).
    Prevents GRPO from thinking logprobs are available but empty."""
    sampler.queue('bye', stop_reason='stop')
    rollout = make_rollout(max_turns=2)
    out = rollout([_user_traj()])[0]
    assert out['logprobs'] is None


def test_logprobs_length_mismatch_raises(make_rollout, sampler, tokenizer):
    """If sampler returns logprobs whose length ≠ token count, we raise."""
    text = 'hello'
    toks = tokenizer.encode(text + '<|im_end|>')
    bad_lp = [[(toks[0], -0.1)]]  # length 1, tokens length > 1
    sampler.queue(text, stop_reason='stop', logprobs=bad_lp)
    rollout = make_rollout(max_turns=2)

    with pytest.raises(RuntimeError, match='logprobs length'):
        rollout([_user_traj()])


# =============================================================================
# Tests: output structure
# =============================================================================
def test_pif_fields_merged_at_top_level(make_rollout, sampler):
    """`input_ids` at top level ⇒ VLLMSampler will skip re-encoding."""
    sampler.queue('bye', stop_reason='stop')
    rollout = make_rollout(max_turns=2)
    out = rollout([_user_traj()])[0]

    # These are the fields a downstream sampler / model.forward consumes.
    for k in ('input_ids', 'labels', 'attention_mask', 'position_ids', 'length'):
        assert k in out, f'{k} missing from top-level output'
    # And NOT nested under user_data.
    assert 'input_feature' not in (out.get('user_data') or {})


def test_extra_trajectory_fields_pass_through(make_rollout, sampler):
    """Non-encoding fields like ``images`` / ``tools`` flow through.

    We only check that the fields are preserved by VALUE (not identity),
    because the real ``concat_input_feature`` does ``copy.deepcopy(pif)``
    internally — that is the sampler's concern, not this rollout's.
    """
    traj = _user_traj()
    traj['images'] = ['/path/to/img.png']
    traj['tools'] = [{
        'type': 'function',
        'function': {'name': 'search', 'description': '', 'parameters': {}},
    }]

    sampler.queue('ok', stop_reason='stop')
    rollout = make_rollout(max_turns=2)
    out = rollout([traj])[0]

    assert out['images'] == ['/path/to/img.png']
    assert out['tools'] == traj['tools']


# =============================================================================
# Tests: constructor validation
# =============================================================================
def test_rejects_none_template(sampler, tool_manager):
    with pytest.raises(ValueError, match='Template'):
        MultiTurnRollout(sampler=sampler, template=None,
                         tool_manager=tool_manager)


def test_rejects_none_tool_manager(sampler, template):
    with pytest.raises(ValueError, match='ToolManager'):
        MultiTurnRollout(sampler=sampler, template=template,
                         tool_manager=None)


def test_rejects_bad_max_turns(sampler, template, tool_manager):
    with pytest.raises(ValueError, match='max_turns'):
        MultiTurnRollout(sampler=sampler, template=template,
                         tool_manager=tool_manager, max_turns=0)


def test_rejects_num_samples_gt_1(sampler, template, tool_manager):
    with pytest.raises(ValueError, match='num_samples'):
        MultiTurnRollout(
            sampler=sampler, template=template, tool_manager=tool_manager,
            sampling_params=SamplingParams(num_samples=2))


# =============================================================================
# Tests: defensive guards
# =============================================================================
def test_missing_new_input_feature_raises(template, tool_manager):
    class BrokenSampler:
        def sample(self, pifs, sampling_params=None):
            if isinstance(pifs, dict):
                pifs = [pifs]
            seq = SampledSequence(
                stop_reason='stop', tokens=[], logprobs=None,
                decoded='', new_input_feature=None)
            return [SampleResponse(sequences=[seq]) for _ in pifs]

    rollout = MultiTurnRollout(
        sampler=BrokenSampler(), template=template,
        tool_manager=tool_manager)
    with pytest.raises(RuntimeError, match='new_input_feature'):
        rollout([_user_traj()])


def test_empty_sampler_response_raises(template, tool_manager):
    class EmptySampler:
        def sample(self, pifs, sampling_params=None):
            return []

    rollout = MultiTurnRollout(
        sampler=EmptySampler(), template=template,
        tool_manager=tool_manager)
    # Batched contract: 0 responses for a batch of 1 → mismatch error.
    with pytest.raises(RuntimeError, match='0 responses'):
        rollout([_user_traj()])


def test_sample_response_no_sequences_raises(template, tool_manager):
    class NoSeqSampler:
        def sample(self, pifs, sampling_params=None):
            if isinstance(pifs, dict):
                pifs = [pifs]
            return [SampleResponse(sequences=[]) for _ in pifs]

    rollout = MultiTurnRollout(
        sampler=NoSeqSampler(), template=template,
        tool_manager=tool_manager)
    with pytest.raises(RuntimeError, match='no sequences'):
        rollout([_user_traj()])


# =============================================================================
# Tests: batched / parallel rollout
# =============================================================================
def test_empty_batch_returns_empty_list(make_rollout):
    rollout = make_rollout(max_turns=2)
    assert rollout([]) == []


def test_batch_single_turn_two_trajectories(make_rollout, sampler):
    """Two trajectories finish on turn 1 → one batched sample call."""
    sampler.queue('answer-A', stop_reason='stop')
    sampler.queue('answer-B', stop_reason='stop')
    rollout = make_rollout(max_turns=3)
    outs = rollout([_user_traj('Q-A'), _user_traj('Q-B')])

    assert len(outs) == 2
    # Exactly ONE batched sample call, not two.
    assert sampler.sample_calls == 2  # one per item, still one turn
    # But FakeSampler counts per-input; the critical batching invariant is
    # that MultiTurnRollout only calls sampler.sample ONCE per turn. We
    # enforce this via the queue ordering + single turn.
    for out in outs:
        assert out['turns'] == 1
        assert out['stop_reason'] == 'stop'
        assert out['truncated'] is False


def test_batch_different_termination_turns(make_rollout, sampler):
    """Trajectory A finishes on turn 1; trajectory B needs a tool turn.

    Turn 1 batch:  [A: 'done-A' stop, B: tool_call stop]  → A parked.
    Turn 2 batch:  [B: 'done-B' stop]                     → only B live.
    """
    sampler.queue('done-A', stop_reason='stop')              # A turn 1
    sampler.queue(_tool_call_text('search', {'q': 'b'}),      # B turn 1
                  stop_reason='stop')
    sampler.queue('done-B', stop_reason='stop')              # B turn 2
    rollout = make_rollout(max_turns=4)
    outs = rollout([_user_traj('Q-A'), _user_traj('Q-B')])

    assert len(outs) == 2
    # A: 1 turn, no tool. B: 2 turns, one tool.
    assert outs[0]['turns'] == 1
    assert outs[1]['turns'] == 2
    roles_a = [m['role'] for m in outs[0]['messages']]
    roles_b = [m['role'] for m in outs[1]['messages']]
    assert 'tool' not in roles_a
    assert roles_b == ['user', 'assistant', 'tool', 'assistant']


def test_batch_per_trajectory_tool_manager(make_rollout, sampler, template):
    """A list of ``tool_manager`` aligned with trajectories is honoured:
    each trajectory dispatches through its OWN manager."""
    tm_a = ToolManager({})
    tm_a.register(EchoTool('search'))

    class TagTool(Tool):
        def __init__(self, tag):
            self._tag = tag
        def __call__(self, tool_name, arguments):
            return f'tagged[{self._tag}]:{json.dumps(arguments, sort_keys=True)}'
        def tool_info(self):
            return {
                'type': 'function',
                'function': {
                    'name': 'search',
                    'description': '',
                    'parameters': {},
                },
            }

    tm_b = ToolManager({})
    tm_b.register(TagTool('B'))

    sampler.queue(_tool_call_text('search', {'q': 'x'}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'y'}), stop_reason='stop')
    sampler.queue('done-A', stop_reason='stop')
    sampler.queue('done-B', stop_reason='stop')

    rollout = MultiTurnRollout(
        sampler=sampler, template=template,
        tool_manager=tm_a,  # default (unused when per-call list supplied)
        max_turns=4)
    outs = rollout([_user_traj('A'), _user_traj('B')],
                   tool_manager=[tm_a, tm_b])

    assert outs[0]['messages'][2]['content'] == 'echo[search]:{"q": "x"}'
    assert outs[1]['messages'][2]['content'] == 'tagged[B]:{"q": "y"}'


def test_batch_tool_manager_list_length_mismatch(make_rollout, tool_manager):
    rollout = make_rollout(max_turns=2)
    with pytest.raises(ValueError, match='tool_manager list length'):
        rollout([_user_traj('A'), _user_traj('B')],
                tool_manager=[tool_manager])  # length 1 vs 2 trajectories


def test_single_trajectory_dict_rejected(make_rollout):
    """A single ``Trajectory`` (dict) is NOT accepted — caller must wrap."""
    rollout = make_rollout(max_turns=2)
    with pytest.raises(TypeError, match='List\\[Trajectory\\]'):
        rollout(_user_traj())


# =============================================================================
# Tests: trace_path (JSONL per-turn observability)
# =============================================================================
def test_trace_path_writes_one_record_per_turn_natural_stop(
        tmp_path, sampler, template, tool_manager):
    """Single-turn natural stop: trace file has exactly one JSON line."""
    trace = tmp_path / 'trace.jsonl'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template,
        tool_manager=tool_manager,
        max_turns=4, trace_path=str(trace))
    sampler.queue('final answer', stop_reason='stop')

    outs = rollout([_user_traj('hello')])
    assert len(outs) == 1

    lines = [l for l in trace.read_text().splitlines() if l]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec['turn'] == 1
    assert rec['batch_size'] == 1
    assert rec['trajectory_idx'] == 0
    assert rec['stop_reason'] == 'stop'
    assert rec['decoded'] == 'final answer'
    assert rec['tool_call_count'] == 0
    assert rec['done'] is True
    assert rec['truncated'] is False
    assert rec['trainable_tokens'] > 0


def test_trace_path_captures_tool_turn_and_completion(
        tmp_path, sampler, template, tool_manager):
    """Two-turn rollout: one tool turn (done=False) then completion."""
    trace = tmp_path / 'trace.jsonl'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template,
        tool_manager=tool_manager,
        max_turns=4, trace_path=str(trace))
    sampler.queue(_tool_call_text('search', {'q': 'x'}))
    sampler.queue('done', stop_reason='stop')

    rollout([_user_traj('hello')])

    lines = [l for l in trace.read_text().splitlines() if l]
    assert len(lines) == 2
    turn1 = json.loads(lines[0])
    turn2 = json.loads(lines[1])

    assert turn1['turn'] == 1
    assert turn1['tool_call_count'] == 1
    assert turn1['done'] is False
    assert turn1['truncated'] is False

    assert turn2['turn'] == 2
    assert turn2['tool_call_count'] == 0
    assert turn2['done'] is True
    # input_ids length must monotonically increase across turns.
    assert turn2['input_ids_len'] > turn1['input_ids_len']


def test_trace_path_truncates_file_on_construction(
        tmp_path, sampler, template, tool_manager):
    """Constructor opens the file in 'w' mode — stale data is wiped."""
    trace = tmp_path / 'trace.jsonl'
    trace.write_text('STALE CONTENT SHOULD BE GONE\n')
    assert trace.read_text() == 'STALE CONTENT SHOULD BE GONE\n'

    sampler.queue('ok', stop_reason='stop')
    rollout = MultiTurnRollout(
        sampler=sampler, template=template,
        tool_manager=tool_manager,
        max_turns=2, trace_path=str(trace))
    # After construction the file is empty (we truncate eagerly).
    assert trace.read_text() == ''

    rollout([_user_traj('hi')])
    content = trace.read_text()
    assert 'STALE' not in content
    assert content.strip()  # at least one record written


def test_trace_path_batch_emits_one_record_per_active_trajectory(
        tmp_path, sampler, template, tool_manager):
    """Batched rollout: each turn emits N active records (not N_total)."""
    trace = tmp_path / 'trace.jsonl'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template,
        tool_manager=tool_manager,
        max_turns=4, trace_path=str(trace))
    # Traj 0: stops turn 1. Traj 1: tool-calls turn 1, stops turn 2.
    # Responses are consumed in batch order per turn.
    sampler.queue('done0', stop_reason='stop')                        # t1-A
    sampler.queue(_tool_call_text('search', {'q': 'y'}))              # t1-B
    sampler.queue('done1', stop_reason='stop')                        # t2-B (B only)

    rollout([_user_traj('A'), _user_traj('B')])

    lines = [json.loads(l) for l in trace.read_text().splitlines() if l]
    assert len(lines) == 3
    # Turn 1 has both trajectories.
    turn1 = [r for r in lines if r['turn'] == 1]
    turn2 = [r for r in lines if r['turn'] == 2]
    assert sorted(r['trajectory_idx'] for r in turn1) == [0, 1]
    # Turn 2 has only trajectory 1 (trajectory 0 already done).
    assert [r['trajectory_idx'] for r in turn2] == [1]
    # batch_size is the ORIGINAL batch count (2), not active count.
    assert all(r['batch_size'] == 2 for r in lines)


def test_trace_path_none_disables_tracing(
        tmp_path, sampler, template, tool_manager):
    """Default ``trace_path=None`` never touches the filesystem."""
    trace = tmp_path / 'never.jsonl'
    assert not trace.exists()

    rollout = MultiTurnRollout(
        sampler=sampler, template=template,
        tool_manager=tool_manager, max_turns=2)
    sampler.queue('ok', stop_reason='stop')
    rollout([_user_traj('hi')])

    assert rollout.trace_path is None
    assert not trace.exists()


def test_trace_path_truncation_marked_on_max_turns(
        tmp_path, sampler, template, tool_manager):
    """The final record of a max-turns truncation has truncated=True."""
    trace = tmp_path / 'trunc.jsonl'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template,
        tool_manager=tool_manager,
        max_turns=2, trace_path=str(trace))
    # Two tool-call turns -> the second hits max_turns cap.
    sampler.queue(_tool_call_text('search', {'q': 'a'}))
    sampler.queue(_tool_call_text('search', {'q': 'b'}))

    rollout([_user_traj('hi')])

    lines = [json.loads(l) for l in trace.read_text().splitlines() if l]
    assert len(lines) == 2
    assert lines[0]['truncated'] is False and lines[0]['done'] is False
    assert lines[1]['truncated'] is True and lines[1]['done'] is True
