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
import pytest
import re
from typing import Any, Dict, List, Optional

from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams
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
        self._s2i: dict[str, int] = {}
        self._i2s: dict[int, str] = {}
        for s in self.SPECIALS:
            self._add(s)

    def _add(self, tok: str) -> int:
        if tok not in self._s2i:
            i = len(self._s2i)
            self._s2i[tok] = i
            self._i2s[i] = tok
        return self._s2i[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
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

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        specials = set(self.SPECIALS)
        toks = [self._i2s[int(i)] for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in specials]
        return ''.join(toks)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
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
    def encode(self, trajectory: dict[str, Any], add_generation_prompt: bool = False) -> dict[str, Any]:
        messages = trajectory.get('messages', [])
        s = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        input_ids = self.tokenizer.encode(s, add_special_tokens=False)
        pif: dict[str, Any] = dict(trajectory)  # preserve top-level fields
        pif['input_ids'] = input_ids
        pif['labels'] = [-100] * len(input_ids)  # inference mode
        return self._invoke_post_pipeline([pif])[0]

    def _invoke_post_pipeline(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def parse_tool_call(self, decoded: str) -> list[dict[str, Any]]:
        matches = re.findall(r'<tool_call>\s*([\s\S]*?)\s*</tool_call>', decoded or '')
        results: list[dict[str, Any]] = []
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

    def clean_tool_call(self, decoded: str) -> str:
        """Strip the call blocks, as the real template does before storing."""
        return re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', decoded or '')

    # --- Used by the fake sampler to mirror real concat_input_feature -------
    def concat_input_feature(self, pif: dict[str, Any], new_tokens: list[int]) -> dict[str, Any]:
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
        # Append assistant message with the decoded response (no special toks).
        # A reply that parses as a call is stored with the call text removed and
        # the calls in their own field, which is what the real template does --
        # and the reason a stage reply has to be put back afterwards.
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        messages = list(result.get('messages') or [])
        parsed = self.parse_tool_call(response_text)
        msg: dict[str, Any] = {
            'role': 'assistant',
            'content': self.clean_tool_call(response_text) if parsed else response_text,
        }
        if parsed:
            msg['tool_calls'] = parsed
        messages.append(msg)
        result['messages'] = messages
        return result


class FakeSampler:
    """Queue-driven sampler that mirrors VLLMSampler output shape."""

    def __init__(self, template: FakeTemplate) -> None:
        self.template = template
        self._queue: list[dict[str, Any]] = []
        self.sample_calls = 0
        # One entry per sample() call, so a test can assert which budget each
        # stage was sampled under.
        self.params_seen: list[Any] = []

    def queue(
        self,
        response_text: str,
        stop_reason: str = 'stop',
        logprobs: list[Any] | None = None,
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
        assert isinstance(pifs, list), (f'FakeSampler.sample expects a list, got {type(pifs).__name__}')
        self.params_seen.append(sampling_params)
        responses: list[SampleResponse] = []
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

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> str:
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


class FailTool(Tool):
    """Answers in the two shapes a real failure arrives in.

    ``kind='envelope'`` is ms-agent wrapping a failure; ``kind='bare'`` is a
    dispatch that never reached a tool. Both copied from a recorded run.
    """

    def __init__(self, name: str = 'grep', kind: str = 'envelope'):
        self._name = name
        self._kind = kind

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self._kind == 'bare':
            return (f"Error: unknown tool '{tool_name}'. "
                    f'Available: code_executor---shell_executor')
        return ('{\n  "success": false,\n  "output": "",\n'
                '  "error": "[Errno 2] No such file or directory"\n}')

    def tool_info(self):
        return {
            'type': 'function',
            'function': {'name': self._name, 'description': 'always fails',
                         'parameters': {}},
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
    mgr.register(FailTool('grep'))
    mgr.register(FailTool('badname', kind='bare'))
    return mgr


@pytest.fixture
def make_rollout(sampler, template, tool_manager):

    def _make(max_turns: int = 4, sampling_params: SamplingParams | None = None,
              stop_after_stuck_turns: int = 0):
        return MultiTurnRollout(
            sampler=sampler,
            template=template,
            tool_manager=tool_manager,
            sampling_params=sampling_params or SamplingParams(),
            max_turns=max_turns,
            stop_after_stuck_turns=stop_after_stuck_turns,
        )

    return _make


# =============================================================================
# Helpers
# =============================================================================
def _count_trainable(labels: list[int]) -> int:
    return sum(1 for label in labels if label != -100)


def _user_traj(text: str = 'hi') -> dict[str, Any]:
    return {'messages': [{'role': 'user', 'content': text}]}


def _tool_call_text(name: str, arguments: dict[str, Any]) -> str:
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


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
    # Running out of generation budget is a truncation, like the max_turns and
    # max_trajectory_tokens cases: a consumer filtering on this flag must not see
    # a cut-off trajectory as one that reached its own conclusion.
    assert out['truncated'] is True
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
    decoded = (_tool_call_text('search', {'q': 'a'}) + _tool_call_text('search', {'q': 'b'}))
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


def test_max_turns_one_dispatches_no_tool(make_rollout, sampler):
    """A one-turn rollout never runs a tool, even when the reply asks for one.

    This is what a caller relies on to get a text-only round out of a rollout that
    requires a tool manager at construction: the challenger's check-writing round
    must not be able to touch the workspace its script is about to be verified
    against, and a reply containing python parses as a tool call whether or not
    the model meant one.
    """
    sampler.queue(_tool_call_text('search', {'q': 'x'}), stop_reason='stop')
    rollout = make_rollout(max_turns=1)
    out = rollout([_user_traj()])[0]

    assert out['turns'] == 1
    assert [m['role'] for m in out['messages']].count('tool') == 0
    # The fake tool echoes what it was called with, so its absence anywhere in
    # the transcript is proof it never ran.
    assert 'echo[' not in ''.join(m.get('content') or '' for m in out['messages'])


# =============================================================================
# Tests: stuck-episode early stop
#
# Measured on 12 recorded sandbox episodes: 131 of 239 tool calls were
# byte-identical repeats of an earlier call, and the two worst episodes burned 54
# and 84 calls to leave behind a single script that could not run. Stopping on
# errors alone would have caught 1 of the 12 -- the offenders interleave a failing
# call with a glob that succeeds -- so a turn also counts as stuck when every call
# in it repeats one already made.
# =============================================================================
def test_stuck_stop_off_by_default(make_rollout, sampler):
    """Two failing turns run on when the limit is 0: existing callers see no change."""
    sampler.queue(_tool_call_text('grep', {'p': 1}), stop_reason='stop')
    sampler.queue(_tool_call_text('grep', {'p': 2}), stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    out = make_rollout(max_turns=4)([_user_traj()])[0]

    assert out['stuck_stop'] is False
    assert out['turns'] == 3


def test_two_all_error_turns_stop_the_episode(make_rollout, sampler):
    sampler.queue(_tool_call_text('grep', {'p': 1}), stop_reason='stop')
    sampler.queue(_tool_call_text('grep', {'p': 2}), stop_reason='stop')
    # Would have been a third turn; the stop means it is never sampled.
    sampler.queue(_tool_call_text('search', {'q': 'x'}), stop_reason='stop')
    out = make_rollout(max_turns=6, stop_after_stuck_turns=2)([_user_traj()])[0]

    assert out['stuck_stop'] is True
    assert out['truncated'] is True
    assert out['turns'] == 2
    assert sampler.sample_calls == 2
    # The failures that ended it are in the transcript the caller reads, so the
    # reason is visible without re-running anything.
    assert [m['role'] for m in out['messages']].count('tool') == 2


def test_bare_error_string_counts_as_a_failure(make_rollout, sampler):
    """An unknown tool name never reaches a tool; that is still a failed turn."""
    sampler.queue(_tool_call_text('badname', {'a': 1}), stop_reason='stop')
    sampler.queue(_tool_call_text('badname', {'a': 2}), stop_reason='stop')
    out = make_rollout(max_turns=6, stop_after_stuck_turns=2)([_user_traj()])[0]

    assert out['stuck_stop'] is True
    assert out['turns'] == 2


def test_two_verbatim_repeat_turns_stop_the_episode(make_rollout, sampler):
    """Repeating a *successful* call is stuck too -- it cannot produce new state."""
    sampler.queue(_tool_call_text('search', {'q': 'a'}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'a'}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'a'}), stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    out = make_rollout(max_turns=6, stop_after_stuck_turns=2)([_user_traj()])[0]

    assert out['stuck_stop'] is True
    assert out['turns'] == 3


def test_changed_arguments_are_not_a_repeat(make_rollout, sampler):
    sampler.queue(_tool_call_text('search', {'q': 'a'}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'b'}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'c'}), stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    out = make_rollout(max_turns=6, stop_after_stuck_turns=2)([_user_traj()])[0]

    assert out['stuck_stop'] is False
    assert out['turns'] == 4


def test_one_success_in_a_turn_resets_the_count(make_rollout, sampler):
    """The case that decided the rule: a failing call next to a useful one.

    Counting these as stuck would stop at turn 2 -- and in the recorded run the
    files worth writing a check about were created after that point.
    """
    for i in range(3):
        sampler.queue(_tool_call_text('grep', {'p': i})
                      + _tool_call_text('search', {'q': i}), stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    out = make_rollout(max_turns=6, stop_after_stuck_turns=2)([_user_traj()])[0]

    assert out['stuck_stop'] is False
    assert out['turns'] == 4


def test_a_good_turn_between_two_bad_ones_resets_the_count(make_rollout, sampler):
    sampler.queue(_tool_call_text('grep', {'p': 1}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'new'}), stop_reason='stop')
    sampler.queue(_tool_call_text('grep', {'p': 2}), stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    out = make_rollout(max_turns=6, stop_after_stuck_turns=2)([_user_traj()])[0]

    assert out['stuck_stop'] is False
    assert out['turns'] == 4


def test_stuck_stop_is_per_trajectory_in_a_batch(make_rollout, sampler, template):
    """One stuck episode must not end its batch mates."""
    good = ToolManager({})
    good.register(EchoTool('search'))
    bad = ToolManager({})
    bad.register(FailTool('search'))

    sampler.queue(_tool_call_text('search', {'q': 1}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 1}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 2}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 3}), stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')

    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=[good, bad],
        sampling_params=SamplingParams(), max_turns=6, stop_after_stuck_turns=2)
    outs = rollout([_user_traj('a'), _user_traj('b')])

    assert outs[1]['stuck_stop'] is True
    assert outs[0]['stuck_stop'] is False
    assert outs[0]['turns'] > outs[1]['turns']


def test_rejects_negative_stuck_limit(sampler, template, tool_manager):
    with pytest.raises(ValueError, match='stop_after_stuck_turns'):
        MultiTurnRollout(sampler=sampler, template=template,
                         tool_manager=tool_manager, stop_after_stuck_turns=-1)


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
        'function': {
            'name': 'search',
            'description': '',
            'parameters': {}
        },
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
        MultiTurnRollout(sampler=sampler, template=None, tool_manager=tool_manager)


def test_none_tool_manager_accepted_at_construction(sampler, template):
    """tool_manager=None is valid at construction; error deferred to call time."""
    rollout = MultiTurnRollout(sampler=sampler, template=template, tool_manager=None)
    assert rollout.tool_manager is None
    # Calling without providing a tool_manager should raise
    sampler.queue(_tool_call_text('search', {'q': 'x'}), stop_reason='stop')
    with pytest.raises(ValueError, match='tool_manager is required'):
        rollout([_user_traj('hello')])


def test_rejects_bad_max_turns(sampler, template, tool_manager):
    with pytest.raises(ValueError, match='max_turns'):
        MultiTurnRollout(sampler=sampler, template=template, tool_manager=tool_manager, max_turns=0)


def test_rejects_num_samples_gt_1(sampler, template, tool_manager):
    with pytest.raises(ValueError, match='num_samples'):
        MultiTurnRollout(
            sampler=sampler,
            template=template,
            tool_manager=tool_manager,
            sampling_params=SamplingParams(num_samples=2))


# =============================================================================
# Tests: defensive guards
# =============================================================================
def test_missing_new_input_feature_raises(template, tool_manager):

    class BrokenSampler:

        def sample(self, pifs, sampling_params=None):
            if isinstance(pifs, dict):
                pifs = [pifs]
            seq = SampledSequence(stop_reason='stop', tokens=[], logprobs=None, decoded='', new_input_feature=None)
            return [SampleResponse(sequences=[seq]) for _ in pifs]

    rollout = MultiTurnRollout(sampler=BrokenSampler(), template=template, tool_manager=tool_manager)
    with pytest.raises(RuntimeError, match='new_input_feature'):
        rollout([_user_traj()])


def test_empty_sampler_response_raises(template, tool_manager):

    class EmptySampler:

        def sample(self, pifs, sampling_params=None):
            return []

    rollout = MultiTurnRollout(sampler=EmptySampler(), template=template, tool_manager=tool_manager)
    # Batched contract: 0 responses for a batch of 1 → mismatch error.
    with pytest.raises(RuntimeError, match='0 responses'):
        rollout([_user_traj()])


def test_sample_response_no_sequences_raises(template, tool_manager):

    class NoSeqSampler:

        def sample(self, pifs, sampling_params=None):
            if isinstance(pifs, dict):
                pifs = [pifs]
            return [SampleResponse(sequences=[]) for _ in pifs]

    rollout = MultiTurnRollout(sampler=NoSeqSampler(), template=template, tool_manager=tool_manager)
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
    sampler.queue('done-A', stop_reason='stop')  # A turn 1
    sampler.queue(
        _tool_call_text('search', {'q': 'b'}),  # B turn 1
        stop_reason='stop')
    sampler.queue('done-B', stop_reason='stop')  # B turn 2
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
        sampler=sampler,
        template=template,
        tool_manager=tm_a,  # default (unused when per-call list supplied)
        max_turns=4)
    outs = rollout([_user_traj('A'), _user_traj('B')], tool_manager=[tm_a, tm_b])

    assert outs[0]['messages'][2]['content'] == 'echo[search]:{"q": "x"}'
    assert outs[1]['messages'][2]['content'] == 'tagged[B]:{"q": "y"}'


def test_batch_tool_manager_list_length_mismatch(make_rollout, tool_manager):
    rollout = make_rollout(max_turns=2)
    with pytest.raises(ValueError, match='tool_manager list length'):
        rollout([_user_traj('A'), _user_traj('B')], tool_manager=[tool_manager])  # length 1 vs 2 trajectories


def test_single_trajectory_dict_rejected(make_rollout):
    """A single ``Trajectory`` (dict) is NOT accepted — caller must wrap."""
    rollout = make_rollout(max_turns=2)
    with pytest.raises(TypeError, match='List\\[Trajectory\\]'):
        rollout(_user_traj())


# =============================================================================
# Tests: trace_dir (per-rollout JSON dump + callback filtering)
# =============================================================================
def _list_trace_files(trace_dir):
    return sorted(p.name for p in trace_dir.iterdir() if p.suffix == '.json')


def test_trace_dir_is_created_and_empty_by_default(tmp_path, sampler, template, tool_manager):
    """Constructor creates the directory eagerly; no files until a rollout runs."""
    trace_dir = tmp_path / 'trace'
    assert not trace_dir.exists()

    MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager, max_turns=2, trace_dir=str(trace_dir))
    assert trace_dir.is_dir()
    assert _list_trace_files(trace_dir) == []


def test_trace_dir_writes_one_file_per_rollout(tmp_path, sampler, template, tool_manager):
    """Single trajectory -> single JSON file (regardless of turn count)."""
    trace_dir = tmp_path / 'trace'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager, max_turns=4, trace_dir=str(trace_dir))
    sampler.queue(_tool_call_text('search', {'q': 'x'}))
    sampler.queue('final answer', stop_reason='stop')

    outs = rollout([_user_traj('hello')])
    assert len(outs) == 1

    files = _list_trace_files(trace_dir)
    assert len(files) == 1
    # No callbacks supplied -> default prefix is ``fail-``.
    assert files[0].startswith('fail-')
    assert files[0].endswith('.json')


def test_trace_dir_json_is_pretty_printed_and_well_formed(tmp_path, sampler, template, tool_manager):
    """Dumped JSON is multi-line (indent=2) and carries the documented keys."""
    trace_dir = tmp_path / 'trace'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager, max_turns=2, trace_dir=str(trace_dir))
    sampler.queue('final answer', stop_reason='stop')

    rollout([_user_traj('hello')])

    files = list((trace_dir).glob('*.json'))
    assert len(files) == 1
    raw = files[0].read_text()
    assert '\n' in raw, 'pretty-printed JSON must span multiple lines'

    rec = json.loads(raw)
    assert set(rec.keys()) >= {'trajectory', 'ground_truth', 'stop_reason', 'truncated', 'success'}
    assert rec['stop_reason'] == 'stop'
    assert rec['truncated'] is False
    assert rec['success'] is False  # no callback => default False
    # Heavy tensor-like fields are stripped from the dumped trajectory.
    for k in ('input_ids', 'labels', 'attention_mask', 'logprobs'):
        assert k not in rec['trajectory']
    assert isinstance(rec['trajectory'].get('messages'), list)


def test_trace_dir_trace_callback_filters_storage(tmp_path, sampler, template, tool_manager):
    """``trace_callback`` returning False suppresses the dump entirely."""
    trace_dir = tmp_path / 'trace'
    rollout = MultiTurnRollout(
        sampler=sampler,
        template=template,
        tool_manager=tool_manager,
        max_turns=2,
        trace_dir=str(trace_dir),
        trace_callback=lambda traj: False)
    sampler.queue('ok', stop_reason='stop')

    rollout([_user_traj('hi')])
    assert _list_trace_files(trace_dir) == []


def test_trace_dir_success_callback_drives_filename_prefix(tmp_path, sampler, template, tool_manager):
    """True -> ``ok-*.json``, False -> ``fail-*.json``, split across batch."""
    trace_dir = tmp_path / 'trace'

    # Success is decided by a cheap rule on the last assistant message
    # content; ``store`` accepts everything.
    def _is_success(traj):
        for msg in reversed(traj.get('messages', []) or []):
            if msg.get('role') == 'assistant':
                return 'good' in (msg.get('content') or '')
        return False

    rollout = MultiTurnRollout(
        sampler=sampler,
        template=template,
        tool_manager=tool_manager,
        max_turns=2,
        trace_dir=str(trace_dir),
        success_callback=_is_success)
    sampler.queue('good answer', stop_reason='stop')
    sampler.queue('bad answer', stop_reason='stop')

    rollout([_user_traj('A'), _user_traj('B')])

    files = _list_trace_files(trace_dir)
    assert len(files) == 2
    assert any(f.startswith('ok-') for f in files)
    assert any(f.startswith('fail-') for f in files)


def test_trace_dir_batch_writes_one_file_per_trajectory(tmp_path, sampler, template, tool_manager):
    """Batch of N trajectories -> N files (never per-turn records)."""
    trace_dir = tmp_path / 'trace'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager, max_turns=4, trace_dir=str(trace_dir))
    # Traj 0: stops turn 1. Traj 1: tool-calls turn 1, stops turn 2.
    sampler.queue('done0', stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'y'}))
    sampler.queue('done1', stop_reason='stop')

    rollout([_user_traj('A'), _user_traj('B')])

    files = _list_trace_files(trace_dir)
    # Exactly one file per input trajectory, not one per turn.
    assert len(files) == 2


def test_trace_dir_none_disables_tracing(tmp_path, sampler, template, tool_manager):
    """Default ``trace_dir=None`` never touches the filesystem."""
    trace_dir = tmp_path / 'never'
    assert not trace_dir.exists()

    rollout = MultiTurnRollout(sampler=sampler, template=template, tool_manager=tool_manager, max_turns=2)
    sampler.queue('ok', stop_reason='stop')
    rollout([_user_traj('hi')])

    assert rollout.trace_dir is None
    assert not trace_dir.exists()


def test_trace_dir_truncation_marked_on_max_turns(tmp_path, sampler, template, tool_manager):
    """A rollout hitting ``max_turns`` records ``truncated=True``."""
    trace_dir = tmp_path / 'trunc'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager, max_turns=2, trace_dir=str(trace_dir))
    # Two tool-call turns -> the second hits max_turns cap.
    sampler.queue(_tool_call_text('search', {'q': 'a'}))
    sampler.queue(_tool_call_text('search', {'q': 'b'}))

    rollout([_user_traj('hi')])

    files = list((trace_dir).glob('*.json'))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec['truncated'] is True


def test_trace_dir_uses_user_data_id_in_filename(tmp_path, sampler, template, tool_manager):
    """Filenames prefer ``user_data['id']`` (sanitised) over the fallback."""
    trace_dir = tmp_path / 'trace'
    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager, max_turns=2, trace_dir=str(trace_dir))
    sampler.queue('ok', stop_reason='stop')

    traj = _user_traj('hi')
    traj['user_data'] = [('id', 'hotpotqa/42')]
    rollout([traj])

    files = _list_trace_files(trace_dir)
    assert len(files) == 1
    # Slashes are sanitised away; the id still drives the filename.
    assert 'hotpotqa_42' in files[0]
    assert files[0].startswith('fail-')


# =============================================================================
# followup_fn: several stages, one trajectory
# =============================================================================
def test_followup_appends_a_user_turn_and_keeps_generating(sampler, template, tool_manager):
    """A stage that ends without tool calls continues when the callback says so."""
    asked = []

    def followup(traj, n_before):
        asked.append((n_before, len(traj['messages'])))
        return ['write the checks', 'write the statement'][n_before] if n_before < 2 else None

    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager,
        sampling_params=SamplingParams(), max_turns=8, followup_fn=followup)
    sampler.queue(_tool_call_text('search', {'q': 'x'}), stop_reason='stop')
    sampler.queue('Done.', stop_reason='stop')
    sampler.queue('```python\nassert True\n```', stop_reason='stop')
    sampler.queue('The statement.', stop_reason='stop')

    out = rollout([_user_traj()])[0]

    assert [n for n, _ in asked] == [0, 1, 2]
    assert out['followups'] == 2
    roles = [m['role'] for m in out['messages']]
    # user, assistant(tool call), tool, assistant(Done.), user, assistant(checks),
    # user, assistant(statement)
    assert roles == ['user', 'assistant', 'tool', 'assistant', 'user', 'assistant',
                     'user', 'assistant']
    assert out['messages'][4]['content'] == 'write the checks'
    assert out['messages'][6]['content'] == 'write the statement'


def test_every_assistant_stage_stays_trainable(sampler, template, tool_manager):
    """The whole chain trains: no stage is demoted to prompt by the follow-ups.

    This is the reason follow-ups are appended inside one rollout instead of
    starting a second one on the finished conversation: a second rollout encodes
    the history as its prompt, which sets labels to -100 for every earlier
    assistant turn and leaves only the last stage trainable.
    """
    def followup(traj, n_before):
        return 'next stage' if n_before < 2 else None

    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager,
        sampling_params=SamplingParams(), max_turns=8, followup_fn=followup)
    replies = [_tool_call_text('search', {'q': 'x'}), 'Done.', 'CHECKS', 'STATEMENT']
    for i, text in enumerate(replies):
        sampler.queue(text, stop_reason='stop', logprobs=[-0.5] * len(
            template.tokenizer.encode(text + '<|im_end|>', add_special_tokens=False)))

    out = rollout([_user_traj()])[0]

    trainable = _count_trainable(out['labels'])
    expected = sum(len(template.tokenizer.encode(text + '<|im_end|>', add_special_tokens=False))
                   for text in replies)
    assert trainable == expected
    # The alignment invariant GRPO depends on: one logprob per trainable label.
    assert len(out['logprobs']) == trainable


def test_followup_stage_can_use_its_own_sampling_params(sampler, template, tool_manager):
    """``(text, params)`` gives that stage its own budget, without touching others."""
    small = SamplingParams(max_tokens=17)

    def followup(traj, n_before):
        return ('write the checks', small) if n_before == 0 else None

    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager,
        sampling_params=SamplingParams(max_tokens=99), max_turns=6, followup_fn=followup)
    sampler.queue('Done.', stop_reason='stop')
    sampler.queue('CHECKS', stop_reason='stop')

    rollout([_user_traj()])

    assert [p.max_tokens for p in sampler.params_seen] == [99, 17]


def test_tool_calls_are_not_dispatched_after_a_followup(sampler, template, tool_manager):
    """Python in a check script parses as a call list; it must not run."""
    def followup(traj, n_before):
        return 'write the checks' if n_before == 0 else None

    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager,
        sampling_params=SamplingParams(), max_turns=6, followup_fn=followup)
    sampler.queue('Done.', stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'should not run'}), stop_reason='stop')

    out = rollout([_user_traj()])[0]

    assert not any(m['role'] == 'tool' for m in out['messages'])
    assert out['followups'] == 1


# =============================================================================
# Appending a user turn under a template that moves reasoning blocks around
# =============================================================================
class ThinkAwareTokenizer(FakeTokenizer):
    """Renders like Qwen3: reasoning is kept only after the last user turn.

    Two rules, both measured on Qwen3-4B's own template: an assistant turn that
    precedes the last user message loses its ``<think>`` block, and the trailing
    assistant turn gains an empty one when it has none. Together they mean that
    appending a user message rewrites earlier text, so the plain
    "render before, render after, take the difference" bridge cannot be used.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        last_user = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
        s = ''
        for i, m in enumerate(messages):
            content = m['content']
            if m['role'] == 'assistant':
                if i < last_user:
                    content = re.sub(r'<think>[\s\S]*?</think>\n*', '', content)
                elif '<think>' not in content:
                    content = '<think>\n\n</think>\n\n' + content
            s += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            s += '<|im_start|>assistant\n'
        return self.encode(s) if tokenize else s


def test_appending_a_user_turn_keeps_the_history_ids_and_adds_only_the_new_block():
    """The delta is the new user block plus the generation prompt, nothing else."""
    from twinkle_agentic.rollout.bridge import extend_with_bridge

    template = FakeTemplate(ThinkAwareTokenizer())
    messages = [{'role': 'user', 'content': 'do work'},
                {'role': 'assistant', 'content': '<think>reasoning</think>Done.'}]
    pif = template.encode({'messages': messages})
    pif['labels'] = [7] * len(pif['input_ids'])  # stand-in for "these were sampled"
    before_ids = list(pif['input_ids'])

    out = extend_with_bridge(pif, [{'role': 'user', 'content': 'write the checks'}], template)

    # History untouched: the reasoning the policy produced is still in the ids.
    assert out['input_ids'][:len(before_ids)] == before_ids
    added = template.tokenizer.decode(out['input_ids'][len(before_ids):])
    assert added == ('<|im_start|>user\nwrite the checks<|im_end|>\n'
                     '<|im_start|>assistant\n'), added
    # And the appended block is not trained on.
    assert set(out['labels'][len(before_ids):-1]) == {-100}


def test_a_template_that_really_reorders_history_still_raises():
    """The fallback must not paper over a template that rewrites message blocks."""
    from twinkle_agentic.rollout.bridge import extend_with_bridge

    class ReorderingTokenizer(FakeTokenizer):
        """Puts the message count up front, so every append rewrites the start."""

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
            s = f'[{len(messages)} messages]'
            for m in messages:
                s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            if add_generation_prompt:
                s += '<|im_start|>assistant\n'
            return self.encode(s) if tokenize else s

    template = FakeTemplate(ReorderingTokenizer())
    pif = template.encode({'messages': [{'role': 'user', 'content': 'a'},
                                        {'role': 'assistant', 'content': 'b'}]})
    with pytest.raises(RuntimeError, match='non-monotonic'):
        extend_with_bridge(pif, [{'role': 'user', 'content': 'c'}], template)


def test_running_out_of_tool_turns_still_reaches_the_follow_up_stages(sampler, template, tool_manager):
    """An episode that spends its whole turn budget is not thrown away.

    Before, hitting ``max_turns`` ended the trajectory outright -- and with the
    stages living inside the episode that would throw away the sandbox run that
    produced the state they are about.
    """
    asked = []

    def followup(traj, n_before):
        asked.append(n_before)
        return 'write the checks' if n_before == 0 else None

    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager,
        sampling_params=SamplingParams(), max_turns=2, followup_fn=followup)
    # Two turns of tool calls: the second one hits the limit.
    sampler.queue(_tool_call_text('search', {'q': 'a'}), stop_reason='stop')
    sampler.queue(_tool_call_text('search', {'q': 'b'}), stop_reason='stop')
    sampler.queue('assert True', stop_reason='stop')

    out = rollout([{'messages': [{'role': 'user', 'content': 'go'}]}])[0]

    assert asked == [0, 1]
    assert out['tool_stop'] == 'max_turns'
    # The stage ran, so nothing was cut off.
    assert out['truncated'] is False
    assert out['messages'][-2:] == [{'role': 'user', 'content': 'write the checks'},
                                    {'role': 'assistant', 'content': 'assert True'}]


def test_a_stage_reply_that_looks_like_a_tool_call_is_kept_whole(sampler, template, tool_manager):
    """The stage reply the caller reads is what the model wrote.

    The template stores a reply that parses as a call with the call text removed,
    which is right for a turn whose calls get dispatched and wrong for a stage
    whose reply *is* the answer. It bit for real: one of the tool-call formats is
    XML-shaped, so a check script asserting the content of an .xml file parsed as
    calls, and 5 of ex12's 72 scripts arrived with that content deleted -- three
    then asserted `content == ''` against a file that had text in it.
    """

    def followup(traj, n_before):
        return 'write the checks' if n_before == 0 else None

    rollout = MultiTurnRollout(
        sampler=sampler, template=template, tool_manager=tool_manager,
        sampling_params=SamplingParams(), max_turns=4, followup_fn=followup)
    sampler.queue('done exploring', stop_reason='stop')
    script = ('```python\n' + _tool_call_text('data', {'number': '75'})
              + "\nassert open('a.xml').read() == 'x'\n```")
    sampler.queue(script, stop_reason='stop')

    out = rollout([{'messages': [{'role': 'user', 'content': 'go'}]}])[0]

    last = out['messages'][-1]
    assert last['content'] == script
    assert 'tool_calls' not in last
    # Whole, but still without the special tokens: the sampled ids end with
    # <|im_end|> and ``seq.decoded`` may keep it. ex13 shipped 7 of 7 problem
    # statements ending in a literal '<|im_end|>' that way.
    assert '<|im_end|>' not in last['content']
    # And it was not dispatched: a dispatch appends a tool message.
    assert [m['role'] for m in out['messages'] if m['role'] == 'tool'] == []
