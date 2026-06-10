# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the heuristic IntentClassifier pipeline.

Focus areas:
- Per-detector recall on representative samples (ZH + EN, R1-distill-flavoured).
- Per-detector FP guards (chitchat, role mismatch, first-turn dissatisfaction).
- Multi-detector ordering: ToolCallDetector short-circuit, ``setdefault`` semantics.
- Edge cases: empty / None / non-dict / list-content messages, empty trajectories.
- Public API contract: ``row['intent']``, user_data List[Tuple] entries ``('key_rounds', ...)`` / ``('intents', ...)``.
- Detector pluggability: custom subclass, overriding ``DEFAULT_DETECTORS``.
"""
import pytest

from twinkle_agentic.preprocessor.intent_classifier import (INTENT_CODE, INTENT_MATH, INTENT_OTHER, INTENT_TOOL_CALL,
                                                            INTENT_USER_DISSATISFACTION, CodeDetector, IntentClassifier,
                                                            IntentDetector, MathDetector, ToolCallDetector,
                                                            UserDissatisfactionDetector, _msg_text, _pair_assistant)
from twinkle.data_format import pack_value, user_data_get

# ── Helpers ───────────────────────────────────────────────────────────────────


def _u(text):
    return {'role': 'user', 'content': text}


def _a(text, **extra):
    msg = {'role': 'assistant', 'content': text}
    msg.update(extra)
    return msg


def _row(*messages):
    return {'messages': list(messages)}


def _classify_one(*messages, detectors=None):
    # Keep no-key-round rows (e.g. chitchat) so tests can assert ``intent == INTENT_OTHER``.
    ic = IntentClassifier(detectors=detectors, drop_no_key_rounds=False)
    kept, _dropped = ic([_row(*messages)])
    return kept[0]


def _ud_get(row, key):
    """Read a value by key from packed user_data (JSON-decoded)."""
    return user_data_get(row.get('user_data'), key)


def _has_key_rounds(row):
    kr = _ud_get(row, 'key_rounds')
    return isinstance(kr, list) and bool(kr)


# ── Helper functions ──────────────────────────────────────────────────────────


class TestHelpers:

    def test_msg_text_string(self):
        assert _msg_text({'content': 'hi'}) == 'hi'

    def test_msg_text_list_with_text_parts(self):
        msg = {
            'content': [
                {
                    'type': 'text',
                    'text': 'foo'
                },
                {
                    'type': 'image',
                    'url': 'x'
                },
                {
                    'type': 'text',
                    'text': 'bar'
                },
            ]
        }
        assert _msg_text(msg) == 'foo bar'

    def test_msg_text_missing_content(self):
        assert _msg_text({}) == ''

    def test_msg_text_none_content(self):
        assert _msg_text({'content': None}) == ''

    def test_msg_text_list_no_text_parts(self):
        assert _msg_text({'content': [{'type': 'image'}]}) == ''

    def test_pair_assistant_user_finds_next_assistant(self):
        msgs = [_u('q'), _a('a1'), _u('follow'), _a('a2')]
        assert _pair_assistant(msgs, 0, 'user') == 1
        assert _pair_assistant(msgs, 2, 'user') == 3

    def test_pair_assistant_assistant_returns_self(self):
        msgs = [_u('q'), _a('a1')]
        assert _pair_assistant(msgs, 1, 'assistant') == 1

    def test_pair_assistant_user_no_following_assistant(self):
        # User turn at the tail with no assistant after — un-pairable.
        msgs = [_a('a1'), _u('dangling')]
        assert _pair_assistant(msgs, 1, 'user') is None

    def test_pair_assistant_other_role(self):
        assert _pair_assistant([{'role': 'system', 'content': 's'}], 0, 'system') is None


# ── ToolCallDetector ─────────────────────────────────────────────────────────


class TestToolCallDetector:

    def test_definitive_flag(self):
        assert ToolCallDetector.definitive is True

    def test_detects_assistant_with_tool_calls(self):
        msgs = [_u('q'), _a('', tool_calls=[{'name': 'f'}])]
        assert ToolCallDetector()(msgs) == [1]

    def test_ignores_assistant_without_tool_calls(self):
        assert ToolCallDetector()([_u('q'), _a('plain')]) == []

    def test_ignores_user_with_tool_calls_field(self):
        # A user dict carrying a tool_calls key must not be picked up.
        msgs = [{'role': 'user', 'content': 'q', 'tool_calls': [{'name': 'x'}]}]
        assert ToolCallDetector()(msgs) == []

    def test_short_circuits_pipeline(self):
        # When ToolCall fires it must suppress later detectors on the same round.
        msgs = [
            _u('解一元二次方程 x^2 - 5x + 6 = 0 的因式分解'),
            _a('answer', tool_calls=[{
                'name': 'calc'
            }]),
        ]
        out = _classify_one(*msgs)
        assert out['intent'] == INTENT_TOOL_CALL
        # math detector must not have written into intents.
        assert _ud_get(out, 'intents') == {'1': INTENT_TOOL_CALL}


# ── CodeDetector ──────────────────────────────────────────────────────────────


class TestCodeDetector:

    def test_fenced_code_block(self):
        text = '```python\ndef f():\n    return 1\n```'
        assert CodeDetector()._match(text)

    def test_short_fenced_block_below_min_length(self):
        # Block content must be ≥10 chars to qualify.
        assert not CodeDetector()._match('```\nhi\n```')

    def test_keyword_threshold_three(self):
        # Three keyword hits must trigger.
        assert CodeDetector()._match('use async function and await the response')

    def test_two_keywords_below_threshold(self):
        assert not CodeDetector()._match('a class and a function')

    def test_arrow_signature_alone_insufficient(self):
        # Single arrow without other signals doesn't reach threshold.
        assert not CodeDetector()._match('x => x + 1')

    def test_call_signature_with_brace(self):
        # `name(args) {` is a strong code indicator.
        assert CodeDetector()._match('function fetchData(url) { return fetch(url); } and async await yield')

    def test_chitchat_with_word_class_no_fp(self):
        assert not CodeDetector()._match('I took a yoga class today')


# ── MathDetector ──────────────────────────────────────────────────────────────


class TestMathDetector:

    @pytest.mark.parametrize('text', [
        '设 $f(x)=x^2$ 求导得 2x',
        '矩阵 A 的行列式 det(A) 不等于 0',
        '三角形 ABC 周长是 12，面积约为 6',
        '数列 {a_n} 是等差数列，公差为 2，首项为 1',
        '4, 3, 4, 3, ()，奇数位是 4',
        'Σ_{i=1}^n A_{ik} B_{kj}',
        'gradient and integral are both fundamental',
        '求一元二次方程 x^2 - 5x + 6 = 0 的解',
        '一个圆形的直径是 10cm，所以周长是 10π',
    ])
    def test_math_recall(self, text):
        assert MathDetector()._match(text), f'should detect: {text!r}'

    @pytest.mark.parametrize(
        'text',
        [
            '今天天气真好',
            '我最近在追一部电视剧',
            '帮我写一首诗',
            '请帮我翻译这句英文',
            # Single math keyword in non-math context — must not trip ≥2 threshold.
            '积分兑换可以兑换礼品',
            '矩阵这个电影很好看',
        ])
    def test_math_fp_guard(self, text):
        assert not MathDetector()._match(text), f'must NOT detect: {text!r}'

    def test_arithmetic_equation_single_hit(self):
        # Only the arithmetic equation matches, threshold ≥2 not met.
        assert not MathDetector()._match('计算 30 ÷ 6 = 5')

    def test_threshold_is_configurable(self):
        # Subclass with looser threshold catches single-hit case.
        class LooseMath(MathDetector):
            threshold = 1

        assert LooseMath()._match('计算 30 ÷ 6 = 5')

    def test_subscript_pattern(self):
        assert MathDetector()._match('矩阵元素 a_{ij} 与 b_{kl} 满足条件')


# ── UserDissatisfactionDetector ───────────────────────────────────────────────


class TestUserDissatisfactionDetector:

    @pytest.mark.parametrize('text', [
        '不对，再来一次',
        '完全错了',
        '答非所问',
        '你这是在胡扯',
        '太离谱了',
        '一塌糊涂',
        '没逻辑啊',
        '你根本没听懂我的意思',
        '我说的不是这个',
        '别瞎编',
        '什么玩意',
        '不靠谱',
        '让我失望',
        '不严谨',
        '没get到',
    ])
    def test_zh_recall(self, text):
        assert UserDissatisfactionDetector()._match(text)

    @pytest.mark.parametrize('text', [
        'this is wrong',
        'totally incorrect',
        'try again please',
        "doesn't make sense",
        'that is garbage',
        'you misunderstood me',
        'low quality response',
        'completely off topic',
        'are you serious',
        'waste of time',
        'this is bullshit',
        'redo it',
        'sub-par answer',
        'do better',
        'WTF is this',
        'nowhere near correct',
    ])
    def test_en_recall(self, text):
        assert UserDissatisfactionDetector()._match(text)

    @pytest.mark.parametrize('text', [
        '今天心情很好',
        '我喜欢这个回答',
        '请帮我修改一下',
        'this is exactly what I wanted',
        'great answer thanks',
        '能再详细一点吗',
    ])
    def test_fp_guard(self, text):
        det = UserDissatisfactionDetector()
        assert not det._match(text), f'FP on: {text!r}'

    def test_first_turn_user_complaint_ignored(self):
        # No prior assistant — the negative phrasing is part of the initial query, not a reaction.
        msgs = [_u('你这答案完全错了，太垃圾'), _a('sorry')]
        assert UserDissatisfactionDetector()(msgs) == []

    def test_system_first_then_user_complaint_ignored(self):
        msgs = [
            {
                'role': 'system',
                'content': 'You are helpful.'
            },
            _u('上次回答简直一塌糊涂'),
            _a('sorry'),
        ]
        # System turn must not satisfy "prior assistant".
        assert UserDissatisfactionDetector()(msgs) == []

    def test_multiturn_reaction_detected(self):
        msgs = [_u('解释勾股定理'), _a('a²+b²=c²'), _u('不对，再来一次'), _a('好的')]
        # The dissat user is at idx 2 → key round is the next assistant idx 3.
        assert UserDissatisfactionDetector()(msgs) == [3]

    def test_dissat_with_no_following_assistant_dropped(self):
        # User dissatisfaction at the tail with no assistant pair → unpaired, no key round.
        msgs = [_u('q'), _a('answer'), _u('完全错了')]
        assert UserDissatisfactionDetector()(msgs) == []

    def test_role_filter_blocks_assistant_self_correction(self):
        # "等等我算错了，重新推导" appearing on assistant must not be tagged dissatisfaction.
        msgs = [_u('推导一下'), _a('等等，我之前算错了，让我重新推导')]
        assert UserDissatisfactionDetector()(msgs) == []


# ── End-to-end IntentClassifier ───────────────────────────────────────────────


class TestIntentClassifierE2E:

    def test_chitchat_other(self):
        out = _classify_one(_u('今天天气真好'), _a('是的，挺适合出门的'))
        assert out['intent'] == INTENT_OTHER
        assert not _has_key_rounds(out)

    def test_math_round(self):
        out = _classify_one(
            _u('求一元二次方程 x^2 - 5x + 6 = 0 的解'),
            _a('由因式分解得 (x-2)(x-3)=0'),
        )
        assert out['intent'] == INTENT_MATH
        assert _ud_get(out, 'key_rounds') == [1]
        assert _ud_get(out, 'intents') == {'1': INTENT_MATH}

    def test_code_round(self):
        out = _classify_one(
            _u('use async function and await the response in JavaScript'),
            _a('try const fetchData = async () => { return await fetch(url); }'),
        )
        assert out['intent'] == INTENT_CODE

    def test_dissat_round(self):
        out = _classify_one(_u('q'), _a('answer'), _u('totally garbage answer, redo'), _a('sorry'))
        assert out['intent'] == INTENT_USER_DISSATISFACTION
        assert _ud_get(out, 'key_rounds') == [3]

    def test_assistant_self_correction_not_dissat(self):
        # Root cause for original FP: role-agnostic regex on assistant text. Must stay fixed.
        out = _classify_one(_u('推导一下'), _a('等等，我之前算错了，让我重新推导...'))
        assert out['intent'] == INTENT_OTHER

    def test_first_turn_user_negative_words_not_dissat(self):
        out = _classify_one(_u('你这答案完全错了，太垃圾'), _a('抱歉'))
        assert out['intent'] == INTENT_OTHER

    def test_setdefault_earlier_detector_wins(self):
        # When a round is first claimed by MathDetector, a later UserDissatisfactionDetector
        # touching the same round must not overwrite it.
        out = _classify_one(
            _u('解一元二次方程 x^2 - 5x + 6 = 0'),
            _a('factoring: (x-2)(x-3)'),
            _u('不对，再来一次'),
            _a('好的'),
        )
        intents = _ud_get(out, 'intents')
        assert intents['1'] == INTENT_MATH
        assert intents['3'] == INTENT_USER_DISSATISFACTION

    def test_tool_call_definitive_short_circuits(self):
        out = _classify_one(
            _u('解一元二次方程 x^2 - 5x + 6 = 0'),
            _a('', tool_calls=[{
                'name': 'calc'
            }]),
        )
        assert out['intent'] == INTENT_TOOL_CALL
        # MathDetector must not have run after the definitive ToolCallDetector.
        assert set(_ud_get(out, 'intents').values()) == {INTENT_TOOL_CALL}

    def test_multimodal_list_content(self):
        # List-content messages must work transparently.
        msgs = [
            _u([{
                'type': 'text',
                'text': '求一元二次方程'
            }, {
                'type': 'image',
                'url': 'x'
            }]),
            _a([{
                'type': 'text',
                'text': '因式分解后得到结果'
            }]),
        ]
        out = _classify_one(*msgs)
        assert out['intent'] == INTENT_MATH


# ── Edge / robustness ─────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_rows(self):
        kept, _ = IntentClassifier()([])
        assert kept == []

    def test_missing_messages_field(self):
        kept, _ = IntentClassifier(drop_no_key_rounds=False)([{'foo': 'bar'}])
        assert kept[0]['intent'] == INTENT_OTHER

    def test_messages_is_none(self):
        kept, _ = IntentClassifier(drop_no_key_rounds=False)([{'messages': None}])
        assert kept[0]['intent'] == INTENT_OTHER

    def test_messages_empty_list(self):
        kept, _ = IntentClassifier(drop_no_key_rounds=False)([{'messages': []}])
        assert kept[0]['intent'] == INTENT_OTHER

    def test_messages_with_non_dict_entries(self):
        # Non-dict entries must be silently skipped.
        kept, _ = IntentClassifier()([{
            'messages': [
                'not a dict',
                None,
                _u('求一元二次方程'),
                _a('因式分解'),
            ]
        }])
        assert kept[0]['intent'] == INTENT_MATH

    def test_user_data_preexists_preserved(self):
        # IntentClassifier must merge into existing packed user_data without clobbering.
        rows = [{
            'messages': [_u('解一元二次方程 x^2'), _a('因式分解 (x-2)(x-3)')],
            'user_data': [('source', pack_value('gsm8k')), ('difficulty', pack_value('easy'))],
        }]
        kept, _ = IntentClassifier()(rows)
        row = kept[0]
        assert _ud_get(row, 'source') == 'gsm8k'
        assert _ud_get(row, 'difficulty') == 'easy'
        assert _ud_get(row, 'key_rounds') == [1]
        assert _ud_get(row, 'intents') == {'1': INTENT_MATH}

    def test_input_row_not_mutated(self):
        # IntentClassifier must shallow-copy rows; original dict must remain untouched.
        original = {'messages': [_u('你好'), _a('hi')]}
        IntentClassifier(drop_no_key_rounds=False)([original])
        assert 'intent' not in original
        assert 'user_data' not in original

    def test_other_intent_does_not_emit_user_data(self):
        out = _classify_one(_u('你好'), _a('hi'))
        # No detectors fired → no key_rounds / intents written.
        assert not _has_key_rounds(out)


# ── Pluggability ──────────────────────────────────────────────────────────────


class TestPluggability:

    def test_custom_detector_via_constructor(self):

        class GreetingDetector(IntentDetector):
            intent = 'greeting'

            def __call__(self, messages):
                return [
                    i for i, m in enumerate(messages) if isinstance(m, dict) and m.get('role') == 'assistant'
                    and isinstance(m.get('content'), str) and 'hello' in m['content'].lower()
                ]

        ic = IntentClassifier(detectors=[GreetingDetector()])
        kept, _ = ic([_row(_u('hi'), _a('Hello there'))])
        assert kept[0]['intent'] == 'greeting'

    def test_empty_detector_list_yields_other(self):
        ic = IntentClassifier(detectors=[], drop_no_key_rounds=False)
        kept, _ = ic([_row(_u('q'), _a('因式分解 一元二次方程'))])
        assert kept[0]['intent'] == INTENT_OTHER

    def test_intent_field_override(self):
        ic = IntentClassifier(intent_field='label', drop_no_key_rounds=False)
        kept, _ = ic([_row(_u('q'), _a('a'))])
        assert 'label' in kept[0]
        assert 'intent' not in kept[0]

    def test_definitive_short_circuits_custom_pipeline(self):
        # User-defined definitive detector must halt the pipeline after firing.
        seen = []

        class StopAll(IntentDetector):
            intent = 'stop'
            definitive = True

            def __call__(self, messages):
                seen.append('stop')
                return [len(messages) - 1]

        class NeverRuns(IntentDetector):
            intent = 'never'

            def __call__(self, messages):
                seen.append('never')
                return [0]

        ic = IntentClassifier(detectors=[StopAll(), NeverRuns()])
        ic([_row(_u('q'), _a('a'))])
        assert seen == ['stop']
