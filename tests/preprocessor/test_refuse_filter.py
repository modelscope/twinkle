# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for RefuseFilter.

RefuseFilter drops rows whose first assistant reply contains a self-referential
refusal signal across en/zh/ja/ko.  Key invariants:

  * Detection is anchored to ``I/we`` + inability + task verb (en) and 我 +
    无法/不能 + 帮/回答/... (zh).  Generic uses of ``cannot`` / ``无法`` that are
    NOT self-referential must NOT trip the filter.
  * Only the FIRST assistant turn is scanned, within ``check_window`` chars.
  * ``<think>...</think>`` blocks are stripped before scanning.
  * Empty / missing assistant content is kept (filter is conservative).
"""
import pytest

from twinkle_agentic.preprocessor.refuse_filter import RefuseFilter, _is_refusal


def _row(messages):
    return {'messages': messages}


def _fil(rows, **kw):
    return RefuseFilter(**kw)(rows)


# ── _is_refusal: English ────────────────────────────────────────────────────


class TestRefusalEnglish:

    def test_core_self_referential(self):
        assert _is_refusal("I can't help with that request.") is True
        assert _is_refusal('I cannot assist with creating malware.') is True
        assert _is_refusal('I am not able to provide that information.') is True
        assert _is_refusal("We won't generate that content.") is True

    def test_apology_opener(self):
        assert _is_refusal("I'm sorry, but I can't help with that.") is True
        assert _is_refusal('Unfortunately, I am unable to comply.') is True
        assert _is_refusal('I apologize, but I cannot answer that.') is True

    def test_policy_violation(self):
        assert _is_refusal('This request violates my guidelines.') is True
        assert _is_refusal('Your question goes against safety policies.') is True

    def test_standalone_decline(self):
        assert _is_refusal('I must decline this request.') is True
        assert _is_refusal('I refuse to participate in that.') is True
        assert _is_refusal("As an AI, I can't help with illegal activities.") is True

    def test_false_positive_cannot_stress(self):
        # "I cannot stress enough" — has I+cannot but NOT a task verb in window
        text = 'I cannot stress enough how important documentation is.'
        assert _is_refusal(text) is False

    def test_false_positive_cannot_find(self):
        # "I cannot find the bug" — find is not in task-verb whitelist
        text = 'I cannot find the bug after looking everywhere.'
        assert _is_refusal(text) is False

    def test_false_positive_third_person(self):
        # No I/we subject → not refusal
        assert _is_refusal('The API cannot handle null inputs.') is False
        assert _is_refusal('You cannot use this without auth.') is False

    def test_case_insensitive(self):
        assert _is_refusal('I CANNOT HELP with that.') is True
        assert _is_refusal("i can't assist you here") is True


# ── _is_refusal: Chinese ────────────────────────────────────────────────────


class TestRefusalChinese:

    def test_apology_inability(self):
        assert _is_refusal('非常抱歉，我无法回答这个问题。') is True
        assert _is_refusal('对不起，我不能提供这类信息。') is True
        assert _is_refusal('抱歉，没有办法帮助您完成这个请求。') is True

    def test_self_referential(self):
        assert _is_refusal('我无法帮助你完成这项任务。') is True
        assert _is_refusal('我不能提供医疗建议。') is True
        assert _is_refusal('我不被允许回答此类问题。') is True

    def test_request_violation(self):
        assert _is_refusal('您的请求违反了使用规则。') is True
        assert _is_refusal('这个问题不合适讨论。') is True

    def test_ai_identity_refusal(self):
        assert _is_refusal('作为AI，我无法回答涉及隐私的问题。') is True
        assert _is_refusal('作为人工智能，我不能提供该类内容。') is True

    def test_false_positive_self_deprecating_preamble(self):
        # "作为AI，我虽无法体验情感，但..." — 无法 not paired with task verb
        text = '作为AI，我虽无法亲身体验情感，但可以分析其特征。'
        assert _is_refusal(text) is False

    def test_false_positive_third_person(self):
        # No 我 subject + task verb pattern
        assert _is_refusal('该接口无法处理空输入。') is False
        # "无法理解" with no 我 / task verb pattern
        assert _is_refusal('系统暂时不能正常运作。') is False


# ── _is_refusal: Japanese ───────────────────────────────────────────────────


class TestRefusalJapanese:

    def test_apology_inability(self):
        # Pattern is `(申し訳|恐れ入り)ます(が|けれど)` — only 恐れ入りますが
        # is grammatically natural.
        assert _is_refusal('恐れ入りますが、お答えできません。') is True
        assert _is_refusal('恐れ入りますけれど、対応できません。') is True

    def test_explicit_inability(self):
        assert _is_refusal('回答することはできません。') is True
        assert _is_refusal('ご要望にはお応えできません。') is True

    def test_request_decline(self):
        # Matches `(その|この)(リクエスト|質問|依頼).{0,20}(お断り|辞退|対応できません)`
        assert _is_refusal('そのリクエストはお断りします。') is True
        assert _is_refusal('このリクエストには対応できません。') is True


# ── _is_refusal: Korean ─────────────────────────────────────────────────────


class TestRefusalKorean:

    def test_apology_inability(self):
        assert _is_refusal('죄송하지만 답변을 드릴 수 없습니다.') is True
        assert _is_refusal('유감스럽게도 도와드릴 수 없습니다.') is True

    def test_action_difficulty(self):
        assert _is_refusal('답변드리기 어렵습니다.') is True
        assert _is_refusal('처리하기 불가능합니다.') is True


# ── check_window ────────────────────────────────────────────────────────────


class TestCheckWindow:

    def test_window_excludes_late_refusal(self):
        # Refusal at position 700 — beyond default 600-char window
        text = 'a' * 700 + " I can't help you complete that task."
        assert _is_refusal(text, check_window=600) is False

    def test_custom_window_includes_late_refusal(self):
        text = 'a' * 700 + " I can't help you complete that task."
        assert _is_refusal(text, check_window=1000) is True

    def test_zero_window_finds_nothing(self):
        assert _is_refusal("I can't help you complete tasks.", check_window=0) is False


# ── RefuseFilter pipeline ───────────────────────────────────────────────────


class TestRefuseFilterPipeline:

    def test_drops_refusal_row(self):
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'do bad thing'
                },
                {
                    'role': 'assistant',
                    'content': "I'm sorry, but I cannot help with that request."
                },
            ])
        ]
        assert _fil(rows) == []

    def test_keeps_normal_reply(self):
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'explain X'
                },
                {
                    'role': 'assistant',
                    'content': 'X is a concept that...'
                },
            ])
        ]
        assert len(_fil(rows)) == 1

    def test_only_first_assistant_scanned(self):
        # Refusal in SECOND assistant turn → kept (filter only checks first).
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q1'
                },
                {
                    'role': 'assistant',
                    'content': 'A clean reply.'
                },
                {
                    'role': 'user',
                    'content': 'q2'
                },
                {
                    'role': 'assistant',
                    'content': "I can't help with that."
                },
            ])
        ]
        assert len(_fil(rows)) == 1

    def test_think_block_stripped(self):
        # Refusal phrasing inside <think>...</think> must NOT trigger.
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q'
                },
                {
                    'role': 'assistant',
                    'content': '<think>I cannot help with this request</think>'
                    'Sure, here is the answer: 42.'
                },
            ])
        ]
        assert len(_fil(rows)) == 1

    def test_no_assistant_kept(self):
        rows = [_row([{'role': 'user', 'content': 'hi'}])]
        assert len(_fil(rows)) == 1

    def test_empty_assistant_kept(self):
        rows = [_row([
            {
                'role': 'user',
                'content': 'q'
            },
            {
                'role': 'assistant',
                'content': ''
            },
        ])]
        assert len(_fil(rows)) == 1

    def test_empty_input(self):
        assert _fil([]) == []

    def test_missing_messages_kept(self):
        # No messages key → no assistant → kept
        rows = [{'id': 'x'}]
        assert len(_fil(rows)) == 1

    def test_mixed_batch(self):
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q1'
                },
                {
                    'role': 'assistant',
                    'content': 'a normal answer'
                },
            ]),
            _row([
                {
                    'role': 'user',
                    'content': 'q2'
                },
                {
                    'role': 'assistant',
                    'content': 'I refuse to help you with that task.'
                },
            ]),
            _row([
                {
                    'role': 'user',
                    'content': 'q3'
                },
                {
                    'role': 'assistant',
                    'content': '抱歉，我无法回答这个问题。'
                },
            ]),
        ]
        out = _fil(rows)
        assert len(out) == 1
        assert out[0]['messages'][0]['content'] == 'q1'

    def test_custom_check_window(self):
        # Default 600 would miss a late refusal; tighten via pipeline kw.
        long_prefix = 'a' * 700
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q'
                },
                {
                    'role': 'assistant',
                    'content': long_prefix + " I can't help you complete that."
                },
            ])
        ]
        # default window → kept
        assert len(_fil(rows)) == 1
        # widen → dropped
        assert _fil(rows, check_window=1000) == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
