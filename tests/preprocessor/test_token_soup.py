# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for TokenSoupFilter.

Covers each garbled-output signal in ``_is_token_soup`` plus the
script-chaos analyzer and the row-filter pipeline.
"""
import pytest

from twinkle_agentic.preprocessor.token_soup import TokenSoupFilter, _is_token_soup, _script_chaos, _script_of


def _row(content):
    return {
        'messages': [
            {
                'role': 'user',
                'content': 'q'
            },
            {
                'role': 'assistant',
                'content': content
            },
        ]
    }


# ── Per-signal detector tests ────────────────────────────────────────────────


class TestReplacementChar:

    def test_above_threshold(self):
        text = '\ufffd' * 5 + 'short'  # 5/10 = 50% > 2%
        assert _is_token_soup(text) is True

    def test_below_threshold(self):
        text = '\ufffd' + 'hello world this is text. ' * 30  # 1/~780 ≈ 0.1% < 2%
        # No other signal should fire
        assert _is_token_soup(text) is False

    def test_no_replacement_char(self):
        assert _is_token_soup('hello world') is False


class TestControlChar:

    def test_above_threshold(self):
        text = '\x01\x02\x03\x04\x05' + 'a' * 100  # 5/105 ≈ 4.8% > 1%
        assert _is_token_soup(text) is True

    def test_keeps_legitimate_whitespace(self):
        text = 'line1\nline2\tindented\rcr'
        assert _is_token_soup(text) is False

    def test_del_char_triggers(self):
        text = '\x7f' * 5 + 'a' * 100
        assert _is_token_soup(text) is True


class TestPrivateUseArea:

    def test_bmp_pua_above_threshold(self):
        text = '\ue000\ue001\ue002\ue003\ue004' + 'a' * 100  # 5/105 ≈ 4.8% > 3%
        assert _is_token_soup(text) is True

    def test_below_threshold(self):
        text = '\ue000' + 'hello world this is text. ' * 30  # ~0.1% < 3%
        assert _is_token_soup(text) is False


class TestSpecialTokens:

    def test_repeated_pipe_token(self):
        text = '<|endoftext|>' * 25
        assert _is_token_soup(text, special_token_count=20) is True

    def test_repeated_bert_uppercase(self):
        text = '[PAD]' * 25
        assert _is_token_soup(text, special_token_count=20) is True

    def test_lowercase_brackets_not_matched(self):
        # ``dp[mask]`` is normal code; lowercase variant must NOT match.
        text = 'arr[mask] = arr[mask] | 1; ' * 30
        assert _is_token_soup(text, special_token_count=20) is False

    def test_byte_token_form(self):
        text = '<0x0A>' * 25
        assert _is_token_soup(text, special_token_count=20) is True

    def test_below_count(self):
        text = '<|endoftext|>' * 5
        assert _is_token_soup(text, special_token_count=20) is False

    def test_unk_pad_html_tags(self):
        text = '<unk>' * 12 + '</unk>' * 13
        assert _is_token_soup(text, special_token_count=20) is True


class TestSingleCharRepeat:

    def test_letter_repeat_triggers(self):
        text = 'aaaaaaaaaaaaaaaaaaaaaaaaaa hello world'  # 26 a's > 19
        assert _is_token_soup(text) is True

    def test_dash_excluded(self):
        text = '-' * 50 + ' separator'
        assert _is_token_soup(text) is False

    def test_equals_excluded(self):
        text = '=' * 50
        assert _is_token_soup(text) is False

    def test_digit_excluded(self):
        text = '9' * 50
        assert _is_token_soup(text) is False

    def test_box_drawing_excluded(self):
        text = '\u2500' * 50  # ─ box-drawing horizontal
        assert _is_token_soup(text) is False

    def test_below_threshold(self):
        text = 'a' * 19  # 19 < 20 (regex requires \1{19,} → 1 + 19 = 20)
        assert _is_token_soup(text) is False

    def test_at_threshold(self):
        text = 'a' * 20  # 20 a's: 1 + 19 repeats → matches
        assert _is_token_soup(text) is True


# ── Script-chaos analyzer ────────────────────────────────────────────────────


class TestScriptOf:

    def test_latin(self):
        assert _script_of(ord('A')) == 'latin'
        assert _script_of(ord('z')) == 'latin'

    def test_cjk(self):
        assert _script_of(ord('中')) == 'cjk'

    def test_hiragana_katakana(self):
        assert _script_of(0x3042) == 'hiragana'  # あ
        assert _script_of(0x30A2) == 'katakana'  # ア

    def test_cyrillic(self):
        assert _script_of(0x0410) == 'cyrillic'

    def test_hangul(self):
        assert _script_of(0xAC00) == 'hangul'

    def test_private(self):
        assert _script_of(0xE000) == 'private'

    def test_other(self):
        assert _script_of(0x2000) == 'other'  # general punctuation


class TestScriptChaos:

    def test_pure_latin_zero_chaos(self):
        assert _script_chaos('hello world this is a long english sentence') == 0.0

    def test_pure_cjk_zero_chaos(self):
        assert _script_chaos('这是一段足够长的中文文本用于测试脚本切换检测' * 2) == 0.0

    def test_short_text_returns_zero(self):
        # Below ``min_chars`` → returns 0.0 regardless of mix.
        assert _script_chaos('aあ', min_chars=40) == 0.0

    def test_high_chaos_alternation(self):
        # Pure letter/number alternation between scripts → chaos ≈ 1.0.
        text = ('aあbいcうdえeお' * 5)  # 50 alternating letters
        score = _script_chaos(text, min_chars=40)
        assert score > 0.9

    def test_filter_with_chaos(self):
        text = ('aあbいcうdえeお' * 5)  # high chaos
        assert _is_token_soup(text, script_chaos_min_chars=40, script_chaos_threshold=0.55) is True

    def test_skips_punct_whitespace(self):
        # Categories not in (L, N) are dropped before script-of pairing.
        text = 'hello, world! how are you?'
        assert _script_chaos(text) == 0.0


# ── max_chars head-sampling ──────────────────────────────────────────────────


class TestMaxChars:

    def test_only_head_examined(self):
        # Soup at the tail; head is clean. With max_chars=100 we should not see it.
        head = 'hello world this is plain text. ' * 4  # ~128 chars, no repeat-20
        text = head[:100] + '\ufffd' * 100
        assert _is_token_soup(text, max_chars=100, replacement_char_ratio=0.02) is False

    def test_full_text_when_max_chars_zero(self):
        head = 'hello world this is plain text. ' * 4
        text = head[:100] + '\ufffd' * 100
        assert _is_token_soup(text, max_chars=0, replacement_char_ratio=0.02) is True


# ── Empty / trivial inputs ───────────────────────────────────────────────────


class TestTrivial:

    def test_empty_text(self):
        assert _is_token_soup('') is False

    def test_short_clean_text(self):
        assert _is_token_soup('Hi there!') is False


# ── Pipeline ─────────────────────────────────────────────────────────────────


class TestTokenSoupFilterPipeline:

    def test_drops_soupy_assistant(self):
        f = TokenSoupFilter()
        rows = [_row('clean response'), _row('aaaaaaaaaaaaaaaaaaaaaaaaaaaaa')]
        out = f(rows)
        assert len(out) == 1
        assert out[0]['messages'][1]['content'] == 'clean response'

    def test_keeps_row_without_assistant(self):
        f = TokenSoupFilter()
        rows = [{'messages': [{'role': 'user', 'content': 'q'}]}]
        out = f(rows)
        assert len(out) == 1

    def test_any_assistant_soupy_drops_row(self):
        f = TokenSoupFilter()
        rows = [{
            'messages': [
                {
                    'role': 'user',
                    'content': 'q'
                },
                {
                    'role': 'assistant',
                    'content': 'fine'
                },
                {
                    'role': 'user',
                    'content': 'q2'
                },
                {
                    'role': 'assistant',
                    'content': '\ufffd' * 10 + 'a' * 5
                },
            ]
        }]
        out = f(rows)
        assert out == []

    def test_strips_whitespace_before_check(self):
        # Leading/trailing whitespace shouldn't bypass detection.
        f = TokenSoupFilter()
        rows = [_row('   ' + 'a' * 30 + '   ')]
        assert f(rows) == []

    def test_threshold_overrides_propagated(self):
        # With a stricter ratio, even small amounts of \ufffd trip it.
        f = TokenSoupFilter(replacement_char_ratio=0.0)
        rows = [_row('hello\ufffdworld')]
        assert f(rows) == []

    def test_empty_rows(self):
        assert TokenSoupFilter()([]) == []

    def test_messages_missing(self):
        f = TokenSoupFilter()
        rows = [{'id': 'no-msgs'}]
        out = f(rows)
        assert len(out) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
