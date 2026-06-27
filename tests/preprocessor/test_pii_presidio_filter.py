# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for pure helpers in pii_presidio_filter.

Only validators and replacement primitives are tested here — the full
``PIIPresidioFilter`` requires presidio_analyzer + spacy + faker which are
heavy/optional deps.  Pure helpers are usable standalone and have clear
mathematical contracts.

Coverage:
  * ``_is_valid_cn_id``      — 18-digit checksum (last digit may be 'X')
  * ``_is_valid_luhn``       — Luhn algorithm with min length 13
  * ``_mask_keep_edges``     — keep head/tail, mask middle
  * ``_hash_short``          — SHA-256 prefix, deterministic w/ salt
  * ``Strategy.coerce``      — enum coercion + strict failure mode
"""
import hashlib
import pytest

from twinkle_agentic.preprocessor.pii_presidio_filter import (Strategy, _hash_short, _is_valid_cn_id, _is_valid_luhn,
                                                              _mask_keep_edges)

# ── _is_valid_cn_id ─────────────────────────────────────────────────────────


class TestIsValidCnId:
    """
    Verified against the official GB 11643-1999 weights:
      weights = (7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2)
      checks  = '10X98765432'
    Test ID `11010519491231002X` is a textbook valid example.
    """

    def test_valid_id_with_x_check(self):
        assert _is_valid_cn_id('11010519491231002X') is True

    def test_valid_id_with_x_lowercase(self):
        # Implementation upper-cases the check digit before compare.
        assert _is_valid_cn_id('11010519491231002x') is True

    def test_invalid_check_digit(self):
        # Flip the last char to a wrong number.
        assert _is_valid_cn_id('110105194912310020') is False

    def test_too_short(self):
        assert _is_valid_cn_id('110105194912310') is False

    def test_too_long(self):
        assert _is_valid_cn_id('11010519491231002X9') is False

    def test_non_digit_in_first_17(self):
        assert _is_valid_cn_id('1101051949123100AX') is False

    def test_empty(self):
        assert _is_valid_cn_id('') is False

    def test_18_digits_invalid_checksum(self):
        # 18 digits but last is wrong number
        assert _is_valid_cn_id('110105194912310029') is False


# ── _is_valid_luhn ──────────────────────────────────────────────────────────


class TestIsValidLuhn:
    """
    `4532015112830366` is a well-known Visa test number that satisfies Luhn.
    """

    def test_valid_visa_test_number(self):
        assert _is_valid_luhn('4532015112830366') is True

    def test_valid_with_separators(self):
        # Implementation strips non-digits via `c.isdigit()`.
        assert _is_valid_luhn('4532-0151-1283-0366') is True
        assert _is_valid_luhn('4532 0151 1283 0366') is True

    def test_invalid_checksum(self):
        # Flip the last digit.
        assert _is_valid_luhn('4532015112830367') is False

    def test_too_short(self):
        # Only 12 digits — below 13-digit minimum.
        assert _is_valid_luhn('453201511283') is False

    def test_empty(self):
        assert _is_valid_luhn('') is False

    def test_no_digits(self):
        assert _is_valid_luhn('abcd-efgh-ijkl-mnop') is False

    def test_amex_test_number(self):
        # 15-digit Amex test card.
        assert _is_valid_luhn('378282246310005') is True

    def test_mastercard_test_number(self):
        assert _is_valid_luhn('5555555555554444') is True


# ── _mask_keep_edges ────────────────────────────────────────────────────────


class TestMaskKeepEdges:

    def test_default_head_tail(self):
        # head=3, tail=4 → keep 3 + mask middle + keep 4
        s = '13800138000'  # 11 chars
        # 11 > 3+4 = 7 → masked = 11 - 7 = 4 stars
        out = _mask_keep_edges(s)
        assert out == '138' + '*' * 4 + '8000'

    def test_short_string_all_masked(self):
        # len ≤ head+tail → entire string masked.
        s = 'short'  # 5 chars; head+tail = 7
        assert _mask_keep_edges(s) == '*****'

    def test_at_threshold_all_masked(self):
        # len == head+tail → all masked (boundary is `<=`)
        s = '1234567'  # 7 chars
        assert _mask_keep_edges(s) == '*' * 7

    def test_custom_head_tail(self):
        s = 'abcdefghij'  # 10 chars
        # head=2, tail=2 → keep ab + 6 stars + ij
        assert _mask_keep_edges(s, head=2, tail=2) == 'ab' + '*' * 6 + 'ij'

    def test_custom_mask_char(self):
        s = '1234567890'
        out = _mask_keep_edges(s, head=1, tail=1, ch='X')
        assert out == '1' + 'X' * 8 + '0'

    def test_empty_string(self):
        # len=0 ≤ head+tail → '' * 0 = ''
        assert _mask_keep_edges('') == ''

    def test_credit_card_default(self):
        s = '4532015112830366'  # 16 chars
        out = _mask_keep_edges(s)
        # head=3, tail=4 → keep 453 + 9 stars + 0366
        assert out == '453' + '*' * 9 + '0366'


# ── _hash_short ─────────────────────────────────────────────────────────────


class TestHashShort:

    def test_length_is_12(self):
        assert len(_hash_short('alice@example.com')) == 12

    def test_deterministic_same_input(self):
        a = _hash_short('hello')
        b = _hash_short('hello')
        assert a == b

    def test_different_inputs_different_outputs(self):
        a = _hash_short('alice@example.com')
        b = _hash_short('bob@example.com')
        assert a != b

    def test_salt_changes_output(self):
        a = _hash_short('hello', salt='')
        b = _hash_short('hello', salt='secret')
        assert a != b

    def test_matches_sha256_prefix(self):
        expected = hashlib.sha256(b'hello').hexdigest()[:12]
        assert _hash_short('hello') == expected

    def test_matches_sha256_with_salt(self):
        expected = hashlib.sha256(b'saltyhello').hexdigest()[:12]
        assert _hash_short('hello', salt='salty') == expected

    def test_empty_string(self):
        # Hash is well-defined for empty input too.
        expected = hashlib.sha256(b'').hexdigest()[:12]
        assert _hash_short('') == expected

    def test_unicode_input(self):
        # UTF-8 encoding before hashing.
        expected = hashlib.sha256('张三'.encode()).hexdigest()[:12]
        assert _hash_short('张三') == expected


# ── Strategy.coerce ─────────────────────────────────────────────────────────


class TestStrategyCoerce:

    def test_coerce_string_to_enum(self):
        assert Strategy.coerce('mask') is Strategy.MASK
        assert Strategy.coerce('replace') is Strategy.REPLACE
        assert Strategy.coerce('redact') is Strategy.REDACT
        assert Strategy.coerce('hash') is Strategy.HASH

    def test_coerce_enum_returns_self(self):
        assert Strategy.coerce(Strategy.MASK) is Strategy.MASK

    def test_coerce_unknown_raises(self):
        with pytest.raises(ValueError) as exc:
            Strategy.coerce('encrypt')
        # Error message lists allowed strategies for diagnosability.
        msg = str(exc.value)
        assert 'mask' in msg
        assert 'replace' in msg
        assert 'redact' in msg
        assert 'hash' in msg

    def test_coerce_empty_string_raises(self):
        with pytest.raises(ValueError):
            Strategy.coerce('')

    def test_string_enum_membership(self):
        # Strategy is a str-Enum: values should compare equal to their str form.
        assert Strategy.MASK == 'mask'
        assert Strategy.REPLACE.value == 'replace'

    def test_coerce_case_sensitive(self):
        # Implementation does not lowercase before lookup.
        with pytest.raises(ValueError):
            Strategy.coerce('MASK')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
