# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the lightweight patches (Qwen3AllowToolTailTemplate, NoSplitModulesPatch)."""
from types import SimpleNamespace

import pytest

from twinkle.patch.no_split_modules import NoSplitModulesPatch
from twinkle.patch.qwen3_chat_template import _NEW_TAIL, _OLD_TAIL, Qwen3AllowToolTailTemplate


class TestQwen3AllowToolTailTemplate:

    def _tokenizer(self, template):
        return SimpleNamespace(chat_template=template)

    def test_template_with_old_tail_is_rewritten(self):
        tokenizer = self._tokenizer(f'head\n{_OLD_TAIL}tail')
        assert Qwen3AllowToolTailTemplate()(tokenizer) is True
        assert tokenizer.chat_template == f'head\n{_NEW_TAIL}tail'

    def test_patch_is_idempotent(self):
        tokenizer = self._tokenizer(f'head\n{_OLD_TAIL}tail')
        patch = Qwen3AllowToolTailTemplate()
        assert patch(tokenizer) is True
        patched = tokenizer.chat_template
        assert patch(tokenizer) is False
        assert tokenizer.chat_template == patched

    def test_missing_chat_template_is_untouched(self):
        for tokenizer in (SimpleNamespace(chat_template=None), SimpleNamespace(chat_template=['not', 'str'])):
            assert Qwen3AllowToolTailTemplate()(tokenizer) is False

    def test_template_without_old_tail_warns_and_skips(self):
        tokenizer = self._tokenizer('a template with no multi_step_tool block')
        with pytest.warns(RuntimeWarning, match='OLD multi_step_tool'):
            assert Qwen3AllowToolTailTemplate()(tokenizer) is False
        assert tokenizer.chat_template == 'a template with no multi_step_tool block'


class TestNoSplitModulesPatch:

    def test_single_string_name_is_normalized_to_a_set(self):
        patch = NoSplitModulesPatch('MyLayer')
        module = SimpleNamespace()
        result = patch(module)
        assert result is module
        assert module._no_split_modules == {'MyLayer'}

    def test_default_names_are_qwen3_5_layers(self):
        module = SimpleNamespace()
        NoSplitModulesPatch()(module)
        assert module._no_split_modules == {'Qwen3_5DecoderLayer'}

    def test_unpatch_removes_the_attribute(self):
        patch = NoSplitModulesPatch({'A', 'B'})
        module = SimpleNamespace()
        patch(module)
        assert patch.unpatch(module) is module
        assert not hasattr(module, '_no_split_modules')

    def test_unpatch_on_a_clean_module_is_a_no_op(self):
        clean = SimpleNamespace()
        assert NoSplitModulesPatch().unpatch(clean) is clean
        assert not hasattr(clean, '_no_split_modules')
