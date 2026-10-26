# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the pure helpers in twinkle.template.utils."""
from types import SimpleNamespace

import pytest
import torch

from twinkle.template.utils import (
    TokenizeByPlaceHolder,
    TokenizeByRound,
    _coerce_ids_to_list,
    _convert_to_vlm_format,
    _is_vlm_processor,
    construct_template,
    transfer_to_standard_message,
)


class TestCoerceIdsToList:

    def test_one_dim_tensor_becomes_list(self):
        assert _coerce_ids_to_list(torch.tensor([1, 2, 3])) == [1, 2, 3]

    def test_single_sample_leading_dims_are_squeezed(self):
        assert _coerce_ids_to_list(torch.tensor([[4, 5]])) == [4, 5]
        assert _coerce_ids_to_list(torch.ones(1, 1, 2, dtype=torch.long)) == [1, 1]

    def test_multi_sample_tensor_is_rejected(self):
        with pytest.raises(ValueError, match='single-sample tensor'):
            _coerce_ids_to_list(torch.tensor([[1, 2], [3, 4]]))

    def test_non_tensor_passes_through(self):
        assert _coerce_ids_to_list([1, 2]) == [1, 2]


class TestConvertToVlmFormat:

    def test_string_content_is_wrapped_in_text_block(self):
        converted = _convert_to_vlm_format([{'role': 'user', 'content': 'hi'}])
        assert converted == [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]

    def test_block_content_and_missing_content_are_untouched(self):
        messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]},
            {'role': 'user', 'tool_calls': []},
        ]
        assert _convert_to_vlm_format(messages) == messages


class TestIsVlmProcessor:

    def test_processor_with_image_processor_is_vlm(self):
        tokenizer = SimpleNamespace(tokenizer=object(), image_processor=object())
        assert _is_vlm_processor(tokenizer)

    def test_plain_tokenizer_is_not_vlm(self):
        assert not _is_vlm_processor(SimpleNamespace(tokenizer=object()))
        assert not _is_vlm_processor(object())


class TestTransferToStandardMessage:

    def test_plain_message_without_mm(self):
        message = {'role': 'user', 'content': 'hello'}
        transferred = transfer_to_standard_message(message, None, None, None, is_mm=False)
        assert transferred['content'] == [{'type': 'text', 'text': 'hello'}]
        assert transferred['role'] == 'user'

    def test_single_image_placeholder_splits_content(self):
        message = {
            'role': 'user',
            'content': 'before <img> after',
            'images': ['http://example.com/a.png'],
        }
        transferred = transfer_to_standard_message(message, '<img>', None, None, is_mm=True)
        assert transferred['content'] == [
            {'type': 'text', 'text': 'before '},
            {'type': 'image', 'url': 'http://example.com/a.png'},
            {'type': 'text', 'text': ' after'},
        ]

    def test_extra_placeholders_stay_as_literal_text(self):
        # Two placeholders but one image: the second placeholder must survive verbatim
        # instead of being silently swallowed.
        message = {'role': 'user', 'content': 'a <img> b <img> c', 'images': ['u1']}
        transferred = transfer_to_standard_message(message, '<img>', None, None, is_mm=True)
        assert transferred['content'] == [
            {'type': 'text', 'text': 'a '},
            {'type': 'image', 'url': 'u1'},
            {'type': 'text', 'text': ' b <img> c'},
        ]

    def test_none_media_entry_consumes_placeholder_without_block(self):
        message = {'role': 'user', 'content': 'a <img> b', 'images': [None]}
        transferred = transfer_to_standard_message(message, '<img>', None, None, is_mm=True)
        # The placeholder is consumed, no media block is emitted, and the surrounding
        # text survives as two separate blocks.
        assert transferred['content'] == [
            {'type': 'text', 'text': 'a '}, {'type': 'text', 'text': ' b'}]

    def test_whitespace_only_text_segments_are_dropped(self):
        message = {'role': 'user', 'content': '  <img>  ', 'images': ['u1']}
        transferred = transfer_to_standard_message(message, '<img>', None, None, is_mm=True)
        assert transferred['content'] == [{'type': 'image', 'url': 'u1'}]

    def test_mixed_media_types_are_ordered_by_position(self):
        message = {
            'role': 'user',
            'content': 'v <vid> i <img> tail',
            'images': ['img-url'],
            'videos': ['vid-url'],
        }
        transferred = transfer_to_standard_message(message, '<img>', '<vid>', None, is_mm=True)
        assert transferred['content'] == [
            {'type': 'text', 'text': 'v '},
            {'type': 'video', 'url': 'vid-url'},
            {'type': 'text', 'text': ' i '},
            {'type': 'image', 'url': 'img-url'},
            {'type': 'text', 'text': ' tail'},
        ]

    def test_empty_content_yields_no_blocks(self):
        message = {'role': 'user', 'content': ''}
        transferred = transfer_to_standard_message(message, '<img>', None, None, is_mm=True)
        assert transferred['content'] == []


class TestConstructTemplateSpecPath:

    def test_extra_kwargs_with_spec_are_rejected(self):
        with pytest.raises(TypeError, match='Unexpected TemplateSpec construction arguments'):
            construct_template(object, template_spec=object(), anything=1)

    def test_undotted_factory_path_is_rejected(self):
        with pytest.raises(ValueError, match='dotted class path'):
            construct_template('NotDotted', template_spec=object())

    def test_class_without_from_spec_is_rejected(self):
        class NoFromSpec:
            pass

        with pytest.raises(TypeError, match='does not implement from_spec'):
            construct_template(NoFromSpec, template_spec=object())

    def test_from_spec_factory_receives_the_spec(self):
        marker = object()

        class WithFromSpec:

            @classmethod
            def from_spec(cls, spec):
                assert spec is marker
                return 'built'

        assert construct_template(WithFromSpec, template_spec=marker) == 'built'


class TestTokenizeByPlaceHolderStatics:

    def test_find_subsequence_basic_and_offset(self):
        seq = [1, 2, 3, 1, 2, 3, 4]
        assert TokenizeByPlaceHolder.find_subsequence(seq, [1, 2, 3]) == 0
        assert TokenizeByPlaceHolder.find_subsequence(seq, [1, 2, 3], start=1) == 3
        assert TokenizeByPlaceHolder.find_subsequence(seq, [1, 2, 3], start=4) == -1
        assert TokenizeByPlaceHolder.find_subsequence(seq, [9]) == -1
        assert TokenizeByPlaceHolder.find_subsequence([1], [1, 2]) == -1

    def test_split_by_subsequence(self):
        assert TokenizeByPlaceHolder.split_by_subsequence([1, 9, 9, 2, 9, 9, 3], [9, 9]) == [[1], [2], [3]]
        assert TokenizeByPlaceHolder.split_by_subsequence([1, 2], [9]) == [[1, 2]]

    def test_build_labels_masks_template_parts(self):
        labels = TokenizeByPlaceHolder.build_labels([7, 1, 2, 8, 3, 4, 9], [[1, 2], [3, 4]])
        assert labels == [7, -100, -100, 8, -100, -100, 9]

    def test_build_labels_missing_part_raises(self):
        with pytest.raises(ValueError, match='Template part not found'):
            TokenizeByPlaceHolder.build_labels([1, 2], [[3]])


class _FakeTokenizer:

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _char_ids(text):
    return [ord(c) for c in text]


def _placeholder_encode_func(trajectory, **kwargs):
    """Render each message as <role-marker> content <tail-marker>."""
    ids = []
    for message in trajectory['messages']:
        ids += [0] + _char_ids(message['content']) + [1]
    return {'input_ids': ids}


class TestTokenizeByPlaceHolderLabels:

    def _trajectory(self, contents):
        return {'messages': [
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': text}
            for i, text in enumerate(contents)
        ]}

    def test_assistant_spans_are_the_only_trained_tokens(self):
        trajectory = self._trajectory(['AB', 'XY', 'CD', 'EF'])
        full_ids, labels, _ = TokenizeByPlaceHolder.tokenize_with_assistant_labels(
            _FakeTokenizer(), _placeholder_encode_func, trajectory)
        assert full_ids == [0] + _char_ids('AB') + [1] + [0] + _char_ids('XY') + [1] \
            + [0] + _char_ids('CD') + [1] + [0] + _char_ids('EF') + [1]
        # Scaffolding before 'XY' (5 tokens), between 'XY' and 'EF' (6 tokens) stays
        # masked; the tail marker after 'EF' is recovered by the trailing-assistant
        # fixup so the final assistant turn trains to the very end.
        expected = [-100] * 5 + _char_ids('XY') + [-100] * 6 + _char_ids('EF') + [1]
        assert labels == expected

    def test_non_terminal_assistant_keeps_template_tail_masked(self):
        trajectory = self._trajectory(['AB', 'XY', 'CD'])
        _, labels, _ = TokenizeByPlaceHolder.tokenize_with_assistant_labels(
            _FakeTokenizer(), _placeholder_encode_func, trajectory)
        # 'XY' is trained, everything else (including the tail marker after XY and the
        # final user turn) stays -100 because the trajectory does not end on assistant.
        trained = [i for i, label in enumerate(labels) if label != -100]
        xy_start = labels.index(ord('X'))
        assert trained == [xy_start, xy_start + 1]

    def test_assistant_with_multiple_text_items_is_rejected(self):
        trajectory = {'messages': [
            {'role': 'user', 'content': 'AB'},
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]},
        ]}
        with pytest.raises(ValueError, match='exactly one text item'):
            TokenizeByPlaceHolder.tokenize_with_assistant_labels(
                _FakeTokenizer(), _placeholder_encode_func, trajectory)


def _round_encode_func(trajectory, add_generation_prompt=False, **kwargs):
    # ChatML-style: an assistant message renders as <assistant-prefix> content <tail>,
    # and the generation prompt emits the same <assistant-prefix> -- so the prefix
    # tokens are shared between both encodings, which is what TokenizeByRound relies on.
    ids = []
    for message in trajectory['messages']:
        if message['role'] == 'assistant':
            ids += [13, 14] + _char_ids(message['content']) + [12]
        else:
            ids += [10] + _char_ids(message['role']) + [11] + _char_ids(message['content']) + [12]
    if add_generation_prompt:
        ids += [13, 14]
    return {'input_ids': ids}


class TestTokenizeByRoundLabels:

    def _trajectory(self, contents):
        return {'messages': [
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': text}
            for i, text in enumerate(contents)
        ]}

    def test_only_assistant_content_is_trained(self):
        trajectory = self._trajectory(['AB', 'XY', 'CD'])
        full_ids, labels, _ = TokenizeByRound.tokenize_with_assistant_labels(
            None, _round_encode_func, trajectory)
        # user turn: 10 'user' 11 'AB' 12 ; assistant prefix 13 14 ; then 'XY' 12.
        trained = [full_ids[i] for i, label in enumerate(labels) if label != -100]
        assert trained == _char_ids('XY') + [12]
        # The assistant prefix tokens must be excluded from training.
        prefix_positions = [i for i, token in enumerate(full_ids) if token in (13, 14)]
        assert all(labels[i] == -100 for i in prefix_positions)

    def test_train_indices_limit_labelled_turns(self):
        trajectory = self._trajectory(['AB', 'XY', 'CD', 'EF'])
        _, labels, _ = TokenizeByRound.tokenize_with_assistant_labels(
            None, _round_encode_func, trajectory, train_indices={3})
        trained = [i for i, label in enumerate(labels) if label != -100]
        # Only the second assistant turn is labelled: prefix [13, 14] at positions
        # 23-24 stays masked, then 'E', 'F' and the tail marker are trained.
        assert trained == [25, 26, 27]
        assert [labels[i] for i in trained] == _char_ids('EF') + [12]

    def test_empty_trajectory_labels_nothing(self):
        full_ids, labels, encoded = TokenizeByRound.tokenize_with_assistant_labels(
            None, _round_encode_func, {'messages': []})
        assert full_ids == []
        assert labels == []
        assert encoded == {}

    def test_add_generation_prompt_kwarg_is_consumed(self):
        # The caller may pass add_generation_prompt; TokenizeByRound pops it before
        # delegating to encode_func.
        trajectory = self._trajectory(['AB', 'XY'])
        full_ids, labels, _ = TokenizeByRound.tokenize_with_assistant_labels(
            None, _round_encode_func, trajectory, add_generation_prompt=True)
        assert full_ids == _round_encode_func(trajectory)['input_ids']
