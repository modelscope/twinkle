# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the pure helpers in twinkle.sampler.generation."""
from types import SimpleNamespace

import pytest
import torch

from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.sampler.generation import (
    attach_features,
    collate_extra_model_inputs,
    error_response,
    group_by_adapter,
    not_encoded,
    run_chunk,
)


class TestNotEncoded:

    def test_raw_trajectory_is_not_encoded(self):
        assert not_encoded({'messages': [{'role': 'user', 'content': 'hi'}]})

    def test_input_ids_or_embedding_means_encoded(self):
        assert not not_encoded({'input_ids': [1, 2]})
        assert not not_encoded({'input_embedding': torch.zeros(2, 2)})

    def test_non_dict_input_is_rejected(self):
        with pytest.raises(AssertionError, match='Expected dict'):
            not_encoded(['not', 'a', 'dict'])


class TestErrorResponse:

    def test_placeholder_carries_prompt_ids_and_error_reason(self):
        response = error_response({'input_ids': [5, 6]})
        assert response.prompt_token_ids == [5, 6]
        assert len(response.sequences) == 1
        sequence = response.sequences[0]
        assert sequence.stop_reason == 'error'
        assert sequence.tokens == []
        assert sequence.decoded == ''

    def test_missing_prompt_ids_default_to_empty(self):
        assert error_response({}).prompt_token_ids == []
        assert error_response({'input_ids': None}).prompt_token_ids == []


class TestCollateExtraModelInputs:

    def test_text_only_batch_collates_to_nothing(self):
        encoded = {
            0: {'input_ids': [1], 'attention_mask': [1], 'labels': [1], 'note': 'str'},
            1: {'input_ids': [2], 'attention_mask': [1], 'labels': [1]},
        }
        assert collate_extra_model_inputs([0, 1], encoded) == {}

    def test_vision_tensors_concatenate_along_dim_zero(self):
        encoded = {
            0: {'input_ids': [1], 'pixel_values': torch.ones(2, 3)},
            1: {'input_ids': [2], 'pixel_values': torch.ones(1, 3)},
        }
        collated = collate_extra_model_inputs([0, 1], encoded)
        assert collated['pixel_values'].shape == (3, 3)

    def test_per_token_companions_are_left_padded_and_stacked(self):
        encoded = {
            0: {'input_ids': [1], 'mm_token_type_ids': torch.tensor([1., 2., 3., 4.])},
            1: {'input_ids': [2], 'mm_token_type_ids': torch.tensor([5., 6.])},
        }
        collated = collate_extra_model_inputs([0, 1], encoded)
        stacked = collated['mm_token_type_ids']
        assert stacked.shape == (2, 4)
        # The narrower row is LEFT-padded to the batch width (engine builds left-padded ids).
        assert torch.equal(stacked[1], torch.tensor([0., 0., 5., 6.]))


class TestAttachFeatures:

    class _Template:

        def __init__(self):
            self.calls = []

        def append_completion(self, feat, tokens):
            self.calls.append((feat, tokens))
            return SimpleNamespace(state=SimpleNamespace(feature={'feature_of': tuple(tokens)}))

    def _response(self, tokens):
        return SampleResponse(
            prompt_token_ids=[1],
            sequences=[SampledSequence(stop_reason='stop', tokens=tokens, decoded='')],
        )

    def test_none_template_is_a_no_op(self):
        template = self._Template()
        attach_features(None, [self._response([3])], {0: {'input_ids': [1]}}, {0: None})
        assert template.calls == []

    def test_feature_is_attached_through_append_completion(self):
        template = self._Template()
        response = self._response([3, 4])
        feat = {'input_ids': [1]}
        attach_features(template, [response], {0: feat}, {0: None})
        assert template.calls == [(feat, [3, 4])]
        assert response.sequences[0].new_input_feature == {'feature_of': (3, 4)}

    def test_empty_tokens_missing_feature_and_none_response_are_skipped(self):
        template = self._Template()
        response = self._response([])
        attach_features(template, [response, None], {0: {}, 1: {'input_ids': [9]}}, {0: None, 1: None})
        assert template.calls == []


class TestGroupByAdapter:

    def test_no_adapter_paths_yields_a_single_bucket(self):
        assert group_by_adapter([3, 1, 2], None, 'a') == {'a': [3, 1, 2]}

    def test_indices_are_bucketed_in_insertion_order(self):
        groups = group_by_adapter([0, 1, 2, 3], ['x', 'y', 'x', 'y'], None)
        assert groups == {'x': [0, 2], 'y': [1, 3]}

    def test_none_paths_form_their_own_bucket(self):
        groups = group_by_adapter([0, 1], ['x', None], None)
        assert groups == {'x': [0], None: [1]}


class TestRunChunk:

    def test_successful_batch_passes_prompts_and_extras(self):
        encoded = {0: {'input_ids': [10]}, 1: {'input_ids': [20, 21]}}
        seen = {}

        def call_batch(prompts, extra):
            seen.update(prompts=prompts, extra=extra)
            return ['r0', 'r1']

        assert run_chunk([0, 1], encoded, {0: None, 1: None}, call_batch) == ['r0', 'r1']
        assert seen['prompts'] == [[10], [20, 21]]
        assert seen['extra'] == {}

    def test_strict_failure_raises(self):
        def call_batch(prompts, extra):
            raise RuntimeError('boom')

        with pytest.raises(RuntimeError, match='boom'):
            run_chunk([0, 1], {0: {'input_ids': [1]}, 1: {'input_ids': [2]}}, {0: None, 1: None}, call_batch)

    def test_single_item_failure_raises_even_when_not_strict(self):
        def call_batch(prompts, extra):
            raise RuntimeError('boom')

        with pytest.raises(RuntimeError, match='boom'):
            run_chunk([0], {0: {'input_ids': [1]}}, {0: None}, call_batch, strict=False)

    def test_non_strict_failure_isolates_the_bad_input(self):
        encoded = {0: {'input_ids': [1]}, 1: {'input_ids': [2]}, 2: {'input_ids': [3]}}
        attempts = []

        def call_batch(prompts, extra):
            attempts.append(list(prompts))
            if len(prompts) == 3:
                raise RuntimeError('batch-wide failure')
            if prompts == [[2]]:
                raise RuntimeError('this row is broken')
            return [SampleResponse(
                prompt_token_ids=prompts[0],
                sequences=[SampledSequence(stop_reason='stop', tokens=[7], decoded='')],
            )]

        results = run_chunk([0, 1, 2], encoded, {0: None, 1: None, 2: None}, call_batch, strict=False)
        # One batch attempt, then one call per input to isolate the failure.
        assert attempts == [[[1], [2], [3]], [[1]], [[2]], [[3]]]
        assert results[0].sequences[0].tokens == [7]
        assert results[1].sequences[0].stop_reason == 'error'
        assert results[1].sequences[0].tokens == []
        assert results[2].sequences[0].tokens == [7]
