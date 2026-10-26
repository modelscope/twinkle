# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the pure parsing/validation helpers of the vLLM engine."""
from types import SimpleNamespace

import pytest
import torch

from twinkle.sampler.vllm_sampler.vllm_engine import (
    _as_flat_int_list,
    _map_finish_reason,
    _parse_chunked_csr,
    _parse_token_rows,
    _to_list,
    _validate_sampling_mask_csr,
    _validate_sampling_replay_params,
    get_vllm_max_lora_rank,
)


class TestMapFinishReason:

    def test_known_reasons_pass_through(self):
        assert _map_finish_reason('stop') == 'stop'
        assert _map_finish_reason('abort') == 'abort'
        assert _map_finish_reason('error') == 'error'

    def test_none_and_unknown_fall_back_to_length(self):
        assert _map_finish_reason(None) == 'length'
        assert _map_finish_reason('length') == 'length'
        assert _map_finish_reason('something new') == 'length'


class TestToList:

    def test_tensor_is_converted(self):
        assert _to_list(torch.tensor([1, 2])) == [1, 2]

    def test_plain_values_pass_through(self):
        assert _to_list([1, 2]) == [1, 2]
        assert _to_list(None) is None


class TestAsFlatIntList:

    def test_flat_list_is_coerced_to_ints(self):
        assert _as_flat_int_list([1, '2', 3.0], 'token_ids') == [1, 2, 3]

    def test_tensor_input_is_accepted(self):
        assert _as_flat_int_list(torch.tensor([4, 5]), 'token_ids') == [4, 5]

    def test_nested_structure_is_rejected(self):
        with pytest.raises(RuntimeError, match='flat integer list'):
            _as_flat_int_list([[1, 2]], 'token_ids')

    def test_non_container_is_rejected(self):
        with pytest.raises(RuntimeError, match='flat integer list'):
            _as_flat_int_list(7, 'token_ids')


class TestValidateSamplingMaskCsr:

    def test_valid_csr_passes(self):
        _validate_sampling_mask_csr([1, 2, 3, 4], [0, 2, 4], num_tokens=2)

    def test_row_count_must_match_num_tokens(self):
        with pytest.raises(RuntimeError, match='rows for'):
            _validate_sampling_mask_csr([1, 2], [0, 1, 2], num_tokens=3)

    def test_endpoints_must_be_zero_and_total(self):
        with pytest.raises(RuntimeError, match='CSR endpoints'):
            _validate_sampling_mask_csr([1, 2], [1, 2], num_tokens=1)
        with pytest.raises(RuntimeError, match='CSR endpoints'):
            _validate_sampling_mask_csr([1, 2], [0, 1], num_tokens=1)

    def test_empty_rows_are_rejected(self):
        # Endpoints are consistent (0 start, 2 == len(token_ids)) but the middle row
        # is empty, which the row check must reject.
        with pytest.raises(RuntimeError, match='empty or invalid CSR row'):
            _validate_sampling_mask_csr([1, 2], [0, 1, 1, 2], num_tokens=3)


class TestParseTokenRows:

    def test_rows_are_flattened_into_csr(self):
        token_ids, offsets = _parse_token_rows([[5, 6], [7], [8, 9]], num_tokens=3)
        assert token_ids == [5, 6, 7, 8, 9]
        assert offsets == [0, 2, 3, 5]

    def test_row_count_mismatch_is_rejected(self):
        with pytest.raises(RuntimeError, match='one token-id row per sampled token'):
            _parse_token_rows([[1], [2]], num_tokens=3)


class TestParseChunkedCsr:

    def test_chunks_are_rebased_and_concatenated(self):
        token_ids, offsets = _parse_chunked_csr(
            [[1, 2], [3]],
            [[0, 2], [0, 1]],
        )
        assert token_ids == [1, 2, 3]
        assert offsets == [0, 2, 3]

    def test_chunk_length_mismatch_is_rejected(self):
        with pytest.raises(RuntimeError, match='matching token-id and offset chunks'):
            _parse_chunked_csr([[1]], [[0, 1], [0, 1]])

    def test_invalid_chunk_endpoints_are_rejected(self):
        with pytest.raises(RuntimeError, match='chunk 0 has invalid CSR offsets'):
            _parse_chunked_csr([[1, 2]], [[0, 1]])


class TestValidateSamplingReplayParams:

    def _ok(self, **extra):
        params = {'repetition_penalty': 1.0, 'presence_penalty': 0.0, 'top_k': 8}
        _validate_sampling_replay_params(params, extra)

    def test_neutral_values_pass(self):
        self._ok()
        self._ok(top_k=16)

    def test_score_changing_processor_is_named_in_the_error(self):
        params = {'repetition_penalty': 1.2, 'top_k': 8}
        with pytest.raises(ValueError, match='repetition_penalty'):
            _validate_sampling_replay_params(params, {})

    def test_logit_bias_and_processors_are_flagged_when_truthy(self):
        with pytest.raises(ValueError, match='logit_bias'):
            _validate_sampling_replay_params({'top_k': 8}, {'logit_bias': {1: 2.0}})
        with pytest.raises(ValueError, match='logits_processor'):
            _validate_sampling_replay_params({'top_k': 8}, {'logits_processor': lambda ids, logits: logits})

    def test_top_k_must_be_a_positive_int(self):
        for bad in (None, 0, -1, '8', 8.0):
            with pytest.raises(ValueError, match='top_k > 0'):
                _validate_sampling_replay_params({'top_k': bad}, {})

    def test_namespace_params_are_supported(self):
        params = SimpleNamespace(repetition_penalty=1.5, top_k=8)
        with pytest.raises(ValueError, match='repetition_penalty'):
            _validate_sampling_replay_params(params, {})

    def test_extra_kwargs_top_k_takes_precedence(self):
        params = {'top_k': 8}
        with pytest.raises(ValueError, match='top_k > 0'):
            _validate_sampling_replay_params(params, {'top_k': None})


class TestGetVllmMaxLoraRank:

    def test_rank_is_rounded_up_to_the_nearest_allowed(self):
        # Allowed ranks are (1, 8, 16, 32, 64, 128, 256, 320, 512); a non-listed
        # rank must round up to the next allowed one.
        assert get_vllm_max_lora_rank(1) == 1
        assert get_vllm_max_lora_rank(4) == 8
        assert get_vllm_max_lora_rank(33) == 64

    def test_fallback_without_vllm_lora_config(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, 'vllm.config.lora', None)
        assert get_vllm_max_lora_rank(24) == 24


class TestTiedEmbeddingAliases:

    def _patched(self, monkeypatch):
        import torch.nn as nn
        import vllm.model_executor.layers.vocab_parallel_embedding as vpe_module

        class _FakeVocabParallelEmbedding(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

        monkeypatch.setattr(vpe_module, 'VocabParallelEmbedding', _FakeVocabParallelEmbedding)
        return _FakeVocabParallelEmbedding

    def test_tied_embedding_duplicate_names_are_reported(self, monkeypatch):
        from twinkle.sampler.vllm_sampler.delta.weight_update_utils import tied_embedding_aliases

        fake_cls = self._patched(monkeypatch)
        import torch.nn as nn

        shared = nn.Parameter(torch.zeros(3, 3))
        model = nn.Module()
        model.embed_tokens = fake_cls(shared)
        model.lm_head = fake_cls(shared)
        # The first traversal name is canonical; the later duplicate is the alias.
        assert tied_embedding_aliases(model) == {'lm_head.weight'}

    def test_untied_embeddings_report_nothing(self, monkeypatch):
        from twinkle.sampler.vllm_sampler.delta.weight_update_utils import tied_embedding_aliases

        fake_cls = self._patched(monkeypatch)
        import torch.nn as nn

        model = nn.Module()
        model.embed_tokens = fake_cls(nn.Parameter(torch.zeros(3, 3)))
        model.lm_head = fake_cls(nn.Parameter(torch.zeros(3, 3)))
        assert tied_embedding_aliases(model) == set()

    def test_plain_modules_are_ignored(self, monkeypatch):
        from twinkle.sampler.vllm_sampler.delta.weight_update_utils import tied_embedding_aliases

        self._patched(monkeypatch)
        import torch.nn as nn

        model = nn.Module()
        model.linear = nn.Linear(2, 2, bias=False)
        assert tied_embedding_aliases(model) == set()
