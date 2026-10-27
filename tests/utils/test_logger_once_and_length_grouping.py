# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for logger once-semantics and length-grouped batch indices."""
import uuid

import torch

from twinkle.utils import get_logger
from twinkle.utils.logger import info_once, warning_once
from twinkle.utils.torch_utils import get_length_grouped_indices


class TestLogOnce:

    def _logger_with_counter(self, monkeypatch, method):
        logger = get_logger()
        calls = []
        monkeypatch.setattr(logger, method, lambda msg, *args, **kwargs: calls.append(msg),
                            raising=False)
        return logger, calls

    def test_warning_once_emits_each_message_only_once(self, monkeypatch):
        logger, calls = self._logger_with_counter(monkeypatch, 'warning')
        message = f'unique-warning-{uuid.uuid4()}'
        warning_once(logger, message)
        warning_once(logger, message)
        assert calls == [message]

    def test_info_once_emits_each_message_only_once(self, monkeypatch):
        logger, calls = self._logger_with_counter(monkeypatch, 'info')
        message = f'unique-info-{uuid.uuid4()}'
        info_once(logger, message)
        info_once(logger, message)
        assert calls == [message]

    def test_distinct_messages_both_pass(self, monkeypatch):
        logger, calls = self._logger_with_counter(monkeypatch, 'warning')
        first = f'unique-a-{uuid.uuid4()}'
        second = f'unique-b-{uuid.uuid4()}'
        warning_once(logger, first)
        warning_once(logger, second)
        assert calls == [first, second]


class TestGetLengthGroupedIndices:

    def _sorted_run(self, lengths, batch_size=2, mega_batch_mult=1):
        generator = torch.Generator().manual_seed(0)
        return get_length_grouped_indices(lengths, batch_size,
                                          mega_batch_mult=mega_batch_mult, generator=generator)

    def test_result_is_a_permutation(self):
        lengths = [10, 1, 5, 3, 8, 2]
        assert sorted(self._sorted_run(lengths)) == list(range(len(lengths)))

    def test_batches_are_sorted_by_descending_length(self):
        lengths = [10, 1, 5, 3, 8, 2]
        result = self._sorted_run(lengths)
        for start in range(0, len(lengths), 2):
            batch = result[start:start + 2]
            if len(batch) == 2:
                assert lengths[batch[0]] >= lengths[batch[1]]

    def test_longest_sample_surfaces_in_the_first_batch(self):
        # The megabatch holding the overall longest element is moved first, so an
        # out-of-distribution length fails on the very first batch.
        lengths = [10, 1, 5, 3, 8, 2]
        result = self._sorted_run(lengths)
        assert lengths[result[0]] == max(lengths)

    def test_tiny_dataset_default_multiplier_stays_positive(self):
        result = get_length_grouped_indices([4, 2], 2, generator=torch.Generator().manual_seed(0))
        assert sorted(result) == [0, 1]
