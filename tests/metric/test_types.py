# Copyright (c) ModelScope Contributors. All rights reserved.
"""Contract of the transport-neutral metric record."""

import dataclasses

import pytest

from twinkle.metric import MetricRecord


def test_defaults_and_copy_on_construction():
    values = {'loss': 1.0}
    attributes = {'scope': 'group'}
    record = MetricRecord(stage='train', values=values, attributes=attributes)

    values['loss'] = 2.0
    attributes['scope'] = 'changed'

    assert record.status == 'completed'
    assert record.sequence is None
    assert record.values == {'loss': 1.0}
    assert record.attributes == {'scope': 'group'}


def test_record_is_frozen():
    record = MetricRecord(stage='train', values={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.stage = 'run'


def test_unknown_stage_or_status_is_rejected():
    with pytest.raises(ValueError, match='unsupported metric stage'):
        MetricRecord(stage='bogus', values={})
    with pytest.raises(ValueError, match='unsupported metric status'):
        MetricRecord(stage='train', values={}, status='bogus')
