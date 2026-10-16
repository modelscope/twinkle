# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure-logic tests for the ``_not_encoded`` probe on both model backends.

``_not_encoded`` decides whether ``forward`` must run ``template.batch_encode`` on its inputs (a raw
Trajectory) or hand them straight to the processor (an already-encoded feature). It is a staticmethod
with no model state, so these cases call it directly on CPU.

The load-bearing case is the GROUP-shaped embedding / reranker row: one row carries an anchor plus its
candidates under prefixed keys (``anchor_input_ids`` / ``positive_input_ids`` / ``negative0_input_ids``)
and the processor splits it into flat per-sequence rows later. A bare-``input_ids``-only probe reads
such a row as raw and re-encodes it, which dies on the missing ``messages`` key. The probe must treat
ANY ``*input_ids`` / ``*input_embedding`` field as encoded.
"""
import pytest

from twinkle.model.transformers.transformers import TransformersModel

#: A raw Trajectory: messages + media, never a tokenized field.
_RAW = {'messages': [{'role': 'user', 'content': 'hi'}], 'images': ['x.png']}
#: A flat encoded causal_lm row.
_FLAT = {'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}
#: A precomputed-embedding row.
_INPUT_EMBEDDING = {'input_embedding': [[0.0, 1.0]]}
#: A GROUP-shaped encoded embedding row (anchor + positive + one negative), as swift's embedding
#: template emits it before the processor flattens it.
_GROUP = {
    'anchor_input_ids': [1, 2],
    'positive_input_ids': [3, 4],
    'negative0_input_ids': [5, 6],
    'labels': [1.0, 0.0, 0.0],
}
#: A packed batch arrives as list[list[row]]; the probe descends to the first leaf.
_PACKED_GROUP = [[_GROUP, _GROUP]]


@pytest.mark.parametrize('row,expected', [
    (_RAW, True),  # raw -> must re-encode
    (_FLAT, False),  # bare input_ids -> encoded
    (_INPUT_EMBEDDING, False),  # precomputed embedding -> encoded
    (_GROUP, False),  # group-shaped encoded row -> encoded (the regression this guards)
])
def test_transformers_not_encoded(row, expected):
    assert TransformersModel._not_encoded(dict(row)) is expected


def test_transformers_not_encoded_descends_into_packed_list():
    assert TransformersModel._not_encoded(_PACKED_GROUP) is False


def test_transformers_not_encoded_empty_list_is_not_raw():
    # An empty batch has nothing to encode; the probe reports False rather than asserting on inputs[0].
    assert TransformersModel._not_encoded([]) is False


def test_megatron_not_encoded_matches_transformers():
    """Both backends duplicate the probe verbatim; pin them together so one cannot drift."""
    megatron = pytest.importorskip('twinkle.model.megatron.megatron')
    for row, expected in [(_RAW, True), (_FLAT, False), (_GROUP, False)]:
        assert megatron.MegatronModel._not_encoded(dict(row)) is expected
