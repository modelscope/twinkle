# Copyright (c) ModelScope Contributors. All rights reserved.
"""Wire schema behaviour for the inline ``inputs`` data plane.

These assertions are made against the schema directly, not through a mock backend: the
mock accepts ``**kwargs`` without inspecting anything, so "the mock did not complain"
is evidence of nothing about validation.
"""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError
from typing import Union, get_args, get_origin

from twinkle.data_format.encoding import ENCODED_INPUT_KEYS, is_encoded
from twinkle.processor.base import InputProcessor
from twinkle_client.types import data as wire

_INPUTS = TypeAdapter(wire.WireInputBatch)


def _parse(payload):
    return _INPUTS.validate_python(payload)


# --------------------------------------------------------------------------- #
# Classification and homogeneity
# --------------------------------------------------------------------------- #


def test_encoded_entry_parses_as_input_feature():
    (entry, ) = _parse([{'input_ids': [1, 2, 3]}])
    assert isinstance(entry, wire.WireInputFeature)


def test_embedding_only_entry_is_encoded_not_a_trajectory():
    """An embedding-only batch is already encoded.

    Reading it as a ``Trajectory`` would send it through ``template.batch_encode`` and
    fail far from the cause, which is why ``input_embedding`` is part of the shared
    predicate rather than only ``input_ids``.
    """
    (entry, ) = _parse([{'input_embedding': [[0.1, 0.2]]}])
    assert isinstance(entry, wire.WireInputFeature)


def test_message_entry_parses_as_trajectory():
    (entry, ) = _parse([{'messages': [{'role': 'user', 'content': 'hi'}]}])
    assert isinstance(entry, wire.WireTrajectory)


def test_mixed_batch_is_rejected():
    with pytest.raises(ValidationError):
        _parse([{'input_ids': [1]}, {'messages': [{'role': 'user', 'content': 'x'}]}])


def test_entry_without_a_required_key_is_rejected():
    with pytest.raises(ValidationError):
        _parse([{'labels': [1, 2]}])


@pytest.mark.parametrize('payload', [[1, 2, 3], 'text', 42, True, None])
def test_non_object_inputs_are_rejected(payload):
    with pytest.raises(ValidationError):
        _parse(payload)


def test_a_single_entry_is_accepted_as_a_one_element_batch():
    assert len(_parse({'input_ids': [1, 2]})) == 1


def test_required_key_rule_matches_the_shared_predicate():
    """The schema's required-key rule and ``is_encoded`` must stay the same rule.

    If they drifted, an entry could be an ``InputFeature`` to one and a ``Trajectory``
    to the other -- the exact divergence the single shared predicate exists to prevent.
    """
    for key in ENCODED_INPUT_KEYS:
        entry = {key: [1] if key == 'input_ids' else [[0.5]]}
        assert is_encoded(entry)
        assert isinstance(_parse([entry])[0], wire.WireInputFeature)
    assert not is_encoded({'messages': []})


# --------------------------------------------------------------------------- #
# Strictness on declared numeric fields
# --------------------------------------------------------------------------- #


def test_bool_tokens_are_rejected():
    """Lax ``int`` would coerce ``[true, false]`` to ``[1, 0]`` and train on it."""
    with pytest.raises(ValidationError):
        _parse([{'input_ids': [True, False]}])


def test_float_tokens_are_rejected():
    """Values come from a tensor's ``tolist()``; a ``1.0`` there is an upstream defect."""
    with pytest.raises(ValidationError):
        _parse([{'input_ids': [1.0]}])


def test_negative_labels_are_accepted():
    (entry, ) = _parse([{'input_ids': [1, 2], 'labels': [-100, 5]}])
    assert entry.labels == [-100, 5]


def test_float_vlm_values_are_accepted():
    (entry, ) = _parse([{'input_ids': [1], 'pixel_values': [[0.5, 0.25]]}])
    assert entry.pixel_values == [[0.5, 0.25]]


@pytest.mark.parametrize('position_ids', [[0, 1], [[0, 1], [2, 3]], [[[0, 1]]]])
def test_position_ids_accept_one_to_three_dimensions(position_ids):
    (entry, ) = _parse([{'input_ids': [1, 2], 'position_ids': position_ids}])
    assert entry.position_ids == position_ids


def test_position_ids_reject_a_scalar():
    with pytest.raises(ValidationError):
        _parse([{'input_ids': [1, 2], 'position_ids': 0}])


def test_routed_experts_require_exactly_three_dimensions():
    with pytest.raises(ValidationError):
        _parse([{'input_ids': [1, 2], 'routed_experts': [0, 1]}])


# --------------------------------------------------------------------------- #
# Extension data and export
# --------------------------------------------------------------------------- #


def test_unknown_json_fields_survive_a_round_trip():
    """A preprocessor's leftover columns must reach the backend, not be dropped.

    ``extra='ignore'`` would accept the request and then silently strip these on export,
    which loses data the caller sent -- a worse outcome than rejecting it.
    """
    payload = {'input_ids': [1, 2], 'source_id': 'row-7', 'score': 0.5}
    exported = wire.export(_parse([payload])[0])
    assert exported == payload


def test_export_omits_unset_optional_fields():
    """Twinkle_Core branches on key *presence*, so ``None`` must not be emitted."""
    exported = wire.export(_parse([{'input_ids': [1, 2]}])[0])
    assert exported == {'input_ids': [1, 2]}


def test_round_trip_is_idempotent():
    samples = [
        {'input_ids': [1, 2, 3]},
        {'input_ids': [[1, 2], [3, 4]]},
        {'input_ids': [1, 2], 'position_ids': [[0, 1]]},
        {'input_ids': [1, 2], 'position_ids': [[[0, 1]]]},
        {'input_ids': [1, 2], 'routed_experts': [[[0, 1]]]},
        {'input_ids': [1, 2], 'labels': [-100, 4]},
        {'input_embedding': [[0.5]]},
        {'messages': [{'role': 'user', 'content': 'x'}], 'user_data': [['k', '"v"']]},
        {'messages': []},
    ]
    for sample in samples:
        once = wire.export(_parse([sample])[0])
        twice = wire.export(_parse([once])[0])
        assert once == twice, sample


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #


def _depth(annotation) -> int:
    depth = 0
    while get_origin(annotation) is list:
        depth += 1
        annotation = get_args(annotation)[0]
    return depth


@pytest.mark.parametrize('alias', ['Ints1to2', 'Ints1to3', 'Numbers1to2', 'Numbers1to4'])
def test_union_members_are_declared_shallowest_first(alias):
    """Deepest-first is ~41x slower on a 2-D input, so the order is load-bearing.

    Asserted structurally rather than by timing: a timing assertion would have to build
    the anti-pattern to compare against, take seconds, and could fail on a busy runner.
    """
    annotation = getattr(wire, alias)
    union = get_args(annotation)[0]  # unwrap Annotated
    assert get_origin(union) is Union
    depths = [_depth(member) for member in get_args(union)]
    assert depths == sorted(depths) and len(set(depths)) == len(depths), depths


def test_schema_covers_every_key_core_reads():
    missing = wire.CORE_INPUT_KEYS - wire.declared_wire_keys()
    assert not missing, f'Twinkle_Core reads these keys but the wire schema drops them: {sorted(missing)}'


def test_vlm_field_set_matches_the_processor():
    """Kept as a test, not an import, so this module stays free of Twinkle_Core's deps.

    A field added to ``VLM_CONCAT_FIELDS`` and not here would be silently absent from
    the wire while the batching code still expects it.
    """
    assert wire.VLM_TENSOR_FIELDS == frozenset(InputProcessor.VLM_CONCAT_FIELDS)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
