# Copyright (c) ModelScope Contributors. All rights reserved.
"""Characterization tests for the shared ``to_backend_inputs`` seam (F003 / P003).

Pins the input-shape rules that the sampler handlers used to re-implement inline,
so the three call sites (``sample`` / ``sample_to_data_plane`` batch form, and
``sample_stream`` single form) now share one definition.
"""
from __future__ import annotations

import pytest

from twinkle.data_format import InputFeature, Trajectory
from twinkle.server.lifecycle.submit import to_backend_inputs


@pytest.mark.parametrize('single', [False, True])
def test_wire_media_restores_tensor_shape_for_swift_collation(single):
    import torch
    from swift.dev.template import Template

    from twinkle.protocol.types.data import WireInputFeature

    wire = WireInputFeature(input_ids=[1, 2], pixel_values=[[0.5, 0.25], [0.1, 0.2]],
                            image_grid_thw=[[1, 1, 2]], target_probs=[[0.2, 0.8]])
    output = to_backend_inputs([wire], single=single)
    row = output if single else output[0]
    media = Template._data_collator_mm_data(object.__new__(Template), [row])
    assert media['pixel_values'].shape == (2, 2)
    assert media['image_grid_thw'].shape == (1, 3)
    assert media['pixel_values'].dtype == torch.float32
    assert media['image_grid_thw'].dtype == torch.int64
    assert row['target_probs'] == [[0.2, 0.8]]
    assert wire.pixel_values == [[0.5, 0.25], [0.1, 0.2]]

_IF = {'input_ids': [1, 2, 3]}
_TRAJ = {'messages': [{'role': 'user', 'content': 'hi'}]}


def test_batch_list_of_input_features():
    # Each element is parsed with InputFeature (dict with input_ids).
    assert to_backend_inputs([_IF, _IF]) == [InputFeature(**_IF), InputFeature(**_IF)]


def test_batch_list_of_trajectories():
    # A dict without input_ids is parsed as a Trajectory.
    assert to_backend_inputs([_TRAJ]) == [Trajectory(**_TRAJ)]


def test_batch_single_dict_becomes_one_element_list():
    assert to_backend_inputs(_IF) == [InputFeature(**_IF)]
    assert to_backend_inputs(_TRAJ) == [Trajectory(**_TRAJ)]


def test_batch_passthrough_for_non_list_non_dict():
    sentinel = object()
    assert to_backend_inputs(sentinel) is sentinel


def test_single_returns_one_object_not_a_list():
    out = to_backend_inputs([_IF], single=True)
    assert not isinstance(out, list)
    assert out == InputFeature(**_IF)
    assert to_backend_inputs(_TRAJ, single=True) == Trajectory(**_TRAJ)


def test_single_rejects_multi_element_list():
    with pytest.raises(ValueError, match='single input'):
        to_backend_inputs([_IF, _IF], single=True)
