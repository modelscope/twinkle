# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for ``twinkle.data_format.tq_fields``.

The packing helpers carry field-consistency validation and a numeric/non-numeric
type branch but previously had zero test coverage. These cover the three branches
required by (empty rows, inconsistent fields, numeric+non-numeric mix), plus a
subprocess assertion (not depending on the ``async-rl`` extra) that guards: the
module must import cleanly without ``tensordict``.
"""
import subprocess
import sys

import pytest

tensordict = pytest.importorskip('tensordict')


def test_rows_to_tq_fields_empty_rows():
    from twinkle.data_format import rows_to_tq_fields

    packed = rows_to_tq_fields([])
    assert packed.batch_size[0] == 0


def test_rows_to_tq_fields_rejects_inconsistent_fields():
    """Rows with differing key sets must raise, not silently pack a ragged TensorDict."""
    from twinkle.data_format import rows_to_tq_fields

    with pytest.raises(ValueError):
        rows_to_tq_fields([{'input_ids': [1]}, {'input_ids': [2], 'labels': [3]}])


def test_columns_to_tq_fields_mixes_numeric_and_non_numeric():
    """Numeric columns go through ``torch.tensor``; the rest through ``NonTensorStack``."""
    import torch

    from twinkle.data_format import columns_to_tq_fields

    packed = columns_to_tq_fields({'scores': [1, 2], 'names': ['a', 'b']}, 2)
    assert packed.batch_size[0] == 2
    assert isinstance(packed['scores'], torch.Tensor)
    assert list(packed['names']) == ['a', 'b']


def test_tq_fields_imports_without_tensordict():
    """The module must import cleanly without the async-rl extra installed.

    Subprocess with ``tensordict`` blocked from ``sys.modules``, asserting that importing
    the module (and reading the constants) does not touch it -- function-level imports are
    what make that true, so hoisting them would break this.
    """
    code = (
        'import sys;'
        "sys.modules['tensordict'] = None;"
        'from twinkle.data_format import tq_fields, ROLLOUT_TRAIN_FIELDS;'
        "assert 'input_ids' in ROLLOUT_TRAIN_FIELDS"
    )
    subprocess.run([sys.executable, '-c', code], check=True)
