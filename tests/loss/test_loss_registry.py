# Copyright (c) ModelScope Contributors. All rights reserved.
"""Registry smoke tests: every advertised loss must import, construct and be re-exported."""
import inspect

import pytest

import twinkle.loss as loss_pkg
from twinkle.loss import torch_loss_mapping
from twinkle.loss.base import Loss

# Only losses without a usable default constructor need entries here. Adding a new loss with a
# required argument must fail ``test_construct_kwargs_cover_required_parameters`` until it is listed.
CONSTRUCT_KWARGS = {
    'ChunkedCrossEntropyLoss': {'chunk_size': 1024},
    'SeqClsLoss': {'problem_type': 'single_label_classification', 'num_labels': 2},
}


def _construct(cls):
    return cls(**CONSTRUCT_KWARGS.get(cls.__name__, {}))


def test_mapping_values_are_loss_subclasses():
    for name, cls in torch_loss_mapping.items():
        assert isinstance(cls, type), name
        assert issubclass(cls, Loss), name


def test_mapping_classes_are_reexported_from_the_package():
    for name, cls in torch_loss_mapping.items():
        assert getattr(loss_pkg, cls.__name__, None) is cls, name


@pytest.mark.parametrize('name,cls', list(torch_loss_mapping.items()))
def test_registered_loss_constructs(name, cls):
    try:
        _construct(cls)
    except ImportError as error:
        # e.g. the Liger-fused losses need ``liger_kernel`` installed before they can build.
        pytest.skip(f'optional dependency missing for {name}: {error}')
    except Exception as error:  # noqa: BLE001 - name the failing registry key
        pytest.fail(f'torch_loss_mapping[{name!r}] ({cls.__name__}) failed to construct: {error!r}')


def test_construct_kwargs_cover_required_parameters():
    for name, cls in torch_loss_mapping.items():
        provided = CONSTRUCT_KWARGS.get(cls.__name__, {})
        params = inspect.signature(cls.__init__).parameters
        required = [
            param.name for param in params.values()
            if param.name != 'self' and param.default is inspect.Parameter.empty
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        ]
        missing = [param for param in required if param not in provided]
        assert not missing, f'{cls.__name__} (registry key {name!r}) requires {missing}'


def test_flags_are_booleans():
    for cls in torch_loss_mapping.values():
        for attr in ('require_logits', 'require_logps', 'require_values', 'require_entropy'):
            assert isinstance(getattr(cls, attr), bool), f'{cls.__name__}.{attr}'
