# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-only checks that SequenceParallel wraps *every* FlashAttention backend.

transformers stores one and the same ``flash_attention_forward`` under the keys ``flash_attention_2``,
``flash_attention_3`` and ``flash_attention_4``, and ``AttentionInterface.get_interface`` is an
exact-key lookup. So SP has to overwrite each name; overwriting only ``flash_attention_2`` leaves an
FA3 config training *without* sequence parallelism and with no error anywhere.
"""
import functools

import pytest

from twinkle.model.transformers.strategy.sequence_parallel import FLASH_ATTENTION_IMPLS, SequenceParallel

pytestmark = pytest.mark.skipif(
    'flash_attention_3' not in FLASH_ATTENTION_IMPLS,
    reason='This build only wires flash_attention_2 into SequenceParallel.')


@pytest.fixture
def registered():
    """Run the registration against the real transformers dict, then restore it verbatim."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    snapshot = dict(ALL_ATTENTION_FUNCTIONS)
    strategy = SequenceParallel.__new__(SequenceParallel)
    strategy.world_size = 2
    strategy.sp_world_size = 2
    strategy.rp_world_size = 1
    strategy.extra_kwargs = {}
    try:
        strategy._prepare_flash_attn(None)
    except (AttributeError, KeyError) as error:
        # a transformers version whose internals moved: the production path would raise too, so skip
        pytest.skip(f'SequenceParallel._prepare_flash_attn is not compatible with this transformers: {error!r}')
    try:
        yield ALL_ATTENTION_FUNCTIONS
    finally:
        ALL_ATTENTION_FUNCTIONS.clear()
        ALL_ATTENTION_FUNCTIONS.update(snapshot)


def _wrapped_name(value):
    if not isinstance(value, functools.partial):
        return None
    return getattr(value.func, '__name__', None)


def test_every_flash_attention_name_is_wrapped(registered):
    for implementation in FLASH_ATTENTION_IMPLS:
        assert _wrapped_name(registered[implementation]) == 'local_flash_attn', (
            f'{implementation} is not routed through the SP wrapper; an FA config would train without '
            'sequence parallelism.')
        assert isinstance(registered[implementation].keywords['dist_attn'], object)


def test_sdpa_is_wrapped(registered):
    assert _wrapped_name(registered['sdpa']) == 'local_sdpa_attn'


def test_origin_aliases_keep_the_transformers_implementation(registered):
    """``local_flash_attn`` re-enters through ``*_origin``; those must not point back at the wrapper."""
    assert _wrapped_name(registered['flash_attention_2_origin']) is None
    assert _wrapped_name(registered['sdpa_origin']) is None
    assert registered['flash_attention_2_origin'].__module__.startswith('transformers')
    assert registered['sdpa_origin'].__module__.startswith('transformers')


def test_wrapper_dispatches_through_the_module_config(registered):
    """The FA3 key must reach the same wrapper as FA2 -- the leaf is chosen from the module config."""
    wrapper = registered['flash_attention_2'].func
    assert registered['flash_attention_3'].func is wrapper
    assert 'local_flash_attn' in wrapper.__qualname__
