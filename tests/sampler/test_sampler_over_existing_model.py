# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampling from a model that already exists, instead of loading a second one.

A ``TransformersSampler`` normally loads its own weights from ``model_id``. During training that is the
wrong thing to do twice over: the weights are already on the device, and a second copy competes with the
gradients and optimizer state for the same memory. Passing ``model=<TransformersModel>`` makes the
sampler a facade that generates inside the model's workers instead.

What the tests below pin is the seam, not generation itself: that the facade forwards rather than
reimplements, and that the methods which would be lying in facade mode refuse instead. Whether tokens
come out correctly is ``TransformersEngine``'s business and is covered where that is tested.
"""
import pytest


class FakeModel:
    """Stands in for a ``TransformersModel`` handle -- only the methods the facade reaches for."""

    def __init__(self):
        self.calls = []

    def generate(self, inputs, **kwargs):
        self.calls.append(('generate', inputs, kwargs))
        return ['response']

    def generate_stream(self, inputs, **kwargs):
        self.calls.append(('generate_stream', inputs, kwargs))
        yield ('hello', None)
        yield ('', 'stop')

    def set_template(self, template_cls, **kwargs):
        self.calls.append(('set_template', template_cls, kwargs))


def _facade(model=None):
    from twinkle.sampler.transformers_sampler import TransformersSampler
    return TransformersSampler(model=model or FakeModel())


def test_a_sampler_needs_either_a_model_id_or_a_model():
    """Neither is unusable and both is ambiguous, so both are refused at construction."""
    from twinkle.sampler.transformers_sampler import TransformersSampler
    with pytest.raises(ValueError, match='not both and not neither'):
        TransformersSampler()
    with pytest.raises(ValueError, match='not both and not neither'):
        TransformersSampler('Qwen/Qwen3-0.6B', model=FakeModel())


def test_sampling_is_forwarded_to_the_model_whole():
    """Every argument reaches the model.

    The point of forwarding rather than half-encoding here: adapter grouping, chunking and error
    tolerance all live on the model's side, and splitting them across the two would mean two places to
    fix when either changes.
    """
    sampler = _facade()
    result = sampler.sample(
        ['prompt'], sampling_params={'max_tokens': 8}, adapter_name='a', adapter_path='/tmp/a', strict=False)

    assert result == ['response']
    (kind, inputs, kwargs), = sampler.model.calls
    assert kind == 'generate' and inputs == ['prompt']
    assert kwargs['sampling_params'] == {'max_tokens': 8}
    assert kwargs['adapter_name'] == 'a' and kwargs['adapter_path'] == '/tmp/a'
    assert kwargs['strict'] is False


def test_streaming_is_forwarded_and_stays_a_generator():
    sampler = _facade()
    assert list(sampler.sample_stream('prompt')) == [('hello', None), ('', 'stop')]
    assert sampler.model.calls[0][0] == 'generate_stream'


def test_a_facade_builds_no_engine_of_its_own():
    """The whole point: no second copy of the weights."""
    assert _facade().engine is None


@pytest.mark.parametrize('method, args', [('sleep', ()), ('wake_up', ()), ('get_state_keys', ())])
def test_memory_controls_refuse_rather_than_pretend(method, args):
    """A caller freeing the device has to be told this sampler cannot do that.

    Silently succeeding would be worse than failing: the device would still be occupied by the model,
    and the caller would believe otherwise.
    """
    sampler = _facade()
    with pytest.raises(RuntimeError, match='no engine here to control'):
        getattr(sampler, method)(*args)


def test_the_template_is_set_on_the_model():
    """Encoding happens where generation happens, so the template has to go there.

    It also cannot be built here: the base implementation defaults ``model_id`` off this object, which
    for a facade is None.
    """
    sampler = _facade()
    sampler.set_template('qwen')
    assert sampler.model.calls == [('set_template', 'qwen', {})]


def test_megatron_refuses_to_generate_in_place():
    """Refused, not approximated: TP/PP-sharded weights under Megatron names are not what an engine reads."""
    from twinkle.model.megatron.megatron import MegatronModel
    model = object.__new__(MegatronModel)
    with pytest.raises(NotImplementedError, match='sharded across TP/PP'):
        model.generate(['prompt'])
    with pytest.raises(NotImplementedError, match='sharded across TP/PP'):
        model.generate_stream(['prompt'])
