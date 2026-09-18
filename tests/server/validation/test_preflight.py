# Copyright (c) ModelScope Contributors. All rights reserved.
"""Preflight: everything decidable about a request before it is enqueued.

The property under test is not "an error is returned" but *where* it is returned. A
rejection that happens inside the queued task has already written a future record and
fanned the call out to every data-parallel rank; a rejection in preflight has done
neither. Each test therefore asserts on the side effects (future records, backend calls)
as well as the status code.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.lifecycle.submit import backend_kwargs, run_submit
from twinkle.server.validation import BackendCapability, EndpointUnavailableError, assert_request_supported
from twinkle.server.validation.backend_compat import resolve_backend
from twinkle_client.types import model as model_types


class _Deployment:
    """A deployment stub that records what the request managed to reach."""

    def __init__(self, backend: str = 'transformers'):
        self.backend = backend
        self.data_world_size = 1
        self.futures: dict[str, dict] = {}
        self.backend_calls: list = []
        self.claimed: list = []
        self._task_queue_config = SimpleNamespace(effective_execution_timeout=60.0)
        self.state = SimpleNamespace(
            claim_seq=self._claim_seq,
            get_future=self._get_future,
            release_seq=self._release_seq,
        )

    async def _claim_seq(self, dedup_key, request_id, ttl):
        self.claimed.append(dedup_key)
        return None

    async def _get_future(self, request_id):
        return self.futures.get(request_id)

    async def _release_seq(self, dedup_key):
        self.claimed.remove(dedup_key)

    async def _on_request_start(self, request):
        return 'token'

    async def submit_and_peek(self, task, *, request_id, **kwargs):
        self.futures[request_id] = {'status': 'queued'}
        return await task()


def _request():
    return SimpleNamespace(state=SimpleNamespace(session_id='sess', request_id='rq'))


async def _call(self, body, adapter_name, token):
    self.backend_calls.append(type(body).__name__)
    return {'ok': True}


# --------------------------------------------------------------------------- #
# Backend resolution
# --------------------------------------------------------------------------- #


def test_backend_is_read_from_the_deployment_not_guessed():
    assert resolve_backend(_Deployment('megatron')) == 'megatron'
    # A sampler deployment has no backend concept at all, and must not be invented.
    assert resolve_backend(SimpleNamespace()) is None


# --------------------------------------------------------------------------- #
# Endpoint capability
# --------------------------------------------------------------------------- #


def test_megatron_rejects_the_split_gradient_endpoints_with_501():
    for capability in (BackendCapability.Forward, BackendCapability.Backward, BackendCapability.CalculateLoss):
        with pytest.raises(EndpointUnavailableError) as raised:
            assert_request_supported(
                _Deployment('megatron'), model_types.AdapterRequest(adapter_name='a'), capability=capability)
        assert raised.value.error_code == 501
        assert 'forward_backward' in str(raised.value), 'the alternative endpoints must be named'


def test_transformers_serves_the_split_gradient_endpoints():
    assert_request_supported(
        _Deployment('transformers'),
        model_types.AdapterRequest(adapter_name='a'),
        capability=BackendCapability.Forward)


def test_mock_backend_serves_every_endpoint():
    """The mock is a test double; restricting it would only break tests."""
    assert_request_supported(
        _Deployment('mock'), model_types.AdapterRequest(adapter_name='a'), capability=BackendCapability.Forward)


@pytest.mark.asyncio
async def test_capability_rejection_writes_no_future_and_calls_no_backend():
    deployment = _Deployment('megatron')
    with pytest.raises(EndpointUnavailableError):
        await run_submit(
            deployment,
            _request(),
            model_types.AdapterRequest(adapter_name='a'),
            task_type='backward',
            backend_call=_call,
            capability=BackendCapability.Backward)
    assert deployment.futures == {}, 'a rejected request must leave no future record'
    assert deployment.backend_calls == [], 'the backend must not run for a rejected request'
    assert deployment.claimed == [], 'no seq claim may be taken before preflight passes'


# --------------------------------------------------------------------------- #
# Backend-only parameters
# --------------------------------------------------------------------------- #


def test_a_megatron_only_parameter_is_rejected_on_transformers():
    with pytest.raises(RequestRejectedError) as raised:
        assert_request_supported(_Deployment('transformers'), model_types.SaveRequest(adapter_name='a', merge_lora=True))
    assert raised.value.error_code == 422
    assert 'merge_lora' in str(raised.value)
    assert 'megatron' in str(raised.value)


def test_a_transformers_only_parameter_is_rejected_on_megatron():
    with pytest.raises(RequestRejectedError):
        assert_request_supported(
            _Deployment('megatron'), model_types.LoadRequest(adapter_name='a', name='ckpt', strict=True))


def test_an_unset_backend_only_parameter_is_not_rejected():
    """Only a *sent* value is checked.

    This is why every restricted field is ``Optional[...] = None``: had one carried its
    backend's own default, it would look sent on every request and the other half of the
    fleet would reject everything.
    """
    assert_request_supported(_Deployment('transformers'), model_types.SaveRequest(adapter_name='a'))
    assert_request_supported(_Deployment('megatron'), model_types.LoadRequest(adapter_name='a', name='ckpt'))


def test_every_restricted_field_is_optional_with_a_none_default():
    from twinkle_client.types.base import FieldRole, fields_with_role, read_backend_only
    offenders = []
    for model_cls in vars(model_types).values():
        if not isinstance(model_cls, type) or not hasattr(model_cls, 'model_fields'):
            continue
        for name, info in fields_with_role(model_cls, FieldRole.BackendKwarg).items():
            if read_backend_only(info) and info.get_default() is not None:
                offenders.append(f'{model_cls.__name__}.{name}')
    assert not offenders, ('these backend-restricted fields carry a non-None default, so "non-None means wrongly '
                          f'targeted" would reject every request on the other backend: {offenders}')


# --------------------------------------------------------------------------- #
# Passthrough keys are forwarded, not judged
# --------------------------------------------------------------------------- #


def test_a_real_parameter_read_from_kwargs_is_not_rejected():
    """The case that removed the passthrough spelling check.

    ``InputProcessor`` declares ``padding_free`` and reads ``padding_side`` via
    ``kwargs.get`` -- both are real parameters, and ``inspect.signature`` only sees the
    first. A similarity check scored them 0.75 and rejected
    ``set_processor('InputProcessor', padding_side='right')``, a call used throughout the
    cookbook and the E2E suite. No threshold fixes that: it also has to catch
    ``bate`` -> ``beta`` at 0.5. Rejecting valid requests is worse than missing a typo, so
    passthrough contents are forwarded unjudged.
    """
    assert_request_supported(
        _Deployment(),
        model_types.SetProcessorRequest(
            processor_cls='InputProcessor', adapter_name='a', init_kwargs={'padding_side': 'right'}))


def test_an_unrecognised_plugin_argument_is_forwarded():
    """Plugins accept ``**kwargs``, so "unknown" cannot mean "wrong"."""
    assert_request_supported(
        _Deployment(),
        model_types.SetLossRequest(loss_cls='DPOLoss', adapter_name='a', init_kwargs={'my_custom_knob': 1}))


def test_no_plugin_download_happens_during_validation(monkeypatch):
    """A validation path must have no side effects, and resolving a remote id downloads."""
    import twinkle.utils.loader as loader
    monkeypatch.setattr(loader.Plugin, 'load_plugin',
                        lambda *a, **k: pytest.fail('validation must not download a plugin'))
    assert_request_supported(
        _Deployment(),
        model_types.SetLossRequest(loss_cls='ms://someone/MyLoss', adapter_name='a', init_kwargs={'beta': 0.1}))


# --------------------------------------------------------------------------- #
# Forwarding
# --------------------------------------------------------------------------- #


def test_control_fields_are_never_forwarded_to_the_backend():
    """``inputs`` / ``adapter_name`` / ``seq_id`` are already passed explicitly.

    Forwarding them again would duplicate a keyword argument, or leak a protocol field
    into a backend signature.
    """
    body = model_types.ForwardBackwardTaskRequest(
        inputs=[{'input_ids': [1, 2]}], adapter_name='a', seq_id=3, task='embedding')
    assert backend_kwargs(body) == {'task': 'embedding'}


def test_unset_backend_parameters_are_not_forwarded():
    body = model_types.ForwardRequest(inputs=[{'input_ids': [1]}], adapter_name='a')
    assert backend_kwargs(body) == {}


def test_passthrough_contents_are_flattened():
    body = model_types.ForwardBackwardTaskRequest(
        inputs=[{'input_ids': [1]}], adapter_name='a', loss_kwargs={'advantages': [0.5]})
    assert backend_kwargs(body) == {'advantages': [0.5]}


def test_a_passthrough_key_shadowing_a_declared_parameter_is_an_error():
    """Silently letting one win would make the effective value depend on merge order."""
    body = model_types.ForwardRequest(inputs=[{'input_ids': [1]}], adapter_name='a', task='causal_lm',
                                      loss_kwargs={'task': 'embedding'})
    with pytest.raises(ValueError, match='collides'):
        backend_kwargs(body)


def test_plugin_identifiers_are_not_forwarded_as_kwargs():
    """The handler passes ``loss_cls`` positionally; forwarding it too would duplicate it."""
    body = model_types.SetLossRequest(loss_cls='DPOLoss', adapter_name='a', init_kwargs={'beta': 0.1})
    assert backend_kwargs(body) == {'beta': 0.1}


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
