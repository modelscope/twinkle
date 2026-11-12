# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side wire shape of the async / data-plane component calls.

The recorded body is the *serialized request model*, not a hand-built dict: every
twinkle-native client method now instantiates its endpoint's model and posts one
``model_dump_json``. Asserting on the parsed JSON therefore checks the real wire shape,
including the fact that a caller's undeclared arguments land in the model's passthrough
region rather than at the top level.
"""
from __future__ import annotations

import asyncio
import json

from twinkle.protocol.types import DataRef


class _Response:

    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


def _completed(result):
    """Wrap a business result in a completed Task_Envelope (the new wire shape)."""
    return {'request_id': 'req-test', 'status': 'completed', 'result': result}


def _recorder(calls, result_factory):
    """A Session.post stand-in that records the URL and decoded JSON body."""

    def post(url, headers=None, data=None, timeout=None, **kwargs):
        body = json.loads(data) if data else kwargs.get('json') or {}
        calls.append((url, body))
        if '/model/' in url and url.endswith('/create'):
            return _Response({'training_handles': True})
        if url.endswith('/add_adapter_to_model'):
            return _Response(_completed({'training_id': 'training-test', 'adapter_name': body['adapter_name']}))
        return _Response(result_factory(url))

    return post


def _patch_transport(monkeypatch, calls, result_factory):
    from twinkle_client.http import ClientContext, ClientTransport
    from twinkle_client.http.context import set_default_transport

    transport = ClientTransport(ClientContext(base_url='http://server', api_key='test-key'))
    monkeypatch.setattr(transport._session, 'post', _recorder(calls, result_factory))
    set_default_transport(transport)


def test_model_forward_backward_sends_multiple_data_refs(monkeypatch) -> None:
    from twinkle_client.model import multi_lora_transformers as module

    calls: list = []
    _patch_transport(monkeypatch, calls, lambda url: {}
                     if url.endswith('/create') else _completed({'result': {
                         'loss': 1.0
                     }}))

    model = module.MultiLoraTransformersModel('ms://base')
    model.add_adapter_to_model('adapter', None)
    refs = [
        DataRef(ref_id='data-1', size=2, fields=['train_input']),
        DataRef(ref_id='data-2', size=2, fields=['train_input']),
    ]
    model.forward_backward_from_data_plane(
        refs,
        input_field='train_input',
        kwarg_fields={'advantages': 'advantage'},
    )

    url, body = calls[-1]
    assert url.endswith('/model/base/twinkle/forward_backward_from_data_plane')
    assert body['input_refs'] == [ref.model_dump() for ref in refs]
    assert body['input_field'] == 'train_input'
    assert body['kwarg_fields'] == {'advantages': 'advantage'}
    assert body['adapter_name'] == 'adapter'


def test_model_inline_forward_methods_keep_the_original_endpoints(monkeypatch) -> None:
    from twinkle_client.model import multi_lora_transformers as module

    calls: list = []
    _patch_transport(monkeypatch, calls, lambda url: {} if url.endswith('/create') else _completed({'result': {}}))

    model = module.MultiLoraTransformersModel('ms://base')
    model.add_adapter_to_model('adapter', None)
    inputs = [{'input_ids': [1, 2]}]
    model.forward(inputs, return_logits=True)
    model.forward_only(inputs, disable_lora=True)
    model.forward_backward(inputs, micro_batch_size=1)

    assert [url.rsplit('/', 1)[-1] for url, _ in calls[-3:]] == [
        'forward',
        'forward_only',
        'forward_backward',
    ]
    assert all(body['inputs'] == inputs for _, body in calls[-3:])
    assert all('input_refs' not in body for _, body in calls[-3:])
    # Declared backend parameters stay top level; ``exclude_none`` keeps the rest off.
    assert calls[-3][1]['return_logits'] is True
    assert calls[-2][1]['disable_lora'] is True
    assert calls[-1][1]['micro_batch_size'] == 1


def test_model_add_adapter_serializes_dict_lora_config(monkeypatch) -> None:
    """Dict input remains compatible with the strict serialized LoRA wire field."""
    from twinkle_client.model import multi_lora_transformers as module

    calls: list = []
    _patch_transport(monkeypatch, calls,
                     lambda url: {} if url.endswith('/create') else _completed({'status': 'ok'}))

    model = module.MultiLoraTransformersModel('ms://base')
    model.add_adapter_to_model('adapter', {'r': 4, 'target_modules': 'all-linear'})

    url, body = calls[-1]
    assert url.endswith('/model/base/twinkle/add_adapter_to_model')
    assert isinstance(body['config'], str)
    assert 'LoraConfig' in body['config']
    assert model.adapter_name == 'adapter'


def test_undeclared_forward_arguments_are_routed_to_loss_kwargs(monkeypatch) -> None:
    """A loss input is not a declared field, so it travels in the passthrough region.

    The public signature is unchanged -- callers still pass ``advantages=...`` -- which is
    what lets the body be strict without breaking existing scripts.
    """
    from twinkle_client.model import multi_lora_transformers as module

    calls: list = []
    _patch_transport(monkeypatch, calls, lambda url: {} if url.endswith('/create') else _completed({'result': {}}))

    model = module.MultiLoraTransformersModel('ms://base')
    model.add_adapter_to_model('adapter', None)
    model.forward_backward([{'input_ids': [1, 2]}], advantages=[0.5], old_logps=[[-1.0, -2.0]])

    _, body = calls[-1]
    assert body['loss_kwargs'] == {'advantages': [0.5], 'old_logps': [[-1.0, -2.0]]}
    assert 'advantages' not in body


def test_model_data_plane_forward_uses_a_separate_api(monkeypatch) -> None:
    from twinkle_client.model import multi_lora_transformers as module

    calls: list = []
    _patch_transport(monkeypatch, calls, lambda url: {}
                     if url.endswith('/create') else _completed({'result': {
                         'value': 1
                     }}))

    model = module.MultiLoraTransformersModel('ms://base')
    model.add_adapter_to_model('adapter', None)
    ref = DataRef(ref_id='data-1', size=2, fields=['train_input'])
    model.forward_from_data_plane(ref, input_field='train_input')

    url, body = calls[-1]
    assert url.endswith('/model/base/twinkle/forward_from_data_plane')
    assert body['input_refs'] == [ref.model_dump()]
    assert body['input_field'] == 'train_input'


def test_model_data_plane_forward_only_can_append_selected_outputs(monkeypatch) -> None:
    from twinkle_client.model import multi_lora_transformers as module

    ref = DataRef(ref_id='data-1', size=2, fields=['input_ids'])
    updated_ref = ref.model_copy(update={'fields': ['input_ids', 'ref_logps']})
    calls: list = []
    # The handler wraps its payload as ``{'result': ...}``, so the stub must too --
    # otherwise the test asserts against a reply shape the server never sends.
    _patch_transport(monkeypatch, calls, lambda url: {}
                     if url.endswith('/create') else _completed({'result': updated_ref.model_dump()}))

    model = module.MultiLoraTransformersModel('ms://base')
    model.add_adapter_to_model('adapter', None)
    result = model.forward_only_from_data_plane(
        ref,
        output_ref=ref,
        output_fields={'logps': 'ref_logps'},
        disable_lora=True,
    )

    url, body = calls[-1]
    assert url.endswith('/model/base/twinkle/forward_only_from_data_plane')
    assert body['input_refs'] == [ref.model_dump()]
    assert body['output_ref'] == ref.model_dump()
    assert result == updated_ref


def test_sampler_async_data_plane_path_returns_reference_without_materializing(monkeypatch) -> None:
    from twinkle_client.sampler import vllm_sampler as module

    output_ref = DataRef(
        ref_id='rollout-1',
        size=4,
        fields=['train_input', 'sampled_logprobs', 'decoded'],
        kind='rollout',
    )
    calls: list = []
    _patch_transport(monkeypatch, calls, lambda url: {}
                     if url.endswith('/create') else _completed(output_ref.model_dump()))

    sampler = module.vLLMSampler('ms://base')

    result = asyncio.run(sampler.asample_to_data_plane(
        [{
            'input_ids': [1]
        }],
        num_samples=4,
        group_ids=['group-1'],
    ))

    assert result == output_ref
    url, body = calls[-1]
    assert url.endswith('/sampler/base/twinkle/sample_to_data_plane')
    assert body['num_samples'] == 4
    assert body['group_ids'] == ['group-1']


def test_old_remove_completion_does_not_clear_new_same_name_binding():
    import threading
    from twinkle_client.http import ClientContext, ClientTransport
    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    entered, resume = threading.Event(), threading.Event()
    class Transport(ClientTransport):
        def __init__(self):
            super().__init__(ClientContext(base_url='http://server', api_key='owner'))
            self.created = 0
            self.operations = []
        def post(self, url, json_data):
            return _Response({'training_handles': True})
        def post_model(self, url, body):
            self.operations.append((url.rsplit('/', 1)[-1], body.model_dump()))
            if url.endswith('/add_adapter_to_model'):
                self.created += 1
                return _Response(_completed({'training_id': f'training-{self.created}', 'adapter_name': 'same'}))
            if url.endswith('/remove_adapter'):
                if not entered.is_set():
                    entered.set()
                    assert resume.wait(3)
            return _Response(_completed(None))
    transport = Transport()
    model = MultiLoraTransformersModel('small', transport=transport)
    model.add_adapter_to_model('same', None)
    errors = []
    def remove():
        try:
            model.remove_adapter()
        except Exception as exc:
            errors.append(exc)
    releasing = threading.Thread(target=remove)
    releasing.start()
    try:
        assert entered.wait(3)
        model.remove_adapter('same')
        model.add_adapter_to_model('same', None)
    finally:
        resume.set()
        releasing.join(3)
    assert not releasing.is_alive() and not errors
    model.zero_grad()
    assert transport.operations[-1][1]['training_id'] == 'training-2'
    removal = next(body for endpoint, body in transport.operations if endpoint == 'remove_adapter')
    assert removal['training_id'] == 'training-1'


def test_duplicate_adapter_is_rejected_before_rpc_and_release_allows_recreate(monkeypatch):
    import pytest
    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    calls = []
    _patch_transport(monkeypatch, calls, lambda url: _completed(None))
    first = MultiLoraTransformersModel('base')
    first.add_adapter_to_model('same', None)
    count = len(calls)
    with pytest.raises(RuntimeError, match='already'):
        first.add_adapter_to_model('same', None)
    assert len(calls) == count
    second = MultiLoraTransformersModel('base')
    second.add_adapter_to_model('same', None)
    first.remove_adapter('same')
    first.add_adapter_to_model('same', None)


def test_concurrent_duplicate_and_failed_creation_leave_reusable_name(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import pytest
    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    calls = []
    _patch_transport(monkeypatch, calls, lambda url: _completed(None))
    model = MultiLoraTransformersModel('base')
    original = model._submit
    entered, resume = threading.Event(), threading.Event()

    def fail(*args, **kwargs):
        entered.set()
        assert resume.wait(3)
        raise RuntimeError('create failed')

    monkeypatch.setattr(model, '_submit', fail)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(model.add_adapter_to_model, 'same', None)
        try:
            assert entered.wait(3)
            with pytest.raises(RuntimeError, match='already being created'):
                model.add_adapter_to_model('same', None)
        finally:
            resume.set()
        with pytest.raises(RuntimeError, match='create failed'):
            future.result()
    monkeypatch.setattr(model, '_submit', original)
    model.add_adapter_to_model('same', None)
    assert model._training_ids['same'] == 'training-test'


def test_unknown_training_rejected_before_handshake(monkeypatch):
    import pytest
    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    calls = []
    _patch_transport(monkeypatch, calls, lambda url: _completed({}))
    model = MultiLoraTransformersModel('base')
    with pytest.raises(RuntimeError, match='add_adapter_to_model'):
        model.step()
    assert not calls


def test_expired_handle_can_be_removed_and_recreated_on_same_object(monkeypatch):
    import pytest
    from twinkle_client.exceptions import TwinkleHTTPError
    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    calls = []
    _patch_transport(monkeypatch, calls, lambda url: _completed(None))
    model = MultiLoraTransformersModel('base')
    model.add_adapter_to_model('same', None)
    submit = model._submit

    def expired(*args, **kwargs):
        raise TwinkleHTTPError('Training handle expired', status_code=404, error_code=404)

    monkeypatch.setattr(model, '_submit', expired)
    with pytest.raises(TwinkleHTTPError) as error:
        model.remove_adapter('same')
    assert error.value.status_code == 404
    monkeypatch.setattr(model, '_submit', submit)
    model.add_adapter_to_model('same', None)
    model.zero_grad()
    assert calls[-1][1]['training_id'] == model._training_ids['same']


def test_failed_release_keeps_handle_when_outcome_is_unknown(monkeypatch):
    import pytest
    from twinkle_client.exceptions import TwinkleHTTPError
    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    calls = []
    _patch_transport(monkeypatch, calls, lambda url: _completed(None))
    model = MultiLoraTransformersModel('base')
    model.add_adapter_to_model('same', None)
    identity = model._training_ids['same']
    for error in (TwinkleHTTPError('Unavailable', status_code=503), TimeoutError('ACK unknown')):
        def failed(*args, **kwargs):
            raise error
        monkeypatch.setattr(model, '_submit', failed)
        with pytest.raises(type(error)):
            model.remove_adapter('same')
        assert model._training_ids['same'] == identity
        assert model.adapter_name == 'same'


def test_sequence_and_handle_selection_are_object_local(monkeypatch):
    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    calls = []
    _patch_transport(monkeypatch, calls, lambda url: _completed({}))
    first, second = MultiLoraTransformersModel('base'), MultiLoraTransformersModel('base')
    for model in (first, second):
        model._created = True
        model.adapter_name = 'default'
    first._training_ids = {'default': 'first-default', 'other': 'first-other'}
    second._training_ids = {'default': 'second-default'}
    first.step()
    first.step(adapter_name='other')
    second.step()
    assert [(body['adapter_name'], body['training_id'], body['seq_id']) for _, body in calls] == [
        ('default', 'first-default', 1), ('other', 'first-other', 2), ('default', 'second-default', 1)]
