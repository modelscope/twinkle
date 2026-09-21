# Copyright (c) ModelScope Contributors. All rights reserved.
"""Level 1: what a malformed request body looks like on the wire.

Driven through FastAPI's ``TestClient`` against the real route table, so what is asserted
is the response a client actually receives -- not a schema call in isolation. A rejection
here happens during body parsing, before any handler runs, which is what makes it free of
queue and backend side effects.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fastapi.exceptions import RequestValidationError

from twinkle.server.deployment import validation_error_handler
from twinkle.protocol.types import model as model_types
from twinkle.protocol.types.base import StrictRequest


@pytest.fixture(scope='module')
def client() -> TestClient:
    """An app carrying the real request models and the shared error handler.

    Handlers are stubs on purpose: the point is that a bad body never reaches one, so a
    stub that records nothing is the strongest possible witness -- if it is invoked, the
    check did not happen.
    """
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.post('/forward')
    async def forward(body: model_types.ForwardRequest):
        return {'reached_handler': True}

    @app.post('/set_loss')
    async def set_loss(body: model_types.SetLossRequest):
        return {'reached_handler': True}

    @app.post('/save')
    async def save(body: model_types.SaveRequest):
        return {'reached_handler': True}

    @app.post('/forward_backward')
    async def forward_backward(body: model_types.ForwardBackwardTaskRequest):
        return {'reached_handler': True}

    return TestClient(app)


def _post(client: TestClient, path: str, body: dict):
    return client.post(path, json=body)


def _valid_forward(**overrides) -> dict:
    return {'inputs': [{'input_ids': [1, 2, 3]}], 'adapter_name': 'a', **overrides}


def test_a_valid_body_reaches_the_handler(client):
    response = _post(client, '/forward', _valid_forward())
    assert response.status_code == 200
    assert response.json() == {'reached_handler': True}


def test_an_unknown_top_level_field_is_rejected(client):
    response = _post(client, '/forward', _valid_forward(adapter_nmae='typo'))
    assert response.status_code == 422
    body = response.json()
    assert body['category'] == 'user'
    assert body['error_code'] == 422
    assert 'adapter_nmae' in body['error']
    assert any(detail['field'] == 'adapter_nmae' for detail in body['details'])
    assert 'reached_handler' not in body


def test_the_error_names_the_client_version_mismatch(client):
    """An unknown top-level field is exactly what an outdated client looks like."""
    response = _post(client, '/forward', _valid_forward(advantages=[0.1]))
    assert response.status_code == 422
    assert 'upgrade' in response.json()['error'].lower()


def test_the_error_body_is_an_error_payload_not_fastapi_detail(client):
    """One error shape on the wire, or a client has to learn two."""
    body = _post(client, '/forward', _valid_forward(unknown=1)).json()
    assert set(body) >= {'error', 'category', 'error_code', 'request_id'}
    assert 'detail' not in body


def test_no_traceback_is_returned(client):
    """A rejected body is the caller's problem, not a crash to be dumped at them."""
    assert 'traceback' not in _post(client, '/forward', _valid_forward(unknown=1)).json()


def test_details_locate_the_field_inside_the_body(client):
    body = _post(client, '/forward', {'inputs': [{'input_ids': [1.5]}], 'adapter_name': 'a'}).json()
    assert body['error_code'] == 422
    assert any('inputs' in detail['path'] for detail in body['details'])


def test_a_token_in_the_body_is_rejected(client):
    """``token`` comes from the Authorization header only.

    Declaring it as a field would make a body-supplied token *legal* under
    ``extra='forbid'``, which is a credential-forgery path, not a convenience.
    """
    assert _post(client, '/forward', _valid_forward(token='stolen')).status_code == 422
    assert 'token' not in model_types.ForwardRequest.model_fields


def test_seq_id_is_still_accepted(client):
    """Strictness must not break the retry idempotency key.

    Were ``seq_id`` undeclared, ``extra='forbid'`` would reject every retried
    gradient-mutating call -- disabling the dedup that prevents a double-apply. It is
    declared on the gradient-mutating models only, which is where the client sends it.
    """
    assert _post(client, '/forward_backward', _valid_forward(seq_id=4)).status_code == 200
    assert 'seq_id' in model_types.ForwardBackwardTaskRequest.model_fields
    assert 'seq_id' in model_types.AdapterRequest.model_fields
    assert 'seq_id' in model_types.DataPlaneForwardRequest.model_fields


def test_a_dynamic_plugin_argument_is_accepted_inside_its_region(client):
    """Strictness at the top level, freedom inside the declared dict."""
    response = _post(client, '/set_loss', {
        'loss_cls': 'DPOLoss',
        'adapter_name': 'a',
        'init_kwargs': {'beta': 0.1, 'anything_at_all': [1, 2]},
    })
    assert response.status_code == 200


def test_a_non_json_value_fails_in_the_client_before_any_request():
    """``JsonValue`` earns its keep at Level 0, not Level 1.

    Anything that arrived as JSON is by definition a JSON value, so this annotation can
    only ever reject something in the caller's own process -- which is the useful place,
    because the caller still has the offending object and a stack trace pointing at it.
    """
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as raised:
        model_types.SetLossRequest(loss_cls='DPOLoss', adapter_name='a', init_kwargs={'beta': object()})
    assert 'init_kwargs' in str(raised.value), 'the error must name the field that holds the bad value'


def test_a_checkpoint_dict_without_its_key_is_a_422_not_a_500(client):
    """It used to surface as ``KeyError`` -> 500, blaming the server for a bad body."""
    response = _post(client, '/save', {'adapter_name': 'a', 'checkpoint_dir': {'wrong': 'shape'}})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Coverage: no twinkle-native route may keep a lax body
# --------------------------------------------------------------------------- #

# The one twinkle-native route with no request body at all.
_BODYLESS = {('model', 'GET /healthz')}


def test_every_twinkle_route_body_is_strict():
    """Enumerated from the live route table, not from a hardcoded count.

    A count would have to be updated by whoever adds a route -- exactly the person who
    would also forget the base class.
    """
    from fastapi.routing import APIRoute
    from tests.server.contract.client_api_harness import build_model_app, build_processor_app, build_sampler_app

    offenders = []
    for app_name, builder in (('model', build_model_app), ('sampler', build_sampler_app), ('processor',
                                                                                           build_processor_app)):
        for route in builder().routes:
            if not isinstance(route, APIRoute):
                continue
            for method in sorted(route.methods & {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}):
                key = f'{method} {route.path}'
                if not route.path.startswith(('/twinkle', '/healthz')) or (app_name, key) in _BODYLESS:
                    continue
                for field in route.dependant.body_params:
                    annotation = field.field_info.annotation
                    if not (isinstance(annotation, type) and issubclass(annotation, StrictRequest)):
                        offenders.append(f'{app_name} {key}: {getattr(annotation, "__name__", annotation)}')
    assert not offenders, ('these twinkle-native routes accept a body that is not a StrictRequest, so an unknown '
                           f'field reaches the backend instead of failing: {offenders}')


# --------------------------------------------------------------------------- #
# Regression: the sampler routes must bind the sampler-domain models, not model.py's
#
# ``sampler.py`` and ``model.py`` once both declared bare ``AddAdapterRequest`` /
# ``SetTemplateRequest``. Because the handler does ``import twinkle.protocol.types as
# types`` and the package ``__init__`` re-exported ``model.py`` first,
# ``types.AddAdapterRequest`` resolved to *model.py*'s model -- whose ``config`` is a
# ``str``, so it rejected the dict a real ``add_adapter_to_sampler`` call sends. The
# Sampler-prefixed names remove the collision; this pins the binding so it cannot
# silently regress.
# --------------------------------------------------------------------------- #


def _body_model(app, path: str):
    from fastapi.routing import APIRoute
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            (param, ) = route.dependant.body_params
            return param.field_info.annotation
    raise AssertionError(f'route {path} not found')


def test_sampler_routes_bind_sampler_domain_models():
    from tests.server.contract.client_api_harness import build_sampler_app
    from twinkle.protocol.types import sampler as sampler_types

    app = build_sampler_app()
    assert _body_model(app, '/twinkle/add_adapter_to_sampler') is sampler_types.SamplerAddAdapterRequest
    assert _body_model(app, '/twinkle/set_template') is sampler_types.SamplerSetTemplateRequest


def test_sampler_add_adapter_accepts_a_dict_config_where_model_rejects_it():
    """The exact divergence the collision hid: the client sends ``config`` as a dict."""
    from twinkle.protocol.types import model as model_types
    from twinkle.protocol.types import sampler as sampler_types

    # The sampler contract (``config: Any``) accepts the LoRA config dict the client sends.
    ok = sampler_types.SamplerAddAdapterRequest(adapter_name='a', config={'r': 8})
    assert ok.config == {'r': 8}
    # model.py's same-shaped-name model declares ``config: Optional[str]`` and would 422 it,
    # which is why the two must not share a bare class name.
    with pytest.raises(ValidationError):
        model_types.AddAdapterRequest(adapter_name='a', config={'r': 8})


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
