# Copyright (c) ModelScope Contributors. All rights reserved.
"""Level 0: the client builds the request from the schema, in-process.

The property worth testing is that a bad call fails *without a network round trip*. Every
test here monkeypatches ``requests.post`` to fail loudly, so any test that passes has
proved no request was sent.
"""
from __future__ import annotations

import pytest
from pydantic import JsonValue, ValidationError
from typing import Dict

from twinkle_client._request_builder import build_request, request_json, to_wire_value
from twinkle_client.exceptions import TwinkleClientValidationError
from twinkle_client.types import model as model_types
from twinkle_client.types.base import StrictRequest, passthrough


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any HTTP call in this module is a bug in the code under test."""
    import twinkle_client.http.http_utils as http_utils
    monkeypatch.setattr(http_utils.requests, 'post',
                        lambda *a, **k: pytest.fail('a Level 0 failure must not produce a request'))


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_a_declared_name_goes_to_its_field():
    body = build_request(model_types.ForwardRequest, inputs=[{'input_ids': [1]}], adapter_name='a', task='embedding')
    assert body.task == 'embedding'
    assert body.loss_kwargs == {}


def test_an_undeclared_name_goes_to_the_single_passthrough_region():
    """Public signatures stay ``**kwargs``; only the wire shape changes."""
    body = build_request(
        model_types.ForwardBackwardTaskRequest,
        inputs=[{'input_ids': [1]}],
        adapter_name='a',
        advantages=[0.5],
        old_logps=[[-1.0]])
    assert body.loss_kwargs == {'advantages': [0.5], 'old_logps': [[-1.0]]}


def test_an_ambiguous_target_is_an_error_rather_than_a_guess():
    """With two regions the builder refuses to choose instead of guessing.

    A model that grows a second region -- constructor arguments *and* invoked-method
    arguments, say -- has no name-based rule that can tell them apart, and guessing wrong
    sends a valid argument to the wrong callable: a wrong result rather than an error. No
    shipped model has two today; the rule exists so that adding one fails loudly.
    """

    class _TwoRegions(StrictRequest):
        target: str
        init_kwargs: Dict[str, JsonValue] = passthrough()
        call_kwargs: Dict[str, JsonValue] = passthrough()

    with pytest.raises(TwinkleClientValidationError) as raised:
        build_request(_TwoRegions, target='t', unplaceable=1)
    assert 'init_kwargs' in str(raised.value) and 'call_kwargs' in str(raised.value)


def test_a_model_without_a_region_rejects_an_unknown_name():
    with pytest.raises(TwinkleClientValidationError) as raised:
        build_request(model_types.SaveRequest, adapter_name='a', name='ckpt', typo=1)
    assert 'typo' in str(raised.value)


def test_an_explicit_region_and_a_routed_key_are_merged():
    body = build_request(
        model_types.SetLossRequest, loss_cls='DPOLoss', adapter_name='a', init_kwargs={'beta': 0.1}, loss_type='sigmoid')
    assert body.init_kwargs == {'beta': 0.1, 'loss_type': 'sigmoid'}


def test_a_key_passed_twice_is_an_error():
    """Silently letting one win would make the effective value depend on merge order."""
    with pytest.raises(TwinkleClientValidationError, match='both directly and inside'):
        build_request(
            model_types.SetLossRequest, loss_cls='DPOLoss', adapter_name='a', init_kwargs={'beta': 0.1}, beta=0.2)


def test_an_omitted_optional_argument_is_not_routed_into_the_region():
    """Client methods pass optionals unconditionally.

    Routing an explicit ``None`` would hand the backend a null argument it never received
    before, changing behaviour for callers who simply did not pass anything.
    """
    body = build_request(model_types.SetLossRequest, loss_cls='DPOLoss', adapter_name='a', unused=None)
    assert body.init_kwargs == {}


# --------------------------------------------------------------------------- #
# Validation before the wire
# --------------------------------------------------------------------------- #


def test_a_wrongly_typed_field_fails_in_process():
    with pytest.raises(ValidationError):
        build_request(model_types.ForwardRequest, inputs=[{'input_ids': [1]}], adapter_name='a', temperature='hot')


def test_an_out_of_range_value_fails_in_process():
    with pytest.raises(ValidationError):
        build_request(model_types.ForwardRequest, inputs=[{'input_ids': [1]}], adapter_name='a', temperature=0)


def test_malformed_inputs_fail_in_process():
    """The client shares the server's schema, so it catches this without asking."""
    with pytest.raises(ValidationError):
        build_request(model_types.ForwardRequest, inputs=[{'input_ids': [True]}], adapter_name='a')


def test_a_missing_required_field_fails_in_process():
    with pytest.raises(ValidationError):
        build_request(model_types.ForwardRequest, inputs=[{'input_ids': [1]}])


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def test_unset_optionals_stay_off_the_wire():
    """Absent and "not requested" must look the same, or the server needs its own defaults."""
    import json
    body = build_request(model_types.ForwardRequest, inputs=[{'input_ids': [1]}], adapter_name='a')
    payload = json.loads(request_json(body))
    assert payload == {'inputs': [{'input_ids': [1]}], 'adapter_name': 'a', 'loss_kwargs': {}}


def test_a_lora_config_is_serialized_to_the_form_the_server_decodes():
    from peft import LoraConfig
    wire = to_wire_value(LoraConfig(target_modules='all-linear'))
    assert isinstance(wire, str) and 'LoraConfig' in wire


def test_a_component_handle_is_sent_as_its_id():
    class _Handle:
        processor_id = 'pid:abc'

    assert to_wire_value(_Handle()) == 'pid:abc'


def test_numpy_values_are_converted_to_lists():
    import numpy as np
    assert to_wire_value(np.array([1, 2])) == [1, 2]


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
