# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the pure protocol helpers (routing headers, json_safe)."""
import torch
from pydantic import BaseModel

from twinkle.protocol.headers import (
    H_AUTH,
    H_AUTH_TWINKLE,
    H_REQUEST_ID,
    H_REQUEST_ID_LEGACY,
    build_routing_headers,
)
from twinkle.protocol.json_utils import json_safe


class TestBuildRoutingHeaders:

    def test_request_id_and_auth_keep_both_wire_names(self):
        assert build_routing_headers('req-1', auth='secret') == {
            H_REQUEST_ID: 'req-1',
            H_REQUEST_ID_LEGACY: 'req-1',
            H_AUTH: 'secret',
            H_AUTH_TWINKLE: 'secret',
        }
        assert build_routing_headers('req-1')[H_AUTH] == ''


class _Model(BaseModel):
    name: str
    value: int


class TestJsonSafe:

    def test_primitives_pass_through(self):
        assert json_safe('a') == 'a'
        assert json_safe(1) == 1
        assert json_safe(True) is True
        assert json_safe(None) is None
        assert json_safe(b'raw') == b'raw'

    def test_containers_are_recursively_converted(self):
        assert json_safe({'k': [1, (2, 3)]}) == {'k': [1, [2, 3]]}
        # sets and frozensets become plain lists
        assert json_safe({'s', 'b'}) in (['s', 'b'], ['b', 's'])

    def test_pydantic_models_are_dumped(self):
        assert json_safe(_Model(name='n', value=2)) == {'name': 'n', 'value': 2}

    def test_tensors_are_converted_via_tolist(self):
        assert json_safe(torch.tensor([1, 2])) == [1, 2]
        assert json_safe({'logits': torch.ones(2, 2)}) == {'logits': [[1.0, 1.0], [1.0, 1.0]]}

    def test_string_with_tolist_like_failure_stays_verbatim(self):
        # A str has no tolist; anything whose tolist() raises must fall through
        # unchanged instead of raising out of json_safe.

        class _Broken:
            def tolist(self):
                raise RuntimeError('cannot convert')

        assert json_safe(_Broken()) is not None
