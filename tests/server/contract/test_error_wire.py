from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tinker.types import RequestFailedResponse

from twinkle.server.gateway.tinker_handlers import _register_tinker_routes


class _State:

    async def get_future(self, request_id: str):
        return {
            'status': 'failed',
            'failure': {
                'reason_code': 'execution_timeout',
                'message': 'backend timed out',
                'attribution': 'server',
            },
        }


class _Gateway:
    state = _State()


def test_retrieve_future_returns_parseable_error_payload():
    app = FastAPI()
    _register_tinker_routes(app, lambda: _Gateway())

    response = TestClient(app).post('/retrieve_future', json={'request_id': 'req-1'})

    assert response.status_code == 200
    body = response.json()
    assert body['error_code'] == 504
    assert body['request_id'] == 'req-1'
    parsed = RequestFailedResponse.model_validate(body)
    assert parsed.category.value == 'server'
