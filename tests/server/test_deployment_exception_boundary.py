from fastapi import FastAPI
from fastapi.testclient import TestClient

from twinkle.server.deployment import build_deployment_app


def test_deployment_app_catches_unhandled_route_exception_and_keeps_serving(monkeypatch):

    class _ReplicaId:
        unique_id = 'replica-test'

    class _Context:
        replica_id = _ReplicaId()

    from twinkle.server import deployment

    monkeypatch.setattr(deployment.serve, 'get_replica_context', lambda: _Context())
    calls = {'health': 0}

    def register_routes(app: FastAPI, _get_self):

        @app.get('/healthz')
        async def healthz():
            calls['health'] += 1
            return {'ok': True, 'health_calls': calls['health']}

        @app.get('/boom')
        async def boom():
            raise RuntimeError('boom with replica header')

    app = build_deployment_app('Test', register_routes, attach_replica_id_header=True)
    client = TestClient(app)

    response = client.get('/boom', headers={'x-request-id': 'boundary-test'})
    assert response.status_code == 500
    assert response.headers['X-Twinkle-Replica-Id'] == 'replica-test'
    # Unhandled exceptions now return the unified ErrorPayload (Server category
    # keeps the traceback) instead of the legacy {'detail': <traceback>} shape.
    body = response.json()
    assert body['category'] == 'server'
    assert body['error_code'] == 500
    assert body['error'] == 'boom with replica header'
    assert 'Traceback' in body['traceback']
    assert 'RuntimeError: boom with replica header' in body['traceback']

    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json() == {'ok': True, 'health_calls': 1}


def test_deployment_app_bounds_overlong_unhandled_error(monkeypatch):

    class _ReplicaId:
        unique_id = 'replica-test'

    class _Context:
        replica_id = _ReplicaId()

    from twinkle.server import deployment

    monkeypatch.setattr(deployment.serve, 'get_replica_context', lambda: _Context())

    def register_routes(app: FastAPI, _get_self):

        @app.get('/boom')
        async def boom():
            raise RuntimeError(f'first line {"X" * 2048}\nsecond line')

    client = TestClient(build_deployment_app('Test', register_routes))
    response = client.get('/boom', headers={'x-request-id': 'long-error'})

    assert response.status_code == 500
    body = response.json()
    assert len(body['error']) == 1024
    assert '\n' not in body['error']
    assert len(body['traceback']) <= 65536
    assert body['traceback'].endswith('second line\n')
    assert body['request_id'] == 'long-error'
