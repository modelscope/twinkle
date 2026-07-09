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
    assert 'Traceback' in response.json()['detail']
    assert 'RuntimeError: boom with replica header' in response.json()['detail']

    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json() == {'ok': True, 'health_calls': 1}
