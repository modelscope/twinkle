from fastapi import FastAPI
from fastapi.testclient import TestClient

from twinkle.server.deployment import build_deployment_app


def test_deployment_app_returns_traceback_and_keeps_serving_after_unhandled_route_exception():
    calls = {'boom': 0, 'health': 0}

    def register_routes(app: FastAPI, _get_self):

        @app.get('/healthz')
        async def healthz():
            calls['health'] += 1
            return {'ok': True, 'health_calls': calls['health']}

        @app.get('/boom')
        async def boom():
            calls['boom'] += 1
            raise RuntimeError(f'boom from route #{calls["boom"]}')

    app = build_deployment_app('Test', register_routes)
    client = TestClient(app)

    for i in range(3):
        response = client.get('/boom', headers={'x-request-id': 'boundary-test'})
        assert response.status_code == 500
        assert f'RuntimeError: boom from route #{i + 1}' in response.json()['detail']
        assert 'Traceback' in response.json()['detail']

    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json() == {'ok': True, 'health_calls': 1}
