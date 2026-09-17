# Copyright (c) ModelScope Contributors. All rights reserved.
"""Task_Envelope coverage check (T5.4 / R5#7, R5#2/#3).

Walks the model and sampler route tables and asserts that every twinkle-native
POST route that enters the Task_Queue declares ``response_model = TaskEnvelope``.
The exemption list (endpoints that do NOT enter the queue, plus the streaming
endpoint) is declared here, in one place.
"""
from __future__ import annotations

from fastapi.routing import APIRoute

from tests.server.contract.client_api_harness import build_model_app, build_sampler_app
from twinkle_client.types.lifecycle import TaskEnvelope

# The single exemption declaration (R5#7), keyed BY APP. A flat path set would be wrong:
# ``/twinkle/set_template`` and ``/twinkle/apply_patch`` exist on both apps, but only the
# sampler's bypass the queue -- the model's are queued and must return a Task_Envelope.
# Sharing one set silently exempted the model's two and left a hole in this guard.
_EXEMPT_BY_APP = {
    'model': {
        # health/session bootstrap only, no queue
        '/twinkle/create',
    },
    'sampler': {
        # direct call_backend, no queue
        '/twinkle/create',
        '/twinkle/set_template',
        '/twinkle/add_adapter_to_sampler',
        '/twinkle/apply_patch',
        '/twinkle/unload_adapter_paths',
        # the one streaming exception (R5#2)
        '/twinkle/sample_stream',
    },
}


def _queued_twinkle_post_routes(app, exempt):
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if 'POST' not in route.methods:
            continue
        if not route.path.startswith('/twinkle/'):
            continue
        if route.path in exempt:
            continue
        yield route


def test_every_queued_twinkle_route_returns_task_envelope():
    violations = []
    for app_name, app in (('model', build_model_app()), ('sampler', build_sampler_app())):
        for route in _queued_twinkle_post_routes(app, _EXEMPT_BY_APP[app_name]):
            if route.response_model is not TaskEnvelope:
                violations.append((app_name, route.path, route.response_model))
    assert violations == [], f'queued routes not returning TaskEnvelope: {violations}'


def test_model_side_set_template_and_apply_patch_are_not_exempt():
    # Regression guard for the hole above: these two are queued on the model app, so
    # they must be covered by the assertion rather than skipped by a shared path set.
    assert '/twinkle/set_template' not in _EXEMPT_BY_APP['model']
    assert '/twinkle/apply_patch' not in _EXEMPT_BY_APP['model']
    covered = {route.path for route in _queued_twinkle_post_routes(build_model_app(), _EXEMPT_BY_APP['model'])}
    assert {'/twinkle/set_template', '/twinkle/apply_patch'} <= covered
