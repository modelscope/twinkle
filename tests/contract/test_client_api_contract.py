# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-facing API contract regression test (R20.3, R20.4, R18.1).

# Feature: server-config-observability-refactor, Property 28: Client-facing API contract invariance

The refactor freezes the client-facing HTTP surface: the route paths, HTTP
methods, and request/response schemas of the Tinker (`/*`, `/tinker/*`) and
Twinkle (`/twinkle/*`) endpoints MUST be identical before and after each
phase. This test rebuilds the FastAPI apps from the current sources, extracts
their OpenAPI surface, and asserts equality with the committed baseline at
``tests/contract/client_api_baseline.json``.

Updating the baseline is intentionally a manual step: run
``python -m tests.contract.update_baseline`` (or call
``client_api_harness.write_baseline()``) only when an API change has been
explicitly approved.
"""
from __future__ import annotations

import json

import pytest

from tests.contract.client_api_harness import (
    APP_BUILDERS,
    BASELINE_PATH,
    extract_full_surface,
    load_baseline,
)


def test_baseline_file_exists() -> None:
    assert BASELINE_PATH.exists(), (
        f'Baseline {BASELINE_PATH} missing. Generate it with '
        '`python -m tests.contract.update_baseline` after confirming the '
        'current client-facing surface is correct.'
    )


@pytest.mark.parametrize('app_name', sorted(APP_BUILDERS.keys()))
def test_app_surface_matches_baseline(app_name: str) -> None:
    """Per-app surface equals the snapshot — narrows failure scope per app."""
    baseline = load_baseline()
    current = extract_full_surface()

    expected = baseline.get(app_name)
    actual = current.get(app_name)
    assert expected is not None, f'baseline is missing app {app_name!r}'
    assert actual is not None, f'current surface is missing app {app_name!r}'

    if actual != expected:
        diff = _surface_diff(expected, actual)
        pytest.fail(
            f'Client-API surface for {app_name!r} drifted from the baseline.\n'
            f'{diff}\n'
            f'If the change is intentional, regenerate the baseline with '
            f'`python -m tests.contract.update_baseline`.'
        )


def test_full_surface_matches_baseline() -> None:
    """Whole-surface equality — the cross-cutting freeze guard."""
    baseline = load_baseline()
    current = extract_full_surface()
    assert current == baseline, (
        'Full client-API surface drifted from the baseline. '
        'See per-app failures for details.'
    )


def _surface_diff(expected: dict, actual: dict) -> str:
    exp_paths = set((expected.get('paths') or {}).keys())
    act_paths = set((actual.get('paths') or {}).keys())
    added = sorted(act_paths - exp_paths)
    removed = sorted(exp_paths - act_paths)
    changed = []
    for p in sorted(exp_paths & act_paths):
        if expected['paths'][p] != actual['paths'][p]:
            exp_methods = set(expected['paths'][p].keys())
            act_methods = set(actual['paths'][p].keys())
            method_added = sorted(act_methods - exp_methods)
            method_removed = sorted(exp_methods - act_methods)
            method_diff = ''
            if method_added or method_removed:
                method_diff = f' methods +{method_added} -{method_removed}'
            changed.append(f'  {p}{method_diff}')
    parts = []
    if added:
        parts.append(f'  added paths: {added}')
    if removed:
        parts.append(f'  removed paths: {removed}')
    if changed:
        parts.append('  changed paths:\n' + '\n'.join(changed))
    return '\n'.join(parts) if parts else json.dumps(
        {'expected_keys': sorted(expected.keys()), 'actual_keys': sorted(actual.keys())},
        indent=2,
    )
