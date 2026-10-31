# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from twinkle.data_format import SamplingParams

from .validation import (
    ValidationConfig,
    _evaluate,
    cancel_evidence_gate,
    learning_gate,
    pipeline_evidence_gate,
    result_payload,
    run_cancel_validation,
    run_validation,
    write_result,
)


def _config(tmp_path):
    return ValidationConfig(
        backend='transformers',
        phase='learning',
        server_url='http://127.0.0.1:8000',
        model_id='ms://Qwen/Qwen3.5-4B',
        template_model_id='/models/Qwen3.5-4B',
        agentenv_template='python-test',
        result_dir=str(tmp_path),
    )


def test_learning_gate_requires_updates_and_independent_improvement():
    assert learning_gate(updates=2, baseline=0.25, final=0.5, minimum_improvement=0.0)
    assert not learning_gate(updates=1, baseline=0.25, final=0.5, minimum_improvement=0.0)
    assert not learning_gate(updates=2, baseline=0.25, final=0.25, minimum_improvement=0.0)
    assert not learning_gate(updates=2, baseline=0.25, final=0.3, minimum_improvement=0.1)


def test_fake_and_real_result_modes_are_explicit(tmp_path):
    config = _config(tmp_path)
    fake = result_payload(config, mode='fake', status='completed', pipeline_gate=True)
    real = result_payload(config, status='blocked', pipeline_gate=False)
    assert fake['mode'] == 'fake'
    assert real['mode'] == 'real'
    assert fake['status'] != real['status']


def test_result_file_contains_no_server_url_or_credentials(tmp_path):
    path = write_result(_config(tmp_path), status='blocked', pipeline_gate=False, learning_gate=None)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['mode'] == 'real'
    assert 'server_url' not in payload['config']
    assert 'api_key' not in payload
    assert 'TWINKLE_SERVER_TOKEN' not in payload


def _pipeline_evidence():
    return {
        'two_updates': True,
        'parameter_change': True,
        'multi_turn_feedback': True,
        'token_alignment': True,
        'distinct_snapshots': True,
        'new_policy_requests': True,
        'sync_async_equivalent': True,
        'independent_episodes': True,
        'environment_isolation': True,
        'cleanup_confirmed': True,
    }


def test_pipeline_gate_fails_when_any_required_evidence_is_missing():
    evidence = _pipeline_evidence()
    assert pipeline_evidence_gate(evidence)
    for key in evidence:
        incomplete = dict(evidence)
        incomplete[key] = False
        assert not pipeline_evidence_gate(incomplete), key


def test_cancel_gate_rejects_early_env_cancel_new_work_and_cleanup_failure():
    assert not cancel_evidence_gate({
        'stage_reached': False,
        'cancelled': True,
        'post_cancel_operations': [],
        'cleanup_issues': [],
    })
    assert not cancel_evidence_gate({
        'stage_reached': True,
        'cancelled': True,
        'post_cancel_operations': ['sample'],
        'cleanup_issues': [],
    })
    assert not cancel_evidence_gate({
        'stage_reached': True,
        'cancelled': True,
        'post_cancel_operations': [],
        'cleanup_issues': ['unknown execution'],
    })


def test_learning_without_frozen_threshold_is_blocked(tmp_path):
    with pytest.raises(ValueError, match='frozen'):
        asyncio.run(run_validation(_config(tmp_path)))


def test_cli_template_is_set_before_recipe_import(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), min_success_improvement=0.1)
    monkeypatch.setenv('AGENTENV_TEMPLATE', 'stale-template')
    seen = []

    def stop_at_import():
        seen.append(os.environ['AGENTENV_TEMPLATE'])
        raise RuntimeError('stop before provider setup')

    monkeypatch.setattr('tests.server.integration.agentic_rl.validation._load_recipe', stop_at_import)
    with pytest.raises(RuntimeError, match='stop before'):
        asyncio.run(run_validation(config))
    with pytest.raises(RuntimeError, match='stop before'):
        asyncio.run(run_cancel_validation(config))
    assert seen == [config.agentenv_template, config.agentenv_template]


def test_evaluation_uses_held_out_tasks_and_distinct_group_seeds(tmp_path):
    config = _config(tmp_path)
    seen = []

    class Recipe:
        _create_attempt = None

        @staticmethod
        async def _rollout_group(rollout, policy_uri, environments, stop_event, sampling_params=None):
            seen.append(sampling_params.seed)
            return [], [0.0, 1.0, 0.0, 0.0]

    rollout = SimpleNamespace(sampling_params=SamplingParams())
    _, evaluations = asyncio.run(_evaluate(Recipe(), rollout, 'policy-0', [], asyncio.Event(), config))
    assert len(seen) == len(set(seen)) == 4
    assert {item['task_id'] for item in evaluations} == {'count-even', 'last-negative'}
    assert all(item['task_id'] != 'smallest-positive' for item in evaluations)


@pytest.mark.skipif(
    not os.environ.get('AGENTENV_TEMPLATE') or not os.environ.get('TWINKLE_SERVER_URL'),
    reason='real AgentENV and Twinkle Server are not configured',
)
def test_real_agentenv_configuration_is_present():
    assert os.environ['AGENTENV_TEMPLATE']
    assert os.environ['TWINKLE_SERVER_URL'].startswith(('http://', 'https://'))
