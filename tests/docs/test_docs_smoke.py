# Copyright (c) ModelScope Contributors. All rights reserved.
"""Smoke checks for the Phase 5 documentation set (R8.3, R11.4, R17)."""
from __future__ import annotations

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------- file presence ------------------------------------------------- #

OBSERVABILITY_EN = REPO_ROOT / 'docs' / 'source_en' / 'Usage Guide' / 'Observability.md'
OBSERVABILITY_ZH = REPO_ROOT / 'docs' / 'source_zh' / '使用指引' / '可观测化.md'
CONFIG_GUIDE_ZH = REPO_ROOT / 'docs' / 'source_zh' / '使用指引' / '服务配置.md'

INDEX_EN = REPO_ROOT / 'docs' / 'source_en' / 'index.rst'
INDEX_ZH = REPO_ROOT / 'docs' / 'source_zh' / 'index.rst'


@pytest.mark.parametrize(
    'path',
    [OBSERVABILITY_EN, OBSERVABILITY_ZH, CONFIG_GUIDE_ZH, INDEX_EN, INDEX_ZH],
)
def test_doc_exists(path: Path) -> None:
    assert path.exists(), f'missing doc: {path}'


# ---------- observability guide content (R11.4, R17.1, R17.2) ------------ #

_CORRELATION_KEYS = (
    'twinkle.session_id',
    'twinkle.model_id',
    'twinkle.replica_id',
    'twinkle.token_id',
    'twinkle.sampling_session_id',
    'twinkle.base_model',
)


@pytest.mark.parametrize('path', [OBSERVABILITY_EN, OBSERVABILITY_ZH])
def test_observability_lists_all_correlation_keys(path: Path) -> None:
    text = path.read_text()
    for key in _CORRELATION_KEYS:
        assert key in text, f'{path.name}: missing correlation key {key}'


@pytest.mark.parametrize('path', [OBSERVABILITY_EN, OBSERVABILITY_ZH])
def test_observability_describes_propagation(path: Path) -> None:
    text = path.read_text()
    # Mentions the carrier helpers + the propagation surface.
    assert 'make_carrier' in text
    assert 'activate_carrier' in text
    assert 'DeploymentHandle' in text


@pytest.mark.parametrize('path', [OBSERVABILITY_EN, OBSERVABILITY_ZH])
def test_observability_has_lgtm_example(path: Path) -> None:
    text = path.read_text()
    assert 'docker compose' in text or 'docker-compose' in text
    assert 'cookbook/observability' in text


# ---------- server-config guide content (R8.3, R17.3) -------------------- #


def test_config_guide_lists_top_level_fields() -> None:
    text = CONFIG_GUIDE_ZH.read_text()
    for field in ('telemetry', 'persistence', 'task_queue', 'applications', 'http_options'):
        assert field in text


def test_config_guide_documents_envvars() -> None:
    text = CONFIG_GUIDE_ZH.read_text()
    assert 'TWINKLE_SERVER_CONFIG' in text
    assert 'TWINKLE_RAY_NAMESPACE' in text


def test_config_guide_includes_yaml_example() -> None:
    text = CONFIG_GUIDE_ZH.read_text()
    assert 'applications:' in text
    assert 'backend: mock' in text or 'backend:' in text
    # Reference to the documented example file.
    assert 'server_config.example.yaml' in text


def test_config_guide_has_migration_table() -> None:
    text = CONFIG_GUIDE_ZH.read_text()
    # Both legacy → current rows must be present (R8.3).
    assert 'telemetry_config' in text and 'telemetry:' in text
    assert 'persistence_config' in text and 'persistence:' in text
    assert 'use_megatron' in text and 'backend:' in text


# ---------- index links (R17.4) ------------------------------------------ #


def test_index_zh_links_both_guides() -> None:
    text = INDEX_ZH.read_text()
    assert '可观测化.md' in text
    assert '服务配置.md' in text


def test_index_en_links_observability() -> None:
    text = INDEX_EN.read_text()
    assert 'Observability.md' in text
