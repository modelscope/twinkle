# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the typed ``ServerConfig``.

Properties covered:
- # Feature: server-config-observability-refactor,
  Property 12: Valid configuration yields a fully validated instance
- # Feature: server-config-observability-refactor,
  Property 13: Any constraint violation is rejected with the offending field named
- # Feature: server-config-observability-refactor,
  Property 14: Configuration round-trip fidelity
- # Feature: server-config-observability-refactor,
  Property 15: Legacy / unknown field names are rejected
"""
from __future__ import annotations

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from pathlib import Path
from pydantic import ValidationError

from twinkle.server.config import ApplicationSpec, ServerConfig
from twinkle.server.exceptions import ConfigParseError
from twinkle.server.launcher import ServerLauncher

# ---------- minimal valid config strategy ---------------------------------- #

_PERSISTENCE_VARIANTS = st.one_of(
    st.fixed_dictionaries({'mode': st.just('memory')}),
    st.fixed_dictionaries({
        'mode': st.just('file'),
        'file_path': st.just('/tmp/state.json')
    }),
    st.fixed_dictionaries({
        'mode': st.just('redis'),
        'redis_url': st.just('redis://localhost:6379/0')
    }),
)

_MODEL_APP = st.fixed_dictionaries({
    'name':
    st.just('m'),
    'route_prefix':
    st.just('/api/v1/m'),
    'import_path':
    st.just('model'),
    'args':
    st.fixed_dictionaries({
        'model_id': st.just('model-id'),
        'device_group': st.just({
            'name': 'g',
            'ranks': 1,
            'device_type': 'CPU'
        }),
        'device_mesh': st.just({
            'device_type': 'CPU',
            'dp_size': 1
        }),
        'backend': st.sampled_from(['mock', 'transformers', 'megatron']),
    }),
})

_VALID_CONFIG = st.fixed_dictionaries({
    'persistence': _PERSISTENCE_VARIANTS,
    'applications': st.lists(_MODEL_APP, min_size=0, max_size=3),
})

# ---------- Property 12: valid → fully validated ----------------------------- #


@settings(max_examples=100)
@given(payload=_VALID_CONFIG)
def test_property_12_valid_payload_yields_full_instance(payload: dict) -> None:
    cfg = ServerConfig.model_validate(payload)
    assert isinstance(cfg, ServerConfig)
    assert all(isinstance(a, ApplicationSpec) for a in cfg.applications)
    # Nested sections instantiated and validated.
    assert cfg.persistence.mode == payload['persistence']['mode']
    assert cfg.task_queue.rps_limit >= 0


# ---------- Property 13: violation → field-named error ---------------------- #


def test_property_13_redis_mode_missing_url() -> None:
    with pytest.raises(ValidationError) as exc:
        ServerConfig.model_validate({'persistence': {'mode': 'redis'}})
    msg = str(exc.value)
    assert 'persistence.redis_url' in msg or 'redis_url' in msg


def test_property_13_file_mode_missing_path() -> None:
    with pytest.raises(ValidationError) as exc:
        ServerConfig.model_validate({'persistence': {'mode': 'file'}})
    msg = str(exc.value)
    assert 'persistence.file_path' in msg or 'file_path' in msg


@settings(max_examples=100)
@given(bad_backend=st.text(min_size=1, max_size=8).filter(lambda s: s not in ('mock', 'transformers', 'megatron')))
def test_property_13_bad_backend_names_field(bad_backend: str) -> None:
    payload = {
        'applications': [{
            'name': 'm',
            'import_path': 'model',
            'args': {
                'model_id': 'x',
                'device_group': {},
                'device_mesh': {},
                'backend': bad_backend,
            },
        }]
    }
    with pytest.raises(ValidationError) as exc:
        ServerConfig.model_validate(payload)
    errors = exc.value.errors()
    assert any('backend' in err['loc'] for err in errors)


@settings(max_examples=100)
@given(bad_max_input_tokens=st.integers(max_value=0, min_value=-1000))
def test_property_13_nested_field_constraint_violation_named(bad_max_input_tokens: int) -> None:
    """Nested-section constraints (here ``task_queue.max_input_tokens``) are
    enforced together with cross-field ones and the offending path is
    visible in the error."""
    with pytest.raises(ValidationError) as exc:
        ServerConfig.model_validate({'task_queue': {'max_input_tokens': bad_max_input_tokens}})
    errors = exc.value.errors()
    assert any('max_input_tokens' in err['loc'] for err in errors)


# ---------- round-trip fidelity ----------------------------------------- #


@settings(max_examples=100)
@given(payload=_VALID_CONFIG)
def test_property_14_round_trip_fidelity(payload: dict) -> None:
    cfg = ServerConfig.model_validate(payload)
    dumped = cfg.to_yaml_dict()
    re_loaded = ServerConfig.model_validate(dumped)
    assert re_loaded == cfg
    assert re_loaded.model_dump() == cfg.model_dump()


# ---------- Property 15: legacy/unknown rejected ----------------------------- #


@pytest.mark.parametrize(
    'legacy_field',
    ['telemetry_config', 'persistence_config'],
)
def test_property_15_legacy_field_rejected(legacy_field: str) -> None:
    payload = {legacy_field: {}}
    with pytest.raises(ValidationError) as exc:
        ServerConfig.model_validate(payload)
    errors = exc.value.errors()
    assert any(err['type'] == 'extra_forbidden' for err in errors)
    assert any(legacy_field in err['loc'] for err in errors)


@settings(max_examples=100)
@given(unknown=st.text(min_size=1, max_size=20).filter(lambda s: not s.startswith('_')))
def test_property_15_unknown_field_rejected(unknown: str) -> None:
    known = {
        'ray_namespace',
        'proxy_location',
        'http_options',
        'telemetry',
        'persistence',
        'task_queue',
        'applications',
    }
    if unknown in known:
        return
    with pytest.raises(ValidationError):
        ServerConfig.model_validate({unknown: 'x'})


@pytest.mark.parametrize('section', ['telemetry', 'persistence'])
def test_property_15_unknown_nested_field_rejected(section: str) -> None:
    """Nested config sections also reject unknown keys (defends against typos
    inside ``telemetry: {...}`` / ``persistence: {...}``)."""
    payload = {section: {'unknown_typo': 1}}
    with pytest.raises(ValidationError) as exc:
        ServerConfig.model_validate(payload)
    assert any('unknown_typo' in err['loc'] for err in exc.value.errors())


# ---------- 3.11: from_yaml error paths + launcher dict rejection ---------- #


def test_from_yaml_missing_path(tmp_path: Path) -> None:
    p = tmp_path / 'does_not_exist.yaml'
    with pytest.raises(FileNotFoundError) as exc:
        ServerConfig.from_yaml(p)
    assert str(p) in str(exc.value)


def test_from_yaml_malformed_yaml(tmp_path: Path) -> None:
    p = tmp_path / 'bad.yaml'
    p.write_text('this is: not: valid: yaml: : :\n  - [unbalanced\n')
    with pytest.raises(ConfigParseError):
        ServerConfig.from_yaml(p)


def test_from_yaml_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / 'list.yaml'
    p.write_text('- a\n- b\n')
    with pytest.raises(ConfigParseError):
        ServerConfig.from_yaml(p)


def test_from_yaml_valid_minimal(tmp_path: Path) -> None:
    p = tmp_path / 'mini.yaml'
    yaml.safe_dump(
        {
            'persistence': {
                'mode': 'memory'
            },
            'applications': []
        },
        p.open('w'),
    )
    cfg = ServerConfig.from_yaml(p)
    assert cfg.persistence.mode == 'memory'
    assert cfg.applications == []


def test_launcher_rejects_raw_dict() -> None:
    with pytest.raises(TypeError) as exc:
        ServerLauncher(config={'applications': []})
    assert 'ServerConfig' in str(exc.value)


def test_launcher_accepts_typed_config() -> None:
    cfg = ServerConfig()
    launcher = ServerLauncher(config=cfg)
    assert launcher.config is cfg


def test_cookbook_examples_load() -> None:
    """Migrated cookbook configs all parse with the new field names."""
    here = Path(__file__).resolve().parents[3]
    examples = [
        here / 'cookbook' / 'client' / 'server' / 'transformer' / 'server_config.yaml',
        here / 'cookbook' / 'client' / 'server' / 'megatron' / 'server_config.yaml',
        here / 'cookbook' / 'client' / 'server' / 'megatron' / 'server_config_4b.yaml',
    ]
    for p in examples:
        cfg = ServerConfig.from_yaml(p)
        assert isinstance(cfg, ServerConfig), p
