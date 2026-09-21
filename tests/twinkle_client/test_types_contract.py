# Copyright (c) ModelScope Contributors. All rights reserved.
"""Contract-base consistency and naming-disambiguation tests.

- ``QueueStateLiteral`` value set equals the server ``QueueState`` enum.
- naming disambiguation guard.

The two SDKs already share public names. The contract freezes that legacy set and
rejects new collisions while requiring explicit aliases when both SDKs are imported
in one module.
"""
from __future__ import annotations

import ast
import pathlib
import typing

import twinkle
from twinkle.server.task_queue.types import QueueState
from twinkle.protocol.types import model as model_types
from twinkle.protocol.types.errors import QueueStateLiteral
from twinkle.protocol.types.server import GetServerCapabilitiesResponse

_TWINKLE_SRC = pathlib.Path(twinkle.__file__).resolve().parent
_LEGACY_PUBLIC_NAME_OVERLAP = frozenset({
    'Checkpoint',
    'CheckpointsListResponse',
    'CreateModelRequest',
    'CreateSessionRequest',
    'CreateSessionResponse',
    'Cursor',
    'ForwardRequest',
    'GetServerCapabilitiesResponse',
    'HealthResponse',
    'LoraConfig',
    'SampleRequest',
    'SessionHeartbeatRequest',
    'SessionHeartbeatResponse',
    'SupportedModel',
    'TrainingRun',
    'TrainingRunsResponse',
    'WeightsInfoResponse',
    'checkpoint',
})


def test_queue_state_literal_matches_server_enum():
    literal_values = set(typing.get_args(QueueStateLiteral))
    enum_values = {state.value for state in QueueState}
    assert literal_values == enum_values, (f'QueueStateLiteral {literal_values} != QueueState {enum_values}')


def test_old_capabilities_response_gets_conservative_defaults():
    response = GetServerCapabilitiesResponse.model_validate({'supported_models': []})
    assert response.protocol_version == 1
    assert response.features.task_envelope is True
    assert response.features.cancel is False
    assert response.features.batch_retrieve is False


def test_capabilities_response_ignores_future_fields():
    response = GetServerCapabilitiesResponse.model_validate({
        'supported_models': [],
        'future_top_level': True,
        'features': {
            'cancel': True,
            'future_feature': True
        },
        'limits': {
            'max_batch_size': 8,
            'future_limit': 9
        },
    })
    assert response.features.cancel is True
    assert response.limits.max_batch_size == 8


def test_data_plane_forward_only_has_no_seq_id() -> None:
    fields = model_types.DataPlaneForwardOnlyRequest.model_fields
    assert 'seq_id' not in fields
    assert 'seq_id' in model_types.DataPlaneForwardRequest.model_fields


def test_void_response_names_are_canonical_ok_response_aliases() -> None:
    for name in (
        'BackwardResponse',
        'StepResponse',
        'ZeroGradResponse',
        'LrStepResponse',
        'SetLossResponse',
        'SetOptimizerResponse',
        'SetLrSchedulerResponse',
        'LoadResponse',
        'SetTemplateResponse',
        'SetProcessorResponse',
        'ClipGradAndStepResponse',
        'ApplyPatchResponse',
        'AddMetricResponse',
    ):
        assert getattr(model_types, name) is model_types.OkResponse


def _origin(module: str | None) -> str | None:
    """Classify an import's source module as 'tinker', 'twinkle_client', or None."""
    if not module:
        return None
    if module == 'tinker' or module.startswith('tinker.'):
        return 'tinker'
    if module == 'twinkle_client' or module.startswith('twinkle_client.'):
        return 'twinkle_client'
    return None


def _binding_collisions(tree: ast.AST) -> set[str]:
    """Return local names bound to BOTH a tinker and a twinkle_client import."""
    tinker_names: set[str] = set()
    twinkle_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            origin = _origin(node.module)
            if origin is None:
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                (tinker_names if origin == 'tinker' else twinkle_names).add(bound)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                origin = _origin(alias.name)
                if origin is None:
                    continue
                bound = alias.asname or alias.name.split('.')[0]
                (tinker_names if origin == 'tinker' else twinkle_names).add(bound)
    return tinker_names & twinkle_names


def test_public_name_overlap_does_not_grow():
    import tinker.types

    import twinkle.protocol.types

    overlap = {name for name in set(dir(tinker.types)) & set(dir(twinkle.protocol.types)) if not name.startswith('_')}
    assert overlap == _LEGACY_PUBLIC_NAME_OVERLAP


def test_no_tinker_twinkle_same_name_binding():
    offenders: dict[str, set[str]] = {}
    for path in _TWINKLE_SRC.rglob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        collisions = _binding_collisions(tree)
        if collisions:
            offenders[str(path.relative_to(_TWINKLE_SRC))] = collisions
    assert not offenders, ('tinker and twinkle_client types bound to the same local name (alias tinker '
                           f'to disambiguate): {offenders}')
