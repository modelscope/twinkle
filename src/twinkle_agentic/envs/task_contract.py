# Copyright (c) ModelScope Contributors. All rights reserved.
"""Validated intent, task, and profile-provenance contracts for sandbox routing."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .profiles import ProfileRegistry, ProfileResolution, SandboxProfile

__all__ = [
    'TaskIntent',
    'FixtureSpec',
    'VerifierSpec',
    'FreshnessSpec',
    'ProfileProvenance',
    'TaskContract',
    'resolve_intent',
    'validate_task_contract',
]

_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_INPUT_KINDS = frozenset({
    'given_text',
    'data_fixture',
    'lean_fixture',
    'repository_fixture',
    'live_web',
    'legacy_task',
})
_OUTPUT_KINDS = frozenset({'answer', 'workspace', 'proof', 'citations'})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _require_unique(values: tuple[str, ...], field: str) -> None:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f'{field} must contain non-empty strings')
    if len(values) != len(set(values)):
        raise ValueError(f'{field} must not contain duplicates')


@dataclass(frozen=True)
class TaskIntent:
    intent_id: str
    requested_profile_id: str
    task_form: str
    capabilities: tuple[str, ...]
    input_kinds: tuple[str, ...]
    output_kinds: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.intent_id or not self.task_form or not self.reason.strip():
            raise ValueError('intent id, task form, and reason must not be empty')
        _require_unique(self.capabilities, 'capabilities')
        _require_unique(self.input_kinds, 'input_kinds')
        _require_unique(self.output_kinds, 'output_kinds')
        unknown_inputs = set(self.input_kinds) - _INPUT_KINDS
        unknown_outputs = set(self.output_kinds) - _OUTPUT_KINDS
        if unknown_inputs:
            raise ValueError(f'unknown input kinds: {sorted(unknown_inputs)}')
        if unknown_outputs:
            raise ValueError(f'unknown output kinds: {sorted(unknown_outputs)}')


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    target_path: str
    sha256: str
    kind: str
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise ValueError('fixture id must not be empty')
        normalized = os.path.normpath(self.target_path)
        if (not self.target_path or os.path.isabs(self.target_path) or normalized in {'.', '..'}
                or normalized.startswith('../')):
            raise ValueError(f'fixture target must stay inside the workspace: {self.target_path!r}')
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError('fixture sha256 must be 64 lowercase hexadecimal characters')
        if self.kind not in _INPUT_KINDS:
            raise ValueError(f'unknown fixture kind {self.kind!r}')
        if not self.read_only:
            raise ValueError('task fixtures must be read-only')


@dataclass(frozen=True)
class VerifierSpec:
    kind: str
    config_json: str = '{}'

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError('verifier kind must not be empty')
        try:
            config = json.loads(self.config_json)
        except json.JSONDecodeError as error:
            raise ValueError(f'invalid verifier config JSON: {error}') from error
        if not isinstance(config, dict):
            raise ValueError('verifier config must be a JSON object')
        if 'kind' in config:
            raise ValueError("verifier config cannot override the reserved 'kind' field")
        if self.config_json != _canonical_json(config):
            raise ValueError('verifier config must use canonical JSON encoding')

    @classmethod
    def from_dict(cls, kind: str, config: dict[str, Any]) -> VerifierSpec:
        return cls(kind=kind, config_json=_canonical_json(config))

    def to_dict(self) -> dict[str, Any]:
        return {'kind': self.kind, **json.loads(self.config_json)}


@dataclass(frozen=True)
class FreshnessSpec:
    retrieved_at: str = ''
    expires_at: str = ''
    receipt_required: bool = False

    @property
    def required(self) -> bool:
        return bool(self.retrieved_at or self.expires_at or self.receipt_required)

    def __post_init__(self) -> None:
        if bool(self.retrieved_at) != bool(self.expires_at):
            raise ValueError('freshness requires both retrieved_at and expires_at')
        if self.retrieved_at:
            try:
                retrieved = datetime.fromisoformat(self.retrieved_at.replace('Z', '+00:00'))
                expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
            except ValueError as error:
                raise ValueError(f'invalid freshness timestamp: {error}') from error
            if retrieved.tzinfo is None or expires.tzinfo is None:
                raise ValueError('freshness timestamps must include a timezone')
            if expires <= retrieved:
                raise ValueError('freshness expiry must be later than retrieval')


@dataclass(frozen=True)
class ProfileProvenance:
    requested_profile_id: str
    resolved_profile_id: str
    profile_digest: str
    registry_digest: str
    router_version: str
    decision_reason: str

    @classmethod
    def from_resolution(cls, resolution: ProfileResolution) -> ProfileProvenance:
        return cls(
            requested_profile_id=resolution.requested_profile_id,
            resolved_profile_id=resolution.profile.profile_id,
            profile_digest=resolution.profile.digest,
            registry_digest=resolution.registry_digest,
            router_version=resolution.router_version,
            decision_reason=resolution.reason,
        )


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    intent_id: str
    query: str
    provenance: ProfileProvenance
    verifiers: tuple[VerifierSpec, ...]
    fixtures: tuple[FixtureSpec, ...] = ()
    output_kinds: tuple[str, ...] = ('answer', )
    freshness: FreshnessSpec = FreshnessSpec()
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError('new sandbox contracts require schema version 3')
        if not self.task_id or not self.intent_id or not self.query.strip():
            raise ValueError('task id, intent id, and query must not be empty')
        if not self.verifiers:
            raise ValueError('a task contract requires at least one verifier')
        _require_unique(self.output_kinds, 'output_kinds')
        paths = [fixture.target_path for fixture in self.fixtures]
        if len(paths) != len(set(paths)):
            raise ValueError('fixture target paths must be unique')

    @property
    def digest(self) -> str:
        payload = self.to_dict()
        return hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version':
            self.schema_version,
            'task_id':
            self.task_id,
            'intent_id':
            self.intent_id,
            'query':
            self.query,
            'profile_provenance': {
                'requested_profile_id': self.provenance.requested_profile_id,
                'resolved_profile_id': self.provenance.resolved_profile_id,
                'profile_digest': self.provenance.profile_digest,
                'registry_digest': self.provenance.registry_digest,
                'router_version': self.provenance.router_version,
                'decision_reason': self.provenance.decision_reason,
            },
            'verifiers': [verifier.to_dict() for verifier in self.verifiers],
            'fixtures': [{
                'fixture_id': fixture.fixture_id,
                'target_path': fixture.target_path,
                'sha256': fixture.sha256,
                'kind': fixture.kind,
                'read_only': fixture.read_only,
            } for fixture in self.fixtures],
            'output_kinds':
            list(self.output_kinds),
            'freshness': {
                'retrieved_at': self.freshness.retrieved_at,
                'expires_at': self.freshness.expires_at,
                'receipt_required': self.freshness.receipt_required,
            },
        }


def resolve_intent(intent: TaskIntent, registry: ProfileRegistry) -> ProfileResolution:
    resolution = registry.resolve(intent.capabilities, intent.task_form, intent.requested_profile_id)
    profile = resolution.profile
    unsupported_inputs = set(intent.input_kinds) - set(profile.input_kinds)
    unsupported_outputs = set(intent.output_kinds) - set(profile.output_kinds)
    if unsupported_inputs:
        raise ValueError(f'profile {profile.profile_id!r} does not accept inputs {sorted(unsupported_inputs)}')
    if unsupported_outputs:
        raise ValueError(f'profile {profile.profile_id!r} does not produce outputs {sorted(unsupported_outputs)}')
    return resolution


def _validate_provenance(provenance: ProfileProvenance, resolution: ProfileResolution) -> None:
    expected = ProfileProvenance.from_resolution(resolution)
    if provenance != expected:
        raise ValueError('task profile provenance does not match the resolved intent')


def validate_task_contract(contract: TaskContract, intent: TaskIntent, registry: ProfileRegistry) -> SandboxProfile:
    if contract.intent_id != intent.intent_id:
        raise ValueError('task contract refers to a different intent')
    resolution = resolve_intent(intent, registry)
    _validate_provenance(contract.provenance, resolution)
    profile = resolution.profile

    verifier_kinds = {verifier.kind for verifier in contract.verifiers}
    unsupported_verifiers = verifier_kinds - set(profile.verifier_kinds)
    if unsupported_verifiers:
        raise ValueError(f'profile {profile.profile_id!r} cannot run verifiers '
                         f'{sorted(unsupported_verifiers)}')
    fixture_kinds = {fixture.kind for fixture in contract.fixtures}
    unsupported_fixtures = fixture_kinds - set(profile.input_kinds)
    if unsupported_fixtures:
        raise ValueError(f'profile {profile.profile_id!r} cannot mount fixtures '
                         f'{sorted(unsupported_fixtures)}')
    if not set(contract.output_kinds).issubset(intent.output_kinds):
        raise ValueError('task contract outputs exceed the declared intent')
    if not set(contract.output_kinds).issubset(profile.output_kinds):
        raise ValueError('task contract outputs exceed the resolved profile')
    if profile.session_kind == 'no_tool' and contract.fixtures:
        raise ValueError('no-tool tasks cannot mount fixtures')
    if profile.name == 'web-research':
        if not contract.freshness.required:
            raise ValueError('web research tasks require a freshness policy')
        if 'web_receipt' not in verifier_kinds or not contract.freshness.receipt_required:
            raise ValueError('web research tasks require receipt verification')
    elif contract.freshness.required:
        raise ValueError('freshness policies are only valid for web research tasks')
    return profile
