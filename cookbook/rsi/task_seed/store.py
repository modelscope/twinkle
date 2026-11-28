# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = 1
FINAL_RECORD_KEYS = {
    'schema_version',
    'global_index',
    'id',
    'text',
    'task_direction',
    'required_artifact',
    'constraints',
    'eos',
    'token_ids',
    'inversion_initial_loss',
    'inversion_final_loss',
    'target_reembedded_cosine',
    'initial_decoded_cosine',
    'prefix_sha256',
    'seed',
    'actual_batch_size',
    'provenance',
}
PROVENANCE_KEYS = {
    'model_hash',
    'tokenizer_hash',
    'config_hash',
    'plan_hash',
    'shard_id',
    'num_shards',
}
MANIFEST_KEYS = {
    'schema_version',
    'release_status',
    'created_at_utc',
    'bank',
    'model',
    'tokenizer',
    'config',
    'config_sha256',
    'planner',
    'coverage',
    'shards',
    'validation_report',
}
READY_KEYS = {'schema_version', 'manifest_sha256', 'bank_sha256'}
_HEX64 = re.compile(r'^[0-9a-f]{64}$')


class ValidationError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b'\n'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(config))


def stable_seed(*parts: object) -> int:
    payload = '\x1f'.join(str(part) for part in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & 0x7FFF_FFFF_FFFF_FFFF


def stable_task_id(text: str) -> str:
    return 'ts_' + sha256_bytes(text.encode('utf-8'))[:24]


def atomic_write_bytes(path: os.PathLike[str] | str, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_line(value))


def atomic_write_jsonl(path: os.PathLike[str] | str, records: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_bytes(path, b''.join(canonical_json_line(record) for record in records))


def load_json(path: os.PathLike[str] | str) -> Any:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f'invalid JSON in {path}: {exc}') from exc
    if canonical_json_line(value) != raw:
        raise ValidationError(f'JSON is not canonical in {path}')
    return value


def iter_strict_jsonl(path: os.PathLike[str] | str) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, 'rb') as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b'\n'):
                raise ValidationError(f'unterminated JSONL line {line_number} in {path}')
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValidationError(f'invalid JSONL line {line_number} in {path}: {exc}') from exc
            if not isinstance(record, dict):
                raise ValidationError(f'JSONL line {line_number} is not an object')
            if canonical_json_line(record) != raw:
                raise ValidationError(f'JSONL line {line_number} is not canonical')
            yield line_number, record


def _hash_file_set(root: Path, files: Sequence[Path]) -> str:
    if not files:
        raise ValidationError(f'no fingerprintable files under {root}')
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        item = {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
        digest.update(canonical_json_line(item))
    return digest.hexdigest()


def model_tree_hash(model_path: os.PathLike[str] | str) -> str:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValidationError(f'model path is not a directory: {root}')
    files = [
        path for path in root.rglob('*') if path.is_file() and (
            path.suffix in {'.safetensors', '.bin', '.pt'} or path.name in {'config.json', 'generation_config.json'})
    ]
    return _hash_file_set(root, files)


def tokenizer_tree_hash(model_path: os.PathLike[str] | str) -> str:
    root = Path(model_path).resolve()
    names = {
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'added_tokens.json',
        'vocab.json',
        'merges.txt',
        'spiece.model',
        'tokenizer.model',
    }
    files = [path for path in root.rglob('*') if path.is_file() and (path.name in names or 'tokenizer' in path.name)]
    return _hash_file_set(root, files)


def validate_source_record(record: Mapping[str, Any], line_number: int) -> None:
    if set(record) != {'source_id', 'text'}:
        raise ValidationError(f'input line {line_number} must contain exactly source_id and text')
    if not isinstance(record['source_id'], (str, int)) or isinstance(record['source_id'], bool):
        raise ValidationError(f'input line {line_number} has invalid source_id')
    if not isinstance(record['text'], str) or not record['text'].strip():
        raise ValidationError(f'input line {line_number} has empty text')


def validate_decoded_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {'task_direction', 'required_artifact', 'constraints'}:
        raise ValidationError('decoded JSON must contain exactly task_direction, required_artifact, constraints')
    for key in ('task_direction', 'required_artifact'):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValidationError(f'decoded {key} must be a non-empty string')
    constraints = payload['constraints']
    if not isinstance(constraints, list) or not constraints:
        raise ValidationError('decoded constraints must be a non-empty list')
    if any(not isinstance(item, str) or not item.strip() for item in constraints):
        raise ValidationError('every decoded constraint must be a non-empty string')
    return payload


def task_text(payload: Mapping[str, Any]) -> str:
    validate_decoded_payload(payload)
    return canonical_json_bytes(payload).decode('utf-8')


def validate_final_record(record: Mapping[str, Any], expected_index: int | None = None) -> None:
    if set(record) != FINAL_RECORD_KEYS:
        raise ValidationError(f'final record keys differ: {sorted(set(record) ^ FINAL_RECORD_KEYS)}')
    if record['schema_version'] != SCHEMA_VERSION:
        raise ValidationError('unsupported record schema_version')
    index = record['global_index']
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValidationError('global_index must be a non-negative integer')
    if expected_index is not None and index != expected_index:
        raise ValidationError(f'expected global_index {expected_index}, got {index}')
    payload = {key: record[key] for key in ('task_direction', 'required_artifact', 'constraints')}
    validate_decoded_payload(payload)
    text = task_text(payload)
    if record['text'] != text or record['id'] != stable_task_id(text):
        raise ValidationError('record text/id does not authenticate decoded payload')
    if record['eos'] is not True:
        raise ValidationError('record did not hit EOS')
    if not isinstance(record['token_ids'], list) or not record['token_ids'] or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in record['token_ids']):
        raise ValidationError('token_ids must be a non-empty integer list')
    for key in (
            'inversion_initial_loss',
            'inversion_final_loss',
            'target_reembedded_cosine',
            'initial_decoded_cosine',
    ):
        if not isinstance(record[key], (int, float)) or isinstance(record[key], bool):
            raise ValidationError(f'{key} must be numeric')
        if not float('-inf') < float(record[key]) < float('inf'):
            raise ValidationError(f'{key} must be finite')
    if not isinstance(record['seed'], int) or isinstance(record['seed'], bool):
        raise ValidationError('seed must be an integer')
    if not isinstance(record['actual_batch_size'], int) or record['actual_batch_size'] < 1:
        raise ValidationError('actual_batch_size must be positive')
    if not isinstance(record['prefix_sha256'], str) or not _HEX64.fullmatch(record['prefix_sha256']):
        raise ValidationError('invalid prefix_sha256')
    provenance = record['provenance']
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise ValidationError('invalid record provenance keys')
    for key in ('model_hash', 'tokenizer_hash', 'config_hash', 'plan_hash'):
        if not isinstance(provenance[key], str) or not _HEX64.fullmatch(provenance[key]):
            raise ValidationError(f'invalid provenance {key}')
    if not isinstance(provenance['shard_id'], int) or not isinstance(provenance['num_shards'], int):
        raise ValidationError('invalid provenance shard identifiers')
    if provenance['num_shards'] != 8 or not 0 <= provenance['shard_id'] < 8:
        raise ValidationError('final records require exactly eight valid shards')
    if index % 8 != provenance['shard_id']:
        raise ValidationError('record is assigned to the wrong shard')


def _expect_file(root: Path, descriptor: Any, expected_path: str) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {'path', 'sha256'}:
        raise ValidationError(f'invalid descriptor for {expected_path}')
    if descriptor['path'] != expected_path or not _HEX64.fullmatch(str(descriptor['sha256'])):
        raise ValidationError(f'invalid descriptor values for {expected_path}')
    path = root / expected_path
    if not path.is_file() or sha256_file(path) != descriptor['sha256']:
        raise ValidationError(f'hash mismatch for {path}')
    return path


class TaskSeedStore(Sequence[dict[str, Any]]):

    def __init__(self, directory: os.PathLike[str] | str):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / 'manifest.json'
        bank_path = self.directory / 'task_seeds.jsonl'
        ready_path = self.directory / 'READY'
        if not manifest_path.is_file() or not bank_path.is_file() or not ready_path.is_file():
            raise ValidationError('task seed release is not finalized')
        manifest = load_json(manifest_path)
        ready = load_json(ready_path)
        if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
            raise ValidationError('invalid manifest schema')
        if manifest['schema_version'] != SCHEMA_VERSION or manifest['release_status'] != 'accepted':
            raise ValidationError('manifest release is not accepted schema version 1')
        if not isinstance(ready, dict) or set(ready) != READY_KEYS or ready['schema_version'] != SCHEMA_VERSION:
            raise ValidationError('invalid READY schema')
        manifest_sha = sha256_file(manifest_path)
        bank_sha = sha256_file(bank_path)
        if ready['manifest_sha256'] != manifest_sha or ready['bank_sha256'] != bank_sha:
            raise ValidationError('READY authentication failed')
        bank = manifest['bank']
        if not isinstance(bank, dict) or set(bank) != {'path', 'sha256', 'bytes', 'records'}:
            raise ValidationError('invalid bank manifest')
        if bank['path'] != 'task_seeds.jsonl' or bank['sha256'] != bank_sha:
            raise ValidationError('bank hash mismatch')
        if bank['bytes'] != bank_path.stat().st_size or not isinstance(bank['records'], int) or bank['records'] < 1:
            raise ValidationError('bank size/count mismatch')
        if not isinstance(manifest['config'], dict) or config_hash(manifest['config']) != manifest['config_sha256']:
            raise ValidationError('config hash mismatch')
        model = manifest['model']
        tokenizer = manifest['tokenizer']
        if not isinstance(model, dict) or set(model) != {'path', 'sha256'}:
            raise ValidationError('invalid model manifest')
        if not isinstance(tokenizer, dict) or set(tokenizer) != {'path', 'sha256'}:
            raise ValidationError('invalid tokenizer manifest')
        if model_tree_hash(model['path']) != model['sha256']:
            raise ValidationError('model hash mismatch')
        if tokenizer['path'] != model['path'] or tokenizer_tree_hash(tokenizer['path']) != tokenizer['sha256']:
            raise ValidationError('tokenizer hash mismatch')
        _expect_file(self.directory, manifest['planner'], 'plan/plan.pt')
        _expect_file(self.directory, manifest['validation_report'], 'validation_report.json')
        records: list[dict[str, Any]] = []
        ids: set[str] = set()
        texts: set[str] = set()
        provenance_reference: tuple[str, str, str, str] | None = None
        for expected_index, (_, record) in enumerate(iter_strict_jsonl(bank_path)):
            validate_final_record(record, expected_index)
            if record['id'] in ids or record['text'] in texts:
                raise ValidationError('duplicate task seed id/text')
            ids.add(record['id'])
            texts.add(record['text'])
            provenance = record['provenance']
            current = tuple(provenance[key] for key in ('model_hash', 'tokenizer_hash', 'config_hash', 'plan_hash'))
            if provenance_reference is None:
                provenance_reference = current
            elif provenance_reference != current:
                raise ValidationError('inconsistent record provenance')
            records.append(record)
        if len(records) != bank['records']:
            raise ValidationError('record count mismatch')
        assert provenance_reference is not None
        expected = (
            model['sha256'],
            tokenizer['sha256'],
            manifest['config_sha256'],
            manifest['planner']['sha256'],
        )
        if provenance_reference != expected:
            raise ValidationError('manifest and record provenance differ')
        self.path = str(self.directory)
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha
        self.bank_sha256 = bank_sha
        self.model_sha256 = model['sha256']
        self.tokenizer_sha256 = tokenizer['sha256']
        self.config_sha256 = manifest['config_sha256']
        self._records = tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        return self._records[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._records)

    def select(self, group_id: int, seed: int, epoch: int = 0) -> dict[str, Any]:
        if not isinstance(group_id, int) or group_id < 0:
            raise ValueError('group_id must be a non-negative integer')
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError('epoch must be a non-negative integer')
        offset = stable_seed(seed, 'task-seed-offset') % len(self._records)
        epoch_offset = stable_seed(seed, 'task-seed-epoch', epoch) % len(self._records)
        return self._records[(offset + epoch_offset + group_id) % len(self._records)]
