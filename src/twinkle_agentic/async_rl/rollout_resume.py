# Copyright (c) ModelScope Contributors. All rights reserved.
"""Durable bookkeeping for pausing and resuming an in-flight RL rollout.

The checkpoint has two layers.  :class:`DurableJournal` records ordered state
transitions and repairs only an interrupted final line.  :class:`ResumeStore`
stores Python state blobs with a content hash and exposes one stable checkpoint
per episode state key.  The journal is intentionally independent of Ray and
torch; callers provide already-plain state or trusted process-local objects.
"""

from __future__ import annotations
import hashlib
import os
import pickle
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import json

_ZERO_DIGEST = "0" * 64
_OBJECT_MAGIC = b"RRS1"
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _thread_path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize one journal across threads and processes."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_path_lock(lock_path):
        with open(lock_path, "a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class ResumeError(RuntimeError):
    """Base error for an unsafe or inconsistent rollout resume."""


class UnsafeResume(ResumeError):
    """The current state cannot be resumed without repeating an external effect."""


class JournalCorrupt(ResumeError):
    """A journal is damaged somewhere other than its final incomplete line."""


class JournalEvent(str, Enum):
    PROMPT_ASSIGNED = "PROMPT_ASSIGNED"
    TURN_COMMITTED = "TURN_COMMITTED"
    TOOL_INTENT = "TOOL_INTENT"
    TOOL_RESULT = "TOOL_RESULT"
    SCORE_RESULT = "SCORE_RESULT"
    GROUP_READY = "GROUP_READY"
    GROUP_DROPPED = "GROUP_DROPPED"
    BATCH_LEASED = "BATCH_LEASED"
    MODEL_CHECKPOINT_COMMITTED = "MODEL_CHECKPOINT_COMMITTED"


@dataclass(frozen=True)
class JournalRecord:
    seq: int
    event: JournalEvent
    state_key: str
    payload: dict[str, Any]
    prev: str
    digest: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(record)).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a rename on filesystems that expose directory fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` with complete bytes; the previous file survives failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(pending, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
        _fsync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def _object_name(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".resume"


class DurableJournal:
    """Append-only, hash-chained JSONL journal with final-line repair.

    A torn final line is the expected result of a process dying after write but
    before fsync.  It is removed.  Damage followed by another complete record is
    never guessed at: that would silently resume from a fabricated history.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_seq = 0
        self._last_digest = _ZERO_DIGEST
        self._records: list[JournalRecord] = []
        with _exclusive_file_lock(self.path):
            self._repair_and_scan()

    def _repair_and_scan(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        offset = 0
        valid_end = 0
        records = []
        prev = _ZERO_DIGEST
        while offset < len(data):
            newline = data.find(b"\n", offset)
            if newline < 0:
                break
            raw = data[offset:newline]
            try:
                payload = json.loads(raw.decode("utf-8"))
                seq = int(payload["seq"])
                body = {
                    "seq": seq,
                    "event": payload["event"],
                    "state_key": payload["state_key"],
                    "payload": payload["payload"],
                    "prev": payload["prev"],
                }
                digest = str(payload["digest"])
                if seq != len(records) or body["prev"] != prev or _record_digest(body) != digest:
                    raise ValueError("journal hash chain mismatch")
                event = JournalEvent(body["event"])
                records.append(JournalRecord(seq, event, body["state_key"], body["payload"], prev, digest))
            except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                if data[newline + 1 :].strip():
                    raise JournalCorrupt(f"{self.path}: corrupt record at line {len(records) + 1}: {exc}") from exc
                break
            prev = records[-1].digest
            offset = newline + 1
            valid_end = offset
        if valid_end != len(data):
            with open(self.path, "r+b") as stream:
                stream.truncate(valid_end)
                stream.flush()
                os.fsync(stream.fileno())
        if records:
            self._next_seq = records[-1].seq + 1
            self._last_digest = records[-1].digest
        self._records = records

    def append(self, event: JournalEvent, state_key: str, payload: dict[str, Any] | None = None) -> JournalRecord:
        with self._lock:
            with _exclusive_file_lock(self.path):
                self._repair_and_scan()
                old_size = self.path.stat().st_size if self.path.exists() else 0
                body = {
                    "seq": self._next_seq,
                    "event": event.value,
                    "state_key": state_key,
                    "payload": payload or {},
                    "prev": self._last_digest,
                }
                digest = _record_digest(body)
                line = (
                    json.dumps({**body, "digest": digest}, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
                )
                try:
                    with open(self.path, "ab") as stream:
                        stream.write(line)
                        stream.flush()
                        os.fsync(stream.fileno())
                except BaseException:
                    try:
                        with open(self.path, "r+b") as stream:
                            stream.truncate(old_size)
                            stream.flush()
                            os.fsync(stream.fileno())
                    except OSError:
                        pass
                    self._repair_and_scan()
                    raise
                record = JournalRecord(self._next_seq, event, state_key, payload or {}, self._last_digest, digest)
                self._records.append(record)
                self._next_seq += 1
                self._last_digest = digest
                return record

    def records(self, *, state_key: str | None = None) -> Iterable[JournalRecord]:
        with self._lock:
            return tuple(record for record in self._records if state_key is None or record.state_key == state_key)

    def last(self, state_key: str, *, events: Iterable[JournalEvent] | None = None) -> JournalRecord | None:
        allowed = set(events) if events is not None else None
        found = None
        for record in self.records(state_key=state_key):
            if allowed is None or record.event in allowed:
                found = record
        return found


class ResumeStore:
    """Checkpoint objects and a transition journal for one training run."""

    def __init__(self, root: str | os.PathLike[str], run_id: str):
        if not run_id:
            raise ValueError("run_id must not be empty")
        self.root = Path(root)
        self.run_id = run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = DurableJournal(self.root / "rollout.jsonl")

    def save_object(self, scope: str, key: str, value: Any) -> str:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(payload).digest()
        path = self.root / "objects" / scope / _object_name(key)
        atomic_write_bytes(path, _OBJECT_MAGIC + digest + payload)
        return str(path)

    def has_object(self, scope: str, key: str) -> bool:
        return (self.root / "objects" / scope / _object_name(key)).is_file()

    def load_object(self, scope: str, key: str) -> Any:
        path = self.root / "objects" / scope / _object_name(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise ResumeError(f"resume object missing: {scope}/{key}") from exc
        if not data.startswith(_OBJECT_MAGIC) or len(data) < len(_OBJECT_MAGIC) + 32:
            raise ResumeError(f"resume object has an invalid header: {path}")
        digest = data[len(_OBJECT_MAGIC) : len(_OBJECT_MAGIC) + 32]
        payload = data[len(_OBJECT_MAGIC) + 32 :]
        if hashlib.sha256(payload).digest() != digest:
            raise ResumeError(f"resume object checksum mismatch: {path}")
        try:
            return pickle.loads(payload)
        except Exception as exc:
            raise ResumeError(f"resume object cannot be decoded: {path}") from exc

    def delete_object(self, scope: str, key: str) -> None:
        path = self.root / "objects" / scope / _object_name(key)
        path.unlink(missing_ok=True)

    def checkpoint_episode(self, state_key: str, state: Any) -> None:
        self.save_object("episodes", state_key, state)
        self.journal.append(JournalEvent.TURN_COMMITTED, state_key, {"object": _object_name(state_key)})

    def load_episode(self, state_key: str) -> Any:
        return self.load_object("episodes", state_key)

    def begin_effect(self, effect_key: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        last = self.journal.last(effect_key, events=(JournalEvent.TOOL_INTENT, JournalEvent.TOOL_RESULT))
        if last is not None and last.event == JournalEvent.TOOL_RESULT:
            return dict(last.payload)
        if last is not None and last.event == JournalEvent.TOOL_INTENT:
            raise UnsafeResume(f"effect {effect_key} has an intent without a result")
        self.journal.append(JournalEvent.TOOL_INTENT, effect_key, payload)
        return None

    def finish_effect(self, effect_key: str, payload: dict[str, Any]) -> None:
        self.journal.append(JournalEvent.TOOL_RESULT, effect_key, payload)

    def pending_effects(self, state_key: str) -> tuple[str, ...]:
        """Effect keys whose last record is an intent rather than a result."""
        prefix = state_key + "/effect/"
        pending: dict[str, bool] = {}
        for record in self.journal.records():
            if not record.state_key.startswith(prefix):
                continue
            if record.event == JournalEvent.TOOL_INTENT:
                pending[record.state_key] = True
            elif record.event == JournalEvent.TOOL_RESULT:
                pending[record.state_key] = False
        return tuple(key for key, is_pending in pending.items() if is_pending)

    def completed_effects(self, state_key: str) -> tuple[str, ...]:
        """Effect keys with a matching result record."""
        prefix = state_key + "/effect/"
        completed: set[str] = set()
        for record in self.journal.records():
            if record.state_key.startswith(prefix) and record.event == JournalEvent.TOOL_RESULT:
                completed.add(record.state_key)
        return tuple(sorted(completed))

    def assert_no_pending_effects(self, state_key: str) -> None:
        pending = self.pending_effects(state_key)
        if pending:
            raise UnsafeResume(f"episode {state_key} has pending tool effects: {pending!r}")
