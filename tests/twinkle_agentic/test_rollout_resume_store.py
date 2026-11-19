import json
import pytest

from twinkle_agentic.async_rl.rollout_resume import (
    DurableJournal,
    JournalCorrupt,
    JournalEvent,
    ResumeError,
    ResumeStore,
    UnsafeResume,
)


def test_journal_repairs_only_a_torn_final_line(tmp_path):
    path = tmp_path / "rollout.jsonl"
    journal = DurableJournal(path)
    journal.append(JournalEvent.PROMPT_ASSIGNED, "run/0", {"row": 0})
    journal.append(JournalEvent.GROUP_READY, "run/0", {"members": 2})
    with open(path, "ab") as stream:
        stream.write(b'{"seq":2')

    repaired = DurableJournal(path)
    record = repaired.append(JournalEvent.BATCH_LEASED, "run/0", {"handed": 1})
    assert record.seq == 2
    assert [item.event for item in repaired.records()] == [
        JournalEvent.PROMPT_ASSIGNED,
        JournalEvent.GROUP_READY,
        JournalEvent.BATCH_LEASED,
    ]


def test_journal_refuses_damage_before_a_complete_later_record(tmp_path):
    path = tmp_path / "rollout.jsonl"
    journal = DurableJournal(path)
    first = journal.append(JournalEvent.PROMPT_ASSIGNED, "run/0", {"row": 0})
    journal.append(JournalEvent.GROUP_READY, "run/0", {"members": 2})
    lines = path.read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[0])
    broken["payload"]["row"] = 99
    lines[0] = json.dumps(broken, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(JournalCorrupt):
        DurableJournal(path)
    assert first.seq == 0


def test_journal_refuses_a_corrupt_complete_final_commit(tmp_path):
    path = tmp_path / "rollout.jsonl"
    journal = DurableJournal(path)
    journal.append(JournalEvent.BATCH_LEASED, "run/batch/0")
    journal.append(JournalEvent.MODEL_CHECKPOINT_COMMITTED, "run/batch/0", {"checkpoint": "step-1"})
    lines = path.read_text(encoding="utf-8").splitlines()
    commit = json.loads(lines[-1])
    commit["payload"]["checkpoint"] = "corrupted"
    lines[-1] = json.dumps(commit, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(JournalCorrupt, match="corrupt record at line 2"):
        DurableJournal(path)


def test_resume_store_checks_object_checksum(tmp_path):
    store = ResumeStore(tmp_path, "run")
    store.save_object("episodes", "run/episode/0", {"tokens": [1, 2, 3]})
    assert store.load_object("episodes", "run/episode/0") == {"tokens": [1, 2, 3]}

    object_path = next((tmp_path / "objects" / "episodes").iterdir())
    data = bytearray(object_path.read_bytes())
    data[-1] ^= 0xFF
    object_path.write_bytes(data)
    with pytest.raises(ResumeError, match="checksum"):
        store.load_object("episodes", "run/episode/0")


def test_effect_intent_without_result_fails_closed_then_result_is_replayable(tmp_path):
    store = ResumeStore(tmp_path, "run")
    effect = "run/group-0/episode-0/turn-1"
    assert store.begin_effect(effect, {"calls": ["write_file"]}) is None
    with pytest.raises(UnsafeResume, match="intent without a result"):
        store.begin_effect(effect, {"calls": ["write_file"]})

    store.finish_effect(effect, {"observations": ["File written"]})
    assert store.begin_effect(effect, {"calls": ["ignored"]}) == {"observations": ["File written"]}
