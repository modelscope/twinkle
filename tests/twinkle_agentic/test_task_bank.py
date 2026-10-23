# Copyright (c) ModelScope Contributors. All rights reserved.
"""The bank of tasks a run has already kept.

``twinkle_agentic.verifier.rubric_score`` judges novelty against a reference set
and skips the criteria that need one when it is empty -- it consumes references
but nothing in the package produces them. TaskBank is that producer: the durable
record of kept task statements, and the nearest-neighbour lookup that turns it
into the reference set one group is scored against.
"""
import json

from twinkle_agentic.challenger import TaskBank
from twinkle_agentic.challenger.task_bank import grams, jaccard_3gram

# ------------------------------------------------------------------ TaskBank
def test_grams_and_jaccard():
    a = grams('the quick brown fox')
    b = grams('the quick brown fox')
    c = grams('completely different words here')
    assert jaccard_3gram(a, b) == 1.0
    assert jaccard_3gram(a, c) < 0.2


def test_add_then_len_and_statements(tmp_path):
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    assert len(bank) == 0
    assert bank.add('Summarise the auth incidents.', 'assert 1') is True
    assert len(bank) == 1
    assert bank.statements() == ['Summarise the auth incidents.']


def test_a_verbatim_repeat_is_refused(tmp_path):
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    bank.add('Count the p1 incidents per service.', 'assert 1')
    assert bank.add('Count the p1 incidents per service.', 'assert 2') is False
    assert len(bank) == 1


def test_the_bank_survives_a_reload(tmp_path):
    path = str(tmp_path / 'bank.jsonl')
    first = TaskBank(path)
    first.add('Task one about the ledger.', 'assert 1')
    first.add('Task two about the cache.', 'assert 1')
    second = TaskBank(path)
    assert len(second) == 2
    assert second.statements() == first.statements()


def test_references_returns_the_nearest_existing_tasks(tmp_path):
    bank = TaskBank(path=str(tmp_path / 'bank.jsonl'), refs=2)
    bank.add('Count the p1 auth incidents per service from the ops database.', 'a')
    bank.add('List every billing migration step in chronological order.', 'a')
    bank.add('Summarise the cache eviction policy from the runbook.', 'a')
    got = bank.references('Count the p2 auth incidents per service from the ops database.')
    assert len(got) <= 2
    assert any('auth incidents' in g for g in got)


def test_references_honours_extra_in_flight_siblings(tmp_path):
    # A group's own siblings are the references that matter most: a whole group
    # can look novel against history while being eight versions of one idea, and
    # GRPO subtracts the group mean, so a term identical across the group
    # produces no gradient at all.
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    got = bank.references('Anything at all.', extra=['sibling one', 'sibling two'])
    assert 'sibling one' in got and 'sibling two' in got


def test_sibling_groups_finds_near_duplicates(tmp_path):
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    bank.add('Count the p1 auth incidents per service.', 'a')
    bank.add('Count the p1 auth incidents per service.', 'a')  # refused as a repeat
    bank.add('Count the p2 auth incidents per service.', 'a')
    bank.add('Something entirely unrelated about cooking.', 'a')
    groups = bank.sibling_groups(minimum=2)
    assert any(len(g) >= 2 for g in groups)


def test_a_corrupt_line_does_not_take_the_bank_down(tmp_path):
    path = tmp_path / 'bank.jsonl'
    path.write_text('{"statement": "kept", "check": "a"}\nnot json at all\n', encoding='utf-8')
    bank = TaskBank(str(path))
    assert bank.statements() == ['kept']


def test_stats_reports_what_a_reader_needs(tmp_path):
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    bank.add('One task.', 'a')
    stats = bank.stats()
    assert isinstance(stats, dict)
    assert stats.get('total') == 1 or stats.get('n') == 1 or len(stats) > 0


# ------------------------------------------------------------- ingest_recorder
def write_index(save_dir, records):
    """The file AgenticChallenger's RolloutRecorder appends to, one record per line."""
    trajs = save_dir / 'trajs'
    trajs.mkdir(parents=True, exist_ok=True)
    (trajs / 'index.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
    return str(save_dir)


def test_ingest_recorder_takes_the_kept_proposals(tmp_path):
    save_dir = write_index(tmp_path / 'run', [
        {
            'statement': 'Count the p1 incidents per service.',
            'check_script': 'assert 1',
            'outcome': 'kept',
            'group_id': 0
        },
        {
            'statement': 'A proposal whose check never ran.',
            'check_script': '',
            'outcome': 'rejected',
            'group_id': 0
        },
        {
            'statement': 'A task every attempt solved.',
            'check_script': 'assert 1',
            'outcome': 'outside_band',
            'group_id': 1
        },
    ])
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    assert bank.ingest_recorder(save_dir) == 1
    assert bank.statements() == ['Count the p1 incidents per service.']


def test_ingest_recorder_is_idempotent_so_a_loop_can_call_it_every_step(tmp_path):
    # The recorder appends while the loop runs, so the natural call site is once
    # per iteration over the same file. Re-reading must not duplicate.
    save_dir = write_index(tmp_path / 'run', [{
        'statement': 'Count the p1 incidents.',
        'check_script': 'assert 1',
        'outcome': 'kept',
        'group_id': 0
    }])
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    assert bank.ingest_recorder(save_dir) == 1
    assert bank.ingest_recorder(save_dir) == 0
    assert len(bank) == 1


def test_ingest_recorder_survives_a_missing_or_partial_index(tmp_path):
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    assert bank.ingest_recorder(str(tmp_path / 'never-ran')) == 0
    trajs = tmp_path / 'run' / 'trajs'
    trajs.mkdir(parents=True)
    (trajs / 'index.jsonl').write_text('{"statement": "Whole task.", "outcome": "kept"}\n{"trunc', encoding='utf-8')
    assert bank.ingest_recorder(str(tmp_path / 'run')) == 1
    assert bank.statements() == ['Whole task.']


def test_ingest_recorder_keeps_the_check_and_group(tmp_path):
    # group_id is what sibling_groups clusters on, and the check is what a novelty
    # judge compares task shapes with, so neither may be dropped on the way in.
    save_dir = write_index(tmp_path / 'run', [{
        'statement': 'Derive one file from the others.',
        'check_script': 'assert open("out").read()',
        'outcome': 'kept',
        'group_id': 7
    }])
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    bank.ingest_recorder(save_dir)
    reloaded = TaskBank(str(tmp_path / 'bank.jsonl'))
    assert reloaded.statements() == ['Derive one file from the others.']
    assert reloaded.sibling_groups(minimum=1) == [['Derive one file from the others.']]


def test_ingest_recorder_can_be_asked_for_other_outcomes(tmp_path):
    save_dir = write_index(tmp_path / 'run', [{
        'statement': 'A task every attempt solved.',
        'outcome': 'outside_band'
    }])
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    assert bank.ingest_recorder(save_dir) == 0
    assert bank.ingest_recorder(save_dir, outcomes=('outside_band', )) == 1
