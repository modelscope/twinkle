# Copyright (c) ModelScope Contributors. All rights reserved.
"""Choosing which sandbox a topic runs in.

A profile registry says what environments exist. This says which one a card
should be routed to, and -- the part that is easy to get wrong -- which ones this
machine can run at all. A profile whose backing service is absent does not raise;
it scores zero on every group routed to it, which reads as "these topics are too
hard" rather than as "the database was not there".
"""
import json

import pytest

from twinkle_agentic.envs import TopicProfiles, default_profile_registry
from twinkle_agentic.envs import topic_profile as tp


class Card:
    """The two attributes ``for_card`` reads, and nothing else."""

    def __init__(self, card_id='c0', source_cue='root > leaf'):
        self.card_id = card_id
        self.source_cue = source_cue


def write_table(path, records):
    path.write_text('\n'.join(json.dumps(record) for record in records) + '\n', encoding='utf-8')
    return str(path)


@pytest.fixture
def affinities(tmp_path):
    """Two cards: one with a real choice, one that offers only the baseline."""
    return write_table(
        tmp_path / 'affinities.jsonl', [
            {
                'card_id': 'c-sql',
                'topic_path': {
                    'root': 'Databases',
                    'leaf': 'Query plans'
                },
                'eligible_profiles': ['python-compute@1', 'sql-analytics@1'],
                'advisory_only': True,
            },
            {
                'card_id': 'c-prose',
                'topic_path': {
                    'root': 'Ethics',
                    'leaf': 'Disclosure'
                },
                'eligible_profiles': ['knowledge-answer@1'],
                'advisory_only': True,
            },
        ])


# ------------------------------------------------------------- the static lists
def test_every_switchable_profile_exists_in_the_registry():
    # The list is hand-maintained and the registry is not, so drift is the
    # expected failure. A name here that is not there would raise on switch.
    registry = default_profile_registry()
    for profile_id in tp.SWITCHABLE:
        assert registry.get(profile_id).profile_id == profile_id


def test_the_baseline_and_lean_are_not_switchable_and_say_why():
    assert tp.BASELINE not in tp.SWITCHABLE
    assert 'lean-proof@1' not in tp.SWITCHABLE
    assert {name for name, _ in tp.UNAVAILABLE} == {tp.BASELINE, 'lean-proof@1'}
    for _, reason in tp.UNAVAILABLE:
        assert reason


def test_preference_covers_the_baseline_last():
    # Last is the point: PREFERENCE is the tiebreak order, and choosing the
    # baseline means the topic offered nothing more specific.
    assert tp.PREFERENCE[-1] == tp.BASELINE
    assert set(tp.SWITCHABLE) <= set(tp.PREFERENCE)


# ----------------------------------------------------------------- reachability
def test_a_service_backed_profile_is_dropped_when_nothing_is_listening(monkeypatch):
    # Port 1 is reserved and nothing binds it, so the connect fails immediately
    # rather than waiting out the timeout.
    monkeypatch.setenv('TWINKLE_WORLD_DB_HOST', '127.0.0.1')
    monkeypatch.setenv('TWINKLE_WORLD_DB_PORT', '1')
    profiles = TopicProfiles()
    assert 'db-analyst@1' in profiles.unreachable
    assert 'db-analyst@1' not in profiles.switchable
    assert 'db-analyst@1' not in profiles.rotation()
    # The sandbox-only profile that answers the same task form survives, which is
    # the whole reason the probe exists.
    assert 'sql-analytics@1' in profiles.rotation()


def test_the_probe_reads_the_same_host_the_world_connects_to(monkeypatch):
    # A probe that read a different variable would test one host and query
    # another, and the run would fail on every group instead of at boot.
    monkeypatch.setenv('TWINKLE_WORLD_DB_HOST', '127.0.0.1')
    monkeypatch.setenv('TWINKLE_WORLD_DB_PORT', '1')
    assert tp._service_reachable('db-analyst@1', timeout=0.5) is False


def test_a_profile_with_no_service_is_reachable_without_probing():
    assert tp._service_reachable('sql-analytics@1') is True
    assert tp._service_reachable('no-such-profile@1') is True


def test_rotation_is_preference_order_minus_the_unreachable():
    rotation = TopicProfiles().rotation()
    assert rotation == [profile for profile in tp.PREFERENCE if profile in set(rotation)]
    assert len(rotation) == len(set(rotation))


# -------------------------------------------------------------- the description
def test_starting_state_prefers_host_over_seeded_over_empty():
    assert tp.starting_state('world-forensics@1') == tp.START_HOST
    assert tp.starting_state('web-research@1') == tp.START_HOST
    assert tp.starting_state('sql-analytics@1') == tp.START_SEEDED
    assert tp.starting_state('python-compute@1') == tp.START_EMPTY


def test_an_unknown_or_absent_profile_describes_an_empty_workspace():
    assert tp.starting_state(None) == tp.START_EMPTY
    assert tp.starting_state('not-a-profile@9') == tp.START_EMPTY
    assert tp.host_tools_of(None) == ()
    assert tp.seed_steps_of(None) == ()


def test_host_tools_and_seed_steps_come_from_the_registry():
    assert tp.host_tools_of('world-forensics@1') == ('world_sql', 'world_schema', 'world_doc')
    assert tp.host_tools_of('sql-analytics@1') == ()
    assert tp.seed_steps_of('sql-analytics@1') == ('seed-sql-analytics', )


# ------------------------------------------------------------------ the table
def test_no_table_configured_means_no_choice(monkeypatch):
    monkeypatch.delenv('TWINKLE_TOPIC_AFFINITIES', raising=False)
    profiles = TopicProfiles()
    assert profiles.loaded is False
    assert profiles.describe() == {}
    assert profiles.for_card(Card()) is None


def test_a_missing_table_warns_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv('TWINKLE_TOPIC_AFFINITIES', str(tmp_path / 'absent.jsonl'))
    profiles = TopicProfiles()
    assert profiles.loaded is False
    assert profiles.for_card(Card('c-sql')) is None


def test_a_card_is_routed_to_its_most_specific_eligible_profile(affinities):
    # The table lists python-compute first, but PREFERENCE ranks the database
    # profile above it. Ranking compute first is how a bank built to exercise
    # databases ends up producing nothing but "write a CSV" tasks.
    profiles = TopicProfiles(affinities)
    assert profiles.loaded is True
    assert profiles.for_card(Card('c-sql')) == 'sql-analytics@1'


def test_a_card_offering_only_the_baseline_stays_where_it_is(affinities):
    # None, not the baseline's name: knowledge-answer is a no_tool session, so
    # switching into it would fail, and "do not switch" is the answer the caller
    # can act on.
    assert TopicProfiles(affinities).for_card(Card('c-prose')) is None


def test_an_unknown_card_falls_back_to_its_topic_path(affinities):
    profiles = TopicProfiles(affinities)
    assert profiles.for_card(Card('never-seen', 'Databases > Query plans')) == 'sql-analytics@1'
    assert profiles.for_card(Card('never-seen', 'Nothing > Matches')) is None


def test_a_profile_this_machine_cannot_run_is_not_chosen(tmp_path, monkeypatch):
    monkeypatch.setenv('TWINKLE_WORLD_DB_PORT', '1')
    table = write_table(
        tmp_path / 'db.jsonl',
        [{
            'card_id': 'c-db',
            'topic_path': {
                'root': 'Databases',
                'leaf': 'Grants'
            },
            'eligible_profiles': ['db-analyst@1', 'sql-analytics@1'],
        }])
    profiles = TopicProfiles(table)
    assert profiles.for_card(Card('c-db')) == 'sql-analytics@1'


def test_a_card_naming_only_unswitchable_profiles_is_skipped(tmp_path):
    table = write_table(tmp_path / 'a.jsonl', [{
        'card_id': 'c-lean',
        'eligible_profiles': ['lean-proof@1', 'knowledge-answer@1'],
    }])
    profiles = TopicProfiles(table)
    assert profiles.loaded is True
    assert profiles.for_card(Card('c-lean')) is None


def test_a_malformed_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / 'a.jsonl'
    path.write_text('{not json\n\n' + json.dumps({'card_id': 'c-ok',
                                                  'eligible_profiles': ['sql-analytics@1']}) + '\n',
                     encoding='utf-8')
    profiles = TopicProfiles(str(path))
    assert profiles.for_card(Card('c-ok')) == 'sql-analytics@1'


def test_the_table_is_read_from_the_environment(monkeypatch, affinities):
    monkeypatch.setenv('TWINKLE_TOPIC_AFFINITIES', affinities)
    assert TopicProfiles().for_card(Card('c-sql')) == 'sql-analytics@1'


def test_describe_counts_what_each_card_would_be_routed_to(affinities):
    counts = TopicProfiles(affinities).describe()
    assert counts['sql-analytics@1'] == 1
    # The prose card chose nothing, and describe() says so rather than hiding it
    # under the baseline.
    assert counts['stay-on-slot'] == 1


def test_a_listening_server_without_a_provisioning_module_is_not_reachable(monkeypatch):
    # The bug this pins: a TCP connect answers "is there a server", not "can we
    # authenticate to it". With the server up and TWINKLE_WORLD_DB_MODULE unset the
    # profile used to stay in the rotation, every db_query returned "cannot reach
    # the shared database", every rollout scored 0, and the group was dropped --
    # 91 of 332 switchable cards went there, so a quarter of the groups were
    # burned and the proposer learned that database topics do not pay.
    monkeypatch.setenv('TWINKLE_WORLD_DB_HOST', '127.0.0.1')
    monkeypatch.setenv('TWINKLE_WORLD_DB_PORT', '5432')
    monkeypatch.delenv('TWINKLE_WORLD_DB_MODULE', raising=False)
    import socket
    try:
        with socket.create_connection(('127.0.0.1', 5432), timeout=0.5):
            server_up = True
    except OSError:
        server_up = False
    if not server_up:
        pytest.skip('nothing listening on 127.0.0.1:5432, so "server up but module missing" cannot be set up')
    # Server up, module missing -> unreachable. This is the case that used to pass.
    assert tp._service_reachable('db-analyst@1', timeout=0.5) is False
    assert 'db-analyst@1' not in TopicProfiles().rotation()
    # Module named (its contents are not imported here, only its presence checked)
    # and server up -> reachable.
    monkeypatch.setenv('TWINKLE_WORLD_DB_MODULE', '/nonexistent-but-set')
    assert tp._service_reachable('db-analyst@1', timeout=0.5) is True
