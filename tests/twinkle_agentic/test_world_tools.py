# Copyright (c) ModelScope Contributors. All rights reserved.
"""Host-side world tools mixed into an ``Env``.

Every test here runs against ``LocalEnv``, which needs no sandbox host: the
point of the mixin being a mixin rather than a change to ``AgentEnv`` is that
the host half can be verified without the sandbox half.
"""
import hashlib
import json
import sqlite3

import pytest

from twinkle_agentic.envs import HostWorld, LocalEnv, WorldToolsMixin, WorldUnavailable


class WorldLocalEnv(WorldToolsMixin, LocalEnv):
    """The mixin goes first: it has to precede the env in the MRO."""


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def world_root(tmp_path):
    """A minimal world: one table, one document, a manifest that matches."""
    root = tmp_path / 'world'
    (root / 'docs').mkdir(parents=True)
    conn = sqlite3.connect(root / 'ops.db')
    conn.execute('CREATE TABLE incidents (id INTEGER PRIMARY KEY, service TEXT, severity TEXT)')
    conn.executemany('INSERT INTO incidents (service, severity) VALUES (?, ?)',
                     [('auth', 'p1'), ('auth', 'p2'), ('billing', 'p1')])
    conn.commit()
    conn.close()
    (root / 'docs' / 'runbook.md').write_text('# Runbook\npage one\n', encoding='utf-8')
    (root / 'manifest.json').write_text(
        json.dumps({'files': {'ops.db': _sha(root / 'ops.db')}}), encoding='utf-8')
    return root


@pytest.fixture
def search_root(tmp_path):
    """A two-document frozen corpus, with a manifest that matches."""
    root = tmp_path / 'search'
    root.mkdir()
    documents = [
        {
            'url': 'https://example.com/a',
            'title': 'Auth outage postmortem',
            'published': '2026-09-01',
            'body': 'The auth service returned 500 for eleven minutes. A leaked pool.'
        },
        {
            'url': 'https://example.com/b',
            'title': 'Billing migration notes',
            'published': '2025-01-01',
            'body': 'Billing moved to a new ledger. Old rows are read only.'
        },
    ]
    (root / 'corpus.json').write_text(json.dumps({'documents': documents}), encoding='utf-8')
    (root / 'manifest.json').write_text(
        json.dumps({'files': {'corpus.json': _sha(root / 'corpus.json')}}), encoding='utf-8')
    return root


@pytest.fixture
def env(world_root, tmp_path, monkeypatch):
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    monkeypatch.delenv('TWINKLE_WORLD_SEARCH_ROOT', raising=False)
    world = HostWorld()
    world.load()
    return WorldLocalEnv(workspace=str(tmp_path / 'slot'), world=world)


# ------------------------------------------------------------------ the world


def test_unconfigured_world_is_unavailable(monkeypatch):
    # A library does not guess where its data lives. Without a root the world
    # reports itself unavailable rather than reading some baked-in path.
    monkeypatch.delenv('TWINKLE_WORLD_ROOT', raising=False)
    world = HostWorld()
    assert world.available is False
    assert world.schemas() == []
    with pytest.raises(WorldUnavailable, match='TWINKLE_WORLD_ROOT'):
        world.load()


def test_load_rejects_a_world_that_drifted(world_root):
    (world_root / 'ops.db').write_bytes(b'not the database the manifest hashed')
    with pytest.raises(WorldUnavailable, match='does not match the manifest'):
        HostWorld(world_root).load()


def test_schemas_advertise_only_what_is_configured(env, search_root, monkeypatch):
    # No search corpus and no database: the model is offered three tools, not
    # seven. Advertising one that always answers "not configured" costs a turn.
    assert [s['function']['name'] for s in env.world.schemas()] == [
        'world_sql', 'world_schema', 'world_doc'
    ]
    monkeypatch.setenv('TWINKLE_WORLD_SEARCH_ROOT', str(search_root))
    with_search = HostWorld(env.world.root)
    with_search.load()
    assert [s['function']['name'] for s in with_search.schemas()] == [
        'world_sql', 'world_schema', 'world_doc', 'web_search', 'fetch_page'
    ]


# ------------------------------------------------------------------- the mixin


def test_tools_appends_world_schemas_to_the_environment_own(env):
    names = [t['function']['name'] for t in env.tools()]
    assert names[:3] == ['run_command', 'write_file', 'read_file']
    assert names[3:] == ['world_sql', 'world_schema', 'world_doc']


def test_environment_tool_of_the_same_name_is_not_shadowed(world_root, tmp_path, monkeypatch):
    # Silently shadowing would promise the model one behaviour and hand it
    # another, so the environment's own tool has to win.
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    world = HostWorld()
    world.load()

    class Colliding(LocalEnv):

        def tools(self):
            return [{
                'type': 'function',
                'function': {
                    'name': 'world_sql',
                    'description': 'the environment one',
                    'parameters': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            }]

    class WorldColliding(WorldToolsMixin, Colliding):
        pass

    env = WorldColliding(workspace=str(tmp_path / 'slot'), world=world)
    tools = env.tools()
    names = [t['function']['name'] for t in tools]
    assert names.count('world_sql') == 1
    assert tools[0]['function']['description'] == 'the environment one'
    # the world's other tools still come through
    assert 'world_schema' in names and 'world_doc' in names


def test_step_answers_a_world_tool(env):
    result = env.step('world_sql', {
        'query': 'SELECT service, COUNT(*) FROM incidents GROUP BY service ORDER BY service',
        'params': []
    })
    assert result.done is False
    assert 'auth' in result.observation and 'billing' in result.observation


def test_step_strips_the_server_prefix_a_harness_adds(env):
    # ms-agent namespaces tools as {server}---{tool}, so the tool name is the
    # suffix. A call arriving that way is still ours.
    result = env.step('msagent---world_sql', {'query': 'SELECT COUNT(*) FROM incidents', 'params': []})
    assert '3' in result.observation


def test_step_forwards_a_tool_that_is_not_the_worlds(env):
    result = env.step('run_command', {'command': 'echo from-the-env'})
    assert 'from-the-env' in result.observation


def test_step_does_not_claim_a_name_that_only_looks_like_ours(env):
    # The tool name is the suffix, so 'world_sql---slot3' names a tool called
    # 'slot3' on a server called 'world_sql'. Not ours; forward it.
    result = env.step('world_sql---slot3', {'query': 'SELECT 1', 'params': []})
    assert 'world_sql---slot3' in result.observation or 'unknown' in result.observation.lower()


def test_unconfigured_database_reports_rather_than_raises(env):
    result = env.step('db_query', {'query': 'SELECT 1'})
    assert 'no database is configured' in result.observation
    assert result.done is False


# ------------------------------------------------------------------ batch path


def test_step_batch_pairs_each_observation_with_its_own_call(env):
    calls = [
        ('run_command', {'command': 'echo A'}),
        ('world_sql', {'query': 'SELECT COUNT(*) FROM incidents', 'params': []}),
        ('run_command', {'command': 'echo B'}),
        ('world_doc', {'name': 'runbook.md'}),
    ]
    results = env.step_batch(calls)
    assert len(results) == 4
    assert 'A' in results[0].observation
    assert '3' in results[1].observation
    assert 'B' in results[2].observation
    assert 'Runbook' in results[3].observation


def test_step_batch_without_a_world_falls_through(env):
    env._world = None
    calls = [('run_command', {'command': 'echo only'})]
    results = env.step_batch(calls)
    assert len(results) == 1 and 'only' in results[0].observation


def test_step_batch_refuses_to_guess_when_the_env_returns_the_wrong_count(env, monkeypatch):
    # Pairing an observation with the wrong call teaches the policy something
    # false and nothing downstream would notice, so this has to be loud.
    monkeypatch.setattr(LocalEnv, 'step_batch', lambda self, calls: [])
    with pytest.raises(RuntimeError, match='cannot pair observations'):
        env.step_batch([('run_command', {'command': 'echo A'}),
                        ('world_sql', {'query': 'SELECT 1', 'params': []})])


# -------------------------------------------------------------------- receipts


def test_web_search_records_what_it_actually_served(world_root, search_root, tmp_path, monkeypatch):
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    monkeypatch.setenv('TWINKLE_WORLD_SEARCH_ROOT', str(search_root))
    world = HostWorld()
    world.load()
    env = WorldLocalEnv(workspace=str(tmp_path / 'slot'), world=world)

    env.step('web_search', {'query': 'auth outage', 'num_results': 2})
    assert len(world.receipts) == 1
    receipt = world.receipts[0]
    assert receipt['tool'] == 'web_search'
    assert receipt['query'] == 'auth outage'
    served = {u['url'] for u in receipt['urls']}
    assert 'https://example.com/a' in served
    # The date travels with the url: a check that only asks "was it served"
    # cannot tell a fresh source from a stale one.
    assert {u['url']: u['published'] for u in receipt['urls']}['https://example.com/a'] == '2026-09-01'


def test_fetch_page_records_the_page_it_served(world_root, search_root, tmp_path, monkeypatch):
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    monkeypatch.setenv('TWINKLE_WORLD_SEARCH_ROOT', str(search_root))
    world = HostWorld()
    world.load()
    env = WorldLocalEnv(workspace=str(tmp_path / 'slot'), world=world)

    env.step('fetch_page', {'url': 'https://example.com/b', 'query': 'ledger'})
    assert [u['url'] for r in world.receipts for u in r['urls']] == ['https://example.com/b']


def test_a_failed_fetch_is_not_a_receipt_for_the_page_it_named(world_root, search_root, tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    monkeypatch.setenv('TWINKLE_WORLD_SEARCH_ROOT', str(search_root))
    world = HostWorld()
    world.load()
    env = WorldLocalEnv(workspace=str(tmp_path / 'slot'), world=world)

    result = env.step('fetch_page', {'url': 'https://example.com/does-not-exist'})
    assert result.observation.startswith('fetch_page:')
    # The call is logged, but it carries no url -- which is the part a citation
    # check reads. "I fetched this" needs evidence, and a failed fetch is not it.
    assert len(world.receipts) == 1
    assert world.receipts[0]['tool'] == 'fetch_page'
    assert world.receipts[0]['urls'] == []


def test_sql_and_doc_calls_produce_no_receipt(env):
    # Receipts exist so a citation can be checked. A tool that serves no url has
    # nothing to cite, so recording one would only dilute the log.
    env.step('world_sql', {'query': 'SELECT 1', 'params': []})
    env.step('world_doc', {'name': 'runbook.md'})
    assert env.world.receipts == []


# -------------------------------------------------------------------- lifetime


def test_world_forks_keep_receipts_and_roles_separate(env):
    first = env.world.fork(db_enabled=True)
    second = env.world.fork()
    first.call('fetch_page', {'url': 'https://example.com/missing'})
    assert len(first.receipts) == 1
    assert second.receipts == env.world.receipts == []
    first._db_creds = {'role': 'first'}
    assert second._db_creds is None
    assert first.db_enabled and not second.db_enabled
    assert first.docs is second.docs


def test_owned_world_state_is_cleared_between_episodes(env):
    env._release_world_on_close = True
    released = []
    env.world.release = lambda: released.append(True)
    env.step('fetch_page', {'url': 'https://example.com/missing'})
    env.clear()
    assert released == [True]
    assert env.world.receipts == []


def test_close_does_not_release_a_shared_world(env):
    released = []
    env.world.release = lambda: released.append(True)
    env.close()
    assert released == []


def test_close_releases_the_world_when_this_env_owns_it(world_root, tmp_path, monkeypatch):
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    world = HostWorld()
    world.load()
    released = []
    world.release = lambda: released.append(True)
    env = WorldLocalEnv(workspace=str(tmp_path / 'slot'), world=world, release_world_on_close=True)
    env.close()
    assert released == [True]
