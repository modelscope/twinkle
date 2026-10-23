# Copyright (c) ModelScope Contributors. All rights reserved.
"""How the RSI cookbook builds one environment per concurrent job.

Loaded from ``cookbook/rsi/agentic/slot.py`` the way
``tests/twinkle_client/test_client_orchestrated_grpo.py`` loads its cookbook
script: the cookbook is not an installed package, so it is read by path.

The property that matters most is the boring one -- a run that configures nothing
has to get exactly the environments it got before this module existed. Everything
else is what the two optional layers add on top, and both are verified against
``LocalEnv``, which needs no sandbox host.
"""
import ast
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from twinkle_agentic.envs import AgentEnv, LocalEnv, ProfiledEnvMixin, WorldToolsMixin

SLOT_PATH = Path(__file__).parents[2] / 'cookbook' / 'rsi' / 'agentic' / 'slot.py'


def _load_slot_module():
    spec = importlib.util.spec_from_file_location('cookbook_rsi_slot', SLOT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def slot():
    return _load_slot_module()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def world_root(tmp_path):
    """One table, one document, and a manifest that matches both."""
    root = tmp_path / 'world'
    (root / 'docs').mkdir(parents=True)
    conn = sqlite3.connect(root / 'ops.db')
    conn.execute('CREATE TABLE incidents (id INTEGER PRIMARY KEY, service TEXT, severity TEXT)')
    conn.executemany('INSERT INTO incidents (service, severity) VALUES (?, ?)',
                     [('auth', 'p1'), ('auth', 'p2'), ('billing', 'p1')])
    conn.commit()
    conn.close()
    (root / 'docs' / 'runbook.md').write_text('# Runbook\npage one\n', encoding='utf-8')
    manifest = json.dumps({'files': {'ops.db': _sha256(root / 'ops.db')}})
    (root / 'manifest.json').write_text(manifest, encoding='utf-8')
    return root


@pytest.fixture
def search_root(tmp_path):
    """A one-document frozen corpus, with a manifest that matches."""
    root = tmp_path / 'search'
    root.mkdir()
    documents = [{
        'url': 'https://example.com/a',
        'title': 'Auth outage postmortem',
        'published': '2026-09-01',
        'body': 'The auth service returned 500 for eleven minutes.'
    }]
    (root / 'corpus.json').write_text(json.dumps({'documents': documents}), encoding='utf-8')
    manifest = json.dumps({'files': {'corpus.json': _sha256(root / 'corpus.json')}})
    (root / 'manifest.json').write_text(manifest, encoding='utf-8')
    return root


@pytest.fixture
def clean_env(monkeypatch):
    """No world, no profiles, no affinity table: the unconfigured run."""
    for name in ('TWINKLE_WORLD_ROOT', 'TWINKLE_WORLD_SEARCH_ROOT', 'TWINKLE_RSI_PROFILES',
                 'TWINKLE_TOPIC_AFFINITIES', 'TWINKLE_WORLD_DB_PORT'):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def tool_names(env):
    names = []
    for tool in env.tools():
        names.append(tool['function']['name'] if isinstance(tool, dict) else getattr(tool, 'name', str(tool)))
    return sorted(set(names))


def close_all(envs):
    for env in envs:
        env.close()


# --------------------------------------------------------------- configuration
def test_an_unconfigured_run_reads_no_world_and_no_profiles(slot, clean_env):
    assert slot.configured_world() is None
    assert slot.configured_profiles() == []


def test_configured_world_loads_what_it_was_pointed_at(slot, clean_env, world_root):
    clean_env.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    world = slot.configured_world()
    assert world is not None and world.available
    assert {s['function']['name'] for s in world.schemas()} == {'world_sql', 'world_schema', 'world_doc'}
    world.release()


def test_an_explicit_profile_list_wins(slot, clean_env, tmp_path):
    # Naming profiles is how a run pins what it measured, so the affinity table
    # must not override it -- not even to drop a profile the run insisted on.
    clean_env.setenv('TWINKLE_RSI_PROFILES', ' sql-analytics@1 , docs-retrieval@1 ')
    clean_env.setenv('TWINKLE_TOPIC_AFFINITIES', str(tmp_path / 'affinities.jsonl'))
    assert slot.configured_profiles() == ['sql-analytics@1', 'docs-retrieval@1']


def test_the_affinity_table_supplies_the_rotation_when_no_list_is_given(slot, clean_env, tmp_path):
    table = tmp_path / 'affinities.jsonl'
    table.write_text(json.dumps({'card_id': 'c0', 'eligible_profiles': ['sql-analytics@1']}) + '\n', encoding='utf-8')
    clean_env.setenv('TWINKLE_TOPIC_AFFINITIES', str(table))
    profiles = slot.configured_profiles()
    assert 'sql-analytics@1' in profiles and 'world-forensics@1' in profiles
    assert profiles == list(dict.fromkeys(profiles))


def test_an_unreachable_service_leaves_the_rotation(slot, clean_env, tmp_path):
    # Port 1 is reserved and nothing binds it, so db-analyst drops out and the
    # sandbox-only profile that answers the same task form takes its cards.
    table = tmp_path / 'affinities.jsonl'
    table.write_text(json.dumps({'card_id': 'c0', 'eligible_profiles': ['db-analyst@1']}) + '\n', encoding='utf-8')
    clean_env.setenv('TWINKLE_TOPIC_AFFINITIES', str(table))
    clean_env.setenv('TWINKLE_WORLD_DB_PORT', '1')
    profiles = slot.configured_profiles()
    assert 'db-analyst@1' not in profiles
    assert 'sql-analytics@1' in profiles


# -------------------------------------------------------------------- the slots
def test_an_unconfigured_pool_is_the_plain_env_classes(slot, clean_env, tmp_path):
    envs = slot.build_slot_envs(num_envs=2, workspace_root=str(tmp_path / 'slots'))
    try:
        assert [type(env) for env in envs] == [LocalEnv, LocalEnv]
        assert all(tool_names(env) == ['read_file', 'run_command', 'write_file'] for env in envs)
    finally:
        close_all(envs)


def test_zero_slots_is_an_empty_pool(slot, clean_env, tmp_path):
    assert slot.build_slot_envs(num_envs=0, workspace_root=str(tmp_path / 'slots')) == []


def test_slots_rotate_over_the_profiles_rather_than_sharing_one(slot, clean_env, tmp_path):
    # A pool whose slots all hold the same fixture can only produce one kind of
    # task, and the policy learns that the fixture is part of the world.
    envs = slot.build_slot_envs(num_envs=3,
                                workspace_root=str(tmp_path / 'slots'),
                                profile_ids=['sql-analytics@1', 'docs-retrieval@1'])
    try:
        assert [env.profile_id for env in envs] == ['sql-analytics@1', 'docs-retrieval@1', 'sql-analytics@1']
    finally:
        close_all(envs)


def test_the_profile_mixin_precedes_the_world_mixin(slot, clean_env, tmp_path, world_root):
    # Reversed, the world would contribute its tools after the profile filtered
    # them, and no profile could restrict the world at all.
    clean_env.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    world = slot.configured_world()
    envs = slot.build_slot_envs(num_envs=1,
                                workspace_root=str(tmp_path / 'slots'),
                                world=world,
                                profile_ids=['world-forensics@1'])
    try:
        mro = type(envs[0]).__mro__
        assert mro.index(ProfiledEnvMixin) < mro.index(WorldToolsMixin) < mro.index(LocalEnv)
    finally:
        close_all(envs)


def test_a_profile_decides_which_world_tools_a_slot_gets(slot, clean_env, tmp_path, world_root, search_root):
    clean_env.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    clean_env.setenv('TWINKLE_WORLD_SEARCH_ROOT', str(search_root))
    world = slot.configured_world()
    envs = slot.build_slot_envs(num_envs=3,
                                workspace_root=str(tmp_path / 'slots'),
                                world=world,
                                profile_ids=['world-forensics@1', 'web-research@1', 'sql-analytics@1'])
    try:
        assert {'world_sql', 'world_schema', 'world_doc'} <= set(tool_names(envs[0]))
        assert {'web_search', 'fetch_page'} <= set(tool_names(envs[1]))
        assert 'world_sql' not in tool_names(envs[1])
        # sql-analytics reads a database seeded into its own workspace, so the
        # shared one is not offered to it at all.
        assert not set(tool_names(envs[2])) & {'world_sql', 'web_search'}

        queried = envs[0].step('world_sql', {'query': 'SELECT service FROM incidents'})
        assert 'auth' in queried.observation
        refused = envs[2].step('world_sql', {'query': 'SELECT 1'})
        assert 'not available in this environment' in refused.observation
    finally:
        close_all(envs)


def test_a_seeded_profile_builds_its_fixture_before_the_first_job(slot, clean_env, tmp_path):
    envs = slot.build_slot_envs(num_envs=1,
                                workspace_root=str(tmp_path / 'slots'),
                                profile_ids=['sql-analytics@1'])
    try:
        assert 'seed-sql-analytics' in [name for name, _, _ in envs[0].manifest_results]
        assert all(code == 0 for _, code, _ in envs[0].manifest_results)
        assert 'shop.db' in envs[0].step('run_command', {'command': 'ls'}).observation
    finally:
        close_all(envs)


def test_a_bad_profile_id_fails_at_build_time_not_on_the_first_episode(slot, clean_env, tmp_path):
    # Raised while the pool is built, so a run with a typo in its profile list
    # never starts rather than discovering it one episode in.
    with pytest.raises(ValueError, match='not registered'):
        slot.build_slot_envs(num_envs=1,
                             workspace_root=str(tmp_path / 'slots'),
                             profile_ids=['no-such-profile@1'])


def test_a_sandbox_template_selects_agentenv_and_carries_the_guard(slot, monkeypatch, tmp_path):
    # Stubbed rather than built: an AgentEnv needs a sandbox host, and what is
    # under test here is which class the pool picks and what it is handed.
    built = []

    class StubAgentEnv(AgentEnv):

        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(slot, 'AgentEnv', StubAgentEnv)

    def guard(tool_name, arguments):
        return None

    envs = slot.build_slot_envs(num_envs=2,
                                sandbox_template='tpl',
                                sandbox_api_url='http://127.0.0.1:1',
                                sandbox_timeout=7,
                                pre_tool_call=guard,
                                workspace_root=str(tmp_path / 'slots'))
    assert len(envs) == 2 and len(built) == 2
    assert all(kwargs['template'] == 'tpl' for kwargs in built)
    assert all(kwargs['api_url'] == 'http://127.0.0.1:1' for kwargs in built)
    assert all(kwargs['sandbox_timeout'] == 7 for kwargs in built)
    assert all(kwargs['pre_tool_call'] is guard for kwargs in built)
    assert [kwargs['metadata']['slot'] for kwargs in built] == ['0', '1']


def test_slots_release_their_own_world_state(slot, clean_env, tmp_path, world_root):
    clean_env.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    world = slot.configured_world()
    envs = slot.build_slot_envs(num_envs=3, workspace_root=str(tmp_path / 'slots'), world=world)
    try:
        assert len({id(env.world) for env in envs}) == 3
        assert all(env.world is not world for env in envs)
        assert all(env._release_world_on_close for env in envs)
    finally:
        close_all(envs)


def test_receipts_do_not_leak_between_slots_or_episodes(slot, clean_env, tmp_path, world_root, search_root):
    clean_env.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    clean_env.setenv('TWINKLE_WORLD_SEARCH_ROOT', str(search_root))
    envs = slot.build_slot_envs(num_envs=2, workspace_root=str(tmp_path / 'slots'), world=slot.configured_world())
    try:
        envs[0].step('fetch_page', {'url': 'https://example.com/a'})
        assert envs[0].world.receipts[0]['urls']
        assert envs[1].world.receipts == []
        envs[0].clear()
        assert envs[0].world.receipts == []
    finally:
        close_all(envs)


def test_database_profile_enables_database_tools_only_for_its_slot(slot, clean_env, tmp_path, world_root):
    clean_env.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    envs = slot.build_slot_envs(num_envs=2, workspace_root=str(tmp_path / 'slots'),
                                world=slot.configured_world(), profile_ids=['db-analyst@1', 'world-forensics@1'])
    try:
        assert {'db_query', 'db_schema'} <= set(tool_names(envs[0]))
        assert not {'db_query', 'db_schema'} & set(tool_names(envs[1]))
    finally:
        close_all(envs)


def test_configuration_rejects_a_bad_profile_before_slots_are_built(slot, clean_env):
    clean_env.setenv('TWINKLE_RSI_PROFILES', 'no-such-profile@1')
    with pytest.raises(ValueError, match='not registered'):
        slot.configured_profiles()


@pytest.mark.parametrize('bad_artifact', ['profile', 'knowledge_bank'])
def test_bad_run_artifacts_fail_before_distributed_initialization(slot, clean_env, tmp_path, bad_artifact):
    # Load only main(), avoiding the executable CLI and its distributed imports.
    from twinkle_agentic.challenger import KnowledgeStore, TaskBank

    script = SLOT_PATH.parents[1] / 'rsi_grpo.py'
    main = next(node for node in ast.parse(script.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    namespace = dict(KnowledgeStore=KnowledgeStore, TaskBank=TaskBank,
                     KNOWLEDGE_CARDS='', KNOWLEDGE_SEED=0, TASK_BANK_PATH='', SAVE_DIR='',
                     configured_world=slot.configured_world, configured_profiles=slot.configured_profiles)
    if bad_artifact == 'profile':
        clean_env.setenv('TWINKLE_RSI_PROFILES', 'no-such-profile@1')
    else:
        bank = tmp_path / 'cards.jsonl'
        bank.write_text('{}\n')
        namespace['KNOWLEDGE_CARDS'] = str(bank)

    def allocate(*args, **kwargs):
        pytest.fail('distributed initialization reached before artifact validation')

    namespace['DeviceGroup'] = allocate
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(script), 'exec'), namespace)
    with pytest.raises(ValueError):
        namespace['main']()


def test_a_run_without_a_search_corpus_still_gets_its_world(slot, clean_env, tmp_path, world_root):
    # Optional by design: a run that only reads the database should not have to
    # build a search index it will never query, and a profile that asked for the
    # search tools simply is not offered them.
    clean_env.setenv('TWINKLE_WORLD_ROOT', str(world_root))
    world = slot.configured_world()
    envs = slot.build_slot_envs(num_envs=1,
                                workspace_root=str(tmp_path / 'slots'),
                                world=world,
                                profile_ids=['web-research@1'])
    try:
        assert 'web_search' not in tool_names(envs[0])
    finally:
        close_all(envs)
