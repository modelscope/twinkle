# Copyright (c) ModelScope Contributors. All rights reserved.
"""Least-privilege profiles, and the mixin that enforces one on an Env.

A profile is what stops every task from getting the same unrestricted tool set:
a topic about querying a database and a topic about office ergonomics should not
arrive in the same environment, and a task whose fixture has to exist before the
episode starts needs the environment seeded rather than merely emptied.
"""
import json

import pytest

from twinkle_agentic.envs import (LocalEnv, ProfiledEnvMixin, ProfileRegistry, SandboxProfile,
                                  default_profile_registry, env_names_for)


def make_profile(name='test-profile',
                 allowed_tools=('read_file', ),
                 capabilities=('data_compute', ),
                 session_kind='agentenv',
                 setup_manifest=(),
                 available=True,
                 enabled_for_new_tasks=True,
                 **kwargs):
    return SandboxProfile(name=name,
                          version=1,
                          session_kind=session_kind,
                          capabilities=capabilities,
                          task_forms=('database_analysis', ),
                          allowed_tools=allowed_tools,
                          verifier_kinds=('file_exists', ),
                          input_kinds=('given_text', ),
                          output_kinds=('answer', ),
                          setup_manifest=setup_manifest,
                          available=available,
                          enabled_for_new_tasks=enabled_for_new_tasks,
                          **kwargs)


# ------------------------------------------------------------- the profile itself
def test_profile_id_is_name_at_version():
    assert make_profile(name='sql-analytics').profile_id == 'sql-analytics@1'


def test_the_default_registry_carries_the_builtin_profiles():
    registry = default_profile_registry()
    ids = {p.profile_id for p in registry.profiles()}
    assert 'sql-analytics@1' in ids and 'web-research@1' in ids and 'lean-proof@1' in ids
    assert len(ids) >= 10


def test_a_profile_cannot_grant_a_forbidden_capability():
    # raw_network in particular: the whole point of serving the world from the
    # host is that the sandbox has no egress, and a profile that granted it
    # would quietly undo that.
    with pytest.raises(ValueError, match='forbidden capabilities'):
        make_profile(capabilities=('raw_network', ))


def test_a_no_tool_profile_cannot_also_allow_tools():
    with pytest.raises(ValueError, match='no-tool profiles cannot advertise tools'):
        make_profile(session_kind='no_tool', allowed_tools=('read_file', ))


def test_a_profile_name_has_to_be_a_valid_identifier():
    with pytest.raises(ValueError, match='invalid profile name'):
        make_profile(name='Not A Name')


def test_version_has_to_be_a_positive_integer():
    with pytest.raises(ValueError, match='positive integer'):
        SandboxProfile(name='x', version=0, session_kind='agentenv', capabilities=('data_compute', ),
                       task_forms=('t', ), allowed_tools=('read_file', ), verifier_kinds=('file_exists', ),
                       input_kinds=('given_text', ), output_kinds=('answer', ))


def test_a_duplicated_entry_in_any_list_field_is_rejected():
    # A repeated tool name reads as two permissions and silently halves the
    # meaning of the list, so it is refused at build time.
    with pytest.raises(ValueError):
        make_profile(allowed_tools=('read_file', 'read_file'))


def test_the_manifest_is_a_digest_so_a_changed_profile_is_detectable():
    a = make_profile().digest
    b = make_profile(allowed_tools=('read_file', 'write_file')).digest
    assert a != b
    assert make_profile().digest == a


# ------------------------------------------------------------------- the registry
def test_registering_the_same_id_twice_is_refused():
    registry = ProfileRegistry()
    registry.register(make_profile())
    with pytest.raises(ValueError, match='already registered'):
        registry.register(make_profile())


def test_get_raises_for_an_unknown_profile():
    with pytest.raises(ValueError, match='not registered'):
        ProfileRegistry().get('nope@1')


def test_registry_digest_covers_every_profile():
    one = ProfileRegistry()
    one.register(make_profile(name='a'))
    two = ProfileRegistry()
    two.register(make_profile(name='a'))
    two.register(make_profile(name='b'))
    assert one.digest != two.digest


# --------------------------------------------------------------------- aliasing
def test_env_names_translates_the_agent_runtime_vocabulary():
    # A profile names tools the way the agent runtime that served them does;
    # an Env declares fewer, coarser names. The table is the only place that
    # knows the mapping.
    profile = make_profile(allowed_tools=('shell_executor', 'read_file', 'glob'))
    assert env_names_for(profile) == frozenset({'run_command', 'read_file'})


def test_env_names_keeps_a_name_that_needs_no_translation():
    profile = make_profile(allowed_tools=('run_command', 'write_file'))
    assert env_names_for(profile) == frozenset({'run_command', 'write_file'})


def test_env_names_passes_through_a_name_it_does_not_know():
    # Unknown means "an environment that declares it", not "drop it": silently
    # discarding would widen nothing but would hide a typo from whoever wrote
    # the profile.
    profile = make_profile(allowed_tools=('some_future_tool', ))
    assert env_names_for(profile) == frozenset({'some_future_tool'})


# -------------------------------------------------------------------- the mixin
class ProfiledLocalEnv(ProfiledEnvMixin, LocalEnv):
    """The mixin goes first: it has to precede the env in the MRO."""


@pytest.fixture
def slot(tmp_path):
    return ProfiledLocalEnv(workspace=str(tmp_path / 'slot'))


def test_with_no_profile_the_environment_is_unrestricted(slot):
    # Mixing this in must not change an environment whose caller has no
    # profiles, or adopting it costs every existing caller something.
    assert slot.profile is None
    assert slot.profile_id == ''
    assert {t['function']['name'] for t in slot.tools()} == {'run_command', 'write_file', 'read_file'}
    assert 'fine' in slot.step('run_command', {'command': 'echo fine'}).observation


def test_a_profile_narrows_what_is_advertised(tmp_path):
    profile = make_profile(allowed_tools=('read_file', ))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profile=profile)
    assert [t['function']['name'] for t in slot.tools()] == ['read_file']


def test_a_tool_outside_the_profile_is_refused_not_forwarded(tmp_path):
    # Advertising is not enforcement: a policy can name a tool it was never
    # shown, and the refusal has to come from here rather than from the
    # environment reporting an unknown tool.
    profile = make_profile(allowed_tools=('read_file', ))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profile=profile)
    result = slot.step('run_command', {'command': 'echo should-not-run'})
    assert 'not available in this environment' in result.observation
    assert 'should-not-run' not in result.observation


def test_a_permitted_tool_still_works(tmp_path):
    profile = make_profile(allowed_tools=('run_command', 'read_file', 'write_file'))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profile=profile)
    assert 'ok' in slot.step('run_command', {'command': 'echo ok'}).observation


def test_the_refusal_names_what_is_allowed(tmp_path):
    # A refusal that only says no teaches the policy nothing; naming the allowed
    # set gives it somewhere to go on the next turn.
    profile = make_profile(allowed_tools=('read_file', ))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profile=profile)
    observation = slot.step('write_file', {'path': 'x', 'content': 'y'}).observation
    assert 'read_file' in observation


def test_step_batch_answers_refusals_locally_and_forwards_the_rest(tmp_path):
    profile = make_profile(allowed_tools=('run_command', ))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profile=profile)
    calls = [
        ('run_command', {'command': 'echo A'}),
        ('write_file', {'path': 'nope', 'content': 'x'}),
        ('run_command', {'command': 'echo B'}),
    ]
    results = slot.step_batch(calls)
    assert len(results) == 3
    assert 'A' in results[0].observation
    assert 'not available in this environment' in results[1].observation
    assert 'B' in results[2].observation


def test_step_batch_with_no_profile_falls_straight_through(tmp_path):
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'))
    results = slot.step_batch([('run_command', {'command': 'echo only'})])
    assert len(results) == 1 and 'only' in results[0].observation


# ------------------------------------------------------------------- switching
def test_switch_profile_seeds_the_workspace(tmp_path):
    registry = ProfileRegistry()
    registry.register(make_profile(name='seeded',
                                   allowed_tools=('run_command', ),
                                   setup_manifest=('seed-sql-analytics', )))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profiles=registry)
    profile = slot.switch_profile('seeded@1')
    assert slot.profile_id == 'seeded@1'
    assert profile.name == 'seeded'
    # The seed built the fixture the task will be graded on. Asserted by name
    # and by content, not by "the directory is not empty": an empty file would
    # satisfy that and every task built on it would then be unsolvable.
    import sqlite3
    db = tmp_path / 'slot' / 'shop.db'
    assert db.is_file()
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {'customers', 'products', 'orders'} <= tables


def test_switch_profile_records_every_manifest_step(tmp_path):
    registry = ProfileRegistry()
    registry.register(make_profile(name='two-steps',
                                   allowed_tools=('run_command', ),
                                   setup_manifest=('clear_workspace', 'python-numpy-sympy-canary')))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profiles=registry)
    slot.switch_profile('two-steps@1')
    steps = [name for name, _, _ in slot.manifest_results]
    assert steps == ['clear_workspace', 'python-numpy-sympy-canary']
    assert all(code == 0 for _, code, _ in slot.manifest_results)


def test_an_unknown_manifest_step_fails_loudly(tmp_path):
    # A typo in a manifest would otherwise produce an environment missing its
    # fixture, and every task built in it would read as unsolvable.
    registry = ProfileRegistry()
    registry.register(make_profile(name='typo',
                                   allowed_tools=('run_command', ),
                                   setup_manifest=('seed-does-not-exist', )))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profiles=registry)
    with pytest.raises(RuntimeError, match='setup step'):
        slot.switch_profile('typo@1')


def test_switching_to_an_unavailable_profile_raises(tmp_path):
    registry = ProfileRegistry()
    registry.register(make_profile(name='off',
                                   allowed_tools=('run_command', ),
                                   available=False,
                                   **{'unavailable_reason': 'no service behind it'}))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profiles=registry)
    with pytest.raises(ValueError, match='not available'):
        slot.switch_profile('off@1')


def test_switching_to_a_retired_profile_raises(tmp_path):
    registry = ProfileRegistry()
    registry.register(make_profile(name='old',
                                   allowed_tools=('run_command', ),
                                   enabled_for_new_tasks=False))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profiles=registry)
    with pytest.raises(ValueError, match='retired'):
        slot.switch_profile('old@1')


def test_switching_narrows_the_advertised_tools(tmp_path):
    registry = ProfileRegistry()
    registry.register(make_profile(name='readers', allowed_tools=('read_file', )))
    slot = ProfiledLocalEnv(workspace=str(tmp_path / 'slot'), profiles=registry)
    assert len(slot.tools()) == 3
    slot.switch_profile('readers@1')
    assert [t['function']['name'] for t in slot.tools()] == ['read_file']


# ------------------------------------------------- combined with the world mixin
def test_the_profile_also_governs_the_world_tools(tmp_path, monkeypatch):
    # MRO order matters: ProfiledEnvMixin ahead of WorldToolsMixin, so the world
    # contributes its tools and the profile then decides which of them survive.
    from twinkle_agentic.envs import HostWorld, WorldToolsMixin

    root = tmp_path / 'world'
    (root / 'docs').mkdir(parents=True)
    import hashlib
    import sqlite3
    conn = sqlite3.connect(root / 'ops.db')
    conn.execute('CREATE TABLE t (a INTEGER)')
    conn.commit()
    conn.close()
    (root / 'manifest.json').write_text(
        json.dumps({'files': {'ops.db': hashlib.sha256((root / 'ops.db').read_bytes()).hexdigest()}}),
        encoding='utf-8')
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(root))
    world = HostWorld()
    world.load()

    registry = ProfileRegistry()
    # world_sql rather than db_query: the world only advertises the database
    # tools when a database is actually provisioned for the run.
    registry.register(make_profile(name='sql-and-files', allowed_tools=('world_sql', 'read_file')))

    class Slot(ProfiledEnvMixin, WorldToolsMixin, LocalEnv):
        pass

    slot = Slot(workspace=str(tmp_path / 'slot'), world=world, profiles=registry)
    slot.switch_profile('sql-and-files@1')
    names = {t['function']['name'] for t in slot.tools()}
    assert names == {'world_sql', 'read_file'}
    # A world tool the profile does not permit is refused, not merely hidden:
    # the policy can name a tool it was never shown.
    refused = slot.step('world_doc', {'name': 'anything.md'}).observation
    assert 'not available in this environment' in refused
    # and the permitted one still reaches the host world
    assert 'no such table' in slot.step('world_sql', {'query': 'SELECT * FROM nope',
                                                      'params': []}).observation.lower()         or slot.step('world_sql', {'query': 'SELECT COUNT(*) FROM t', 'params': []}).observation


def test_world_tools_survive_when_the_profile_permits_them(tmp_path, monkeypatch):
    import hashlib
    import sqlite3
    from twinkle_agentic.envs import HostWorld, WorldToolsMixin

    root = tmp_path / 'world'
    (root / 'docs').mkdir(parents=True)
    conn = sqlite3.connect(root / 'ops.db')
    conn.execute('CREATE TABLE t (a INTEGER)')
    conn.commit()
    conn.close()
    (root / 'manifest.json').write_text(
        json.dumps({'files': {'ops.db': hashlib.sha256((root / 'ops.db').read_bytes()).hexdigest()}}),
        encoding='utf-8')
    monkeypatch.setenv('TWINKLE_WORLD_ROOT', str(root))
    world = HostWorld()
    world.load()

    registry = ProfileRegistry()
    registry.register(make_profile(name='sql-world', allowed_tools=('world_sql', 'world_schema')))

    class Slot(ProfiledEnvMixin, WorldToolsMixin, LocalEnv):
        pass

    slot = Slot(workspace=str(tmp_path / 'slot'), world=world, profiles=registry)
    slot.switch_profile('sql-world@1')
    assert {t['function']['name'] for t in slot.tools()} == {'world_sql', 'world_schema'}
    result = slot.step('world_sql', {'query': 'SELECT COUNT(*) FROM t', 'params': []})
    assert '0' in result.observation


def test_every_builtin_profile_names_setup_steps_something_can_resolve():
    # A manifest step that does not resolve is a typo in the registry, and it is
    # invisible until the first switch_profile: clear() raises "unknown setup
    # step", which reads as a broken slot rather than as a broken profile. The
    # retired legacy profile carried one for exactly that reason.
    from twinkle_agentic.envs.seeds import CLEAR_WORKSPACE, LIBRARY_CANARIES, SEED_PROGRAMS

    known = {'clear_workspace'} | set(SEED_PROGRAMS) | set(LIBRARY_CANARIES)
    for profile in default_profile_registry().profiles():
        for step in profile.setup_manifest:
            assert step in known, f'{profile.profile_id} names unknown setup step {step!r}'
    assert CLEAR_WORKSPACE.strip()
