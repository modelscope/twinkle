# Copyright (c) ModelScope Contributors. All rights reserved.
"""The task contract: one declaration, cross-checked against the environment that runs it.

A contract says what a task needs (capabilities, task form, inputs, outputs), what
grades it (verifier kinds), what has to be in the workspace first (fixtures), and
which profile actually served it (provenance). Each piece is checkable alone; the
value is in :func:`validate_task_contract`, which checks them *against each other*
-- a task declaring a verifier its profile cannot run, or an output its profile
cannot produce, is a task that will fail in a way that reads as "hard".

Like ``verifier/rubric_score``, this module has no in-tree caller yet: it is the
declaration format a task builder fills in. Both halves are tested here so the
format is pinned before something depends on it.
"""
import pytest

from twinkle_agentic.envs import (ProfileProvenance, ProfileRegistry, SandboxProfile, TaskContract, TaskIntent,
                                  VerifierSpec, default_profile_registry, resolve_intent, validate_task_contract)
from twinkle_agentic.envs.task_contract import FixtureSpec, FreshnessSpec

SHA = 'a' * 64
REGISTRY = default_profile_registry()


def make_intent(**overrides):
    kwargs = dict(intent_id='i-sql',
                  requested_profile_id='',
                  task_form='database_analysis',
                  capabilities=('database_query', 'data_compute'),
                  input_kinds=('given_text', 'data_fixture'),
                  output_kinds=('answer', 'workspace'),
                  reason='the card is about querying a database')
    kwargs.update(overrides)
    return TaskIntent(**kwargs)


def make_contract(provenance, **overrides):
    kwargs = dict(task_id='t-1',
                  intent_id='i-sql',
                  query='How many orders shipped late?',
                  provenance=provenance,
                  verifiers=(VerifierSpec.from_dict('sql_query', {'sql': 'SELECT 1'}), ),
                  fixtures=(),
                  output_kinds=('answer', ))
    kwargs.update(overrides)
    return TaskContract(**kwargs)


def provenance_for(intent):
    return ProfileProvenance.from_resolution(resolve_intent(intent, REGISTRY))


# ------------------------------------------------------------------- TaskIntent
def test_an_intent_has_to_say_what_it_is_and_why():
    for field, value in (('intent_id', ''), ('task_form', ''), ('reason', '   ')):
        with pytest.raises(ValueError, match='must not be empty'):
            make_intent(**{field: value})


def test_a_repeated_capability_is_refused():
    # A repeated entry reads as two permissions and silently halves the meaning
    # of the list.
    with pytest.raises(ValueError, match='must not contain duplicates'):
        make_intent(capabilities=('database_query', 'database_query'))


def test_unknown_input_and_output_kinds_are_refused():
    with pytest.raises(ValueError, match='unknown input kinds'):
        make_intent(input_kinds=('telepathy', ))
    with pytest.raises(ValueError, match='unknown output kinds'):
        make_intent(output_kinds=('vibes', ))


# ------------------------------------------------------------------ FixtureSpec
def test_a_fixture_cannot_point_outside_the_workspace():
    for path in ('/etc/passwd', '../escape.csv', 'a/../../escape.csv', '.', '..'):
        with pytest.raises(ValueError, match='must stay inside the workspace'):
            FixtureSpec(fixture_id='f', target_path=path, sha256=SHA, kind='data_fixture')


def test_a_fixture_hash_has_to_be_a_real_hash():
    with pytest.raises(ValueError, match='64 lowercase hexadecimal'):
        FixtureSpec(fixture_id='f', target_path='data/shop.db', sha256='not-a-hash', kind='data_fixture')


def test_a_writable_fixture_is_refused():
    # The whole point of a fixture is that the answer depends on data the policy
    # cannot edit. A writable one is not a fixture.
    with pytest.raises(ValueError, match='must be read-only'):
        FixtureSpec(fixture_id='f',
                    target_path='data/shop.db',
                    sha256=SHA,
                    kind='data_fixture',
                    read_only=False)


# ----------------------------------------------------------------- VerifierSpec
def test_a_verifier_config_has_to_be_canonical_json_object():
    with pytest.raises(ValueError, match='must not be empty'):
        VerifierSpec(kind='')
    with pytest.raises(ValueError, match='invalid verifier config JSON'):
        VerifierSpec(kind='python', config_json='{not json')
    with pytest.raises(ValueError, match='must be a JSON object'):
        VerifierSpec(kind='python', config_json='[1, 2]')
    with pytest.raises(ValueError, match="reserved 'kind'"):
        VerifierSpec(kind='python', config_json='{"kind": "shell"}')
    # Same content, non-canonical encoding: spaces after the separators.
    with pytest.raises(ValueError, match='canonical JSON'):
        VerifierSpec(kind='python', config_json='{"a": 1, "b": 2}')


def test_from_dict_and_to_dict_round_trip():
    spec = VerifierSpec.from_dict('answer_number', {'value': 42, 'abs_tol': 0.5})
    assert spec.to_dict() == {'kind': 'answer_number', 'value': 42, 'abs_tol': 0.5}
    assert VerifierSpec.from_dict('file_exists', {}) == VerifierSpec(kind='file_exists')


# ---------------------------------------------------------------- FreshnessSpec
def test_freshness_needs_both_timestamps_or_neither():
    with pytest.raises(ValueError, match='both retrieved_at and expires_at'):
        FreshnessSpec(retrieved_at='2026-01-01T00:00:00+00:00')
    assert FreshnessSpec().required is False


def test_freshness_timestamps_need_a_timezone_and_an_order():
    with pytest.raises(ValueError, match='must include a timezone'):
        FreshnessSpec(retrieved_at='2026-01-01T00:00:00', expires_at='2026-02-01T00:00:00')
    with pytest.raises(ValueError, match='must be later than retrieval'):
        FreshnessSpec(retrieved_at='2026-02-01T00:00:00+00:00', expires_at='2026-01-01T00:00:00+00:00')


def test_receipt_alone_makes_freshness_required():
    assert FreshnessSpec(receipt_required=True).required is True


# ---------------------------------------------------------------- TaskContract
def test_a_contract_pins_its_schema_version():
    with pytest.raises(ValueError, match='schema version 3'):
        make_contract(provenance_for(make_intent()), schema_version=2)


def test_a_contract_needs_an_id_a_query_and_at_least_one_verifier():
    prov = provenance_for(make_intent())
    with pytest.raises(ValueError, match='must not be empty'):
        make_contract(prov, query='   ')
    with pytest.raises(ValueError, match='at least one verifier'):
        make_contract(prov, verifiers=())


def test_duplicate_outputs_and_duplicate_fixture_paths_are_refused():
    prov = provenance_for(make_intent())
    with pytest.raises(ValueError, match='must not contain duplicates'):
        make_contract(prov, output_kinds=('answer', 'answer'))
    fixture = FixtureSpec(fixture_id='f', target_path='data/shop.db', sha256=SHA, kind='data_fixture')
    with pytest.raises(ValueError, match='target paths must be unique'):
        make_contract(prov, fixtures=(fixture, fixture))


def test_the_digest_is_stable_and_moves_with_the_content():
    intent = make_intent()
    prov = provenance_for(intent)
    first = make_contract(prov).digest
    assert first == make_contract(prov).digest
    assert first != make_contract(prov, query='A different question.').digest


# --------------------------------------------------------------- resolve_intent
def test_an_intent_resolves_to_the_least_privilege_profile():
    resolution = resolve_intent(make_intent(), REGISTRY)
    assert resolution.profile.profile_id == 'sql-analytics@1'
    assert 'least-privilege' in resolution.reason


def test_an_intent_cannot_escalate_past_the_profile_it_named():
    # docs-retrieval cannot do database_query, so asking for it by name while
    # declaring that capability is a request for more than the named profile has.
    with pytest.raises(ValueError, match='escalation beyond requested profile'):
        resolve_intent(make_intent(requested_profile_id='docs-retrieval@1'), REGISTRY)


def test_a_profile_that_cannot_take_the_declared_inputs_is_refused():
    intent = make_intent(input_kinds=('given_text', 'data_fixture', 'live_web'))
    with pytest.raises(ValueError, match='does not accept inputs'):
        resolve_intent(intent, REGISTRY)


def test_a_profile_that_cannot_produce_the_declared_outputs_is_refused():
    intent = make_intent(output_kinds=('answer', 'workspace', 'proof'))
    with pytest.raises(ValueError, match='does not produce outputs'):
        resolve_intent(intent, REGISTRY)


def test_a_forbidden_capability_is_refused_before_any_profile_is_considered():
    with pytest.raises(ValueError, match='forbidden capabilities'):
        resolve_intent(make_intent(capabilities=('raw_network', 'data_compute')), REGISTRY)


# ------------------------------------------------------- validate_task_contract
def test_a_consistent_contract_validates_and_returns_its_profile():
    intent = make_intent()
    profile = validate_task_contract(make_contract(provenance_for(intent)), intent, REGISTRY)
    assert profile.profile_id == 'sql-analytics@1'


def test_a_contract_has_to_name_the_intent_it_was_built_from():
    intent = make_intent()
    contract = make_contract(provenance_for(intent), intent_id='some-other-intent')
    with pytest.raises(ValueError, match='refers to a different intent'):
        validate_task_contract(contract, intent, REGISTRY)


def test_provenance_that_does_not_match_the_resolution_is_refused():
    # This is the check that makes a run auditable: the contract has to name the
    # profile that actually served it, digest and all, not the one its author
    # meant.
    intent = make_intent()
    honest = provenance_for(intent)
    tampered = ProfileProvenance(requested_profile_id=honest.requested_profile_id,
                                 resolved_profile_id=honest.resolved_profile_id,
                                 profile_digest='b' * 64,
                                 registry_digest=honest.registry_digest,
                                 router_version=honest.router_version,
                                 decision_reason=honest.decision_reason)
    with pytest.raises(ValueError, match='provenance does not match'):
        validate_task_contract(make_contract(tampered), intent, REGISTRY)


def test_a_verifier_the_profile_cannot_run_is_refused():
    intent = make_intent()
    contract = make_contract(provenance_for(intent),
                             verifiers=(VerifierSpec.from_dict('web_receipt', {}), ))
    with pytest.raises(ValueError, match='cannot run verifiers'):
        validate_task_contract(contract, intent, REGISTRY)


def test_a_fixture_the_profile_cannot_mount_is_refused():
    intent = make_intent()
    fixture = FixtureSpec(fixture_id='f',
                          target_path='proof.lean',
                          sha256=SHA,
                          kind='lean_fixture')
    contract = make_contract(provenance_for(intent), fixtures=(fixture, ))
    with pytest.raises(ValueError, match='cannot mount fixtures'):
        validate_task_contract(contract, intent, REGISTRY)


def test_outputs_cannot_exceed_the_intent_or_the_profile():
    intent = make_intent(output_kinds=('answer', ))
    contract = make_contract(provenance_for(intent), output_kinds=('answer', 'workspace'))
    with pytest.raises(ValueError, match='exceed the declared intent'):
        validate_task_contract(contract, intent, REGISTRY)


def test_a_fixture_kind_the_profile_does_not_take_is_refused_first():
    # knowledge-answer@1 declares only `given_text`, so a data_fixture is turned
    # down by the fixture-kind check before the no-tool guard below ever runs.
    intent = make_intent(intent_id='i-prose',
                         task_form='knowledge_answer',
                         capabilities=('objective_answer', ),
                         input_kinds=('given_text', 'data_fixture'),
                         output_kinds=('answer', ),
                         reason='the card is about a definition')
    with pytest.raises(ValueError, match='does not accept inputs'):
        resolve_intent(intent, REGISTRY)


def test_a_no_tool_task_cannot_mount_a_fixture():
    # Reached only by a no-tool profile that *does* declare a fixture input kind:
    # with the shipped registry the check above fires first. Kept because the
    # invariant is about the session kind, not about which kinds a profile lists
    # today -- a future no-tool profile that declared `data_fixture` would
    # otherwise get a workspace it has no tool to read.
    registry = ProfileRegistry()
    registry.register(
        SandboxProfile(name='no-tool-with-fixture',
                       version=1,
                       session_kind='no_tool',
                       capabilities=('objective_answer', ),
                       task_forms=('knowledge_answer', ),
                       allowed_tools=(),
                       verifier_kinds=('answer_equals', ),
                       input_kinds=('given_text', 'data_fixture'),
                       output_kinds=('answer', )))
    intent = make_intent(intent_id='i-prose',
                         task_form='knowledge_answer',
                         capabilities=('objective_answer', ),
                         input_kinds=('given_text', 'data_fixture'),
                         output_kinds=('answer', ),
                         reason='the card is about a definition')
    resolution = resolve_intent(intent, registry)
    assert resolution.profile.profile_id == 'no-tool-with-fixture@1'
    fixture = FixtureSpec(fixture_id='f', target_path='data.csv', sha256=SHA, kind='data_fixture')
    contract = TaskContract(task_id='t-2',
                            intent_id='i-prose',
                            query='State the definition.',
                            provenance=ProfileProvenance.from_resolution(resolution),
                            verifiers=(VerifierSpec.from_dict('answer_equals', {'value': 'x'}), ),
                            fixtures=(fixture, ))
    with pytest.raises(ValueError, match='no-tool tasks cannot mount fixtures'):
        validate_task_contract(contract, intent, registry)
    # Without the fixture the same contract is fine, which is what pins the
    # refusal to the fixture and not to the profile.
    assert validate_task_contract(make_contract(ProfileProvenance.from_resolution(resolution),
                                                task_id='t-2b',
                                                intent_id='i-prose',
                                                query='State the definition.',
                                                verifiers=(VerifierSpec.from_dict(
                                                    'answer_equals', {'value': 'x'}), )), intent,
                                  registry).profile_id == 'no-tool-with-fixture@1'


def test_web_research_has_to_carry_a_freshness_policy_and_a_receipt_check():
    intent = make_intent(intent_id='i-web',
                         task_form='web_research',
                         capabilities=('web_search', 'web_fetch', 'citation'),
                         input_kinds=('given_text', ),
                         output_kinds=('answer', 'citations'),
                         reason='the card is about verifying a source')
    prov = provenance_for(intent)
    assert prov.resolved_profile_id == 'web-research@1'

    bare = TaskContract(task_id='t-3',
                        intent_id='i-web',
                        query='When did the auth outage start?',
                        provenance=prov,
                        verifiers=(VerifierSpec.from_dict('answer_contains', {'value': '2026'}), ),
                        output_kinds=('answer', 'citations'))
    with pytest.raises(ValueError, match='require a freshness policy'):
        validate_task_contract(bare, intent, REGISTRY)

    fresh = FreshnessSpec(retrieved_at='2026-09-01T00:00:00+00:00', expires_at='2026-10-01T00:00:00+00:00')
    no_receipt = TaskContract(task_id='t-4',
                              intent_id='i-web',
                              query='When did the auth outage start?',
                              provenance=prov,
                              verifiers=(VerifierSpec.from_dict('answer_contains', {'value': '2026'}), ),
                              output_kinds=('answer', 'citations'),
                              freshness=fresh)
    with pytest.raises(ValueError, match='require receipt verification'):
        validate_task_contract(no_receipt, intent, REGISTRY)

    complete = TaskContract(task_id='t-5',
                            intent_id='i-web',
                            query='When did the auth outage start?',
                            provenance=prov,
                            verifiers=(
                                VerifierSpec.from_dict('answer_contains', {'value': '2026'}),
                                VerifierSpec.from_dict('web_receipt', {'url': 'https://example.com/a'}),
                            ),
                            output_kinds=('answer', 'citations'),
                            freshness=FreshnessSpec(retrieved_at='2026-09-01T00:00:00+00:00',
                                                    expires_at='2026-10-01T00:00:00+00:00',
                                                    receipt_required=True))
    assert validate_task_contract(complete, intent, REGISTRY).profile_id == 'web-research@1'


def test_a_freshness_policy_on_a_non_web_task_is_refused():
    # A stale answer is only meaningful against something retrieved, so a
    # freshness policy anywhere else is a mistake rather than a stricter task.
    intent = make_intent()
    fresh = FreshnessSpec(retrieved_at='2026-09-01T00:00:00+00:00', expires_at='2026-10-01T00:00:00+00:00')
    contract = make_contract(provenance_for(intent), freshness=fresh)
    with pytest.raises(ValueError, match='only valid for web research'):
        validate_task_contract(contract, intent, REGISTRY)
