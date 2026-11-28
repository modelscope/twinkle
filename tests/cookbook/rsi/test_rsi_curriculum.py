# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the migrated curriculum through real environment and scoring boundaries."""
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.rsi.agentic import prompts as P
from cookbook.rsi.agentic.challenger import RsiChallenger, novelty_factor, overlong_penalty
from cookbook.rsi.agentic.classify_sandbox_affinity import generate_overlay
from cookbook.rsi.agentic.knowledge import KnowledgeCard, KnowledgeCardSeeder, KnowledgeRoute, TaskSeedSeeder
from cookbook.rsi.agentic.slot import build_slot_envs
from cookbook.rsi.agentic.task_bank import TaskBank
from cookbook.rsi.agentic.topic_profile import TopicProfiles
from cookbook.rsi.agentic.verification import declared_derived_paths, verify_check
from cookbook.rsi.agentic.world import HostWorld
from twinkle.data_format import attach_user_data, user_data_get
from twinkle_agentic.challenger.agentic import _ProposalResult, _Unit
from twinkle_agentic.envs import LocalEnv

CHECK = """DERIVED_OUTPUTS = ['answer.json']
import json
import sqlite3
with sqlite3.connect('shop.db') as db:
    expected = db.execute('SELECT COUNT(*) FROM customers').fetchone()[0]
with open('answer.json') as handle:
    assert json.load(handle)['count'] == expected
"""
BUILD = """import json, sqlite3
with sqlite3.connect('shop.db') as db:
    count = db.execute('SELECT COUNT(*) FROM customers').fetchone()[0]
with open('answer.json', 'w') as handle:
    json.dump({'count': count}, handle)
"""


@pytest.fixture(autouse=True)
def generation_is_supplied_by_the_test(monkeypatch):
    # Keep the real challenger scheduler and environment interactions. Model
    # generation is replaced below, so no sampler/template is constructed here.
    monkeypatch.setattr('twinkle_agentic.challenger.agentic.MultiTurnRollout',
                        lambda *args, **kwargs: SimpleNamespace(close=lambda: None))


def test_api_authored_turns_are_context_instead_of_training_targets(tmp_path, monkeypatch):
    from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
    monkeypatch.setattr('twinkle_agentic.challenger.agentic.MultiTurnRollout', MultiTurnRollout)
    challenger = RsiChallenger(
        None,
        envs=[LocalEnv(workspace=str(tmp_path))],
        template=SimpleNamespace(truncation_strategy='raise'),
        api=lambda *args, **kwargs: None)
    try:
        assert challenger._rollout.api.appended_as == 'context'
    finally:
        challenger.close()


def test_generated_verifier_rejects_empty_and_mutated_states_and_restores_author(tmp_path):
    env = LocalEnv(workspace=str(tmp_path))
    output = tmp_path / 'answer.json'
    output.write_text('{"count": 42}')
    check = "DERIVED_OUTPUTS = ['answer.json']\nimport json\nassert json.load(open('answer.json'))['count'] == 42"
    before = output.read_bytes()
    result, error = verify_check(env, check)
    assert not error
    assert result['author_state_passed'] and result['empty_workspace_failed']
    assert result['mutation_sensitive_path'] == 'answer.json'
    assert output.read_bytes() == before
    # Existence alone survives a content mutation and must be rejected.
    result, error = verify_check(env,
                                 "DERIVED_OUTPUTS = ['answer.json']\nimport os\nassert os.path.isfile('answer.json')")
    assert error and not result['passed']
    assert output.read_bytes() == before


@pytest.mark.parametrize('declaration', ["['../escape']", "['/tmp/result']", "['a', 'a']", '[]', "['.']"])
def test_derived_outputs_cannot_escape_or_be_ambiguous(declaration):
    with pytest.raises(ValueError):
        declared_derived_paths('DERIVED_OUTPUTS = ' + declaration)


def test_switching_database_profile_updates_backing_world_tools(tmp_path):
    envs = build_slot_envs(num_envs=1, workspace_root=str(tmp_path), world=HostWorld(), profile_ids=['sql-analytics@1'])
    env = envs[0]
    try:
        env.switch_profile('db-analyst@1')
        assert env.world.db_enabled
        assert {'db_query', 'db_schema'} <= {tool['function']['name'] for tool in env.tools()}
        env.switch_profile('sql-analytics@1')
        assert not env.world.db_enabled
        assert 'db_query' not in {tool['function']['name'] for tool in env.tools()}
        assert Path(env.workspace, 'shop.db').is_file()
    finally:
        env.close()


def test_matched_knowledge_control_routes_keep_the_same_profile_without_disclosing_card(tmp_path, monkeypatch):
    monkeypatch.delenv('TWINKLE_WORLD_DB_MODULE', raising=False)
    card = KnowledgeCard('a' * 24, 0, 'databases > aggregation', 'SECRET_CARD_MATERIAL', 20, 'model', 'b' * 64,
                         'c' * 64, 'd' * 64, {})
    store = SimpleNamespace(
        blocks_per_epoch=1,
        route=lambda gid, epoch: KnowledgeRoute('base_open' if gid % 2 == 0 else 'knowledge', card, None
                                                if gid % 2 == 0 else card))
    seeder = KnowledgeCardSeeder(store)
    overlay = tmp_path / 'affinities.jsonl'
    overlay.write_text(json.dumps({'card_id': card.card_id, 'eligible_profiles': ['sql-analytics@1']}))
    envs = build_slot_envs(num_envs=1, workspace_root=str(tmp_path / 'slots'), profile_ids=['docs-retrieval@1'])
    challenger = RsiChallenger(
        None,
        envs=envs,
        seed_fn=seeder,
        knowledge_seeder=seeder,
        profile_ids=['docs-retrieval@1', 'sql-analytics@1'],
        topic_profiles=TopicProfiles(str(overlay)),
        num_solver_rollouts=2,
        pass_band=(1, 1))
    try:
        control = challenger._build_challenge_prompt()
        knowledge = challenger._build_challenge_prompt()
        assert 'SECRET_CARD_MATERIAL' not in control['messages'][-1]['content']
        assert 'SECRET_CARD_MATERIAL' in knowledge['messages'][-1]['content']
        assert user_data_get(control['user_data'], 'profile_id') == 'sql-analytics@1'
        assert user_data_get(knowledge['user_data'], 'profile_id') == 'sql-analytics@1'
        challenger._prepare_profile(control, envs[0])
        assert envs[0].profile_id == 'sql-analytics@1'
        assert Path(envs[0].workspace, 'shop.db').is_file()
        challenger.topic_profiles.by_card[card.card_id] = []
        fallback_control = challenger._build_challenge_prompt()
        fallback_knowledge = challenger._build_challenge_prompt()
        assert user_data_get(fallback_control['user_data'], 'profile_id') == 'docs-retrieval@1'
        assert user_data_get(fallback_knowledge['user_data'], 'profile_id') == 'docs-retrieval@1'
    finally:
        challenger.close()


class _ActingRollout:
    """Replace model generation only; keep the production followup/verifier path."""

    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()
        self.followups = []

    def __call__(self, prompts, *, tool_manager, followup_fn):
        with self.lock:
            index = self.counter
            self.counter += 1
        env = next(iter(tool_manager._tools.values()))._env
        assert env.profile_id == 'sql-analytics@1'
        assert env.run_script(BUILD)[0] == 0
        trajectory = dict(prompts[0])
        trajectory['messages'] = list(trajectory['messages'])
        trajectory.update(labels=[1, 2], input_ids=[1, 2])
        if index == 0:
            trajectory['messages'].append({'role': 'assistant', 'content': 'truncated'})
            return [trajectory]
        check_prompt = followup_fn(trajectory, 0)[0]
        assert 'DERIVED_OUTPUTS' in check_prompt
        trajectory['messages'].append({'role': 'assistant', 'content': '```python\n' + CHECK + '\n```'})
        statement_prompt = followup_fn(trajectory, 1)[0]
        assert P.START_SEEDED in statement_prompt
        self.followups.append(statement_prompt)
        trajectory['messages'].append({
            'role':
            'assistant',
            'content':
            'Read customers from shop.db. Write answer.json with the count of customers.'
        })
        return [trajectory]

    def close(self):
        pass


class _SolvingRollout:

    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()
        self.profiles = []

    def __call__(self, prompts, *, env, **kwargs):
        assert env.profile_id == 'sql-analytics@1'
        assert not Path(env.workspace, 'answer.json').exists()
        with self.lock:
            index = self.counter
            self.counter += 1
            self.profiles.append(env.profile_id)
        assert env.run_script(BUILD)[0] == 0
        if index % 2:
            Path(env.workspace, 'answer.json').write_text('{"count": -1}')
        return [dict(prompts[0], labels=[3, 4], input_ids=[3, 4])]

    def close(self):
        pass


def test_curriculum_runs_propose_verify_solve_reward_record_and_history(tmp_path):
    pytest.importorskip('sklearn')
    save_dir = tmp_path / 'run'
    bank = TaskBank(str(save_dir / 'tasks.jsonl'))
    envs = build_slot_envs(
        num_envs=2, workspace_root=str(tmp_path / 'slots'), profile_ids=['docs-retrieval@1', 'python-compute@1'])
    solver = _SolvingRollout()
    challenger = RsiChallenger(
        None,
        envs=envs,
        profile_ids=['sql-analytics@1'],
        num_challenger_rollouts=2,
        num_solver_rollouts=2,
        pass_band=(1, 1),
        solver_rollout=solver,
        task_bank=bank,
        novelty_floor=0.5,
        save_dir=str(save_dir))
    actor = _ActingRollout()
    challenger._rollout = actor
    try:
        assert challenger._launch()
        deadline = time.monotonic() + 20
        while challenger._jobs and time.monotonic() < deadline:
            time.sleep(0.01)
        assert challenger._jobs == 0
        proposing, solving = challenger._finished.get(timeout=1)
        assert len(proposing) == len(solving) == 1
        assert sorted(t['rewards'] for t in solving[0]) == [0.0, 1.0]
        assert all(user_data_get(t['user_data'], 'profile_id') == 'sql-analytics@1' for t in solving[0])
        assert max(t['rewards'] for t in proposing[0]) > 0
        assert actor.followups and solver.profiles == ['sql-analytics@1'] * 2
        assert len(bank) == 1
        records = [json.loads(line) for line in (save_dir / 'trajs/index.jsonl').read_text().splitlines()]
        kept = next(r for r in records if r['outcome'] == 'kept')
        assert {r['outcome'] for r in records if r['side'] == 'solve'} == {'passed', 'failed'}
        metadata = kept['user_data']
        assert user_data_get(metadata, 'selected')
        assert user_data_get(metadata, 'profile_id') == 'sql-analytics@1'
        assert user_data_get(metadata, 'verification')['mutation_sensitive_path'] == 'answer.json'
        assert user_data_get(metadata, 'novelty') == 1.0
        assert TaskBank(str(save_dir / 'tasks.jsonl')).statements() == bank.statements()
        recovered = TaskBank(str(save_dir / 'recovered.jsonl'))
        assert recovered.ingest_recorder(str(save_dir)) == 1
        assert recovered.statements() == bank.statements()
        assert (save_dir / 'novelty_scores.jsonl').is_file()
    finally:
        challenger.close()


def _proposal(statement, n_pass=1):
    task = attach_user_data({'messages': [{'role': 'user', 'content': statement}]}, check_script='assert True')
    return _ProposalResult(
        attach_user_data({'messages': []}), group_id='group', task=task, attempts=[({}, i < n_pass) for i in range(2)])


def test_computed_novelty_changes_rewards_and_task_selection(tmp_path, monkeypatch):

    class Scorer:

        def __init__(self, corpus, **kwargs):
            assert 'earlier historical task' in corpus

        def score(self, statements):
            return [0.0, 1.0]

    monkeypatch.setattr('cookbook.rsi.agentic.challenger.NoveltyScorer', Scorer)
    bank = TaskBank(str(tmp_path / 'tasks.jsonl'))
    bank.add('earlier historical task')
    challenger = RsiChallenger(
        None,
        envs=[LocalEnv(workspace=str(tmp_path / 'env'))],
        num_solver_rollouts=2,
        pass_band=(1, 1),
        task_bank=bank,
        novelty_floor=0.5)
    try:
        proposals = [_proposal('repeated idea'), _proposal('different idea')]
        challenger._score(_Unit('group', proposals, 0))
        assert proposals[0].reward == pytest.approx(proposals[1].reward * 0.5)
        assert 'different idea' in bank.statements() and 'repeated idea' not in bank.statements()
        assert novelty_factor(None, 0.5) == 1.0
        assert challenger.challenger_reward(0) == 0.0
    finally:
        challenger.close()


@pytest.mark.parametrize('raises', [False, True])
def test_judge_receives_history_and_siblings_and_drops_unscored_groups(tmp_path, raises):
    bank = TaskBank(str(tmp_path / 'tasks.jsonl'))
    bank.add('historical example')
    calls = []

    def judge(payload, **kwargs):
        calls.append(payload)
        assert kwargs['extra_body'] == {'thinking_budget': 128}
        if raises:
            raise RuntimeError('API unavailable')
        return [SimpleNamespace(scores={'novelty': None}, error='unavailable') for _ in payload]

    challenger = RsiChallenger(
        None,
        envs=[LocalEnv(workspace=str(tmp_path / 'env'))],
        num_solver_rollouts=2,
        pass_band=(1, 1),
        task_bank=bank,
        novelty_source='judge',
        api_kwargs={'extra_body': {
            'thinking_budget': 128
        }},
        novelty_judge=judge,
        novelty_tries=2)
    try:
        proposals = [_proposal('first'), _proposal('second')]
        challenger._score(_Unit('group', proposals, 0))
        assert len(calls) == 2
        assert calls[0][0]['references'] == ['second', 'historical example']
        assert challenger._finished.get(timeout=1) == ([], [])
        assert all(p.outcome == 'novelty_unscored' for p in proposals)
        assert challenger.n_proposed == 2
        assert all(p.n_pass == 1 for p in proposals)
        assert bank.statements() == ['historical example']
    finally:
        challenger.close()


def test_task_seed_sampling_advances_and_resets_per_epoch():
    calls = []

    class Store:
        manifest_sha256 = 'a' * 64

        def select(self, group, seed, epoch):
            calls.append((group, seed, epoch))
            return dict(
                id='task',
                global_index=group,
                provenance={},
                task_direction='analyze logs',
                required_artifact='report.json',
                constraints=['use the given inputs'])

    seeder = TaskSeedSeeder(Store(), seed=17)
    assert 'analyze logs' in seeder()
    seeder()
    seeder.set_epoch(2)
    seeder()
    assert calls == [(0, 17, 0), (1, 17, 0), (0, 17, 2)]
    assert seeder.last_route.metadata()['task_seed_id'] == 'task'


def test_length_penalty_uses_generated_tokens_and_stays_disabled_without_budget():
    trajectory = {'labels': [-100] * 100 + [1] * 9}
    assert overlong_penalty(trajectory, 10, 4, 0.5) == 0.375
    assert overlong_penalty(trajectory, 0, 4, 0.5) == 0.0


def test_affinity_producer_release_is_consumed_by_the_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv('TWINKLE_WORLD_DB_MODULE', raising=False)
    cards = []
    for index, leaf in enumerate(('SQL database query optimization', 'Web source verification')):
        cards.append(
            dict(
                card_id=f'{index:024x}',
                global_index=index,
                topic_path={
                    'root': 'systems',
                    'leaf': leaf
                },
                knowledge='reference material',
                token_count=10,
                model='test-model',
                checkpoint_sha256='a' * 64,
                tokenizer_sha256='b' * 64,
                prompt_sha256='c' * 64,
                schema_version=1,
                finish_reason='eos',
                created_unix=0,
                decode_params={},
                generated_token_count=10,
                seed=0,
                local_index=index,
                num_shards=1,
                shard_id=0))
    frozen = tmp_path / 'bank/cards.jsonl'
    frozen.parent.mkdir()
    frozen.write_text(''.join(json.dumps(card) + '\n' for card in cards))
    expected = hashlib.sha256(frozen.read_bytes()).hexdigest()
    output = tmp_path / 'overlay'
    manifest = generate_overlay(frozen, output, expected_sha256=expected, expected_count=2)
    assert manifest['card_count'] == 2
    runtime = TopicProfiles(str(output / 'affinities.jsonl'))
    assert runtime.for_card(SimpleNamespace(card_id=cards[0]['card_id'])) == 'sql-analytics@1'
    assert runtime.for_card(SimpleNamespace(card_id=cards[1]['card_id'])) == 'web-research@1'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == expected
