# Copyright (c) ModelScope Contributors. All rights reserved.
"""Routing a mined knowledge bank into the challenger's opening instruction.

The property under test is coverage. A challenger left to invent topics returns
to the ones it can already verify, and drawing uniformly from a bank leaves the
rest to chance: with the epoch fixed the draw is a pure function of the group
index, so every pass replays the same anchors. Walking a seeded permutation reaches every card in
``len(cards) / blocks_per_epoch`` epochs and repeats none within a pass, which
is what these tests pin.
"""
import hashlib
import json

import pytest

from twinkle_agentic.challenger import (KNOWLEDGE_ARMS, KnowledgeCardSeeder, KnowledgeStore)

SCHEMA = 1


def card(i, root='systems', leaf=None):
    return {
        'card_id': f'card-{i:04d}',
        'global_index': i,
        'topic_path': {
            'root': root,
            'leaf': leaf or f'leaf topic number {i}'
        },
        'knowledge': f'Knowledge paragraph {i} about {leaf or i}.',
        'token_count': 40 + i,
        'model': 'test-model',
        'checkpoint_sha256': 'a' * 64,
        'tokenizer_sha256': 'b' * 64,
        'schema_version': SCHEMA,
        'finish_reason': 'eos',
    }


def write_bank(tmp_path, n=8, monkeypatch=None, mutate=None):
    directory = tmp_path / 'bank'
    directory.mkdir(exist_ok=True)
    records = [card(i) for i in range(n)]
    if mutate is not None:
        records = mutate(records)
    path = directory / 'knowledge_cards.jsonl'
    body = '\n'.join(json.dumps(r) for r in records) + '\n'
    path.write_text(body, encoding='utf-8')
    (directory / 'manifest.json').write_text(
        json.dumps({
            'schema_version': SCHEMA,
            'knowledge_cards_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest()
        }),
        encoding='utf-8')
    return str(path)


@pytest.fixture
def bank(tmp_path):
    return write_bank(tmp_path)


# ------------------------------------------------------------------- the store
def test_loads_a_legacy_bank(bank):
    store = KnowledgeStore(bank, seed=1701)
    assert len(store.cards) == 8
    assert store.schema_version == SCHEMA


def test_a_card_carries_its_own_provenance(bank):
    one = KnowledgeStore(bank, seed=1).cards[0]
    assert one.card_id == 'card-0000'
    assert one.source_cue == 'systems > leaf topic number 0'
    meta = one.metadata()
    assert meta['knowledge_file_sha256'] == hashlib.sha256(
        open(bank, 'rb').read()).hexdigest()
    assert len(meta['text_sha256']) == 64


def test_the_bank_sha_can_be_pinned_by_the_environment(bank, monkeypatch):
    # A run that names the bank it used has to fail loudly when handed another
    # one: silently training on a different bank produces results that cannot be
    # reproduced from the run's own record.
    real = hashlib.sha256(open(bank, 'rb').read()).hexdigest()
    monkeypatch.setenv('TWINKLE_KNOWLEDGE_CARDS_SHA256', real)
    assert len(KnowledgeStore(bank, seed=1).cards) == 8
    monkeypatch.setenv('TWINKLE_KNOWLEDGE_CARDS_SHA256', 'f' * 64)
    with pytest.raises(ValueError, match='differs from'):
        KnowledgeStore(bank, seed=1)


def test_a_missing_bank_is_an_error_not_an_empty_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeStore(str(tmp_path / 'nope.jsonl'), seed=1)


def test_a_duplicate_card_id_is_refused(tmp_path):
    path = write_bank(tmp_path, n=4, mutate=lambda rs: [rs[0], rs[0], rs[1], rs[2]])
    with pytest.raises(ValueError, match='duplicate'):
        KnowledgeStore(path, seed=1)


def test_an_unfinished_card_is_refused(tmp_path):
    def cut(rs):
        rs[2]['finish_reason'] = 'length'
        return rs

    path = write_bank(tmp_path, n=4, mutate=cut)
    with pytest.raises(ValueError, match='unfinished'):
        KnowledgeStore(path, seed=1)


def test_an_unsupported_schema_is_refused(tmp_path):
    path = write_bank(tmp_path, n=4)
    manifest = tmp_path / 'bank' / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 99}), encoding='utf-8')
    with pytest.raises(ValueError, match='unsupported schema_version'):
        KnowledgeStore(path, seed=1)


def test_blocks_per_epoch_has_to_be_a_positive_integer(bank):
    with pytest.raises(ValueError, match='positive integer'):
        KnowledgeStore(bank, seed=1, blocks_per_epoch=0)


# ------------------------------------------------------------------- the routing
def test_one_block_is_a_matched_pair_of_arms(bank):
    # Two consecutive groups share an anchor and differ in arm, so the pair is a
    # comparison rather than two unrelated draws.
    store = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    first = store.route(0, epoch=0)
    second = store.route(1, epoch=0)
    assert {first.arm, second.arm} == set(KNOWLEDGE_ARMS)
    assert first.anchor.card_id == second.anchor.card_id


def test_only_the_knowledge_arm_delivers_a_card(bank):
    store = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    for group_id in range(8):
        route = store.route(group_id, epoch=0)
        if route.arm == 'knowledge':
            assert route.delivered is not None
            assert route.delivered.card_id == route.anchor.card_id
        else:
            assert route.delivered is None


def test_routing_is_deterministic_for_a_seed(bank):
    a = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    b = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    assert [a.route(i, i // 4).arm for i in range(16)] == [b.route(i, i // 4).arm for i in range(16)]


def test_a_different_seed_routes_differently(bank):
    a = KnowledgeStore(bank, seed=1, blocks_per_epoch=2)
    b = KnowledgeStore(bank, seed=99, blocks_per_epoch=2)
    arms_a = [a.route(i, i // 4).arm for i in range(16)]
    arms_b = [b.route(i, i // 4).arm for i in range(16)]
    assert arms_a != arms_b


def test_the_epoch_moves_the_arm_order(bank):
    # With the epoch fixed every pass replays the identical sequence, which is
    # the bug that made every pass replay the same anchors.
    store = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    per_epoch = [tuple(store.route(g, epoch=e).arm for g in range(4)) for e in range(4)]
    assert len(set(per_epoch)) > 1


def test_a_pass_covers_every_card_exactly_once(bank):
    # 8 cards, 2 anchors per epoch -> a pass is 4 epochs; 2 arms -> 4 groups per
    # epoch, so 16 groups per pass.
    store = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    seen = []
    for group_id in range(16):
        epoch = group_id // 4
        route = store.route(group_id, epoch)
        if route.arm == 'knowledge':
            seen.append(route.delivered.card_id)
    anchors = {store.route(g, g // 4).anchor.card_id for g in range(16)}
    assert len(anchors) == 8, f'a pass should touch all 8 cards, touched {sorted(anchors)}'
    # 16 groups over 2 arms: half are the control arm and deliver nothing, so the
    # knowledge half is 8 deliveries over 8 distinct cards -- one each, no repeat.
    assert len(seen) == 8
    assert len(set(seen)) == 8


def test_route_metadata_names_the_bank_and_the_arm(bank):
    store = KnowledgeStore(bank, seed=1701)
    meta = store.route(0, epoch=0).metadata()
    assert meta['arm'] in KNOWLEDGE_ARMS
    assert meta['anchor_card_id'].startswith('card-')
    assert meta['schema_version'] == SCHEMA
    assert len(store.manifest_sha256) == 64


# -------------------------------------------------------------------- the seeder
def test_the_seeder_returns_text_for_knowledge_and_nothing_for_the_control(bank):
    store = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    seeder = KnowledgeCardSeeder(store)
    got = [seeder() for _ in range(4)]
    arms = [seeder.store.route(i, i // 4).arm for i in range(4)]
    for text, arm in zip(got, arms):
        if arm == 'knowledge':
            assert isinstance(text, str) and text.startswith('Knowledge paragraph')
        else:
            # None is not a failure here: it is the protocol's "propose from
            # scratch", which is exactly what the control arm means.
            assert text is None


def test_the_seeder_infers_the_epoch_from_its_call_count(bank):
    # 2 blocks per epoch x 2 arms = 4 calls per epoch.
    store = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    seeder = KnowledgeCardSeeder(store)
    delivered = []
    for _ in range(16):
        seeder()
        if seeder.last_route.arm == 'knowledge':
            delivered.append(seeder.last_route.delivered.card_id)
    assert len(set(delivered)) == len(delivered), 'a pass should not repeat a card'
    assert len(delivered) == 8


def test_a_pinned_epoch_overrides_the_inferred_one(bank):
    store = KnowledgeStore(bank, seed=1701, blocks_per_epoch=2)
    seeder = KnowledgeCardSeeder(store)
    seeder.set_epoch(3)
    for _ in range(8):
        seeder()
        assert seeder.last_route is not None
    # Every call used epoch 3, so the anchor cursor never advanced past it.
    assert seeder.store.route(0, 3).anchor.card_id == store.route(0, 3).anchor.card_id


def test_set_epoch_validates(tmp_path, bank):
    seeder = KnowledgeCardSeeder(KnowledgeStore(bank, seed=1))
    for bad in (-1, 'x', True, 1.5):
        with pytest.raises(ValueError, match='non-negative integer'):
            seeder.set_epoch(bad)
    seeder.set_epoch(0)
    seeder.set_epoch(7)


def test_the_seeder_is_deterministic_for_a_seed(bank):
    a = KnowledgeCardSeeder(KnowledgeStore(bank, seed=1701, blocks_per_epoch=2))
    b = KnowledgeCardSeeder(KnowledgeStore(bank, seed=1701, blocks_per_epoch=2))
    assert [a() for _ in range(12)] == [b() for _ in range(12)]


def test_last_route_exposes_what_the_run_needs_to_log(bank):
    seeder = KnowledgeCardSeeder(KnowledgeStore(bank, seed=1701, blocks_per_epoch=2))
    assert seeder.last_route is None
    seeder()
    route = seeder.last_route
    assert route.arm in KNOWLEDGE_ARMS
    assert route.anchor.card_id.startswith('card-')
    assert route.metadata()['schema_version'] == SCHEMA


def test_the_seeder_satisfies_the_cookbook_seeder_protocol():
    # The cookbook's Seeder is a one-method protocol; asserting the shape here
    # keeps the re-export in cookbook/rsi/seed/knowledge.py honest without
    # importing the cookbook from tests.
    import inspect
    signature = inspect.signature(KnowledgeCardSeeder.__call__)
    assert list(signature.parameters) == ['self']
