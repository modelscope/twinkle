# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-process loopback of the HCCL delta-sync protocol (gloo stands in for PyHccl, zmq is real).

Validates message ordering / rendezvous / decoding across real processes. It does NOT validate HCCL.
Every scenario is bounded by ``run_loopback(timeout=...)`` (children are always killed afterwards).
"""
from __future__ import annotations

from ._hccl_loopback import build_rounds, make_state, run_loopback, state_to_bytes, wire_bytes

W1, W2, B = 'layer.w1', 'layer.w2', 'layer.bias'


def _assert_states(res, expected, n_recv, session=0, rounds=None):
    rounds = range(len(expected)) if rounds is None else rounds
    for rk in range(1, n_recv + 1):
        got = res.recv[(session, rk)]
        for r in rounds:
            assert got[r]['error'] is None, (rk, r, got[r]['error'])
            assert got[r]['state'] == state_to_bytes(expected[r]), f'receiver {rk} differs after round {r}'


def _flags(payload):
    return [m['is_last'] for m in payload['metas']]


def test_seed_only_values_flushes():
    init = make_state(0)
    res = run_loopback([(init, [])], bucket_size=1024)
    _assert_states(res, [init], 2)
    for rk in (1, 2):
        metas = res.recv[(0, rk)][0]['metas']
        assert len(metas) == 3 and all(m['encoding'] == 'dense' and m['pos_numel'] == 0 for m in metas)
        assert _flags(res.recv[(0, rk)][0]) == [False, False, True]
        assert res.recv[(0, rk)][0]['encoding'] == 'dense'
    assert res.trainer_sent[0] == (1, 1)  # communicator built once, diff base primed once


def test_seed_then_sparse_steady_rounds():
    init = make_state(1)
    rounds, expected = build_rounds(init, [{W1: 0, W2: 500, B: 3}, {W1: 5, W2: 0, B: 64}], seed=2)
    res = run_loopback([(init, rounds)], bucket_size=1024)
    _assert_states(res, expected, 2)
    seed_bytes = None
    for rk in (1, 2):
        got = res.recv[(0, rk)]
        seed_bytes = wire_bytes(got[0]['metas'])
        for r in (1, 2):
            assert got[r]['encoding'] == 'indices'
            flags = _flags(got[r])
            assert flags == [False] * (len(flags) - 1) + [True]
            # round 1: w2 (3000 B) overflows the 1 KiB bucket -> 2 flushes; round 2 fits in one
            assert len(flags) == (2 if r == 1 else 1)
            assert wire_bytes(got[r]['metas']) < seed_bytes
    assert len(res.recv[(0, 1)][1]['metas']) == len(res.recv[(0, 2)][1]['metas'])
    assert res.trainer_rounds[(0, 1)][0]['checkpoint_engine/flushes'] == float(len(res.recv[(0, 1)][1]['metas']))


def test_zero_change_round_is_terminal_only_and_comm_persists():
    init = make_state(3)
    rounds, expected = build_rounds(init, [{W1: 7, W2: 20, B: 2}, {W1: 0, W2: 0, B: 0}, {W2: 33}], seed=4)
    res = run_loopback([(init, rounds)], bucket_size=1024)
    _assert_states(res, expected, 2)
    assert all((expected[1][n] == expected[2][n]).all() for n in expected[1])
    for rk in (1, 2):
        m = res.recv[(0, rk)][2]['metas']
        assert len(m) == 1 and m[0]['terminal_empty'] and m[0]['is_last']
        assert res.recv[(0, rk)][2]['state'] == res.recv[(0, rk)][1]['state']
    # the communicator was built exactly once, at the first publish, and persisted for all 4 rounds
    assert res.trainer_sent[0][0] == 1
    assert all(res.trainer_rounds[(0, r)][1] == 1 for r in range(4))


def test_tiny_bucket_slices_one_param_into_pieces():
    init = make_state(5)
    rounds, expected = build_rounds(init, [{W2: 500, B: 10}, {W2: 301}], seed=6)
    res = run_loopback([(init, rounds)], bucket_size=1024, max_entry_elems=64)
    _assert_states(res, expected, 2)
    for rk in (1, 2):
        metas = res.recv[(0, rk)][1]['metas']
        assert len(metas) > 1
        entries = [p for m in metas for p in m['spec']['params'] if p['name'] == W2]
        assert len(entries) >= 8  # 500 changes / 64 per entry
        assert all(p['val_end'] - p['val_start'] <= 64 for p in entries)


def test_close_then_reseed_new_session():
    init0, init1 = make_state(7), make_state(8)
    r0, e0 = build_rounds(init0, [{W2: 40, B: 4}], seed=9)
    r1, e1 = build_rounds(init1, [{W1: 30, W2: 60}], seed=10)
    res = run_loopback([(init0, r0), (init1, r1)], bucket_size=1024, timeout=90)
    _assert_states(res, e0, 2, session=0)
    _assert_states(res, e1, 2, session=1)
    for s in (0, 1):
        assert res.trainer_sent[s] == (1, 1)  # fresh engine: one new communicator, one new seed
        closed = res.trainer_closed[s]
        assert closed == dict(comm=None, socket=None, seeded=False, rank=None)
    # receivers of the second session saw a FULL seed (dense) first, not a delta
    assert res.recv[(1, 1)][0]['encoding'] == 'dense'


def test_corrupted_payload_is_detected_end_to_end():
    init = make_state(11)
    rounds, expected = build_rounds(init, [{W2: 50, B: 3}], seed=12)
    # broadcast calls: 0 = seed values; 1 = steady positions; 2 = steady values (flipped after checksum)
    res = run_loopback([(init, rounds)], bucket_size=1 << 20, corrupt_call=2)
    _assert_states(res, expected, 2, rounds=[0])
    for rk in (1, 2):
        err = res.recv[(0, rk)][1]['error']
        assert err is not None and err[0] == 'RuntimeError' and 'checksum mismatch' in err[1]
        assert res.recv[(0, rk)][1]['state'] == state_to_bytes(expected[0])  # corrupt flush NOT applied
    # single-flush round: the trainer is never blocked by the failed receivers and exits cleanly
    assert res.trainer_exited and (0, 1) in res.trainer_rounds


def test_dp_layout_four_receivers_shuffled_start():
    init = make_state(13)
    rounds, expected = build_rounds(init, [{W1: 9, W2: 200, B: 5}, {W2: 77}], seed=14)
    res = run_loopback([(init, rounds)], n_recv=4, bucket_size=2048, shuffle=True, seed=15, timeout=90)
    _assert_states(res, expected, 4)
    assert res.trainer_sent[0][0] == 1


def test_sub_after_join_loses_first_manifest():
    """Negative demo of the SUB-before-join rule. The late SUB connects 1s after the group formed, long
    after the trainer published, so the loss is deterministic by construction (not a timing race)."""
    init = make_state(16)
    res = run_loopback([(init, [])], sub_after_join=1.0, pump_timeout_ms=3000, allow_errors=True,
                       require_trainer_exit=False, timeout=45)
    for rk in (1, 2):
        p = res.recv[(0, rk)][0]
        assert p['error'] is not None and p['error'][0] == 'Again'  # nothing ever arrived
        assert p['metas'] == []
    assert (0, 0) not in res.trainer_rounds  # trainer cannot finish: receivers never joined the broadcast
