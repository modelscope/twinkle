# Copyright (c) ModelScope Contributors. All rights reserved.
"""C2-G42 (QUEUE2316): the ``noise_reply`` contract and idempotent replay in ``_serve``.

``MultiTurnRollout._serve`` (twinkle_agentic/rollout/multi_turn.py) answers an agent's
endpoint request three ways: a generation to train on, a harness ``noise_reply`` answered
verbatim and recorded masked (no sampling), or a verbatim replay of ``last_reply`` when a
retry re-sends the identical request. These cases drive the real ``_serve`` over a real
message-level ``MessageLedger`` (template=None path) with a counting ``response_callback``
as the sampling oracle -- the number of engine calls is the quantity the contract pins.

PLAN (case | input | expected | failure it kills):
  none_generates            | noise_reply->None, fresh [u1]        | 1 engine call, segments=1, reply 'ok' | None wrongly served as noise (no sampling)
  replay_idempotent         | re-send identical [u1]               | reply==last_reply, engine calls/segments/ledger unchanged | retry re-samples / re-records / opens a segment
  replay_requires_identical | changed payload [u1']                | NOT replay: engine call increases | over-eager replay on a rewritten request
  noise_extends_masked      | noise dict on extending history      | masked observe, 0 new engine calls, segments unchanged, reply==noise | noise that samples / swaps segment on append / drops the record
  noise_rewrite_opens_seg   | noise dict on rewritten history [U2] | fresh segment (segments+1), observe([msg]), 0 engine calls | noise on compaction not swapping the segment (account drift)
  none_vs_absent_identical  | harness=None vs noise_reply->None    | bit-identical reply/ledger/segments | the None default diverging from no-harness
  replay_after_noise        | re-send identical noise request      | replay the noise reply, 0 engine calls | replay only working for generation replies
  record_sync               | ep.sent after a noise extend         | == canonical(request)+[canonical(noise msg)] | ep.sent drift breaking the next extend/replay
  collect_freeze_on_rewrite | segment_policy=collect, noise rewrite| old segment frozen into closed_rows | collect boundary losing the written-out segment
  replay_precedes_noise     | identical retry after filter changes | replay the old reply, NOT the new noise | replay predicate not checked before noise

A "rewrite" input must differ from ep.sent[:-1] (e.g. [U2] after a [U1] generation): an
identical payload is a replay regardless of noise, because the replay predicate is checked
first. The fakes are confined to the two inputs the contract is ABOUT (the harness's
noise_reply answer and the engine's sampled reply); the ledger, the canonicalisation, the
segment bookkeeping and the replay predicate are the real product code under test.

Reverse-verified (tools/probe_reverse_verify_g42.py): defeating the replay predicate
(_canonical nonce) turns replay_idempotent RED; defeating masked observe turns
noise_extends_masked_no_sample RED; control all GREEN.
"""
from __future__ import annotations

import pytest

from twinkle.data_format.sampling import SampledSequence, SamplingParams
from twinkle_agentic.rollout.endpoint import EndpointRequest
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout, _Episode


class _Engine:
    """The sampling oracle: counts calls and returns a fixed plain-stop reply."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, view, state, params) -> SampledSequence:
        self.calls += 1
        return SampledSequence(stop_reason='stop', tokens=[], decoded='ok')


class _Harness:
    """A harness whose noise_reply answer is programmed per request."""

    def __init__(self) -> None:
        self.noise = None
        self.calls = 0

    def noise_reply(self, messages, tools, params):
        self.calls += 1
        return self.noise


def _noise(content: str) -> dict:
    return {'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop', 'usage': {}}


def _make(segment_policy: str = 'drop', harness=None):
    engine = _Engine()
    rollout = MultiTurnRollout(response_callback=engine, template=None, segment_policy=segment_policy)
    # Set after construction: _serve reads self.harness directly; the constructor's harness
    # validation (env/protocol) belongs to _open, not to the serving contract under test.
    rollout.harness = harness
    return rollout, engine


def _ep(messages):
    return _Episode('e0', {'messages': list(messages), 'episode_id': 'e0'}, SamplingParams())


def _serve(rollout, ep, messages, tools=None):
    request = EndpointRequest(messages=list(messages), tools=list(tools or []), params={})
    rollout._serve(ep, request)
    return request.reply.result()


U1 = {'role': 'user', 'content': 'hello'}
U2 = {'role': 'user', 'content': 'tool result'}
OK = {'role': 'assistant', 'content': 'ok'}


def test_none_generates():
    harness = _Harness()  # noise_reply -> None
    rollout, engine = _make(harness=harness)
    ep = _ep([U1])
    reply = _serve(rollout, ep, [U1])
    assert engine.calls == 1
    assert ep.segments == 1
    assert reply['message']['content'] == 'ok'
    assert ep.ledger.messages == [U1, OK]
    assert harness.calls == 1  # the filter was consulted


def test_replay_idempotent():
    rollout, engine = _make(harness=_Harness())
    ep = _ep([U1])
    first = _serve(rollout, ep, [U1])
    ledger_before = ep.ledger.messages
    segments_before = ep.segments
    again = _serve(rollout, ep, [U1])  # identical retry
    assert again == first
    assert engine.calls == 1  # no second sampling
    assert ep.segments == segments_before  # no new segment
    assert ep.ledger.messages == ledger_before  # no re-record


def test_replay_requires_identical():
    rollout, engine = _make(harness=_Harness())
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [{'role': 'user', 'content': 'different'}])  # rewritten, not a retry
    assert engine.calls == 2  # a changed payload is served, not replayed


def test_noise_extends_masked_no_sample():
    harness = _Harness()
    rollout, engine = _make(harness=harness)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])  # generation: ep.sent = [U1, OK]
    harness.noise = _noise('PONG')
    reply = _serve(rollout, ep, [U1, OK, U2])  # extends by U2
    assert reply == harness.noise
    assert engine.calls == 1  # noise costs no sampling
    assert ep.segments == 1  # an append does not swap the segment
    assert ep.ledger.messages == [U1, OK, U2, {'role': 'assistant', 'content': 'PONG'}]
    assert ep.last_reply == harness.noise


def test_noise_rewrite_opens_segment():
    harness = _Harness()
    rollout, engine = _make(harness=harness)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])  # segments=1, ep.sent=[U1, OK]
    harness.noise = _noise('PONG2')
    # A genuine rewrite: [U2] differs from ep.sent[:-1]=[U1] (so it is NOT a replay) and does
    # not extend (len 1 <= 2) -> the noise swaps in a fresh segment.
    reply = _serve(rollout, ep, [U2])
    assert reply == harness.noise
    assert engine.calls == 1  # still no sampling
    assert ep.segments == 2  # the compaction swapped the segment
    assert ep.ledger.messages == [U2, {'role': 'assistant', 'content': 'PONG2'}]


def test_none_vs_absent_bit_identical():
    # harness=None and a harness whose noise_reply returns None must serve identically.
    r_absent, e_absent = _make(harness=None)
    r_none, e_none = _make(harness=_Harness())
    ep_a, ep_n = _ep([U1]), _ep([U1])
    reply_a = _serve(r_absent, ep_a, [U1])
    reply_n = _serve(r_none, ep_n, [U1])
    assert reply_a == reply_n
    assert ep_a.ledger.messages == ep_n.ledger.messages
    assert ep_a.segments == ep_n.segments
    assert e_absent.calls == e_none.calls == 1


def test_replay_after_noise():
    harness = _Harness()
    rollout, engine = _make(harness=harness)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    harness.noise = _noise('PONG')
    first = _serve(rollout, ep, [U1, OK, U2])  # noise served
    segments_before = ep.segments
    again = _serve(rollout, ep, [U1, OK, U2])  # identical retry of the noise request
    assert again == first  # the noise reply is replayed verbatim
    assert engine.calls == 1
    assert ep.segments == segments_before


def test_record_sync_after_noise():
    from twinkle_agentic.rollout.multi_turn import _canonical
    harness = _Harness()
    rollout, _ = _make(harness=harness)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    harness.noise = _noise('PONG')
    request_messages = [U1, OK, U2]
    _serve(rollout, ep, request_messages)
    expected = [_canonical(m) for m in request_messages] + [_canonical(harness.noise['message'])]
    assert ep.sent == expected  # a drift here would misfire the next extend/replay


def test_collect_freeze_on_noise_rewrite():
    harness = _Harness()
    rollout, engine = _make(segment_policy='collect', harness=harness)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])  # segment 0 banked
    harness.noise = _noise('PONG2')
    _serve(rollout, ep, [U2])  # genuine rewrite freezes segment 0 into closed_rows
    assert ep.segments == 2
    assert len(ep.closed_rows) == 1  # the written-out segment was frozen, not dropped
    assert ep.closed_rows[0]['messages'] == [U1, OK]
    assert engine.calls == 1


def test_replay_takes_precedence_over_noise():
    # An identical retry is replayed even when the harness would now answer noise: the replay
    # predicate is checked first, so a retry never re-serves (the agent gets the reply it saw).
    harness = _Harness()
    rollout, engine = _make(harness=harness)
    ep = _ep([U1])
    first = _serve(rollout, ep, [U1])  # generation 'ok'
    harness.noise = _noise('PONG')  # filter changed its mind
    again = _serve(rollout, ep, [U1])  # identical retry
    assert again == first  # replayed the generation, NOT the new noise
    assert engine.calls == 1
    assert ep.segments == 1