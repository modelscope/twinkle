# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect110: chat-protocol continue-on-length in ``MultiTurnRollout._serve``.

A chat-protocol agent (ms-agent) answers ``finish_reason='length'`` by appending the
partial assistant message to its history and re-requesting: the endpoint must CONTINUE
that open turn from the ledger's current state -- not open a segment, which re-encodes
an assistant-ending prompt with ``add_generation_prompt=True`` and crashes (the template
contract refuses it), or silently resets the turn/token account on the message face.
The agent's merged view of the split replies follows its send-time strip recurrence
(``echo_k = strip(echo_{k-1} + piece_k)``), NOT the raw concatenation -- boundary
whitespace is gone, and a raw-concatenation view reads the second continuation as a
rewrite (the attempt4 crash).

Both ledger faces are driven through the real ``_serve``: message-level (template=None)
and token-in (a char-level fake template enforcing the real encode contract). The fakes
are confined to the engine's sampled reply and the tokenizer; detection, canonicalisation
and the segment/turn bookkeeping are the product code under test. The full oracle matrix
and its two-layer HEAD-worktree reverse-verification live in the regression campaign's
``check2/tools/probe_defect110_continuation.py`` / ``probe_defect110_reverse.py``.
"""
from __future__ import annotations

from twinkle.data_format.sampling import SampledSequence, SamplingParams
from twinkle.template.base import Template
from twinkle_agentic.rollout.endpoint import EndpointRequest
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout, _Episode

U1 = {'role': 'user', 'content': 'hello'}
U2 = {'role': 'user', 'content': 'tool result'}


def _a(content):
    return {'role': 'assistant', 'content': content}


class _Engine:
    """Scripted sampling oracle. token_face builds the sampler's new_input_feature
    (prompt + sampled ids, labelled, post-pipeline shifted order as the real sampler
    returns it); else the message-level path."""

    def __init__(self, script, token_face=False, tokenizer=None):
        self.script = list(script)
        self.token_face = token_face
        self.tokenizer = tokenizer
        self.calls = 0

    def __call__(self, view, state, params):
        text, stop = self.script[self.calls]
        self.calls += 1
        if not self.token_face:
            return SampledSequence(stop_reason=stop, tokens=[], decoded=text)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        feature = dict(state)
        feature['input_ids'] = list(state['input_ids']) + tokens
        # shifted order: drop the trailing -100, append the new labels, re-mask the tail
        feature['labels'] = list(state.get('labels') or [])[:-1] + tokens + [-100]
        feature['messages'] = list(state.get('messages') or []) + [_a(text)]
        return SampledSequence(stop_reason=stop, tokens=tokens, decoded=text,
                               new_input_feature=feature, logprobs=[-0.5] * len(tokens))


class _CharTokenizer:
    SPECIALS = ('<|im_start|>', '<|im_end|>')

    def __init__(self):
        self._s2i, self._i2s = {}, {}
        for s in self.SPECIALS:
            self._add(s)

    def _add(self, tok):
        if tok not in self._s2i:
            i = len(self._s2i)
            self._s2i[tok] = i
            self._i2s[i] = tok
        return self._s2i[tok]

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for sp in self.SPECIALS:
                if text.startswith(sp, i):
                    ids.append(self._add(sp))
                    i += len(sp)
                    break
            else:
                ids.append(self._add(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        toks = [self._i2s[int(i)] for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in self.SPECIALS]
        return ''.join(toks)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        s = ''
        for m in messages:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += '<|im_start|>assistant\n'
        return self.encode(s) if tokenize else s


class _FakeTemplate:
    """Char-level template enforcing the REAL encode contract: a conversation ending
    with an assistant turn refuses add_generation_prompt=True (swift intent.py).
    finalize_appended_features rolls labels left by one (the post-pipeline shift the
    real Template applies), so features stay in the sampler's shifted order."""
    model_id = 'qwen-fake'
    truncation_strategy = 'right'
    enable_thinking = False
    max_length = None
    render_delta = Template.render_delta
    decode_tool_calls = staticmethod(Template.decode_tool_calls)
    _collapse_system_messages = staticmethod(Template._collapse_system_messages)
    _engine_input_ids = Template._engine_input_ids

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.encode_calls = 0

    def encode(self, trajectory, add_generation_prompt=False):
        self.encode_calls += 1
        messages = trajectory.get('messages', [])
        if add_generation_prompt and messages and messages[-1].get('role') == 'assistant':
            raise ValueError('add_generation_prompt=True but the conversation already ends with an '
                             'assistant turn; drop the trailing response or encode it as a complete conversation.')
        s = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=add_generation_prompt)
        input_ids = self.tokenizer.encode(s)
        pif = dict(trajectory)
        pif['input_ids'] = input_ids
        pif['labels'] = [-100] * len(input_ids)
        return self.finalize_appended_features([pif])[0]

    def finalize_appended_features(self, inputs):
        out = []
        for pif in inputs:
            pif = dict(pif)
            n = len(pif['input_ids'])
            labels = list(pif.get('labels') or [])
            if len(labels) != n:
                raise RuntimeError('labels length != input_ids length')
            pif['labels'] = labels[1:] + labels[:1]  # np.roll(labels, -1): shifted order
            pif['attention_mask'] = [1] * n
            pif['position_ids'] = list(range(n))
            pif['length'] = n
            out.append(pif)
        return out

    def parse_tool_call(self, decoded):
        return []

    def tool_call_errors(self, decoded):
        return []

    def clean_tool_call(self, text):
        return text


def _make(script, token_face=False, segment_policy='drop', max_length=None):
    tokenizer = _CharTokenizer() if token_face else None
    template = _FakeTemplate(tokenizer) if token_face else None
    if template is not None:
        template.max_length = max_length
    engine = _Engine(script, token_face=token_face, tokenizer=tokenizer)
    rollout = MultiTurnRollout(response_callback=engine, template=template, segment_policy=segment_policy)
    rollout.harness = None
    return rollout, engine, template, tokenizer


def _ep(messages, max_tokens=None):
    return _Episode('e0', {'messages': list(messages), 'episode_id': 'e0'},
                    SamplingParams(max_tokens=max_tokens))


def _serve(rollout, ep, messages):
    request = EndpointRequest(messages=list(messages), tools=[], params={})
    rollout._serve(ep, request)
    return request.reply.result()


# ---------------------------------------------------------------- message face

def test_m1_continue_merges():
    rollout, engine, _, _ = _make([('p1', 'length'), ('p2', 'stop')])
    ep = _ep([U1])
    first = _serve(rollout, ep, [U1])
    assert first['finish_reason'] == 'length' and first['message']['content'] == 'p1'
    reply = _serve(rollout, ep, [U1, _a('p1')])
    assert reply['message']['content'] == 'p2', 'reply must carry ONLY the continuation'
    assert reply['finish_reason'] == 'stop'
    assert engine.calls == 2 and ep.segments == 1 and ep.ledger.turns == 1
    assert ep.ledger.messages == [U1, _a('p1p2')]


def test_m2_repeated_continuation():
    rollout, engine, _, _ = _make([('p1', 'length'), ('p2', 'length'), ('p3', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('p1')])
    reply = _serve(rollout, ep, [U1, _a('p1p2')])  # the agent's MERGED echo
    assert reply['message']['content'] == 'p3' and reply['finish_reason'] == 'stop'
    assert engine.calls == 3 and ep.segments == 1 and ep.ledger.turns == 1
    assert ep.ledger.messages == [U1, _a('p1p2p3')]


def test_m3_continuation_retry_replays():
    rollout, engine, _, _ = _make([('p1', 'length'), ('p2', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    first = _serve(rollout, ep, [U1, _a('p1')])
    again = _serve(rollout, ep, [U1, _a('p1')])  # identical retry of the continuation
    assert again == first
    assert engine.calls == 2 and ep.segments == 1 and ep.ledger.turns == 1
    assert ep.ledger.messages == [U1, _a('p1p2')]


def test_m4_merged_suffix_same_segment():
    rollout, engine, _, _ = _make([('p1', 'length'), ('p2', 'stop'), ('ok', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('p1')])
    reply = _serve(rollout, ep, [U1, _a('p1p2'), U2])  # merged turn + environment suffix
    assert reply['message']['content'] == 'ok'
    assert engine.calls == 3 and ep.segments == 1
    assert ep.ledger.turns == 2
    assert ep.ledger.messages == [U1, _a('p1p2'), U2, _a('ok')]


def test_m5_length_then_user_direct():
    rollout, engine, _, _ = _make([('p1', 'length'), ('ok', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    reply = _serve(rollout, ep, [U1, _a('p1'), U2])  # agent treats the partial as final
    assert reply['message']['content'] == 'ok'
    assert ep.segments == 1 and ep.ledger.turns == 2
    assert ep.ledger.messages == [U1, _a('p1'), U2, _a('ok')]


def test_m6_foreign_assistant_rewrite():
    rollout, engine, _, _ = _make([('ok', 'stop'), ('ok2', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('other')])  # rewritten assistant content: not our reply
    assert ep.segments == 2 and engine.calls == 2
    assert ep.ledger.messages == [U1, _a('other'), _a('ok2')]


def test_m7_normal_two_turn():
    rollout, engine, _, _ = _make([('ok', 'stop'), ('ok2', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('ok'), U2])
    assert ep.segments == 1 and ep.ledger.turns == 2
    assert ep.ledger.messages == [U1, _a('ok'), U2, _a('ok2')]


def test_m8_whitespace_boundary_repeated():
    # The attempt4 crash shape: ms-agent strips stored content at send time, so the merged
    # echo LOSES boundary whitespace ('answer = ' + '42 and ' echoes as 'answer =42 and').
    # A raw-concatenation merged view reads the second continuation as a rewrite.
    rollout, engine, _, _ = _make([('answer = ', 'length'), ('42 and ', 'length'), ('done.', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    r2 = _serve(rollout, ep, [U1, _a('answer =')])  # echo_1 = strip(partial)
    assert r2['message']['content'] == '42 and ' and r2['finish_reason'] == 'length'
    r3 = _serve(rollout, ep, [U1, _a('answer =42 and')])  # echo_2 = strip(echo_1 + piece_2)
    assert r3['message']['content'] == 'done.' and r3['finish_reason'] == 'stop'
    assert engine.calls == 3 and ep.segments == 1 and ep.ledger.turns == 1
    assert ep.ledger.messages == [U1, _a('answer = 42 and done.')]  # ledger keeps the RAW stream


def test_m9_whitespace_boundary_merged_suffix():
    rollout, engine, _, _ = _make([('answer = ', 'length'), ('42.', 'stop'), ('ok', 'stop')])
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('answer =')])
    reply = _serve(rollout, ep, [U1, _a('answer =42.'), U2])  # merged echo + environment suffix
    assert reply['message']['content'] == 'ok'
    assert ep.segments == 1 and ep.ledger.turns == 2
    assert ep.ledger.messages == [U1, _a('answer = 42.'), U2, _a('ok')]


# ------------------------------------------------------------------ token face

def test_t1_token_accounting():
    rollout, engine, template, tok = _make([('p1', 'length'), ('p2', 'stop')], token_face=True)
    ep = _ep([U1])
    prompt_ids = tok.encode(tok.apply_chat_template([U1], add_generation_prompt=True))
    _serve(rollout, ep, [U1])
    reply = _serve(rollout, ep, [U1, _a('p1')])
    assert reply['message']['content'] == 'p2'
    p1, p2 = tok.encode('p1'), tok.encode('p2')
    feature = ep.ledger.input_feature
    assert feature['input_ids'] == prompt_ids + p1 + p2
    input_order_labels = [-100] * len(prompt_ids) + p1 + p2
    assert feature['labels'] == input_order_labels[1:] + input_order_labels[:1]
    assert ep.ledger._logprobs == [-0.5] * (len(p1) + len(p2))
    assert ep.ledger.turns == 1 and ep.segments == 1
    assert ep.ledger.messages == [U1, _a('p1p2')]
    assert template.encode_calls == 1, 'continuation must generate from state, never re-encode'


def test_t2_no_reencode_crash():
    # The defect110 crash shape: pre-fix the continuation request re-encodes an
    # assistant-ending prompt and the template contract raises. Same drive as T1,
    # asserted on the absence of the raise and of any second encode.
    rollout, engine, template, _ = _make([('p1', 'length'), ('p2', 'stop')], token_face=True)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('p1')])  # pre-fix: ValueError here
    assert template.encode_calls == 1 and engine.calls == 2


def test_t3_merged_suffix_token_bridge():
    rollout, engine, template, tok = _make([('p1', 'length'), ('p2', 'stop'), ('ok', 'stop')],
                                           token_face=True)
    ep = _ep([U1])
    prompt_len = len(tok.encode(tok.apply_chat_template([U1], add_generation_prompt=True)))
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('p1')])
    before = list(ep.ledger.input_feature['input_ids'])
    reply = _serve(rollout, ep, [U1, _a('p1p2'), U2])
    assert reply['message']['content'] == 'ok'
    feature = ep.ledger.input_feature
    assert len(feature['input_ids']) > len(before), 'the observation bridge must add tokens'
    # unshift to input order: prompt masked, p1+p2 trainable, bridge masked, ok trainable
    labels = feature['labels']
    input_order = [labels[-1]] + labels[:-1]
    p1, p2, ok = tok.encode('p1'), tok.encode('p2'), tok.encode('ok')
    bridge_len = len(feature['input_ids']) - prompt_len - len(p1) - len(p2) - len(ok)
    assert bridge_len > 0
    assert input_order == [-100] * prompt_len + p1 + p2 + [-100] * bridge_len + ok
    assert ep.ledger.turns == 2 and ep.segments == 1
    assert template.encode_calls == 1, 'merged suffix must observe, not re-encode'
    assert ep.ledger.messages == [U1, _a('p1p2'), U2, _a('ok')]


def test_t4_unfit_continuation_truncates():
    tokenizer = _CharTokenizer()
    prompt_len = len(tokenizer.encode(tokenizer.apply_chat_template([U1], add_generation_prompt=True)))
    rollout, engine, _, _ = _make([('p1', 'length'), ('p2', 'stop')], token_face=True,
                                  max_length=prompt_len + 4)
    ep = _ep([U1], max_tokens=64)
    _serve(rollout, ep, [U1])  # first turn capped to the 4-token room, banks 'p1' (2 tokens)
    assert ep.ledger.turns == 1
    request = EndpointRequest(messages=[U1, _a('p1')], tools=[], params={})
    rollout._serve(ep, request)
    try:
        request.reply.result()
        raise AssertionError('continuation with no room must end the episode truncated')
    except RuntimeError as error:
        assert 'length limit' in str(error)
    assert ep.truncated and engine.calls == 1
    assert ep.ledger.messages == [U1, _a('p1')], 'the banked turn survives truncation'


def test_t5_continuation_retry_token():
    rollout, engine, template, _ = _make([('p1', 'length'), ('p2', 'stop')], token_face=True)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    first = _serve(rollout, ep, [U1, _a('p1')])
    again = _serve(rollout, ep, [U1, _a('p1')])
    assert again == first
    assert engine.calls == 2 and template.encode_calls == 1
    assert ep.ledger.turns == 1 and ep.segments == 1


def test_t6_normal_two_turn_token():
    rollout, engine, template, _ = _make([('ok', 'stop'), ('ok2', 'stop')], token_face=True)
    ep = _ep([U1])
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('ok'), U2])
    assert ep.ledger.turns == 2 and ep.segments == 1 and template.encode_calls == 1
    assert ep.ledger.messages == [U1, _a('ok'), U2, _a('ok2')]


def test_t7_whitespace_boundary_token():
    # Token face of the attempt4 crash: pre-follow-up the second continuation re-encodes
    # an assistant-ending prompt and the template contract raises.
    rollout, engine, template, tok = _make([('answer = ', 'length'), ('42 and ', 'length'), ('done.', 'stop')],
                                           token_face=True)
    ep = _ep([U1])
    prompt_ids = tok.encode(tok.apply_chat_template([U1], add_generation_prompt=True))
    _serve(rollout, ep, [U1])
    _serve(rollout, ep, [U1, _a('answer =')])
    reply = _serve(rollout, ep, [U1, _a('answer =42 and')])
    assert reply['message']['content'] == 'done.'
    pieces = tok.encode('answer = ') + tok.encode('42 and ') + tok.encode('done.')
    assert ep.ledger.input_feature['input_ids'] == prompt_ids + pieces
    assert ep.ledger.turns == 1 and ep.segments == 1 and template.encode_calls == 1
    assert ep.ledger.messages == [U1, _a('answer = 42 and done.')]


