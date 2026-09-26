# Copyright (c) ModelScope Contributors. All rights reserved.
"""The token account of one episode, separated from the policy that grows it.

``MultiTurnRollout._run_one`` is two things braided together. One is policy --
how many turns, what a malformed reply costs, when a follow-up stage is granted.
The other is bookkeeping: which token ids the trajectory now consists of, which
of them the policy produced, and the logprob for each of those. This file is the
second one, and only the second one.

They are split because the policy is not shared and the bookkeeping is. An agent
that ships as its own program drives its own loop and reaches the policy over
HTTP (see ``endpoint.py``), so none of the turn accounting above applies to it --
but the account below applies unchanged, because it is what makes a run trainable
at all:

    the tokens trained on are the tokens the sampler returned

Not text re-encoded afterwards. A tokenizer is free to encode the same string to
different ids depending on what precedes it, so a trajectory rebuilt from its own
transcript can differ from what was sampled -- and every logprob then belongs to
a position that has moved. The gradient is still computed, against the wrong
tokens, and nothing raises. That is why :meth:`record` takes a ``SampledSequence``
and reads ``new_input_feature`` off it rather than encoding anything, and why
:meth:`graft` -- the entry point for an episode driven from outside -- compares
prompt ids against what is already banked and refuses a prompt that does not
extend it, rather than re-encoding to make it fit.
"""
import threading
from typing import Any, Dict, List, Optional, Sequence

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampledSequence
from twinkle.template.base import Template
from twinkle_agentic.utils.token_utils import _to_plain, append_ids, extend_with_bridge


class TurnLedger:
    """Token ids, labels and logprobs for one episode, and nothing else.

    Not thread-safe and not meant to be: one ledger belongs to one episode, and
    an episode runs in one thread. See :class:`LedgerBook` for the many-episode
    case, which is a lock around this and not a change to it.

    Usage is one :meth:`open`, then :meth:`record` after each generation and
    :meth:`observe` for each thing appended that the model did not write, then
    :meth:`merge` to read the episode out. Both appending methods answer False
    when the append did not fit, which is the caller's cue to stop -- the ledger
    is left holding the last state that did fit, so a trajectory that overflowed
    is still complete up to the turn before.

    ``input_feature`` is exposed rather than hidden because the surrounding loop
    legitimately reads and annotates it (assistant metadata, withdrawing tool
    schemas for a follow-up stage). What it must not do is *replace* the token
    arrays in it; use :meth:`adopt` for a feature that came back from a step
    which rebuilt them.
    """

    def __init__(self, template: Template, *, label: str = 'trajectory', max_tokens: Optional[int] = None) -> None:
        """
        Args:
            template: encodes the opening turn and every bridge after it. One
                template for the whole episode: two would disagree about special
                tokens and the disagreement would land mid-trajectory.
            label: how this episode is named in error messages. The alignment
                failures raised here are found by reading them, so an index or
                task id is worth passing.
            max_tokens: the length past which the episode is over. None lets it
                run to whatever the template's own limit is.
        """
        self.template = template
        self.label = label
        self.max_tokens = max_tokens
        self._pif: Dict[str, Any] = {}
        self._logprobs: List[Any] = []
        self._turns = 0

    # ---------------------------------------------------------------- reading

    @property
    def input_feature(self) -> Dict[str, Any]:
        """The encoded episode so far: ``input_ids``, ``labels``, ``messages``, …"""
        return self._pif

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return list(self._pif.get('messages') or [])

    @property
    def logprobs(self) -> List[Any]:
        """One entry per policy-produced token, in order. Empty when unsampled."""
        return self._logprobs

    @property
    def turns(self) -> int:
        """Generations banked. Not turns *attempted*: a failed call is not here."""
        return self._turns

    def full(self) -> bool:
        """Has the episode reached :attr:`max_tokens`? Never true without one."""
        if self.max_tokens is None:
            return False
        return len(self._pif.get('input_ids') or []) >= self.max_tokens

    # ------------------------------------------------------- text operations
    #
    # The loop needs three things that only the template knows how to do: read a
    # tool call out of a reply's text, report markup that did not parse, and
    # decode a generation back to text. They are delegated here so the loop talks
    # to one mode-aware collaborator (the ledger) instead of reaching for
    # ``self.template`` directly -- :class:`MessageLedger` answers the same three
    # trivially, because it has no template. The implementations stay in the
    # template; nothing is moved, this is a forwarding surface.

    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        """Structured tool calls the model emitted in ``decoded`` (template markup)."""
        return self.template.parse_tool_call(decoded)

    def tool_call_errors(self, decoded: str) -> List[str]:
        """Malformed tool-call markup errors in ``decoded`` (template-specific)."""
        return self.template.tool_call_errors(decoded)

    def decode_tokens(self, seq: SampledSequence) -> str:
        """Text of a generation, decoded from its tokens without the special tokens.

        Falls back to ``seq.decoded`` when there is no tokenizer or no tokens, so
        the loop's ``content`` rewrite works for a reply that carries text only.
        """
        tokenizer = getattr(self.template, 'tokenizer', None)
        if tokenizer is None or not seq.tokens:
            return seq.decoded or ''
        return tokenizer.decode(seq.tokens, skip_special_tokens=True)

    # ---------------------------------------------------------------- writing

    def open(self, trajectory: Trajectory, *, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        """Encode the opening messages. The one encode of the episode.

        Everything after this extends the ids this produced; nothing re-encodes
        them. ``tools`` overrides what the trajectory carries, for a caller whose
        executing tool list comes from somewhere else than its prompt (an Env
        that reported its own schemas, say).
        """
        pif = _to_plain(self.template.encode(trajectory, add_generation_prompt=True))
        # The template is not obliged to echo these back, and every consumer
        # downstream reads the episode off the feature rather than off the
        # trajectory it came from.
        pif.setdefault('messages', list(trajectory.get('messages') or []))
        if tools is not None:
            pif['tools'] = list(tools)
        elif 'tools' in trajectory:
            pif['tools'] = list(trajectory.get('tools') or [])
        self._pif = pif
        self._logprobs = []
        self._turns = 0

    def record(self, seq: SampledSequence) -> None:
        """Bank one generation, taking its ids from the sampler.

        ``new_input_feature`` is the prompt plus what was just sampled, already
        labelled, as the sampler saw it -- the whole reason a multi-turn episode
        can be trained on. A sampler that does not return one cannot be used for
        this, and saying so here is better than the alternative: silently
        re-encoding the reply and training on ids that drift from the sampled
        ones.
        """
        if seq.new_input_feature is None or 'input_ids' not in seq.new_input_feature:
            raise RuntimeError(f'sampler returned a SampledSequence without '
                               f'new_input_feature.input_ids for {self.label}; '
                               f'cannot continue multi-turn.')
        self._pif = _to_plain(dict(seq.new_input_feature))
        self._turns += 1
        self._extend_logprobs(seq)

    def _extend_logprobs(self, seq: SampledSequence) -> None:
        """Append one generation's logprobs -- one per sampled token, or raise.

        The check that keeps an account honest: a logprob array slipped by a
        token trains against the wrong positions and nothing else would notice.
        ``None`` means the run was unsampled, which is allowed.
        """
        if seq.logprobs is None:
            return
        if len(seq.logprobs) != len(seq.tokens):
            raise RuntimeError(f'logprobs length ({len(seq.logprobs)}) does not match '
                               f'sampled token count ({len(seq.tokens)}) at turn '
                               f'{self._turns} ({self.label})')
        self._logprobs.extend(seq.logprobs)

    def observe(self, messages: Sequence[Dict[str, Any]]) -> bool:
        """Append messages the model did not write: tool results, a new question.

        Their tokens are masked out of the loss -- they are the environment's
        words, and training on them teaches the model to predict its own
        observations. False means the append did not fit.
        """
        extended = extend_with_bridge(self._pif, list(messages), self.template)
        if extended is None:
            return False
        self._pif = extended
        return True

    def adopt(self, input_feature: Dict[str, Any]) -> None:
        """Take an already-encoded feature as the current state.

        For the one caller that legitimately rebuilds it: an opening rewritten
        before the first generation, which has to be re-encoded because there is
        no append that expresses it. Called after a generation instead, this is
        how a run silently starts training on drifted ids.
        """
        self._pif = _to_plain(dict(input_feature))

    def graft(self,
              prompt_token_ids: Sequence[int],
              seq: SampledSequence,
              *,
              messages: Optional[Sequence[Dict[str, Any]]] = None) -> bool:
        """Bank one round of an episode this loop did not drive.

        An agent running as its own program resends the whole conversation on
        every call, as text, and never sends token ids back. It does not have to:
        the prompt it resends is encoded on the way in to be sampled at all, and
        *those* ids are the real ones -- what the model ran on, not a
        reconstruction. So the account is kept on this side, one call at a time,
        and each call is checked against it: ``prompt_token_ids`` must begin with
        every id already banked.

        What the prompt adds beyond that is whatever happened out there between
        the two calls -- a tool result, a file the agent read, a question it asked
        itself -- and it is appended masked, because none of it is the policy's
        writing. ``seq.tokens`` is, and is appended trainable.

        False, and nothing changed, when the prompt does not extend what is
        banked. The agent compacted or rewrote its history, which no append can
        express. The only honest response is to keep the episode as it stands and
        open a fresh ledger for what follows: splicing the new prompt onto the old
        ids would produce a trajectory that never existed.

        ``messages`` is recorded for whoever reads the episode afterwards -- a
        trace, a reward function -- and has no bearing on the ids.
        """
        banked = list(self._pif.get('input_ids') or [])
        prompt = list(prompt_token_ids)
        if len(prompt) < len(banked) or prompt[:len(banked)] != banked:
            return False
        pif: Optional[Dict[str, Any]] = self._pif
        observed = prompt[len(banked):]
        if observed:
            pif = append_ids(pif, observed, self.template, trainable=False)
            if pif is None:
                return False
        if not seq.tokens:
            raise RuntimeError(f'the endpoint returned an empty continuation for {self.label}; '
                               'there is nothing to train on and nothing to append.')
        pif = append_ids(pif, list(seq.tokens), self.template, trainable=True)
        if pif is None:
            return False
        if messages is not None:
            pif['messages'] = list(messages)
        self._pif = pif
        self._turns += 1
        self._extend_logprobs(seq)
        return True

    # ---------------------------------------------------------------- closing

    def merge(self, trajectory: Trajectory, **fields: Any) -> Trajectory:
        """The trajectory plus the account, with the account checked first.

        Token fields land at top level, which is what tells a sampler downstream
        that this is already encoded and must not be encoded again.
        """
        self.audit()
        out = dict(trajectory)
        out.update(self._pif)
        out['messages'] = list(self._pif.get('messages') or trajectory.get('messages') or [])
        out['logprobs'] = self._logprobs if self._logprobs else None
        out.update(fields)
        return out

    def audit(self) -> None:
        """Raise unless there is exactly one logprob per trainable token.

        The one check that catches a drifted account, and the reason it is worth
        raising over: a trajectory whose logprobs have slipped by one position
        trains perfectly well against the wrong tokens. Nothing downstream can
        notice, because both arrays are the length they are supposed to be.

        Skipped when nothing was sampled -- an episode that never got a reply has
        no logprobs to align, and demanding them would turn an empty run into a
        crash.
        """
        if not self._logprobs:
            return
        labels = self._pif.get('labels') or []
        completion_mask = self._pif.get('completion_mask')
        if completion_mask is None:
            expected = sum(1 for label in labels if label != -100)
        elif len(completion_mask) != len(labels):
            raise RuntimeError(f'completion_mask/labels misaligned for {self.label}: '
                               f'{len(completion_mask)} != {len(labels)}')
        else:
            expected = sum(1 for label, flag in zip(labels, completion_mask) if label != -100 and flag)
        if len(self._logprobs) != expected:
            raise RuntimeError(f'logprobs/policy-token alignment failed for {self.label}: '
                               f'{len(self._logprobs)} logprobs vs {expected} positions selected '
                               'by (labels != -100) & completion_mask.')


class MessageLedger:
    """The message-level counterpart of :class:`TurnLedger`: an episode account
    with no tokens in it.

    Some backends produce text only -- a remote OpenAI-compatible teacher, say,
    returns a finished reply and no token ids, and there is no local tokenizer to
    encode with. Such a backend cannot feed :class:`TurnLedger`, whose
    :meth:`record` requires ``new_input_feature.input_ids`` and whose whole reason
    for existing is that a trajectory is trainable down to the sampled id. It can
    still drive a multi-turn conversation, and the loop that does so is identical
    down to the bookkeeping -- so this keeps the same call surface (``open`` /
    ``record`` / ``observe`` / ``merge`` / ``input_feature`` / ``turns`` / ``full``
    plus the three text operations) and accounts in messages instead of tokens.

    What it deliberately does not do: it never encodes; ``merge`` emits no
    ``input_ids`` / ``labels`` / ``logprobs``; ``audit`` is a no-op (there is no
    logprob-to-token alignment to check); ``full`` is always False (there is no
    token budget to hit). A trajectory it produces is a transcript, not a
    trainable sample. Tool calls cannot be parsed from text without a template,
    so they must arrive structurally: the backend puts them on the assistant
    message it returns (through ``new_input_feature['messages']``), which the loop
    already reads before falling back to text parsing.

    Duck-typed against :class:`TurnLedger`, not subclassed: the two share a call
    surface, not state -- one holds an encoded feature, the other a message list
    -- and a common base would be thinner than the disagreement between them.
    """

    def __init__(self, *, label: str = 'trajectory', max_tokens: Optional[int] = None) -> None:
        self.label = label
        # Taken for signature parity with TurnLedger and ignored: there are no
        # tokens to budget at the message level.
        self.max_tokens = max_tokens
        # One live dict, handed out by ``input_feature`` and annotated in place by
        # the loop (it withdraws tool schemas with ``['tools'] = []`` and rewrites
        # the last message). ``messages`` is the very list the writer methods
        # append to, so those annotations stay visible.
        self._feature: Dict[str, Any] = {'messages': [], 'tools': []}
        self._turns = 0

    # ---------------------------------------------------------------- reading

    @property
    def input_feature(self) -> Dict[str, Any]:
        """The episode so far as a live ``{'messages', 'tools'}`` dict.

        Live rather than a copy because the loop annotates it in place. There are
        no token arrays in it to protect, which is the one freedom message-level
        accounting has over :class:`TurnLedger`.
        """
        return self._feature

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return list(self._feature['messages'])

    @property
    def logprobs(self) -> List[Any]:
        """Always empty: a text-only backend reports no logprobs."""
        return []

    @property
    def turns(self) -> int:
        """Replies banked."""
        return self._turns

    def full(self) -> bool:
        """Never full: there is no token budget at the message level."""
        return False

    # ------------------------------------------------------- text operations

    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        """No template, so no text parser: tool calls arrive structurally only."""
        return []

    def tool_call_errors(self, decoded: str) -> List[str]:
        """No markup to validate without a template."""
        return []

    def decode_tokens(self, seq: SampledSequence) -> str:
        """No tokenizer: the reply's text is all there is."""
        return seq.decoded or ''

    # ---------------------------------------------------------------- writing

    def open(self, trajectory: Trajectory, *, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        """Take the opening messages as they are -- nothing is encoded."""
        self._feature = {
            'messages': list(trajectory.get('messages') or []),
            'tools': list(tools if tools is not None else (trajectory.get('tools') or [])),
        }
        self._turns = 0

    def record(self, seq: SampledSequence) -> None:
        """Bank one reply as an assistant message.

        ``new_input_feature`` is optional here, unlike :class:`TurnLedger`. When
        the backend supplies it as ``{'messages': [...]}`` -- the conversation plus
        the assistant turn it just produced, carrying any structured ``tool_calls``
        -- those messages are adopted wholesale, which is how a text-only backend
        gets tool calls into the episode. Otherwise the reply is appended as a
        plain assistant message built from ``seq.decoded``.
        """
        feature = seq.new_input_feature
        if isinstance(feature, dict) and feature.get('messages'):
            self._feature['messages'] = list(feature['messages'])
            if 'tools' in feature:
                self._feature['tools'] = list(feature['tools'])
        else:
            self._feature['messages'].append({'role': 'assistant', 'content': seq.decoded or ''})
        self._turns += 1

    def observe(self, messages: Sequence[Dict[str, Any]]) -> bool:
        """Append messages the model did not write. Always fits: no token budget."""
        self._feature['messages'].extend(dict(m) for m in messages)
        return True

    # ---------------------------------------------------------------- closing

    def merge(self, trajectory: Trajectory, **fields: Any) -> Trajectory:
        """The trajectory plus the transcript. No token fields, and no audit.

        ``audit`` is a no-op because there is no logprob-to-token alignment to
        check, and ``input_ids`` / ``labels`` / ``logprobs`` are absent so a
        downstream consumer sees a transcript rather than an encoded feature.
        """
        out = dict(trajectory)
        out['messages'] = list(self._feature['messages'])
        out.update(fields)
        return out

    def audit(self) -> None:
        """Nothing to align at the message level."""
        return


class LedgerBook:
    """Ledgers filed under a key, for episodes whose turns arrive unannounced.

    A ledger belongs to an episode, but an externally driven episode has no loop
    here to hold one: rounds arrive one HTTP request at a time, interleaved with
    every other episode in the batch, and the only thing tying a request to an
    episode is the key it came in under. This keeps the accounts and does the
    filing.

    A key can end up with more than one ledger, and that is the interesting part.
    An agent is free to compact or rewrite its own history -- summarise the first
    twenty turns into a paragraph, drop a file it no longer needs -- and when it
    does, the next prompt is not an extension of what is banked. There is no
    append that expresses it and no honest way to splice the two. So the ledger in
    hand is left exactly as it is, complete up to the last round that did fit, and
    a fresh one takes over from the rewritten history. One episode becomes two
    trajectories, which is what actually happened.

    Thread-safe, because the requests are: one lock per key so that concurrent
    episodes do not wait on each other, and the key registry guarded separately
    so two first-requests cannot each create an account.
    """

    def __init__(self, template: Template, *, max_tokens: Optional[int] = None) -> None:
        """
        Args:
            template: one template for every account here, for the reason
                :class:`TurnLedger` gives: two would disagree about special tokens.
            max_tokens: passed to each ledger as its length limit.
        """
        self.template = template
        self.max_tokens = max_tokens
        self._filed: Dict[str, List[TurnLedger]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def __contains__(self, key: str) -> bool:
        with self._guard:
            return key in self._filed

    def bank(self,
             key: str,
             prompt_token_ids: Sequence[int],
             seq: SampledSequence,
             *,
             messages: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[TurnLedger]:
        """Record one round against ``key``, opening or splitting as needed.

        A key seen for the first time gets an empty account, and the whole prompt
        the agent sent -- its system prompt, its tool descriptions, the task -- is
        appended masked. That is correct rather than convenient: none of it is the
        policy's writing, and we did not compose it.

        Returns the ledger the round landed in, or None when it landed nowhere:
        the sequence no longer fits the template's length limit. The caller should
        let the agent carry on -- it has its own reasons to stop, and killing its
        request over our bookkeeping teaches it nothing -- while knowing that what
        follows is not being recorded.
        """
        ledgers, lock = self._file(key)
        with lock:
            current = ledgers[-1]
            if current.graft(prompt_token_ids, seq, messages=messages):
                return current
            fresh = self._ledger(key, len(ledgers))
            if not fresh.graft(prompt_token_ids, seq, messages=messages):
                return None
            ledgers.append(fresh)
            return fresh

    def close(self, key: str) -> List[TurnLedger]:
        """Take the accounts for ``key`` away, in the order they were opened.

        Removed, not just read: the key is done, and a request arriving under it
        afterwards is a new episode that reused a name, not a continuation of one
        already handed to the trainer. Empty list for a key that never banked
        anything -- an agent that failed to make a single call.
        """
        with self._guard:
            self._locks.pop(key, None)
            return self._filed.pop(key, [])

    def _file(self, key: str) -> tuple:
        with self._guard:
            if key not in self._filed:
                self._filed[key] = [self._ledger(key, 0)]
                self._locks[key] = threading.Lock()
            return self._filed[key], self._locks[key]

    def _ledger(self, key: str, part: int) -> TurnLedger:
        label = key if part == 0 else f'{key}#{part}'
        return TurnLedger(self.template, label=label, max_tokens=self.max_tokens)


__all__ = ['LedgerBook', 'MessageLedger', 'TurnLedger']
