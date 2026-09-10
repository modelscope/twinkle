# Copyright (c) ModelScope Contributors. All rights reserved.
"""The token account of one episode, separated from the policy that grows it.

``MultiTurnRollout._run_one`` is two things braided together. One is policy --
how many turns, what a malformed reply costs, when a follow-up stage is granted.
The other is bookkeeping: which token ids the trajectory now consists of, which
of them the policy produced, and the logprob for each of those. This file is the
second one, and only the second one.

They are split because the policy is not shared and the bookkeeping is. An agent
that ships as its own program drives its own loop (see ``harness/base.py``, "Who
drives"), so none of the turn accounting above applies to it -- but the account
below applies unchanged, because it is what makes a run trainable at all:

    the tokens trained on are the tokens the sampler returned

Not text re-encoded afterwards. A tokenizer is free to encode the same string to
different ids depending on what precedes it, so a trajectory rebuilt from its own
transcript can differ from what was sampled -- and every logprob then belongs to
a position that has moved. The gradient is still computed, against the wrong
tokens, and nothing raises. That is why :meth:`record` takes a ``SampledSequence``
and reads ``new_input_feature`` off it rather than encoding anything, and why
:meth:`graft` -- the entry point for a caller that supplies messages instead of
driving turns -- refuses a history that does not extend what is already banked
rather than re-encoding to make it fit.
"""
from typing import Any, Dict, List, Optional, Sequence

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampledSequence
from twinkle.template.base import Template
from .bridge import _to_plain, extend_with_bridge


class TurnLedger:
    """Token ids, labels and logprobs for one episode, and nothing else.

    Not thread-safe and not meant to be: one ledger belongs to one episode, the
    way a harness does.

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

    def __init__(self, template: Template, *, label: str = 'trajectory',
                 max_tokens: Optional[int] = None) -> None:
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
        if seq.logprobs is not None:
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

        For the one caller that legitimately rebuilds it: a harness that rewrote
        history before the first generation, which has to be re-encoded because
        there is no append that expresses it. Called after a generation instead,
        this is how a run silently starts training on drifted ids.
        """
        self._pif = _to_plain(dict(input_feature))

    def graft(self, messages: Sequence[Dict[str, Any]], *,
              tools: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Extend the account with a history that arrived whole, not turn by turn.

        For a driver that is not this loop -- an agent process that runs its own
        turns and hands over the conversation it has accumulated. The prefix it
        sends must be the prefix already banked, message for message; what is new
        is appended as observations, and appended *masked*, because a caller
        supplying messages is by definition not supplying sampled tokens. The
        model's own replies still enter through :meth:`record`, from the sampler
        that produced them.

        False, and nothing changed, when the history is not an extension of what
        is banked -- the caller edited or summarised earlier turns, and there is
        no way to represent that as an append. What to do about it is the caller's
        call, and there is only one honest option: bank the episode as it stands
        and start a fresh ledger. Stitching the new history onto the old ids
        produces a trajectory that never existed, which is the failure this whole
        file is arranged to prevent.
        """
        banked = self.messages
        supplied = list(messages)
        if len(supplied) < len(banked):
            return False
        for mine, theirs in zip(banked, supplied):
            if mine != theirs:
                return False
        appended = supplied[len(banked):]
        if not appended:
            if tools is not None:
                self._pif['tools'] = list(tools)
            return True
        if not banked:
            # Nothing to extend: this is the opening, and the tokens for it have
            # to come from an encode like any other opening.
            self.open({'messages': supplied}, tools=tools)
            return True
        if not self.observe(appended):
            return False
        if tools is not None:
            self._pif['tools'] = list(tools)
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

        Skipped when nothing was sampled -- an episode assembled by hand has no
        logprobs to align, and demanding them would fail the very case
        ``graft`` exists to serve.
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
            expected = sum(1 for label, flag in zip(labels, completion_mask)
                           if label != -100 and flag)
        if len(self._logprobs) != expected:
            raise RuntimeError(f'logprobs/policy-token alignment failed for {self.label}: '
                               f'{len(self._logprobs)} logprobs vs {expected} positions selected '
                               'by (labels != -100) & completion_mask.')


__all__ = ['TurnLedger']
