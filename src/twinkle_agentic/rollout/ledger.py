# Copyright (c) ModelScope Contributors. All rights reserved.
"""The token account of one episode, separated from the policy that grows it.

``MultiTurnRollout`` is two things braided together. One is policy --
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
and reads ``new_input_feature`` off it rather than encoding anything, and why an
episode driven from outside appends only what extends the recorded conversation
(:meth:`observe`) and starts a fresh account when it does not, rather than
re-encoding to make it fit.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from twinkle.data_format import Trajectory, copy_trajectory
from twinkle.data_format import last_assistant_text as _traj_last_assistant_text
from twinkle.data_format import turn_token_spans, validate_trajectory
from twinkle.data_format.trajectory import _trainable_positions
from twinkle.data_format.sampling import SampledSequence, SamplingMask, sampled_token_logprobs
from twinkle.template.base import Template
from twinkle.template.contract import coerce_tool_call_arguments, engine_input_ids
from twinkle_agentic.utils.token_utils import append_ids



def last_assistant_text(messages: list[dict] | None) -> str:
    """The newest assistant turn's text, for logging / rule-based reward.

    Delegates to the shared trajectory helper; multimodal content (a list of parts) is
    flattened to its text parts rather than read as a plain string.
    """
    return _traj_last_assistant_text({"messages": messages})


class _TrajectoryLedger:
    """Shared ownership and message operations for token and message accounting."""

    @property
    def input_feature(self) -> Trajectory:
        return self._trajectory

    @property
    def messages(self) -> list[dict[str, Any]]:
        return copy_trajectory({'messages': self._trajectory.get('messages') or []})['messages']

    @property
    def tools(self) -> list[dict[str, Any]]:
        return copy_trajectory({'tools': self._trajectory.get('tools') or []})['tools']

    def replace_last_message(self, message: dict[str, Any]) -> None:
        if self._trajectory.get("messages"):
            self._trajectory["messages"][-1] = copy_trajectory(message)

    def clear_tools(self) -> None:
        self._trajectory["tools"] = []

    def _sampled_trajectory(self, seq: SampledSequence) -> Trajectory:
        feature = seq.new_input_feature
        if feature is None or 'messages' not in feature:
            raise RuntimeError(f"sampler must return a complete trajectory for {self.label}")
        sampled = copy_trajectory(feature)
        previous = self.messages
        messages = sampled.get('messages') or []
        if (len(messages) != len(previous) + 1 or messages[-1].get('role') != 'assistant'
                or messages[:-1] != previous):
            raise RuntimeError(f"sampler must preserve the history and append one assistant message for {self.label}")
        for key in ('tools', 'user_data'):
            if (sampled.get(key) or []) != (self._trajectory.get(key) or []):
                raise RuntimeError(f"sampler changed the recorded {key} for {self.label}")
        return sampled

    def _continue_messages(self, previous, sampled) -> None:
        if not previous or previous[-1].get("role") != "assistant":
            raise RuntimeError(f"continuation without an open assistant turn at {self.label}")
        incoming = sampled.get("messages") or []
        if len(incoming) != len(previous) + 1 or incoming[-1].get("role") != "assistant":
            raise RuntimeError(f"continuation must append one assistant message at {self.label}")
        last, delta = dict(previous[-1]), incoming[-1]
        for key in ("content", "reasoning_content"):
            if key in delta:
                before, after = last.get(key) or "", delta.get(key) or ""
                if isinstance(before, list) or isinstance(after, list):
                    before = before if isinstance(before, list) else ([{"type": "text", "text": before}] if before else [])
                    after = after if isinstance(after, list) else ([{"type": "text", "text": after}] if after else [])
                last[key] = before + after
        if delta.get("tool_calls"):
            last["tool_calls"] = list(last.get("tool_calls") or []) + list(delta["tool_calls"])
        last.update({key: value for key, value in delta.items()
                     if key not in ('role', 'content', 'reasoning_content', 'tool_calls')})
        sampled["messages"] = previous[:-1] + [last]


class TurnLedger(_TrajectoryLedger):
    """Token ids, labels and logprobs for one episode, and nothing else.

    Not thread-safe and not meant to be: one ledger belongs to one episode, and
    an episode takes one step at a time.

    Usage is one :meth:`open`, then :meth:`record` after each generation and
    :meth:`observe` for each thing appended that the model did not write, then
    :meth:`merge` to read the episode out. Both appending methods answer False
    when the append did not fit, which is the caller's cue to stop -- the ledger
    is left holding the last state that did fit, so a trajectory that overflowed
    is still complete up to the turn before.

    ``input_feature`` is the complete current trajectory.
    """

    def __init__(
        self, template: Template, *, label: str = "trajectory", max_tokens: int | None = None, backend: str = "vllm"
    ) -> None:
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
        self.backend = backend
        self._trajectory: Trajectory = {}
        self._logprobs: list[Any] = []
        self._sampling_masks: Any = None
        self._turns = 0

    # ---------------------------------------------------------------- reading



    @property
    def logprobs(self) -> list[Any]:
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
        return len(self._trajectory.get("input_ids") or []) >= self.max_tokens

    def unfit(self, params: Any) -> bool:
        """Would this turn's completion budget overrun the template's ``max_length``?

        The sampler caps a TURN at its own ``max_tokens``; nothing caps the EPISODE at
        ``max_length`` before an append runs out of room. Checked before each turn so a
        budget that no longer fits ends the episode as truncated -- with the turns it
        already banked -- instead of dying inside the append. A dict or object
        ``params`` both work; one without ``max_tokens`` (an API default) is not
        checkable and answers False.
        """
        max_length = getattr(self.template, "max_length", None)
        turn_budget = params.get("max_tokens") if isinstance(params, dict) else getattr(params, "max_tokens", None)
        if max_length is None or turn_budget is None:
            return False
        return len(self._trajectory.get("input_ids") or []) + int(turn_budget) > max_length

    def capped_turn_params(self, params: Any) -> Any | None:
        """Cap the turn's completion budget to the room left under ``max_length``.

        :meth:`unfit` answers whether the FULL turn budget overruns ``max_length``. When it
        does on the FIRST turn (nothing banked yet), ending the episode there emits a
        zero-response sample: no labels, so nothing trainable, and it breaks the driver-side
        length statistic and the worker forward (under ray the sample's ``encoded`` is an
        ObjectRef the driver must not dereference). Capping the budget to the remaining room
        yields a truncated partial response instead -- the standard "``max_length`` truncates
        the generation" semantics. Returns the capped params, or ``None`` when not even one
        token fits (the prompt alone is at/over ``max_length``: a degenerate sample with
        nothing trainable, which the caller ends truncated rather than forcing a generation).
        """
        max_length = getattr(self.template, "max_length", None)
        turn_budget = params.get("max_tokens") if isinstance(params, dict) else getattr(params, "max_tokens", None)
        if max_length is None or turn_budget is None:
            return None
        room = int(max_length) - len(self._trajectory.get("input_ids") or [])
        if room <= 0:
            return None
        capped = min(int(turn_budget), room)
        if isinstance(params, dict):
            return {**params, "max_tokens": capped}
        return replace(params, max_tokens=capped)

    # ------------------------------------------------------- text operations
    #
    # The loop needs three things that only the template knows how to do: read a
    # tool call out of a reply's text, report markup that did not parse, and
    # decode a generation back to text. They are delegated here so the loop talks
    # to one mode-aware collaborator (the ledger) instead of reaching for
    # ``self.template`` directly -- :class:`MessageLedger` answers the same three
    # trivially, because it has no template. The implementations stay in the
    # template; nothing is moved, this is a forwarding surface.

    def parse_tool_call(self, decoded: str) -> list[dict[str, Any]]:
        """Structured tool calls, with arguments restored from this episode's tool schemas."""
        calls = self.template.parse_tool_call(decoded)
        tools = self._trajectory.get("tools")
        return list(coerce_tool_call_arguments(calls, tools))

    def tool_call_errors(self, decoded: str) -> list[str]:
        """Malformed tool-call markup errors in ``decoded`` (template-specific)."""
        return self.template.tool_call_errors(decoded)

    def decode_tokens(self, seq: SampledSequence) -> str:
        """Text of a generation, decoded from its tokens without the special tokens.

        Falls back to ``seq.decoded`` when there is no tokenizer or no tokens, so
        the loop's ``content`` rewrite works for a reply that carries text only.
        """
        tokenizer = getattr(self.template, "tokenizer", None)
        if tokenizer is None or not seq.tokens:
            return seq.decoded or ""
        return tokenizer.decode(seq.tokens, skip_special_tokens=True)

    # ---------------------------------------------------------------- writing

    def open(self, trajectory: Trajectory, *, tools: list[dict[str, Any]] | None = None) -> None:
        """Encode an unencoded opening once, or adopt its existing token history."""
        feature = copy_trajectory(trajectory)
        if tools is not None:
            feature["tools"] = copy_trajectory({"tools": tools})["tools"]
        if "input_ids" not in feature:
            feature = self.template.encode_prompt(feature, backend=self.backend, add_generation_prompt=True)
        engine_input_ids(feature)
        self.adopt(feature)
        self._logprobs = []
        self._sampling_masks = None
        self._turns = 0

    def record(self, seq: SampledSequence) -> None:
        """Adopt the sampler's complete trajectory without re-encoding its reply."""
        self._trajectory = self._sampled_trajectory(seq)
        self._turns += 1
        self._extend_logprobs(seq)
        self._extend_sampling_masks(seq)

    def _sampled_trajectory(self, seq: SampledSequence) -> Trajectory:
        sampled = super()._sampled_trajectory(seq)
        if 'input_ids' not in sampled:
            raise RuntimeError(f"sampler must return a complete encoded trajectory for {self.label}")
        for key in ("input_ids", "labels", "completion_mask", "engine_input_ids"):
            if key in sampled and hasattr(sampled[key], "tolist"):
                sampled[key] = sampled[key].tolist()
        expected = list(self._trajectory.get("input_ids", [])) + list(seq.tokens)
        if list(sampled["input_ids"]) != expected:
            raise RuntimeError(f"sampler changed the recorded token prefix for {self.label}")
        if engine_input_ids(sampled) != engine_input_ids(self._trajectory) + list(seq.tokens):
            raise RuntimeError(f"sampler changed the recorded engine token prefix for {self.label}")
        for key, fill in (('labels', -100), ('completion_mask', 0)):
            values = sampled.get(key)
            if values is None or len(values) != len(expected) or (values and values[-1] != fill):
                raise RuntimeError(f"sampler returned misaligned {key} for {self.label}")
            before = self._trajectory.get(key)
            width = max(len(expected) - len(seq.tokens) - 1, 0)
            if before is not None and list(values[:width]) != list(before[:width]):
                raise RuntimeError(f"sampler changed the recorded {key} for {self.label}")
        return sampled

    def record_continuation(self, seq: SampledSequence) -> None:
        """Continue the existing assistant turn while retaining sampled token positions."""
        previous = self.messages
        sampled = self._sampled_trajectory(seq)
        self._continue_messages(previous, sampled)
        self._trajectory = sampled
        self._extend_logprobs(seq)
        self._extend_sampling_masks(seq)

    def _extend_sampling_masks(self, seq: SampledSequence) -> None:
        """Keep CSR support sets aligned across sampled turns and continuations."""
        width = len(_trainable_positions(self._trajectory['input_ids'], self._trajectory.get('labels'),
                                        self._trajectory.get('completion_mask')))
        previous = self._sampling_masks
        previous_width = len(previous.offsets) - 1 if previous is not None else 0
        mask = seq.sampling_masks
        if mask is None:
            if width != previous_width:
                self._sampling_masks = None
            return
        if len(mask.offsets) != len(seq.tokens) + 1:
            raise RuntimeError(f"sampling masks do not align with the generated tokens for {self.label}")
        if width != previous_width + len(seq.tokens):
            self._sampling_masks = None
            return
        tokens = list(previous.token_ids) if previous is not None else []
        offsets = list(previous.offsets) if previous is not None else [0]
        base = len(tokens)
        tokens.extend(mask.token_ids)
        offsets.extend(base + offset for offset in mask.offsets[1:])
        self._sampling_masks = SamplingMask(token_ids=tokens, offsets=offsets)

    def _extend_logprobs(self, seq: SampledSequence) -> None:
        """Append one generation's logprobs -- one per sampled token, or raise.

        The check that keeps an account honest: a logprob array slipped by a
        token trains against the wrong positions and nothing else would notice.
        ``None`` means the run was unsampled, which is allowed.
        """
        if seq.logprobs is None:
            return
        if len(seq.logprobs) != len(seq.tokens):
            raise RuntimeError(
                f"logprobs length ({len(seq.logprobs)}) does not match "
                f"sampled token count ({len(seq.tokens)}) at turn "
                f"{self._turns} ({self.label})"
            )
        self._logprobs.extend(seq.logprobs)

    def observe(self, messages: Sequence[dict[str, Any]]) -> bool:
        """Append environment messages with separate training and engine bridges."""
        previous = self.messages
        kwargs = {"gen_prompt_before": False, "gen_prompt_after": True, "tools": self.tools}
        bridge = self.template.render_delta(previous, list(messages), **kwargs)
        engine_bridge = self.template.render_delta(previous, list(messages), engine=True,
                                                    backend=self.backend, **kwargs)
        extended = append_ids(self._trajectory, bridge, self.template, trainable=False,
                              engine_ids=engine_bridge)
        if extended is None:
            return False
        extended["messages"] = previous + copy_trajectory({"messages": list(messages)})["messages"]
        self._trajectory = extended
        return True



    def adopt(self, input_feature: Trajectory) -> None:
        """Adopt an encoded opening; sampled history must only be extended incrementally."""
        self._trajectory = copy_trajectory(input_feature)
        for key in ("input_ids", "labels", "completion_mask", "engine_input_ids"):
            if key in self._trajectory and hasattr(self._trajectory[key], "tolist"):
                self._trajectory[key] = self._trajectory[key].tolist()

    # ---------------------------------------------------------------- closing

    def merge(self, trajectory: Trajectory, **fields: Any) -> Trajectory:
        """The trajectory plus the account, with the account checked first.

        Token fields land at top level, which is what tells a sampler downstream
        that this is already encoded and must not be encoded again. The TRAINING
        VIEW comes with it: ``response_token_ids`` (the policy-produced tokens,
        input order), ``response_positions`` (the prediction positions those
        tokens are scored at -- ``labels[position]`` IS the token, the shifted-labels
        contract ``validate_trajectory`` asserts), ``rollout_logps`` (one scalar per
        response token -- the old_logps a consumer like GRPO needs), and
        ``replay_mask`` beside a routed ``routed_experts`` -- so no caller has to
        re-derive them from the labels.
        """
        self.audit()
        out = {**copy_trajectory(trajectory), **copy_trajectory(self._trajectory)}
        out["messages"] = self.messages or copy_trajectory({'messages': trajectory.get('messages') or []})['messages']
        out["tools"] = self.tools
        out["logprobs"] = copy_trajectory({'logprobs': self._logprobs})['logprobs'] if self._logprobs else None
        labels = self._trajectory.get("labels") or []
        input_ids = self._trajectory.get("input_ids") or []
        completion_mask = self._trajectory.get("completion_mask")
        positions = _trainable_positions(input_ids, labels, completion_mask)
        if positions and positions[0] == 0:
            # A trainable input-order position 0 means the very first token of the
            # input is policy-produced: there is no prompt. Its label lives nowhere
            # under the shifted convention (``labels[-1]`` is the last position's
            # target), so refusing beats guessing an index that scores nothing.
            raise RuntimeError(
                f"the first input token of {self.label} is policy-produced; an episode with no "
                "prompt has no response position to score."
            )
        out["response_token_ids"] = [input_ids[j] for j in positions]
        out["response_positions"] = [j - 1 for j in positions]
        if self._logprobs:
            # An engine that returns no logprobs leaves the field ABSENT, not empty: ``rollout_logps``
            # is response-framed, so an empty list reads as "this response has zero tokens" and
            # ``validate_trajectory`` rejects it, while an absent field is the documented signal that
            # this trajectory simply carries no sampling logps.
            out["rollout_logps"] = sampled_token_logprobs(out["response_token_ids"], self._logprobs)
        if "routed_experts" in self._trajectory:
            out["replay_mask"] = (
                list(completion_mask) if completion_mask is not None else [int(label != -100) for label in labels]
            )
        out.pop('sampling_masks', None)
        if self._sampling_masks is not None:
            out["sampling_masks"] = SamplingMask(token_ids=list(self._sampling_masks.token_ids),
                                                  offsets=list(self._sampling_masks.offsets))
        out.update(fields)
        # The response contract is asserted HERE, at the trajectory's producer: the driver's
        # hand-out merge stays row-level work, and a bad encoding fails this episode's step
        # instead of a training batch far downstream.
        validate_trajectory(out)
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
        labels = self._trajectory.get("labels") or []
        completion_mask = self._trajectory.get("completion_mask")
        if completion_mask is None:
            expected = sum(1 for label in labels if label != -100)
        elif len(completion_mask) != len(labels):
            raise RuntimeError(
                f"completion_mask/labels misaligned for {self.label}: {len(completion_mask)} != {len(labels)}"
            )
        else:
            expected = sum(1 for label, flag in zip(labels, completion_mask, strict=False) if label != -100 and flag)
        if len(self._logprobs) != expected:
            raise RuntimeError(
                f"logprobs/policy-token alignment failed for {self.label}: "
                f"{len(self._logprobs)} logprobs vs {expected} positions selected "
                "by (labels != -100) & completion_mask."
            )


class MessageLedger(_TrajectoryLedger):
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

    Shares trajectory ownership and message operations with :class:`TurnLedger`;
    token budgets and logprob accounting remain exclusive to the encoded path.
    """

    def __init__(
        self, *, label: str = "trajectory", max_tokens: int | None = None, template: Template | None = None
    ) -> None:
        self.label = label
        # Taken for signature parity with TurnLedger and ignored: there are no
        # tokens to budget at the message level.
        self.max_tokens = max_tokens
        # An optional template used ONLY for its text operations (parse_tool_call / tool_call_errors /
        # decode_tokens), so a message-level episode that nonetheless has a local template -- the TUI's
        # streaming tool turn -- can read tool-call markup out of a reply's text. Accounting stays at the
        # message level either way: this never encodes, never banks token ids, and ``merge`` still emits
        # no ``input_ids`` / ``labels`` / ``logprobs``. Without a template (a text-only remote teacher)
        # the three text operations fall back to their structural-only behaviour.
        self._template = template
        # One live dict, handed out by ``input_feature`` and annotated in place by
        # the loop (it withdraws tool schemas with ``['tools'] = []`` and rewrites
        # the last message). ``messages`` is the very list the writer methods
        # append to, so those annotations stay visible.
        self._trajectory: dict[str, Any] = {"messages": [], "tools": []}
        self._turns = 0

    # ---------------------------------------------------------------- reading



    @property
    def logprobs(self) -> list[Any]:
        """Always empty: a text-only backend reports no logprobs."""
        return []

    @property
    def turns(self) -> int:
        """Replies banked."""
        return self._turns

    def full(self) -> bool:
        """Never full: there is no token budget at the message level."""
        return False

    def unfit(self, params: Any) -> bool:
        """Never unfit: a message-level episode has no token ids to budget against."""
        return False

    # ------------------------------------------------------- text operations

    def parse_tool_call(self, decoded: str) -> list[dict[str, Any]]:
        """Tool calls in ``decoded``: parsed via the template when one is present, else structural-only.

        A message-level episode with a local template (the TUI streaming tool turn) can read markup out
        of the reply text; a text-only backend has no parser, so its tool calls must arrive structurally
        on the assistant message (which the loop reads before falling back to this).
        """
        if self._template is not None:
            return self._template.parse_tool_call(decoded)
        return []

    def tool_call_errors(self, decoded: str) -> list[str]:
        """Malformed tool-call markup errors in ``decoded`` (template-specific); empty without a template."""
        if self._template is not None:
            return self._template.tool_call_errors(decoded)
        return []

    def decode_tokens(self, seq: SampledSequence) -> str:
        """Text of a reply: decoded from its tokens when a template/tokenizer is present, else ``seq.decoded``."""
        tokenizer = getattr(self._template, "tokenizer", None) if self._template is not None else None
        if tokenizer is None or not seq.tokens:
            return seq.decoded or ""
        return tokenizer.decode(seq.tokens, skip_special_tokens=True)

    # ---------------------------------------------------------------- writing

    def open(self, trajectory: Trajectory, *, tools: list[dict[str, Any]] | None = None) -> None:
        """Own the complete opening without tokenizing it."""
        self._trajectory = copy_trajectory(trajectory)
        self._trajectory.setdefault("messages", [])
        self._trajectory["tools"] = copy_trajectory({"tools": tools if tools is not None else trajectory.get("tools", [])})["tools"]
        self._turns = 0

    def record(self, seq: SampledSequence) -> None:
        """Adopt a structured reply, or append a text-only streaming result."""
        if seq.new_input_feature is not None:
            self._trajectory = self._sampled_trajectory(seq)
        elif not seq.tokens:
            self._trajectory["messages"].append({"role": "assistant", "content": seq.decoded or ""})
        else:
            raise RuntimeError(f"sampler omitted its trajectory for {self.label}")
        self._turns += 1

    def record_continuation(self, seq: SampledSequence) -> None:
        """Extend the same assistant message, including structured continuation fields."""
        previous = self.messages
        if seq.new_input_feature is not None:
            sampled = self._sampled_trajectory(seq)
        elif not seq.tokens:
            sampled = copy_trajectory(self._trajectory)
            sampled["messages"] = previous + [{"role": "assistant", "content": seq.decoded or ""}]
        else:
            raise RuntimeError(f"sampler omitted its trajectory for {self.label}")
        self._continue_messages(previous, sampled)
        self._trajectory = sampled

    def observe(self, messages: Sequence[dict[str, Any]]) -> bool:
        """Append messages the model did not write. Always fits: no token budget."""
        self._trajectory["messages"].extend(copy_trajectory({'messages': list(messages)})['messages'])
        return True



    # ---------------------------------------------------------------- closing

    def merge(self, trajectory: Trajectory, **fields: Any) -> Trajectory:
        """The trajectory plus the transcript. No token fields, and no audit.

        ``audit`` is a no-op because there is no logprob-to-token alignment to
        check, and ``input_ids`` / ``labels`` / ``logprobs`` are absent so a
        downstream consumer sees a transcript rather than an encoded feature.
        """
        out = {**copy_trajectory(trajectory), **copy_trajectory(self._trajectory)}
        out["messages"] = self.messages
        out["tools"] = self.tools
        out.update(fields)
        return out

    def audit(self) -> None:
        """Nothing to align at the message level."""
        return


__all__ = ['MessageLedger', 'TurnLedger', 'turn_token_spans']
