# Copyright (c) ModelScope Contributors. All rights reserved.
"""Deterministic (LLM-free) hard scorer for agentic trajectories.

Where :class:`RubricVerifier` judges *soft* quality via an LLM, this scorer
judges *hard* facts with plain code: did the agent call declared tools, were
the arguments valid, did the calls execute, is the OpenAI tool protocol
consistent, did the run terminate cleanly, is there a final answer, and did it
avoid degenerate repetition. These signals are **free** and, crucially,
**un-hackable by the policy** — the policy cannot talk its way past a JSON
parse error or a hallucinated tool name.

The score is a weighted mean of independent checks, each producing a
``CheckResult`` in ``[0, 1]``. Two aggregation modes:

- ``mode='mean'`` (default): weighted average of all checks.
- ``mode='gate'``: any *critical* check that scores 0 caps the whole score at
  0 (a strict gatekeeper — one hallucinated tool call fails the segment).

Checks are pluggable: pass your own callables to extend/override. The public
``__call__`` returns an ``int`` in ``[0, NUM_LEVELS)`` per the
:class:`Verifier` contract; ``score_detail`` returns the full breakdown.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import Verifier

# A check takes the parsed trajectory view and returns a CheckResult.
CheckFn = Callable[['TrajectoryView'], 'CheckResult']

_ERROR_PREFIX_RE = re.compile(r'^\s*(error|exception|traceback|failed)\b[:\s]', re.IGNORECASE)


@dataclass
class CheckResult:
    name: str
    score: float                 # in [0, 1]
    weight: float
    critical: bool               # if True and score==0, gate mode caps total at 0
    n: int = 0                   # number of items this check evaluated
    detail: str = ''

    def __post_init__(self):
        self.score = min(1.0, max(0.0, float(self.score)))


@dataclass
class HardScoreDetail:
    level: int
    scalar: float                # continuous hard score in [0, 1]
    gated: bool                  # a critical check zeroed the score (gate mode)
    checks: List[CheckResult] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level,
            'scalar': self.scalar,
            'gated': self.gated,
            'checks': {c.name: {'score': c.score, 'weight': c.weight,
                                'critical': c.critical, 'n': c.n, 'detail': c.detail}
                       for c in self.checks},
        }


@dataclass
class _ToolCall:
    name: Optional[str]
    raw_args: Any
    call_id: Optional[str]
    msg_index: int


class TrajectoryView:
    """Parsed, check-friendly view over a trajectory segment.

    Precomputes the message list, the assistant tool_calls, the tool-result
    messages and the declared tool schema so individual checks stay cheap and
    don't each re-walk the messages.
    """

    def __init__(self, trajectory: dict):
        self.trajectory = trajectory or {}
        self.messages: List[dict] = list(self.trajectory.get('messages', []) or [])
        self.tools: List[dict] = list(self.trajectory.get('tools', []) or [])

        # declared tool names + parameter schemas
        self.declared_names: set = set()
        self.declared_required: Dict[str, List[str]] = {}
        for t in self.tools:
            fn = t.get('function') if isinstance(t, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get('name')
            if not isinstance(name, str) or not name:
                continue
            self.declared_names.add(name)
            params = fn.get('parameters')
            if isinstance(params, dict):
                req = params.get('required')
                if isinstance(req, list):
                    self.declared_required[name] = [r for r in req if isinstance(r, str)]

        # assistant tool calls, in order
        self.tool_calls: List[_ToolCall] = []
        for i, m in enumerate(self.messages):
            if m.get('role') != 'assistant':
                continue
            for tc in (m.get('tool_calls') or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get('function') or {}
                self.tool_calls.append(_ToolCall(
                    name=fn.get('name') if isinstance(fn, dict) else None,
                    raw_args=fn.get('arguments') if isinstance(fn, dict) else None,
                    call_id=tc.get('id'),
                    msg_index=i,
                ))

        # tool-result messages, indexed by tool_call_id where present
        self.tool_msgs_by_id: Dict[str, dict] = {}
        self.tool_msgs: List[dict] = []
        for m in self.messages:
            if m.get('role') == 'tool':
                self.tool_msgs.append(m)
                cid = m.get('tool_call_id')
                if isinstance(cid, str) and cid:
                    self.tool_msgs_by_id[cid] = m

    # -- shared helpers reused by checks --
    def parsed_args(self, tc: _ToolCall) -> Optional[dict]:
        raw = tc.raw_args
        if isinstance(raw, dict):
            return raw
        if raw is None:
            return {}
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return {}
            try:
                v = json.loads(s)
                return v if isinstance(v, dict) else None
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def result_for(self, tc: _ToolCall) -> Optional[dict]:
        """Find the tool result for a call: prefer id match, else next tool msg."""
        if tc.call_id and tc.call_id in self.tool_msgs_by_id:
            return self.tool_msgs_by_id[tc.call_id]
        # positional fallback: first tool message after the call's assistant msg
        for j in range(tc.msg_index + 1, len(self.messages)):
            m = self.messages[j]
            role = m.get('role')
            if role == 'tool':
                return m
            if role == 'assistant':
                break
        return None

    @staticmethod
    def text_of(msg: Optional[dict]) -> str:
        if not msg:
            return ''
        content = msg.get('content')
        if isinstance(content, list):
            return '\n'.join(p.get('text', '') for p in content
                             if isinstance(p, dict) and p.get('type') == 'text')
        return content if isinstance(content, str) else ''

    def last_assistant_text(self) -> str:
        for m in reversed(self.messages):
            if m.get('role') == 'assistant':
                return self.text_of(m)
        return ''

    def last_assistant_msg(self) -> Optional[dict]:
        for m in reversed(self.messages):
            if m.get('role') == 'assistant':
                return m
        return None


# ---------------------------------------------------------------------------
# Individual checks (pure, deterministic)
# ---------------------------------------------------------------------------
def check_args_valid_json(view: TrajectoryView) -> CheckResult:
    """Every tool call's arguments must parse as a JSON object."""
    calls = view.tool_calls
    if not calls:
        return CheckResult('args_valid_json', 1.0, 1.0, critical=True, n=0,
                           detail='no tool calls')
    ok = sum(1 for tc in calls if view.parsed_args(tc) is not None)
    return CheckResult('args_valid_json', ok / len(calls), 1.0, critical=True,
                       n=len(calls), detail=f'{ok}/{len(calls)} valid')


def check_tool_declared(view: TrajectoryView) -> CheckResult:
    """Called tools must be in the declared tool set (no hallucinated tools)."""
    calls = view.tool_calls
    if not calls or not view.declared_names:
        # can't verify without a declared schema -> neutral pass, non-critical
        return CheckResult('tool_declared', 1.0, 1.0, critical=False, n=0,
                           detail='no tools declared or no calls')
    ok = sum(1 for tc in calls if tc.name in view.declared_names)
    return CheckResult('tool_declared', ok / len(calls), 1.5, critical=True,
                       n=len(calls), detail=f'{ok}/{len(calls)} declared')


def check_required_args(view: TrajectoryView) -> CheckResult:
    """Parsed arguments must contain the schema's required fields."""
    calls = [tc for tc in view.tool_calls if tc.name in view.declared_required]
    if not calls:
        return CheckResult('required_args', 1.0, 1.0, critical=False, n=0,
                           detail='no schema-required fields to check')
    ok = 0
    for tc in calls:
        args = view.parsed_args(tc)
        if args is None:
            continue
        req = view.declared_required.get(tc.name, [])
        if all(r in args for r in req):
            ok += 1
    return CheckResult('required_args', ok / len(calls), 1.0, critical=False,
                       n=len(calls), detail=f'{ok}/{len(calls)} complete')


def check_tool_executed(view: TrajectoryView) -> CheckResult:
    """Each tool call must have a non-empty, non-error result message."""
    calls = view.tool_calls
    if not calls:
        return CheckResult('tool_executed', 1.0, 1.0, critical=False, n=0,
                           detail='no tool calls')
    ok = 0
    for tc in calls:
        res = view.result_for(tc)
        text = view.text_of(res).strip()
        if text and not _ERROR_PREFIX_RE.match(text):
            ok += 1
    return CheckResult('tool_executed', ok / len(calls), 2.0, critical=False,
                       n=len(calls), detail=f'{ok}/{len(calls)} succeeded')


def check_protocol_pairing(view: TrajectoryView) -> CheckResult:
    """OpenAI protocol: every tool_call id should have a matching tool msg, and
    every tool msg should reference a known call id (when ids are used)."""
    calls = view.tool_calls
    if not calls:
        return CheckResult('protocol_pairing', 1.0, 1.0, critical=False, n=0,
                           detail='no tool calls')
    call_ids = {tc.call_id for tc in calls if tc.call_id}
    if not call_ids:
        # ids not used in this trace; fall back to counting result coverage
        paired = sum(1 for tc in calls if view.result_for(tc) is not None)
        return CheckResult('protocol_pairing', paired / len(calls), 1.0,
                           critical=False, n=len(calls),
                           detail=f'{paired}/{len(calls)} have a result (no ids)')
    matched_calls = sum(1 for cid in call_ids if cid in view.tool_msgs_by_id)
    # orphan tool messages referencing unknown ids
    orphans = sum(1 for m in view.tool_msgs
                  if isinstance(m.get('tool_call_id'), str)
                  and m['tool_call_id'] not in call_ids)
    total = len(call_ids) + orphans
    score = matched_calls / total if total else 1.0
    return CheckResult('protocol_pairing', score, 1.0, critical=False,
                       n=len(call_ids),
                       detail=f'{matched_calls}/{len(call_ids)} paired, {orphans} orphan tool msgs')


def check_no_repeated_calls(view: TrajectoryView) -> CheckResult:
    """Penalize degenerate tool-call loops.

    Two independent signals, worst one wins:
    1. exact-duplicate ``(name, args)`` calls — classic redundant repetition;
    2. single-tool domination — one tool name fired over and over (even with
       *different* args), the "spin the same tool forever" failure that (1)
       misses because the arguments differ each time. Only kicks in once there
       are enough calls (``>= _SPIN_MIN_CALLS``) so a legitimate 3-4 step loop
       of the same tool is not punished.
    """
    calls = view.tool_calls
    if len(calls) < 2:
        return CheckResult('no_repeated_calls', 1.0, 1.0, critical=False,
                           n=len(calls), detail='fewer than 2 calls')

    seen: set = set()
    dupes = 0
    name_counts: Dict[str, int] = {}
    for tc in calls:
        args = view.parsed_args(tc)
        key = (tc.name, json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(tc.raw_args))
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
        name_counts[tc.name or ''] = name_counts.get(tc.name or '', 0) + 1
    dup_score = 1.0 - dupes / len(calls)

    # Single-tool domination: one tool fired over and over (a spin loop). This
    # is only a *soft* signal — a legitimate agent may batch-read 8 files with
    # the same tool — so it is deliberately lenient: it only triggers on long
    # sequences that are almost entirely one tool, and it floors the penalty so
    # a batch operation is nudged down, not zeroed. Real dead-loops (empty
    # repeated spins) get further penalized by the rubric / final-answer checks.
    _SPIN_MIN_CALLS = 8
    _SPIN_TOLERATED_SHARE = 0.8
    _SPIN_FLOOR = 0.4
    spin_score = 1.0
    top_name, top_n = max(name_counts.items(), key=lambda kv: kv[1])
    top_share = top_n / len(calls)
    if len(calls) >= _SPIN_MIN_CALLS and top_share > _SPIN_TOLERATED_SHARE:
        # Linearly map (tolerated..1.0] share onto (1.0.._SPIN_FLOOR] score.
        frac = (top_share - _SPIN_TOLERATED_SHARE) / (1.0 - _SPIN_TOLERATED_SHARE)
        spin_score = max(_SPIN_FLOOR, 1.0 - (1.0 - _SPIN_FLOOR) * frac)

    score = min(dup_score, spin_score)
    detail = f'{dupes} duplicate calls'
    if spin_score < dup_score:
        detail = (f"tool '{top_name}' dominates {top_n}/{len(calls)} "
                  f'calls ({top_share:.0%})')
    return CheckResult('no_repeated_calls', score, 1.0, critical=False,
                       n=len(calls), detail=detail)


def check_clean_termination(view: TrajectoryView) -> CheckResult:
    """The trajectory should end on an assistant answer, not a dangling tool
    call or a length-truncated turn."""
    last = view.last_assistant_msg()
    if last is None:
        return CheckResult('clean_termination', 0.0, 1.0, critical=False, n=1,
                           detail='no assistant message')
    # last message overall should be the assistant answer (no trailing tool call
    # left unanswered / no pending tool msg after it)
    last_role = view.messages[-1].get('role') if view.messages else None
    finish = last.get('finish_reason')
    truncated = finish == 'length'
    dangling = last_role == 'assistant' and bool(last.get('tool_calls'))
    ok = (not truncated) and (not dangling) and (last_role in ('assistant', 'tool'))
    detail = []
    if truncated:
        detail.append('length-truncated')
    if dangling:
        detail.append('dangling tool_call')
    return CheckResult('clean_termination', 1.0 if ok else 0.0, 1.0,
                       critical=False, n=1, detail=', '.join(detail) or 'clean')


def check_final_answer(view: TrajectoryView) -> CheckResult:
    """There must be a non-empty final assistant answer."""
    text = view.last_assistant_text().strip()
    return CheckResult('final_answer', 1.0 if text else 0.0, 1.5,
                       critical=False, n=1,
                       detail=f'{len(text)} chars' if text else 'empty')


DEFAULT_CHECKS: Tuple[CheckFn, ...] = (
    check_args_valid_json,
    check_tool_declared,
    check_required_args,
    check_tool_executed,
    check_protocol_pairing,
    check_no_repeated_calls,
    check_clean_termination,
    check_final_answer,
)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
class HardScorer(Verifier):
    """Deterministic hard score for an agentic trajectory segment.

    Args:
        checks: Ordered check callables. Defaults to :data:`DEFAULT_CHECKS`.
            Pass your own to extend or replace.
        mode: ``'mean'`` (weighted average) or ``'gate'`` (a critical check
            scoring 0 zeroes the total).
        weights: Optional ``{check_name: weight}`` overrides.
    """

    def __init__(
        self,
        checks: Optional[List[CheckFn]] = None,
        *,
        mode: str = 'mean',
        weights: Optional[Dict[str, float]] = None,
        gate_threshold: float = 1.0,
    ):
        if mode not in ('mean', 'gate'):
            raise ValueError("mode must be 'mean' or 'gate'")
        if not 0.0 <= gate_threshold <= 1.0:
            raise ValueError('gate_threshold must be in [0, 1]')
        self.checks: List[CheckFn] = list(checks) if checks is not None else list(DEFAULT_CHECKS)
        self.mode = mode
        self.weights = dict(weights or {})
        # In gate mode, a critical check scoring BELOW this threshold zeroes the
        # total. 1.0 = any violation gates (strict); 0.0 = only total failure.
        self.gate_threshold = float(gate_threshold)

    def __call__(self, trajectory: dict, **kwargs) -> int:
        return self.score_detail(trajectory, **kwargs).level

    def score_detail(self, trajectory: dict, **kwargs) -> HardScoreDetail:
        view = TrajectoryView(trajectory)
        results: List[CheckResult] = []
        for fn in self.checks:
            r = fn(view)
            if r.name in self.weights:
                r.weight = float(self.weights[r.name])
            results.append(r)

        gated = False
        if self.mode == 'gate':
            # A critical check that is not fully satisfied gates the segment:
            # a single hallucinated tool or JSON parse error fails the whole
            # thing, regardless of how many other calls were fine.
            for r in results:
                if r.critical and r.n > 0 and r.score < self.gate_threshold:
                    gated = True
                    break

        if gated:
            scalar = 0.0
        else:
            num = sum(r.score * r.weight for r in results)
            den = sum(r.weight for r in results)
            scalar = num / den if den else 0.0

        return HardScoreDetail(
            level=self._to_level(scalar),
            scalar=scalar,
            gated=gated,
            checks=results,
        )

    def _to_level(self, scalar: float) -> int:
        scalar = min(1.0, max(0.0, scalar))
        level = int(round(scalar * (self.NUM_LEVELS - 1)))
        return min(self.NUM_LEVELS - 1, max(0, level))
