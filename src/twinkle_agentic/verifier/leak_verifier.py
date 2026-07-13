# Copyright (c) ModelScope Contributors. All rights reserved.
"""General answer-leak verifier for auxiliary hints / skills / notes.

Decides whether a piece of *auxiliary content* — a hint, a distilled skill, a
retrieved note, a rationale — that will be shown to a solver ALONGSIDE a task
leaks the answer (or an essentially complete solution) to THAT task. A good hint
carries only transferable strategy ("factor the radicand before adding roots");
a leaking hint hands over the result, a task-specific decisive step, or a full
derivation.

**One layer only: an LLM judge.** Leak detection is inherently *semantic* — the
common case is a hint that describes the solution structure without ever writing
the answer, which no string rule can catch. A cheap deterministic pre-filter
(verbatim answer / answer-number matching) was measured to replace only a few
percent of the judge's catches, is not domain-general, and false-flags short
answers ("D", "1") that appear in almost any text. So it is deliberately absent:
this verifier is a single ``llm_backup``-distilled LEAK / CLEAN judge that runs
on the student model when confident and falls back to the teacher API otherwise
— the same progressive-distillation path the other verifiers here use.

The :class:`Verifier` contract (``__call__ -> int``) maps CLEAN -> ``NUM_LEVELS-1``
(a clean hint is the "good" end of the reward scale) and LEAK -> ``0``.
:meth:`leak_detail` exposes the boolean + reason, and :meth:`leak_batch` runs
many checks in parallel (the judge dominates latency).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

from twinkle_agentic.utils.llm_backup import llm_backup

from .base import Verifier
from .domain_checks import _ground_truths

if TYPE_CHECKING:
    from twinkle.data_format.sampling import SamplingParams  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401


# ---------------------------------------------------------------------------
# Prompts (task-agnostic; a reference answer is optional context for the judge)
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM = """\
You check whether a HINT that will be shown to someone solving a TASK leaks the \
answer.

A good hint gives ONLY transferable strategy — general methods, common \
pitfalls, or sanity checks that would help on a whole class of similar tasks.

The hint LEAKS if it does ANY of:
- reveals the final answer or final result (a value, expression, choice, label, \
or verbatim output);
- states a specific decisive intermediate result or fact that is unique to THIS \
task;
- gives a derivation or step-by-step plan that essentially solves THIS task.

The hint does NOT leak if it only names general methods, common mistakes, or \
checks that are not specific to this task's answer.

Reply with exactly one word: LEAK or CLEAN."""

# Answer-only variant: flags ONLY final-answer disclosure and explicitly PERMITS
# revealing the method / approach / step plan. For skill-generation training, the
# method IS the product we want, so the default prompt's "decisive intermediate
# result" and "step-by-step plan that solves this task" clauses over-flag good
# skills; this variant drops them.
_JUDGE_SYSTEM_ANSWER_ONLY = """\
You check whether a HINT that will be shown to someone solving a TASK reveals the 
TASK's FINAL ANSWER.

The hint may FREELY describe general methods, the solution approach, which 
technique to apply, the steps to take, common pitfalls, or sanity checks — even 
if that strongly points at how to solve THIS task. Revealing the METHOD or PLAN 
is allowed and expected of a good hint.

The hint LEAKS only if it reveals the FINAL ANSWER itself — the concrete final 
value, expression, choice, label, or verbatim result the task asks for (or a 
trivially equivalent restatement of it). Describing HOW to get there WITHOUT 
stating the resulting final value does NOT leak.

Reply with exactly one word: LEAK or CLEAN."""

_JUDGE_USER = """\
## Task
{query}
{reference_block}
## Hint to check
{hint}

Does the hint leak the answer or an essentially complete solution to THIS task? \
Reply LEAK or CLEAN."""

_REFERENCE_BLOCK = """
## Known answer (for your judgement only; do not treat its wording as the hint)
{reference}
"""


# ---------------------------------------------------------------------------
# Result holder
# ---------------------------------------------------------------------------
@dataclass
class LeakDetail:
    """Outcome of one leak check.

    Attributes:
        leaked: True if the content leaks the answer/solution.
        reason: machine-readable reason — ``''`` (clean), ``'llm_leak'``,
            ``'llm_uncertain'`` (judge reply unparseable), or ``'no_llm'`` (no
            student sampler and no teacher API configured).
        source: which layer decided — ``'llm'`` | ``'none'``.
    """
    leaked: bool
    reason: str
    source: str


# ---------------------------------------------------------------------------
# Verdict parsing / comparator (for the judge + its distillation)
# ---------------------------------------------------------------------------
def _verdict_of(raw: str) -> Optional[bool]:
    """Parse a LEAK/CLEAN reply -> True (leak) / False (clean) / None (unclear)."""
    v = (raw or '').strip().upper()
    if 'CLEAN' in v:
        return False
    if 'LEAK' in v:
        return True
    return None


def _verdict_close(a: str, b: str) -> bool:
    """llm_backup comparator: student/teacher agree iff same LEAK/CLEAN verdict."""
    va, vb = _verdict_of(a), _verdict_of(b)
    if va is None or vb is None:
        return (a or '').strip() == (b or '').strip()
    return va == vb


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
class LeakVerifier(Verifier):
    """Verify that an auxiliary hint does not leak the answer to its task.

    Args:
        sampler: student model sampler (local inference). If ``None`` the judge
            is served entirely by the teacher API via ``llm_backup`` (useful
            before a student exists); if no teacher is configured either, the
            verifier reports ``no_llm`` and cannot judge.
        model_path: identifier (bookkeeping only).
        sampling_params: default sampling params for the judge call.
        judge_lora_path: LoRA adapter for the distilled judge student.
        max_content_chars / max_query_chars: truncation caps for the judge input.
        uncertain_is_leak: if the judge reply is unparseable, treat it as a leak
            (default False: keep the hint, the conservative choice).
        judge_system: optional custom judge system prompt that overrides the built-in
            answer_only/legacy criteria (for task-specific leak policies).
    """

    def __init__(
        self,
        sampler: Optional['Sampler'] = None,
        *,
        model_path: str = '',
        sampling_params: Optional['SamplingParams'] = None,
        judge_lora_path: Optional[str] = None,
        max_content_chars: int = 4000,
        max_query_chars: int = 4000,
        uncertain_is_leak: bool = False,
        answer_only: bool = False,
        judge_system: Optional[str] = None,
    ):
        self.sampler = sampler
        self.model_path = model_path
        self.sampling_params = sampling_params
        self.judge_lora_path = judge_lora_path or None
        self.max_content_chars = int(max_content_chars)
        self.max_query_chars = int(max_query_chars)
        self.uncertain_is_leak = bool(uncertain_is_leak)
        # answer_only: flag ONLY final-answer disclosure, permit method/plan (see
        # _JUDGE_SYSTEM_ANSWER_ONLY). Default keeps the strict legacy behaviour.
        self.answer_only = bool(answer_only)
        # judge_system: caller-supplied criterion that overrides both built-ins, for
        # task-specific leak policies (e.g. permit method/plan but still flag concrete
        # intermediate key results). None -> fall back to the answer_only/legacy pair.
        self.judge_system = judge_system or None

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def __call__(self, trajectory: dict, *, query: Optional[str] = None,
                 reference: Optional[Union[str, Sequence[str]]] = None,
                 **kwargs) -> int:
        detail = self.leak_detail(
            self._content_of(trajectory),
            query=query or self._infer_query(trajectory),
            reference=reference if reference is not None else _ground_truths(trajectory),
        )
        return 0 if detail.leaked else self.NUM_LEVELS - 1

    def is_leak(self, content: str, *, query: str,
                reference: Optional[Union[str, Sequence[str]]] = None) -> bool:
        return self.leak_detail(content, query=query, reference=reference).leaked

    def leak_detail(self, content: str, *, query: str,
                    reference: Optional[Union[str, Sequence[str]]] = None
                    ) -> LeakDetail:
        """LLM judge; ``no_llm`` when neither a student nor a teacher exists."""
        content = content or ''
        references = self._as_list(reference)

        if not self._llm_available():
            return LeakDetail(False, 'no_llm', 'none')

        verdict = self._judge(content, query or '', references)
        if verdict is True:
            return LeakDetail(True, 'llm_leak', 'llm')
        if verdict is False:
            return LeakDetail(False, '', 'llm')
        return LeakDetail(self.uncertain_is_leak, 'llm_uncertain', 'llm')

    def leak_batch(self, items: Sequence[dict], *, max_workers: int = 8
                   ) -> List[LeakDetail]:
        """Check many hints in parallel; each item is ``{content, query, reference?}``.

        The judge (network / student inference) dominates latency, so the checks
        fan out over a thread pool. Results are returned in input order.
        """
        items = list(items)
        if not items:
            return []
        workers = max(1, min(max_workers, len(items)))
        if workers == 1:
            return [self._leak_detail_item(it) for it in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._leak_detail_item, items))

    def _leak_detail_item(self, item: dict) -> LeakDetail:
        return self.leak_detail(item.get('content', ''), query=item.get('query', ''),
                                reference=item.get('reference'))

    # ------------------------------------------------------------------
    # judge (student, distilled via llm_backup)
    # ------------------------------------------------------------------
    def _judge(self, content: str, query: str,
               references: Sequence[str]) -> Optional[bool]:
        trajectory = self._judge_trajectory(content, query, references)
        raw = self._judge_once(
            trajectory=trajectory,
            sampling_params=self._judge_sampling_params(self.sampling_params),
            judge_key='leak')
        return _verdict_of(raw)

    # Distilled on the LEAK/CLEAN verdict. A single shared key ('leak') tracks
    # student/teacher agreement on the judging skill as a whole; the comparator
    # matches on the verdict, not byte-identical text.
    @llm_backup(key_params=['judge_key'], comparator=_verdict_close)
    def _judge_once(self, trajectory, sampling_params, judge_key: str = 'leak') -> str:
        return self._sample_text(trajectory, sampling_params, self.judge_lora_path)

    def _judge_trajectory(self, content: str, query: str,
                          references: Sequence[str]) -> dict:
        ref_block = ''
        refs = [r for r in references if (r or '').strip()]
        if refs:
            ref_block = _REFERENCE_BLOCK.format(reference='; '.join(refs))
        user = _fill(
            _JUDGE_USER,
            query=self._trim(query, self.max_query_chars),
            reference_block=ref_block,
            hint=self._trim(content, self.max_content_chars))
        system = self.judge_system or (
            _JUDGE_SYSTEM_ANSWER_ONLY if self.answer_only else _JUDGE_SYSTEM)
        return {'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]}

    def _judge_sampling_params(self, override):
        if override is not None:
            return override
        from twinkle.data_format.sampling import SamplingParams
        # A one-word verdict; keep the budget tiny. Small headroom absorbs models
        # that prepend a stray token before LEAK/CLEAN.
        return SamplingParams(temperature=0.0, max_tokens=8)

    # ------------------------------------------------------------------
    # LLM plumbing (mirrors RubricVerifier)
    # ------------------------------------------------------------------
    def _llm_available(self) -> bool:
        if self.sampler is not None:
            return True
        return bool(os.environ.get('LLM_BACKUP_API_KEY')
                    or os.environ.get('OPENAI_API_KEY')
                    or os.environ.get('LLM_BACKUP_BASE_URL'))

    def _sample_text(self, trajectory, sampling_params, lora_path) -> str:
        if self.sampler is None:
            return ''
        sample_kwargs: dict[str, Any] = {'sampling_params': sampling_params}
        if lora_path is None:
            sample_kwargs['use_base_model'] = True
        else:
            sample_kwargs['adapter_path'] = lora_path
        responses = self.sampler.sample([trajectory], **sample_kwargs)
        resp = list(responses)[0] if responses else None
        if resp is None:
            return ''
        seqs = getattr(resp, 'sequences', None) or []
        return (getattr(seqs[0], 'decoded', None) or '') if seqs else ''

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_list(reference: Optional[Union[str, Sequence[str]]]) -> List[str]:
        if reference is None:
            return []
        if isinstance(reference, str):
            return [reference]
        return [str(r) for r in reference]

    @staticmethod
    def _trim(text: str, cap: int) -> str:
        text = text or ''
        return text if len(text) <= cap else text[:cap]

    @staticmethod
    def _content_of(trajectory: dict) -> str:
        """The hint under check = last assistant message text (else any content)."""
        if isinstance(trajectory, str):
            return trajectory
        msgs = trajectory.get('messages', []) or []
        for m in reversed(msgs):
            if m.get('role') == 'assistant':
                c = m.get('content')
                if isinstance(c, list):
                    c = '\n'.join(p.get('text', '') for p in c
                                  if isinstance(p, dict) and p.get('type') == 'text')
                if isinstance(c, str) and c.strip():
                    return c
        return str(trajectory.get('content', '') or '')

    @staticmethod
    def _infer_query(trajectory: dict) -> str:
        if isinstance(trajectory, str):
            return '(no explicit task)'
        for m in trajectory.get('messages', []) or []:
            if m.get('role') == 'user':
                c = m.get('content')
                if isinstance(c, str) and c.strip():
                    return c.strip()
        return '(no explicit task)'


def _fill(template: str, **kw) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace('{' + k + '}', str(v))
    return out
