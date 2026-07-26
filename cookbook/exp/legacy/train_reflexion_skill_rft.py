"""RFT cold-start for the reflexion skill generator (see reflexion.md §6).

Trains an INDEPENDENT skill model to write reusable, transferable skills that,
when injected into a FROZEN base solver's system prompt, let the base solve problems
it first got wrong. The base is never trained — it only produces the reward signal.
Scoring is SEAM-style DETERMINISTIC: the base runs each candidate skill once at
temperature 0 (M=1), so the reward ``R in {0,1}`` (answer correct) carries no
sampling noise; the per-candidate advantage is group-relative within a problem
(``A = (R - mean) / (std + eps)``) and the skill model is updated online by GRPO —
problem-groups where every skill scores alike (std=0) contribute no gradient.

Direction: skill GENERATION + recall. Skill-gen always runs with thinking ON, and
each hard problem is routed to EXACTLY ONE of two views (no reuse — kills memory
leak and holds cost at 1x): view A ``(problem + attempt) -> think + skills`` keeps
the online generator self-bootstrapping; view B ``(problem only) -> think + skills``
is the deployment form, where the think is grounded on the query alone so it cannot
hallucinate an attempt. Both share the verified skill; the distilled ``<skills>``
block is recalled into the base's system prompt at solve time.

8-GPU layout (three DeviceGroups, one twinkle.initialize):
  - ranks 0-3  : ``train``         — skill model, full-param FSDP2, dp=4
  - ranks 4-5  : ``skill_sampler`` — skill model rollouts (vLLM, tp1 dp2)
  - ranks 6-7  : ``base_sampler``  — frozen base solver (vLLM, tp1 dp2)
CheckpointEngineManager syncs train -> skill_sampler after every optimizer step;
base_sampler is never synced.

Leak filtering uses ``LeakVerifier(sampler=None)`` via the backup teacher API
(no local judge, no distillation): set LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL /
LLM_BACKUP_MODEL.

Launch (8 GPUs):
    LLM_BACKUP_API_KEY=... LLM_BACKUP_BASE_URL=... LLM_BACKUP_MODEL=... \
    python cookbook/exp/embedding/train_reflexion_skill_rft.py --n 2000 --chunk-size 16
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle_agentic.verifier import LeakVerifier, RubricVerifier
from twinkle_agentic.verifier.rubric_verifier import RubricItem

# Reuse the reference eval's dataset + grading + prompts + sampling config, and the
# phase-0 pipeline's parsing / rollout / injection helpers (Find > Create).
from eval_gpqa_rag import (GEN_GPU_MEM, GEN_MODEL_ID, build_direct_prompt,  # noqa: F401
                           load_aops, load_math)
from eval_reflexion_skill import (_EX_PROBLEM, _clean_text,  # noqa: F401
                                  _parse_seq, _run_samples, build_skill_solve_prompt)

logger = get_logger()

try:
    import swanlab
except ImportError:  # optional; metric logging degrades to stdout + jsonl only
    swanlab = None


# -- GPU layout ---------------------------------------------------------------
TRAIN_GPUS = int(os.environ.get('TRAIN_GPUS', 4))
SKILL_SAMPLER_GPUS = int(os.environ.get('SKILL_SAMPLER_GPUS', 2))
BASE_SAMPLER_GPUS = int(os.environ.get('BASE_SAMPLER_GPUS', 2))
NUM_GPUS = TRAIN_GPUS + SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS
# FSDP shard group size within a dp replica; TRAIN_DP is the data-parallel axis that
# ``forward_backward`` (slice_dp) splits each mini-batch over.
TRAIN_FSDP = int(os.environ.get('TRAIN_FSDP', 2))
TRAIN_DP = max(1, TRAIN_GPUS // TRAIN_FSDP)


# ---------------------------------------------------------------------------
# Skill-generation prompt (DISTILL the useful approach, per the new direction)
# ---------------------------------------------------------------------------
# --- Previous STRICT view-A system prompt (commented out; kept for easy revert). It
# hard-required 3-5 bullets, "output nothing after </skills>", one-imperative-sentence
# items, no-narration, and a strict do-not-reveal block. The soft SEAM-style version
# below drops those four format demands, frames the skills as advisory reminders, and
# explains how they are used. ---
# SKILL_GEN_SYSTEM = (
#     'You are distilling reusable problem-solving SKILLS from one worked episode. '
#     'You are shown a competition problem, the guidance the solver was given, and the '
#     "solver's own attempt (its reasoning may be partly right and partly wrong).\n\n"
#     'FIRST, in your private thinking, do ALL of: (a) work out what this TYPE of problem '
#     'fundamentally requires; (b) pinpoint WHERE THIS attempt actually went wrong '
#     '(when a process-check report is provided below, use its flagged criteria as '
#     'evidence, but confirm each against the attempt yourself) — '
#     'the decisive misstep, a missing idea, a wrong turn, or the way it stalled, looped '
#     'on the same step, or ran the length budget out without ever committing to an '
#     'answer; and (c) imagine AS MANY DIFFERENT angles as you can — distinct approaches '
#     'or representations that could crack this problem, alternative solution paths, and '
#     'the various ways a solver could plausibly go wrong on it (a few words each, do NOT '
#     'develop them fully). THEN commit to the angle you find most decisive and write a '
#     'SHORT list of skills that would have PREVENTED that specific '
#     'failure and would raise the success rate of a SIMILAR solver on SIMILAR problems. '
#     'Ground each skill in the concrete mistake you found, but state it as a GENERAL, '
#     'transferable rule — not a patch hard-coded to this problem. Across the 3-5 '
#     'bullets, prioritise in this order:\n'
#     '1. the decisive method or representation this class of problem calls for (what to '
#     'set up or reach for first);\n'
#     '2. the specific mistake that derailed THIS attempt, recast as a general pitfall, '
#     'plus the quick check that catches it;\n'
#     '3. a compact, ordered procedure that reliably drives toward a final answer;\n'
#     '4. convergence discipline: once the key quantity is in hand, commit to a single '
#     'concrete final answer in the required format instead of re-deriving, endless '
#     'case-splitting, looping on the same check, or overrunning the length budget.\n\n'
#     'OUTPUT FORMAT (strict):\n'
#     '- Keep the thinking COMPACT — a quick brainstorm of angles then a decision, not a '
#     'full solution and not a re-statement of these instructions. AFTER it, '
#     'output ONLY a markdown bullet list of 3-5 items WRAPPED IN <skills> and </skills> '
#     'tags — no preamble, no narration outside the tags. Output nothing after '
#     '</skills>.\n'
#     '- Each item is ONE short imperative sentence (a method, heuristic, check, or '
#     'habit).\n'
#     '- Inside the tags: no narration, no "The student...", no headings, no restating '
#     'the problem.\n\n'
#     'CONTENT RULES (strict):\n'
#     '- Do NOT reveal the final answer or the multiple-choice option.\n'
#     '- Do NOT state the specific numbers, values, or key intermediate results of THIS '
#     'problem.\n'
#     '- Every item must be GENERAL and transferable, not a step-by-step solution to '
#     'THIS problem.\n\n'
#     'Follow the example below for the exact tags, style, and level of generality.'
# )
SKILL_GEN_SYSTEM = (
    'You are a mathematics coach. You are shown a competition problem together with an '
    'automated process-check of an earlier solver attempt at it -- which solution '
    'criteria the attempt passed or failed, and suggested fixes for the failures. You '
    'do NOT see the attempt itself, only this check. Treat the check as privileged '
    'training scaffolding: study it together with the problem, identify the '
    'problem-visible features that make each useful flagged failure relevant, then '
    'rephrase those lessons as self-contained reusable skills. The goal is not to '
    'continue from the check, cite it, or hide it silently; the goal is to turn it to '
    'a skill pattern which prevents the model falls into similar pitfalls in the future.\n\n'
    'Good skills name the observable trigger, the method worth reaching for, the '
    'pitfall to watch, and a quick verification habit. Prefer formulations like '
    '"When a configuration has ...", "Before setting up ...", or "Check whether ..." '
    'over references to the process-check, failed criteria, or the earlier attempt. '
    "These tips are advisory: they will be placed in a solver's system prompt as gentle "
    'reminders before it works through a SIMILAR problem on its own, without seeing '
    'this process-check. So keep them general and transferable rather than a worked '
    'solution to this exact problem, and do not state its specific intermediate values '
    'or final answer. Think briefly first, then give your tips as a markdown bullet '
    'list wrapped in <skills> and </skills>, like the example below.'
)

# One-shot demo of the recommended mix (method / pitfall+check / procedure /
# convergence), answer-free — anchors both the format and the content priorities.
_EX_SKILLS = (
    '<skills>\n'
    '- Rewrite each square root by factoring its radicand into a perfect square times '
    'a remainder, then move the perfect-square factor outside.\n'
    '- Avoid the classic trap $\\sqrt{a}+\\sqrt{b}\\ne\\sqrt{a+b}$; only combine terms '
    'sharing the same simplest radical, and sanity-check by estimating each root.\n'
    '- Procedure: simplify every radical, group like radical terms, add their '
    'coefficients, then reduce to simplest form.\n'
    '- Once the expression is in simplest form, commit to that single result as the '
    'final answer rather than re-checking indefinitely.\n'
    '</skills>')


# View A user template: the problem + the automated rubric process-check of an earlier
# attempt (PASS/FAIL per criterion + suggested fixes). The attempt trajectory is NOT
# shown -- the rubric findings are the evidence the skill model grounds its tips on,
# which avoids feeding the (often long, non-terminating) attempt into the prompt.
SKILL_GEN_USER_RUBRIC = (
    'Problem:\n{problem}\n\n'
    'Process check of an earlier attempt (automated rubric verifier -- treat as '
    'evidence, not gospel; PASS/FAIL per criterion with suggested fixes for failures):\n'
    '{diagnosis}\n\n'
    'Now output a self-contained skills bullet list. Each bullet should still be useful '
    'if the process check were removed: connect any useful flagged failure to '
    'problem-visible features, general methods, and quick checks rather than citing the '
    'rubric or the earlier attempt. \n\n'
    'Note: **Do not solve the problem, only generate skills**. Now Begin:'
)


def build_skillgen_prompt(problem: str, diagnosis: str) -> Dict[str, Any]:
    """View A skill-gen prompt: system + one-shot format demo + the real episode
    (problem + the rubric process-check of an earlier attempt). The attempt trajectory
    is deliberately NOT shown -- the rubric findings localise the failure without the
    generator having to re-chew (and often re-solve) a long, possibly non-terminating
    attempt. The one-shot demo is query-only; only the real turn carries the diagnosis."""
    return {'messages': [
        {'role': 'system', 'content': SKILL_GEN_SYSTEM},
        # {'role': 'user', 'content': SKILL_GEN_USER_Q.format(problem=_EX_PROBLEM)},
        # {'role': 'assistant', 'content': _EX_SKILLS},
        {'role': 'user',
         'content': SKILL_GEN_USER_RUBRIC.format(problem=problem, diagnosis=diagnosis)},
    ]}


# ---------------------------------------------------------------------------
# View B: query-only skill-gen (deployment form). No attempt is shown — the model
# must reason about the problem TYPE from the query alone, so the think is grounded
# on the query and cannot narrate/fabricate an attempt. Format is deliberately
# distinct from view A so the model learns the two modes as separate contracts.
# ---------------------------------------------------------------------------
# --- Previous STRICT view-B (query-only) system prompt (commented out; kept for revert).
# Same four format demands as the old view A. Soft SEAM-style version below. ---
# SKILL_GEN_SYSTEM_Q = (
#     'You are distilling reusable problem-solving SKILLS for a CLASS of problems. You '
#     'are shown ONE competition problem and NOTHING else — no solution, no attempt. '
#     'FIRST, in your private thinking, imagine AS MANY DIFFERENT angles as you can — '
#     'distinct approaches or representations that could crack this TYPE of problem, '
#     'alternative solution paths, and the various ways a solver could plausibly go wrong '
#     'on it (a few words each, do NOT develop them fully). THEN commit to what you find '
#     'most decisive and write a SHORT list of skills that would raise a solver\'s success '
#     'rate on SIMILAR problems. Across the 3-5 bullets, prioritise in this order:\n'
#     '1. the decisive method or representation this class of problem calls for (what to '
#     'set up or reach for first);\n'
#     '2. the specific pitfall that derails such problems, plus the quick check that '
#     'catches it;\n'
#     '3. a compact, ordered procedure that reliably drives toward a final answer;\n'
#     '4. convergence discipline: once the key quantity is in hand, commit to a single '
#     'concrete final answer in the required format instead of re-deriving, endless '
#     'case-splitting, or overrunning the length budget.\n\n'
#     'OUTPUT FORMAT (strict):\n'
#     '- Keep the thinking COMPACT — a quick brainstorm of angles then a decision, not a '
#     'full solution and not a re-statement of these instructions. AFTER '
#     'it, output ONLY a markdown bullet list of 3-5 items WRAPPED IN <skills> and '
#     '</skills> tags — no preamble, no narration outside the tags. Output nothing after '
#     '</skills>.\n'
#     '- Each item is ONE short imperative sentence (a method, heuristic, check, or '
#     'habit).\n'
#     '- Inside the tags: no narration, no headings, no restating the problem, and no '
#     'reference to any attempt, student, or solution.\n\n'
#     'CONTENT RULES (strict):\n'
#     '- Do NOT solve THIS problem or reveal its final answer or multiple-choice option.\n'
#     '- Do NOT state the specific numbers, values, or key intermediate results of THIS '
#     'problem.\n'
#     '- Every item must be GENERAL and transferable to other problems of the same '
#     'type.\n\n'
#     'Follow the example below for the exact tags, style, and level of generality.'
# )
SKILL_GEN_SYSTEM_Q = (
    'You are a mathematics coach. You are shown ONE competition problem and nothing '
    'else — no solution and no attempt. Think about what approach this KIND of problem '
    'calls for and where solvers tend to slip, then distil a few reusable tips.\n\n'
    "These tips are advisory: they will be placed in a solver's system prompt as gentle "
    'reminders before it works through a SIMILAR problem on its own. So keep them '
    'general and transferable — the method worth reaching for, the pitfall to watch and '
    'a quick check, and the discipline to settle on a final answer — rather than a '
    'worked solution to this exact problem, and without stating its specific '
    'intermediate values or its final answer. Think briefly first, then give your tips '
    'as a markdown bullet list wrapped in <skills> and </skills>, like the example below.'
)

SKILL_GEN_USER_Q = (
    'Problem:\n{problem}\n\n'
    'Now reason about this TYPE of problem, then output the skills bullet list.'
)


def build_querygen_prompt(problem: str) -> Dict[str, Any]:
    """View B skill-gen prompt: system + one-shot demo + the problem ALONE (no
    attempt) — matching what is available at deployment (query only)."""
    return {'messages': [
        {'role': 'system', 'content': SKILL_GEN_SYSTEM_Q},
        # {'role': 'user', 'content': SKILL_GEN_USER_Q.format(problem=_EX_PROBLEM)},
        # {'role': 'assistant', 'content': _EX_SKILLS},
        {'role': 'user', 'content': SKILL_GEN_USER_Q.format(problem=problem)},
    ]}


def _assign_view(problem: str, args: argparse.Namespace) -> str:
    """Deterministically route a problem to exactly one view (stable across restarts
    and across the generation/SFT sides). ``--view-b-frac`` of problems go to view B."""
    h = int(hashlib.md5(f'{args.seed}:{problem}'.encode('utf-8')).hexdigest(), 16)
    return 'B' if (h % 100000) / 100000.0 < args.view_b_frac else 'A'


def _skillgen_messages(problem: str, view: str, diagnosis: str) -> List[Dict[str, Any]]:
    """Single source of truth for the skill-gen prompt, used at BOTH generation and
    training time so they can never diverge. View A with a localisable failure uses
    problem + rubric findings (NO trajectory); view B -- or a view-A problem whose rubric
    flagged NO failure (``[FAIL]`` absent: all-pass or missing diagnosis) -- is query-only.
    So view A DEGRADES to view B whenever there is nothing concrete to correct."""
    if view == 'B' or '[FAIL]' not in (diagnosis or ''):
        return build_querygen_prompt(problem)['messages']
    return build_skillgen_prompt(problem, diagnosis)['messages']


def _view_prompt(r: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """The skill-gen prompt for problem ``r`` under its assigned view (routing in
    ``_skillgen_messages``: view A carries the rubric process-check; view B, and any
    view-A problem with no rubric failure, is query-only)."""
    return {'messages': _skillgen_messages(r['problem'], r['_view'], r.get('_rubric_diag', ''))}


_BULLET_RE = re.compile(r'(?m)^\s*(?:[-*]|\d+[.)])\s')
# Trajectory/meta references that betray CoT fragments leaking into the block; any
# hit fails the purity gate (the problem is then re-sampled, per --skill-retries).
_META_RE = re.compile(
    r'\b(the student|the solver|the attempt|this attempt|the trace|the response|'
    r'in the (?:attempt|trace|response|solution)|as (?:shown|seen|noted) above|'
    r'the (?:above|previous|earlier)|my (?:reasoning|analysis)|i (?:think|need|will))\b',
    re.IGNORECASE)


def _is_clean_block(block: str) -> bool:
    """Purity gate for thinking-ON skill-gen: the block must be a pure bullet list
    (every non-empty line a bullet — no prose/CoT fragments) with no meta/trajectory
    reference. Answer leak is caught separately by the backup-teacher leak stage."""
    lines = [ln for ln in (l.strip() for l in block.splitlines()) if ln]
    if not lines or not all(_BULLET_RE.match(ln) for ln in lines):
        return False
    return _META_RE.search(block) is None


def _extract_skills_block(text: str) -> Optional[str]:
    """Return the clean ``<skills>...</skills>`` block, or None if not parseable.

    Skill-gen runs with thinking ON, so the model must end its reasoning with an explicit
    ``</think>`` before committing an answer (whether the opening ``<think>`` is emitted by
    the model or pre-injected by the chat template). We therefore REQUIRE ``</think>`` and
    read only the text after the last one; its absence means the token budget was exhausted
    mid-reasoning (nothing committed, per reflexion.md §6.8) — reject so a draft or a
    system-prompt demo echo inside the CoT can never be mistaken for the answer. Within
    the answer take the ``<skills>`` block (closing tag optional), strip stray tags, and
    require ``_is_clean_block`` — prose-mixed / meta-referencing fragments are rejected
    for re-sampling."""
    low = text.lower()
    end_think = low.rfind('</think>')
    if end_think < 0:
        return None  # reasoning never closed -> no committed answer
    answer = text[end_think + len('</think>'):]
    low_a = answer.lower()
    s = low_a.find('<skills>')
    if s < 0:
        return None
    inner = s + len('<skills>')
    e = low_a.find('</skills>', inner)
    block = (answer[inner:e] if e >= 0 else answer[inner:]).strip()
    block = re.sub(r'</?(?:skills|think)>', '', block, flags=re.IGNORECASE).strip()
    if not _is_clean_block(block):
        return None
    return block


# ---------------------------------------------------------------------------
# OPTIONAL stricter leak criterion (currently UNUSED -- the run uses answer_only=True,
# which flags ONLY the final answer). This variant ALSO flags concrete intermediate KEY
# results, while still permitting method / plan / pitfalls / checks. To enable, pass
# judge_system=_LEAK_JUDGE_SYSTEM to the LeakVerifier below.
# ---------------------------------------------------------------------------
_LEAK_JUDGE_SYSTEM = """\
You check whether a HINT that will be shown to someone solving a math TASK gives away
this task's own results.

The hint may FREELY describe the general method, which approach or technique to use, the
steps or plan to follow, common pitfalls, and sanity checks -- even when that points
strongly at HOW to solve THIS task. Describing the approach is expected of a good hint.

The hint LEAKS only if, for THIS specific task, it states either:
- the final answer or final result (a value, expression, choice, label, or verbatim
  output); or
- a concrete decisive INTERMEDIATE key result -- a specific computed value, quantity, or
  fact unique to this task that hands over a key step of the answer.

If it names only the method / plan / pitfalls / checks WITHOUT stating those concrete
intermediate values or the final result, it does NOT leak.

Reply with exactly one word: LEAK or CLEAN."""


# ---------------------------------------------------------------------------
# Rubric process-check (view A only): a frozen teacher diagnoses the base's failed
# attempt so the skill model grounds its error analysis on a verified fault
# localisation instead of guessing. Teacher-only (sampler=None -> every diagnose()
# hits llm_backup); mirrors eval_dualline_math's fixed math rubric.
# ---------------------------------------------------------------------------
_RFT_DIAG_SYSTEM = """\
You are a process error checker for a math solution attempt. You are given a math
problem, a rubric, and one attempted solution segment. Decide PASS or FAIL for each
criterion and explain only the process error type.

Output STRICT JSON (no prose outside it) with this shape:
{
  "items": [
    {"index": 1, "verdict": "PASS", "reason": "<why the process satisfies it>",
     "fix": ""},
    {"index": 2, "verdict": "FAIL", "reason": "<what process step is wrong>",
     "fix": "<method-level correction, without computing the corrected result>"}
  ],
  "overall": "OK" | "ISSUES",
  "summary": "<one sentence naming the process issue, not the answer>"
}

Rules:
- Judge every criterion independently and literally; a [Hard Rule] is FAIL unless
  unambiguously satisfied.
- Judge ONLY what is observable in THIS segment.
- Content inside <think>...</think> (or <thinking>) is internal reasoning, not
  user-facing output; ignore it for "output only X" style criteria.
- For PASS items, leave "fix" as "".
- For FAIL items, "reason", "fix", and "summary" must describe only the flawed
  step, theorem, arithmetic operation, case split, or verification habit.
- NEVER try to solve the query or state the correct final answer, corrected final expression, option letter,
  graph/choice label, or any exact value that the answer should become.
- NEVER write phrases like "the correct answer is", "which gives", "yielding",
  "should be <value>", "Option <letter>", or "Graph <letter>".
- If a fix would require naming a corrected value, replace it with a method-level
  instruction such as "redo that computation carefully" or "apply the theorem with
  the correct quantities".
- Keep every "reason" and "fix" clear and concise.
- "overall" is "OK" only if NO criterion is FAIL.
- Output only the JSON object."""


_RFT_DIAG_USER = """\
## Task / query (context)
{query}

## Rubric
{rubric}

## Segment
{segment}

Now output the diagnostic JSON object."""


class _RftRubricVerifier(RubricVerifier):
    def _diagnose_trajectory(self, query: str, rubric_block: str, segment_text: str) -> dict:
        user = _RFT_DIAG_USER.format(query=query, rubric=rubric_block, segment=segment_text)
        return {'messages': [
            {'role': 'system', 'content': _RFT_DIAG_SYSTEM},
            {'role': 'user', 'content': user},
        ]}


_MATH_RUBRIC = [
    ('The reasoning contains no arithmetic or algebraic error', True),
    ('Each step follows logically from the previous ones', True),
    ('No formula or theorem is misstated or misapplied', True),
    ('The approach is on track to answer the actual question asked', False),
    ('No step contradicts an earlier established fact', False),
]


def _build_rubric_checker() -> Optional['RubricVerifier']:
    """Fixed math-process rubric verifier, teacher-served. None if no LLM backup env."""
    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('LLM_BACKUP_BASE_URL')
            or os.environ.get('OPENAI_API_KEY')):
        return None
    return _RftRubricVerifier(
        fixed_rubric=[RubricItem(t, is_hard=h) for t, h in _MATH_RUBRIC], gate=True)


def _format_diagnosis(detail) -> str:
    """One line per criterion (PASS/FAIL + reason + fix on FAIL) then a summary — the
    compact evidence block appended to the view-A skill-gen prompt."""
    rub = detail.rubric
    lines = []
    for it in detail.items:
        text = rub[it.index - 1].text if 0 < it.index <= len(rub) else f'criterion {it.index}'
        if it.verdict:
            lines.append(f'- [PASS] {text}')
        else:
            tail = f': {it.reason}' if it.reason else ''
            tail += f' (fix: {it.fix})' if it.fix else ''
            lines.append(f'- [FAIL] {text}{tail}')
    if detail.summary:
        lines.append(f'Summary: {detail.summary}')
    return '\n'.join(lines)


def _diagnose_views(checker, hard: List[Dict[str, Any]], args: argparse.Namespace,
                    diag_cache: Optional[Dict[str, str]] = None) -> None:
    """Rubric-check every view-A problem's greedy attempt in parallel, stashing the
    formatted findings on ``r['_rubric_diag']`` (view B stays empty). A checker error
    or empty result degrades to no diagnosis (the plain view-A prompt)."""
    from concurrent.futures import ThreadPoolExecutor
    targets = [r for r in hard if r.get('_view') == 'A']
    if not checker or not targets:
        return

    def _cache_key(r: Dict[str, Any]) -> str:
        init_text = r.get('_init', [{}])[0].get('text', '')
        return hashlib.md5(f'{r["problem"]}\n{init_text}'.encode('utf-8')).hexdigest()

    pending = []
    for r in targets:
        key = _cache_key(r)
        if diag_cache is not None and key in diag_cache:
            r['_rubric_diag'] = diag_cache[key]
        else:
            pending.append((r, key))
    if not pending:
        return

    def _run(item: Tuple[Dict[str, Any], str]) -> Tuple[Dict[str, Any], str, str, bool]:
        r, key = item
        seg = {'messages': [
            {'role': 'user', 'content': r['problem']},
            {'role': 'assistant', 'content': r['_init'][0]['text']},
        ]}
        try:
            return r, key, _format_diagnosis(checker.diagnose(seg, query=r['problem'])), True
        except Exception as exc:  # teacher hiccup -> fall back to no-diagnosis prompt
            logger.warning(f'[rubric] diagnose error: {exc}')
            return r, key, '', False

    workers = max(1, min(args.rubric_workers, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, key, diag, ok in ex.map(_run, pending):
            r['_rubric_diag'] = diag
            if ok and diag_cache is not None:
                diag_cache[key] = diag


# ---------------------------------------------------------------------------
# Online data generation (one chunk; every candidate is recorded, untruncated)
# ---------------------------------------------------------------------------
def _roll(x: Dict[str, Any]) -> Dict[str, Any]:
    """Full (untruncated) rollout record for offline analysis."""
    return {k: x[k] for k in ('pred', 'correct', 'terminated', 'passed',
                              'stop_reason', 'gen_tokens', 'text')}


def _empty_roll() -> Dict[str, Any]:
    """Fallback rollout when the sampler returned nothing for a prompt."""
    return {'pred': '', 'correct': False, 'terminated': False, 'passed': False,
            'stop_reason': 'empty', 'gen_tokens': 0, 'text': ''}


def _assign_advantages(hard: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Group-relative advantage (SEAM-style) over each problem's clean, scored candidates,
    using the DETERMINISTIC greedy reward ``R in {0, 1}`` (answer CORRECT only;
    termination is NOT part of the reward -- monitored via `terminated`/`passed` only):

        A_j = (R_j - mean_R) / (std_R + eps)

    Groups where every candidate shares the same reward (``std_R == 0``: all solve or all
    fail) get advantage 0 and contribute no gradient -- GRPO's own group variance
    auto-selects the informative problems, so no explicit difficulty / marginal gate is
    needed. Because the reward is deterministic (M=1 greedy, no pass@k sampling), the
    std-normalisation no longer amplifies rollout noise (the reason it was dropped for the
    old stochastic marginal). ``kept`` marks above-average candidates (for reporting only).
    """
    eps = 1e-6
    for r in hard:
        for c in r['_cands']:
            c['advantage'], c['grpo_adv'], c['kept'] = 0.0, 0.0, False
        cs = ([c for c in r['_cands'] if c.get('reward') is not None] if args.format_in_reward
              else [c for c in r['_cands'] if c['leaked'] is False and c.get('reward') is not None])
        if len(cs) < 2:
            continue
        rewards = [c['reward'] for c in cs]
        mean_r = sum(rewards) / len(rewards)
        std = (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
        if std < 1e-9:
            continue  # all candidates equal (all solve / all fail) -> no learning signal
        for c in cs:
            adv = (c['reward'] - mean_r) / (std + eps)
            c['advantage'], c['grpo_adv'], c['kept'] = adv, adv, c['reward'] > mean_r


def _full_record(r: Dict[str, Any], ci: int) -> Dict[str, Any]:
    """Complete per-problem record: init attempt, baseline, and ALL candidates
    (parseable/leaked/scored alike) with full text — nothing dropped or truncated."""
    init = r['_init'][0]
    rec: Dict[str, Any] = {
        'record_type': 'problem', 'chunk': ci, 'problem': r['problem'],
        'reference_answer': r['reference_answer'], 'level': r.get('level', ''),
        'failed_first_try': r['_failed'],
        'init_attempt': {'text': init['text'], 'pred': init['pred'],
                         'correct': init['correct'], 'terminated': init['terminated'],
                         'stop_reason': init['stop_reason'], 'gen_tokens': init['gen_tokens']},
    }
    rec['baseline_pass'] = r['_baseline_pass']
    rec['is_hard'] = r['_hard']
    rec['view'] = r.get('_view', '')
    rec['rubric_diag'] = r.get('_rubric_diag', '')
    rec['baseline_rolls'] = [_roll(x) for x in r['_baseline_rolls']]
    rec['candidates'] = [{
        'skills': c['skills'], 'response': c['response'], 'parseable': c['parseable'],
        'leaked': c['leaked'], 'leak_reason': c['leak_reason'], 'leak_source': c['leak_source'],
        'with_pass': c['with_pass'], 'reward': c.get('reward'),
        'advantage': c.get('advantage'), 'grpo_adv': c.get('grpo_adv'), 'kept': c.get('kept'),
        'rolls': [_roll(x) for x in c['rolls']],
    } for c in r['_cands']]
    return rec


def _view_stats(hard: List[Dict[str, Any]], view: str) -> Dict[str, Any]:
    """Per-view yield: hard problems, clean candidates, and the ADOPTION rate —
    the fraction of hard problems that produced at least one clean, non-zero-advantage
    candidate (i.e. a record that actually reaches training). Watching A vs B and
    early vs late tells whether query-only (B) catches up to trajectory-grounded (A)."""
    hv = [r for r in hard if r.get('_view') == view]
    cands = [c for r in hv for c in r['_cands'] if c['parseable']]
    clean = [c for c in cands if c['leaked'] is False]
    adopted = sum(1 for r in hv
                  if any(c['leaked'] is False and abs(c.get('advantage') or 0.0) > 1e-9
                         for c in r['_cands']))
    return {
        'n_hard': len(hv), 'n_candidates_parseable': len(cands),
        'n_clean': len(clean), 'n_adopted_problems': adopted,
        'adoption_rate': (adopted / len(hv)) if hv else 0.0,
    }


def _is_trainable(c: Dict[str, Any], args: argparse.Namespace) -> bool:
    """A candidate reaches the GRPO update iff its advantage is non-zero. With
    --format-in-reward every candidate carries a reward (unparseable/leaked score 0),
    so non-zero advantage is the only gate; otherwise it must also be clean and scored.
    Single source of truth for both the summary counts and ``_group_records``."""
    adv_nz = abs(c.get('advantage') or 0.0) > 1e-9
    if args.format_in_reward:
        return adv_nz
    return c['leaked'] is False and c.get('with_pass') is not None and adv_nz


def _chunk_summary(chunk: List[Dict[str, Any]], ci: int, args: argparse.Namespace) -> Dict[str, Any]:
    """Per-chunk aggregates — watch these across chunks to see if the RFT'd skill
    model produces better skills over time (yield, leak rate, lift, termination)."""
    failed = [r for r in chunk if r['_failed']]
    hard = [r for r in chunk if r['_hard']]
    all_cands = [c for r in chunk for c in r['_cands']]
    cands = [c for c in all_cands if c['parseable']]
    scored = [c for c in cands if c['with_pass'] is not None]
    ws_rolls = [x for c in scored for x in c['rolls']]
    # With --format-in-reward, unparseable/leaked candidates also carry a (0) reward and are
    # trained, so count trainables over ALL candidates; else only clean scored ones.
    train_cands = [c for c in all_cands if _is_trainable(c, args)]
    base_acc = (sum(r['_baseline_pass'] for r in hard) / len(hard)) if hard else 0.0
    ws_acc = (sum(c['with_pass'] for c in scored) / len(scored)) if scored else 0.0
    # -- signal-source monitor: how much of the GRPO signal comes from base-FAIL problems
    # (the offensive "rescue a failure" signal we want) vs base-success (defensive "don't
    # break an easy one"). abs_adv_from_fail_frac ~0.1 was the diagnosed failure mode. --
    fail_cands = [c for r in chunk if r['_failed'] for c in r['_cands']]
    abs_adv = lambda cs: sum(abs(c.get('advantage') or 0.0) for c in cs)
    total_abs = abs_adv(all_cands)
    return {
        'record_type': 'summary', 'chunk': ci, 'n': len(chunk),
        'n_failed_first_try': len(failed), 'n_hard': len(hard),
        'n_generated': len(all_cands),
        'n_candidates_parseable': len(cands),
        'n_unparseable': len(all_cands) - len(cands),
        'n_leaked': sum(1 for c in cands if c['leaked']),
        'n_clean': sum(1 for c in cands if c['leaked'] is False),
        'n_reward_pos': sum(1 for c in scored if c['reward']),
        'n_train_samples': len(train_cands),
        'n_train_from_fail': sum(1 for c in fail_cands if _is_trainable(c, args)),
        'abs_adv_from_fail_frac': (abs_adv(fail_cands) / total_abs) if total_abs > 0 else 0.0,
        'avg_baseline_pass_on_hard': base_acc,
        'avg_withskill_pass': ws_acc,
        'avg_lift': ws_acc - base_acc,
        'termination_rate_withskill': (sum(1 for x in ws_rolls if x['terminated']) / len(ws_rolls)) if ws_rolls else 0.0,
        'view_A': _view_stats(hard, 'A'), 'view_B': _view_stats(hard, 'B'),
    }


def _group_records(chunk: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """GRPO training records: every clean, scored skill candidate with a NON-ZERO
    advantage (positive pushes the skill up, negative down; the group-relative
    usefulness-over-base advantage was set in _assign_advantages). Each carries its
    ``view`` and the rubric ``diagnosis``; the prompt (identical to generation) is rebuilt
    from those by ``_skillgen_messages`` -- no trajectory is stored or replayed."""
    out = []
    for r in chunk:
        if not r['_hard']:
            continue
        view = r.get('_view', 'A')
        for c in r['_cands']:
            if _is_trainable(c, args):
                out.append({
                    'problem': r['problem'], 'reference_answer': r['reference_answer'],
                    'view': view,
                    'diagnosis': r.get('_rubric_diag', ''),
                    'response': c['response'], 'skills': c['skills'],
                    'advantage': c['advantage'], 'grpo_adv': c['grpo_adv'], 'kept': c['kept'],
                    'reward': c['reward'], 'with_pass': c['with_pass'],
                })
    return out


def _apply_baseline(r: Dict[str, Any], roll: Dict[str, Any]) -> None:
    """Write a (cached or fresh) greedy baseline roll onto a problem and RESET the per-chunk
    working state, so a problem reused in a later chunk never carries prior skill candidates."""
    r['_baseline_rolls'], r['_cands'] = [roll], []
    r['_init'] = [roll]
    r['_failed'] = not roll['correct']
    r['_baseline_pass'] = 1.0 if roll['correct'] else 0.0
    r['_hard'] = True                                   # process EVERY selected problem; group variance selects


def _baseline_rollout(base_sampler, problems: List[Dict[str, Any]], base_dp: int,
                      args: argparse.Namespace, cache: Dict[str, Dict[str, Any]]) -> int:
    """Phase 1: base solves each problem GREEDILY once (T=0, M=1), keyed-cached by problem
    text across chunks. The base sampler is FROZEN and decoding is greedy, so a problem's
    baseline never changes over the run -- a cache hit is exact and skips the sampler.
    Returns the number of FRESH sampler rollouts (cache misses) for efficiency reporting."""
    todo = [r for r in problems if r['problem'] not in cache]
    if todo:
        base_out = _run_samples(base_sampler, [build_direct_prompt(r['problem']) for r in todo],
                                1, args.max_tokens, base_dp, temperature=0.0)
        for r, seqs in zip(todo, base_out):
            cache[r['problem']] = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
    for r in problems:
        _apply_baseline(r, cache[r['problem']])
    return len(todo)


def _baseline_class(r: Dict[str, Any]) -> str:
    """Bucket a baselined problem by its greedy outcome: ``success`` (base solved it),
    ``fail_loop`` (ran the length budget out / never terminated -- the mode skills rescue
    best), or ``fail_wrong`` (terminated cleanly but the answer is wrong)."""
    roll = r['_init'][0]
    if roll['correct']:
        return 'success'
    if roll['stop_reason'] == 'length' or not roll['terminated']:
        return 'fail_loop'
    return 'fail_wrong'


_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _norm_num_text(num: str) -> str:
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return str(num).strip()


def _numeric_value(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().strip('$').strip()
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    s = re.sub(r'\\!|\\,|\\;|\\ |\\left|\\right|\s', '', s)
    m = re.fullmatch(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return _norm_num_text(str(a / b)) if b else None
    m = re.fullmatch(r'(-?\d+)/(-?\d+)', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return _norm_num_text(str(a / b)) if b else None
    if _NUM_RE.fullmatch(s):
        return _norm_num_text(s)
    return None


def _numeric_only_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    out = []
    dropped = 0
    for r in records:
        ref = _numeric_value(r.get('reference_answer'))
        if ref is None:
            dropped += 1
            continue
        rr = dict(r)
        rr['reference_answer'] = ref
        out.append(rr)
    return out, dropped


def _load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    need_split = args.eval_size > 0
    load_n = 0 if (args.numeric_only or need_split) else args.n
    records = (load_aops(n=load_n, seed=args.seed) if args.dataset == 'aops'
               else load_math(n=load_n, seed=args.seed))
    raw_n = len(records)
    dropped = 0
    if args.numeric_only:
        records, dropped = _numeric_only_records(records)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(records)
    eval_n = min(args.eval_size, len(records)) if args.eval_size > 0 else 0
    eval_records = [dict(r) for r in records[:eval_n]]
    train_pool = records[eval_n:]
    train_n = args.n if args.n > 0 else len(train_pool)
    train_records = [dict(r) for r in train_pool[:train_n]]
    overlap = {r['problem'] for r in train_records} & {r['problem'] for r in eval_records}
    if overlap:
        raise ValueError(f'fixed eval/train overlap detected: {len(overlap)} duplicated problems')
    stats = {'raw_loaded': raw_n, 'numeric_dropped': dropped,
             'train_records': len(train_records), 'eval_records': len(eval_records)}
    return train_records, eval_records, stats


class _ProblemPool:
    """Cyclic draw source over the loaded problems. Each full pass reshuffles with
    ``seed + epoch`` and bumps ``epoch`` (matching the old per-epoch reshuffle); the
    initial pass keeps the loader's shuffled order. Draws never run out."""

    def __init__(self, records: List[Dict[str, Any]], seed: int):
        self._records = list(records)
        self._seed = seed
        self._cursor = 0
        self.epoch = 0
        self.baseline_cache: Dict[str, Dict[str, Any]] = {}  # problem text -> frozen greedy roll

    def draw(self, k: int) -> List[Dict[str, Any]]:
        """Return ``k`` DISTINCT problems (unique within this call, so one chunk never
        processes the same problem twice even when the cursor wraps mid-draw). ``k`` is
        always << pool size, so this terminates."""
        out: List[Dict[str, Any]] = []
        seen: set = set()
        while len(out) < k:
            if self._cursor >= len(self._records):
                self.epoch += 1
                np.random.RandomState(self._seed + self.epoch).shuffle(self._records)
                self._cursor = 0
            r = self._records[self._cursor]
            self._cursor += 1
            if id(r) not in seen:
                seen.add(id(r))
                out.append(r)
        return out


def _select_balanced(buckets: Dict[str, List[Dict[str, Any]]], n_success: int,
                     n_fail: int, n_fail_loop: int) -> List[Dict[str, Any]]:
    """Pick the chunk from the baselined buckets: ``n_fail`` base-fails (split toward
    ``n_fail_loop`` loop-fails, best-effort) + ``n_success`` base-successes. If a bucket
    is too thin to hit ``chunk_size`` the shortfall is topped up from leftovers (the
    ratio then drifts, which the caller logs)."""
    loop, wrong, succ = buckets['fail_loop'], buckets['fail_wrong'], buckets['success']
    take_loop = min(n_fail_loop, len(loop))
    take_wrong = min(n_fail - take_loop, len(wrong))
    take_loop = min(n_fail - take_wrong, len(loop))   # give loop the remainder if wrong is short
    sel = loop[:take_loop] + wrong[:take_wrong] + succ[:n_success]
    target = n_success + n_fail
    if len(sel) < target:
        used = {id(x) for x in sel}
        leftover = [x for b in (loop, wrong, succ) for x in b if id(x) not in used]
        sel += leftover[:target - len(sel)]
    return sel


def _draw_chunk(pool: _ProblemPool, base_sampler, base_dp: int, args: argparse.Namespace
                ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Draw one training chunk, running baseline rollout (Phase 1) on every drawn problem.

    With ``--balance`` off, draw ``chunk_size`` problems and return them. With it on, keep
    drawing+baselining in ``chunk_size`` batches, bucketing by ``_baseline_class``, until the
    target base fail:success mix is reachable or the draw budget is hit; then select a
    balanced subset. Returns ``(chunk, stats)`` where stats records the realised mix."""
    if not args.balance:
        chunk = pool.draw(args.chunk_size)
        n_fresh = _baseline_rollout(base_sampler, chunk, base_dp, args, pool.baseline_cache)
        return chunk, {'enabled': False, 'n_drawn': len(chunk), 'n_baseline_fresh': n_fresh}

    n_success = max(0, min(args.chunk_size, round(args.chunk_size * args.balance_success_frac)))
    n_fail = args.chunk_size - n_success
    n_fail_loop = round(n_fail * args.balance_loop_frac)
    buckets: Dict[str, List[Dict[str, Any]]] = {'success': [], 'fail_loop': [], 'fail_wrong': []}
    budget = args.chunk_size * args.balance_max_draws_mult
    n_drawn, n_fresh = 0, 0
    seen: set = set()  # dedupe across batches: the pool can re-serve a problem after a wrap
    while n_drawn < budget:
        if (len(buckets['success']) >= n_success
                and len(buckets['fail_loop']) + len(buckets['fail_wrong']) >= n_fail):
            break  # enough of both classes buffered to satisfy the target split
        batch = pool.draw(args.chunk_size)
        n_fresh += _baseline_rollout(base_sampler, batch, base_dp, args, pool.baseline_cache)
        n_drawn += len(batch)
        for r in batch:
            if id(r) in seen:
                continue
            seen.add(id(r))
            buckets[_baseline_class(r)].append(r)

    target_reached = (len(buckets['success']) >= n_success
                      and len(buckets['fail_loop']) + len(buckets['fail_wrong']) >= n_fail)
    chunk = _select_balanced(buckets, n_success, n_fail, n_fail_loop)
    sel_success = sum(1 for r in chunk if not r['_failed'])
    stats = {
        'enabled': True, 'n_drawn': n_drawn, 'n_baseline_fresh': n_fresh, 'n_selected': len(chunk),
        'target_success': n_success, 'target_fail': n_fail, 'target_fail_loop': n_fail_loop,
        'selected_success': sel_success, 'selected_fail': len(chunk) - sel_success,
        'selected_fail_loop': sum(1 for r in chunk if _baseline_class(r) == 'fail_loop'),
        'selected_fail_wrong': sum(1 for r in chunk if _baseline_class(r) == 'fail_wrong'),
        'selected_success_frac': (sel_success / len(chunk)) if chunk else 0.0,
        'budget_hit': not target_reached,   # stopped short of the target mix, not by choice
    }
    return chunk, stats


def process_chunk(base_sampler, skill_sampler, leak: LeakVerifier,
                  chunk: List[Dict[str, Any]], ci: int, base_dp: int, skill_dp: int,
                  args: argparse.Namespace, checker=None,
                  diag_cache: Optional[Dict[str, str]] = None
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """base-solve -> rubric-check (view A) -> skill-gen -> leak-filter -> with-skill pass
    -> GRPO advantages, for one chunk.

    Sequential (generate-one-chunk-train-one): generation and the trainer's weight
    sync never overlap, so no lock is needed. ``base_sampler`` is frozen (never
    synced); ``skill_sampler`` is synced by the trainer between chunks.

    ``chunk`` arrives ALREADY baselined by ``_draw_chunk`` (Phase 1 ran during the
    balanced draw), so every problem carries ``_init``/``_failed``/``_baseline_pass``/
    ``_hard``/``_cands`` -- Phase 1 is not repeated here.
    """
    # Phase 1 (base greedy solve) ran in _draw_chunk so the balancer could classify by
    # outcome; every selected problem is processed (no difficulty gate, SEAM-style): the
    # group-relative advantage (Phase 6) gives zero gradient to any problem whose skills
    # all score alike, so GRPO's own group variance selects the informative problems.
    hard = chunk

    # --- Phase 2: assign each problem's view, then rubric-check the view-A attempts so
    # the skill model diagnoses from verified findings instead of guessing. View B is
    # query-only and deliberately gets NO rubric (nothing to diagnose without an attempt). ---
    for r in hard:
        r['_view'] = _assign_view(r['problem'], args)
        r['_rubric_diag'] = ''
    _diagnose_views(checker, hard, args, diag_cache)

    # --- Phase 3: skill-gen (thinking ON), per-view prompt; re-sample empties. ---
    flat: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    if hard:
        pending = list(hard)  # problems still without any clean candidate
        for _ in range(args.skill_retries + 1):
            if not pending:
                break
            prompts = [_view_prompt(r, args) for r in pending]
            sg_out = _run_samples(skill_sampler, prompts, args.n_skills,
                                  args.skill_max_tokens, skill_dp,
                                  temperature=args.skill_gen_temperature,
                                  top_p=args.skill_gen_top_p,
                                  top_k=args.skill_gen_top_k)
            still = []
            for r, seqs in zip(pending, sg_out):
                got = False
                for s in seqs:
                    resp = _clean_text(getattr(s, 'decoded', '') or '')
                    block = _extract_skills_block(resp)
                    cand = {'skills': block or '', 'response': resp, 'parseable': bool(block),
                            'view': r['_view'], 'leaked': None, 'leak_reason': '',
                            'leak_source': '', 'with_pass': None,
                            'reward': None, 'rolls': []}
                    r['_cands'].append(cand)
                    if block:
                        flat.append((r, cand))
                        got = True
                if not got:
                    still.append(r)  # nothing parseable yet -> retry this problem
            pending = still

    # --- Phase 4: leak filter via backup teacher (network only, no lock). VIEW A ONLY --
    # view B is query-only (no trajectory to leak from) and is left exactly like SEAM, which
    # runs NO leak filter: its candidates skip the check and are treated as clean. To restore
    # leak-checking on view B, drop the ``_view == 'A'`` guard below. ---
    for r, c in flat:
        if r.get('_view') != 'A':
            c['leaked'], c['leak_reason'], c['leak_source'] = False, '', 'skipped_viewB'
    flat_a = [(r, c) for r, c in flat if r.get('_view') == 'A']
    if flat_a:
        details = leak.leak_batch(
            [{'content': c['skills'], 'query': r['problem'], 'reference': r['reference_answer']}
             for r, c in flat_a], max_workers=args.leak_workers)
        for (r, c), d in zip(flat_a, details):
            c['leaked'], c['leak_reason'], c['leak_source'] = bool(d.leaked), d.reason, d.source

    # --- Phase 5: with-skill GREEDY pass (T=0, M=1) on clean candidates. Binary reward
    # R = answer CORRECT (deterministic, no pass@k noise), ABSOLUTE -- no baseline
    # subtraction; the group mean in Phase 6 is the only baseline. Termination is NOT
    # required (monitored only) -- see reflexion.md §7.6. ---
    clean = [(r, c) for r, c in flat if c['leaked'] is False]
    if clean:
        ws_out = _run_samples(base_sampler,
                              [build_skill_solve_prompt(r['problem'], c['skills']) for r, c in clean],
                              1, args.max_tokens, base_dp, temperature=0.0)
        for (r, c), seqs in zip(clean, ws_out):
            c['rolls'] = [_parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()]
            c['with_pass'] = 1.0 if c['rolls'][0]['correct'] else 0.0
            c['reward'] = c['with_pass']                       # valid + clean + correct -> 1
    # Validity-in-reward (SEAM-style, --format-in-reward): every candidate that never reached
    # the executor -- unparseable/impure format OR answer-leaked -- scores 0 and STILL joins its
    # group, so its whole response (think tokens included) is trained DOWN. Off => those
    # candidates are excluded, as before.
    if args.format_in_reward:
        for r in hard:
            for c in r['_cands']:
                if c['reward'] is None:
                    c['reward'] = 0.0

    # --- Phase 6: group-relative GRPO advantage per problem-group. ---
    _assign_advantages(hard, args)

    return ([_full_record(r, ci) for r in chunk],
            _chunk_summary(chunk, ci, args),
            _group_records(chunk, args))


# ---------------------------------------------------------------------------
# Online RFT training
# ---------------------------------------------------------------------------
def _train_trajectory(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Training sample = the exact skill-gen prompt for this record's view + the
    generated (think + skills) response as the target; the GRPO advantage is attached
    separately at forward_backward time.

    The prompt is rebuilt by ``_skillgen_messages`` (the same function used at generation),
    so train/inference stay identical: view A replays problem + rubric findings, view B (and
    no-failure view A) replays the query-only prompt. ``key_rounds`` selects the final
    assistant turn (index ``len(msgs)``); the plain ``Template`` then masks the prompt and
    trains the whole generated response (reasoning + ``</think>`` + skills) -- the key-round
    prefix already excludes the prompt-provided ``<think>``, so no extra masking is needed."""
    msgs = _skillgen_messages(rec['problem'], rec.get('view', 'A'), rec.get('diagnosis', ''))
    full = msgs + [{'role': 'assistant', 'content': rec['response']}]
    return {'messages': full, 'user_data': {'key_rounds': [len(msgs)]}}


def _train_chunk(skill_model, ckpt: Optional[CheckpointEngineManager],
                 samples: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """One on-policy GRPO optimizer update on THIS chunk's skill candidates, then sync weights.

    Sequential design (generate-one-chunk-train-one): the skills were sampled from the
    current policy and trained immediately, so ``old_logps`` is omitted and the GRPO
    ratio is ~1. All driver-side mini-batches accumulate into one optimizer step so the
    whole rollout chunk stays under the same pre-update policy. The batch is padded to a
    multiple of ``sft_batch_size`` with advantage-0 copies that contribute zero gradient.
    """
    trajs = [_train_trajectory(rec) for rec in samples]
    advs = [float(rec['advantage']) for rec in samples]
    rem = (-len(trajs)) % args.sft_batch_size
    if rem:
        trajs += [trajs[-1]] * rem       # zero-advantage pads -> forward only, no gradient
        advs += [0.0] * rem
    micro_batches = 0
    for i in range(0, len(trajs), args.sft_batch_size):
        skill_model.forward_backward(inputs=trajs[i:i + args.sft_batch_size],
                                     advantages=advs[i:i + args.sft_batch_size])
        micro_batches += 1
    skill_model.clip_grad_and_step()
    if ckpt is not None:
        ckpt.sync_weights(merge_and_sync=True)
    metric = skill_model.calculate_metric(is_training=True)
    return {'n_samples': len(samples), 'n_steps': 1, 'n_micro_batches': micro_batches,
            'advantages': [float(rec['advantage']) for rec in samples],
            'metric': {k: (float(v) if _is_num(v) else v) for k, v in (metric or {}).items()}}


def _is_num(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops',
                   help='Problem source. aops (AI-MO competition problems) is much '
                        'harder than MATH, so the base fails more often -> more offensive '
                        'training signal after balanced sampling.')
    p.add_argument('--n', type=int, default=2000,
                   help='Problems to load into the draw pool (cycled/reshuffled across '
                        'epochs; with --balance many more baseline rollouts than this '
                        'may run, but the pool size is fixed here).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True,
                   help='Keep only answers that collapse to one integer/decimal/fraction, '
                        'matching SEAM numeric reward and avoiding non-scalar grading noise.')
    p.add_argument('--eval-size', type=int, default=128,
                   help='Fixed holdout problems, sampled before the train pool after all '
                        'filters; set 0 to disable fixed eval.')
    p.add_argument('--eval-every', type=int, default=10,
                   help='Run fixed holdout eval every N generation chunks when --eval-size > 0.')
    p.add_argument('--chunk-size', type=int, default=16,
                   help='Problems per generation chunk (all sampler calls batched).')
    # -- online baseline-balanced sampling (draw+baseline until the chunk hits the
    #    target base fail:success mix, so the offensive signal is not starved) --
    p.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True,
                   help='Keep drawing+baselining problems until the chunk matches the '
                        'target base fail:success composition, then select a balanced '
                        'subset. --no-balance draws chunk_size problems directly.')
    p.add_argument('--balance-success-frac', type=float, default=0.4,
                   help='Target fraction of the chunk that the base solves (base-success). '
                        '0.4 => 3:2 fail:success; 0.2 => 4:1. The remainder are base-fail.')
    p.add_argument('--balance-loop-frac', type=float, default=0.5,
                   help='Within the base-fail portion, SOFT target fraction of loop-fails '
                        '(ran out of length / never terminated) vs non-loop wrong answers. '
                        'Best-effort only: the fail count is filled from whichever bucket '
                        'is available so a thin bucket never starves the chunk.')
    p.add_argument('--balance-max-draws-mult', type=int, default=8,
                   help='Draw budget per chunk as a multiple of chunk_size; once this many '
                        'problems have been baselined the chunk is assembled from whatever '
                        'the buckets hold (ratio may drift; the actual mix is logged).')
    p.add_argument('--n-skills', type=int, default=8,
                   help='Candidate skills generated per hard problem.')
    p.add_argument('--view-b-frac', type=float, default=0.5,
                   help='Fraction of hard problems routed to view B (query-only, '
                        'deployment form); the rest go to view A (problem + attempt). '
                        'Each problem is assigned to EXACTLY ONE view.')
    p.add_argument('--skill-retries', type=int, default=2,
                   help='Extra skill-gen rounds for a hard problem that yielded no '
                        'clean, parseable candidate (thinking-ON purity gate rejects).')
    p.add_argument('--skill-gen-temperature', type=float, default=1.0,
                   help='Sampling temperature for skill-gen (BOTH views). >0 so the '
                        'n_skills candidates per problem are genuinely DIVERSE — a group '
                        'of near-duplicate skills gives GRPO no real good-vs-bad contrast.')
    p.add_argument('--skill-gen-top-p', type=float, default=1.0,
                   help='top_p for skill-gen; 1.0 keeps the full tail for diversity.')
    p.add_argument('--skill-gen-top-k', type=int, default=-1,
                   help='top_k for skill-gen; -1 disables truncation (max diversity). '
                        'A finite value only narrows the candidate pool.')
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192,
                   help='Max generated tokens for solve rollouts.')
    p.add_argument('--skill-max-tokens', type=int, default=8192,
                   help='Max tokens for skill-gen (thinking ON: the model must close '
                        '</think> within this budget or the candidate is dropped, so '
                        'leave ample room).')
    p.add_argument('--leak-workers', type=int, default=16,
                   help='Parallel workers for the LeakVerifier backup judge (capped at 16 '
                        'to avoid the teacher API burst-rate limit; leak and rubric run in '
                        'separate phases so peak teacher concurrency is max(leak,rubric)).')
    p.add_argument('--rubric-workers', type=int, default=16,
                   help='Parallel workers for the view-A rubric diagnose() calls '
                        '(teacher-served; requires LLM_BACKUP_* env).')
    # -- online GRPO (one on-policy update per generated chunk) --
    p.add_argument('--sft-batch-size', type=int, default=8,
                   help='Driver-side micro-batch size before the chunk-level optimizer step; '
                        'MUST be a multiple of the training dp size (sliced across dp ranks).')
    p.add_argument('--grpo-epsilon', type=float, default=0.2,
                   help='PPO clip epsilon for GRPOLoss (ratio~1 on-policy, so rarely binds).')
    p.add_argument('--format-in-reward', action=argparse.BooleanOptionalAction, default=True,
                   help='Fold output validity into the reward (SEAM-style): unparseable/impure '
                        'or answer-leaked candidates score 0 and join their group to be trained '
                        'DOWN (the whole response, think tokens included). '
                        '--no-format-in-reward keeps the reject-and-exclude gate.')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max-train-rounds', type=int, default=200,
                   help='Cap on train rounds = trained chunks (also sizes the LR schedule).')
    p.add_argument('--save-rounds', type=int, default=50)
    p.add_argument('--trend-every', type=int, default=10,
                   help='Every N chunks, print a [trend] line contrasting the first N '
                        'vs the most recent N chunks (adoption + lift + pos/chunk) '
                        'so the training effect on fresh problems is visible at a glance.')
    p.add_argument('--output-dir', default='./output/reflexion_skill_rft')
    p.add_argument('--swanlab-project', default='twinkle',
                   help='swanlab project; logging is skipped when swanlab is not '
                        'installed or SWANLAB_MODE=disabled.')
    p.add_argument('--swanlab-exp', default='',
                   help='swanlab experiment (run) name; empty = auto.')
    return p.parse_args()


def _trend_line(hist: List[Dict[str, float]], window: int, rounds_done: int) -> Optional[str]:
    """Contrast the FIRST ``window`` chunks with the most recent ``window`` chunks so
    the online training effect on fresh, never-trained problems is glanceable: if RFT
    is working, adoption and lift on recent chunks exceed the early baseline."""
    if len(hist) < 2 * window:
        return None  # need two non-overlapping windows for a clean before/after
    base, rec = hist[:window], hist[-window:]
    m = lambda xs, k: sum(h[k] for h in xs) / len(xs)
    return (f'[trend] first {window} vs last {window} chunks | '
            f'adopt A {m(base,"aA"):.2f}->{m(rec,"aA"):.2f} '
            f'B {m(base,"aB"):.2f}->{m(rec,"aB"):.2f} | '
            f'lift {m(base,"lift"):+.3f}->{m(rec,"lift"):+.3f} | '
            f'pos/chunk {m(base,"pos"):.1f}->{m(rec,"pos"):.1f} | rounds={rounds_done}')


def _query_rows(full: List[Dict[str, Any]]) -> List[Tuple[float, float, float, int, str]]:
    """Per hard problem that produced >=1 scored candidate: its no-skill baseline
    pass@k, the BEST and MEAN with-skill pass@k over its N skill candidates, the scored
    count, and the problem text. Drives both the per-query print and the swanlab passk/*
    aggregates."""
    rows = []
    for rec in full:
        if rec.get('record_type') != 'problem' or not rec.get('is_hard'):
            continue
        ps = [c['with_pass'] for c in rec.get('candidates', []) if c.get('with_pass') is not None]
        if not ps:
            continue
        rows.append((rec['baseline_pass'], max(ps), sum(ps) / len(ps), len(ps), rec['problem']))
    return rows


def _clean_metric(metric: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Numeric GRPO metrics for swanlab: collapse the duplicate per-group LR to a single
    ``lr`` and drop non-numeric fields (e.g. 'total time elapse')."""
    out: Dict[str, float] = {}
    for k, v in (metric or {}).items():
        if not _is_num(v):
            continue
        if k.startswith('learning rate'):
            if 'group 1' in k:
                out['lr'] = float(v)
        else:
            out[k.replace(' ', '_')] = float(v)
    return out


def _swan_metrics(summary: Dict[str, Any], log: Optional[Dict[str, Any]],
                  rows: List[Tuple[float, float, float, int, str]]) -> Dict[str, float]:
    """Flat metric dict for swanlab = external reflexion metrics + (when this chunk was
    trained) the GRPO built-in metric. acc/adopt/term are only emitted on chunks that had
    hard problems, and passk/* only when scored candidates exist, so idle chunks don't dip
    the charts to zero."""
    d: Dict[str, float] = {
        'gen/n_hard': summary['n_hard'], 'gen/n_clean': summary['n_clean'],
        'gen/n_leaked': summary['n_leaked'], 'gen/n_train_samples': summary['n_train_samples'],
        'gen/n_reward_pos': summary['n_reward_pos'],
        'gen/n_train_from_fail': summary['n_train_from_fail'],
        'gen/abs_adv_from_fail_frac': summary['abs_adv_from_fail_frac'],
    }
    bal = summary.get('balance') or {}
    if bal.get('enabled'):
        d.update({'balance/n_drawn': bal['n_drawn'],
                  'balance/n_baseline_fresh': bal['n_baseline_fresh'],
                  'balance/selected_success_frac': bal['selected_success_frac'],
                  'balance/selected_fail_loop': bal['selected_fail_loop'],
                  'balance/selected_fail_wrong': bal['selected_fail_wrong']})
    if summary['n_hard'] > 0:
        d.update({
            'acc/baseline_pass': summary['avg_baseline_pass_on_hard'],
            'acc/withskill_pass': summary['avg_withskill_pass'],
            'acc/lift': summary['avg_lift'],
            'adopt/A': summary['view_A']['adoption_rate'],
            'adopt/B': summary['view_B']['adoption_rate'],
            'term/withskill': summary['termination_rate_withskill'],
        })
    if rows:
        m = lambda i: sum(r[i] for r in rows) / len(rows)
        d.update({'passk/baseline_mean': m(0), 'passk/bestN_mean': m(1), 'passk/avgN_mean': m(2)})
    if log:
        d['train/n_steps'] = log['n_steps']
        if 'n_micro_batches' in log:
            d['train/n_micro_batches'] = log['n_micro_batches']
        d.update({f'train/{k}': v for k, v in _clean_metric(log.get('metric')).items()})
    return d


def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f'{prefix}/{k}': v for k, v in metrics.items()}


def _greedy_eval_metrics(recs: List[Dict[str, Any]], ci: int, rounds: int
                         ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Aggregate the greedy holdout into SEAM ``mean@1`` metrics: overall + per-view acc,
    the frozen-baseline acc, and their lift -- all single-sample-per-problem means (no
    candidate averaging, no pass@k), so acc is directly comparable to SEAM's
    ``val-core/math/acc/mean@1`` (correctness only; format/leak not gated)."""
    def acc(rs: List[Dict[str, Any]]) -> float:
        return sum(1 for x in rs if x['withskill_correct']) / len(rs) if rs else 0.0
    def bacc(rs: List[Dict[str, Any]]) -> float:
        return sum(x['baseline_pass'] for x in rs) / len(rs) if rs else 0.0
    A = [x for x in recs if x['view'] == 'A']
    B = [x for x in recs if x['view'] == 'B']
    ws, base = acc(recs), bacc(recs)
    summary = {
        'record_type': 'eval_summary', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
        'n': len(recs), 'n_A': len(A), 'n_B': len(B),
        'acc_mean1': ws, 'baseline_acc_mean1': base, 'lift_mean1': ws - base,
        'acc_A_mean1': acc(A), 'acc_B_mean1': acc(B),
        'format_mean1': (sum(1 for x in recs if x['skill_parseable']) / len(recs)) if recs else 0.0,
        'term_mean1': (sum(1 for x in recs if x['withskill_terminated']) / len(recs)) if recs else 0.0,
    }
    metrics = {
        'core/math/acc/mean@1': ws,
        'core/math/baseline_acc/mean@1': base,
        'core/math/lift/mean@1': ws - base,
        'core/math/format/mean@1': summary['format_mean1'],
        'core/math/term/mean@1': summary['term_mean1'],
    }
    if A:
        metrics['core/math/acc_A/mean@1'] = summary['acc_A_mean1']
    if B:
        metrics['core/math/acc_B/mean@1'] = summary['acc_B_mean1']
    return summary, metrics


def _run_greedy_eval(base_sampler, skill_sampler,
                     eval_records: List[Dict[str, Any]], eval_cache: Dict[str, Dict[str, Any]],
                     ci: int, rounds: int, base_dp: int, skill_dp: int,
                     args: argparse.Namespace, checker=None,
                     diag_cache: Optional[Dict[str, str]] = None
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    """SEAM ``val-core/math/acc/mean@1`` analogue on the fixed holdout: ONE greedy skill per
    problem (T=0) injected into ONE greedy base solve (T=0), so acc is a single-sample
    pass@1 per problem averaged over problems. Each problem keeps its assigned view; view A
    still gets the rubric process-check, view B stays query-only -- the mixed A/B acc is the
    deployment number. No leak filter: like SEAM's val, acc scores correctness alone."""
    _baseline_rollout(base_sampler, eval_records, base_dp, args, eval_cache)  # frozen greedy baseline
    for r in eval_records:
        r['_view'] = _assign_view(r['problem'], args)
        r['_rubric_diag'] = ''
    _diagnose_views(checker, eval_records, args, diag_cache)
    sg_out = _run_samples(skill_sampler, [_view_prompt(r, args) for r in eval_records],
                          1, args.skill_max_tokens, skill_dp, temperature=0.0)
    skills = []
    for seqs in sg_out:
        resp = _clean_text(getattr(seqs[0], 'decoded', '') or '') if seqs else ''
        skills.append((_extract_skills_block(resp) or '', resp))
    ws_out = _run_samples(base_sampler,
                          [build_skill_solve_prompt(r['problem'], sk) for r, (sk, _) in zip(eval_records, skills)],
                          1, args.max_tokens, base_dp, temperature=0.0)
    recs = []
    for r, (sk, sresp), seqs in zip(eval_records, skills, ws_out):
        roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
        recs.append({
            'record_type': 'eval_problem', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
            'problem': r['problem'], 'reference_answer': r['reference_answer'],
            'view': r['_view'], 'rubric_diag': r.get('_rubric_diag', ''),
            'baseline_pass': r['_baseline_pass'],
            'skill': sk, 'skill_parseable': bool(sk), 'skill_response': sresp,
            'withskill_pred': roll['pred'], 'withskill_correct': roll['correct'],
            'withskill_terminated': roll['terminated'], 'withskill_stop_reason': roll['stop_reason'],
            'withskill_text': roll['text'],
        })
    summary, metrics = _greedy_eval_metrics(recs, ci, rounds)
    return recs, summary, metrics


def _validate_run_config(args: argparse.Namespace, records: List[Dict[str, Any]]) -> None:
    """Fail fast on configs that would SILENTLY hang the online sampler: _ProblemPool.draw(k)
    dedups within a call, so it never returns unless the pool holds >= chunk_size problems;
    a zero draw budget or chunk size yields empty chunks that never advance ``rounds``."""
    if not records:
        raise ValueError(f'loaded 0 {args.dataset} problems; check the dataset source')
    if args.chunk_size < 1:
        raise ValueError(f'--chunk-size must be >= 1 (got {args.chunk_size})')
    if args.eval_size < 0:
        raise ValueError(f'--eval-size must be >= 0 (got {args.eval_size})')
    if args.eval_size > 0 and args.eval_every < 1:
        raise ValueError(f'--eval-every must be >= 1 when eval is enabled (got {args.eval_every})')
    if len(records) < args.chunk_size:
        raise ValueError(f'--chunk-size ({args.chunk_size}) exceeds loaded problems '
                         f'({len(records)}); raise --n or lower --chunk-size')
    if args.balance_max_draws_mult < 1:
        raise ValueError(f'--balance-max-draws-mult must be >= 1 (got {args.balance_max_draws_mult})')
    if not 0.0 <= args.balance_success_frac <= 1.0:
        raise ValueError(f'--balance-success-frac must be in [0, 1] (got {args.balance_success_frac})')
    if not 0.0 <= args.balance_loop_frac <= 1.0:
        raise ValueError(f'--balance-loop-frac must be in [0, 1] (got {args.balance_loop_frac})')


def main() -> None:
    args = _build_args()
    if args.sft_batch_size % TRAIN_DP != 0:
        raise ValueError(f'--sft-batch-size ({args.sft_batch_size}) must be a multiple '
                         f'of the training dp size ({TRAIN_DP})')
    # LR schedule now follows chunk-level optimizer updates, not driver micro-batches.
    steps_per_round = 1
    records, eval_records, data_stats = _load_records(args)
    _validate_run_config(args, records)
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, 'skill_dataset.jsonl')
    gen_path = os.path.join(args.output_dir, 'gen_records.jsonl')
    eval_path = os.path.join(args.output_dir, 'eval_records.jsonl')
    train_log_path = os.path.join(args.output_dir, 'train_log.jsonl')

    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('OPENAI_API_KEY')):
        sys.stderr.write('[rft] WARNING: no LLM_BACKUP_API_KEY/OPENAI_API_KEY — '
                         'LeakVerifier will report no_llm and skip leak filtering\n')

    use_swan = swanlab is not None and os.environ.get('SWANLAB_MODE') != 'disabled'
    if use_swan:
        swanlab.init(project=args.swanlab_project,
                     experiment_name=(args.swanlab_exp or None),
                     config={'model': GEN_MODEL_ID, 'dataset': args.dataset,
                             'n': len(records), 'eval_n': len(eval_records),
                             'raw_loaded': data_stats['raw_loaded'],
                             'numeric_only': args.numeric_only,
                             'numeric_dropped': data_stats['numeric_dropped'],
                             'n_skills': args.n_skills, 'view_b_frac': args.view_b_frac,
                             'balance': args.balance,
                             'balance_success_frac': args.balance_success_frac,
                             'skill_gen_temp': args.skill_gen_temperature,
                             'grpo_epsilon': args.grpo_epsilon, 'lr': args.lr})

    # -- Device groups: train (FSDP2) + two independent vLLM samplers. --
    r0, r1, r2 = TRAIN_GPUS, TRAIN_GPUS + SKILL_SAMPLER_GPUS, NUM_GPUS
    device_groups = [
        DeviceGroup(name='train', ranks=list(range(0, r0)), device_type='GPU'),
        DeviceGroup(name='skill_sampler', ranks=list(range(r0, r1)), device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(r1, r2)), device_type='GPU'),
    ]
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups,
                       lazy_collect=False)

    # -- Skill model: full-param FSDP2, GRPO policy update. --
    train_mesh = DeviceMesh.from_sizes(world_size=TRAIN_GPUS, dp_size=TRAIN_DP, fsdp_size=TRAIN_FSDP)
    skill_model = TransformersModel(model_id=GEN_MODEL_ID, device_mesh=train_mesh,
                                    remote_group='train',
                                    ddp_config={'find_unused_parameters': False})
    from twinkle.patch.no_split_modules import NoSplitModulesPatch
    skill_model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    skill_model.set_template(Template, model_id=GEN_MODEL_ID,
                             enable_thinking=True, max_length=args.max_model_len,
                             truncation_strategy='delete')
    skill_model.set_processor(InputProcessor, padding_free=False)
    skill_model.set_loss('GRPOLoss', epsilon=args.grpo_epsilon)
    skill_model.set_optimizer('AdamW', lr=args.lr)
    skill_model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=10,
                                 num_training_steps=args.max_train_rounds * steps_per_round)

    # -- Two vLLM samplers: skill (synced) + base (frozen). --
    skill_dp, base_dp = SKILL_SAMPLER_GPUS, BASE_SAMPLER_GPUS
    skill_sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={'gpu_memory_utilization': GEN_GPU_MEM,
                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
        device_mesh=DeviceMesh.from_sizes(world_size=SKILL_SAMPLER_GPUS, dp_size=skill_dp),
        remote_group='skill_sampler')
    skill_sampler.set_template(Template, model_id=GEN_MODEL_ID,
                               enable_thinking=True, max_length=args.max_model_len)
    base_sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={'gpu_memory_utilization': GEN_GPU_MEM,
                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
        device_mesh=DeviceMesh.from_sizes(world_size=BASE_SAMPLER_GPUS, dp_size=base_dp),
        remote_group='base_sampler')
    base_sampler.set_template(Template, model_id=GEN_MODEL_ID,
                              enable_thinking=True, max_length=args.max_model_len)

    ckpt = CheckpointEngineManager(model=skill_model, sampler=skill_sampler)
    leak = LeakVerifier(sampler=None, answer_only=True)  # flag ONLY the final answer (view A only; view B skips leak)
    # leak = LeakVerifier(sampler=None, judge_system=_LEAK_JUDGE_SYSTEM)  # stricter: also flag concrete intermediate key results
    checker = _build_rubric_checker()                    # view-A process-check (teacher-only); None if no LLM backup
    if checker is None:
        sys.stderr.write('[rft] no LLM backup env -> view-A rubric process-check DISABLED '
                         '(skill-gen diagnoses from the attempt alone)\n')

    sys.stderr.write(f'[rft] raw={data_stats["raw_loaded"]} numeric_drop={data_stats["numeric_dropped"]} '
                     f'train={len(records)} eval={len(eval_records)} {args.dataset} problems; '
                     f'train_gpus={TRAIN_GPUS} skill_dp={skill_dp} base_dp={base_dp}\n')

    # -- Sequential: generate one chunk, train on it, sync -> exact on-policy GRPO.
    # Generation dominates wall-clock, so not overlapping training costs little, and
    # it removes all producer/consumer concurrency (no thread, no lock). --
    cfg = {'record_type': 'config', 'model': GEN_MODEL_ID, 'dataset': args.dataset,
           'n': len(records), 'eval_n': len(eval_records), 'seed': args.seed,
           'numeric_only': args.numeric_only,
           'raw_loaded': data_stats['raw_loaded'],
           'numeric_dropped': data_stats['numeric_dropped'],
           'eval_every': args.eval_every,
           'n_skills': args.n_skills,
           'view_b_frac': args.view_b_frac, 'skill_retries': args.skill_retries,
           'balance': args.balance, 'balance_success_frac': args.balance_success_frac,
           'balance_loop_frac': args.balance_loop_frac,
           'balance_max_draws_mult': args.balance_max_draws_mult,
           'skill_gen_temp': args.skill_gen_temperature,
           'skill_gen_top_p': args.skill_gen_top_p, 'skill_gen_top_k': args.skill_gen_top_k,
           'reward': 'greedy_binary(correct)', 'advantage': 'group_relative',
           'format_in_reward': args.format_in_reward,
           'rubric_check': 'fixed_math_5crit(viewA)' if checker else 'disabled',
           'grpo_epsilon': args.grpo_epsilon, 'lr': args.lr,
           'max_train_rounds': args.max_train_rounds, 'started': int(time.time())}
    hist: List[Dict[str, float]] = []
    rounds = 0
    pool = _ProblemPool(records, args.seed)
    eval_cache: Dict[str, Dict[str, Any]] = {}
    rubric_cache: Dict[str, str] = {}
    eval_rubric_cache: Dict[str, str] = {}
    with open(gen_path, 'w', encoding='utf-8') as gen_f, \
            open(eval_path, 'w', encoding='utf-8') as eval_f, \
            open(data_path, 'w', encoding='utf-8') as data_f, \
            open(train_log_path, 'w', encoding='utf-8') as tlog:
        for f in (gen_f, eval_f, data_f, tlog):
            f.write(json.dumps(cfg, ensure_ascii=False) + '\n')
            f.flush()
        gstep = 0
        # Each chunk is drawn fresh from the pool (which reshuffles + bumps epoch on every
        # full pass) and RE-GENERATED with the current (improved) policy, so every chunk
        # stays on-policy (no importance correction) -- the online analogue of SEAM's
        # fixed-data epochs. With --balance, _draw_chunk keeps drawing+baselining until the
        # base fail:success mix hits the target before this chunk is trained on.
        while rounds < args.max_train_rounds:
            chunk, balance = _draw_chunk(pool, base_sampler, base_dp, args)
            full, summary, groups = process_chunk(
                base_sampler, skill_sampler, leak, chunk, gstep, base_dp, skill_dp,
                args, checker, rubric_cache)
            summary['balance'] = balance

            log = None
            if groups:  # on-policy GRPO update on this chunk, then weights sync
                log = _train_chunk(skill_model, ckpt, groups, args)
                rounds += 1
                log.update({'record_type': 'train_round', 'round': rounds,
                            'chunk': gstep, 'epoch': pool.epoch, 'ts': int(time.time())})
                tlog.write(json.dumps(log, ensure_ascii=False) + '\n')
                tlog.flush()
                if rounds % args.save_rounds == 0:
                    skill_model.save(f'skill-rft-{rounds}', output_dir=args.output_dir)

            summary['rounds_done'], summary['epoch'] = rounds, pool.epoch
            for rec in full:
                gen_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            gen_f.write(json.dumps(summary, ensure_ascii=False) + '\n')
            gen_f.flush()
            for v in groups:
                data_f.write(json.dumps(v, ensure_ascii=False) + '\n')
            data_f.flush()

            sa, sb = summary['view_A'], summary['view_B']
            hist.append({'aA': sa['adoption_rate'], 'aB': sb['adoption_rate'],
                         'lift': summary['avg_lift'], 'pos': summary['n_reward_pos']})
            bal_str = (f'bal {balance["selected_fail"]}f/{balance["selected_success"]}s '
                       f'(loop {balance["selected_fail_loop"]} drew {balance["n_drawn"]}/'
                       f'fresh {balance["n_baseline_fresh"]}'
                       + ('!' if balance.get('budget_hit') else '') + ') '
                       ) if balance.get('enabled') else ''
            sys.stderr.write(
                f'[gen] e{pool.epoch} g{gstep}: {bal_str}hard={summary["n_hard"]} '
                f'clean={summary["n_clean"]} train={summary["n_train_samples"]} '
                f'(fail {summary["n_train_from_fail"]} adv%{summary["abs_adv_from_fail_frac"]:.2f}) '
                f'acc={summary["avg_baseline_pass_on_hard"]:.2f}->{summary["avg_withskill_pass"]:.2f} '
                f'lift={summary["avg_lift"]:+.3f} '
                f'A[{sa["n_hard"]}h {sa["adoption_rate"]:.2f}] '
                f'B[{sb["n_hard"]}h {sb["adoption_rate"]:.2f}] '
                f'rounds={rounds}'
                + (f' metric={log.get("metric")}' if log else '') + '\n')
            # -- per-query passk (base vs best/avg of N skills) + swanlab metrics --
            rows = _query_rows(full)
            for base_p, best_p, avg_p, nsc, prob in rows:
                logger.info(f'[q] g{gstep} base={base_p:.2f} bestN={best_p:.2f} avgN={avg_p:.2f} '
                            f'n={nsc} | {prob[:70].replace(chr(10), " ")}')
            if use_swan:
                swanlab.log(_swan_metrics(summary, log, rows), step=gstep)

            if eval_records and (gstep + 1) % args.eval_every == 0:
                eval_recs, eval_summary, eval_metrics = _run_greedy_eval(
                    base_sampler, skill_sampler, eval_records, eval_cache, gstep,
                    rounds, base_dp, skill_dp, args, checker, eval_rubric_cache)
                for rec in eval_recs:
                    eval_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                eval_f.write(json.dumps(eval_summary, ensure_ascii=False) + '\n')
                eval_f.flush()
                if use_swan:
                    swanlab.log(_prefix_metrics(eval_metrics, 'eval'), step=gstep)
                sys.stderr.write(
                    f'[eval] g{gstep}: n={eval_summary["n"]} mean@1 '
                    f'acc={eval_summary["baseline_acc_mean1"]:.3f}->{eval_summary["acc_mean1"]:.3f} '
                    f'lift={eval_summary["lift_mean1"]:+.3f} '
                    f'A[{eval_summary["n_A"]} {eval_summary["acc_A_mean1"]:.3f}] '
                    f'B[{eval_summary["n_B"]} {eval_summary["acc_B_mean1"]:.3f}] '
                    f'fmt={eval_summary["format_mean1"]:.2f} rounds={rounds}\n')

            if (gstep + 1) % args.trend_every == 0:
                tl = _trend_line(hist, args.trend_every, rounds)
                if tl:
                    sys.stderr.write(tl + '\n')
            gstep += 1

    skill_model.save('skill-rft-final', output_dir=args.output_dir)
    sys.stderr.write(f'[rft] done: {rounds} train rounds over {gstep} chunks / {pool.epoch} epochs; '
                     f'data -> {data_path}\n')


if __name__ == '__main__':
    main()
