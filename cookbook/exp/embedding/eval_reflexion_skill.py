"""Phase-0 measurement for the reflexion self-skill scheme (see reflexion.md).

Question this script answers: **on problems the base model first gets wrong, does
letting the SAME base model reflect on its failed attempt, distill a general
"skill", and re-solve WITH that skill in the system prompt, actually raise its
pass@k?** No LoRA is trained here — this is the upper-bound / go-no-go gate before
investing in a Skill-LoRA. If the base model's own skills don't help, training a
LoRA to produce them is pointless.

It deliberately reuses ``eval_gpqa_rag`` verbatim (dataset, grader, prompts, sampling
config) so numbers are comparable with the other AoPS lines. Only the base model +
one vLLM sampler are used; the dataset is AoPS; validation is on the SAME problem
(no similar-problem retrieval).

Per chunk of problems (all sampler calls are BATCHED across the whole chunk — never
one problem at a time):
  1. Initial solve — 1 rollout each; keep only problems the model got wrong.
  2. Skill generation — for each failed problem, the base model reads its own failed
     attempt and produces N candidate skills (general reminders, no answer/solution).
  3. Leak filter — drop skills that leak the gold answer or a full solution.
  4. Baseline pass@k — K rollouts of the plain problem (the "no-skill" control).
  5. With-skill pass@k — K rollouts of the problem with each surviving skill in the
     system prompt.
  6. Score — marginal = with-skill pass@k − baseline pass@k; keep the best skill.
A "pass" = answer correct AND generation terminated (no length cutoff).

Everything useful (failed attempt, every candidate skill + leak flag, baseline and
per-skill rollout stats, marginals, best skill) is written to a JSONL **incrementally
after each chunk**, so partial runs are fully analysable.

Launch (8 GPUs, tp=1 dp=8 by default):
    python cookbook/exp/embedding/eval_reflexion_skill.py --n 64 --chunk-size 16
"""
import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams as TwinkleSamplingParams
from twinkle.sampler import vLLMSampler
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

# Reuse the reference eval's dataset + grading + prompts + sampling config so this
# line is directly comparable with eval_gpqa_rag / eval_dualline_math.
from eval_gpqa_rag import (DIRECT_SYSTEM, GEN_GPU_MEM, GEN_GPUS, GEN_MODEL_ID,
                           GEN_TEMPERATURE, GEN_TOP_P, MCQ_INSTRUCTION, answers_match,
                           build_direct_prompt, extract_boxed, load_aops)

logger = get_logger()

# vLLM parallel: tp=1, dp=GEN_GPUS by default (override GEN_TP; keep GEN_GPUS=8).
GEN_TP = int(os.environ.get('GEN_TP', 1))

# Leak-detector API (reuses eval_gpqa_rag's env names). A strong external model
# judges whether a candidate skill leaks THIS problem's answer/solution — catching
# what the string filter cannot (multiple-choice letters, derived-result leakage).
LEAK_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
LEAK_BASE_URL = os.environ.get('COMPRESS_BASE_URL',
                               'https://dashscope.aliyuncs.com/compatible-mode/v1')
LEAK_API_MODEL = os.environ.get('COMPRESS_MODEL', 'qwen3.7-max')

# Global call spacer so bursts of leak-judge calls stay under the QPS limit.
_api_lock = threading.Lock()
_api_next = [0.0]


def _api_throttle(min_interval: float) -> None:
    with _api_lock:
        now = time.monotonic()
        wait = max(0.0, _api_next[0] - now)
        _api_next[0] = max(now, _api_next[0]) + min_interval
    if wait > 0:
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Prompts (self-reflection skill generation + skill-conditioned solving)
# ---------------------------------------------------------------------------
SKILL_GEN_SYSTEM = (
    "You are a meticulous mathematics coach. You are shown a competition problem and a "
    "student's FAILED attempt. Produce a SHORT list of general, reusable skills that "
    'would prevent this class of mistake on SIMILAR problems.\n\n'
    'OUTPUT FORMAT (strict):\n'
    '- You may reason briefly first, but the final answer MUST be a markdown bullet '
    'list of 3-5 items WRAPPED IN <skills> and </skills> tags. Output nothing after '
    '</skills>.\n'
    '- Each item is ONE short imperative sentence (a rule, check, or habit).\n'
    '- Inside the tags: no diagnosis narration, no "The student...", no headings, no '
    'restating the problem or the examples.\n\n'
    'CONTENT RULES (strict):\n'
    '- Do NOT reveal the final answer or the multiple-choice option.\n'
    '- Do NOT state the specific numbers, values, or key intermediate results of THIS '
    'problem.\n'
    '- Do NOT give a step-by-step solution to THIS problem. Every item must be GENERAL '
    'and transferable to other problems of the same type.\n\n'
    'Follow the example below for the exact tags, style, and level of generality.'
)

SKILL_GEN_USER = (
    'Problem:\n{problem}\n\n'
    "The student's failed attempt (it may be long or may fail to terminate):\n"
    '{attempt}\n\n'
    'Now output the skills bullet list.'
)

# One-shot demonstration of the required format and generality (answer-free).
_EX_PROBLEM = 'Simplify $\\sqrt{72} + \\sqrt{18}$ and give the result.'
_EX_ATTEMPT = (
    'The student added the radicands directly to get $\\sqrt{90}$ and concluded it '
    'could not be simplified, never factoring out the perfect squares first.')
_EX_SKILLS = (
    '<skills>\n'
    '- Before adding square roots, factor each radicand into a perfect square times a '
    'remainder and move the perfect-square root outside.\n'
    '- Never add radicands directly: $\\sqrt{a}+\\sqrt{b}\\ne\\sqrt{a+b}$.\n'
    '- Only combine radical terms after reducing them to the same simplest radical '
    'form.\n'
    '- Sanity-check the simplified result by estimating each root numerically.\n'
    '</skills>')


def build_skillgen_prompt(problem: str, attempt: str) -> Dict[str, Any]:
    return {'messages': [
        {'role': 'system', 'content': SKILL_GEN_SYSTEM},
        {'role': 'user',
         'content': SKILL_GEN_USER.format(problem=_EX_PROBLEM, attempt=_EX_ATTEMPT)},
        {'role': 'assistant', 'content': _EX_SKILLS},
        {'role': 'user',
         'content': SKILL_GEN_USER.format(problem=problem, attempt=attempt)},
    ]}


# The skill is injected into the SYSTEM prompt (per reflexion.md), on top of the
# exact DIRECT_SYSTEM used by the baseline so the only difference is the reminders.
# Built by concatenation (NOT str.format): DIRECT_SYSTEM and the skill may contain
# literal braces (e.g. ``\boxed{}``, LaTeX), which would break ``.format``.
_SKILL_SOLVE_PREFIX = (
    DIRECT_SYSTEM + '\n\n'
    'Before you start, keep these reminders in mind to avoid common mistakes on this '
    'type of problem:\n')
_SKILL_SOLVE_SUFFIX = (
    '\nApply them where relevant, but rely on your own reasoning to reach the answer.')


def build_skill_solve_prompt(problem: str, skill: str) -> Dict[str, Any]:
    return {'messages': [
        {'role': 'system', 'content': _SKILL_SOLVE_PREFIX + skill + _SKILL_SOLVE_SUFFIX},
        {'role': 'user', 'content': problem + MCQ_INSTRUCTION},
    ]}


# ---------------------------------------------------------------------------
# Parsing / grading / leak filtering
# ---------------------------------------------------------------------------
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


def _clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()


_BULLET_RE = re.compile(r'^\s*(?:[-*]|\d+[.)])\s')


def _extract_skill_list(text: str) -> str:
    """Pull just the clean skill list out of a (possibly thinking-laden) output.

    The model is instructed to wrap the final list in ``<skills>...</skills>``, so
    prefer that (robust to any preceding reasoning, closed or unterminated). Fall
    back to dropping a ``<think>`` block and keeping from the first bullet onward.
    """
    low = text.lower()
    if '<skills>' in low:
        start = low.index('<skills>') + len('<skills>')
        end = low.index('</skills>') if '</skills>' in low else len(text)
        return text[start:end].strip()
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1]
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _BULLET_RE.match(line):
            return '\n'.join(lines[i:]).strip()
    return text.strip()


def _bound_attempt(text: str, gen_tokens: int, budget_tokens: int) -> str:
    """Keep a failed attempt within the skill-gen context budget.

    Round-2 (skill-gen) input contains the FULL round-1 attempt, and failed
    attempts are often the ones that ran to the token cap (repetition loops), so
    feeding them verbatim overflows max_model_len. Keep the head (real reasoning +
    where it went wrong) plus a short tail (the final wrong answer); drop the
    redundant middle. Token->char conversion uses THIS attempt's observed
    chars-per-token so the cut fits precisely.
    """
    if not text or gen_tokens <= budget_tokens:
        return text
    cpt = len(text) / max(1, gen_tokens)
    head_tok = int(budget_tokens * 0.7)
    tail_tok = budget_tokens - head_tok
    head = text[:int(head_tok * cpt)]
    tail = text[-int(tail_tok * cpt):] if tail_tok > 0 else ''
    return f'{head}\n\n[... attempt truncated for length ...]\n\n{tail}'


def _parse_seq(seq, gold: str) -> Dict[str, Any]:
    """Turn one sampled sequence into a graded rollout record.

    ``pass`` requires BOTH a correct boxed answer AND clean termination (a length
    cutoff means the model never actually committed to the answer).
    """
    text = _clean_text(getattr(seq, 'decoded', '') or '')
    pred = extract_boxed(text)
    correct = bool(pred) and answers_match(pred, gold)
    terminated = getattr(seq, 'stop_reason', None) != 'length'
    return {
        'pred': pred,
        'correct': correct,
        'terminated': terminated,
        'passed': bool(correct and terminated),
        'stop_reason': getattr(seq, 'stop_reason', None),
        'gen_tokens': len(getattr(seq, 'tokens', None) or []),
        'text': text,
    }


def _pass_rate(rolls: List[Dict[str, Any]]) -> float:
    return sum(1 for r in rolls if r['passed']) / len(rolls) if rolls else 0.0


def _skill_leaks(skill: str, gold: str) -> Tuple[bool, str]:
    """Reject a skill that leaks the answer, or is a degenerate / non-list output."""
    if not skill.strip():
        return True, 'empty'
    bullets = [ln for ln in skill.splitlines() if _BULLET_RE.match(ln)]
    if len(bullets) < 2:
        return True, 'too_short'
    low = skill.lower()
    if 'item 1' in low and 'item 2' in low:  # model echoed the format placeholder
        return True, 'placeholder'
    if '\\boxed' in skill:
        return True, 'contains_boxed'
    g = (gold or '').strip()
    # Raw substring match is only trustworthy when the answer is specific enough that
    # an incidental hit is unlikely. Short answers ('D', 'E', '1') would match almost
    # any text, so leave those to the API judge instead of false-flagging every skill.
    if len(g) >= 4 and g.lower() in skill.lower():
        return True, 'contains_gold_answer'
    # Standalone multi-digit numbers from the gold answer leaking into the skill.
    for num in re.findall(r'-?\d{2,}', g):
        if re.search(r'(?<!\d)' + re.escape(num) + r'(?!\d)', skill):
            return True, f'contains_gold_number:{num}'
    return False, ''


# --- API leak judge: catches what the string filter cannot (MCQ letters, a
#     derived key result, or a near-complete solution laid out as a "skill"). ---
_LEAK_JUDGE_SYSTEM = (
    'You are a strict grader deciding whether a "skill" hint LEAKS the solution to a '
    'math problem. A skill should be a GENERAL, transferable reminder. It LEAKS if it '
    'does ANY of: reveal the final answer (a number, expression, or multiple-choice '
    'option); state a specific numeric/geometric result or key intermediate value of '
    'THIS problem; or give a derivation that essentially solves THIS problem. It does '
    'NOT leak if it only names general methods, common pitfalls, or checks. Reply with '
    'exactly one word: LEAK or CLEAN.'
)
_LEAK_JUDGE_USER = (
    'Problem:\n{problem}\n\nGold answer: {gold}\n\nSkill hint to check:\n{skill}\n\n'
    'Does the skill leak the answer or a full solution to THIS problem? Reply LEAK or '
    'CLEAN.'
)


def _api_leak_judge_one(api: OpenAIClient, problem: str, gold: str, skill: str,
                        min_interval: float, retries: int) -> Optional[bool]:
    """Return True (leak) / False (clean) / None (unparseable or API error after retries).

    Only transient API errors are retried (with exponential backoff); an unparseable
    verdict is deterministic at temperature 0, so retrying it is pointless.
    """
    msgs = [
        {'role': 'system', 'content': _LEAK_JUDGE_SYSTEM},
        {'role': 'user', 'content': _LEAK_JUDGE_USER.format(
            problem=problem[:4000], gold=gold, skill=skill[:4000])},
    ]
    for attempt in range(retries + 1):
        _api_throttle(min_interval)
        try:
            reply = api({'messages': msgs},
                        TwinkleSamplingParams(temperature=0.0, max_tokens=16),
                        extra_body={'enable_thinking': False})
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional
            logger.warning(f'[leak-judge] error (attempt {attempt + 1}/{retries + 1}): {exc}')
            if attempt < retries:
                time.sleep(min(4.0, 0.5 * 2 ** attempt))  # exponential backoff
                continue
            return None
        verdict = (reply.get('content') or '').strip().upper()
        if 'CLEAN' in verdict:
            return False
        if 'LEAK' in verdict:
            return True
        return None  # unparseable — deterministic at temp 0, no point retrying


def _api_leak_batch(api: OpenAIClient, items: List[Tuple[int, str, str, str]],
                    concurrency: int, min_interval: float,
                    retries: int) -> Dict[int, Optional[bool]]:
    """Judge many (key, problem, gold, skill) tuples in parallel; key -> verdict."""
    verdicts: Dict[int, Optional[bool]] = {}
    if not items:
        return verdicts
    with ThreadPoolExecutor(max_workers=min(len(items), concurrency)) as pool:
        futs = {pool.submit(_api_leak_judge_one, api, p, g, s, min_interval, retries): k
                for (k, p, g, s) in items}
        for fut in as_completed(futs):
            verdicts[futs[fut]] = fut.result()
    return verdicts


# ---------------------------------------------------------------------------
# Batched sampling (one shared sampler.sample per phase — never per problem)
# ---------------------------------------------------------------------------
def _pad_for_dp(prompts: List[Any], gen_dp: int) -> List[Any]:
    """vLLM dp needs batch len >= dp; pad tail rounds and let the caller slice back."""
    if gen_dp <= 1 or not prompts or len(prompts) >= gen_dp:
        return prompts
    pad = [copy.deepcopy(prompts[-1]) for _ in range(gen_dp - len(prompts))]
    return prompts + pad


def _run_samples(sampler, prompts: List[Any], num_samples: int, max_tokens: int,
                 gen_dp: int) -> List[List[Any]]:
    """One batched sampler call; return per-prompt list of raw sampled sequences."""
    if not prompts:
        return []
    params = TwinkleSamplingParams(
        max_tokens=max_tokens, temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
        num_samples=num_samples)
    padded = _pad_for_dp(prompts, gen_dp)
    responses = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in responses]


def _set_thinking(sampler, args: argparse.Namespace, enabled: bool) -> None:
    """Toggle the remote template's thinking mode.

    Skill generation wants thinking OFF so the model emits the short ``<skills>``
    list directly (with thinking ON it burns the token budget reasoning and often
    never reaches the list); solving wants it ON. ``set_template`` is a
    remote_function, so this propagates to every sampler worker.
    """
    sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                         enable_thinking=enabled, max_length=args.max_model_len)


def _bounded_attempt_for(r: Dict[str, Any], args: argparse.Namespace) -> str:
    """Per-problem bound so problem + attempt + skill output fits the context window."""
    prob_est = len(r['problem']) // 2  # conservative problem token estimate
    budget = max(1024, args.max_model_len - args.skill_max_tokens
                 - args.attempt_reserve_tokens - prob_est)
    return _bound_attempt(r['_init'][0]['text'], r['_init'][0]['gen_tokens'], budget)


def _filter_candidates(api: Optional[OpenAIClient],
                       cands: List[Tuple[Dict[str, Any], str]],
                       args: argparse.Namespace) -> None:
    """Apply string + API leak filters to (problem, skill) candidates; append results.

    The cheap string filter runs first; the API judge only sees skills that pass it,
    which is what catches MCQ-letter / derived-result / full-solution leakage.
    """
    prepared = []  # [r, text, leaked(bool|None), reason]
    for r, text in cands:
        leaked, reason = _skill_leaks(text, r['reference_answer'])
        prepared.append([r, text, True if leaked else None, reason])
    if api is not None:
        items = [(i, prepared[i][0]['problem'], prepared[i][0]['reference_answer'],
                  prepared[i][1]) for i in range(len(prepared)) if prepared[i][2] is None]
        verdicts = _api_leak_batch(api, items, args.api_concurrency, args.api_min_interval,
                                   args.api_retries)
        for key, _p, _g, _s in items:
            v = verdicts.get(key)
            if v is True:
                prepared[key][2], prepared[key][3] = True, 'api_leak'
            elif v is False:
                prepared[key][2], prepared[key][3] = False, ''
            else:
                prepared[key][2], prepared[key][3] = False, 'api_uncertain'
    for r, text, leaked, reason in prepared:
        r['_skills'].append({'skill': text, 'leaked': bool(leaked), 'leak_reason': reason})


def _build_skills(sampler, api: Optional[OpenAIClient], failed: List[Dict[str, Any]],
                  gen_dp: int, args: argparse.Namespace) -> None:
    """Generate + extract + leak-filter skills, re-rolling problems short on clean ones.

    Clean skills accumulate across rounds; only problems still below ``min_survivors``
    clean skills are re-rolled, up to ``skill_retries`` extra rounds.
    """
    for r in failed:
        r['_skills'] = []
    todo = list(failed)
    _set_thinking(sampler, args, False)  # skill-gen: emit the list directly, no CoT
    try:
        for _ in range(args.skill_retries + 1):
            if not todo:
                break
            sg_out = _run_samples(
                sampler,
                [build_skillgen_prompt(r['problem'], _bounded_attempt_for(r, args)) for r in todo],
                args.n_skills, args.skill_max_tokens, gen_dp)
            cands = [(r, _extract_skill_list(_clean_text(getattr(s, 'decoded', '') or '')))
                     for r, seqs in zip(todo, sg_out) for s in seqs]
            _filter_candidates(api, cands, args)
            todo = [r for r in failed
                    if sum(1 for sk in r['_skills'] if not sk['leaked']) < args.min_survivors]
    finally:
        _set_thinking(sampler, args, True)  # restore for solving phases
    tot = sum(len(r['_skills']) for r in failed)
    leaked = sum(1 for r in failed for sk in r['_skills'] if sk['leaked'])
    sys.stderr.write(f'  phase2: skills={tot} leaked={leaked} '
                     f'({leaked / max(1, tot):.0%}); {len(todo)} still short of '
                     f'{args.min_survivors} clean\n')


# ---------------------------------------------------------------------------
# Per-chunk pipeline
# ---------------------------------------------------------------------------
def process_chunk(sampler, api: Optional[OpenAIClient], chunk: List[Dict[str, Any]],
                  gen_dp: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run all 6 phases for one chunk (batched) and return per-problem records."""
    # --- Phase 1: initial solve, keep only the ones the model got wrong. ---
    init_out = _run_samples(
        sampler, [build_direct_prompt(r['problem']) for r in chunk],
        args.init_samples, args.max_tokens, gen_dp)
    for r, seqs in zip(chunk, init_out):
        r['_init'] = [_parse_seq(s, r['reference_answer']) for s in seqs]
        r['_init_pass'] = _pass_rate(r['_init'])
        r['_failed'] = r['_init_pass'] == 0.0
    failed = [r for r in chunk if r['_failed']]
    sys.stderr.write(f'  phase1: {len(chunk)-len(failed)}/{len(chunk)} solved on '
                     f'first try, {len(failed)} failed -> reflect\n')

    if failed:
        # --- Baseline pass@k FIRST: defines which failures are genuinely hard. ---
        # (A single initial rollout is noisy; an easy problem can fail phase 1 yet
        # have a high pass@k, so measure the marginal only on truly hard problems.)
        base_out = _run_samples(
            sampler, [build_direct_prompt(r['problem']) for r in failed],
            args.pass_k, args.max_tokens, gen_dp)
        for r, seqs in zip(failed, base_out):
            r['_baseline'] = [_parse_seq(s, r['reference_answer']) for s in seqs]
            r['_baseline_pass'] = _pass_rate(r['_baseline'])
            r['_hard'] = r['_baseline_pass'] <= args.hard_baseline_max
            r['_skills'] = []
            r['_best'] = None
        hard = [r for r in failed if r['_hard']]
        sys.stderr.write(f'  baseline: {len(hard)}/{len(failed)} failures are hard '
                         f'(pass@{args.pass_k} <= {args.hard_baseline_max})\n')

        if hard:
            # --- Skills (generate + leak filter + re-rollout) for HARD problems only. ---
            _build_skills(sampler, api, hard, gen_dp, args)

            # --- With-skill pass@k (flatten hard-problem x surviving skill). ---
            flat: List[Tuple[int, int]] = []
            ws_prompts: List[Any] = []
            for ri, r in enumerate(hard):
                for si, sk in enumerate(r['_skills']):
                    if sk['leaked'] or not sk['skill'].strip():
                        continue
                    flat.append((ri, si))
                    ws_prompts.append(build_skill_solve_prompt(r['problem'], sk['skill']))
            ws_out = _run_samples(sampler, ws_prompts, args.pass_k, args.max_tokens, gen_dp)
            for (ri, si), seqs in zip(flat, ws_out):
                r = hard[ri]
                sk = r['_skills'][si]
                sk['rolls'] = [_parse_seq(s, r['reference_answer']) for s in seqs]
                sk['with_pass'] = _pass_rate(sk['rolls'])
                sk['marginal'] = sk['with_pass'] - r['_baseline_pass']

            # --- Pick the best (highest marginal) surviving skill per hard problem. ---
            for r in hard:
                scored = [sk for sk in r['_skills'] if 'marginal' in sk]
                r['_best'] = max(scored, key=lambda s: s['marginal']) if scored else None

    return [_make_record(r, args) for r in chunk]


def _roll_summary(roll: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = {k: roll[k] for k in ('pred', 'correct', 'terminated', 'passed',
                                'stop_reason', 'gen_tokens')}
    if args.store_rollout_text:
        out['text'] = roll['text'][:args.store_rollout_chars]
    return out


def _make_record(r: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble the incremental JSONL record for one problem (solved or failed)."""
    rec: Dict[str, Any] = {
        'problem': r['problem'],
        'reference_answer': r['reference_answer'],
        'tags': r.get('tags', []),
        'failed_first_try': r['_failed'],
        'init_pass_rate': r['_init_pass'],
        'init_attempt': {
            'text': r['_init'][0]['text'][:args.store_init_chars],
            'pred': r['_init'][0]['pred'],
            'stop_reason': r['_init'][0]['stop_reason'],
            'gen_tokens': r['_init'][0]['gen_tokens'],
        },
    }
    if not r['_failed']:
        return rec

    best = r.get('_best')
    rec['baseline_pass'] = r['_baseline_pass']
    # Genuinely hard = low baseline pass@k; only these count in the marginal stats.
    rec['is_hard'] = bool(r.get('_hard'))
    rec['baseline_rolls'] = [_roll_summary(x, args) for x in r['_baseline']]
    rec['skills'] = [{
        'skill': sk['skill'],
        'leaked': sk['leaked'],
        'leak_reason': sk['leak_reason'],
        'with_pass': sk.get('with_pass'),
        'marginal': sk.get('marginal'),
        'rolls': [_roll_summary(x, args) for x in sk.get('rolls', [])],
    } for sk in r.get('_skills', [])]
    rec['best_skill'] = best['skill'] if best else None
    rec['best_marginal'] = best['marginal'] if best else None
    rec['best_with_pass'] = best['with_pass'] if best else None
    # "rescued" = a leak-free skill turned a fully-failing problem into some passes.
    rec['rescued'] = bool(best and r['_baseline_pass'] == 0.0 and best['with_pass'] > 0.0)
    rec['helped'] = bool(best and best['marginal'] > 0.0)
    return rec


# ---------------------------------------------------------------------------
# Running summary
# ---------------------------------------------------------------------------
def _update_summary(summ: Dict[str, Any], recs: List[Dict[str, Any]]) -> None:
    for rec in recs:
        summ['n_total'] += 1
        if not rec['failed_first_try']:
            summ['n_solved_first'] += 1
            continue
        summ['n_failed'] += 1
        if not rec.get('is_hard'):
            summ['n_failed_easy'] += 1  # failed phase 1 but easy on pass@k — excluded
            continue
        summ['n_hard'] += 1
        base = rec.get('baseline_pass', 0.0)
        summ['sum_baseline_pass'] += base
        if rec.get('best_marginal') is not None:
            summ['n_with_skill'] += 1
            summ['sum_best_with_pass'] += rec.get('best_with_pass', 0.0)
            summ['sum_best_marginal'] += rec.get('best_marginal', 0.0)
        else:
            # No clean skill produced for this hard problem -> skill adds no gain
            # (count it honestly as marginal 0 rather than dropping it from the average).
            summ['sum_best_with_pass'] += base
        summ['n_helped'] += int(rec.get('helped', False))
        summ['n_rescued'] += int(rec.get('rescued', False))


def _summary_report(summ: Dict[str, Any]) -> Dict[str, Any]:
    nh = max(1, summ['n_hard'])
    return {
        'record_type': 'summary',
        'n_total': summ['n_total'],
        'n_solved_first_try': summ['n_solved_first'],
        'n_failed_first_try': summ['n_failed'],
        'n_failed_but_easy': summ['n_failed_easy'],
        'n_hard': summ['n_hard'],
        'n_hard_with_skill': summ['n_with_skill'],
        # Averages are over ALL hard problems; a hard problem with no clean skill
        # counts as zero gain (with_pass == baseline), so with - base == marginal.
        'avg_baseline_pass_on_hard': summ['sum_baseline_pass'] / nh,
        'avg_best_with_skill_pass_on_hard': summ['sum_best_with_pass'] / nh,
        'avg_best_marginal_on_hard': summ['sum_best_marginal'] / nh,
        'n_helped_by_skill': summ['n_helped'],
        'n_rescued_from_zero': summ['n_rescued'],
        'frac_hard_helped': summ['n_helped'] / nh,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=64, help='AoPS problems to sample.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--chunk-size', type=int, default=16,
                   help='Problems per chunk. All sampler calls within a chunk are '
                        'batched; results are flushed to disk after each chunk.')
    p.add_argument('--init-samples', type=int, default=1,
                   help='Rollouts for the initial solve. A problem is "failed" (and '
                        'sent to reflection) only if all initial rollouts are wrong.')
    p.add_argument('--n-skills', type=int, default=8,
                   help='Candidate skills generated per failed problem.')
    p.add_argument('--pass-k', type=int, default=8,
                   help='Rollouts per (baseline / with-skill) pass@k estimate.')
    p.add_argument('--hard-baseline-max', type=float, default=0.25,
                   help='A failed problem counts as "hard" (included in the marginal '
                        'stats) only if its baseline pass@k <= this. Filters out easy '
                        'problems that merely failed the single initial rollout.')
    p.add_argument('--max-model-len', type=int, default=30000,
                   help='Context window (engine + template). MUST exceed --max-tokens: '
                        'the round-2 skill-gen input holds the full round-1 attempt '
                        'plus the problem.')
    p.add_argument('--max-tokens', type=int, default=20000,
                   help='Max generated tokens for solving rollouts (round-1 output cap).')
    p.add_argument('--skill-max-tokens', type=int, default=2048,
                   help='Max tokens for skill generation. Enough to finish any thinking '
                        'and emit the short bullet list (which is then extracted).')
    p.add_argument('--attempt-reserve-tokens', type=int, default=2048,
                   help='Tokens reserved for system prompt + wrappers when bounding the '
                        'failed attempt fed into skill generation (the problem length '
                        'is accounted for separately, per-problem).')
    p.add_argument('--min-survivors', type=int, default=2,
                   help='Re-roll a problem\'s skills if fewer than this many survive the '
                        'leak filters.')
    p.add_argument('--skill-retries', type=int, default=1,
                   help='Max extra skill-generation rounds for problems short on clean '
                        'skills (0 = no retry).')
    p.add_argument('--api-concurrency', type=int, default=32,
                   help='Parallel workers for the API leak judge (max 32 recommended).')
    p.add_argument('--api-min-interval', type=float, default=0.1,
                   help='Minimum seconds between API leak-judge calls (QPS guard).')
    p.add_argument('--api-retries', type=int, default=3,
                   help='Retries on transient API errors per leak-judge call (exponential '
                        'backoff); only after these are exhausted is a skill kept as '
                        'api_uncertain.')
    p.add_argument('--disable-api-leak', action='store_true',
                   help='Skip the API leak judge even if COMPRESS_API_KEY is set '
                        '(string filter only).')
    p.add_argument('--output', default='./output/reflexion_phase0/aops_results.jsonl')
    p.add_argument('--store-init-chars', type=int, default=8000,
                   help='Truncate the stored failed-attempt text to this many chars.')
    p.add_argument('--store-rollout-text', action='store_true',
                   help='Also store (truncated) text of every rollout, not just stats.')
    p.add_argument('--store-rollout-chars', type=int, default=2000)
    args = p.parse_args()

    records = load_aops(n=args.n, seed=args.seed)
    sys.stderr.write(f'[reflexion] {len(records)} AoPS problems, chunk={args.chunk_size}, '
                     f'init_samples={args.init_samples}, n_skills={args.n_skills}, '
                     f'pass_k={args.pass_k}, max_tokens={args.max_tokens}\n')

    # --- 8-GPU vLLM sampler (tp=GEN_TP, dp=GEN_GPUS/GEN_TP). ---
    if GEN_GPUS % GEN_TP != 0:
        raise ValueError(f'GEN_GPUS ({GEN_GPUS}) must be divisible by GEN_TP ({GEN_TP})')
    gen_dp = GEN_GPUS // GEN_TP
    gen_mesh = DeviceMesh.from_sizes(world_size=GEN_GPUS, dp_size=gen_dp, tp_size=GEN_TP)
    twinkle.initialize(
        mode='ray', nproc_per_node=GEN_GPUS,
        groups=[DeviceGroup(name='sampler', ranks=list(range(GEN_GPUS)),
                            device_type='GPU', gpus_per_worker=GEN_TP)],
        lazy_collect=False)
    sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={'gpu_memory_utilization': GEN_GPU_MEM,
                     'max_model_len': args.max_model_len,
                     'tensor_parallel_size': GEN_TP},
        device_mesh=gen_mesh, remote_group='sampler')
    sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                         enable_thinking=True, max_length=args.max_model_len)
    sys.stderr.write(f'[reflexion] sampler ready (model={GEN_MODEL_ID}, tp={GEN_TP}, '
                     f'dp={gen_dp})\n')

    # --- API leak judge (optional): reuses eval_gpqa_rag's COMPRESS_* env. ---
    api: Optional[OpenAIClient] = None
    if LEAK_API_KEY and not args.disable_api_leak:
        api = OpenAIClient(model=LEAK_API_MODEL, api_key=LEAK_API_KEY,
                           base_url=LEAK_BASE_URL)
        sys.stderr.write(f'[reflexion] leak judge ON via API model={LEAK_API_MODEL} '
                         f'(concurrency={args.api_concurrency})\n')
    else:
        sys.stderr.write('[reflexion] leak judge OFF (string filter only) — set '
                         'COMPRESS_API_KEY to enable the API judge\n')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    summ = {k: 0 for k in ('n_total', 'n_solved_first', 'n_failed', 'n_failed_easy',
                           'n_hard', 'n_with_skill', 'n_helped', 'n_rescued')}
    summ.update({'sum_baseline_pass': 0.0, 'sum_best_with_pass': 0.0,
                 'sum_best_marginal': 0.0})

    with open(args.output, 'w', encoding='utf-8') as out_f:
        # Line 1: run config, for reproducibility / later analysis.
        out_f.write(json.dumps({
            'record_type': 'config', 'model': GEN_MODEL_ID, 'dataset': 'aops',
            'n': len(records), 'seed': args.seed, 'init_samples': args.init_samples,
            'n_skills': args.n_skills, 'pass_k': args.pass_k,
            'hard_baseline_max': args.hard_baseline_max,
            'max_model_len': args.max_model_len, 'max_tokens': args.max_tokens,
            'skill_max_tokens': args.skill_max_tokens,
            'api_leak_judge': api is not None,
            'api_leak_model': LEAK_API_MODEL if api is not None else None,
            'min_survivors': args.min_survivors, 'skill_retries': args.skill_retries,
            'gpus': GEN_GPUS, 'tp': GEN_TP, 'started': int(time.time()),
        }, ensure_ascii=False) + '\n')
        out_f.flush()

        n_chunks = (len(records) + args.chunk_size - 1) // args.chunk_size
        for ci in range(n_chunks):
            chunk = records[ci * args.chunk_size:(ci + 1) * args.chunk_size]
            sys.stderr.write(f'[reflexion] chunk {ci+1}/{n_chunks} ({len(chunk)} problems)\n')
            recs = process_chunk(sampler, api, chunk, gen_dp, args)
            for rec in recs:                       # incremental write per problem
                out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out_f.flush()
            _update_summary(summ, recs)
            rep = _summary_report(summ)
            sys.stderr.write(
                f'  running: failed={rep["n_failed_first_try"]} '
                f'(easy={rep["n_failed_but_easy"]}) hard={rep["n_hard"]} '
                f'base_pass={rep["avg_baseline_pass_on_hard"]:.3f} '
                f'skill_pass={rep["avg_best_with_skill_pass_on_hard"]:.3f} '
                f'helped={rep["n_helped_by_skill"]} rescued={rep["n_rescued_from_zero"]}\n')

        report = _summary_report(summ)
        out_f.write(json.dumps(report, ensure_ascii=False) + '\n')
        out_f.flush()

    print('\n' + '=' * 60)
    print(f'Reflexion Phase-0 — model={GEN_MODEL_ID}, dataset=aops, n={report["n_total"]}')
    print('=' * 60)
    print(f'solved on first try      : {report["n_solved_first_try"]}/{report["n_total"]}')
    print(f'failed first try         : {report["n_failed_first_try"]} '
          f'(easy, excluded: {report["n_failed_but_easy"]})')
    print(f'hard (baseline pass@{args.pass_k}<= {args.hard_baseline_max}) : {report["n_hard"]}')
    print(f'  avg baseline pass@{args.pass_k:<2}   : {report["avg_baseline_pass_on_hard"]:.4f}')
    print(f'  avg best-skill pass@{args.pass_k:<2} : {report["avg_best_with_skill_pass_on_hard"]:.4f}')
    print(f'  avg best marginal      : {report["avg_best_marginal_on_hard"]:+.4f}')
    print(f'  helped by a skill      : {report["n_helped_by_skill"]}/{report["n_hard"]}')
    print(f'  rescued from 0 pass    : {report["n_rescued_from_zero"]}/{report["n_hard"]}')
    print(f'\n[output] {args.output}')


if __name__ == '__main__':
    main()
