"""Re-annotate HotpotQA ground truth using a super-LLM to ensure correctness.

The original HotpotQA dataset has annotation issues:
  - GT doesn't match the question type (asks "where", GT gives a name)
  - Partial/incomplete answers for multi-hop questions
  - Single form when multiple valid forms exist (e.g. "2" vs "two")
  - Question itself malformed (wrong question word, truncation, presupposition
    mismatch with the answer type)

This script:
  1. Loads HotpotQA fullwiki train split.
  2. By default (--only-forced), re-annotates ONLY the IDs listed in
     wrong_ids.txt (the 340 known-bad cases).
     Pass --no-only-forced to fall back to stratified 3000-per-level sampling
     with wrong_ids force-included.
  3. For each row, sends question + full context + original GT to a super-LLM.
  4. The LLM emits one of four verdicts and (when applicable) a multi-form
     answer list and/or a repaired question:
       - keep:         original Q + A are both correct
       - fix_answer:   Q is fine; A is wrong/incomplete
       - fix_question: Q is malformed but repairable into a well-formed Q
                       that the same passages answer with the same gold facts
       - drop:         Q cannot be repaired without changing the fact, OR
                       passages do not support any answer
  5. Outputs ONE JSONL file containing all rows (including drop). Each row has
     verdict, question, question_fixed, answers, reasoning. Downstream filters
     by verdict.

Run (re-clean wrong_ids.txt only, default):
    python reannotate_groundtruth.py \
        --model qwen-max --api-key $OPENAI_API_KEY \
        --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --output hotpotqa_reannotated_wrong.jsonl --concurrency 16
"""
import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI


VERIFY_SYSTEM = """You are a dataset quality auditor for a multi-hop QA benchmark (HotpotQA).

Given a Question, supporting Context passages, and the dataset's Original Answer, output ONE of four verdicts and a multi-form answer list grounded in the passages.

VERDICTS
- "keep":          original question + original answer are both correct.
- "fix_answer":    question is fine; original answer is wrong/incomplete.
- "fix_question":  question is malformed (wrong question word, broken grammar, truncated, or presupposition mismatch with the answer type) but can be REPAIRED into a well-formed question that the SAME passages answer with the SAME gold facts.
- "drop":          question cannot be repaired without changing the underlying fact, OR the passages do not support any answer.

MULTI-FORM ANSWER RULES (apply to keep / fix_answer / fix_question)
1. Output ALL acceptable surface forms whenever applicable:
   - Number variants: arabic + english word + hyphen-prefix form (e.g. "3", "three", "three-door", "3-door")
   - Range variants: start, end, and full range string (e.g. "1901", "1902", "1901-1902", "1901-2")
   - Location variants: city / state-or-province / country (e.g. "Everett", "Washington", "WA", "United States")
   - Person variants: legal name / nickname / full name (e.g. "Allan", "Heywood", "Allan Stewart Konigsberg")
   - Entity-role pairs for role-of-X questions: BOTH the role AND the entity (e.g. "chauffeur", "Hitler's chauffeur")
   - Show-vs-character pairs for best-known-for questions: BOTH the show AND the character (e.g. "M*A*S*H", "Major Frank Burns")
   - Common abbreviations (e.g. "NYC", "New York City", "New York")
   - With/without titles (e.g. "Dr. Smith", "Smith")
   - Different date formats if applicable (e.g. "July 4, 1776", "4 July 1776")
2. Each answer is SHORT (a name, entity, number, date, or yes/no).
3. yes/no answers MUST be lowercase ["yes"] or ["no"].
4. Do NOT hallucinate. Every answer must be grounded in the provided passages.

QUESTION REWRITE RULES (verdict = fix_question)
1. question_fixed MUST be answerable by the SAME passages and yield the SAME factual answer as the original gold facts.
2. Allowed edits: swap question word (Where -> Did / Who / What), repair grammar, complete truncation, align question word with the answer type.
3. FORBIDDEN: changing intent, injecting the answer into the question, adding facts not in the passages.
4. If you cannot satisfy these constraints, downgrade to "drop".

DROP RULES (verdict = drop)
- answers MUST be [] and question_fixed MUST be null.

OUTPUT FORMAT (JSON only, no markdown fence, no explanation)
{"verdict": "keep|fix_answer|fix_question|drop", "question_fixed": "..." | null, "answers": ["..."], "reasoning": "one sentence"}"""

VERIFY_USER = """## Question
{question}

## Original Answer (may be wrong)
{original_answer}

## Supporting Passages
{context}

## Task
Audit the row per the system rules. Pick exactly one verdict (keep / fix_answer / fix_question / drop), produce the multi-form answers list (or [] for drop), and write a one-sentence reasoning. If verdict=fix_question, also produce question_fixed; otherwise set it to null.
Return a single JSON object only."""


LEVELS: Tuple[str, str, str] = ('easy', 'medium', 'hard')


def _format_context(context: Dict[str, Any]) -> str:
    titles = context.get('title', []) or []
    sentences = context.get('sentences', []) or []
    lines = []
    for i, (title, sents) in enumerate(zip(titles, sentences), start=1):
        if isinstance(sents, list):
            body = ' '.join(s.strip() for s in sents if s and s.strip())
        else:
            body = str(sents).strip()
        lines.append(f'[{i}] {title}: {body}')
    return '\n\n'.join(lines)


_JSON_RE = re.compile(r'\{[^{}]*"verdict"\s*:\s*"[^"]+"[^{}]*"answers"\s*:\s*\[.*?\][^{}]*\}', re.DOTALL)

_VALID_VERDICTS = ('keep', 'fix_answer', 'fix_question', 'drop')


def _parse_response(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith('```'):
        first_nl = text.find('\n')
        last_fence = text.rfind('```')
        if first_nl != -1 and last_fence > first_nl:
            text = text[first_nl + 1:last_fence].strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'answers' in obj:
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _validate_verdict(
    verdict: Optional[str], answers: List[str],
    qfix: Optional[str], original_question: str,
) -> bool:
    if verdict not in _VALID_VERDICTS:
        return False
    if verdict == 'drop':
        return not answers and qfix is None
    if not answers:
        return False
    if verdict == 'fix_question':
        return bool(qfix) and qfix.strip() != original_question.strip()
    return qfix is None


def verify_answer(
    api: OpenAI, model: str, row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    question = row['question']
    original_answer = row.get('answer', '') or ''
    context_str = _format_context(row.get('context', {}) or {})

    user_content = VERIFY_USER.format(
        question=question,
        original_answer=original_answer,
        context=context_str)

    trajectory = {
        'messages': [
            {'role': 'system', 'content': VERIFY_SYSTEM},
            {'role': 'user', 'content': user_content},
        ]
    }
    sp = SamplingParams(temperature=0.1, max_tokens=512)

    for attempt in range(3):
        try:
            reply = api(trajectory, sp, extra_body={'enable_thinking': True})
        except Exception as exc:
            sys.stderr.write(f'[verify] {row["id"]}: API error: {exc}\n')
            if attempt < 2:
                continue
            return None

        content = reply.get('content') or ''
        parsed = _parse_response(content)
        if parsed:
            verdict = parsed.get('verdict')
            answers_raw = parsed.get('answers')
            answers = (
                [str(a).strip() for a in answers_raw if str(a).strip()]
                if isinstance(answers_raw, list) else [])
            qfix_raw = parsed.get('question_fixed')
            qfix = (qfix_raw.strip() or None) if isinstance(qfix_raw, str) else None
            if _validate_verdict(verdict, answers, qfix, question):
                return {
                    'id': row['id'],
                    'verdict': verdict,
                    'question': question,
                    'question_fixed': qfix,
                    'original_answer': original_answer,
                    'answers': answers,
                    'reasoning': parsed.get('reasoning', ''),
                    'level': row.get('level', ''),
                    'type': row.get('type', ''),
                    'context': row.get('context', {}),
                    'supporting_facts': row.get('supporting_facts', {}),
                }
        sys.stderr.write(
            f'[verify retry {attempt+1}] {row["id"]}: '
            f'parse failed, content={content[:200]!r}\n')

    sys.stderr.write(f'[verify drop] {row["id"]}: all attempts failed\n')
    return None


def stratified_sample_with_forced(
    ds, per_level: Dict[str, int], forced_ids: frozenset, seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {lv: [] for lv in LEVELS}
    forced_indices: List[int] = []
    forced_levels: Dict[str, int] = {lv: 0 for lv in LEVELS}

    for i in range(len(ds)):
        row_id = ds[i]['id']
        level = (ds[i].get('level') or '').strip().lower()
        if row_id in forced_ids:
            forced_indices.append(i)
            if level in forced_levels:
                forced_levels[level] += 1
        elif level in buckets:
            buckets[level].append(i)

    picked_set = set(forced_indices)
    for lv in LEVELS:
        need = max(0, per_level[lv] - forced_levels[lv])
        pool = [idx for idx in buckets[lv] if idx not in picked_set]
        if len(pool) < need:
            sys.stderr.write(
                f'Warning: level={lv} has {len(pool)} available, need {need}\n')
            need = len(pool)
        sampled = rng.sample(pool, need)
        picked_set.update(sampled)

    picked = sorted(picked_set)
    rng.shuffle(picked)
    return [ds[int(i)] for i in picked]


def select_forced_only(ds, forced_ids: frozenset, seed: int) -> List[Dict[str, Any]]:
    """Pick exactly the rows whose id is in forced_ids; warn on missing."""
    indices: List[int] = []
    found: set = set()
    for i in range(len(ds)):
        rid = ds[i]['id']
        if rid in forced_ids:
            indices.append(i)
            found.add(rid)
    missing = forced_ids - found
    if missing:
        sys.stderr.write(
            f'Warning: {len(missing)} forced ids not found in dataset, '
            f'e.g. {sorted(missing)[:5]}\n')
    rng = random.Random(seed)
    rng.shuffle(indices)
    return [ds[int(i)] for i in indices]


def load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get('id')
            if rid:
                done.add(rid)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--total', type=int, default=12000)
    parser.add_argument('--easy', type=int, default=2000)
    parser.add_argument('--medium', type=int, default=4000)
    parser.add_argument('--hard', type=int, default=6000)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wrong-ids', default='cookbook/rl/wrong_ids.txt')
    parser.add_argument('--hf-subset', default='fullwiki')
    parser.add_argument('--hf-split', default='train')
    parser.add_argument(
        '--only-forced', action=argparse.BooleanOptionalAction, default=False,
        help='If set, re-annotate ONLY IDs in --wrong-ids; default is stratified sampling with wrong_ids force-included.')
    args = parser.parse_args()

    forced_ids: frozenset = frozenset()
    if args.wrong_ids and os.path.exists(args.wrong_ids):
        with open(args.wrong_ids, 'r', encoding='utf-8') as fh:
            forced_ids = frozenset(ln.strip() for ln in fh if ln.strip())
        sys.stderr.write(f'Forced IDs loaded: {len(forced_ids)}\n')

    if args.only_forced and not forced_ids:
        raise ValueError(
            f'--only-forced is set but no IDs loaded from {args.wrong_ids!r}')

    sys.stderr.write(
        f'Loading hotpotqa/hotpot_qa:{args.hf_subset}:{args.hf_split}...\n')
    ds = load_dataset(
        'hotpotqa/hotpot_qa', args.hf_subset, split=args.hf_split)

    if args.only_forced:
        rows = select_forced_only(ds, forced_ids=forced_ids, seed=args.seed)
        sys.stderr.write(
            f'Selected {len(rows)} rows (only-forced mode, '
            f'requested={len(forced_ids)})\n')
    else:
        if args.easy + args.medium + args.hard != args.total:
            raise ValueError(
                f'--easy + --medium + --hard ({args.easy + args.medium + args.hard}) '
                f'must equal --total ({args.total})')
        per_level = {'easy': args.easy, 'medium': args.medium, 'hard': args.hard}
        rows = stratified_sample_with_forced(
            ds, per_level=per_level, forced_ids=forced_ids, seed=args.seed)
        sys.stderr.write(
            f'Selected {len(rows)} rows (stratified per_level={per_level}, '
            f'forced={len(forced_ids)})\n')

    done = load_done_ids(args.output)
    sys.stderr.write(f'Resume: {len(done)} rows already done, skipping.\n')
    pending = [row for row in rows if row['id'] not in done]
    sys.stderr.write(f'Pending: {len(pending)} / {len(rows)}\n')

    api = OpenAI(
        model=args.model, api_key=args.api_key, base_url=args.base_url)

    write_lock = threading.Lock()
    out_fh = open(args.output, 'a', encoding='utf-8')
    rows_done = 0
    rows_failed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(verify_answer, api, args.model, row): row['id']
                for row in pending
            }
            for fut in as_completed(futures):
                rid = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    sys.stderr.write(f'[row {rid}] crashed: {exc}\n')
                    rows_failed += 1
                    continue
                if result is None:
                    rows_failed += 1
                    continue
                with write_lock:
                    out_fh.write(
                        json.dumps(result, ensure_ascii=False) + '\n')
                    out_fh.flush()
                rows_done += 1
                if rows_done % 100 == 0:
                    sys.stderr.write(
                        f'[progress] done={rows_done} '
                        f'failed={rows_failed}\n')
    finally:
        out_fh.close()

    sys.stderr.write(
        f'Done. rows_done={rows_done}, failed={rows_failed}, '
        f'total_pending={len(pending)}\n')


if __name__ == '__main__':
    main()
