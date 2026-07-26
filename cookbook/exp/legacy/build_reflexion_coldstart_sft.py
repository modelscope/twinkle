"""Build a cold-start SFT corpus for reflexion skill generation on AOPS.

Pipeline:
  AOPS problems -> frozen base greedy attempt -> strategy-level rubric API diagnosis
  -> answer-free API skill target -> query-only SFT examples.

This is intentionally offline: no GRPO, no actor training, and no skill-model rollout.
The API is treated as an external teacher, so both diagnosis and generated skill targets
are filtered if they reveal the target final answer.

Example:
  LLM_BACKUP_API_KEY=... LLM_BACKUP_BASE_URL=... LLM_BACKUP_MODEL=... \
  python cookbook/exp/embedding/build_reflexion_coldstart_sft.py \
    --dataset aops --n 10000 --output-dir ./output/reflexion_coldstart_sft --overwrite
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.sampler import vLLMSampler

from cookbook.exp.embedding.train_reflexion_skill import (
    MODEL_ID,
    GPU_MEM,
    SamplingParams,
    DiskCache,
    _MATH_RUBRIC,
    _RUBRIC_VERSION,
    _answer_leaked,
    _clean_text,
    _empty_roll,
    _format_diagnosis,
    _numeric_value,
    _parse_seq,
    _run_samples,
    _skillgen_messages,
    _load_excluded_records,
    build_direct_prompt,
    build_skill_solve_prompt,
    build_rubric_checker,
    extract_boxed,
    load_problems,
)

logger = get_logger()

COLDSTART_SYSTEM = """\
You are writing cold-start training targets for a math skill generator. You are given a
competition problem and an answer-free process diagnosis of a previous attempt.

Write concise, reusable guidance that a query-only solver could use before solving this
problem or similar problems. Focus on route choice, structural observations, constraints,
validity checks, and length-control habits.

Output exactly one XML-style block:
<skills>
Your reusable guidance here.
</skills>

Rules:
- Do not mention the diagnosis, rubric, previous attempt, or API.
- Do not reveal the final answer, a corrected value/expression, an option label, or a
  step-by-step solution.
- It is okay to name methods, checks, pitfalls, and local strategy directions.
- Keep it short and useful: 3-6 compact sentences or bullets.
"""

COLDSTART_USER = """\
Problem:
{problem}

Answer-free process diagnosis:
{diagnosis}

Now write the reusable skill guidance.
"""

_SPECIAL_TOKEN_NOTE = 'process diagnosis leaked target answer'


def _api_config() -> Tuple[str, str, str]:
    api_key = os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('LLM_BACKUP_BASE_URL') or os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    model = os.environ.get('LLM_BACKUP_MODEL') or os.environ.get('OPENAI_MODEL') or 'gpt-4o-mini'
    if not api_key:
        raise RuntimeError('Set LLM_BACKUP_API_KEY or OPENAI_API_KEY for cold-start API generation.')
    return api_key, base_url.rstrip('/'), model


def _chat_complete(messages: List[Dict[str, str]], max_tokens: int, temperature: float,
                   retries: int = 3, timeout: int = 120) -> str:
    api_key, base_url, model = _api_config()
    url = f'{base_url}/chat/completions'
    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    last_err = None
    for attempt in range(max(1, retries)):
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode('utf-8'))
            return obj['choices'][0]['message']['content']
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt + 1 < max(1, retries):
                time.sleep(min(8.0, 1.0 * (2 ** attempt)))
                continue
    raise RuntimeError(f'chat completion failed after {retries} attempts: {last_err}')


def _extract_skill_block(text: str) -> Optional[str]:
    low = (text or '').lower()
    end_think = low.rfind('</think>')
    answer = text[end_think + len('</think>'):] if end_think >= 0 else (text or '')
    low = answer.lower()
    s = low.rfind('<skills>')
    if s < 0:
        return None
    inner = s + len('<skills>')
    e = low.find('</skills>', inner)
    if e < 0:
        return None
    block = answer[inner:e].strip()
    return block or None


def _skill_response(block: str) -> str:
    return f'<skills>\n{block.strip()}\n</skills>'


def _baseline_rollout(base_sampler, problems: List[Dict[str, Any]], base_dp: int,
                      args: argparse.Namespace, cache: DiskCache) -> int:
    todo = [r for r in problems if DiskCache.key_for(r['problem']) not in cache]
    if todo:
        outs = _run_samples(base_sampler, [build_direct_prompt(r['problem']) for r in todo],
                            1, args.max_tokens, base_dp, temperature=0.0)
        for r, seqs in zip(todo, outs):
            roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
            cache.put(DiskCache.key_for(r['problem']), roll)
    for r in problems:
        roll = cache.get(DiskCache.key_for(r['problem']))
        r['_init'] = [roll]
        r['_baseline_pass'] = 1.0 if roll.get('correct') else 0.0
        r['_failed'] = not roll.get('correct')
    return len(todo)


def _diagnose_one(checker, r: Dict[str, Any], args: argparse.Namespace) -> str:
    init = r['_init'][0]
    seg_text = init.get('text', '')
    if init.get('stop_reason') == 'length' or not init.get('terminated'):
        seg_text += ('\n\n[Process note: this attempt was cut off at the token budget '
                     'and never produced a final \\boxed{} answer.]')
    seg = {'messages': [{'role': 'user', 'content': r['problem']},
                        {'role': 'assistant', 'content': seg_text}]}
    attempts = max(1, args.rubric_retries + 1)
    for attempt in range(attempts):
        try:
            return _format_diagnosis(checker.diagnose(seg, query=r['problem']))
        except Exception as exc:
            if attempt + 1 < attempts:
                logger.warning(f'[rubric] diagnose error: {exc}; retry {attempt + 1}/{args.rubric_retries}')
                time.sleep(min(4.0, 0.5 * (2 ** attempt)))
            else:
                logger.warning(f'[rubric] diagnose failed: {exc}')
    return ''


def _diagnose_batch(checker, rows: List[Dict[str, Any]], args: argparse.Namespace,
                    cache: DiskCache) -> int:
    pending = []
    for r in rows:
        init = r['_init'][0]
        term = 'L' if (init.get('stop_reason') == 'length' or not init.get('terminated')) else 'T'
        key = DiskCache.key_for(r['problem'], init.get('text', ''), _RUBRIC_VERSION, term)
        if key in cache:
            r['_rubric_diag'] = cache.get(key)
        else:
            pending.append((r, key))
    if not pending:
        return 0

    def run(item):
        r, key = item
        diag = _diagnose_one(checker, r, args)
        return r, key, diag

    workers = max(1, min(args.rubric_workers, len(pending)))
    fresh = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, key, diag in ex.map(run, pending):
            r['_rubric_diag'] = diag or ''
            if diag:
                cache.put(key, diag)
                fresh += 1
    return fresh


def _target_key(problem: str, diagnosis: str, sample_idx: int) -> str:
    return DiskCache.key_for('coldstart_skill_v2', str(sample_idx), problem, diagnosis)


def _generate_skill_targets(r: Dict[str, Any], args: argparse.Namespace,
                            cache: DiskCache) -> List[Dict[str, Any]]:
    out = []
    messages = [
        {'role': 'system', 'content': COLDSTART_SYSTEM},
        {'role': 'user', 'content': COLDSTART_USER.format(problem=r['problem'], diagnosis=r.get('_rubric_diag', ''))},
    ]
    for sample_idx in range(max(1, int(args.api_samples))):
        key = _target_key(r['problem'], r.get('_rubric_diag', ''), sample_idx)
        if key in cache:
            resp = cache.get(key)
        else:
            resp = _chat_complete(messages, max_tokens=args.api_max_tokens,
                                  temperature=args.api_temperature, retries=args.api_retries,
                                  timeout=args.api_timeout)
            cache.put(key, resp)
        block = _extract_skill_block(resp) or ''
        leaked = _answer_leaked(resp + '\n' + block, r['reference_answer'])
        out.append({'sample_idx': sample_idx, 'raw_response': resp,
                    'skills': block, 'skill_leak': leaked})
    return out


def _sft_messages(problem: str, response: str) -> List[Dict[str, str]]:
    msgs = _skillgen_messages(problem, 'B', '')
    return msgs + [{'role': 'assistant', 'content': response}]


def _init_base_sampler(args: argparse.Namespace):
    twinkle.initialize(mode='ray', nproc_per_node=args.base_gpus, lazy_collect=False,
                       groups=[DeviceGroup(name='base_sampler', ranks=list(range(args.base_gpus)), device_type='GPU')])
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={'gpu_memory_utilization': GPU_MEM,
                     'max_model_len': args.max_model_len,
                     'tensor_parallel_size': 1},
        device_mesh=DeviceMesh.from_sizes(world_size=args.base_gpus, dp_size=args.base_gpus),
        remote_group='base_sampler')
    sampler.set_template('Template', model_id=MODEL_ID, enable_thinking=True,
                         max_length=args.max_model_len)
    return sampler, args.base_gpus


def _select_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    load_n = 0 if args.numeric_only or args.eval_size > 0 else max(args.n, args.target_size + args.eval_size)
    records = load_problems(args.dataset, load_n, args.seed)
    raw_n = len(records)
    if args.numeric_only:
        records = [{**r, 'reference_answer': v}
                   for r, v in ((r, _numeric_value(r.get('reference_answer'))) for r in records)
                   if v is not None]
    import numpy as np
    np.random.RandomState(args.seed).shuffle(records)
    exclude_ids, exclude_problems = _load_excluded_records(getattr(args, 'exclude_data_ids', ''))
    excluded = 0
    if exclude_ids or exclude_problems:
        before = len(records)
        records = [r for r in records
                   if str(r.get('data_id', '')) not in exclude_ids
                   and str(r.get('problem', '')).strip() not in exclude_problems]
        excluded = before - len(records)
    eval_n = min(args.eval_size, len(records)) if args.eval_size > 0 else 0
    pool = [dict(r) for r in records[eval_n:]]
    pool_offset = max(0, int(getattr(args, 'pool_offset', 0) or 0))
    if pool_offset:
        if pool_offset >= len(pool):
            raise ValueError(f'--pool-offset ({pool_offset}) leaves no cold-start records from pool size {len(pool)}')
        pool = pool[pool_offset:]
    n = min(args.n, len(pool)) if args.n > 0 else min(len(pool), max(args.target_size * 2, args.target_size + 512))
    stats = {'raw_loaded': raw_n, 'numeric_dropped': raw_n - len(records) - excluded,
             'excluded_records': excluded, 'eval_size': eval_n,
             'pool_offset': pool_offset, 'pool_selected': n}
    return pool[:n], stats


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--target-size', type=int, default=10000, help='Number of accepted SFT examples to write.')
    p.add_argument('--n', type=int, default=0, help='Raw train-pool size after eval split; 0 auto-selects.')
    p.add_argument('--pool-offset', type=int, default=0,
                   help='Skip this many shuffled non-eval records before building the cold-start pool.')
    p.add_argument('--exclude-data-ids', default='',
                   help='Comma-separated jsonl files whose data_id/problem keys are excluded, '
                        'useful for building non-overlapping shards.')
    p.add_argument('--eval-size', type=int, default=128)
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='./output/reflexion_coldstart_sft')
    p.add_argument('--cache-dir', default='')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--base-gpus', type=int, default=int(os.environ.get('BASE_GPUS', 4)))
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--rubric-workers', type=int, default=16)
    p.add_argument('--rubric-retries', type=int, default=2)
    p.add_argument('--api-workers', type=int, default=16)
    p.add_argument('--api-samples', type=int, default=4,
                   help='API skill targets sampled per problem before executor verification.')
    p.add_argument('--verify-targets', action=argparse.BooleanOptionalAction, default=True,
                   help='Run frozen base executor with each API skill target and keep a successful one.')
    p.add_argument('--keep-unverified-targets', action='store_true',
                   help='If all executor checks fail, keep the first clean target anyway. Default skips it.')
    p.add_argument('--api-retries', type=int, default=3)
    p.add_argument('--api-timeout', type=int, default=120)
    p.add_argument('--api-max-tokens', type=int, default=768)
    p.add_argument('--api-temperature', type=float, default=0.2)
    p.add_argument('--require-fail', action=argparse.BooleanOptionalAction, default=True,
                   help='Only keep API diagnoses containing [FAIL]. Use --no-require-fail to keep OK diagnoses too.')
    return p.parse_args()


def _write(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = _build_args()
    if args.target_size <= 0:
        raise ValueError('--target-size must be positive')
    records, data_stats = _select_records(args)
    if not records:
        raise ValueError('no records selected')

    os.makedirs(args.output_dir, exist_ok=True)
    sft_path = os.path.join(args.output_dir, 'coldstart_sft.jsonl')
    rec_path = os.path.join(args.output_dir, 'coldstart_records.jsonl')
    for path in (sft_path, rec_path):
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f'{path} exists; pass --overwrite')

    checker = build_rubric_checker()
    if checker is None:
        raise RuntimeError('No rubric checker available; set LLM_BACKUP_API_KEY/BASE_URL or OPENAI_API_KEY.')
    _api_config()
    base_sampler, base_dp = _init_base_sampler(args)

    cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    use_cache = not args.no_cache
    base_cache = DiskCache(os.path.join(cache_dir, 'baseline.jsonl'), use_cache)
    rubric_cache = DiskCache(os.path.join(cache_dir, 'rubric.jsonl'), use_cache)
    skill_cache = DiskCache(os.path.join(cache_dir, 'api_skill.jsonl'), use_cache)

    cfg = {
        'record_type': 'config', 'mode': 'coldstart_sft_build', 'dataset': args.dataset,
        'target_size': args.target_size, 'selected_records': len(records), 'seed': args.seed,
        'pool_offset': args.pool_offset, 'exclude_data_ids': args.exclude_data_ids,
        'numeric_only': args.numeric_only, **data_stats,
        'rubric_version': _RUBRIC_VERSION, 'rubric_check': f'fixed_math_{len(_MATH_RUBRIC)}crit',
        'api_model': os.environ.get('LLM_BACKUP_MODEL') or os.environ.get('OPENAI_MODEL') or 'gpt-4o-mini',
        'api_samples': args.api_samples, 'verify_targets': args.verify_targets,
        'keep_unverified_targets': args.keep_unverified_targets,
        'require_fail': args.require_fail, 'started': int(time.time()),
    }

    accepted = 0
    skipped_no_diag = skipped_no_fail = skipped_api_leak = 0
    skipped_no_skill = skipped_skill_leak = skipped_executor_fail = 0
    processed = 0
    with open(sft_path, 'w', encoding='utf-8') as sft_f, open(rec_path, 'w', encoding='utf-8') as rec_f:
        _write(rec_f, cfg)
        for start in range(0, len(records), args.chunk_size):
            if accepted >= args.target_size:
                break
            chunk = [dict(r) for r in records[start:start + args.chunk_size]]
            _baseline_rollout(base_sampler, chunk, base_dp, args, base_cache)
            _diagnose_batch(checker, chunk, args, rubric_cache)

            def gen_one(r: Dict[str, Any]):
                return r, _generate_skill_targets(r, args, skill_cache)

            candidates = []
            for r in chunk:
                processed += 1
                diag = r.get('_rubric_diag', '') or ''
                if not diag:
                    skipped_no_diag += 1
                    continue
                if args.require_fail and '[FAIL]' not in diag:
                    skipped_no_fail += 1
                    continue
                if _answer_leaked(diag, r['reference_answer']):
                    skipped_api_leak += 1
                    continue
                candidates.append(r)

            generated = []
            workers = max(1, min(args.api_workers, len(candidates)))
            if candidates:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for r, targets in ex.map(gen_one, candidates):
                        for target in targets:
                            skills = target.get('skills', '')
                            if not skills:
                                skipped_no_skill += 1
                                continue
                            if target.get('skill_leak'):
                                skipped_skill_leak += 1
                                continue
                            target['r'] = r
                            target['response'] = _skill_response(skills)
                            generated.append(target)

            selected = []
            selected_keys = set()
            if generated and args.verify_targets:
                verify_prompts = [build_skill_solve_prompt(g['r']['problem'], g['skills']) for g in generated]
                verify_outs = _run_samples(base_sampler, verify_prompts, 1, args.max_tokens,
                                           base_dp, temperature=0.0)
                attempted_keys = set()
                for g, seqs in zip(generated, verify_outs):
                    r = g['r']
                    key = r.get('data_id') or r['problem']
                    attempted_keys.add(key)
                    roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
                    g['target_roll'] = roll
                    if key not in selected_keys and roll.get('correct') and roll.get('terminated'):
                        g['executor_verified'] = True
                        selected.append(g)
                        selected_keys.add(key)
                if args.keep_unverified_targets:
                    for g in generated:
                        r = g['r']
                        key = r.get('data_id') or r['problem']
                        if key not in selected_keys:
                            g['executor_verified'] = False
                            g.setdefault('target_roll', {})
                            selected.append(g)
                            selected_keys.add(key)
                skipped_executor_fail += len(attempted_keys - selected_keys)
            elif generated:
                for g in generated:
                    r = g['r']
                    key = r.get('data_id') or r['problem']
                    if key not in selected_keys:
                        g['executor_verified'] = False
                        selected.append(g)
                        selected_keys.add(key)

            for g in selected:
                if accepted >= args.target_size:
                    break
                r = g['r']
                response = g['response']
                messages = _sft_messages(r['problem'], response)
                sft_row = {
                    'messages': messages,
                    'user_data': {'key_rounds': [len(messages) - 1]},
                    'data_id': r.get('data_id'),
                    'problem': r['problem'], 'reference_answer': r['reference_answer'],
                    'skills': g['skills'], 'response': response,
                    'view': 'B', 'sft': True, 'source': 'api_coldstart',
                    'api_sample_idx': g.get('sample_idx'),
                    'executor_verified': g.get('executor_verified', False),
                    'baseline_correct': r['_init'][0].get('correct'),
                    'baseline_terminated': r['_init'][0].get('terminated'),
                    'baseline_stop_reason': r['_init'][0].get('stop_reason'),
                    'target_correct': (g.get('target_roll') or {}).get('correct'),
                    'target_terminated': (g.get('target_roll') or {}).get('terminated'),
                    'target_stop_reason': (g.get('target_roll') or {}).get('stop_reason'),
                    'diagnosis': r.get('_rubric_diag', ''),
                }
                audit = {
                    'record_type': 'coldstart_problem', 'accepted': True,
                    'data_id': r.get('data_id'),
                    'problem': r['problem'], 'reference_answer': r['reference_answer'],
                    'baseline': r['_init'][0], 'diagnosis': r.get('_rubric_diag', ''),
                    'raw_skill_response': g.get('raw_response'), 'skills': g['skills'],
                    'api_sample_idx': g.get('sample_idx'),
                    'executor_verified': g.get('executor_verified', False),
                    'target_roll': g.get('target_roll'),
                }
                _write(sft_f, sft_row)
                _write(rec_f, audit)
                accepted += 1
            sys.stderr.write(
                f'[coldstart] processed={processed} accepted={accepted}/{args.target_size} '
                f'skip(no_diag={skipped_no_diag}, no_fail={skipped_no_fail}, api_leak={skipped_api_leak}, '
                f'no_skill={skipped_no_skill}, skill_leak={skipped_skill_leak}, '
                f'executor_fail={skipped_executor_fail})\n')
            sft_f.flush(); rec_f.flush()

    summary = {
        'record_type': 'summary', 'processed': processed, 'accepted': accepted,
        'skipped_no_diag': skipped_no_diag, 'skipped_no_fail': skipped_no_fail,
        'skipped_api_leak': skipped_api_leak, 'skipped_no_skill': skipped_no_skill,
        'skipped_skill_leak': skipped_skill_leak,
        'skipped_executor_fail': skipped_executor_fail, 'finished': int(time.time()),
    }
    with open(rec_path, 'a', encoding='utf-8') as rec_f:
        _write(rec_f, summary)
    sys.stderr.write(f'[coldstart] wrote {accepted} SFT rows to {sft_path}\n')


if __name__ == '__main__':
    main()
