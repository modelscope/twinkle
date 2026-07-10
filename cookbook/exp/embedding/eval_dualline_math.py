"""Dual-line math evaluation: baseline vs online process-checking + rubric injection.

This is **Phase 0 of DESIGN §11.6** ("参数化 memory: 查错 LoRA"): before training any
LoRA, test the *upper bound* of the mechanism "pause every N tokens, let a strong
teacher check the partial reasoning for rubric errors, inject the found issue back
into the context, then resume". If even the strongest teacher checking online cannot
lift math accuracy, distilling that ability into a LoRA is pointless — so we gate on
this first.

It deliberately reuses the SAME dataset loader, sampling params and answer grader as
``eval_gpqa_rag.py`` so the two lines are directly comparable:

  - **Line A — baseline** (``--mode baseline``): the student model solves each problem
    in a single pass (identical to ``eval_gpqa_rag.py --mode direct``).
  - **Line B — dualline** (``--mode dualline``, default): the student generates in
    ``--chunk-tokens`` slices; between slices a teacher ``RubricVerifier.diagnose()``
    inspects the partial reasoning. When it reports process issues, the concrete
    finding is injected as a ``[Checker]`` note and generation resumes.

The teacher checker is the ``llm_backup`` teacher API (no student sampler is given to
the verifier, so every check is served by the teacher — exactly the Phase-0 setup).
Configure it via the ``LLM_BACKUP_*`` env vars (see ``utils/llm_backup.py``).

Continuation is done at the token level (crude on purpose — §11.6 says experiment
performance is not a concern): each slice re-feeds the prior ``new_input_feature`` and,
on injection, splices the tokenized note in before resuming.

Launch examples:
    # Dual-line on 200 MATH problems (needs LLM_BACKUP_* for the teacher checker)
    LLM_BACKUP_API_KEY=sk-... LLM_BACKUP_BASE_URL=... \\
        python cookbook/exp/embedding/eval_dualline_math.py --target-eval 200

    # Paired baseline on the same subset (no checker calls)
    python cookbook/exp/embedding/eval_dualline_math.py --mode baseline --target-eval 200
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams as TwinkleSamplingParams
from twinkle.sampler import vLLMSampler

# Reuse the reference eval's dataset + grading + prompts verbatim so the two
# lines are measured on identical footing.
from eval_gpqa_rag import (GEN_MAX_MODEL_LEN, GEN_MAX_TOKENS, GEN_MODEL_ID,
                           GEN_GPU_MEM, GEN_GPUS, GEN_TEMPERATURE, GEN_TOP_P,
                           answers_match, build_direct_prompt, extract_boxed,
                           load_math)

logger = get_logger()

# ---------------------------------------------------------------------------
# Dual-line config
# ---------------------------------------------------------------------------
CHUNK_TOKENS = int(os.environ.get('DUALLINE_CHUNK_TOKENS', 512))
MAX_CHECKS = int(os.environ.get('DUALLINE_MAX_CHECKS', 8))
MAX_INJECTIONS = int(os.environ.get('DUALLINE_MAX_INJECTIONS', 3))
# Only inject when the checker is confident enough that something is wrong.
CHECK_SCORE_FLOOR = float(os.environ.get('DUALLINE_CHECK_FLOOR', 0.6))

# The note format wraps the teacher's finding so the student treats it as an
# external hint rather than its own reasoning. Kept short to limit disruption.
INJECT_TEMPLATE = (
    '\n\n[Checker] A quick review of the reasoning so far found an issue: {issue}\n'
    'Please account for this and continue solving.\n\n')


# ---------------------------------------------------------------------------
# Teacher checker (Phase-0: pure teacher via llm_backup)
# ---------------------------------------------------------------------------
def _build_checker():
    """RubricVerifier with no student sampler -> every diagnose() hits the teacher.

    Uses a fixed, math-oriented process rubric so we do not spend a rubric-
    generation call per slice (the segment here is a partial CoT, not a finished
    trajectory). Falls back to auto-generated rubrics if fixed_rubric is cleared.
    """
    from twinkle_agentic.verifier import RubricVerifier
    from twinkle_agentic.verifier.rubric_verifier import RubricItem

    fixed = [
        RubricItem('The reasoning contains no arithmetic or algebraic error so far',
                   is_hard=True),
        RubricItem('Each step follows logically from the previous ones', is_hard=True),
        RubricItem('No formula or theorem is misstated or misapplied', is_hard=True),
        RubricItem('The approach is on track to answer the actual question asked',
                   is_hard=False),
        RubricItem('No step contradicts an earlier established fact', is_hard=False),
    ]
    return RubricVerifier(fixed_rubric=fixed, gate=True)


def _checker_available() -> bool:
    return bool(os.environ.get('LLM_BACKUP_API_KEY')
                or os.environ.get('LLM_BACKUP_BASE_URL')
                or os.environ.get('OPENAI_API_KEY'))


def _diagnose_partial(checker, problem: str, partial_cot: str):
    """Run the teacher checker on the partial reasoning; return (issue_or_None, detail)."""
    seg = {'messages': [
        {'role': 'user', 'content': problem},
        {'role': 'assistant', 'content': partial_cot},
    ]}
    try:
        detail = checker.diagnose(seg, query=problem)
    except Exception as exc:
        logger.warning(f'[dualline] checker error: {exc}')
        return None, None
    if detail.overall_ok:
        return None, detail
    if detail.scalar >= CHECK_SCORE_FLOOR:
        # Checker leans "mostly fine"; don't disrupt on a marginal signal.
        return None, detail
    fails = [it for it in detail.items if not it.verdict]
    if not fails:
        return None, detail
    # Prefer a fix if the checker gave one; else the reason.
    parts = []
    for it in fails[:2]:
        msg = it.fix or it.reason
        if msg:
            parts.append(msg)
    issue = ' '.join(parts).strip() or detail.summary
    return (issue or None), detail


# ---------------------------------------------------------------------------
# Token-level segmented generation with mid-stream injection
# ---------------------------------------------------------------------------
def _decode(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def generate_dualline(sampler, tokenizer, problem: str, checker,
                      base_params: TwinkleSamplingParams,
                      chunk_tokens: int) -> Dict[str, Any]:
    """Generate the reasoning in slices, checking + injecting between slices.

    Returns a dict with the final text, number of checks/injections, and the
    per-injection findings (for the debug log / future SFT corpus).
    """
    prompt = build_direct_prompt(problem)

    # First slice: encode the trajectory (adds the generation prompt), generate
    # up to chunk_tokens. Subsequent slices reuse the returned new_input_feature.
    chunk_params = TwinkleSamplingParams(
        max_tokens=chunk_tokens, temperature=base_params.temperature,
        top_p=base_params.top_p, num_samples=1)

    cur_inputs: Any = [prompt]
    gen_ids: List[int] = []           # student-generated token ids only
    injected_ids: List[int] = []      # ids we spliced in (excluded from answer)
    n_checks = 0
    n_injections = 0
    findings: List[Dict[str, Any]] = []
    total_new = 0
    finished = False

    while total_new < GEN_MAX_TOKENS:
        responses = sampler.sample(cur_inputs, chunk_params)
        seq = (responses[0].sequences[0]
               if responses and responses[0].sequences else None)
        if seq is None:
            break
        gen_ids.extend(seq.tokens)
        total_new += len(seq.tokens)

        if seq.stop_reason != 'length':
            finished = True
            break  # hit EOS / stop -> generation complete

        # Length-capped slice: this is a pause point. Check the partial CoT.
        if n_checks >= MAX_CHECKS or not checker:
            cur_inputs = [seq.new_input_feature]
            continue

        partial_cot = _decode(tokenizer, gen_ids)
        n_checks += 1
        issue, _detail = _diagnose_partial(checker, problem, partial_cot)

        next_feat = dict(seq.new_input_feature)
        if issue and n_injections < MAX_INJECTIONS:
            note = INJECT_TEMPLATE.format(issue=issue)
            note_ids = tokenizer.encode(note, add_special_tokens=False)
            next_feat['input_ids'] = list(next_feat['input_ids']) + note_ids
            if 'labels' in next_feat:
                next_feat['labels'] = list(next_feat['labels']) + note_ids
            injected_ids.extend(note_ids)
            n_injections += 1
            findings.append({'at_token': total_new, 'issue': issue})
        cur_inputs = [next_feat]

    final_text = _decode(tokenizer, gen_ids)
    return {
        'text': final_text,
        'finished': finished,
        'n_checks': n_checks,
        'n_injections': n_injections,
        'findings': findings,
        'gen_tokens': len(gen_ids),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['baseline', 'dualline'], default='dualline')
    p.add_argument('--math-split', default='test')
    p.add_argument('--per-level', type=int, default=0,
                   help='Problems per difficulty level. 0 => --n split across levels.')
    p.add_argument('--n', type=int, default=200,
                   help='Pool size sampled from MATH (stratified by level).')
    p.add_argument('--target-eval', type=int, default=200,
                   help='Stop after this many problems are evaluated (0 = all sampled).')
    p.add_argument('--chunk-tokens', type=int, default=CHUNK_TOKENS,
                   help='Generate this many tokens between checker pauses.')
    p.add_argument('--batch-size', type=int, default=16,
                   help='Baseline mode batch size (dualline runs per-problem).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = f'./output/dualline/math_{args.mode}_results.jsonl'

    is_dual = (args.mode == 'dualline')
    if is_dual and not _checker_available():
        sys.stderr.write(
            '[dualline] ERROR: --mode dualline needs a teacher checker but no '
            'LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL / OPENAI_API_KEY is set.\n'
            '  Set them, or run --mode baseline for the paired baseline.\n')
        sys.exit(1)

    records = load_math(n=args.n, seed=args.seed, split=args.math_split,
                        per_level=args.per_level)
    if args.target_eval > 0:
        records = records[:args.target_eval]
    sys.stderr.write(f'[dualline] evaluating {len(records)} problems (mode={args.mode})\n')

    device_groups = [
        DeviceGroup(name='sampler', ranks=list(range(GEN_GPUS)),
                    device_type='GPU', gpus_per_worker=GEN_GPUS),
    ]
    gen_mesh = DeviceMesh.from_sizes(world_size=GEN_GPUS, tp_size=GEN_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=GEN_GPUS,
                       groups=device_groups, lazy_collect=False)

    sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={'gpu_memory_utilization': GEN_GPU_MEM,
                     'max_model_len': GEN_MAX_MODEL_LEN},
        device_mesh=gen_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                         enable_thinking=True, max_length=GEN_MAX_MODEL_LEN)
    sys.stderr.write(f'[dualline] vLLM sampler ready (model={GEN_MODEL_ID})\n')

    gen_params = TwinkleSamplingParams(
        max_tokens=GEN_MAX_TOKENS, temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P, num_samples=1)

    checker = None
    tokenizer = None
    if is_dual:
        checker = _build_checker()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, trust_remote_code=True)
        sys.stderr.write('[dualline] teacher checker ready (llm_backup teacher)\n')

    correct = 0
    total = 0
    debug_records: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_f = open(args.output, 'w', encoding='utf-8')

    def _grade_and_log(rec, idx, raw_output, extra=None):
        nonlocal correct, total
        predicted = extract_boxed(raw_output)
        is_correct = answers_match(predicted, rec['reference_answer'])
        if is_correct:
            correct += 1
        total += 1
        debug_rec = {
            'idx': idx,
            'reference_answer': rec['reference_answer'],
            'predicted': predicted,
            'is_correct': is_correct,
            'problem': rec['problem'],
            'model_output': raw_output,
        }
        if rec.get('level'):
            debug_rec['level'] = rec['level']
        if rec.get('type'):
            debug_rec['type'] = rec['type']
        if extra:
            debug_rec.update(extra)
        debug_records.append(debug_rec)
        out_f.write(json.dumps(debug_rec, ensure_ascii=False) + '\n')
        out_f.flush()

    if is_dual:
        for idx, rec in enumerate(records):
            result = generate_dualline(sampler, tokenizer, rec['problem'],
                                       checker, gen_params, args.chunk_tokens)
            _grade_and_log(rec, idx, result['text'], extra={
                'n_checks': result['n_checks'],
                'n_injections': result['n_injections'],
                'findings': result['findings'],
                'finished': result['finished'],
                'gen_tokens': result['gen_tokens'],
            })
            acc = correct / total if total else 0
            sys.stderr.write(
                f'  [{total}/{len(records)}] acc={acc:.4f} ({correct}/{total}) '
                f'checks={result["n_checks"]} inj={result["n_injections"]}\n')
    else:
        import re
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start:batch_start + args.batch_size]
            prompts = [build_direct_prompt(r['problem']) for r in batch]
            responses = sampler.sample(prompts, gen_params)
            for i, (rec, resp) in enumerate(zip(batch, responses)):
                seq = resp.sequences[0] if resp and resp.sequences else None
                raw_output = ''
                if seq is not None:
                    raw_output = re.sub(r'<\|[^|]+\|>', '', seq.decoded or '').rstrip()
                _grade_and_log(rec, batch_start + i, raw_output)
            acc = correct / total if total else 0
            sys.stderr.write(f'  [{total}/{len(records)}] acc={acc:.4f} '
                             f'({correct}/{total})\n')

    overall = correct / total if total else 0
    print(f'\n{"=" * 60}')
    print(f'MATH dual-line — mode={args.mode}, model={GEN_MODEL_ID}')
    print(f'  n={total}, seed={args.seed}, chunk_tokens={args.chunk_tokens}')
    print(f'{"=" * 60}')
    print(f'Overall accuracy: {overall:.4f}  ({correct}/{total})')

    if is_dual:
        tot_checks = sum(r.get('n_checks', 0) for r in debug_records)
        tot_inj = sum(r.get('n_injections', 0) for r in debug_records)
        n_with_inj = sum(1 for r in debug_records if r.get('n_injections', 0) > 0)
        print(f'  checker: {tot_checks} checks, {tot_inj} injections across '
              f'{n_with_inj}/{total} problems')

    if any(r.get('level') for r in debug_records):
        per = defaultdict(lambda: [0, 0])
        for r in debug_records:
            lv = r.get('level', 'Unknown')
            per[lv][1] += 1
            if r['is_correct']:
                per[lv][0] += 1
        print('\nPer-level accuracy:')
        for lv in sorted(per.keys()):
            c, t = per[lv]
            print(f'  {lv:>10}: {c/t:.4f}  ({c}/{t})')

    out_f.close()
    print(f'\n[output] {len(debug_records)} records saved to {args.output}')


if __name__ == '__main__':
    main()
