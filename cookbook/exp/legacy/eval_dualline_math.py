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
    inspects the full reasoning so far (query + all prior response). When it reports
    process issues, the finding is injected back as a first-person self-correction
    (in the student's own voice) and generation resumes.

The teacher checker is the ``llm_backup`` teacher API (no student sampler is given to
the verifier, so every check is served by the teacher — exactly the Phase-0 setup).
Configure it via the ``LLM_BACKUP_*`` env vars (see ``utils/llm_backup.py``).

Continuation is done at the token level (crude on purpose — §11.6 says experiment
performance is not a concern): each slice re-feeds the prior ``new_input_feature`` and,
on injection, splices the tokenized note in before resuming.

The dataset defaults to AoPS (``--dataset aops``), which auto-downloads from
ModelScope so no local data path is needed; pass ``--dataset math`` to use the
local Hendrycks MATH set instead. Both lines MUST share ``--dataset``, ``--n``,
``--target-eval`` and ``--seed`` to stay a paired comparison.

Launch examples:
    # Dual-line on 200 AoPS problems (needs LLM_BACKUP_* for the teacher checker)
    LLM_BACKUP_API_KEY=sk-... LLM_BACKUP_BASE_URL=... \\
        python cookbook/exp/embedding/eval_dualline_math.py \\
            --n 200 --target-eval 200 --seed 42

    # Paired baseline on the same subset (no checker calls)
    python cookbook/exp/embedding/eval_dualline_math.py --mode baseline \\
        --n 200 --target-eval 200 --seed 42
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams as TwinkleSamplingParams
from twinkle.sampler import vLLMSampler

# Reuse the reference eval's dataset + grading + prompts verbatim so the two
# lines are measured on identical footing.
from eval_gpqa_rag import (GEN_MODEL_ID, GEN_GPU_MEM, GEN_GPUS, GEN_TEMPERATURE,
                           GEN_TOP_P, answers_match, build_direct_prompt,
                           extract_boxed, load_aops, load_math)

# Dualline eval defaults (override via --max-model-len or DUALLINE_MAX_MODEL_LEN).
DUALLINE_DEFAULT_MAX_MODEL_LEN = int(os.environ.get('DUALLINE_MAX_MODEL_LEN', 32000))
DUALLINE_DEFAULT_MAX_GEN_TOKENS = int(
    os.environ.get('DUALLINE_MAX_GEN_TOKENS', DUALLINE_DEFAULT_MAX_MODEL_LEN))

# vLLM parallel: default tp=1, dp=GEN_GPUS (override with GEN_TP / keep GEN_GPUS=8).
GEN_TP = int(os.environ.get('GEN_TP', 1))

logger = get_logger()

# ---------------------------------------------------------------------------
# Dual-line config
# ---------------------------------------------------------------------------
CHUNK_TOKENS = int(os.environ.get('DUALLINE_CHUNK_TOKENS', 512))
MAX_CHECKS = int(os.environ.get('DUALLINE_MAX_CHECKS', 8))
MAX_INJECTIONS = int(os.environ.get('DUALLINE_MAX_INJECTIONS', 3))
# Only inject when the checker is confident enough that something is wrong.
CHECK_SCORE_FLOOR = float(os.environ.get('DUALLINE_CHECK_FLOOR', 0.6))
# The note is written in the student's own first-person voice so, when spliced
# back in, the running model treats it as its own mid-thought self-correction
# rather than an external interruption (which tended to derail generation toward
# max-length). Kept short to limit disruption.
INJECT_TEMPLATE = (
    '\n\nWait — reviewing my reasoning above, I realize there is a problem: {issue}\n'
    'Let me correct this and continue.\n\n')

# When context hits max_model_len (or sample fails), dump query + generation here.
OVERFLOW_DUMP_DIR = os.environ.get(
    'DUALLINE_OVERFLOW_DUMP_DIR', './output/dualline/overflow_dumps')


def _decode(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _input_ids_len(cur_inputs: Any) -> Optional[int]:
    """Length of the tokenized prompt fed to vLLM on this step, if known."""
    if not cur_inputs:
        return None
    item = cur_inputs[0]
    if isinstance(item, dict) and 'input_ids' in item:
        ids = item['input_ids']
        return len(ids) if ids is not None else None
    return None


def _dump_dualline_state(
    *,
    reason: str,
    problem: str,
    debug_idx: Optional[int],
    chunk_tokens: int,
    cur_inputs: Any,
    gen_ids: List[int],
    injected_ids: List[int],
    tokenizer,
    n_checks: int,
    n_injections: int,
    findings: List[Dict[str, Any]],
    total_new: int,
    finished: bool,
    max_model_len: int,
    error: Optional[str] = None,
) -> str:
    """Persist state for post-mortem (student CoT vs checker injection). Returns path."""
    os.makedirs(OVERFLOW_DUMP_DIR, exist_ok=True)
    tag = f'idx{debug_idx}' if debug_idx is not None else 'idx_unknown'
    path = os.path.join(
        OVERFLOW_DUMP_DIR, f'{tag}_{reason}_{int(time.time())}.json')

    partial_cot = _decode(tokenizer, gen_ids) if tokenizer and gen_ids else ''
    injected_text = (_decode(tokenizer, injected_ids)
                     if tokenizer and injected_ids else '')
    ctx_len = _input_ids_len(cur_inputs)

    payload: Dict[str, Any] = {
        'reason': reason,
        'error': error,
        'query': problem,
        'debug_idx': debug_idx,
        'gen_token_count': len(gen_ids),
        'injected_token_count': len(injected_ids),
        'context_input_ids_len': ctx_len,
        'max_model_len': max_model_len,
        'chunk_tokens': chunk_tokens,
        'total_new': total_new,
        'n_checks': n_checks,
        'n_injections': n_injections,
        'findings': findings,
        'finished': finished,
        'partial_cot': partial_cot,
        'injected_text': injected_text,
        'partial_cot_chars': len(partial_cot),
        'context_is_message_prompt': ctx_len is None,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cot_path = path.replace('.json', '_partial_cot.txt')
    with open(cot_path, 'w', encoding='utf-8') as f:
        f.write(partial_cot)
    sys.stderr.write(f'[dualline] overflow dump -> {path}\n')
    return path

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
    """Run the teacher checker on the reasoning so far; return (issue_or_None, detail).

    ``partial_cot`` is the FULL reasoning generated so far (all prior chunks plus
    any self-corrections already spliced in), not just the latest slice, so the
    teacher judges the whole derivation in context. We label it as in-progress so
    it grades correctness of the steps rather than penalizing the absence of a
    final answer.
    """
    seg_content = (
        '[The following is the full reasoning so far, still in progress and not '
        'yet complete. Judge only whether the reasoning up to this point is '
        'mathematically correct; do not expect a final answer here.]\n\n' + partial_cot)
    seg = {'messages': [
        {'role': 'user', 'content': problem},
        {'role': 'assistant', 'content': seg_content},
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


def _pad_batch_for_dp(items: List[Any], gen_dp: int) -> List[Any]:
    """``slice_dp`` needs batch len >= DP world size (every rank gets work).

    Only kicks in on the tail rounds when fewer than ``gen_dp`` problems are
    still active; the padded replicas are dropped by the caller.
    """
    if gen_dp <= 1 or not items or len(items) >= gen_dp:
        return items
    pad = [copy.deepcopy(items[-1]) for _ in range(gen_dp - len(items))]
    return items + pad


class _DualState:
    """Per-problem generation state for the batched dualline loop.

    All problems advance together, one ``chunk_tokens`` slice per round. A
    problem stays *active* until it emits EOS, hits ``max_gen_tokens``, would
    overflow ``max_model_len``, or a sample call fails. Because the problems
    share every round's ``sampler.sample`` call, the vLLM engine batches them
    (and, with dp>1, spreads them across ranks) instead of running one at a
    time.
    """

    __slots__ = ('idx', 'problem', 'cur_input', 'gen_ids', 'injected_ids',
                 'n_checks', 'n_injections', 'findings', 'total_new',
                 'finished', 'stopped_reason', 'context_input_ids_len',
                 'pending_partial_cot', 'prompt_len')

    def __init__(self, idx: int, problem: str, prompt: Any):
        self.idx = idx
        self.problem = problem
        self.cur_input: Any = prompt      # str prompt (round 0) or input_feature
        self.gen_ids: List[int] = []       # student-generated token ids only
        self.injected_ids: List[int] = []  # spliced-in ids (excluded from answer)
        self.n_checks = 0
        self.n_injections = 0
        self.findings: List[Dict[str, Any]] = []
        self.total_new = 0
        self.finished = False
        self.stopped_reason: Optional[str] = None
        self.context_input_ids_len: Optional[int] = None
        self.pending_partial_cot: Optional[str] = None
        self.prompt_len: Optional[int] = None  # token len of the fixed prompt prefix

    def cur_input_len(self) -> Optional[int]:
        item = self.cur_input
        if isinstance(item, dict) and 'input_ids' in item:
            ids = item['input_ids']
            return len(ids) if ids is not None else None
        return None

    def result(self, tokenizer) -> Dict[str, Any]:
        if self.context_input_ids_len is None:
            self.context_input_ids_len = self.cur_input_len()
        return {
            'text': _decode(tokenizer, self.gen_ids),
            'finished': self.finished,
            'stopped_reason': self.stopped_reason,
            'context_input_ids_len': self.context_input_ids_len,
            'n_checks': self.n_checks,
            'n_injections': self.n_injections,
            'findings': self.findings,
            'gen_tokens': len(self.gen_ids),
        }


def _dump_state_obj(st: '_DualState', tokenizer, chunk_tokens: int,
                    max_model_len: int, reason: str, error: str) -> None:
    _dump_dualline_state(
        reason=reason,
        problem=st.problem,
        debug_idx=st.idx,
        chunk_tokens=chunk_tokens,
        cur_inputs=[st.cur_input],
        gen_ids=st.gen_ids,
        injected_ids=st.injected_ids,
        tokenizer=tokenizer,
        n_checks=st.n_checks,
        n_injections=st.n_injections,
        findings=st.findings,
        total_new=st.total_new,
        finished=st.finished,
        max_model_len=max_model_len,
        error=error,
    )


# ---------------------------------------------------------------------------
# Batched token-level segmented generation with mid-stream injection
# ---------------------------------------------------------------------------
def run_dualline_batch(sampler, tokenizer, problems: List[str], checker,
                       base_params: TwinkleSamplingParams,
                       chunk_tokens: int,
                       max_model_len: int,
                       max_gen_tokens: int,
                       gen_dp: int = 1,
                       diagnose_workers: int = 8) -> List[Dict[str, Any]]:
    """Advance every problem in lock-step slices, sharing one sampler call/round.

    Each round: (1) preflight-drop any problem that would overflow the context,
    (2) one ``sampler.sample`` over all still-active problems (vLLM batches +
    spreads over dp ranks), (3) for the length-capped ones, run the teacher
    diagnoses concurrently and splice injections, then loop.

    Returns per-problem result dicts in the original ``problems`` order.
    """
    from concurrent.futures import ThreadPoolExecutor

    chunk_params = TwinkleSamplingParams(
        max_tokens=chunk_tokens, temperature=base_params.temperature,
        top_p=base_params.top_p, num_samples=1)

    states = [_DualState(i, p, build_direct_prompt(p))
              for i, p in enumerate(problems)]
    active = list(states)
    round_no = 0

    while active:
        round_no += 1

        # (1) Preflight: drop problems that would overflow the context window,
        # and those that already reached the generation-token cap.
        survivors: List[_DualState] = []
        for st in active:
            if st.total_new >= max_gen_tokens:
                st.stopped_reason = st.stopped_reason or 'max_gen_tokens'
                continue
            ctx_len = st.cur_input_len()
            if ctx_len is not None and ctx_len + chunk_tokens >= max_model_len:
                st.context_input_ids_len = ctx_len
                st.stopped_reason = 'context_full'
                _dump_state_obj(
                    st, tokenizer, chunk_tokens, max_model_len,
                    reason='preflight_context_full',
                    error=(f'context len {ctx_len} + chunk {chunk_tokens} '
                           f'>= max_model_len {max_model_len}'))
                continue
            survivors.append(st)
        active = survivors
        if not active:
            break

        # (2) One shared sampler call over all active problems. On tail rounds
        # with fewer active problems than dp ranks, pad to keep slice_dp happy
        # and drop the padded responses. The context-overflow preflight above
        # guarantees every input still fits, so a length-capped slice should
        # never raise here; let any genuine engine error propagate instead of
        # masking it as a whole-round failure.
        batch_inputs = [st.cur_input for st in active]
        padded = _pad_batch_for_dp(batch_inputs, gen_dp)
        responses = sampler.sample(padded, chunk_params)
        responses = responses[:len(active)]

        # (3) Consume each problem's slice; queue the ones needing a check.
        to_diagnose: List[_DualState] = []
        next_active: List[_DualState] = []
        for st, resp in zip(active, responses):
            seq = resp.sequences[0] if resp and resp.sequences else None
            if seq is None:
                st.stopped_reason = st.stopped_reason or 'empty_response'
                continue
            st.gen_ids.extend(seq.tokens)
            st.total_new += len(seq.tokens)
            st.cur_input = seq.new_input_feature
            if st.prompt_len is None:
                # Fixed prompt prefix = everything before this round's generation.
                st.prompt_len = len(st.cur_input['input_ids']) - len(seq.tokens)

            if seq.stop_reason != 'length':
                st.finished = True          # EOS / stop -> done
                continue
            if st.n_checks >= MAX_CHECKS or not checker:
                next_active.append(st)      # keep generating, no more checks
                continue
            # Diagnose the FULL reasoning generated so far (all prior chunks plus
            # any self-corrections already spliced in), so the teacher judges the
            # whole derivation in context rather than an isolated tail slice.
            st.pending_partial_cot = _decode(
                tokenizer, st.cur_input['input_ids'][st.prompt_len:])
            st.n_checks += 1
            to_diagnose.append(st)

        # Concurrent teacher diagnoses for this round's length-capped problems.
        if to_diagnose:
            def _run(st: _DualState):
                return st, _diagnose_partial(
                    checker, st.problem, st.pending_partial_cot)
            workers = max(1, min(diagnose_workers, len(to_diagnose)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for st, (issue, _detail) in ex.map(_run, to_diagnose):
                    st.pending_partial_cot = None
                    if issue and st.n_injections < MAX_INJECTIONS:
                        note = INJECT_TEMPLATE.format(issue=issue)
                        note_ids = tokenizer.encode(note, add_special_tokens=False)
                        feat = dict(st.cur_input)
                        feat['input_ids'] = list(feat['input_ids']) + note_ids
                        if 'labels' in feat:
                            feat['labels'] = list(feat['labels']) + note_ids
                        st.cur_input = feat
                        st.injected_ids.extend(note_ids)
                        st.n_injections += 1
                        st.findings.append(
                            {'at_token': st.total_new, 'issue': issue})
                    next_active.append(st)

        active = next_active
        n_done = sum(1 for s in states if s.finished or s.stopped_reason)
        sys.stderr.write(
            f'[dualline] round {round_no}: active={len(active)} '
            f'done={n_done}/{len(states)}\n')

    return [st.result(tokenizer) for st in states]


def _load_tokenizer(model_id: str):
    """Load the tokenizer from ModelScope (matches the vLLM sampler source).

    The box runs offline, so ``transformers.AutoTokenizer`` (which resolves via
    the HF hub) fails with ``Network is unreachable``. ModelScope's AutoTokenizer
    downloads/reads from the ModelScope cache instead — the same place the vLLM
    sampler already pulled the model from. Falls back to transformers only if the
    ModelScope path is unavailable.
    """
    try:
        from modelscope import AutoTokenizer as MSAutoTokenizer
        return MSAutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        sys.stderr.write(f'[dualline] modelscope tokenizer load failed ({exc}); '
                         f'falling back to transformers\n')
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['baseline', 'dualline'], default='dualline')
    p.add_argument('--dataset', choices=['aops', 'math'], default='aops',
                   help='Evaluation dataset. "aops" (default) auto-downloads from '
                        'ModelScope (no local path needed); "math" reads local '
                        'MATH_DATA_DIR, stratified by difficulty level.')
    p.add_argument('--math-split', default='test')
    p.add_argument('--per-level', type=int, default=0,
                   help='MATH only: problems per level. 0 => --n split across levels.')
    p.add_argument('--n', type=int, default=32,
                   help='Pool size sampled from the dataset (MATH is stratified '
                        'by level; AoPS is a flat shuffle).')
    p.add_argument('--target-eval', type=int, default=32,
                   help='Stop after this many problems are evaluated (0 = all sampled).')
    p.add_argument('--max-model-len', type=int, default=DUALLINE_DEFAULT_MAX_MODEL_LEN,
                   help='vLLM max_model_len / template max_length (default 32000).')
    p.add_argument('--max-gen-tokens', type=int, default=DUALLINE_DEFAULT_MAX_GEN_TOKENS,
                   help='Cap total generated tokens per problem (default: same as '
                        'max-model-len / DUALLINE_MAX_GEN_TOKENS).')
    p.add_argument('--chunk-tokens', type=int, default=CHUNK_TOKENS,
                   help='Generate this many tokens between checker pauses.')
    p.add_argument('--batch-size', type=int, default=16,
                   help='Baseline mode batch size. Dualline runs all problems '
                        'concurrently (one shared sampler call per slice-round).')
    p.add_argument('--diagnose-workers', type=int,
                   default=int(os.environ.get('DUALLINE_DIAGNOSE_WORKERS', 8)),
                   help='Concurrency for teacher diagnose() calls within a round.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = f'./output/dualline/{args.dataset}_{args.mode}_results.jsonl'

    is_dual = (args.mode == 'dualline')
    if is_dual and not _checker_available():
        sys.stderr.write(
            '[dualline] ERROR: --mode dualline needs a teacher checker but no '
            'LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL / OPENAI_API_KEY is set.\n'
            '  Set them, or run --mode baseline for the paired baseline.\n')
        sys.exit(1)

    if args.dataset == 'math':
        records = load_math(n=args.n, seed=args.seed, split=args.math_split,
                            per_level=args.per_level)
    else:
        records = load_aops(n=args.n, seed=args.seed)
    if args.target_eval > 0:
        records = records[:args.target_eval]
    max_model_len = args.max_model_len
    max_gen_tokens = args.max_gen_tokens
    sys.stderr.write(
        f'[dualline] evaluating {len(records)} problems '
        f'(mode={args.mode}, dataset={args.dataset}, '
        f'max_model_len={max_model_len}, max_gen_tokens={max_gen_tokens})\n')

    device_groups = [
        DeviceGroup(name='sampler', ranks=list(range(GEN_GPUS)),
                    device_type='GPU', gpus_per_worker=GEN_TP),
    ]
    if GEN_GPUS % GEN_TP != 0:
        raise ValueError(f'GEN_GPUS ({GEN_GPUS}) must be divisible by GEN_TP ({GEN_TP})')
    gen_dp = GEN_GPUS // GEN_TP
    gen_mesh = DeviceMesh.from_sizes(world_size=GEN_GPUS, dp_size=gen_dp, tp_size=GEN_TP)
    twinkle.initialize(mode='ray', nproc_per_node=GEN_GPUS,
                       groups=device_groups, lazy_collect=False)

    sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': GEN_GPU_MEM,
            'max_model_len': max_model_len,
            'tensor_parallel_size': GEN_TP,
        },
        device_mesh=gen_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                         enable_thinking=True, max_length=max_model_len)
    sys.stderr.write(
        f'[dualline] vLLM sampler ready (model={GEN_MODEL_ID}, '
        f'tp={GEN_TP}, dp={gen_dp})\n')

    gen_params = TwinkleSamplingParams(
        max_tokens=max_gen_tokens, temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P, num_samples=1)

    checker = None
    tokenizer = None
    if is_dual:
        checker = _build_checker()
        tokenizer = _load_tokenizer(GEN_MODEL_ID)
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
        problems = [rec['problem'] for rec in records]
        results = run_dualline_batch(
            sampler, tokenizer, problems, checker, gen_params,
            args.chunk_tokens, max_model_len, max_gen_tokens,
            gen_dp=gen_dp, diagnose_workers=args.diagnose_workers)
        for idx, (rec, result) in enumerate(zip(records, results)):
            _grade_and_log(rec, idx, result['text'], extra={
                'n_checks': result['n_checks'],
                'n_injections': result['n_injections'],
                'findings': result['findings'],
                'finished': result['finished'],
                'stopped_reason': result.get('stopped_reason'),
                'context_input_ids_len': result.get('context_input_ids_len'),
                'gen_tokens': result['gen_tokens'],
            })
            stop_tag = (f' stop={result["stopped_reason"]}'
                        if result.get('stopped_reason') else '')
            sys.stderr.write(
                f'  [idx {idx}] correct={debug_records[-1]["is_correct"]} '
                f'gen={result["gen_tokens"]} checks={result["n_checks"]} '
                f'inj={result["n_injections"]}{stop_tag}\n')
        acc = correct / total if total else 0
        sys.stderr.write(
            f'[dualline] batched eval done: acc={acc:.4f} ({correct}/{total})\n')
    else:
        import re
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start:batch_start + args.batch_size]
            prompts = [build_direct_prompt(r['problem']) for r in batch]
            if gen_dp > 1 and len(prompts) < gen_dp:
                prompts = _pad_batch_for_dp(prompts, gen_dp)
                pad_n = len(prompts) - len(batch)
            else:
                pad_n = 0
            responses = sampler.sample(prompts, gen_params)
            if pad_n:
                responses = responses[:len(batch)]
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
    print(f'  n={total}, seed={args.seed}, chunk_tokens={args.chunk_tokens}, '
          f'max_model_len={max_model_len}')
    print(f'{"=" * 60}')
    print(f'Overall accuracy: {overall:.4f}  ({correct}/{total})')

    if is_dual:
        tot_checks = sum(r.get('n_checks', 0) for r in debug_records)
        tot_inj = sum(r.get('n_injections', 0) for r in debug_records)
        n_with_inj = sum(1 for r in debug_records if r.get('n_injections', 0) > 0)
        print(f'  checker: {tot_checks} checks, {tot_inj} injections across '
              f'{n_with_inj}/{total} problems')
        n_ctx_full = sum(
            1 for r in debug_records if r.get('stopped_reason') == 'context_full')
        n_sample_fail = sum(
            1 for r in debug_records if r.get('stopped_reason') == 'sample_failed')
        n_unfinished = sum(1 for r in debug_records if not r.get('finished'))
        print(f'  length: context_full={n_ctx_full}/{total}, '
              f'sample_failed={n_sample_fail}/{total}, '
              f'unfinished(no EOS)={n_unfinished}/{total}')

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
