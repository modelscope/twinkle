"""RFT cold-start for the reflexion skill generator (see reflexion.md §6).

Trains an INDEPENDENT skill model to write reusable, transferable skills that,
when injected into a FROZEN base solver's system prompt, raise the base's pass@k
on problems it first got wrong. The base is never trained — it only produces the
reward signal (marginal pass@k gain). This is STaR/RFT self-bootstrapping: skills
are generated online, only those with a positive marginal (and no answer leak)
are kept, and the skill model is periodically SFT-ed on them, so each round's
generator is a little better than the last.

Direction: skill GENERATION + recall. The skill model reads the problem, the
guidance the solver had, and the solver's own attempt, then DISTILLS the genuinely
useful method / reasoning direction into a short skill list (not a mistake
diagnosis). The distilled ``<skills>`` block is recalled into the base's system
prompt at solve time.

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
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle_agentic.verifier import LeakVerifier

# Reuse the reference eval's dataset + grading + prompts + sampling config, and the
# phase-0 pipeline's parsing / rollout / injection helpers (Find > Create).
from eval_gpqa_rag import (GEN_GPU_MEM, GEN_MODEL_ID, build_direct_prompt,  # noqa: F401
                           load_math)
from eval_reflexion_skill import (SKILL_GEN_USER, _EX_ATTEMPT, _EX_PROBLEM,  # noqa: F401
                                  _EX_SKILLS, _bound_attempt, _clean_text,
                                  _pass_rate, _parse_seq, _run_samples,
                                  build_skill_solve_prompt)

logger = get_logger()

# -- GPU layout ---------------------------------------------------------------
TRAIN_GPUS = int(os.environ.get('TRAIN_GPUS', 4))
SKILL_SAMPLER_GPUS = int(os.environ.get('SKILL_SAMPLER_GPUS', 2))
BASE_SAMPLER_GPUS = int(os.environ.get('BASE_SAMPLER_GPUS', 2))
NUM_GPUS = TRAIN_GPUS + SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS


# ---------------------------------------------------------------------------
# Skill-generation prompt (DISTILL the useful approach, per the new direction)
# ---------------------------------------------------------------------------
SKILL_GEN_SYSTEM = (
    'You are distilling a reusable problem-solving SKILL from one worked episode. '
    'You are shown a competition problem, the guidance the solver was given, and the '
    "solver's own attempt (its reasoning may be partly right and partly wrong).\n\n"
    'Your job: summarize the GENUINELY USEFUL parts — the effective method, the '
    'correct reasoning direction, and the kind of guidance that transfers — into a '
    'short list of reusable skills for SIMILAR problems. Distill the useful approach; '
    'do NOT merely criticise this attempt.\n\n'
    'OUTPUT FORMAT (strict):\n'
    '- You may think first, but the final answer MUST be a markdown bullet list of 3-5 '
    'items WRAPPED IN <skills> and </skills> tags. Output nothing after </skills>.\n'
    '- Each item is ONE short imperative sentence (a method, heuristic, or check).\n'
    '- Inside the tags: no narration, no "The student...", no headings, no restating '
    'the problem.\n\n'
    'CONTENT RULES (strict):\n'
    '- Do NOT reveal the final answer or the multiple-choice option.\n'
    '- Do NOT state the specific numbers, values, or key intermediate results of THIS '
    'problem.\n'
    '- Every item must be GENERAL and transferable, not a step-by-step solution to '
    'THIS problem.\n\n'
    'Follow the example below for the exact tags, style, and level of generality.'
)


def build_skillgen_prompt(problem: str, attempt: str) -> Dict[str, Any]:
    """Skill-gen chat prompt: system + one-shot format demo + the real episode."""
    return {'messages': [
        {'role': 'system', 'content': SKILL_GEN_SYSTEM},
        {'role': 'user',
         'content': SKILL_GEN_USER.format(problem=_EX_PROBLEM, attempt=_EX_ATTEMPT)},
        {'role': 'assistant', 'content': _EX_SKILLS},
        {'role': 'user',
         'content': SKILL_GEN_USER.format(problem=problem, attempt=attempt)},
    ]}


def _extract_skills_block(text: str) -> Optional[str]:
    """Return the inner ``<skills>...</skills>`` block, or None if not parseable.

    RFT keeps thinking ON, so a candidate is usable only if it actually produced a
    CLOSED tag block; unterminated / tag-less generations are dropped (never fall
    back to scraping raw reasoning) — see reflexion.md §6.8.
    """
    low = text.lower()
    if '<skills>' not in low or '</skills>' not in low:
        return None
    start = low.index('<skills>') + len('<skills>')
    end = low.index('</skills>')
    if end <= start:
        return None
    block = text[start:end].strip()
    return block or None


def _bounded_attempt(r: Dict[str, Any], args: argparse.Namespace) -> str:
    """Bound the failed attempt so problem + attempt + skill output fits context.

    Capped by ``--attempt-max-tokens`` so the SFT sequences (which embed the same
    attempt) stay short enough to train; the same bound is used at generation and
    SFT time, keeping train/inference identical.
    """
    prob_est = len(r['problem']) // 2
    budget = max(1024, args.max_model_len - args.skill_max_tokens
                 - args.attempt_reserve_tokens - prob_est)
    budget = min(budget, args.attempt_max_tokens)
    init = r['_init'][0]
    return _bound_attempt(init['text'], init['gen_tokens'], budget)


# ---------------------------------------------------------------------------
# Online data generation (one chunk, all sampler calls batched)
# ---------------------------------------------------------------------------
def generate_chunk(base_sampler, skill_sampler, leak: LeakVerifier,
                   chunk: List[Dict[str, Any]], base_dp: int, skill_dp: int,
                   args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run base-solve -> hard-filter -> skill-gen -> leak-filter -> marginal, and
    return the records whose skill gave a positive marginal pass@k gain."""
    # --- Phase 1: base solves once; keep only what it got wrong. ---
    init_out = _run_samples(
        base_sampler, [build_direct_prompt(r['problem']) for r in chunk],
        args.init_samples, args.max_tokens, base_dp)
    failed = []
    for r, seqs in zip(chunk, init_out):
        r['_init'] = [_parse_seq(s, r['reference_answer']) for s in seqs]
        if _pass_rate(r['_init']) == 0.0:
            failed.append(r)
    sys.stderr.write(f'  phase1: {len(chunk)-len(failed)}/{len(chunk)} solved on '
                     f'first try, {len(failed)} failed\n')
    if not failed:
        return []

    # --- Phase 2: baseline pass@k on the failures -> keep the genuinely hard. ---
    base_out = _run_samples(
        base_sampler, [build_direct_prompt(r['problem']) for r in failed],
        args.pass_k, args.max_tokens, base_dp)
    hard = []
    for r, seqs in zip(failed, base_out):
        r['_baseline_pass'] = _pass_rate([_parse_seq(s, r['reference_answer']) for s in seqs])
        if r['_baseline_pass'] <= args.hard_baseline_max:
            hard.append(r)
    sys.stderr.write(f'  phase2: {len(hard)}/{len(failed)} failures are hard '
                     f'(pass@{args.pass_k} <= {args.hard_baseline_max})\n')
    if not hard:
        return []

    # --- Phase 3: skill-gen (thinking ON) -> require a parseable <skills> block. ---
    sg_out = _run_samples(
        skill_sampler,
        [build_skillgen_prompt(r['problem'], _bounded_attempt(r, args)) for r in hard],
        args.n_skills, args.skill_max_tokens, skill_dp)
    cands: List[Dict[str, Any]] = []  # {r, response, block}
    for r, seqs in zip(hard, sg_out):
        for s in seqs:
            resp = _clean_text(getattr(s, 'decoded', '') or '')
            block = _extract_skills_block(resp)
            if block:
                cands.append({'r': r, 'response': resp, 'block': block})
    if not cands:
        sys.stderr.write('  phase3: 0 parseable skill blocks\n')
        return []

    # --- Phase 4: leak filter via backup teacher (drops answer-leaking skills). ---
    details = leak.leak_batch(
        [{'content': c['block'], 'query': c['r']['problem'],
          'reference': c['r']['reference_answer']} for c in cands],
        max_workers=args.leak_workers)
    clean = [c for c, d in zip(cands, details) if not d.leaked]
    sys.stderr.write(f'  phase4: {len(clean)}/{len(cands)} skills clean '
                     f'({len(cands)-len(clean)} leaked)\n')
    if not clean:
        return []

    # --- Phase 5: with-skill pass@k -> marginal = with - baseline. ---
    ws_out = _run_samples(
        base_sampler,
        [build_skill_solve_prompt(c['r']['problem'], c['block']) for c in clean],
        args.pass_k, args.max_tokens, base_dp)
    records: List[Dict[str, Any]] = []
    for c, seqs in zip(clean, ws_out):
        r = c['r']
        with_pass = _pass_rate([_parse_seq(s, r['reference_answer']) for s in seqs])
        marginal = with_pass - r['_baseline_pass']
        if marginal > 0.0:  # keep only skills that actually helped
            records.append({
                'problem': r['problem'],
                'reference_answer': r['reference_answer'],
                'attempt': _bounded_attempt(r, args),
                'response': c['response'],
                'skills': c['block'],
                'baseline_pass': r['_baseline_pass'],
                'with_pass': with_pass,
                'marginal': marginal,
            })
    sys.stderr.write(f'  phase5: {len(records)} skills with positive marginal kept\n')
    return records


# ---------------------------------------------------------------------------
# Online RFT training
# ---------------------------------------------------------------------------
def _sft_trajectory(rec: Dict[str, Any]) -> Dict[str, Any]:
    """SFT sample = the exact skill-gen prompt + the kept generation as the target.

    Matching the generation prompt keeps train/inference consistent; the template
    masks the prompt to -100 and supervises the assistant turns.
    """
    msgs = build_skillgen_prompt(rec['problem'], rec['attempt'])['messages']
    return {'messages': msgs + [{'role': 'assistant', 'content': rec['response']}]}


def _train_round(skill_model, ckpt: CheckpointEngineManager,
                 pool: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """One RFT round: weighted-sample ``train_batch`` records, SFT them in driver-
    side mini-batches (each an optimizer step, matching short_math_grpo), then sync.

    The transformers backend runs one forward per ``forward_backward`` call (no
    internal micro split), and ``slice_dp`` needs each mini-batch divisible across
    the training dp ranks — hence ``sft_batch_size`` is a multiple of TRAIN_GPUS.
    """
    weights = [max(rec['marginal'], args.weight_eps) ** args.weight_alpha for rec in pool]
    batch = random.choices(pool, weights=weights, k=args.train_batch)
    trajs = [_sft_trajectory(rec) for rec in batch]
    for i in range(0, len(trajs), args.sft_batch_size):
        skill_model.forward_backward(inputs=trajs[i:i + args.sft_batch_size])
        skill_model.clip_grad_and_step()
    ckpt.sync_weights(merge_and_sync=True)  # full-param: push all weights to vLLM


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=2000, help='MATH problems to stream.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--chunk-size', type=int, default=16,
                   help='Problems per generation chunk (all sampler calls batched).')
    p.add_argument('--init-samples', type=int, default=1)
    p.add_argument('--n-skills', type=int, default=8,
                   help='Candidate skills generated per hard problem.')
    p.add_argument('--pass-k', type=int, default=8)
    p.add_argument('--hard-baseline-max', type=float, default=0.25)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192,
                   help='Max generated tokens for solve rollouts.')
    p.add_argument('--skill-max-tokens', type=int, default=4096,
                   help='Max tokens for skill-gen (thinking ON: room for CoT + the '
                        '<skills> block).')
    p.add_argument('--attempt-reserve-tokens', type=int, default=1024)
    p.add_argument('--attempt-max-tokens', type=int, default=3072,
                   help='Hard cap on the failed-attempt tokens embedded in skill-gen '
                        'and SFT prompts (keeps SFT sequences trainable).')
    p.add_argument('--leak-workers', type=int, default=32,
                   help='Parallel workers for the LeakVerifier backup judge.')
    # -- online RFT --
    p.add_argument('--train-every', type=int, default=64,
                   help='Trigger one train round after this many new valid records.')
    p.add_argument('--train-batch', type=int, default=64,
                   help='Records weighted-sampled (with replacement) per train round.')
    p.add_argument('--sft-batch-size', type=int, default=8,
                   help='Driver-side mini-batch per optimizer step; MUST be a multiple '
                        'of TRAIN_GPUS (sliced across training dp ranks) and divide '
                        '--train-batch.')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max-train-rounds', type=int, default=200,
                   help='Cap on train rounds (also sizes the LR schedule).')
    p.add_argument('--save-rounds', type=int, default=50)
    p.add_argument('--weight-alpha', type=float, default=1.0,
                   help='Sampling weight exponent: w = max(marginal, eps) ** alpha.')
    p.add_argument('--weight-eps', type=float, default=0.01)
    p.add_argument('--output-dir', default='./output/reflexion_skill_rft')
    return p.parse_args()


def main() -> None:
    args = _build_args()
    if args.sft_batch_size % TRAIN_GPUS != 0 or args.train_batch % args.sft_batch_size != 0:
        raise ValueError(f'--sft-batch-size ({args.sft_batch_size}) must be a multiple '
                         f'of TRAIN_GPUS ({TRAIN_GPUS}) and divide --train-batch '
                         f'({args.train_batch})')
    steps_per_round = args.train_batch // args.sft_batch_size
    records = load_math(n=args.n, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, 'skill_dataset.jsonl')

    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('OPENAI_API_KEY')):
        sys.stderr.write('[rft] WARNING: no LLM_BACKUP_API_KEY/OPENAI_API_KEY — '
                         'LeakVerifier will report no_llm and skip leak filtering\n')

    # -- Device groups: train (FSDP2) + two independent vLLM samplers. --
    r0, r1, r2 = TRAIN_GPUS, TRAIN_GPUS + SKILL_SAMPLER_GPUS, NUM_GPUS
    device_groups = [
        DeviceGroup(name='train', ranks=list(range(0, r0)), device_type='GPU'),
        DeviceGroup(name='skill_sampler', ranks=list(range(r0, r1)), device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(r1, r2)), device_type='GPU'),
    ]
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups,
                       lazy_collect=False)

    # -- Skill model: full-param FSDP2, causal-LM SFT. --
    train_mesh = DeviceMesh.from_sizes(world_size=TRAIN_GPUS, dp_size=TRAIN_GPUS)
    skill_model = TransformersModel(model_id=GEN_MODEL_ID, device_mesh=train_mesh,
                                    remote_group='train',
                                    ddp_config={'find_unused_parameters': False})
    from twinkle.patch.no_split_modules import NoSplitModulesPatch
    skill_model.apply_patch(NoSplitModulesPatch({'Qwen3_5DecoderLayer'}))
    skill_model.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                             enable_thinking=True, max_length=args.max_model_len,
                             truncation_strategy='delete')
    skill_model.set_processor(InputProcessor, padding_free=True)
    skill_model.set_loss('CrossEntropyLoss')
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
    skill_sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                               enable_thinking=True, max_length=args.max_model_len)
    base_sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={'gpu_memory_utilization': GEN_GPU_MEM,
                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
        device_mesh=DeviceMesh.from_sizes(world_size=BASE_SAMPLER_GPUS, dp_size=base_dp),
        remote_group='base_sampler')
    base_sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                              enable_thinking=True, max_length=args.max_model_len)

    ckpt = CheckpointEngineManager(model=skill_model, sampler=skill_sampler)
    leak = LeakVerifier(sampler=None)  # backup teacher only, no local judge

    sys.stderr.write(f'[rft] {len(records)} MATH problems; train_gpus={TRAIN_GPUS} '
                     f'skill_dp={skill_dp} base_dp={base_dp}\n')

    pool: List[Dict[str, Any]] = []
    pending = 0
    rounds = 0
    n_chunks = (len(records) + args.chunk_size - 1) // args.chunk_size

    with open(data_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps({
            'record_type': 'config', 'model': GEN_MODEL_ID, 'dataset': 'math',
            'n': len(records), 'seed': args.seed, 'pass_k': args.pass_k,
            'hard_baseline_max': args.hard_baseline_max, 'n_skills': args.n_skills,
            'train_every': args.train_every, 'train_batch': args.train_batch,
            'lr': args.lr, 'started': int(time.time()),
        }, ensure_ascii=False) + '\n')
        fout.flush()

        for ci in range(n_chunks):
            if rounds >= args.max_train_rounds:
                break
            chunk = records[ci * args.chunk_size:(ci + 1) * args.chunk_size]
            sys.stderr.write(f'[rft] chunk {ci+1}/{n_chunks} ({len(chunk)} problems)\n')
            recs = generate_chunk(base_sampler, skill_sampler, leak, chunk,
                                  base_dp, skill_dp, args)
            for rec in recs:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
            pool.extend(recs)
            pending += len(recs)

            while pending >= args.train_every and rounds < args.max_train_rounds:
                _train_round(skill_model, ckpt, pool, args)
                pending -= args.train_every
                rounds += 1
                sys.stderr.write(f'[rft] train round {rounds}/{args.max_train_rounds} '
                                 f'(pool={len(pool)})\n')
                if rounds % args.save_rounds == 0:
                    skill_model.save(f'skill-rft-{rounds}', output_dir=args.output_dir)

    skill_model.save('skill-rft-final', output_dir=args.output_dir)
    sys.stderr.write(f'[rft] done: {len(pool)} valid records, {rounds} train rounds; '
                     f'dataset -> {data_path}\n')


if __name__ == '__main__':
    main()
