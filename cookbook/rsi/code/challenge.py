# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI self-play, code half: generate training problems with a local sampler.

One model plays both roles. It writes a problem plus a reference solution; the
solution is executed to turn the problem's check expressions into asserts; then
the same model attempts the problem several times and only problems it solves
*sometimes* are kept -- an all-pass or all-fail group gives GRPO nothing to learn
from.

The machinery lives in :mod:`twinkle_agentic.challenger`; the prompts live in
``prompts.py`` next to this file. What is here is the wiring: which model, how
many, where the output goes.

Output is what ``rsi_rl`` reads directly, no prepare/refine stage in between:

    --out-flows  {id, system, query, tools, rounds:[code round]}
    --out-tests  {id, test_list, test_setup_code}

Run it as a Ray job (sampler only, no trainer)::

    python cookbook/rsi/code/challenge.py --keep-target 500 --seed-file seeds.jsonl
"""
import argparse
import json
import os
import sys

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams, user_data_get
from twinkle.sampler import vLLMSampler
from twinkle_agentic.challenger import CodeChallenger, KeywordStore, load_seeds
from twinkle_agentic.rollout import build_rollout
from twinkle_agentic.tools.tool_manager import ToolManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import CATEGORIES, CATEGORY_DESC, code_prompts  # noqa: E402

logger = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are the ones the previous env-var script shipped with, so a run
    # started without flags produces what earlier iterations produced.
    p.add_argument('--model-id', default='ms://Qwen/Qwen3-4B')
    p.add_argument('--template', default='Template',
                   help='template class in twinkle.template; the text one for Qwen3-4B')
    p.add_argument('--sampler-gpus', type=int, default=4)
    p.add_argument('--max-model-len', type=int, default=16384)

    p.add_argument('--keep-target', type=int, default=500,
                   help='how many problems to keep; generation stops once reached')
    p.add_argument('--batch-size', type=int, default=0,
                   help='problems per written batch (0 = one batch of --keep-target)')
    p.add_argument('--max-proposals-per-round', type=int, default=2000,
                   help='ceiling on one proposing round, i.e. one batched generate')
    p.add_argument('--seed-file', default='', help='seed jsonl with query [+ code]')
    p.add_argument('--seed-mix-prob', type=float, default=0.5)
    p.add_argument('--no-two-step', action='store_true',
                   help='never take the two-call path, even for seeds carrying code')

    p.add_argument('--propose-temp', type=float, default=1.1)
    p.add_argument('--propose-max-tokens', type=int, default=8192)
    p.add_argument('--problem-max-chars', type=int, default=4000)

    p.add_argument('--keywords-n', type=int, default=128,
                   help='per-category refill target; 0 disables the keyword bank')
    p.add_argument('--keyword-db', default='output/rsi/keywords.jsonl')
    p.add_argument('--keyword-gen-calls', type=int, default=8)
    p.add_argument('--keyword-refill-tries', type=int, default=2)
    p.add_argument('--keyword-temp', type=float, default=1.3)
    p.add_argument('--keyword-max-tokens', type=int, default=1024)
    p.add_argument('--single-kw-prob', type=float, default=0.1)
    p.add_argument('--combo-arity', default='triple', choices=['triple', 'mix'])
    p.add_argument('--arity-weights', default='',
                   help="'w1,w2,w3' for --combo-arity mix (empty = uniform)")
    p.add_argument('--low-pass-expand', type=int, default=0,
                   help='expand topics of problems solved at most this many times')
    p.add_argument('--expand-per-kw', type=int, default=8)
    p.add_argument('--expand-max-kws', type=int, default=32)

    p.add_argument('--solver-rollouts', type=int, default=8)
    p.add_argument('--solver-temp', type=float, default=1.0)
    p.add_argument('--solver-max-tokens', type=int, default=2048)
    p.add_argument('--keep-min-pass', type=int, default=1)
    p.add_argument('--keep-max-margin', type=int, default=1)

    p.add_argument('--sandbox-timeout', type=int, default=30)
    p.add_argument('--max-checks', type=int, default=6)
    p.add_argument('--keep-constant-answer', action='store_true',
                   help='keep problems where one constant satisfies every assert')
    p.add_argument('--no-sort-by-difficulty', action='store_true',
                   help='write in generation order instead of hardest-last')
    p.add_argument('--random-seed', type=int, default=0)

    p.add_argument('--out-flows', default='output/rsi/challenge_flows.jsonl')
    p.add_argument('--out-tests', default='output/rsi/challenge_tests.jsonl')
    p.add_argument('--dump-rejected', default='output/rsi/challenge_rejected.jsonl')
    return p.parse_args()


def main():
    args = parse_args()
    for path in (args.out_flows, args.out_tests, args.dump_rejected, args.keyword_db):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    twinkle.initialize(
        mode='ray', nproc_per_node=args.sampler_gpus, lazy_collect=False,
        groups=[DeviceGroup(name='sampler', ranks=list(range(args.sampler_gpus)),
                            device_type='GPU')])
    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': args.max_model_len},
        device_mesh=DeviceMesh.from_sizes(world_size=args.sampler_gpus,
                                          dp_size=args.sampler_gpus),
        remote_group='sampler',
    )
    sampler.set_template(args.template, model_id=args.model_id, enable_thinking=True,
                         max_length=args.max_model_len)

    import twinkle.template as template_module
    template = getattr(template_module, args.template)(
        args.model_id, max_length=args.max_model_len, enable_thinking=True)
    # Single-turn generation, but through the same rollout the RL loop uses, so a
    # challenger that should be allowed to run code while inventing only needs a
    # tool manager here rather than a different code path.
    explorer = build_rollout(
        sampler,
        template=template,
        tool_manager=ToolManager([]),
        max_turns=1,
        sampling_params=SamplingParams(max_tokens=args.propose_max_tokens, num_samples=1,
                                       logprobs=1, temperature=args.propose_temp, top_p=0.95),
    )

    store = None
    if args.keywords_n > 0:
        store = KeywordStore(args.keyword_db, CATEGORIES)
        logger.info('[challenge] keyword bank loaded: '
                    + ', '.join(f'{c}={len(store.items[c])}' for c in CATEGORIES))

    seeds = load_seeds(args.seed_file)
    logger.info(f'[challenge] seeds: {len(seeds)} from {args.seed_file!r} '
                f'(seed_mix_prob={args.seed_mix_prob if seeds else 0.0})')

    rejected = open(args.dump_rejected, 'w', encoding='utf-8') if args.dump_rejected else None

    def _reject(record):
        if rejected is not None:
            rejected.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    challenger = CodeChallenger(
        code_prompts(),
        explorer,
        seeds=seeds,
        keyword_store=store,
        category_desc=CATEGORY_DESC if store else None,
        seed_mix_prob=args.seed_mix_prob,
        two_step=not args.no_two_step,
        combo_arity=args.combo_arity,
        arity_weights=[float(x) for x in args.arity_weights.split(',')] if args.arity_weights
        else None,
        single_kw_prob=args.single_kw_prob,
        keyword_refill_target=args.keywords_n,
        keyword_gen_calls=args.keyword_gen_calls,
        keyword_refill_tries=args.keyword_refill_tries,
        keyword_params=SamplingParams(max_tokens=args.keyword_max_tokens, num_samples=1,
                                      logprobs=1, temperature=args.keyword_temp, top_p=0.98),
        # A batch smaller than the sampler's data-parallel width leaves workers idle.
        min_batch=args.sampler_gpus,
        problem_max_chars=args.problem_max_chars,
        max_checks=args.max_checks,
        sandbox_timeout=args.sandbox_timeout,
        drop_constant_answer=not args.keep_constant_answer,
        low_pass_expand=args.low_pass_expand,
        expand_per_kw=args.expand_per_kw,
        expand_max_kws=args.expand_max_kws,
        reject_sink=_reject,
        max_proposals_per_round=args.max_proposals_per_round,
        solver_rollouts=args.solver_rollouts,
        keep_min_pass=args.keep_min_pass,
        keep_max_pass_margin=args.keep_max_margin,
        solver_params=SamplingParams(max_tokens=args.solver_max_tokens, num_samples=1,
                                     logprobs=1, temperature=args.solver_temp, top_p=0.95),
        seed=args.random_seed,
    )

    batch_size = args.batch_size or args.keep_target
    kept = []
    for batch in challenger(batch_size=batch_size, total=args.keep_target):
        kept.extend(batch)
        logger.info(f'[challenge] kept {len(kept)}/{args.keep_target} so far; '
                    f'proposal stats {challenger.stats}')
    if rejected is not None:
        rejected.close()

    if store is not None:
        challenger.expand_hard_keywords()
        store.save()
        logger.info('[challenge] keyword bank saved: '
                    + ', '.join(f'{c}={len(store.items[c])}' for c in CATEGORIES)
                    + f' -> {args.keyword_db}')

    # File order = decreasing pass count, i.e. hardest last. The fixed-pool
    # validation in rsi_rl relies on this ordering.
    if not args.no_sort_by_difficulty:
        kept.sort(key=lambda t: -(user_data_get(t.get('user_data'), 'n_pass', 0) or 0))

    write_flows(kept, args)
    logger.info(f'[challenge] wrote {len(kept)} problems -> {args.out_flows} + {args.out_tests}')
    dist = {}
    for task in kept:
        n = user_data_get(task.get('user_data'), 'n_pass', 0)
        dist[n] = dist.get(n, 0) + 1
    logger.info(f'[challenge] kept pass-count distribution: {dict(sorted(dist.items()))}')


def write_flows(kept, args):
    """Write the two files rsi_rl reads: one flow and one test row per problem."""
    with open(args.out_flows, 'w', encoding='utf-8') as ff, \
            open(args.out_tests, 'w', encoding='utf-8') as ft:
        for i, task in enumerate(kept):
            data = task.get('user_data')
            cid = f'ch_{i:06d}'
            messages = task.get('messages') or []
            system = next((m for m in messages if m.get('role') == 'system'), None)
            query = next((m for m in messages if m.get('role') == 'user'), None)
            flow = {
                'id': cid,
                'system': system,
                'query': query,
                'tools': [],
                # Difficulty audit, ignored by rsi_rl: how many solver attempts
                # passed, so a stored flow can be analysed without re-running.
                'n_pass': user_data_get(data, 'n_pass'),
                'n_rollouts': user_data_get(data, 'n_rollouts'),
                'keywords': user_data_get(data, 'keywords', []),
                'seeded': user_data_get(data, 'seeded', False),
                'two_step': user_data_get(data, 'two_step', False),
                'rounds': [{
                    'intent': 'solve the problem',
                    'type': 'code',
                    'tool_call': None,
                    # The challenger's own passing solution; OPSD reads this.
                    'code': user_data_get(data, 'solution', ''),
                    'result': '',
                    'reward_method': 'rubric',
                }],
            }
            ff.write(json.dumps(flow, ensure_ascii=False) + '\n')
            ft.write(json.dumps({'id': cid,
                                 'test_list': user_data_get(data, 'asserts', []),
                                 'test_setup_code': ''}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
