# Copyright (c) ModelScope Contributors. All rights reserved.
"""The self-play loop as one resident process: collect, step, hand the new weights
to the live sampler, repeat.

This replaces loop.sh, which ran challenge.py and train.py as a fresh pair of
processes per iteration. What that cost, in the order the numbers matter:

Accumulation. loop.sh's only channel between iterations was a bf16 HF checkpoint,
so the trainer's fp32 master weights and its Adam moments were thrown away and
rebuilt every iteration. Measured on v3 after 12 iterations at lr 1e-6: 98.54% of
the 4.02 B weights were still bit-identical to the base model, and the largest
change anywhere was 2.289e-05 -- one bf16 step at that magnitude, and the same
value in eight different tensors, which is quantisation showing through rather
than learning. A step displaces an element by about 2e-6, bf16 near |w|=1e-2
cannot record less than ~4e-5, so each iteration's update was rounded away instead
of added to the last one. Here the optimizer never leaves memory and 12 steps are
12 steps.

Startup. 5.5 minutes of vLLM and 5.4 minutes of Megatron per iteration, about 29%
of a 38-minute iteration, plus 7.6 GB written and ~50 GB read as every sampler
worker reloaded the checkpoint.

Memory. The trainer and the sampler own disjoint GPUs, so neither can starve the
other. Time-sharing all eight cards instead -- vLLM asleep during the step -- would
put 29 GB of resident trainer against ~65 GB of woken vLLM inside 97 GB, on the
machine where a metric gather has already died for want of 200 MB.

The split costs idle capacity: the trainer's cards wait out the ~35 minutes of
collection and the sampler's wait out the ~6 minutes of the step. Collection is
bound by sandbox round trips and API latency rather than generation -- 128
trajectories of at most 1.16 M tokens in 30 minutes is under 700 tok/s across all
engines, far under what a 4B model does on one H20 -- so buying wall-clock with
sampler width is the cheap direction and buying it with trainer width is not.

    python cookbook/rsi/agentic/rsi.py --tag v4

Resuming is by the same marker loop.sh used: iter<n>/iteration.done, written last.
A resident optimizer is state that only exists in memory, so it is checkpointed
every --save-optimizer-every iterations; a crash between two of those resumes with
the weights but with Adam starting from zero moments, which is the old behaviour
for exactly one step rather than for every step.
"""
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import twinkle
from twinkle import DeviceGroup, get_device_placement, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import challenge as C  # noqa: E402
import train as T  # noqa: E402
from sandbox import close_pool  # noqa: E402

logger = get_logger()


def next_iteration(root: str) -> int:
    """The first iteration with no ``iteration.done``.

    Counted from the marker rather than from what is on disk: a directory exists
    as soon as collection starts writing into it, and a train_summary.json is
    there after a step whose checkpoint may not have been saved.
    """
    i = 1
    while os.path.exists(os.path.join(root, f'iter{i}', 'iteration.done')):
        i += 1
    return i


def collect_once(args, sampler, template, slots, out_dir: str) -> Dict[str, Any]:
    """One collection pass into ``out_dir``; returns its metrics.

    The body of what challenge.py's main() did, minus the resources: the sampler,
    the template and the sandbox pool are owned by the caller and outlive this.
    """
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir
    recorder = C.Recorder(out_dir)
    run = C.Run(args, sampler, template, slots, recorder)
    started = time.time()
    try:
        run.run()
        # After the loop, not during: what it adds is for the next iteration, and
        # doing it here means a crash in collection does not also lose the bank.
        if args.keyword_expand:
            run.expand_hard_keywords()
    finally:
        recorder.close()
        run.store.save()
        # A Run per iteration means a thread pool per iteration. close_pool cannot
        # do this because the sandbox pool is the one thing that is not per-Run.
        run.api_pool.shutdown(wait=False)
        if run.bank is not None:
            logger.info(f'[rsi] task bank: {run.bank.stats()}')
        # In the finally block because a run that crashed is the one whose numbers
        # are most worth having, and after recorder.close() so groups.jsonl is
        # flushed before collect_metrics reads it back.
        metrics = C.collect_metrics(out_dir, run.counts, run.n_launched,
                                    args.solver_rollouts, time.time() - started)
        with open(os.path.join(out_dir, 'challenge_metrics.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f'[rsi] {len(run.kept)}/{run.n_launched} groups kept in '
                    f'{time.time() - started:.0f}s: {metrics["scalars"]}')
    return metrics


def main():
    args = C.parse_args()
    root = os.path.join(args.root, args.tag)
    os.makedirs(root, exist_ok=True)
    ckpt_dir = args.ckpt_dir or os.path.join(root, 'ckpt')
    # save() writes <output_dir>/<name>, and --model-id takes an HF directory, so
    # the next start reads back exactly what the last one wrote.
    hf_dir = os.path.join(ckpt_dir, 'model')

    start = next_iteration(root)
    model_id, resume_from = args.model_id, None
    if start > 1:
        if not os.path.exists(os.path.join(hf_dir, 'config.json')):
            raise SystemExit(
                f'[rsi] {start - 1} iteration(s) finished under {root} but there '
                f'is no checkpoint at {hf_dir}. One directory holds the whole '
                f'loop and each save overwrites the last, so those weights are '
                f'gone: start a new --tag, or delete the iteration.done markers '
                f'to redo them from {args.model_id}.')
        model_id = hf_dir
        # Written by save(save_optimizer=True). Its absence is not an error, it
        # means the crash landed between two optimizer checkpoints.
        if os.path.exists(os.path.join(hf_dir, 'trainer_state.json')):
            resume_from = hf_dir
        else:
            logger.warning(f'[rsi] no optimizer state in {hf_dir}; resuming from '
                           f'the weights with Adam at zero moments')

    total_gpus = args.model_gpus + args.sampler_gpus
    logger.info(f'[rsi] tag {args.tag}, iterations from {start}'
                f'{"" if not args.iterations else f" for {args.iterations}"}, '
                f'{args.model_gpus} trainer + {args.sampler_gpus} sampler GPUs, '
                f'model {model_id}, checkpoint {hf_dir}, lr {args.lr}')

    # Before the GPUs: a dashboard that will not accept this client is worth
    # finding out about now rather than 35 minutes in, and there is nothing to
    # lose yet if it raises.
    T.init_swanlab(tag=args.tag, project=args.swanlab_project,
                   mode=args.swanlab_mode,
                   config={'model_id': args.model_id, 'learning_rate': args.lr,
                           'sides': args.sides, 'model_gpus': args.model_gpus,
                           'sampler_gpus': args.sampler_gpus})

    # Both groups are named here, once, and every remote object below is pinned to
    # one of them. Disjoint rank ranges are what keeps the two halves from sharing
    # a card.
    twinkle.initialize(
        mode='ray', nproc_per_node=total_gpus, lazy_collect=False,
        groups=[
            DeviceGroup(name='model', ranks=list(range(args.model_gpus)),
                        device_type='GPU'),
            DeviceGroup(name='sampler', ranks=list(range(args.model_gpus, total_gpus)),
                        device_type='GPU'),
        ])

    model = T.build_model(model_id=model_id, model_gpus=args.model_gpus, lr=args.lr,
                          template=args.template, max_length=args.max_train_len)
    if resume_from:
        state = model.resume_from_checkpoint(resume_from)
        logger.info(f'[rsi] optimizer resumed from {resume_from}: {state}')
    sampler, template = C.build_sampler(args)
    # Model rank 0 serves the TCPStore the sampler ranks connect to, so this must
    # be built after both halves exist. Its first call is what sends the weights.
    weights = CheckpointEngineManager(model=model, sampler=sampler)
    slots = C.initialize_sandbox(args)
    logger.info(get_device_placement())

    i = start
    try:
        while not args.iterations or i < start + args.iterations:
            out_dir = os.path.join(root, f'iter{i}')
            logger.info(f'[rsi] iteration {i}: collect -> {out_dir}')
            challenge_metrics = collect_once(args, sampler, template, slots, out_dir)

            logger.info(f'[rsi] iteration {i}: train on {out_dir}')
            summary = T.train_one_step(
                model, out_dir, sides=args.sides, max_length=args.max_train_len,
                micro_batch_size=args.micro_batch_size,
                mini_batch_size=args.mini_batch_size or args.model_gpus * args.micro_batch_size,
                lr=args.lr)

            # The whole point of one process: the weights go to the engines that
            # are already running, over NCCL, instead of through the filesystem.
            # merge_and_sync=True is the full-parameter path -- there is no adapter
            # here, so the merge is a no-op and every weight is sent.
            t0 = time.time()
            weights.sync_weights(merge_and_sync=True)
            # The cache holds keys computed under the old weights. Cheap to drop,
            # and wrong to keep.
            sampler.reset_prefix_cache()
            logger.info(f'[rsi] iteration {i}: weights synced to the sampler in '
                        f'{time.time() - t0:.1f}s')

            with_optimizer = (i % args.save_optimizer_every == 0)
            t0 = time.time()
            model.save('model', output_dir=ckpt_dir, save_optimizer=with_optimizer)
            logger.info(f'[rsi] iteration {i}: checkpoint at {hf_dir} in '
                        f'{time.time() - t0:.0f}s'
                        f'{" with optimizer state" if with_optimizer else ""}')

            T.upload(challenge_metrics.get('scalars') or {}, summary, iteration=i)
            # Last, so a resume counts only iterations whose weights are on disk.
            open(os.path.join(out_dir, 'iteration.done'), 'w').close()
            logger.info(f'[rsi] iteration {i} done')
            i += 1
    finally:
        rebuilds = close_pool(slots)
        if rebuilds:
            logger.warning(f'[rsi] sandboxes were rebuilt {rebuilds} time(s); the '
                           f'jobs in flight at those moments were lost')
        logger.info(f'[rsi] stopped after iteration {i - 1}; model at {hf_dir}')


if __name__ == '__main__':
    main()
