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
_RSI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# recorder.py sits one level up, shared with the code half, and the code half
# itself is a sibling directory. Appended rather than inserted, and behind the
# agentic directory on purpose: both halves have a challenge.py, and the one this
# process means by that name is the agentic one.
sys.path.insert(1, _RSI)
sys.path.append(os.path.join(_RSI, 'code'))
import challenge as C  # noqa: E402
import collect as CODE  # noqa: E402
import train as T  # noqa: E402
from recorder import Recorder  # noqa: E402
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


def collect_agentic(args, sampler, template, slots, recorder: Recorder,
                    out_dir: str) -> Dict[str, Any]:
    """The agentic half of one collection pass; returns its metrics.

    The body of what challenge.py's main() did, minus the resources: the sampler,
    the template, the sandbox pool and the recorder are owned by the caller and
    outlive this.
    """
    run = C.Run(args, sampler, template, slots, recorder)
    started = time.time()
    try:
        run.run()
        # After the loop, not during: what it adds is for the next iteration, and
        # doing it here means a crash in collection does not also lose the bank.
        if args.keyword_expand:
            run.keywords.expand_hard()
    finally:
        run.keywords.save()
        # A Run per iteration means a thread pool per iteration. close_pool cannot
        # do this because the sandbox pool is the one thing that is not per-Run.
        run.api_pool.shutdown(wait=False)
        if run.bank is not None:
            logger.info(f'[rsi] task bank: {run.bank.stats()}')
        # In the finally block because a run that crashed is the one whose numbers
        # are most worth having. Reading groups.jsonl back no longer waits on a
        # close -- the recorder flushes every line as it writes it, and its handles
        # outlive this half now that the code half writes through the same ones.
        metrics = C.collect_metrics(out_dir, run.counts, run.n_launched,
                                    args.solver_rollouts, time.time() - started)
        with open(os.path.join(out_dir, 'challenge_metrics.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f'[rsi] {len(run.kept)}/{run.n_launched} groups kept in '
                    f'{time.time() - started:.0f}s: {metrics["scalars"]}')
    return metrics


def collect_code(args, sampler, template, recorder: Recorder,
                 out_dir: str) -> Dict[str, Any]:
    """The code half of the same pass; returns its metrics.

    Takes no sandbox slot and asks for none. A code problem is checked by running
    its asserts in a subprocess -- milliseconds, against the hundreds a microVM
    round trip costs -- and the difficulty stage runs one per candidate per
    rollout, so routing that through the pool would make it the dominant cost of
    the iteration. The slots stay with the agentic half, whose episodes have
    nowhere else to run at all.
    """
    challenger = CODE.build_challenger(args, sampler, template, recorder=recorder)
    metrics = CODE.collect(args, challenger, recorder)
    with open(os.path.join(out_dir, 'code_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    return metrics


def collect_once(args, sampler, template, slots, out_dir: str) -> Dict[str, Any]:
    """Collect from every task source ``--sides`` names, into one ``out_dir``.

    One recorder for all of them, so the numbering is global and index.jsonl
    interleaves the halves. That is the whole of what makes a mixed step possible:
    train.py groups on ``(side, group_id)`` and never learns that two different
    generators wrote the file it read.

    Each half keeps its own metrics file. Their ``counts`` use the same words for
    different things -- ``groups`` is a set of sibling proposals on one side and a
    single problem's attempts on the other -- and adding those together produces a
    number that means neither. Only the ``scalars`` are merged, and only because
    their names are disjoint by construction: the code half prefixes all of its own.
    """
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir
    recorder = Recorder(out_dir)
    scalars: Dict[str, Any] = {}
    try:
        if 'propose' in args.sides_list or 'solve' in args.sides_list:
            # Named by either side, the agentic pair is collected whole: one build
            # is what produces the task its attempts are graded on, so there is no
            # way to collect the solving side without the proposing one.
            metrics = collect_agentic(args, sampler, template, slots, recorder, out_dir)
            scalars.update(metrics.get('scalars') or {})
        if 'code' in args.sides_list:
            metrics = collect_code(args, sampler, template, recorder, out_dir)
            scalars.update(metrics.get('scalars') or {})
    finally:
        recorder.close()
    return {'scalars': scalars}


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
        # Written by save(save_optimizer=True), which only fires every
        # --save-optimizer-every iterations. Staleness is not a matter of losing a
        # few moments: _load_mcore_optimizer reads latest_checkpointed_iteration.txt
        # and restores the model from that sub-checkpoint too, so an optimizer state
        # older than the weights sitting beside it rolls the weights back to
        # whichever iteration wrote it -- silently, since both come from the same
        # directory. Only the iteration that saved it may load it back.
        saved_at = (start - 1) - (start - 1) % args.save_optimizer_every
        if saved_at != start - 1:
            logger.warning(
                f'[rsi] the optimizer state under {hf_dir} is from iteration '
                f'{saved_at} and the weights are from {start - 1}; loading it would '
                f'take the weights back with it, so it is skipped and Adam starts '
                f'at zero moments. Every {args.save_optimizer_every} iterations is '
                f'a resume point; the others cost the fp32 master residue.')
        elif os.path.exists(os.path.join(hf_dir, 'trainer_state.json')):
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
    # Only if a task source needs them: --sides code boots no microVMs at all,
    # which is 32 fewer machines to wait for and to be billed for. close_pool of
    # an empty list is a no-op, so the teardown below needs no second condition.
    agentic = 'propose' in args.sides_list or 'solve' in args.sides_list
    slots = C.initialize_sandbox(args) if agentic else []
    if not agentic:
        logger.info(f'[rsi] --sides {args.sides!r} names no agentic side, so no '
                    f'sandbox pool is opened')
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
