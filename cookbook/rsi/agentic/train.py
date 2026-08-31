# Copyright (c) ModelScope Contributors. All rights reserved.
"""One GRPO step on what challenge.py collected.

Read ``trajs/index.jsonl``, group it, turn rewards into advantages, accumulate the
whole collection into a single optimizer step, and overwrite the checkpoint the
next iteration loads.

There is no filtering here. Every rule about what is worth training on was applied
while collecting -- a group is on disk only if it was kept, and a kept group is
exactly its 8 proposals plus the 8 attempts at its selected task -- so anything
this script dropped would be a second, invisible policy on top of that one. What
it does refuse is a trajectory the model cannot be stepped on at all: no logprobs,
no trainable token, a logprob count that disagrees with the trainable count, more
tokens than the model accepts, or a group left with fewer than two members. Each
refusal is named and counted in the summary rather than folded into a total.

One step, not several: every trajectory here was sampled from one set of weights,
so a second step would be training weights that no longer produced their own data,
``old_logps`` would stop matching, and epsilon would start clipping for a reason
that has nothing to do with the policy being wrong. The cost is update frequency,
one per collection.

    RSI_RUN_DIR=output/rsi_agentic python cookbook/rsi/agentic/train.py \\
        --model_gpus 8 --lr 1e-6
"""
import collections
import json
import os
from typing import Any, Dict, List

import numpy as np

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.cli import CLI
from twinkle.processor import InputProcessor

logger = get_logger()
args = CLI.from_args()

MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3-4B'
MODEL_GPUS = args.infra.model_gpus or 8
# The base text template. Qwen3-4B is text-only and the multimodal subclass
# crashes on encode for it.
TEMPLATE = os.environ.get('RSI_TEMPLATE', 'Template')
# Whatever --lr says, or the CLI's own default. Not written as ``or <number>``:
# the CLI default is never zero, so a fallback here would be dead code that reads
# like the default.
LEARNING_RATE = args.optimizer.learning_rate

# One trajectory per micro batch, because padding_free is off: a micro batch is
# padded to its longest member, so pairing a short solver attempt with a long
# build episode pays for the long one twice. Builds are whole agentic rollouts --
# median 7.5k tokens, up to 15.7k -- and at 2 per micro batch a 31k-token padded
# batch died of CUDA OOM with activation recompute already at its most aggressive
# setting. Read from the environment rather than args.training, whose
# micro_batch_size defaults to 2 rather than None.
MICRO_BATCH_SIZE = int(os.environ.get('RSI_MICRO_BATCH_SIZE', 1))
# forward_backward is declared dispatch='slice_dp', so a mini batch is sliced
# across the data-parallel ranks and each rank collates only its share. That share
# has to hold at least one micro batch, so the floor is MODEL_GPUS * MICRO_BATCH.
MINI_BATCH_SIZE = args.training.mini_batch_size or MODEL_GPUS * MICRO_BATCH_SIZE

RUN_DIR = os.environ.get('RSI_RUN_DIR', 'output/rsi_agentic')
SAVE_DIR = os.environ.get('RSI_SAVE_DIR', 'output/rsi_agentic/ckpt')
SAVE_NAME = os.environ.get('RSI_SAVE_NAME', 'agentic')
# Longest trajectory fed to the model. Above this the model refuses it mid-step.
MAX_MODEL_LEN = int(os.environ.get('RSI_MAX_MODEL_LEN', 32768))
# Which side(s) to train: 'both', 'solve', 'propose'.
SIDES = os.environ.get('RSI_SIDES', 'both')

# swanlab. One experiment for the whole loop rather than one per iteration: the
# question these charts answer is whether iteration k+1 is better than k, which a
# chart that ends after one point cannot show. ``id`` is the tag, so re-running a
# tag appends to its curve and a new tag starts a new one. Both are read from the
# environment because loop.sh is what knows them; the fallbacks parse the run
# directory, which is ``<root>/<tag>/iter<n>``, so a bare ``python train.py`` still
# lands somewhere sensible instead of failing.
SWANLAB_PROJECT = os.environ.get('RSI_SWANLAB_PROJECT', 'twinkle-rsi-agentic')
TAG = os.environ.get('RSI_TAG') or os.path.basename(os.path.dirname(
    os.path.abspath(RUN_DIR)))
ITERATION = int(os.environ.get('RSI_ITER')
                or ''.join(c for c in os.path.basename(os.path.abspath(RUN_DIR))
                           if c.isdigit()) or 0)
# 'disabled' skips it entirely, for a run that should not appear on the dashboard.
SWANLAB_MODE = os.environ.get('RSI_SWANLAB_MODE', 'online')


def upload(challenge: Dict[str, Any], training: Dict[str, Any]) -> None:
    """Send this iteration's numbers to swanlab, as one step.

    Called after the checkpoint is saved, so a swanlab failure costs this
    iteration's charts and not its weights. The cost of that order is the
    reverse: a crash between the step and here loses the numbers, which are still
    on disk in challenge_metrics.json and train_summary.json.

    Only ``challenge['scalars']`` goes up, not ``challenge['counts']``: the counts
    have keys that exist in one iteration and not the next
    (``group_dropped:rubric_error``), and a chart that appears halfway through a
    run is read as a change in the run rather than a change in what was recorded.
    """
    import swanlab
    swanlab.init(project=SWANLAB_PROJECT, name=TAG, id=TAG, resume='allow',
                 mode=SWANLAB_MODE, config={'tag': TAG, 'model_id': MODEL_ID,
                                            'learning_rate': LEARNING_RATE,
                                            'sides': SIDES, 'gpus': MODEL_GPUS})
    log = {f'challenge/{k}': v for k, v in challenge.items()}
    log.update({f'train/{k}': v for k, v in training.items()})
    swanlab.log(log, step=ITERATION)
    logger.info(f'[train] swanlab {SWANLAB_PROJECT}/{TAG} step {ITERATION}: '
                f'{len(log)} metrics')



def load(run_dir: str) -> tuple:
    """Read the index into GRPO groups; returns (groups, skipped).

    A group is ``(side, group_id)`` for the proposing side and
    ``(side, group_id, proposal_idx)`` for the solving side -- one prompt answered
    several times, which is what an advantage is computed over.
    """
    traj_dir = os.path.join(run_dir, 'trajs')
    index = os.path.join(traj_dir, 'index.jsonl')
    if not os.path.exists(index):
        raise SystemExit(f'[train] no {index}; run challenge.py first')
    wanted = {'both': ('propose', 'solve')}.get(SIDES, (SIDES, ))
    skipped: collections.Counter = collections.Counter()
    by_key: Dict[Any, List[Dict[str, Any]]] = collections.OrderedDict()
    with open(index, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped['unparseable index line'] += 1
                continue
            side = record.get('side')
            if side not in wanted:
                skipped[f'side {side!r} not requested'] += 1
                continue
            if not record.get('has_logprobs'):
                # Nothing to compare a new forward pass against, so GRPO has no
                # ratio. Counted rather than dropped silently: a collection that
                # produced many of these is a collection whose sampler was not
                # returning logprobs, which is a wiring fault, not attrition.
                skipped['trajectory has no logprobs'] += 1
                continue
            arrays = np.load(os.path.join(traj_dir, record['npz']))
            ids = arrays['input_ids'].astype(np.int64)
            labels = arrays['labels'].astype(np.int64)
            logps = arrays['logprobs'].astype(np.float64)
            n_train = int((labels != -100).sum())
            if not n_train:
                skipped['no trainable tokens'] += 1
                continue
            if logps.size != n_train:
                # Off by anything here pairs each logprob with the wrong token and
                # the loss still comes out a number, so it is a hard stop rather
                # than something to trim to the shorter of the two.
                skipped[f'logps {logps.size} != trainable {n_train}'] += 1
                continue
            if ids.size > MAX_MODEL_LEN:
                # Dropped before the model is built, so the count is in the log
                # rather than arriving as an exception in the middle of a step.
                # Reachable in normal operation: challenge.py samples at
                # max_model_len 40960, which is above this.
                skipped[f'longer than MAX_MODEL_LEN={MAX_MODEL_LEN}'] += 1
                continue
            key = ((side, record.get('group_id')) if side == 'propose' else
                   (side, record.get('group_id'), record.get('proposal_idx')))
            by_key.setdefault(key, []).append({
                'side': side,
                'input_ids': ids.tolist(),
                # Labels are stored already shifted by one -- labels[i] is the
                # token at input_ids[i+1] -- which is how the sampler wrote them.
                # Passed through untouched; re-deriving them here would be guessing
                # at an alignment that is already correct on disk.
                'labels': labels.tolist(),
                'attention_mask': [1] * len(ids),
                'position_ids': list(range(len(ids))),
                'logps': logps.tolist(),
                'reward': float(record.get('reward') or 0.0),
            })
    groups = []
    for key, members in by_key.items():
        if len(members) < 2:
            # One member means the advantage is the reward minus itself.
            skipped[f'group of {len(members)} (no gradient)'] += 1
            continue
        groups.append({'key': key, 'side': members[0]['side'], 'members': members})
    return groups, skipped


def score(groups: List[Dict[str, Any]]) -> collections.Counter:
    """Advantage per member, in place. Groups may differ in size."""
    advantage_fn = GRPOAdvantage()
    notes: collections.Counter = collections.Counter()
    for group in groups:
        rewards = [m['reward'] for m in group['members']]
        adv = advantage_fn(rewards, num_generations=len(rewards), scale='group').tolist()
        if all(abs(a) < 1e-9 for a in adv):
            # Every member scored the same, so the group cancels out. Reported
            # because it is the one failure that costs a full collection and looks
            # like a successful run: the step happens and moves nothing.
            notes[f'{group["side"]}: group with no gradient after scoring'] += 1
        for member, a in zip(group['members'], adv):
            member['advantage'] = a
    return notes


def interleave(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Spread each side's groups evenly through the order.

    At one optimizer step this changes nothing about the update -- every group is
    in the one step either way -- but the order is what the per-step log reports,
    and a mixed order makes the composition line readable.

    Each side's share of the update is by trajectory count, not by token count.
    GRPOLoss averages per sequence and then across the batch and reports
    num_tokens=0, which puts the model on the path that weights every micro group
    equally rather than dividing by a global token sum. Measured on run_clean9's
    lengths the two readings differ by about 4 points; they are not the same thing.
    """
    by_side: Dict[str, List[Dict[str, Any]]] = collections.OrderedDict()
    for group in groups:
        by_side.setdefault(group['side'], []).append(group)
    if len(by_side) < 2:
        return list(groups)
    marked = [((i + 0.5) / len(gs), g)
              for gs in by_side.values() for i, g in enumerate(gs)]
    marked.sort(key=lambda t: t[0])
    return [g for _, g in marked]


def main():
    groups, skipped = load(RUN_DIR)
    if not groups:
        raise SystemExit(f'[train] nothing trainable in {RUN_DIR}: {dict(skipped)}')
    skipped.update(score(groups))
    batch = [m for g in interleave(groups) for m in g['members']]
    mix = collections.Counter(m['side'] for m in batch)
    sizes = collections.Counter((g['side'], len(g['members'])) for g in groups)
    logger.info(f'[train] {len(groups)} groups, {len(batch)} trajectories {dict(mix)}; '
                f'group sizes {dict(sizes)}')
    if skipped:
        for note, n in sorted(skipped.items()):
            logger.warning(f'[train] skipped: {note} x{n}')

    twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS, lazy_collect=False,
                       groups=[DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)),
                                           device_type='GPU')])
    # Full-parameter: no adapter, so every weight is trained and the checkpoint is
    # a whole model rather than something to merge before the next iteration.
    from twinkle.model.megatron import MegatronModel
    # variable_seq_lengths stays off with padding_free: both switches send
    # collate_fn down the packed path, and Megatron's TE extension then reads
    # PackedSeqParams.pad_between_seqs, which this Megatron-LM checkout does not
    # define. Padded batches cost throughput but keep attention on plain sequences.
    model = MegatronModel(model_id=MODEL_ID, device_mesh=DeviceMesh.from_sizes(
        world_size=MODEL_GPUS, dp_size=MODEL_GPUS), remote_group='model',
        mixed_precision='bf16', variable_seq_lengths=False)
    model.set_optimizer('default', lr=LEARNING_RATE)
    # Inert at one step: the scheduler is read before lr_step advances it, so the
    # update happens at max_lr and there is no second step to decay over. Wired up
    # so that splitting the run into steps would give a decay across them.
    model.set_lr_scheduler('default', lr_decay_steps=1, max_lr=LEARNING_RATE)
    # beta=0: there is no reference model here, and the KL term needs beta>0 AND
    # ref_logps, so any beta above 0 would silently do nothing.
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
    model.set_processor(InputProcessor, padding_free=False)
    model.set_template(TEMPLATE, model_id=MODEL_ID, max_length=MAX_MODEL_LEN,
                       enable_thinking=True)
    # approx_kl on landed data is the check for whether this collection belongs to
    # the weights being trained: it should start near zero, and a large value means
    # the dump came from a different checkpoint.
    model.add_metric('GRPOMetric', is_training=True, epsilon=0.2)
    logger.info(get_device_placement())

    inputs = [{k: m[k] for k in ('input_ids', 'labels', 'attention_mask', 'position_ids')}
              for m in batch]
    old_logps = [m['logps'] for m in batch]
    advantages = [m['advantage'] for m in batch]
    dropped = 0
    for lo in range(0, len(inputs), MINI_BATCH_SIZE):
        hi = min(lo + MINI_BATCH_SIZE, len(inputs))
        # A tail shorter than a whole mini batch is dropped rather than handed
        # over: dispatch 'slice_dp' splits it across all ranks, and a batch that
        # cannot give every rank its own micro batch raises inside _dispatch_args
        # before collate_fn ever runs.
        if hi - lo < MINI_BATCH_SIZE:
            dropped = hi - lo
            logger.warning(f'[train] dropping the last {dropped} trajectories, under '
                           f'the mini batch of {MINI_BATCH_SIZE}')
            break
        model.forward_backward(inputs=inputs[lo:hi], old_logps=old_logps[lo:hi],
                               advantages=advantages[lo:hi],
                               micro_batch_size=MICRO_BATCH_SIZE)
    # Once, after every mini batch: forward_backward neither steps nor zeroes, so
    # the mini batches above simply add their gradients together and one step
    # consumes all of them.
    model.clip_grad_and_step()

    log = model.calculate_metric(is_training=True)
    high_kl = log.pop('_high_kl_records', None)
    logger.info(f'[train] one step over {len(batch) - dropped} trajectories '
                f'adv[{min(advantages):+.3f},{max(advantages):+.3f}] {log}')
    if high_kl:
        logger.warning(f'[train] {len(high_kl)} sequences disagree with the sampler '
                       f'logps; this collection may not be from these weights')
    summary = {
        'groups': len(groups),
        'trajectories': len(batch),
        'trained': len(batch) - dropped,
        'dropped_tail': dropped,
        'sides': dict(mix),
        'group_sizes': {f'{s}:{n}': c for (s, n), c in sizes.items()},
        'advantage_min': min(advantages),
        'advantage_max': max(advantages),
        'learning_rate': LEARNING_RATE,
        'metrics': log,
        'high_kl_records': high_kl or [],
        # Named, not summed: a collection that lost half its trajectories to one
        # reason and one that lost none read the same from the metrics alone.
        'skipped': dict(skipped),
    }
    with open(os.path.join(RUN_DIR, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    model.save(SAVE_NAME, output_dir=SAVE_DIR)
    logger.info(f'[train] checkpoint at {os.path.join(SAVE_DIR, SAVE_NAME)}')

    # The collection's own numbers, written by challenge.py in this same directory.
    # Absent when train.py is pointed at a directory collected before this existed,
    # in which case the training half still goes up alone.
    challenge_path = os.path.join(RUN_DIR, 'challenge_metrics.json')
    challenge: Dict[str, Any] = {}
    if os.path.exists(challenge_path):
        with open(challenge_path, encoding='utf-8') as f:
            challenge = json.load(f).get('scalars') or {}
    else:
        logger.warning(f'[train] no {challenge_path}; uploading training metrics only')
    training = {
        'groups': len(groups),
        'trajectories': len(batch),
        'trained': len(batch) - dropped,
        'dropped_tail': dropped,
        'propose_trajectories': mix.get('propose', 0),
        'solve_trajectories': mix.get('solve', 0),
        'advantage_min': min(advantages),
        'advantage_max': max(advantages),
        'learning_rate': LEARNING_RATE,
        'skipped_total': sum(skipped.values()),
        'high_kl_sequences': len(high_kl or []),
        # Every numeric metric GRPOMetric returned: loss, clip fractions, approx_kl.
        **{k: v for k, v in log.items() if isinstance(v, (int, float))},
    }
    upload(challenge, training)


if __name__ == '__main__':
    main()
