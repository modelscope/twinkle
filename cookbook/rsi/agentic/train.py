# Copyright (c) ModelScope Contributors. All rights reserved.
"""One GRPO step on what a collection pass left in a run directory.

A library, not a script: rsi.py owns the process, the model and the iteration
loop, and calls in here once per iteration. The model it hands over is resident,
which is the whole point -- see rsi.py's docstring for what the old
process-per-iteration arrangement did to the updates.

There is no filtering here. Every rule about what is worth training on was applied
while collecting -- a group is on disk only if it was kept, and a kept group is
exactly its proposals plus the attempts at its selected task -- so anything this
dropped would be a second, invisible policy on top of that one. What it does
refuse is a trajectory the model cannot be stepped on at all: no logprobs, no
trainable token, a logprob count that disagrees with the trainable count, more
tokens than the model accepts, or a group left with fewer than two members. Each
refusal is named and counted in the summary rather than folded into a total.

One step per collection, not several: every trajectory was sampled from one set of
weights, so a second step would be training weights that no longer produced their
own data, ``old_logps`` would stop matching, and epsilon would start clipping for a
reason that has nothing to do with the policy being wrong.
"""
import collections
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from twinkle import DeviceMesh, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.processor import InputProcessor

logger = get_logger()


def build_model(*, model_id: str, model_gpus: int, lr: float, template: str,
                max_length: int):
    """The resident trainer. Full-parameter: no adapter, so the checkpoint is a
    whole model rather than something to merge before the next iteration.
    """
    from twinkle.model.megatron import MegatronModel
    # variable_seq_lengths stays off with padding_free: both switches send
    # collate_fn down the packed path, and Megatron's TE extension then reads
    # PackedSeqParams.pad_between_seqs, which this Megatron-LM checkout does not
    # define. Padded batches cost throughput but keep attention on plain sequences.
    model = MegatronModel(
        model_id=model_id,
        device_mesh=DeviceMesh.from_sizes(world_size=model_gpus, dp_size=model_gpus),
        remote_group='model', mixed_precision='bf16', variable_seq_lengths=False)
    model.set_optimizer('default', lr=lr)
    # 'constant' rather than the default cosine, and this matters here in a way it
    # did not when each iteration was its own process: the scheduler is stepped
    # after every optimizer step and lr_decay_steps=1 would put the second step
    # and everything after it at min_lr, which is 0. Constant returns max_lr
    # before that check is reached, so every iteration steps at the same rate.
    model.set_lr_scheduler('default', lr_decay_steps=1, max_lr=lr,
                           lr_decay_style='constant')
    # beta=0: there is no reference model here, and the KL term needs beta>0 AND
    # ref_logps, so any beta above 0 would silently do nothing.
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
    model.set_processor(InputProcessor, padding_free=False)
    model.set_template(template, model_id=model_id, max_length=max_length,
                       enable_thinking=True)
    # approx_kl on landed data is the check for whether this collection belongs to
    # the weights being trained: it should start near zero, and a large value means
    # the sampler was not holding these weights.
    model.add_metric('GRPOMetric', is_training=True, epsilon=0.2)
    return model


def load(run_dir: str, *, sides: str, max_length: int) -> tuple:
    """Read the index into GRPO groups; returns (groups, skipped).

    A group is ``(side, group_id)`` for the proposing side and
    ``(side, group_id, proposal_idx)`` for the solving side -- one prompt answered
    several times, which is what an advantage is computed over.
    """
    traj_dir = os.path.join(run_dir, 'trajs')
    index = os.path.join(traj_dir, 'index.jsonl')
    if not os.path.exists(index):
        raise SystemExit(f'[train] no {index}')
    wanted = {'both': ('propose', 'solve')}.get(sides, (sides, ))
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
            if ids.size > max_length:
                # Dropped here rather than arriving as an exception in the middle
                # of a step. Reachable in normal operation: collection samples at
                # max_model_len 40960, which is above this.
                skipped[f'longer than max_length={max_length}'] += 1
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


def train_one_step(model, run_dir: str, *, sides: str, max_length: int,
                   micro_batch_size: int, mini_batch_size: int,
                   lr: float) -> Dict[str, Any]:
    """Accumulate everything in ``run_dir`` into one optimizer step.

    Writes train_summary.json next to the collection it trained on and returns it.
    """
    groups, skipped = load(run_dir, sides=sides, max_length=max_length)
    if not groups:
        raise SystemExit(f'[train] nothing trainable in {run_dir}: {dict(skipped)}')
    skipped.update(score(groups))
    batch = [m for g in interleave(groups) for m in g['members']]
    mix = collections.Counter(m['side'] for m in batch)
    sizes = collections.Counter((g['side'], len(g['members'])) for g in groups)
    logger.info(f'[train] {len(groups)} groups, {len(batch)} trajectories {dict(mix)}; '
                f'group sizes {dict(sizes)}')
    for note, n in sorted(skipped.items()):
        logger.warning(f'[train] skipped: {note} x{n}')

    inputs = [{k: m[k] for k in ('input_ids', 'labels', 'attention_mask', 'position_ids')}
              for m in batch]
    old_logps = [m['logps'] for m in batch]
    advantages = [m['advantage'] for m in batch]
    dropped = 0
    for lo in range(0, len(inputs), mini_batch_size):
        hi = min(lo + mini_batch_size, len(inputs))
        # A tail shorter than a whole mini batch is dropped rather than handed
        # over: dispatch 'slice_dp' splits it across all ranks, and a batch that
        # cannot give every rank its own micro batch raises inside _dispatch_args
        # before collate_fn ever runs.
        if hi - lo < mini_batch_size:
            dropped = hi - lo
            logger.warning(f'[train] dropping the last {dropped} trajectories, under '
                           f'the mini batch of {mini_batch_size}')
            break
        model.forward_backward(inputs=inputs[lo:hi], old_logps=old_logps[lo:hi],
                               advantages=advantages[lo:hi],
                               micro_batch_size=micro_batch_size)
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
        'learning_rate': lr,
        'metrics': log,
        'high_kl_records': high_kl or [],
        # Named, not summed: a collection that lost half its trajectories to one
        # reason and one that lost none read the same from the metrics alone.
        'skipped': dict(skipped),
    }
    with open(os.path.join(run_dir, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


# swanlab state for this process. ``init`` may be called once and only once here:
# a second call raises 'DataPorter instance already exists', which the old
# process-per-iteration arrangement never hit because every iteration was a fresh
# interpreter. So it happens once, at startup, before anything expensive has been
# built -- a dashboard that will not accept this client is worth finding out about
# in the first second rather than after the first iteration.
def init_swanlab(*, tag: str, project: str, mode: str, config: Dict[str, Any]) -> None:
    """Open the one experiment this process logs to. Raises if it cannot.

    One experiment for the whole loop rather than one per iteration: the question
    these charts answer is whether iteration k+1 is better than k, which a chart
    that ends after one point cannot show. ``id`` is the tag, so a later process
    under the same tag appends to its curve and a new tag starts a new one.
    """
    import swanlab
    swanlab.init(project=project, name=tag, id=tag, resume='allow', mode=mode,
                 config={'tag': tag, **config})
    logger.info(f'[train] swanlab {project}/{tag}, mode {mode}')


def upload(challenge: Dict[str, Any], summary: Dict[str, Any], *,
           iteration: int) -> None:
    """Send one iteration's numbers to the experiment ``init_swanlab`` opened.

    Only ``challenge['scalars']`` goes up, not its counts: those have keys that
    exist in one iteration and not the next (``group_dropped:rubric_error``), and a
    chart that appears halfway through a run is read as a change in the run rather
    than a change in what was recorded.

    Called after the checkpoint is saved, and it does not swallow anything: the
    connection was proved at startup by ``init_swanlab``, so a failure here is a
    dashboard that went away mid-run and that is worth stopping on. The numbers
    are in challenge_metrics.json and train_summary.json either way.
    """
    import swanlab
    log = {f'challenge/{k}': v for k, v in challenge.items()}
    metrics = summary.get('metrics') or {}
    log.update({
        'train/groups': summary['groups'],
        'train/trajectories': summary['trajectories'],
        'train/trained': summary['trained'],
        'train/dropped_tail': summary['dropped_tail'],
        'train/propose_trajectories': summary['sides'].get('propose', 0),
        'train/solve_trajectories': summary['sides'].get('solve', 0),
        'train/advantage_min': summary['advantage_min'],
        'train/advantage_max': summary['advantage_max'],
        'train/learning_rate': summary['learning_rate'],
        'train/skipped_total': sum(summary['skipped'].values()),
        'train/high_kl_sequences': len(summary['high_kl_records']),
        # Every numeric metric GRPOMetric returned: loss, clip fractions, approx_kl.
        **{f'train/{k}': v for k, v in metrics.items() if isinstance(v, (int, float))},
    })
    swanlab.log(log, step=iteration)
    logger.info(f'[train] swanlab step {iteration}: {len(log)} metrics')
