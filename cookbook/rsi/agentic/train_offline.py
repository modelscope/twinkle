"""Offline GRPO on both sides of a challenge run's dump: full-parameter.

``challenge.py`` already generated the problems and solved each one eight times,
and it landed the token ids, the labels and the sampler's logprobs for every one
of those trajectories. This trains on that dump directly -- nothing is generated
here and nothing is re-encoded, so the tokens trained on are byte-for-byte the
tokens that were sampled.

Two sides come out of one run:

* proposing -- one trajectory per proposal, grouped by ``group_id`` (the
  proposals answering one identical prompt). Reward is ``challenger_reward``:
  ``1 - 2|p - 1/2|`` for p the fraction of solver attempts that passed, so a
  proposal is worth most when the solver got it right about half the time.
* solving -- eight trajectories per task, grouped by task. Reward is the check
  script's exit code, 1.0 or 0.0.

Both sides go into the same optimizer step and are weighted only by how many
trainable tokens they carry; no coefficient is applied to either.

Usage, after a run of challenge.py --proposals-per-group 8:

    python cookbook/rsi/agentic/train_offline.py \\
        --run-dir output/rsi_agentic/run_clean10 \\
        --model-id ms://Qwen/Qwen3-4B --model-gpus 8

The saved checkpoint is HF-format weights plus tokenizer, which is what
``challenge.py --model-id`` takes, so the next round of the loop is a shell line
rather than a conversion step.
"""
import collections
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.cli import CLI
from twinkle.processor import InputProcessor

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3-4B'
MODEL_GPUS = args.infra.model_gpus or 8
# Base text template, as in cookbook/rsi/rl.py:102. Qwen3-4B is text-only, and
# the multimodal subclass crashes on encode for it.
TEMPLATE = os.environ.get('RSI_TEMPLATE', 'Template')

# Whatever --lr says, or the CLI's own 1e-5. Not written as ``or <number>``: the
# CLI default is never zero, so a fallback here would be dead code that reads
# like the default.
LEARNING_RATE = args.optimizer.learning_rate
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
SAVE_STEPS = args.training.save_steps or 0

RUN_DIR = os.environ.get('RSI_RUN_DIR', '')
SAVE_DIR = os.environ.get('RSI_SAVE_DIR', 'output/rsi_agentic/ckpt')
SAVE_NAME = os.environ.get('RSI_SAVE_NAME', 'agentic-offline')

# Trajectories per optimizer step. 32 is BATCH_SIZE 4 x NUM_GENERATIONS 8, the
# same as the online rl.py loop, so a step here moves the weights by as much as a
# step there.
STEP_SIZE = int(os.environ.get('RSI_STEP_SIZE', 32))
# Longest trajectory fed to the model. Above this the model refuses it mid-step.
MAX_MODEL_LEN = int(os.environ.get('RSI_MAX_MODEL_LEN', 32768))

# Which side(s) to train. 'both', 'solver', 'proposer'.
SIDES = os.environ.get('RSI_SIDES', 'both')

# Cap on proposing-side groups per run, 0 for no cap. The solving side is held
# constant by challenge.py's --keep-target, but the proposing side is however
# many proposals it took to reach that target, which grows as the model improves
# and fewer of its proposals land in the band. Capping keeps the two sides' share
# of each step the same from run to run; the proposals above the cap still did
# their job of measuring difficulty, they just do not also become training data.
MAX_PROPOSER_GROUPS = int(os.environ.get('RSI_MAX_PROPOSER_GROUPS', 0))

# Same for the solving side, 0 for no cap. ``--keep-target`` stops challenge.py
# once it has that many tasks in the band, but the internal round that crosses
# the target finishes measuring everything it started -- run_clean9's last round
# added 14 tasks to reach 53 from 39 -- so a run overshoots by however much that
# round produced. Capping makes every run's solving side exactly the same size.
MAX_SOLVER_GROUPS = int(os.environ.get('RSI_MAX_SOLVER_GROUPS', 0))

# Where the numbers go. Two files under the run directory:
#   train_summary.json  one object: the settings this run used, what the
#                       challenger collected, and what got trained on
#   train_steps.jsonl   one line per optimizer step, every metric the model
#                       reported plus the batch's own composition
# Written rather than uploaded, and written as they happen rather than at the end,
# so a run that dies partway still leaves the steps it did finish.
SUMMARY_NAME = os.environ.get('RSI_SUMMARY_NAME', 'train_summary.json')
STEPS_NAME = os.environ.get('RSI_STEPS_NAME', 'train_steps.jsonl')

# Solver groups outside this pass-count range carry one reward for all eight
# members, so their advantages are all zero and a forward+backward over them adds
# nothing. The bounds exclude only 0 and n_rollouts, which is what makes this
# 'the groups that have a gradient' rather than a difficulty judgement.
KEEP_MIN_PASS = int(os.environ.get('RSI_KEEP_MIN_PASS', 1))
KEEP_MAX_PASS_MARGIN = int(os.environ.get('RSI_KEEP_MAX_MARGIN', 1))


def _cap(groups: List[Dict[str, Any]], limit: int, side: str,
         notes: collections.Counter) -> List[Dict[str, Any]]:
    """Hold a side's group count to ``limit`` so every run trains on the same amount.

    Kept in file order, which is the order they came out of the challenger. The
    weights do not change inside one challenge.py run, so the groups dropped are
    not different in kind from the ones kept -- they are just later.
    """
    if not limit:
        return groups
    if len(groups) > limit:
        notes[f'{side} groups over the cap of {limit}'] += len(groups) - limit
        return groups[:limit]
    if len(groups) < limit:
        # Worth saying out loud: this run trains on less than the cap promises, so
        # its steps are not mixed the same way as a run that reached it.
        logger.warning(f'[offline] only {len(groups)} {side} groups, below the cap '
                       f'of {limit}; this run is not comparable to one that '
                       f'reached the cap')
    return groups


def _collection_metrics(run_dir: str) -> Dict[str, float]:
    """Summarise what the challenger produced, before any of the training filters.

    Read off the two dumps rather than from anything challenge.py logs, so these
    numbers describe the whole collection and not the subset that survived the
    pass band and the caps.

    Two pass rates are reported and they mean different things. ``acc/all`` is
    over every attempt on every task whose difficulty got measured, including the
    tasks where all eight attempts agreed. ``acc/trained`` is over the attempts on
    the tasks that go into training, and is bounded to [1/8, 7/8] by that band --
    it cannot report an all-correct or all-wrong task even if the model produces
    one. Neither is a capability measurement on its own: the tasks change every
    iteration and the challenger is being trained to push them toward half
    passing, so a flat curve is what success looks like for both.
    """
    m: Dict[str, float] = {}
    idx = os.path.join(run_dir, 'propose_traj', 'index.jsonl')
    if os.path.isfile(idx):
        rows = [json.loads(line) for line in open(idx) if line.strip()]
        outcomes = collections.Counter(r.get('outcome') for r in rows)
        m['collect/proposals'] = len(rows)
        for name, n in outcomes.items():
            m[f'collect/outcome_{name}'] = n
        rewards = [r['challenger_reward'] for r in rows if r.get('challenger_reward') is not None]
        if rewards:
            m['collect/proposer_reward_mean'] = sum(rewards) / len(rewards)

        measured = [r['n_pass'] for r in rows
                    if r.get('n_pass') is not None and r.get('n_rollouts')]
        rollouts = [r['n_rollouts'] for r in rows
                    if r.get('n_pass') is not None and r.get('n_rollouts')]
        if measured:
            n_att = sum(rollouts)
            m['collect/measured_tasks'] = len(measured)
            m['collect/never_measured'] = len(rows) - len(measured)
            m['acc/all'] = sum(measured) / n_att
            dist = collections.Counter(measured)
            for k in range(max(rollouts) + 1):
                m[f'collect/n_pass_{k}'] = dist.get(k, 0)
            # All eight attempts agreeing means one reward for the whole group, so
            # the group mean equals it and every advantage is zero. Tracking the
            # share of tasks like that is tracking how much of the collection did
            # no work.
            flat = sum(1 for p, r in zip(measured, rollouts) if p == 0 or p == r)
            m['collect/zero_gradient_frac'] = flat / len(measured)
            band = [(p, r) for p, r in zip(measured, rollouts)
                    if KEEP_MIN_PASS <= p <= r - KEEP_MAX_PASS_MARGIN]
            if band:
                m['acc/trained'] = sum(p for p, _ in band) / sum(r for _, r in band)
                m['collect/band_tasks'] = len(band)
                # How many proposals it costs to land one trainable task. Expected
                # to climb as the model gets better at its own proposals, which is
                # what makes an iteration take longer than the one before it.
                m['collect/proposals_per_band_task'] = len(rows) / len(band)

    attempts = os.path.join(run_dir, 'solver_attempts.jsonl')
    if os.path.isfile(attempts):
        rows = [json.loads(line) for line in open(attempts) if line.strip()]
        if rows:
            m['collect/solver_attempts'] = len(rows)
            m['collect/solver_truncated_frac'] = \
                sum(1 for r in rows if r.get('truncated')) / len(rows)
    return m


def _load_solver_groups(run_dir: str) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Groups of solver attempts on one task, with their exit-code rewards."""
    path = os.path.join(run_dir, 'solver_attempts.jsonl')
    if not os.path.exists(path):
        return [], collections.Counter({'no solver_attempts.jsonl': 1})
    by_task: Dict[str, List[Dict[str, Any]]] = collections.OrderedDict()
    notes: collections.Counter = collections.Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                notes['unparseable line'] += 1
                continue
            by_task.setdefault(rec['statement'], []).append(rec)

    groups: List[Dict[str, Any]] = []
    for statement, recs in by_task.items():
        rewards = [1.0 if r['check_exit'] == 0 else 0.0 for r in recs]
        n_pass = int(sum(rewards))
        high = len(recs) - KEEP_MAX_PASS_MARGIN
        if not (KEEP_MIN_PASS <= n_pass <= high):
            notes[f'solver n_pass={n_pass} of {len(recs)} (no gradient)'] += 1
            continue
        members = []
        for rec, reward in zip(recs, rewards):
            att = rec.get('attempt') or {}
            members.append({
                'input_ids': att.get('input_ids') or [],
                'labels': att.get('labels') or [],
                'attention_mask': att.get('attention_mask') or [],
                'position_ids': att.get('position_ids') or [],
                # The sampler's own logprobs, in the [[token, logp]] shape the
                # rollout stored them in. Reusing them rather than a fresh
                # forward is what keeps old_logps free of engine differences.
                'logps': [lp[0][1] for lp in (att.get('logprobs') or [])],
                'reward': reward,
            })
        groups.append({'side': 'solver', 'key': statement[:60], 'members': members})
    return _cap(groups, MAX_SOLVER_GROUPS, 'solver', notes), notes


def _load_proposer_groups(run_dir: str) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Groups of proposals answering one prompt, with their 50%-target rewards."""
    d = os.path.join(run_dir, 'propose_traj')
    index = os.path.join(d, 'index.jsonl')
    notes: collections.Counter = collections.Counter()
    if not os.path.exists(index):
        return [], collections.Counter({'no propose_traj/index.jsonl': 1})

    by_group: Dict[Any, List[Dict[str, Any]]] = collections.OrderedDict()
    with open(index) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                notes['unparseable line'] += 1
                continue
            gid = rec.get('group_id')
            if gid is None:
                # Landed by a run from before proposals were grouped. Every such
                # proposal is its own group of one, so its advantage would be
                # zero; counted and skipped rather than trained on as noise.
                notes['proposal has no group_id (pre-grouping run)'] += 1
                continue
            by_group.setdefault(gid, []).append(rec)

    groups: List[Dict[str, Any]] = []
    for gid, recs in by_group.items():
        if len(recs) < 2:
            notes[f'proposer group of {len(recs)} (no gradient)'] += 1
            continue
        members = []
        for rec in recs:
            npz_name = rec.get('npz')
            if not npz_name:
                notes['proposal has no npz (text-only rollout)'] += 1
                continue
            z = np.load(os.path.join(d, npz_name))
            if 'r0_input_ids' not in z.files:
                notes['npz has no r0_input_ids'] += 1
                continue
            ids = z['r0_input_ids'].astype(np.int64)
            # Labels are stored already shifted by one -- labels[i] is the token
            # at input_ids[i+1] -- which is the convention the sampler wrote them
            # in. Passed through untouched; re-deriving them here would be
            # guessing at an alignment that is already correct on disk.
            labels = z['r0_labels'].astype(np.int64)
            logps = z['r0_logprobs'].astype(np.float64) if 'r0_logprobs' in z.files else None
            if logps is None:
                notes['npz has no r0_logprobs'] += 1
                continue
            members.append({
                'input_ids': ids.tolist(),
                'labels': labels.tolist(),
                'attention_mask': [1] * len(ids),
                'position_ids': list(range(len(ids))),
                'logps': logps.tolist(),
                'reward': float(rec.get('challenger_reward') or 0.0),
            })
        if len(members) < 2:
            notes['proposer group lost members to missing arrays'] += 1
            continue
        groups.append({'side': 'proposer', 'key': f'group{gid}', 'members': members})
    return _cap(groups, MAX_PROPOSER_GROUPS, 'proposer', notes), notes


def _score(groups: List[Dict[str, Any]]) -> collections.Counter:
    """Advantage per member, in place. Groups may differ in size."""
    advantage_fn = GRPOAdvantage()
    notes: collections.Counter = collections.Counter()
    for g in groups:
        rewards = [m['reward'] for m in g['members']]
        adv = advantage_fn(rewards, num_generations=len(rewards), scale='group').tolist()
        if all(abs(a) < 1e-9 for a in adv):
            notes[f'{g["side"]}: group with no gradient after scoring'] += 1
        for m, a in zip(g['members'], adv):
            m['advantage'] = a
    return notes


def _interleave(by_side: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Spread each side's groups evenly through the order.

    Both sides have to appear in every step, or a step's update comes from one
    side only and 'weighted by token count' stops describing anything. The ratio
    is not a free choice: it falls out of using all of both sides' groups over
    the same number of steps.
    """
    sides = [s for s in ('solver', 'proposer') if by_side.get(s)]
    if len(sides) < 2:
        return list(by_side.get(sides[0], [])) if sides else []
    # Place each group at its fractional position within its own side, then sort
    # by that position: a side with three times the groups contributes three for
    # every one of the other's, spread out rather than in a block.
    marked: List[Tuple[float, Dict[str, Any]]] = []
    for s in sides:
        gs = by_side[s]
        for i, g in enumerate(gs):
            marked.append(((i + 0.5) / len(gs), g))
    marked.sort(key=lambda t: t[0])
    return [g for _pos, g in marked]


def main():
    if not RUN_DIR:
        raise SystemExit('set RSI_RUN_DIR to a challenge.py output directory')

    collect = _collection_metrics(RUN_DIR)

    by_side: Dict[str, List[Dict[str, Any]]] = {}
    all_notes: collections.Counter = collections.Counter()
    if SIDES in ('both', 'solver'):
        gs, notes = _load_solver_groups(RUN_DIR)
        by_side['solver'] = gs
        all_notes.update(notes)
    if SIDES in ('both', 'proposer'):
        gs, notes = _load_proposer_groups(RUN_DIR)
        by_side['proposer'] = gs
        all_notes.update(notes)

    all_notes.update(_score([g for gs in by_side.values() for g in gs]))

    for side, gs in by_side.items():
        n_traj = sum(len(g['members']) for g in gs)
        sizes = collections.Counter(len(g['members']) for g in gs)
        logger.info(f'[offline] {side}: {len(gs)} groups, {n_traj} trajectories, '
                    f'group sizes {dict(sorted(sizes.items()))}')
    for note, n in all_notes.most_common():
        logger.info(f'[offline] skipped {n}: {note}')
    if not any(by_side.values()):
        raise SystemExit(f'[offline] nothing trainable in {RUN_DIR}')

    order = _interleave(by_side)
    flat: List[Dict[str, Any]] = []
    for g in order:
        for m in g['members']:
            m['side'] = g['side']
            flat.append(m)

    # Trajectories the model would refuse, dropped before anything is loaded so
    # the count is in the log rather than showing up as a mid-step exception.
    kept: List[Dict[str, Any]] = []
    for m in flat:
        n_train = sum(1 for label in m['labels'] if label != -100)
        if not n_train:
            all_notes['no trainable tokens'] += 1
            continue
        if len(m['logps']) != n_train:
            # Off-by-anything here pairs each logprob with the wrong token and
            # the loss still comes out a number, so it has to be a hard stop.
            all_notes[f'logps {len(m["logps"])} != trainable {n_train}'] += 1
            continue
        if len(m['input_ids']) > MAX_MODEL_LEN:
            all_notes[f'longer than MAX_MODEL_LEN={MAX_MODEL_LEN}'] += 1
            continue
        kept.append(m)

    n_steps = (len(kept) + STEP_SIZE - 1) // STEP_SIZE
    mix = collections.Counter(m['side'] for m in kept)
    logger.info(f'[offline] {len(kept)} trainable trajectories '
                f'({dict(mix)}), {n_steps} steps of {STEP_SIZE}')
    for note, n in all_notes.most_common():
        logger.info(f'[offline] skipped {n}: {note}')

    summary = {
        'run_dir': RUN_DIR,
        'config': {'model_id': MODEL_ID, 'lr': LEARNING_RATE, 'step_size': STEP_SIZE,
                   'sides': SIDES, 'keep_min_pass': KEEP_MIN_PASS,
                   'keep_max_pass_margin': KEEP_MAX_PASS_MARGIN,
                   'max_solver_groups': MAX_SOLVER_GROUPS,
                   'max_proposer_groups': MAX_PROPOSER_GROUPS,
                   'mini_batch_size': MINI_BATCH_SIZE,
                   'micro_batch_size': MICRO_BATCH_SIZE,
                   'max_model_len': MAX_MODEL_LEN, 'template': TEMPLATE,
                   'model_gpus': MODEL_GPUS},
        'collect': collect,
        'train': {'trainable_trajectories': len(kept), 'steps': n_steps,
                  **{f'{side}_trajectories': n for side, n in mix.items()},
                  'groups_per_side': {side: len(gs) for side, gs in by_side.items()}},
        # Every reason anything was left out, with its count. Kept in the summary
        # rather than only in the log so a later comparison between iterations can
        # tell a change in the model from a change in how much survived the load.
        'skipped': dict(all_notes),
    }
    summary_path = os.path.join(RUN_DIR, SUMMARY_NAME)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    steps_path = os.path.join(RUN_DIR, STEPS_NAME)
    steps_file = open(steps_path, 'w')
    logger.info(f'[offline] numbers going to {summary_path} and {steps_path}')
    for k in sorted(collect):
        logger.info(f'[offline]   {k} = {collect[k]}')

    device_groups = [DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU')]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS, groups=device_groups,
                       lazy_collect=False)

    # Full-parameter: no adapter is added, so every weight is trained and the
    # checkpoint is a whole model rather than something that needs merging before
    # the next challenger round can load it.
    from twinkle.model.megatron import MegatronModel
    model = MegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
                          mixed_precision='bf16', variable_seq_lengths=True)
    model.set_optimizer('default', lr=LEARNING_RATE)
    model.set_lr_scheduler('default', lr_decay_steps=max(1, n_steps), max_lr=LEARNING_RATE)
    # beta=0: no reference model here, and grpo.py:315 needs beta>0 AND ref_logps
    # for the KL term, so any beta above 0 would silently do nothing.
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template(TEMPLATE, model_id=MODEL_ID, max_length=MAX_MODEL_LEN,
                       enable_thinking=True)
    # approx_kl at the first inner step compares the sampler's logps against the
    # trainer's on the same tokens. On landed data that is the check for whether
    # this dump belongs to the weights being trained: it should start near zero,
    # and a large value means the dump came from a different checkpoint.
    model.add_metric('GRPOMetric', is_training=True, epsilon=0.2)
    logger.info(get_device_placement())

    for step in range(n_steps):
        lo, hi = step * STEP_SIZE, min((step + 1) * STEP_SIZE, len(kept))
        batch = kept[lo:hi]
        inputs = [{'input_ids': m['input_ids'], 'labels': m['labels'],
                   'attention_mask': m['attention_mask'],
                   'position_ids': m['position_ids']} for m in batch]
        old_logps = [m['logps'] for m in batch]
        advantages = [m['advantage'] for m in batch]

        for mb in range(0, len(inputs), MINI_BATCH_SIZE):
            end = min(mb + MINI_BATCH_SIZE, len(inputs))
            model.forward_backward(
                inputs=inputs[mb:end],
                old_logps=old_logps[mb:end],
                advantages=advantages[mb:end],
                micro_batch_size=MICRO_BATCH_SIZE,
            )
        # Once per step, not per mini-batch: forward_backward neither steps nor
        # zeroes, and clip_grad_norm divides by the tokens accumulated across all
        # of them, so every trajectory in the step carries the same weight no
        # matter how the mini-batches split -- which is also how the two sides end
        # up weighted by their token counts and nothing else.
        model.clip_grad_and_step()

        side_mix = collections.Counter(m['side'] for m in batch)
        log = model.calculate_metric(is_training=True)
        # A list of records rather than a number, marked with a leading underscore
        # at grpo.py:451 for that reason. Going to a file, so the records go in
        # whole instead of being reduced to a count.
        high_kl = log.pop('_high_kl_records', None)
        logger.info(f'[offline] step {step + 1}/{n_steps} '
                    f'{len(batch)} traj {dict(side_mix)} '
                    f'adv[{min(advantages):+.3f},{max(advantages):+.3f}] {log}')
        if high_kl:
            logger.warning(f'[offline] step {step + 1}: {len(high_kl)} sequences '
                           f'with high kl against the sampler logps')
        row = {'step': step + 1, **log,
               'trajectories': len(batch),
               'solver': side_mix.get('solver', 0),
               'proposer': side_mix.get('proposer', 0),
               'adv_min': min(advantages), 'adv_max': max(advantages),
               'high_kl_records': high_kl or []}
        steps_file.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        steps_file.flush()
        if SAVE_STEPS and (step + 1) % SAVE_STEPS == 0:
            model.save(f'{SAVE_NAME}-step{step + 1}', output_dir=SAVE_DIR)

    steps_file.close()
    model.save(SAVE_NAME, output_dir=SAVE_DIR)
    logger.info(f'[offline] done, {n_steps} steps; checkpoint at '
                f'{os.path.join(SAVE_DIR, SAVE_NAME)}')


if __name__ == '__main__':
    main()
