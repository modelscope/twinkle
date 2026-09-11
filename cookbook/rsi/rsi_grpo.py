"""RSI in one file: the model invents its own tasks, and that same work trains it.

One challenger batch, and everything it produced is used:

  1. AgenticChallenger acts in a workspace, writes a check script that verifies
     what it produced, then states the task someone else would be given.
  2. Each surviving statement is attempted ``num_solver_rollouts`` times to
     measure how hard it is; only tasks whose pass count lands inside
     ``pass_band`` are kept.
  3. Both halves of that are trained on, in the same optimizer step: the
     proposing episodes against the pass rate they achieved, the attempts against
     the task's own check script. Nothing is rolled out a second time.

Seeds are optional inspiration, not training data: what a round proposes *about*
comes from the seeders in ``seed/``, asked once per round and appended to the
challenger's opening instruction -- a keyword pool the model fills itself, and the
rows of a hub dataset read straight off with ``twinkle.Dataset``, no parquet to
prepare. A run adds a kind of variety by adding a seeder, not by teaching the
challenger about it.

Usage:
    python cookbook/rsi/rsi_grpo.py
    python cookbook/rsi/rsi_grpo.py --seed-dataset ms://mlabonne/ToolACE --pass-band 1,3
"""
import collections
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

# One sample call in flight per workspace: the challenger runs one trajectory per
# job and holds a workspace for the whole of it, so a cap below ``--num-envs``
# leaves workspaces standing still, waiting their turn at the sampler. Set here
# because the cap is read when vLLMSampler is defined, which is before there are
# any parsed arguments to read it from -- a run with more workspaces than this
# raises the variable too.
os.environ.setdefault('TWINKLE_SAMPLER_MAX_CONCURRENCY', '32')

import twinkle  # noqa: E402
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger  # noqa: E402
from twinkle.advantage import GRPOAdvantage  # noqa: E402
from twinkle.checkpoint_engine import CheckpointEngineManager  # noqa: E402
from twinkle.cli import CLI  # noqa: E402
from twinkle.data_format import SamplingParams, Trajectory, user_data_get  # noqa: E402
from twinkle.dataset import Dataset, DatasetMeta  # noqa: E402
from twinkle.metric import CompletionRewardMetric  # noqa: E402
from twinkle.model import TransformersModel  # noqa: E402
from twinkle.processor import InputProcessor  # noqa: E402
from twinkle.sampler import vLLMSampler  # noqa: E402
from twinkle.template import Qwen3_5Template  # noqa: E402
from twinkle_agentic.agents import MsAgent  # noqa: E402
from twinkle_agentic.challenger import AgenticChallenger, ChallengeBatch  # noqa: E402
from twinkle_agentic.envs import AgentEnv, LocalEnv  # noqa: E402
from twinkle_agentic.protocol.openai import OpenAI  # noqa: E402
from twinkle_agentic.rollout import ExternalRollout  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import brittle_check_reason  # noqa: E402
from seed import ChainSeeder, KeywordSeeder, Seeder, TrajectorySeeder  # noqa: E402

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
USE_MEGATRON = args.model.strategy != 'native_fsdp'

MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
# The KL anchor, when there is one: frozen weights on their own GPUs.
REF_GPUS = args.infra.ref_model_gpus or 0
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS + REF_GPUS

MAX_NEW_TOKENS = args.sampling.max_tokens or 4096
# One bound for the whole loop: rollouts are encoded against it and a trajectory
# longer than it is not trained on. The measured value for a 4B policy on 4 trainer
# GPUs is 16384 -- the limit is vocab x length x 2 bytes of logits against whatever
# the card has left after the weights, so pass --max-length to match the hardware.
MAX_LENGTH = args.template.max_length
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 200
# Trajectories a challenger batch has to reach before it is trained on, and the
# solving side's share of them. Rounded to whole groups: with 8 rollouts a side,
# --batch-size 128 at 0.5 is 8 proposing groups + 8 solving ones, the shape the
# earlier experiments ran.
BATCH_SIZE = args.training.batch_size or 8
SOLVER_RATIO = args.challenger.solver_ratio
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
SAVE_STEPS = args.training.save_steps or 50
OUTPUT_DIR = args.training.output_dir or './output'
EPSILON = args.loss.epsilon
# Off by default. Anything above zero needs --ref-model-gpus as well.
KL_BETA = args.rl.kl_coef

# ========== Challenger configuration ==========
# One workspace per concurrent job, for the challenger and the solver alike: a
# task built in one workspace and checked in another is a task nobody can pass.
NUM_ENVS = args.challenger.num_envs
WORKSPACE_ROOT = args.challenger.workspace_root
SAVE_DIR = args.challenger.save_dir
MAX_TURNS = args.challenger.max_turns

# Rollouts spent proposing, and rollouts spent measuring how hard the proposal is.
# The band is in attempt counts: a task no attempt solves is unverifiable, a task
# every attempt solves teaches nothing.
CHALLENGER_ROLLOUTS = args.challenger.num_challenger_rollouts
DIFFICULTY_ROLLOUTS = args.challenger.num_solver_rollouts
PASS_BAND = tuple(args.challenger.pass_band)

# The API backend, when there is one. Only the appended check-script and statement
# turns go through it; the acting turns stay on the policy being trained, since
# those are what the gradient comes from.
API_MODEL = args.challenger.api_model

# One tool call per reply. A second call in the same reply is made blind -- the
# observation the first one produced does not exist yet when it is written.
ONE_CALL_PER_REPLY = args.challenger.one_call_per_reply

# Optional. Empty means the topics the keyword seeder invents are the only variety.
SEED_DATASET = args.challenger.seed_dataset
SEED_SUBSET = args.challenger.seed_subset
SEED_SPLIT = args.challenger.seed_split
SEED_LIMIT = args.challenger.seed_limit

# Empty keeps every slot local, which has no isolation beyond a memory cap: the
# policy is being trained to run commands it wrote itself, in the trainer's own
# process tree. A template name switches every slot to its own microVM instead,
# and the workspace then lives inside the VM rather than under workspace_root.
SANDBOX_TEMPLATE = args.challenger.sandbox_template
SANDBOX_API_URL = args.challenger.sandbox_api_url
SANDBOX_TIMEOUT = args.challenger.sandbox_timeout

# An agent framework's config hands the solving half over to that framework's own
# program: it is started on the task, works until it decides it is done, and the
# policy is trained on the requests it made to the endpoint this process serves.
# That is the agent deployment runs, tools and context management included. Off,
# the solver runs on the loop in this repo against the env's built-in three tools
# -- serviceable, and not what anything deploys. The proposing half is unaffected
# either way: it needs to interrupt the conversation to ask for a check script,
# which is exactly what an agent that owns its loop will not allow.
# Sandboxed runs only: the agent needs a machine of its own to work in.
AGENT_CONFIG = args.challenger.agent_config
AGENT_ENDPOINT_HOST = args.challenger.agent_endpoint_host
AGENT_ENDPOINT_PORT = args.challenger.agent_endpoint_port
AGENT_TIMEOUT = args.challenger.agent_timeout
if AGENT_CONFIG and not SANDBOX_TEMPLATE:
    raise SystemExit('--agent-config needs --sandbox-template: the agent runs commands it '
                     'wrote itself, and a local workspace is the trainer\'s own process tree')
if AGENT_CONFIG and not AGENT_ENDPOINT_HOST:
    raise SystemExit('--agent-config needs --agent-endpoint-host: the agent calls the policy '
                     'from inside the sandbox, where loopback is the sandbox itself. Give the '
                     'address of this host that the sandbox can reach.')

KEYWORD_PATH = args.challenger.keyword_path
KEYWORD_QUERIES = [
    'data files: parsing, reshaping, and summarising CSV/JSON/YAML on disk',
    'text processing: extracting, rewriting, and validating structured text',
    'small algorithms with a verifiable numeric answer',
    'command-line utilities that leave their result in a file',
]

# One audit line per trained trajectory, next to the proposals it came from.
REWARD_DUMP = os.path.join(SAVE_DIR, 'rewards.jsonl') if SAVE_DIR else ''


def create_seed_trajectories() -> List[Trajectory]:
    """Read seed rows off a hub dataset. Built in memory; nothing is written out.

    Seeds are read as inspiration for the challenger's prompt, not as training
    data, so no template, no encode, and no preprocessing pass: whatever the rows
    look like, the challenger only ever sees a summary of one of them.
    """
    if not SEED_DATASET:
        return []
    meta_kwargs: Dict[str, Any] = {'dataset_id': SEED_DATASET, 'split': SEED_SPLIT}
    if SEED_SUBSET:
        meta_kwargs['subset_name'] = SEED_SUBSET
    dataset = Dataset(DatasetMeta(**meta_kwargs))
    columns = dataset.dataset.column_names
    rows = dataset.dataset.to_list()[:SEED_LIMIT]

    seeds: List[Trajectory] = []
    for row in rows:
        if 'messages' in columns and row.get('messages'):
            seeds.append({'messages': row['messages']})
            continue
        text = next((row[key] for key in ('query', 'problem', 'prompt', 'instruction', 'text')
                     if row.get(key)), '')
        if text:
            seeds.append({'messages': [{'role': 'user', 'content': str(text)}]})
    logger.info(f'[rsi] {len(seeds)} seeds from {SEED_DATASET}')
    return seeds


def create_seeder(sampler, rollout_template, sampling_params) -> Seeder:
    """Where a round's variety comes from: earlier tasks first, then fresh topics.

    Both are asked every round and their texts land in that order, so a round
    seeded from a dataset row is still pushed somewhere new by the topics. The
    keyword pool is generated by the policy itself against ``KEYWORD_QUERIES`` and
    cached on disk, so a resumed run does not spend rollouts inventing the topics
    it already has.
    """
    seeders: List[Seeder] = []
    trajectories = create_seed_trajectories()
    if trajectories:
        # No summarizer: the rows are prompts, not episodes, and short enough to
        # hand over whole. One that quotes whole transcripts wants one.
        seeders.append(TrajectorySeeder(trajectories))
    seeders.append(
        KeywordSeeder(
            query=KEYWORD_QUERIES,
            backend=sampler,
            path=KEYWORD_PATH,
            num_keywords=args.challenger.num_keywords,
            keywords_group_size=args.challenger.keywords_group_size,
            recycle=args.challenger.keyword_recycle,
            template=rollout_template,
            sampling_params=sampling_params,
        ))
    return seeders[0] if len(seeders) == 1 else ChainSeeder(seeders)


class TrainingBatch:
    """The trainable part of one challenger batch: scored, then filtered.

    The challenger hands over groups, each member carrying its own reward, so the
    advantage is taken inside a group whatever that group's size -- proposals are
    grouped by the round they were proposed in, attempts by the task they attempted.

    A trajectory is dropped only after its group has been scored. Filtering first
    would leave the survivors of a group compared against a baseline that included
    what was removed. Every drop is counted under its reason: a batch that lost
    half its trajectories to one wiring fault and a batch that lost none read the
    same from the loss alone.
    """

    def __init__(self, batch: ChallengeBatch, max_length: int = MAX_LENGTH):
        self.inputs: List[Trajectory] = []
        self.old_logps: List[List[float]] = []
        self.advantages: List[float] = []
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.keys: List[Tuple[str, str]] = []
        self.sides: collections.Counter = collections.Counter()
        self.skipped: collections.Counter = collections.Counter()
        self.rewards_by_side: Dict[str, List[float]] = collections.defaultdict(list)
        advantage_fn = GRPOAdvantage()
        for members in list(batch.challenger) + list(batch.solver):
            if not members:
                continue
            # Read off the group rather than off which list it came in: the names are
            # the challenger's to choose, and they are what the audit lines are keyed by.
            data = members[0].get('user_data')
            side = user_data_get(data, 'side', '')
            group_id = user_data_get(data, 'group_id', '')
            rewards = [float(trajectory.get('rewards') or 0.0) for trajectory in members]
            if len(members) < 2:
                # The advantage would be the reward minus itself.
                self.skipped[f'{side}: group of 1'] += 1
                continue
            advantages = advantage_fn(rewards, num_generations=len(members), scale='group').tolist()
            if all(abs(advantage) < 1e-9 for advantage in advantages):
                # Every member scored the same, so the group cancels out. Counted
                # because it is the one failure that looks like a successful step:
                # the update happens and moves nothing.
                self.skipped[f'{side}: group with no spread'] += 1
                continue
            for trajectory, reward, advantage in zip(members, rewards, advantages):
                self._add(trajectory, reward, advantage, side, group_id, max_length)

    def _add(self, trajectory: Trajectory, reward: float, advantage: float, side: str,
             group_id: str, max_length: int) -> None:
        labels = trajectory.get('labels') or []
        logprobs = trajectory.get('logprobs') or []
        trainable = sum(1 for label in labels if label != -100)
        if not logprobs:
            # Nothing for a new forward pass to be compared against, so GRPO has
            # no ratio. A sampler not returning logprobs is a wiring fault, not
            # attrition, which is why it is named rather than summed.
            self.skipped[f'{side}: no logprobs'] += 1
            return
        if not trainable:
            self.skipped[f'{side}: no trainable tokens'] += 1
            return
        if len(logprobs) != trainable:
            # Off by anything here pairs every logprob with the wrong token, and
            # the loss still comes out a plausible number.
            self.skipped[f'{side}: {len(logprobs)} logprobs != {trainable} trainable'] += 1
            return
        if len(trajectory.get('input_ids') or labels) > max_length:
            self.skipped[f'{side}: longer than max_length={max_length}'] += 1
            return
        self.inputs.append(trajectory)
        self.old_logps.append([logprob[0][1] for logprob in logprobs])
        self.advantages.append(advantage)
        self.rewards.append(reward)
        self.lengths.append(trainable)
        self.keys.append((side, group_id))
        self.sides[side] += 1
        self.rewards_by_side[side].append(reward)

    def __len__(self) -> int:
        return len(self.inputs)

    def mini_batches(self, size: int) -> Iterator[slice]:
        """Whole mini batches only, in order.

        ``forward_backward`` dispatches with 'slice_dp': it splits what it is given
        across every rank, and a batch that cannot hand each rank its own micro
        batch raises inside the dispatch, before the loss is ever reached. So a
        tail shorter than a mini batch is dropped here instead.
        """
        usable = len(self.inputs) - len(self.inputs) % size
        if usable < len(self.inputs):
            logger.warning(f'[rsi] dropping the last {len(self.inputs) - usable} trajectories, '
                           f'under the mini batch of {size}')
        for start in range(0, usable, size):
            yield slice(start, start + size)

    def head(self) -> str:
        """The first group, verbatim.

        A high reward carrying a negative advantage is a reordering bug between
        the rollout and the loss, and it is invisible in any average.
        """
        if not self.keys:
            return 'empty'
        first = self.keys[0]
        span = [i for i, key in enumerate(self.keys) if key == first]
        return (f'{first[0]} rewards={[round(self.rewards[i], 3) for i in span]} '
                f'advantages={[round(self.advantages[i], 3) for i in span]} '
                f'lens={[self.lengths[i] for i in span]}')

    def dump(self, path: str, step: int) -> None:
        """Append one audit line per trained trajectory. Reads, never changes."""
        if not path:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            for index, trajectory in enumerate(self.inputs):
                data = trajectory.get('user_data')
                handle.write(
                    json.dumps(
                        {
                            'step': step,
                            'side': self.keys[index][0],
                            'group_id': self.keys[index][1],
                            'reward': round(self.rewards[index], 4),
                            'advantage': round(self.advantages[index], 4),
                            'len': self.lengths[index],
                            'n_pass': user_data_get(data, 'n_pass', None),
                            'passed': user_data_get(data, 'passed', None),
                            'outcome': user_data_get(data, 'outcome', ''),
                        },
                        ensure_ascii=False) + '\n')


def reference_logps(ref_model, inputs: List[Trajectory]) -> Optional[List[Any]]:
    """One row of per-token logps per input, or None when there is no anchor.

    ``forward_only`` collects the ranks into a single [N, L], padded to the longest
    sequence across them; the loss wants one row per sample, in input order. Rows go
    over whole: the padding is masked out there, and trimming it here would need a
    length this side does not have. The shape is asserted rather than coerced -- a
    mismatch is a dispatch fault, and the loss would take a wrong pairing and still
    come out a plausible number.
    """
    if ref_model is None:
        return None
    import torch
    logps = ref_model.forward_only(inputs=inputs, micro_batch_size=MICRO_BATCH_SIZE)['logps']
    if not isinstance(logps, torch.Tensor) or logps.dim() != 2 or logps.shape[0] != len(inputs):
        shape = tuple(logps.shape) if isinstance(logps, torch.Tensor) else type(logps).__name__
        raise RuntimeError(f'reference returned {shape} for {len(inputs)} inputs, expected one row each')
    return list(logps)


def optimizer_step(model, optim_step: int) -> int:
    """Step on whatever gradient has accumulated, and checkpoint if one is due.

    ``forward_backward`` neither steps nor zeroes, so the mini batches before this
    have simply been adding their gradients together.
    """
    model.clip_grad_and_step()
    optim_step += 1
    if optim_step % SAVE_STEPS == 0:
        model.save(f'rsi-grpo-checkpoint-{optim_step}', output_dir=OUTPUT_DIR)
    return optim_step


def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, MODEL_GPUS + SAMPLER_GPUS)),
                    device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    ref_mesh = None
    if REF_GPUS:
        device_groups.append(
            DeviceGroup(name='ref', ranks=list(range(MODEL_GPUS + SAMPLER_GPUS, NUM_GPUS)),
                        device_type='GPU'))
        ref_mesh = DeviceMesh.from_sizes(world_size=REF_GPUS, dp_size=REF_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # The dashboard, if one was asked for. Init once: a second call raises.
    swan = None
    if args.report.swanlab_project and args.report.swanlab_mode != 'disabled':
        import swanlab
        swanlab.init(project=args.report.swanlab_project,
                     experiment_name=args.report.swanlab_experiment or None,
                     logdir=args.report.swanlab_log_dir,
                     mode=args.report.swanlab_mode)
        swan = swanlab

    # The actor. Full-parameter: no adapter is added, so every weight is trained and
    # the whole model is what gets pushed to the sampler. The trained weights stay
    # fp32 either way -- Megatron keeps fp32 master weights under mixed_precision,
    # the Transformers path is asked for fp32 directly. GRPO's ratio is the difference
    # of two log-probabilities of the same token, and at bf16 that difference is
    # mostly the rounding.
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
                              mixed_precision=args.model.mixed_precision,
                              variable_seq_lengths=args.model.variable_seq_lengths)
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
                                  torch_dtype='float32')
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    if KL_BETA > 0 and not REF_GPUS:
        raise RuntimeError(f'--kl-coef {KL_BETA} needs ref_logps to act on, and there is no '
                           f'reference model without --ref-model-gpus. Set one, or set '
                           f'--kl-coef 0.')
    model.set_loss('GRPOLoss', epsilon=EPSILON, beta=KL_BETA)
    model.set_processor(InputProcessor, padding_free=args.training.padding_free)
    # 'raise', not the rollout template's 'delete': inputs / old_logps / advantages
    # are handed to forward_backward as parallel lists, so a row dropped during
    # encoding would pair every later row with someone else's advantage. Nothing
    # oversized reaches here anyway -- TrainingBatch drops it against the same
    # max_length -- so this firing means that filter was bypassed.
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=MAX_LENGTH,
                       enable_thinking=True, truncation_strategy='raise')
    # Observability: approx_kl / clip_ratio / entropy per step. approx_kl at the
    # first inner step also reconciles the sampler's logps against the trainer's,
    # which is the check for whether the weight sync actually landed.
    model.add_metric('GRPOMetric', is_training=True, epsilon=EPSILON)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': args.sampler.gpu_memory_utilization,
            'max_model_len': args.sampler.max_model_len or MAX_LENGTH,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    # 'delete' here, as on the rollout template: an over-long prompt costs one
    # rollout, and taking down the run for it is the more expensive answer.
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=MAX_LENGTH,
                         enable_thinking=True, truncation_strategy='delete')
    # The KL anchor: frozen weights, forward only, no optimizer. Same template and
    # processor as the actor, so the logps it returns line up token for token with
    # the actor's own forward. bf16 is enough for a term only compared against itself.
    ref_model = None
    if REF_GPUS:
        if USE_MEGATRON:
            from twinkle.model.megatron import MegatronModel
            ref_model = MegatronModel(model_id=MODEL_ID, device_mesh=ref_mesh, remote_group='ref',
                                      mixed_precision=args.model.mixed_precision,
                                      variable_seq_lengths=args.model.variable_seq_lengths)
        else:
            ref_model = TransformersModel(model_id=MODEL_ID, device_mesh=ref_mesh, remote_group='ref')
        # advantages=None on this path, so GRPOLoss short-circuits to a zero loss and
        # only the logps are harvested.
        ref_model.set_loss('GRPOLoss', epsilon=EPSILON)
        ref_model.set_processor(InputProcessor, padding_free=args.training.padding_free)
        # 'raise' for the actor's reason, and one more: the rows this returns are
        # zipped with the actor's own forward, so a row missing on one side only
        # would anchor the KL of every token after it to the wrong sequence.
        ref_model.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=MAX_LENGTH,
                               enable_thinking=True, truncation_strategy='raise')

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    optim_step = 0
    if args.training.resume_from_checkpoint:
        state = model.resume_from_checkpoint(args.training.resume_from_checkpoint,
                                             resume_only_model=args.training.resume_only_model)
        # There is no dataloader to skip forward: the tasks this loop trains on do
        # not exist yet. Only the step counter carries over, so the schedule and
        # MAX_STEPS mean the same thing across a restart.
        optim_step = int(state.get('cur_step') or 0)
        logger.info(f'[rsi] resumed {args.training.resume_from_checkpoint} at step {optim_step}')

    # The local template every rollout encodes with. 'delete' rather than a
    # truncation: half an episode teaches the wrong lesson about its own reward.
    rollout_template = Qwen3_5Template(MODEL_ID, max_length=MAX_LENGTH, enable_thinking=True,
                                       truncation_strategy='delete')
    # Stopping at the end of a tool call is what keeps a reply to one call: the
    # second one would be answered with an observation the model never saw. The
    # marker comes from the template, since the format belongs to the model and
    # not to this script -- a format that has none gets no stop. The stop string
    # stays in the output, or every turn trains on an unclosed block.
    tool_call_stop = rollout_template.tool_call_stop if ONE_CALL_PER_REPLY else None
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
                                     temperature=1.0, top_p=0.95,
                                     stop=[tool_call_stop] if tool_call_stop else None,
                                     include_stop_str_in_output=bool(tool_call_stop))

    # No model configured keeps the whole loop local. The thinking budget rides in
    # extra_body because it is a provider extension rather than part of the
    # chat-completions body, and it is worth setting: a reasoning model left
    # uncapped spends thousands of tokens on a reply of a few lines, and the calls
    # then time out under concurrency.
    api = None
    api_kwargs = None
    if API_MODEL:
        api = OpenAI(API_MODEL, api_key=args.challenger.api_key or None,
                     base_url=args.challenger.api_base or None,
                     concurrency=args.challenger.api_concurrency)
        if args.challenger.api_thinking_budget > 0:
            api_kwargs = {'extra_body': {'thinking_budget': args.challenger.api_thinking_budget}}

    # The task factory: act, verify, describe, then keep only what lands in band.
    # One workspace per concurrent job either way, and how many there are is how
    # many jobs run at once; the sandboxed ones boot on first use, from the clear()
    # the pool does before handing one over.
    if SANDBOX_TEMPLATE:
        envs = [
            AgentEnv(template=SANDBOX_TEMPLATE, api_url=SANDBOX_API_URL or None,
                     sandbox_timeout=SANDBOX_TIMEOUT, command_timeout=120,
                     metadata={'run': 'rsi_grpo', 'slot': str(i)})
            for i in range(NUM_ENVS)
        ]
    else:
        envs = [
            LocalEnv(workspace=os.path.join(WORKSPACE_ROOT, f'slot_{i}'), command_timeout=120)
            for i in range(NUM_ENVS)
        ]
    # The solving half, when an agent program owns it. The endpoint it serves lives
    # in this process on purpose: it answers out of the sampler the trainer syncs,
    # so what the agent talked to is what the gradient updates. Same decoding
    # settings as the proposing half, logprobs included -- GRPO needs the rollout
    # logprobs, and an agent has no way to ask for them.
    solver_rollout = None
    if AGENT_CONFIG:
        solver_rollout = ExternalRollout(
            sampler,
            MsAgent(config=AGENT_CONFIG),
            template=rollout_template,
            sampling_params=sampling_params,
            timeout=AGENT_TIMEOUT,
            endpoint_host=AGENT_ENDPOINT_HOST,
            endpoint_port=AGENT_ENDPOINT_PORT,
        )
    challenger = AgenticChallenger(
        sampler,
        envs=envs,
        seed_fn=create_seeder(sampler, rollout_template, sampling_params),
        solver_rollout=solver_rollout,
        num_challenger_rollouts=CHALLENGER_ROLLOUTS,
        num_solver_rollouts=DIFFICULTY_ROLLOUTS,
        pass_band=PASS_BAND,
        pass_rate_target=args.challenger.pass_rate_target,
        pass_rate_width=args.challenger.pass_rate_width,
        max_empty_rounds=args.challenger.max_empty_rounds,
        check_language=args.challenger.check_language,
        check_retries=args.challenger.check_retries,
        problem_max_chars=args.challenger.problem_max_chars,
        brittle_check_fn=brittle_check_reason,
        api=api,
        use_api=args.challenger.use_api,
        save_dir=SAVE_DIR,
        save_failed_rollouts=args.challenger.save_failed_rollouts,
        # Passed through to the challenger's own MultiTurnRollout.
        template=rollout_template,
        sampling_params=sampling_params,
        max_turns=MAX_TURNS,
        api_kwargs=api_kwargs,
    )
    metrics = CompletionRewardMetric()
    logger.info(get_device_placement())

    # The sync straddles the loop: the challenger draws the next batch as soon as
    # the loop asks for it, so the weights it proposes with are the ones this line
    # pushed, not the ones from the step before. Full weights, no adapter.
    ckpt_manager.sync_weights(merge_and_sync=True)
    sampler.reset_prefix_cache()

    for batch in challenger(BATCH_SIZE, solver_ratio=SOLVER_RATIO):
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        data = TrainingBatch(batch)
        logger.info(f'[Step {optim_step}] {len(batch.challenger)} proposal groups + '
                    f'{len(batch.solver)} attempt groups, {len(batch)} trajectories, '
                    f'{len(data)} trainable {dict(data.sides)}')
        for note, count in sorted(data.skipped.items()):
            logger.warning(f'[rsi] skipped: {note} x{count}')
        if len(data) < MINI_BATCH_SIZE:
            logger.warning(f'[Step {optim_step}] {len(data)} trainable trajectories is under one '
                           f'mini batch ({MINI_BATCH_SIZE}); skipping this batch')
            ckpt_manager.sync_weights(merge_and_sync=True)
            sampler.reset_prefix_cache()
            continue
        logger.info(f'[group0] {data.head()}')
        data.dump(REWARD_DUMP, optim_step + 1)
        metrics.accumulate(completion_lengths=data.lengths, rewards=dict(data.rewards_by_side))

        pending = 0
        for window in data.mini_batches(MINI_BATCH_SIZE):
            model.forward_backward(
                inputs=data.inputs[window],
                old_logps=data.old_logps[window],
                advantages=data.advantages[window],
                ref_logps=reference_logps(ref_model, data.inputs[window]),
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            pending += 1
            if pending < GRADIENT_ACCUMULATION_STEPS:
                continue
            optim_step, pending = optimizer_step(model, optim_step), 0
            if optim_step >= MAX_STEPS:
                break
        if pending:
            # The weights go to the sampler below, so a gradient held back here
            # would be applied to data drawn from weights that no longer exist.
            optim_step = optimizer_step(model, optim_step)

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        high_kl = log_dict.pop('_high_kl_records', None)
        log_dict['kept_tasks'] = f'{challenger.n_kept}/{challenger.n_proposed}'
        log_dict['trained'] = dict(data.sides)
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')
        if high_kl:
            logger.warning(f'[rsi] {len(high_kl)} sequences disagree with the sampler logps; '
                           f'this batch may not be from these weights')
        if swan is not None:
            swan.log({key: value for key, value in log_dict.items() if isinstance(value, (int, float))},
                     step=optim_step)

        ckpt_manager.sync_weights(merge_and_sync=True)
        sampler.reset_prefix_cache()

    challenger.close()
    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('rsi-grpo-checkpoint', output_dir=OUTPUT_DIR)


if __name__ == '__main__':
    main()
