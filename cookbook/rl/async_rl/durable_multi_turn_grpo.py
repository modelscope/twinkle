# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end GSM8K multi-turn GRPO with durable group checkpoints.

The example trains a LoRA policy on the official GSM8K train split.  Each
prompt is sampled as one complete GRPO group, may receive one format-repair
follow-up, and is trained with a batch lease owned by ``StreamingDriver``.

The durable transaction is:

1. finish every mini-batch and optimizer step for the leased rollout batch;
2. save adapter, optimizer, scheduler and RNG state;
3. call ``driver.commit_batch(checkpoint, manifest)``;
4. publish the new policy and admit the next version.

A crash before step 3 reloads the previous committed checkpoint and re-delivers
the same batch lease.  A crash after step 3 reloads this checkpoint and does not
train the batch twice.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.async_rl.rollout_resume import JournalEvent, ResumeStore, atomic_write_bytes
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout

logger = get_logger()
args = CLI.from_args()

MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 4
BATCH_SIZE = args.training.batch_size or 2
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_GENERATIONS
MINI_BATCH_SIZE = args.training.mini_batch_size or GLOBAL_BATCH_SIZE
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
MAX_STEPS = args.training.max_steps or 100
MAX_NEW_TOKENS = args.sampling.max_tokens or 1024
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
LORA_RANK = args.lora.lora_r or 8
ADAPTER_NAME = args.lora.adapter_name or 'durable-grpo'

RUN_ID = os.environ.get('TWINKLE_DURABLE_RUN_ID', 'gsm8k-durable-grpo')
RESUME_ROOT = Path(os.environ.get('TWINKLE_RESUME_ROOT', 'output/durable-gsm8k'))
CHECKPOINT_ROOT = RESUME_ROOT / 'model-checkpoints'

if MINI_BATCH_SIZE != GLOBAL_BATCH_SIZE:
    raise ValueError(
        'mini_batch_size must equal batch_size * num_generations so one driver lease maps to '
        f'exactly one optimizer step, got {MINI_BATCH_SIZE} and {GLOBAL_BATCH_SIZE}'
    )

SYSTEM_PROMPT = (
    'You are a careful math assistant. Solve the problem and put only the final '
    'answer in \\boxed{}.'
)


class _ImmediateFuture:
    """Adapt a synchronous reward function to the driver's future contract."""

    def __init__(self, value: Any):
        self._value = value

    def exception(self) -> None:
        return None

    def result(self) -> Any:
        return self._value

    def add_done_callback(self, callback) -> None:
        callback(self)


class _BrevityReward(Reward):
    """Small deterministic format reward, aligned with the short-math example."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            reply = ''
            for message in reversed(trajectory.get('messages') or ()):
                if message.get('role') == 'assistant':
                    reply = str(message.get('content') or '')
                    break
            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', reply)
                or re.search(r'####\s*[\-\d,\.]+', reply)
            )
            if not has_answer:
                rewards.append(0.0)
            elif len(reply) <= 300:
                rewards.append(1.0)
            else:
                rewards.append(max(0.0, 1.0 - (len(reply) - 300) / 3000))
        return rewards


def _reward_future(reward: Reward):
    def score(trajectories):
        return _ImmediateFuture(reward(trajectories))

    return score


class _PolicyIndex:
    """Translate monotonic driver versions to retained adapter directories."""

    def __init__(self, path: Path):
        self.path = path
        self._paths: dict[int, str] = {}
        self._current_version = 0
        if path.is_file():
            raw = json.loads(path.read_text(encoding='utf-8'))
            self._paths = {int(version): str(value) for version, value in raw.items()}
            self._current_version = max(self._paths, default=0)

    @property
    def latest_version(self) -> int:
        return max(self._paths, default=0)

    def version(self) -> int:
        return self._current_version

    def adapter_path(self, version: int) -> str | None:
        return self._paths.get(int(version))

    def register(self, version: int, checkpoint: str) -> None:
        version = int(version)
        checkpoint = str(checkpoint)
        if version < 0 or not checkpoint:
            raise ValueError('policy registration needs a non-negative version and checkpoint')
        self._paths[version] = checkpoint
        self._current_version = max(self._current_version, version)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            self.path,
            (json.dumps(self._paths, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8'),
        )

    def recover_from_journal(self, records: Iterable[Any]) -> None:
        """Recover a commit that became durable before the JSON index write."""
        for record in records:
            if record.event != JournalEvent.MODEL_CHECKPOINT_COMMITTED:
                continue
            manifest = record.payload.get('manifest') or {}
            version = manifest.get('policy_version')
            checkpoint = record.payload.get('checkpoint')
            if version is not None and checkpoint:
                self.register(int(version), str(checkpoint))


def create_gsm8k_dataset():
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template(
        'Qwen3_5Template',
        model_id=MODEL_ID,
        max_length=4096,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def _followup(trajectory: Dict[str, Any], followups: int):
    if followups >= 1:
        return None
    replies = [
        str(message.get('content') or '')
        for message in trajectory.get('messages') or ()
        if message.get('role') == 'assistant'
    ]
    if replies and re.search(r'\\boxed\{[^}]+\}', replies[-1]):
        return None
    return 'Please restate the result with the final answer inside \\boxed{}.'


def _score_vector(rows: List[Any], facts: Dict[str, Any]) -> List[float]:
    scores = facts.get('scores') or ()
    if len(scores) != len(rows):
        raise RuntimeError(f'driver returned {len(rows)} rows but {len(scores)} score records')
    rewards = []
    for item in scores:
        values = (item.get('rewards') or {}).get('orm') or {}
        accuracy = values.get('accuracy')
        brevity = values.get('brevity')
        rewards.append(float(accuracy or 0.0) + float(brevity or 0.0))
    return rewards


def _old_logps(rows: List[Any]) -> List[List[float]]:
    values = []
    for row in rows:
        logps = row.get('rollout_logps')
        if not logps:
            raise RuntimeError(f'row {row.get("episode_id")!r} has no rollout log probabilities')
        values.append(list(logps))
    return values


def main() -> None:
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
        mixed_precision='bf16',
    )
    model.add_adapter_to_model(
        ADAPTER_NAME,
        LoraConfig(target_modules='all-linear', r=LORA_RANK, lora_alpha=LORA_RANK * 2),
        gradient_accumulation_steps=1,
    )
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': max(32, LORA_RANK),
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)
    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    store = ResumeStore(RESUME_ROOT, RUN_ID)
    policy_index = _PolicyIndex(RESUME_ROOT / 'policy-index.json')
    policy_index.recover_from_journal(store.journal.records())

    committed_records = [
        record for record in store.journal.records()
        if record.event == JournalEvent.MODEL_CHECKPOINT_COMMITTED
    ]
    latest_checkpoint = policy_index.adapter_path(policy_index.latest_version)
    if latest_checkpoint is None:
        latest_checkpoint = model.save(
            'policy-0',
            output_dir=str(CHECKPOINT_ROOT),
            adapter_name=ADAPTER_NAME,
            save_optimizer=False,
        )
        policy_index.register(0, latest_checkpoint)
        optim_step = 0
    elif not committed_records:
        # The initial adapter index survives a crash before the first commit,
        # but it intentionally has no optimizer/trainer_state files.
        model.load(latest_checkpoint, adapter_name=ADAPTER_NAME)
        optim_step = 0
    else:
        progress = model.resume_from_checkpoint(latest_checkpoint, adapter_name=ADAPTER_NAME)
        optim_step = int(progress.get('cur_step') or 0)
        logger.info(f'resumed model checkpoint {latest_checkpoint} at step {optim_step}')

    template = Qwen3_5Template(MODEL_ID, max_length=4096, enable_thinking=False)
    template.truncation_strategy = 'delete'
    rollout = MultiTurnRollout(
        sampler=sampler,
        template=template,
        sampling_params=SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            num_samples=1,
            logprobs=1,
            temperature=1.0,
            top_p=0.95,
        ),
        max_turns=2,
        max_trajectory_tokens=4096,
        followup_fn=_followup,
        resume_store=store,
    )

    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )
    driver = StreamingDriver(
        dataloader,
        rollout,
        batch_size=GLOBAL_BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        max_steps=MAX_STEPS,
        max_staleness=1,
        checkpoint_manager=policy_index,
        orms=[_reward_future(GSM8KAccuracyReward()), _reward_future(_BrevityReward())],
        orm_names=['accuracy', 'brevity'],
        resume_store=store,
        run_id=RUN_ID,
    )
    if store.has_object('driver', RUN_ID):
        driver.load_state_dict()
        logger.info('resumed StreamingDriver from the latest stable boundary')

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    logger.info(get_device_placement())

    try:
        for rows, facts in driver:
            if optim_step >= MAX_STEPS:
                break
            rewards = _score_vector(rows, facts)
            advantages = advantage_fn(
                rewards,
                num_generations=NUM_GENERATIONS,
                scale='group',
            ).tolist()
            old_logps = _old_logps(rows)
            completion_lengths = [
                sum(1 for label in (row.get('labels') or ()) if label != -100)
                for row in rows
            ]
            metrics.accumulate(
                completion_lengths=completion_lengths,
                rewards={'total': rewards},
            )

            for start in range(0, len(rows), MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                model.forward_backward(
                    inputs=rows[start:end],
                    old_logps=old_logps[start:end],
                    advantages=advantages[start:end],
                    micro_batch_size=MICRO_BATCH_SIZE,
                )
                model.clip_grad_and_step()
                optim_step += 1

            checkpoint = model.save(
                f'policy-step-{optim_step}',
                output_dir=str(CHECKPOINT_ROOT),
                adapter_name=ADAPTER_NAME,
                save_optimizer=True,
                async_save=False,
            )
            next_version = policy_index.version() + 1
            driver.commit_batch(
                checkpoint,
                manifest={
                    'optimizer_step': optim_step,
                    'policy_version': next_version,
                },
            )
            policy_index.register(next_version, checkpoint)

            ckpt_manager.sync_weights(merge_and_sync=False)
            sampler.reset_prefix_cache()
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            logger.info(f'[durable step {optim_step}/{MAX_STEPS}] {log_dict}')
            metrics.reset()
    finally:
        driver.close()

    logger.info(f'durable GRPO completed: optimizer_step={optim_step}')


if __name__ == '__main__':
    main()
