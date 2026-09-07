"""Client-orchestrated async GRPO built from the low-level component APIs."""
from __future__ import annotations

import asyncio
import inspect
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from peft import LoraConfig

from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle_client import DataPlaneClient, init_twinkle_client
from twinkle_client.async_rl import Worker, WorkerPipeline
from twinkle_client.common.json_utils import json_safe
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

BASE_MODEL = 'Qwen/Qwen3.5-4B'
MODEL_ID = f'ms://{BASE_MODEL}'
TEMPLATE_MODEL_ID = os.environ.get('TWINKLE_TEMPLATE_MODEL_ID', MODEL_ID)
TEMPLATE_CLS = 'Qwen3_5Template'
DATASET_ID = os.environ.get('TWINKLE_DATASET_ID', 'ms://modelscope/gsm8k')
ADAPTER_NAME = os.environ.get('TWINKLE_ADAPTER_NAME', 'client-grpo')
MAX_PARTITIONS = int(os.environ.get('TWINKLE_MAX_PARTITIONS', '100'))
MAX_STALENESS = int(os.environ.get('TWINKLE_MAX_STALENESS', '2'))
ROLLOUT_CONCURRENCY = int(os.environ.get('TWINKLE_ROLLOUT_CONCURRENCY', '8'))
NUM_GENERATIONS = int(os.environ.get('TWINKLE_NUM_GENERATIONS', '4'))
BATCH_SIZE = int(os.environ.get('TWINKLE_BATCH_SIZE', '8'))
TRAIN_MINI_BATCH_SIZE = int(os.environ.get('TWINKLE_TRAIN_MINI_BATCH_SIZE', '8'))
MICRO_BATCH_SIZE = int(os.environ.get('TWINKLE_MICRO_BATCH_SIZE', '4'))
MAX_TOKENS_PER_MICRO_BATCH = int(os.environ.get('TWINKLE_MAX_TOKENS_PER_MICRO_BATCH', '4096'))


@dataclass(frozen=True)
class _Policy:
    version: int
    adapter_uri: str


@dataclass
class _RolloutPartition:
    """One DataLoader batch bound to one immutable policy snapshot."""

    partition_id: int
    policy: _Policy
    prompts: list[dict[str, Any]]
    rollouts: list[asyncio.Task[Any]]
    ready: asyncio.Queue['_ReadyGroup'] = field(default_factory=asyncio.Queue)


@dataclass
class _ReadyGroup:
    group_index: int
    ref: Any


@dataclass
class _RolloutResult:
    partition: _RolloutPartition
    group_index: int
    prompt: dict[str, Any]
    ref: Any


class _GRPOState:

    def __init__(self, policy: _Policy):
        self.policy = policy
        self.live: deque[_RolloutPartition] = deque()
        self.input_done = False
        self.failure: BaseException | None = None
        self.condition = asyncio.Condition()

    async def wait_for_admission(self) -> _Policy:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.failure is not None or len(self.live) < MAX_STALENESS + 1)
            if self.failure is not None:
                raise self.failure
            return self.policy

    async def add_partition(self, partition: _RolloutPartition) -> None:
        async with self.condition:
            self.live.append(partition)
            self.condition.notify_all()

    async def finish_input(self) -> None:
        async with self.condition:
            self.input_done = True
            self.condition.notify_all()

    async def fail(self, error: BaseException) -> None:
        async with self.condition:
            if self.failure is None:
                self.failure = error
            self.condition.notify_all()

    async def oldest_partition(self) -> _RolloutPartition | None:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.failure is not None or bool(self.live) or self.input_done)
            if self.failure is not None:
                raise self.failure
            return self.live[0] if self.live else None

    async def publish(self, partition: _RolloutPartition, policy: _Policy) -> None:
        async with self.condition:
            if not self.live or self.live[0] is not partition:
                raise RuntimeError(f'partition {partition.partition_id} attempted out-of-order publication')
            self.policy = policy
            self.live.popleft()
            self.condition.notify_all()


def create_dataset() -> Dataset:
    dataset = Dataset(DatasetMeta(DATASET_ID, subset_name='main', split='train'))
    dataset.set_template(TEMPLATE_CLS, model_id=TEMPLATE_MODEL_ID, max_length=2048, enable_thinking=False)
    dataset.map(GSM8KProcessor(system='Put the final answer within \\boxed{}.'))
    dataset.encode(add_generation_prompt=True)
    return dataset


async def rollout_group(
    sampler: vLLMSampler,
    prompt: dict[str, Any],
    policy: _Policy,
    semaphore: asyncio.Semaphore,
    group_id: str,
) -> Any:
    """Submit one GRPO group and keep its sample-level TQ DataRef alive."""
    async with semaphore:
        return await _submit(
            sampler.asample_to_data_plane,
            [prompt],
            adapter_name=ADAPTER_NAME,
            adapter_uri=policy.adapter_uri,
            policy_version=policy.version,
            group_ids=[group_id],
            sampling_params={
                'max_tokens': 1024,
                'temperature': 1.0,
                'top_p': 0.95,
                'logprobs': 1,
            },
            num_samples=NUM_GENERATIONS,
        )


def start_partition(
    partition_id: int,
    batch: list[dict[str, Any]],
    policy: _Policy,
    sampler: vLLMSampler,
    semaphore: asyncio.Semaphore,
) -> _RolloutPartition:
    """Capture the snapshot before submitting any rollout in this partition."""
    prompts = json_safe(batch)
    return _RolloutPartition(
        partition_id=partition_id,
        policy=policy,
        prompts=prompts,
        rollouts=[
            asyncio.create_task(
                rollout_group(
                    sampler,
                    prompt,
                    policy,
                    semaphore,
                    f'partition-{partition_id}/group-{group_index}',
                ))
            for group_index, prompt in enumerate(prompts)
        ],
    )


async def _submit(method, *args, **kwargs):
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    task = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(task):
        return await task
    return task


def _checkpoint_path(saved: Any) -> str:
    if isinstance(saved, dict):
        return str(saved['twinkle_path'])
    return str(saved.twinkle_path)


class _RolloutWorker(Worker):

    def __init__(self, dataloader, sampler, state: _GRPOState, output: asyncio.Queue, semaphore):
        super().__init__('rollout')
        self.dataloader = dataloader
        self.sampler = sampler
        self.state = state
        self.output = output
        self.semaphore = semaphore

    async def _collect(self, partition, group_index, task):
        try:
            ref = await task
            await self.output.put(
                _RolloutResult(partition, group_index, partition.prompts[group_index], ref))
        except BaseException as error:
            await self.state.fail(error)
            raise

    async def run(self) -> None:
        batches = iter(self.dataloader)
        collectors: list[asyncio.Task] = []
        try:
            for partition_id in range(MAX_PARTITIONS):
                policy = await self.state.wait_for_admission()
                batch = next(batches, None)
                if batch is None:
                    break
                prompts = batch if isinstance(batch, list) else [batch]
                if len(prompts) != BATCH_SIZE:
                    print(f'dropping incomplete final batch with {len(prompts)} prompts')
                    break
                partition = start_partition(
                    partition_id, prompts, policy, self.sampler, self.semaphore)
                await self.state.add_partition(partition)
                collectors.extend(
                    asyncio.create_task(self._collect(partition, index, task))
                    for index, task in enumerate(partition.rollouts)
                )
            await self.state.finish_input()
            await asyncio.gather(*collectors)
            await self.output.put(None)
        except BaseException as error:
            await self.state.fail(error)
            for task in collectors:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*collectors, return_exceptions=True)
            raise


class _AdvantageWorker(Worker):

    def __init__(self, data_plane, state: _GRPOState, source: asyncio.Queue):
        super().__init__('advantage')
        self.data_plane = data_plane
        self.state = state
        self.source = source

    async def run(self) -> None:
        try:
            while True:
                result = await self.source.get()
                if result is None:
                    return
                group_id = f'partition-{result.partition.partition_id}/group-{result.group_index}'
                try:
                    rows = await self.data_plane.aget(result.ref, fields=['decoded'])
                    if len(rows) != NUM_GENERATIONS:
                        raise RuntimeError(
                            f'group {group_id} expected {NUM_GENERATIONS} generations, '
                            f'got {len(rows)}')
                    trajectories = []
                    for row in rows:
                        messages = list(result.prompt.get('messages') or [])
                        messages.append({'role': 'assistant', 'content': row.get('decoded') or ''})
                        trajectories.append({**result.prompt, 'messages': messages})
                    rewards = await asyncio.to_thread(GSM8KAccuracyReward(), trajectories)
                    advantages = await asyncio.to_thread(
                        GRPOAdvantage(), rewards, num_generations=NUM_GENERATIONS)
                    ref = await self.data_plane.aappend(
                        result.ref,
                        [{
                            'reward': float(reward),
                            'advantage': float(advantage),
                        } for reward, advantage in zip(rewards, advantages)],
                    )
                    await result.partition.ready.put(
                        _ReadyGroup(result.group_index, ref))
                except BaseException:
                    await self.data_plane.arelease(result.ref)
                    raise
        except BaseException as error:
            await self.state.fail(error)
            raise


class _TrainerWorker(Worker):

    def __init__(self, model, data_plane, state: _GRPOState):
        super().__init__('trainer')
        self.model = model
        self.data_plane = data_plane
        self.state = state
        self.optimizer_step = 0

    async def _train(self, groups: list[_ReadyGroup]) -> None:
        refs = [group.ref for group in groups]
        try:
            await _submit(
                self.model.forward_backward_from_data_plane,
                refs,
                input_field='train_input',
                kwarg_fields={
                    'old_logps': 'sampled_logprobs',
                    'advantages': 'advantage',
                },
                dynamic_batching=True,
                micro_batch_size=MICRO_BATCH_SIZE,
                max_tokens_per_micro_batch=MAX_TOKENS_PER_MICRO_BATCH,
            )
            await _submit(self.model.clip_grad_and_step, max_grad_norm=1.0)
            self.optimizer_step += 1
            metric_response = await _submit(self.model.calculate_metric, is_training=True)
            metrics = dict(
                metric_response['result']
                if isinstance(metric_response, dict)
                else metric_response.result
            )
            values = ' '.join(f'{name}={value}' for name, value in sorted(metrics.items()))
            print(f'optimizer_step={self.optimizer_step} {values}'.rstrip())
        finally:
            await asyncio.gather(*(self.data_plane.arelease(ref) for ref in refs))

    async def run(self) -> None:
        try:
            while True:
                partition = await self.state.oldest_partition()
                if partition is None:
                    return
                staleness = self.state.policy.version - partition.policy.version
                if staleness > MAX_STALENESS:
                    raise RuntimeError(
                        f'partition {partition.partition_id} staleness {staleness} exceeds {MAX_STALENESS}')
                groups_per_step = TRAIN_MINI_BATCH_SIZE // NUM_GENERATIONS
                ready = []
                for _ in range(len(partition.rollouts)):
                    ready.append(await partition.ready.get())
                    if len(ready) == groups_per_step:
                        await self._train(ready)
                        ready.clear()
                if ready:
                    raise RuntimeError('partition ended with an incomplete train mini-batch')
                publish_version = self.state.policy.version + 1
                saved = await _submit(self.model.save, f'policy-{publish_version}')
                policy = _Policy(publish_version, _checkpoint_path(saved))
                await self.state.publish(partition, policy)
                print(
                    f'partition={partition.partition_id} policy={policy.version} '
                    f'optimizer_step={self.optimizer_step} staleness={staleness}')
        except BaseException as error:
            await self.state.fail(error)
            raise


async def run_grpo(
    dataloader: DataLoader,
    model: MultiLoraTransformersModel,
    sampler: vLLMSampler,
    data_plane: DataPlaneClient,
) -> None:
    """Overlap rollout partitions while training and publishing them in FIFO order."""
    if MAX_STALENESS < 0:
        raise ValueError('MAX_STALENESS must be non-negative')
    if min(ROLLOUT_CONCURRENCY, NUM_GENERATIONS, BATCH_SIZE, TRAIN_MINI_BATCH_SIZE) <= 0:
        raise ValueError('rollout concurrency and all batch sizes must be positive')
    if TRAIN_MINI_BATCH_SIZE % NUM_GENERATIONS:
        raise ValueError('TRAIN_MINI_BATCH_SIZE must be divisible by NUM_GENERATIONS')
    groups_per_step = TRAIN_MINI_BATCH_SIZE // NUM_GENERATIONS
    if BATCH_SIZE % groups_per_step:
        raise ValueError('BATCH_SIZE * NUM_GENERATIONS must be divisible by TRAIN_MINI_BATCH_SIZE')

    initial = await _submit(model.save, 'policy-0')
    state = _GRPOState(_Policy(version=0, adapter_uri=_checkpoint_path(initial)))
    semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)
    rollout_results: asyncio.Queue = asyncio.Queue()
    await WorkerPipeline((
        _RolloutWorker(dataloader, sampler, state, rollout_results, semaphore),
        _AdvantageWorker(data_plane, state, rollout_results),
        _TrainerWorker(model, data_plane, state),
    )).run()


async def train() -> None:
    client = init_twinkle_client(
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )
    try:
        model = MultiLoraTransformersModel(MODEL_ID)
        sampler = vLLMSampler(MODEL_ID)
        data_plane = DataPlaneClient()

        model.add_adapter_to_model(
            ADAPTER_NAME,
            LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
        )
        model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
        model.set_optimizer('AdamW', lr=2e-5)
        model.set_processor('InputProcessor', padding_free=False)
        model.set_template(TEMPLATE_CLS, model_id=TEMPLATE_MODEL_ID)
        sampler.set_template(TEMPLATE_CLS, model_id=TEMPLATE_MODEL_ID)

        dataloader = DataLoader(dataset=create_dataset(), batch_size=BATCH_SIZE, num_workers=0)
        await run_grpo(dataloader, model, sampler, data_plane)
    finally:
        client.close()


if __name__ == '__main__':
    asyncio.run(train())
