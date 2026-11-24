# Copyright (c) ModelScope Contributors. All rights reserved.
"""Single-turn GSM8K GRPO with sampling and training overlapped by StreamingDriver."""

import os
import uuid

import dotenv
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import FunctionWorker, GSM8KAccuracyReward
from twinkle.template import Qwen3_5Template
from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.rollout import MultiTurnRollout
from twinkle_client.resources import RolloutResources
from twinkle_client.rollout import ClientCheckpointManager, HTTPSampler

dotenv.load_dotenv('.env')
logger = get_logger()
MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
if '://' not in MODEL_ID:
    MODEL_ID = f'ms://{MODEL_ID}'
TEMPLATE_MODEL_ID = os.environ.get('TWINKLE_TEMPLATE_MODEL_ID', MODEL_ID)
DATASET_ID = os.environ.get('TWINKLE_DATASET_ID', 'ms://modelscope/gsm8k')
NUM_GENERATIONS = 4
MAX_STEPS = int(os.environ.get('TWINKLE_MAX_STEPS', '6'))


def create_dataloader():
    dataset = Dataset(DatasetMeta(DATASET_ID, subset_name='main', split='train'))
    dataset.map(GSM8KProcessor(), remove_columns=['question', 'answer'])
    return DataLoader(dataset=dataset, batch_size=1, num_workers=0)


def train():
    with RolloutResources(init_twinkle_client) as resources:
        client = resources.client
        adapter_name = 'gsm8k'
        model = client.model(MODEL_ID)
        model.add_adapter_to_model(adapter_name, LoraConfig(target_modules='all-linear', r=8, lora_alpha=32))
        model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
        model.set_optimizer('Adam', lr=2e-5)
        model.set_processor('InputProcessor', padding_free=False)
        model.set_template('Qwen3_5Template', enable_thinking=False)

        sampler = client.sampler(MODEL_ID)
        sampler.set_template('Qwen3_5Template', enable_thinking=False)
        template = Qwen3_5Template(model_id=TEMPLATE_MODEL_ID, max_length=2048, enable_thinking=False)
        template.truncation_strategy = 'delete'
        sampler = HTTPSampler(sampler, template, adapter_name=adapter_name)
        # With no tools, each trajectory is a single prompt and a single reply.
        rollout = resources.own_rollout(MultiTurnRollout(
            sampler=sampler, template=template, max_turns=1, max_trajectory_tokens=2048,
            sampling_params=SamplingParams(max_tokens=512, num_samples=1, logprobs=1, temperature=1.0, top_p=1.0)))
        checkpoints = ClientCheckpointManager(model, sampler, name_prefix=f'gsm8k-{uuid.uuid4().hex[:12]}')
        checkpoints.sync_weights()

        # The driver generates and scores complete groups while the client trains on the previous batch.
        driver = resources.own_rollout(StreamingDriver(
            create_dataloader(), rollout, batch_size=NUM_GENERATIONS, num_generations=NUM_GENERATIONS,
            checkpoint_manager=checkpoints, max_staleness=1, max_steps=MAX_STEPS,
            orms=[FunctionWorker(GSM8KAccuracyReward())], orm_names=['accuracy'], on_sample=lambda sample: not sample.truncated))
        advantage = GRPOAdvantage(num_generations=NUM_GENERATIONS)
        for rows, facts in driver:
            rows, stats = advantage(rows, scores=facts['scores'])
            if stats['zero_advantage_rows'] == 1.0:
                logger.info('All answers earned the same reward; skipping the update.')
                continue
            model.forward_backward(inputs=rows)
            model.clip_grad_and_step(max_grad_norm=1.0)
            checkpoints.sync_weights()
            logger.info(f'Published {sampler.adapter_path}; metrics={model.calculate_metric(is_training=True)}')
        logger.info(f'Optimizer updates: {model.completed_step_calls}')


if __name__ == '__main__':
    train()
