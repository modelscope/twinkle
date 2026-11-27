# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stream Blackjack games from a Twinkle sampler, then train and publish a LoRA."""

import json
import os
import uuid

import dotenv
from peft import LoraConfig

from twinkle import get_logger
from twinkle_client import init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams, pack_user_data, user_data_get
from twinkle.template import Qwen3_5Template
from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.envs import OpenEnv, StepResult
from twinkle_agentic.rollout import MultiTurnRollout
from twinkle_client.resources import RolloutResources
from twinkle_client.rollout import ClientCheckpointManager, HTTPSampler

if __package__:
    from .blackjack_game import BlackjackTask, cards_observation
else:
    from blackjack_game import BlackjackTask, cards_observation

dotenv.load_dotenv('.env')
logger = get_logger()
MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
if '://' not in MODEL_ID:
    MODEL_ID = f'ms://{MODEL_ID}'
TEMPLATE_MODEL_ID = os.environ.get('TWINKLE_TEMPLATE_MODEL_ID', MODEL_ID)
NUM_GENERATIONS = 4
MAX_GROUPS = 16
MAX_STEPS = 2
BASE_SEED = 20261109


class BlackjackEnv(BlackjackTask, OpenEnv):
    """Use the embedded OpenEnv game, with independent RNG state per episode."""

    def __init__(self):
        super().__init__(env_name='openspiel_env', action_mapper=self._action)

    @staticmethod
    def _action(tool_name, arguments):
        return {'action_id': {'hit': 0, 'stand': 1}[arguments['action']], 'game_name': 'blackjack'}

    def _open(self, trajectory):
        from open_spiel.python.rl_environment import ChanceEventSampler

        seed = user_data_get(trajectory['user_data'], 'seed')
        game = self._env_cls(game_name='blackjack', game_params={'chance_event_sampler': ChanceEventSampler(seed)})
        try:
            result = game.reset()
            opening = StepResult(observation=cards_observation(result.info_state, result.done))
        except BaseException:
            super()._close(game)
            raise
        return (game, opening), opening

    def _step(self, state, tool_name, arguments):
        game, last = state
        if last.done:
            return StepResult(observation=last.observation, done=True)
        # Keep the terminal result in this session so extra calls cannot score it twice.
        result = super()._step(game, tool_name, arguments)
        last.observation = cards_observation(json.loads(result.observation)['info_state'], result.done)
        last.done = result.done
        return StepResult(observation=last.observation, done=result.done, reward=result.reward)

    def _close(self, state):
        super()._close(state[0])


def prompt(seed):
    return {'messages': [{'role': 'system', 'content': BlackjackTask.system_prompt}], 'user_data': pack_user_data({'seed': seed})}


def setup_training(resources, env):
    client = resources.client
    model = client.model(MODEL_ID)
    adapter_name = 'blackjack'
    model.add_adapter_to_model(adapter_name, LoraConfig(target_modules='all-linear', r=8, lora_alpha=32))
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
    model.set_optimizer('Adam', lr=2e-5)
    model.set_processor('InputProcessor')
    model.set_template('Qwen3_5Template', enable_thinking=False)

    sampler = client.sampler(MODEL_ID)
    sampler.set_template('Qwen3_5Template', enable_thinking=False)
    template = Qwen3_5Template(model_id=TEMPLATE_MODEL_ID, max_length=4096, enable_thinking=False)
    template.truncation_strategy = 'delete'
    sampler = HTTPSampler(sampler, template, adapter_name=adapter_name)

    # Each trajectory samples one reply per turn and keeps the sampled-token logprobs.
    rollout = resources.own_rollout(MultiTurnRollout(
        sampler=sampler, template=template, env=env, max_turns=12, max_trajectory_tokens=4096,
        sampling_params=SamplingParams(max_tokens=256, num_samples=1, logprobs=1, temperature=1.0, top_p=1.0),
        on_episode_end=env.episode_outcome))
    checkpoints = ClientCheckpointManager(model, sampler, name_prefix=f'blackjack-{uuid.uuid4().hex[:12]}')
    checkpoints.sync_weights()
    return model, rollout, checkpoints


def train_batch(model, checkpoints, advantage, rows):
    scores = [{'rewards': {'orm': {'blackjack': row['rewards']['env']}}} for row in rows]
    rows, summary = advantage(rows, scores=scores)
    if summary['zero_advantage_rows'] == 1.0:
        logger.info('All advantages are zero; skipping the update.')
        return False
    model.forward_backward(inputs=rows)
    model.clip_grad_and_step(max_grad_norm=1.0)
    checkpoints.sync_weights()
    logger.info(f'Published {checkpoints.sampler.adapter_path}; metrics={model.calculate_metric(is_training=True)}')
    return True


def report_training(model, groups):
    checkpoint = model.last_checkpoint
    lora_uri = checkpoint.twinkle_path if checkpoint is not None else None
    updates = model.completed_step_calls
    logger.info(f'Blackjack summary: completed_groups={groups} optimizer_updates={updates} latest_lora={lora_uri}')
    if updates == 0:
        logger.info('No training updates. Check whether games finish and rewards differ within a group.')


def train(env_factory=BlackjackEnv):
    with RolloutResources(init_twinkle_client) as resources:
        env = resources.own_environment(env_factory())
        model, rollout, checkpoints = setup_training(resources, env)
        advantage = GRPOAdvantage(num_generations=NUM_GENERATIONS)

        # One seed per deal. The driver draws several independent games from each deal.
        prompts = ([prompt(BASE_SEED + index)] for index in range(MAX_GROUPS))
        driver = resources.own_rollout(StreamingDriver(
            prompts, rollout, batch_size=NUM_GENERATIONS, num_generations=NUM_GENERATIONS,
            checkpoint_manager=checkpoints,
            # Keep only groups from the current policy at delivery.
            max_staleness=0,
            on_sample=lambda sample: not sample.truncated and sample.extra['game_finished'],
        ))
        groups = 0
        try:
            for rows, _ in driver:
                groups += 1
                train_batch(model, checkpoints, advantage, rows)
                if model.completed_step_calls == MAX_STEPS:
                    break
        finally:
            report_training(model, groups)


if __name__ == '__main__':
    train()
