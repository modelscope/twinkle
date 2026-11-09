# Copyright (c) ModelScope Contributors. All rights reserved.
"""Streaming GRPO on Blackjack: play -> environment reward -> train -> publish.

Reuses the embedded OpenEnv/OpenSpiel game from client/twinkle/multi_turn_rollout.py.
No environment server, generated-code execution or custom verifier is needed.
"""
from __future__ import annotations

import json
import math
import os
import uuid

import dotenv
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.envs import EnvTool, OpenEnv
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.operations import SequentialOperations
from twinkle_client.rollout import ClientMultiTurnRollout

dotenv.load_dotenv('.env')
logger = get_logger()
MODEL_ID = 'ms://' + os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
TEMPLATE_MODEL_ID = os.environ.get('TWINKLE_TEMPLATE_MODEL_ID', MODEL_ID)
ADAPTER_NAME = 'streaming-blackjack'
RUN_ID = f'blackjack-{uuid.uuid4().hex[:12]}'
NUM_GENERATIONS = 4
TRAIN_BATCH_GROUPS = 1
BUFFER_GROUPS = 3
# Leave enough finite draws for sampling to continue after the first publish.
MAX_GROUPS = 16
MAX_STEPS = 2
MAX_STALENESS = 1
BASE_SEED = 20261109

SYSTEM_PROMPT = (
    'Play Blackjack. Use play(action="hit") to draw or play(action="stand") to stop. '
    'Try to get close to 21 without going over; an ace counts as 1 or 11. '
    'After the game ends, give a short final answer without more tool calls.')
PLAY_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'play', 'description': 'Take the next Blackjack action.',
        'parameters': {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': ['hit', 'stand']}}, 'required': ['action']},
    },
}


def blackjack_action_mapper(tool_name, arguments):
    return {'action_id': {'hit': 0, 'stand': 1}[arguments['action']], 'game_name': 'blackjack'}


class BlackjackEnv(OpenEnv):
    """Existing game with readable cards and an episode-local chance RNG."""

    def __init__(self, seed):
        from open_spiel.python.rl_environment import ChanceEventSampler

        super().__init__(env_name='openspiel_env', env_kwargs={
            'game_name': 'blackjack', 'game_params': {'chance_event_sampler': ChanceEventSampler(seed)}},
            action_mapper=blackjack_action_mapper)

    @staticmethod
    def _describe(result):
        # Same visible-card offsets as OpenEnv's Blackjack renderer.
        state = json.loads(result.observation)['info_state']
        if len(state) != 189:
            raise RuntimeError('unexpected OpenSpiel Blackjack observation')
        ranks = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
        player = [ranks[index % 13] for index in range(52) if state[85 + index]]
        dealer = [ranks[index % 13] for index in range(52) if state[137 + index]]
        result.observation = f'Your cards: {player}. Dealer visible cards: {dealer}. Game over: {result.done}.'
        return result

    def reset(self, trajectory=None):
        return self._describe(super().reset(trajectory))

    def step(self, tool_name, arguments=None):
        return self._describe(super().step(tool_name, arguments))


def _identity_sampling_params(seed=None):
    # Shared processed-logprobs must match raw training logits.
    return SamplingParams(max_tokens=256, seed=seed, temperature=1.0, top_p=1.0,
                          top_k=-1, repetition_penalty=1.0, logprobs=1, num_samples=1)


def _build_driver(rollout, initial_uri):
    episodes = {}  # Reward access only; SDK owns handles, admission and draining.

    def submit(ref, trajectory_idx, policy):
        # Equal chance seeds within a group; separate model seeds allow exploration.
        env = BlackjackEnv(ref.row['seed'])
        initial = env.reset()
        tool = EnvTool(env, tool_name='play', description=PLAY_SCHEMA['function']['description'],
                       parameters=PLAY_SCHEMA['function']['parameters'])
        manager = ToolManager([tool])
        handle = rollout.admit(
            [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': initial.observation}],
            trajectory_idx, prompt_idx=ref.ordinal,
            sampling_params=_identity_sampling_params(BASE_SEED + ref.ordinal * NUM_GENERATIONS + trajectory_idx),
            prompt_extras={'tools': manager.tool_infos()}, tool_manager=manager,
            adapter_path=policy.adapter_path, policy_version=policy.version)
        episodes[handle.submission_id] = tool
        return handle

    def collect(handle):
        sample = rollout.collect(handle)
        tool = episodes.pop(handle.submission_id)
        # Answering without finishing the game is not a successful episode.
        sample.truncated = sample.truncated or not tool.done
        sample.orm_reward = [tool.episode_reward]
        return sample

    return StreamingDriver(
        prompt_stream=[{'seed': BASE_SEED + index} for index in range(MAX_GROUPS)],
        num_train_epochs=1, num_generations=NUM_GENERATIONS,
        group_pull=True, need_rows=TRAIN_BATCH_GROUPS * NUM_GENERATIONS,
        groups_per_partition=TRAIN_BATCH_GROUPS, buffer_depth=BUFFER_GROUPS * NUM_GENERATIONS,
        submit=submit, poll=rollout.poll, collect=collect, rollout=rollout,
        version_control=True, initial_adapter_path=lambda: initial_uri, max_staleness=MAX_STALENESS,
        max_steps=MAX_STEPS, max_group_regenerations=0, consume=None,
        prune=lambda: None, snapshot_model=None, score=None, resource_adapter=None)


def _validate_group(group):
    for sample in group:
        if sample.rollout_infos.get('error') is not None:
            raise RuntimeError(f'rollout failed: {sample.rollout_infos}')
    if len(group) != NUM_GENERATIONS:
        return 'incomplete'
    if {sample.extra['trajectory_idx'] for sample in group} != set(range(NUM_GENERATIONS)):
        raise RuntimeError('invalid group members')
    if len({(sample.policy_version, sample.extra['adapter_uri']) for sample in group}) != 1:
        return 'mixed_policy'
    if any(sample.truncated for sample in group):
        return 'truncated'
    for sample in group:
        if (not sample.response_token_ids or len(sample.response_token_ids) != len(sample.rollout_logprobs)
                or not all(math.isfinite(p) for p in sample.rollout_logprobs)):
            raise RuntimeError('missing or invalid rollout probabilities')
        if sample.orm_reward is None or not math.isfinite(sample.orm_reward[0]):
            raise RuntimeError('missing or invalid environment reward')
    return None


def _training_groups(samples, advantage_fn):
    groups = {}
    for sample in samples:
        groups.setdefault(sample.group_key, []).append(sample)
    for group in groups.values():
        reason = _validate_group(group)
        if reason:
            logger.info(f'skip group={group[0].group_key}: {reason}')
            continue
        advantages = advantage_fn([s.orm_reward[0] for s in group],
                                  num_generations=NUM_GENERATIONS, scale='group').tolist()
        if any(abs(value) > 1e-8 for value in advantages):
            yield group, advantages
        else:
            logger.info(f'skip group={group[0].group_key}: zero advantage')


def _train_loop(driver, model, operations):
    confirmed_steps = 0

    def updated(_result):
        nonlocal confirmed_steps
        confirmed_steps += 1

    def publish(_window):
        name = f'{RUN_ID}-policy-{driver.stats.publishes + 1}'
        saved = operations.run(model.save, name, save_optimizer=False)
        logger.info(f'published {saved.twinkle_path}')
        return saved.twinkle_path

    try:
        for samples in driver:
            steps = 0
            for group, advantages in _training_groups(samples, GRPOAdvantage()):
                if confirmed_steps >= MAX_STEPS:
                    logger.info('remaining groups dropped: update budget reached')
                    break
                operations.run(model.forward_backward, inputs=[s.encoded for s in group],
                               advantages=advantages, old_logps=[s.rollout_logprobs for s in group])
                operations.run(model.clip_grad_and_step, max_grad_norm=1.0, on_result=updated)
                steps += 1
                logger.info(f'step={confirmed_steps} rewards={[s.orm_reward[0] for s in group]} '
                            f'metrics={operations.run(model.calculate_metric, is_training=True)}')
            if steps and confirmed_steps < MAX_STEPS:
                driver.request_publish(publish)
            driver.commit(steps)  # Explicit 0 never consumes the optimizer budget.
    finally:
        logger.info(f'confirmed_steps={confirmed_steps} committed_steps={driver.stats.optimizer_steps}')


def train():
    operations, clients = SequentialOperations(), []
    rollout = None
    primary, issues = None, []
    try:
        client = operations.run(init_twinkle_client, on_result=clients.append)
        model = operations.run(client.model, MODEL_ID)
        sampler = operations.run(client.sampler, MODEL_ID)
        operations.run(model.add_adapter_to_model, ADAPTER_NAME, LoraConfig(target_modules='all-linear', r=8, lora_alpha=32))
        operations.run(model.set_loss, 'GRPOLoss', epsilon=0.2, beta=0.0)
        operations.run(model.set_optimizer, 'Adam', lr=2e-5)
        operations.run(model.set_processor, 'InputProcessor')
        operations.run(model.set_template, 'Qwen3_5Template', model_id=TEMPLATE_MODEL_ID, enable_thinking=False)
        operations.run(sampler.set_template, 'Qwen3_5Template', model_id=TEMPLATE_MODEL_ID, enable_thinking=False)
        template = operations.run(Qwen3_5Template, model_id=TEMPLATE_MODEL_ID, max_length=4096, enable_thinking=False)
        template.truncation_strategy = 'delete'
        initial = operations.run(model.save, f'{RUN_ID}-policy-0', save_optimizer=False)
        rollout = ClientMultiTurnRollout(sampler, template, adapter_name=ADAPTER_NAME,
                                       sampling_params=_identity_sampling_params(), max_turns=12,
                                       max_trajectory_tokens=4096)
        # SDK owns cancellation/draining; the training steps stay visible above.
        with rollout.streaming(_build_driver(rollout, initial.twinkle_path)) as driver:
            _train_loop(driver, model, operations)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            operations.close()
        except BaseException as error:
            issues.append(error)
        if clients and (rollout is None or rollout.is_idle):
            try:
                clients[0].close()
            except BaseException as error:
                issues.append(error)
        elif clients:
            issues.append(RuntimeError('rollout execution is unconfirmed; transport retained'))
        if primary is not None:
            for issue in issues:
                primary.add_note(f'cleanup: {issue}')
        elif issues:
            raise BaseExceptionGroup('client cleanup failed', issues)


if __name__ == '__main__':
    train()
