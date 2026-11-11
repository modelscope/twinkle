# Copyright (c) ModelScope Contributors. All rights reserved.
"""Streaming GRPO on Blackjack: play -> environment reward -> train -> publish.

Reuses the embedded OpenEnv/OpenSpiel game from client/twinkle/multi_turn_rollout.py.
No environment server, generated-code execution or custom verifier is needed.
Start with train() for setup and _train_loop() for the training steps.
Use e2b_multi_turn_grpo.py to run the same game inside E2B sandboxes.
"""
from __future__ import annotations
import math
import os
import uuid

import dotenv
import json
from blackjack_game import BlackjackTask
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.envs import EnvTool, OpenEnv
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.resources import RolloutResources
from twinkle_client.rollout import ClientMultiTurnRollout

dotenv.load_dotenv('.env')
logger = get_logger()
MODEL_ID = 'ms://' + os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
TEMPLATE_MODEL_ID = os.environ.get('TWINKLE_TEMPLATE_MODEL_ID', MODEL_ID)
ADAPTER_NAME = 'streaming-blackjack'
NUM_GENERATIONS = 4
BUFFER_GROUPS = 3
MAX_GROUPS = 16  # Finite prompt-group attempts, including skipped groups.
MAX_STEPS = 2  # Optimizer-update budget; a short run may use fewer steps.
MAX_STALENESS = 1  # Allow samples from at most one published policy behind.
BASE_SEED = 20261109


class BlackjackEnv(BlackjackTask, OpenEnv):
    """Existing game with readable cards and an episode-local chance RNG."""

    def __init__(self, seed):
        from open_spiel.python.rl_environment import ChanceEventSampler

        super().__init__(env_name='openspiel_env', env_kwargs={
            'game_name': 'blackjack', 'game_params': {'chance_event_sampler': ChanceEventSampler(seed)}},
            action_mapper=self._action)

    @staticmethod
    def _action(tool_name, arguments):
        return {'action_id': {'hit': 0, 'stand': 1}[arguments['action']], 'game_name': 'blackjack'}

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


def _grpo_sampling_params(seed=None):
    # Shared processed-logprobs must match raw training logits.
    return SamplingParams(max_tokens=256, seed=seed, temperature=1.0, top_p=1.0,
                          top_k=-1, repetition_penalty=1.0, logprobs=1, num_samples=1)


def _build_driver(rollout, initial_uri, resources, env_factory=None):
    """Adapt Blackjack to the driver; env_factory selects the game's execution backend."""
    episodes = {}  # Reward access only; SDK owns handles, admission and draining.
    env_factory = env_factory or BlackjackEnv

    def submit(ref, trajectory_idx, policy):
        # Equal chance seeds within a group; separate model seeds allow exploration.
        env = resources.own_environment(env_factory(ref.row['seed']))
        initial = env.reset()
        tools = EnvTool.from_schemas(env, env.tools())
        if len(tools) != 1 or tools[0].tool_info()['function']['name'] != 'play':
            raise ValueError('Blackjack expects a single play tool; adapt submit/collect for a different task')
        tool = tools[0]
        manager = ToolManager([tool])
        handle = rollout.admit(
            [{'role': 'system', 'content': env.system_prompt}, {'role': 'user', 'content': initial.observation}],
            trajectory_idx, prompt_idx=ref.ordinal,
            sampling_params=_grpo_sampling_params(BASE_SEED + ref.ordinal * NUM_GENERATIONS + trajectory_idx),
            prompt_extras={'tools': manager.tool_infos()}, tool_manager=manager,
            adapter_path=policy.adapter_path, policy_version=policy.version)
        episodes[handle.submission_id] = (env, tool)
        return handle

    def collect(handle):
        sample = rollout.collect(handle)
        env, tool = episodes.pop(handle.submission_id)
        # Answering without finishing the game is not a successful episode.
        sample.truncated = sample.truncated or not tool.done
        sample.orm_reward = [tool.episode_reward]
        if tool.last_result is not None and tool.last_result.info.get('error'):
            sample.rollout_infos['error'] = tool.last_result.info['error']
        resources.close_environment(env)
        return sample

    return StreamingDriver(
        prompt_stream=[{'seed': BASE_SEED + index} for index in range(MAX_GROUPS)],
        num_generations=NUM_GENERATIONS,
        # Pull one complete group for each training iteration.
        group_pull=True, need_rows=NUM_GENERATIONS,
        groups_per_partition=1, buffer_depth=BUFFER_GROUPS * NUM_GENERATIONS,
        rollout=rollout, submit=submit, collect=collect,
        version_control=True, initial_adapter_path=lambda: initial_uri, max_staleness=MAX_STALENESS,
        # No snapshot_model: the driver leaves server checkpoints untouched.
        max_steps=MAX_STEPS)


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


def _complete_groups(samples):
    """Keep complete, single-policy groups; invalid rollout data fails loudly."""
    groups = {}
    for sample in samples:
        groups.setdefault(sample.group_key, []).append(sample)
    for group in groups.values():
        reason = _validate_group(group)
        if reason:
            logger.info(f'skip group={group[0].group_key}: {reason}')
            continue
        yield group


def _train_loop(driver, model, operations, checkpoint_prefix):
    """Complete groups -> GRPO advantages -> backward/step -> publish -> acknowledge."""
    confirmed_steps = 0
    advantage_fn = GRPOAdvantage()

    def record_step(_result):
        # Count a confirmed update even if the caller is interrupted while waiting.
        nonlocal confirmed_steps
        confirmed_steps += 1

    def publish(_window):
        name = f'{checkpoint_prefix}-policy-{driver.stats.publishes + 1}'
        saved = operations.run(model.save, name, save_optimizer=False)
        logger.info(f'published {saved.twinkle_path}')
        return saved.twinkle_path

    try:
        for samples in driver:
            steps = 0
            for group in _complete_groups(samples):
                if confirmed_steps >= MAX_STEPS:
                    logger.info('remaining groups dropped: update budget reached')
                    break
                rewards = [sample.orm_reward[0] for sample in group]
                advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
                if not any(abs(value) > 1e-8 for value in advantages):
                    logger.info(f'skip group={group[0].group_key}: zero advantage')
                    continue
                operations.run(model.forward_backward, inputs=[s.encoded for s in group],
                               advantages=advantages, old_logps=[s.rollout_logprobs for s in group])
                operations.run(model.clip_grad_and_step, max_grad_norm=1.0, on_result=record_step)
                steps += 1
                logger.info(f'step={confirmed_steps} rewards={rewards} '
                            f'metrics={operations.run(model.calculate_metric, is_training=True)}')
            if steps and confirmed_steps < MAX_STEPS:
                driver.request_publish(publish)
            driver.commit(steps)  # Explicit 0 never consumes the optimizer budget.
    finally:
        logger.info(f'confirmed_steps={confirmed_steps} committed_steps={driver.stats.optimizer_steps}')


def train(env_factory=None):
    """Run training with a Blackjack Env exposing system_prompt, tools(), reset() and step()."""
    # Automatic checkpoint names only: this is not a tenant/session ID or a sampling seed.
    checkpoint_prefix = f'blackjack-{uuid.uuid4().hex[:12]}'
    with RolloutResources(init_twinkle_client) as resources:
        client, operations = resources.client, resources.operations
        # 1. Connect to the training model and sampling model on the same server.
        model = operations.run(client.model, MODEL_ID)
        sampler = operations.run(client.sampler, MODEL_ID)
        operations.run(model.add_adapter_to_model, ADAPTER_NAME,
                       LoraConfig(target_modules='all-linear', r=8, lora_alpha=32))
        operations.run(model.set_loss, 'GRPOLoss', epsilon=0.2, beta=0.0)
        operations.run(model.set_optimizer, 'Adam', lr=2e-5)
        operations.run(model.set_processor, 'InputProcessor')
        operations.run(model.set_template, 'Qwen3_5Template', model_id=TEMPLATE_MODEL_ID, enable_thinking=False)
        operations.run(sampler.set_template, 'Qwen3_5Template', model_id=TEMPLATE_MODEL_ID, enable_thinking=False)
        template = operations.run(Qwen3_5Template, model_id=TEMPLATE_MODEL_ID, max_length=4096, enable_thinking=False)
        template.truncation_strategy = 'delete'
        # 2. Freeze policy-0; every trajectory uses one immutable policy URI.
        initial = operations.run(model.save, f'{checkpoint_prefix}-policy-0', save_optimizer=False)
        rollout = resources.own_rollout(ClientMultiTurnRollout(
            sampler, template, adapter_name=ADAPTER_NAME, sampling_params=_grpo_sampling_params(),
            max_turns=12, max_trajectory_tokens=4096))
        # 3. The driver yields complete groups; the SDK drains rollout on exit.
        with rollout.streaming(_build_driver(rollout, initial.twinkle_path, resources, env_factory)) as driver:
            _train_loop(driver, model, operations, checkpoint_prefix)


if __name__ == '__main__':
    train()
