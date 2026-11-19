# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side streaming GRPO: Blackjack -> environment reward -> train -> publish.

StreamingDriver and MultiTurnRollout run locally; HTTPSampler sends each generation
to the Model/Sampler server. Install open_spiel==2.0.2 for the local game, or use
e2b_multi_turn_grpo.py to run the same blackjack_game.py inside E2B sandboxes.
Start with train() for setup and _train_loop() for the training steps.
"""
from __future__ import annotations
import math
import os
import threading
import uuid
from concurrent.futures import wait
from dataclasses import replace
from types import SimpleNamespace

import dotenv
from blackjack_game import BlackjackTask, play
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams, pack_user_data, user_data_get
from twinkle.template import Qwen3_5Template
from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.envs import Env, StepResult
from twinkle_agentic.rollout import MultiTurnRollout
from twinkle_client.resources import RolloutResources
from twinkle_client.rollout import ClientCheckpointManager, HTTPSampler

dotenv.load_dotenv('.env')
logger = get_logger()
MODEL_ID = 'ms://' + os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
TEMPLATE_MODEL_ID = os.environ.get('TWINKLE_TEMPLATE_MODEL_ID', MODEL_ID)
ADAPTER_NAME = 'streaming-blackjack'
NUM_GENERATIONS = 4
MAX_GROUPS = 16  # Finite prompt groups, including groups rejected or skipped.
MAX_STEPS = 2  # Confirmed optimizer updates, not Driver deliveries or publications.
MAX_STALENESS = 1  # Maximum published-version lag; also sizes the initial sampling window.
BASE_SEED = 20261109


class _RolloutTasks:
    """Keep the client and environments alive until issued local rollout calls settle.

    Driver.close() stops scheduling; MultiTurnRollout.close() releases episodes. Neither
    waits for an already issued HTTP generation to return, so this scope waits for its
    step/stop Futures before RolloutResources shuts down the environments and client.
    """

    def __init__(self, rollout):
        self.rollout = rollout
        self._pending = set()
        self._lock = threading.RLock()
        self._closed = False

    def _submit(self, method, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError('rollout task scope is closed')
            future = method(*args, **kwargs)
            self._pending.add(future)
            future.add_done_callback(self._settled)
            return future

    def _settled(self, future):
        with self._lock:
            self._pending.discard(future)

    def step(self, *args, **kwargs):
        return self._submit(self.rollout.step, *args, **kwargs)

    def stop(self, *args, **kwargs):
        return self._submit(self.rollout.stop, *args, **kwargs)

    def close(self):
        with self._lock:
            self._closed = True
            pending = list(self._pending)
        wait(pending)
        self.rollout.close()


class BlackjackEnv(BlackjackTask, Env):
    """One component, one independent game history per episode session."""

    def _open(self, trajectory):
        seed = int(user_data_get(trajectory['user_data'], 'seed'))
        # Replaying this tiny game uses a fresh chance RNG, just as in the E2B game.
        # Equal seeds within a group never mean sharing a mutable RNG or history.
        result = StepResult(**play(seed, []))
        return SimpleNamespace(seed=seed, actions=[], last_result=result), result

    def _step(self, state, tool_name, arguments):
        action = arguments.get('action')
        if tool_name != 'play' or action not in ('hit', 'stand'):
            return StepResult(observation='Use play(action="hit") or play(action="stand").')
        if state.last_result.done:
            # Repeated calls after the terminal action must not pay its reward again.
            return replace(state.last_result, reward=0.0)
        state.actions.append(action)
        state.last_result = StepResult(**play(state.seed, state.actions))
        return state.last_result


def _grpo_sampling_params():
    # Independent stochastic generations; only the environment seed is shared by a group.
    # Processed sampling logprobs must match raw training logits.
    return SamplingParams(max_tokens=256, temperature=1.0, top_p=1.0,
                          top_k=-1, repetition_penalty=1.0, logprobs=1, num_samples=1)


def _episode_end(trajectory, env, episode_id):
    return env.episode_outcome(episode_id)


def _accept_sample(sample):
    # A final model answer does not prove the game finished. Rejecting a member here
    # drops its incomplete group before delivery and returns its admission budget.
    if sample.extra.get('error'):
        raise RuntimeError(f'Blackjack episode {sample.episode_id} failed: {sample.extra["error"]}')
    if not math.isfinite(sample.env_reward):
        raise ValueError(f'Blackjack episode {sample.episode_id} returned a non-finite reward')
    return not sample.truncated and bool(sample.extra.get('game_done'))


def _build_driver(rollout, checkpoint_manager):
    prompts = [
        {'messages': [{'role': 'system', 'content': BlackjackTask.system_prompt}],
         'user_data': pack_user_data({'seed': BASE_SEED + index})}
        for index in range(MAX_GROUPS)
    ]
    return StreamingDriver(
        dataloader=prompts, rollout=rollout,
        num_generations=NUM_GENERATIONS, batch_size=NUM_GENERATIONS,
        checkpoint_manager=checkpoint_manager, max_staleness=MAX_STALENESS,
        sync_interval=1, on_sample=_accept_sample,
        # max_steps counts deliveries; MAX_STEPS is enforced after actual model updates.
    )


def _train_loop(driver, model, checkpoint_manager):
    """Complete groups -> GRPO targets -> optional optimizer update -> publish."""
    initial_step_calls = model.completed_step_calls
    confirmed_steps = 0
    advantage_fn = GRPOAdvantage(num_generations=NUM_GENERATIONS, scale_rewards='group')
    try:
        if MAX_STEPS <= 0:
            return
        for rows, facts in driver:
            # Env reward is stored on the trajectory; GRPO reads named score columns.
            scores = [{'rewards': {'orm': {'blackjack': row['rewards']['env']}}} for row in rows]
            train_rows, _ = advantage_fn(rows, scores=scores, global_step=confirmed_steps)
            advantages = [row['advantages'] for row in train_rows]
            if any(abs(value) > 1e-8 for values in advantages for value in values):
                model.forward_backward(
                    inputs=[{key: row[key] for key in ('input_ids', 'labels', 'completion_mask')}
                            for row in train_rows],
                    advantages=advantages, old_logps=[row['old_logps'] for row in train_rows])
                model.clip_grad_and_step(max_grad_norm=1.0)
                confirmed_steps = model.completed_step_calls - initial_step_calls
                logger.info(f'step={confirmed_steps} rewards={[row["rewards"]["env"] for row in rows]} '
                            f'metrics={model.calculate_metric(is_training=True)}')
            else:
                logger.info(f'skip group={rows[0]["group_key"]}: zero advantage')
            # Publish once per consumed batch, even when its update is skipped: the
            # Driver's admission bound advances with published versions. These versions
            # count snapshots, not optimizer steps. Save the final trained policy too.
            checkpoint_manager.sync_weights()
            logger.info(f'published version={checkpoint_manager.version()} rows={facts["num_rows"]}')
            if confirmed_steps >= MAX_STEPS:
                break
    finally:
        confirmed_steps = model.completed_step_calls - initial_step_calls
        logger.info(f'confirmed_steps={confirmed_steps} delivered_batches={driver.stats()["handed"]}')


def train(env_factory=None):
    """Run with a session-based Blackjack Env; env_factory creates the component once."""
    checkpoint_prefix = f'blackjack-{uuid.uuid4().hex[:12]}'
    with RolloutResources(
        init_twinkle_client,
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    ) as resources:
        client = resources.client
        model = client.model(MODEL_ID)
        sampler = client.sampler(MODEL_ID)
        model.add_adapter_to_model(ADAPTER_NAME, LoraConfig(target_modules='all-linear', r=8, lora_alpha=32))
        model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
        model.set_optimizer('Adam', lr=2e-5)
        model.set_processor('InputProcessor')
        model.set_template('Qwen3_5Template', model_id=TEMPLATE_MODEL_ID, enable_thinking=False)
        sampler.set_template('Qwen3_5Template', model_id=TEMPLATE_MODEL_ID, enable_thinking=False)
        template = Qwen3_5Template(model_id=TEMPLATE_MODEL_ID, max_length=4096, enable_thinking=False)
        template.truncation_strategy = 'delete'

        http_sampler = HTTPSampler(sampler, template, adapter_name=ADAPTER_NAME)
        checkpoint_manager = ClientCheckpointManager(model, http_sampler, name_prefix=checkpoint_prefix)
        checkpoint_manager.sync_weights()
        env = resources.own_environment((env_factory or BlackjackEnv)(
            max_sessions=(MAX_STALENESS + 1) * NUM_GENERATIONS))
        rollout = resources.own_rollout(_RolloutTasks(MultiTurnRollout(
            sampler=http_sampler, template=template, env=env,
            sampling_params=_grpo_sampling_params(), on_episode_end=_episode_end,
            max_turns=12, max_trajectory_tokens=4096)))
        # No fixed adapter_path: later turns read the HTTP sampler's latest published URI.
        driver = _build_driver(rollout, checkpoint_manager)
        try:
            _train_loop(driver, model, checkpoint_manager)
        finally:
            driver.close()  # Then RolloutResources closes rollout, Env sessions and client.


if __name__ == '__main__':
    train()
