"""Multi-turn GRPO training with EnvPool (integrated environment pool).

Demonstrates how to train an LLM agent via GRPO on interactive environments
(e.g. Blackjack) using EnvPool and Twinkle's MultiTurnRollout.

EnvPool is deployed as a @remote_class component — either:
  - With remote_group='env': runs on a dedicated CPU DeviceGroup (isolated)
  - Without remote_group: runs locally in the driver (zero RPC overhead)

The agent interacts with environments through tool calls:
  1. EnvPool manages N env instances; each trajectory maps to one slot.
  2. MultiTurnRollout drives the multi-turn loop: model generates tool calls,
     EnvTool dispatches them to env.step(), observations are fed back.
  3. Episode reward is extracted after rollout completes.
  4. GRPO advantages are computed across the batch and used for policy update.

Usage:
  # No need to start a separate server — environments are instantiated
  # directly inside the EnvPool worker:
  #   python multi_turn_grpo.py
  #
  # To run envs on a dedicated CPU worker (isolated):
  #   ENV_REMOTE=1 python multi_turn_grpo.py

References:
  - OpenEnv GRPO Blackjack: https://github.com/huggingface/OpenEnv/tree/main/examples/grpo_blackjack
  - cookbook/rl/grpo/short_math_grpo.py (single-turn GRPO template)
"""
import os
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.envs import EnvPool, EnvPoolAdapter, EnvTool
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
USE_MEGATRON = False

MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 2048
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 1000
BATCH_SIZE = args.training.batch_size or 4
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
ADAPTER_NAME = args.lora.adapter_name or 'default'
SAVE_STEPS = args.training.save_steps or 500
LORA_RANK = args.lora.lora_r or 16
MAX_TURNS = int(os.environ.get('MAX_TURNS', '6'))

# Environment configuration
ENV_GAME = os.environ.get('ENV_GAME', 'blackjack')
# ENV_REMOTE: set to '1' to deploy envs on a dedicated CPU DeviceGroup
ENV_REMOTE = os.environ.get('ENV_REMOTE', '0') == '1'
# ENV_NUM_WORKERS: number of CPU workers for distributed EnvPool (default 1)
ENV_NUM_WORKERS = int(os.environ.get('ENV_NUM_WORKERS', '1'))
# Pool size = total trajectories per batch
ENV_POOL_SIZE = int(os.environ.get('ENV_POOL_SIZE', '0'))  # 0 = auto

# ========== Tool Schema (Blackjack example) ==========
# Define tools the model can use in the environment.
# For blackjack: a single "play" tool with hit/stand actions.
# Override TOOL_SCHEMA for different environments.
BLACKJACK_TOOL_SCHEMA = [
    {
        'type': 'function',
        'function': {
            'name': 'play',
            'description': 'Take an action in the blackjack game.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action': {
                        'type': 'string',
                        'enum': ['hit', 'stand'],
                        'description': 'The action to take: "hit" to draw a card, "stand" to keep current hand.',
                    }
                },
                'required': ['action'],
            },
        },
    }
]

TOOL_SCHEMA = BLACKJACK_TOOL_SCHEMA

# Action name → OpenSpiel action_id mapping for blackjack.
# OpenSpiel blackjack: 0 = HIT, 1 = STAND
BLACKJACK_ACTION_MAP = {'hit': 0, 'stand': 1}


def blackjack_action_mapper(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Map tool calls to OpenSpielAction format.

    Converts  play(action='hit')  →  {action_id: 0, game_name: 'blackjack'}
    """
    action_str = str(arguments.get('action', 'stand')).lower().strip()
    action_id = BLACKJACK_ACTION_MAP.get(action_str, 1)  # default STAND
    return {'action_id': action_id, 'game_name': 'blackjack'}


SYSTEM_PROMPT = """You are a skilled blackjack player. You will be told your current hand and the dealer's visible card.

Your goal is to win the game by getting as close to 21 as possible without going over.

Strategy guidelines:
- Hit if your hand total is below 12
- Consider the dealer's visible card when deciding
- Stand if you have 17 or higher
- Be cautious with hard hands (no ace counted as 11)

Use the `play` tool to take actions. Always reason briefly before acting."""


# ========== Environment Setup ==========
def prepare_trajectories(
    env_pool: EnvPool,
    n_trajectories: int,
    tool_schema: List[Dict],
    system_prompt: str,
) -> Tuple[List[Dict[str, Any]], List[ToolManager], List[List[EnvTool]]]:
    """Reset environments via EnvPool and build initial trajectories.

    For each trajectory:
      1. Get an EnvPoolAdapter (standard Env interface) from the pool
      2. Reset the env slot to get initial observation
      3. Build a trajectory dict with system + user messages and tools

    Args:
        env_pool: The EnvPool instance managing all environments.
        n_trajectories: Total number of trajectories to create.
        tool_schema: Tool definitions for the environment.
        system_prompt: System prompt for the agent.

    Returns:
        Tuple of (trajectories, tool_managers, env_tools_list).
    """
    adapters = env_pool.get_adapters(n=n_trajectories)

    trajectories = []
    tool_managers = []
    env_tools_list = []

    for adapter in adapters:
        # Reset env slot to start a new episode
        initial_result = adapter.reset()
        initial_obs = initial_result.observation

        # Create EnvTool and ToolManager for this trajectory
        env_tools = EnvTool.from_env(adapter)
        tm = ToolManager(env_tools)

        # Build trajectory with initial observation as user message
        traj = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': initial_obs},
            ],
            'tools': tool_schema,
        }

        trajectories.append(traj)
        tool_managers.append(tm)
        env_tools_list.append(env_tools)

    return trajectories, tool_managers, env_tools_list


def extract_rewards(env_tools_list: List[List[EnvTool]]) -> List[float]:
    """Extract episode rewards from EnvTool instances after rollout.

    Each EnvTool tracks the cumulative episode reward from env.step() calls.
    """
    rewards = []
    for env_tools in env_tools_list:
        if env_tools:
            reward = env_tools[0].episode_reward
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


# ========== Main ==========
def main():
    # Determine pool size
    n_trajectories = BATCH_SIZE * NUM_GENERATIONS
    pool_size = ENV_POOL_SIZE if ENV_POOL_SIZE > 0 else n_trajectories

    # Device groups: model + sampler + (optionally) env
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]

    if ENV_REMOTE:
        # Add a CPU-only DeviceGroup for env pool
        # ranks=ENV_NUM_WORKERS: each rank becomes one EnvPool worker
        device_groups.append(
            DeviceGroup(name='env', ranks=ENV_NUM_WORKERS, device_type='CPU'),
        )

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
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
            'max_lora_rank': 32,
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    # ========== EnvPool: environment instances managed by Twinkle ==========
    # OpenEnv embedded mode: Ray workers replace the HTTP server process pool.
    # Equivalent to: OPENSPIEL_GAME=blackjack python -m envs.openspiel_env.server.app
    # but fully managed by Ray — no manual server startup needed.
    env_pool_kwargs = dict(
        pool_size=pool_size,
        env_kwargs={
            'env_name': 'openspiel_env',
            'env_kwargs': {'game_name': ENV_GAME},
            'action_mapper': blackjack_action_mapper,
        },
    )
    if ENV_REMOTE:
        env_mesh = DeviceMesh.from_sizes(world_size=ENV_NUM_WORKERS, dp_size=ENV_NUM_WORKERS)
        env_pool_kwargs['remote_group'] = 'env'
        env_pool_kwargs['device_mesh'] = env_mesh

    env_pool = EnvPool(**env_pool_kwargs)
    logger.info(f'EnvPool created: game={ENV_GAME}, '
                f'pool_size={pool_size}, remote={ENV_REMOTE}')

    # Local template for MultiTurnRollout bridge computation
    rollout_template = Qwen3_5Template(MODEL_ID, max_length=8192, enable_thinking=False)
    rollout_template.truncation_strategy = 'delete'

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    # MultiTurnRollout: tool_manager is optional at construction time;
    # the actual per-trajectory ToolManagers are provided at call time.
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95,
    )
    rollout = MultiTurnRollout(
        sampler=sampler,
        template=rollout_template,
        sampling_params=sampling_params,
        max_turns=MAX_TURNS,
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    optim_step = 0
    logger.info('Starting multi-turn GRPO training with EnvPool')
    logger.info(get_device_placement())

    while optim_step < MAX_STEPS:
        metrics.reset()

        # Total trajectories per batch: BATCH_SIZE * NUM_GENERATIONS
        # Each trajectory is an independent game episode.
        n_traj = BATCH_SIZE * NUM_GENERATIONS

        # 1. Prepare environments and initial trajectories
        logger.info(f'[Step {optim_step}] Resetting {n_traj} environments...')
        expand_prompts, tool_managers, env_tools_list = prepare_trajectories(
            env_pool=env_pool,
            n_trajectories=n_traj,
            tool_schema=TOOL_SCHEMA,
            system_prompt=SYSTEM_PROMPT,
        )

        # 2. Sync model weights to sampler
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        # 3. Run multi-turn rollout with per-trajectory ToolManagers
        all_trajectories: List[Dict[str, Any]] = rollout(
            expand_prompts,
            tool_manager=tool_managers,
        )

        # 4. Extract rewards and logprobs
        env_rewards = extract_rewards(env_tools_list)

        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []
        n_turns_per_rollout: List[int] = []

        for traj in all_trajectories:
            logprobs = traj.get('logprobs') or []
            old_logps = [lp[0][1] for lp in logprobs] if logprobs else []
            all_old_logps.append(old_logps)
            # Completion length = number of trainable tokens (labels != -100)
            labels = traj.get('labels') or []
            comp_len = sum(1 for l in labels if l != -100)
            all_completion_lengths.append(comp_len)
            n_turns_per_rollout.append(int(traj.get('turns') or 0))

        # 5. Compute advantages (group-relative within NUM_GENERATIONS)
        total_rewards = env_rewards
        advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group',
        ).tolist()

        # 6. Log metrics
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={'total': total_rewards},
        )

        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        avg_turns = sum(n_turns_per_rollout) / len(n_turns_per_rollout) if n_turns_per_rollout else 0.0
        logger.info(f'[Step {optim_step}] avg_reward={avg_reward:.3f}, avg_turns={avg_turns:.1f}')

        # 7. Forward-backward with mini-batches
        # Filter out oversized/truncated trajectories (strategy='delete'),
        # keep only those with valid completions and ensure >= MODEL_GPUS inputs.
        all_input_data: List[Dict[str, Any]] = []
        filtered_old_logps: List[List[float]] = []
        filtered_advantages: List[float] = []
        max_len = rollout_template.max_length or float('inf')
        for i, traj in enumerate(all_trajectories):
            traj_len = len(traj.get('input_ids') or traj.get('labels') or [])
            comp_len = sum(1 for l in (traj.get('labels') or []) if l != -100)
            if traj_len > max_len or comp_len == 0:
                continue
            all_input_data.append(traj)
            filtered_old_logps.append(all_old_logps[i])
            filtered_advantages.append(advantages[i])

        if len(all_input_data) < MODEL_GPUS:
            logger.warning(f'[Step {optim_step}] Only {len(all_input_data)} valid trajectories '
                           f'after filtering (need >= {MODEL_GPUS}), skipping this batch.')
            continue

        all_old_logps = filtered_old_logps
        advantages = filtered_advantages
        total_completions = len(all_input_data)
        logger.info(f'[Step {optim_step}] {total_completions}/{n_traj} trajectories '
                    f'passed length filter (max_len={max_len})')

        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]

            # Print trajectory lengths before forward_backward
            traj_lengths = []
            for idx, traj in enumerate(mb_inputs):
                labels = traj.get('labels') or traj.get('input_ids') or []
                traj_lengths.append(len(labels))
            logger.info(f'[Step {optim_step}] mini-batch [{mb_start}:{mb_end}] '
                        f'n_inputs={len(mb_inputs)}, dp_world={MODEL_GPUS}, '
                        f'traj_lengths={traj_lengths}')

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'multi-turn-grpo-checkpoint-{optim_step}')

        # 8. Log step summary
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict['avg_turns'] = avg_turns
        log_dict['avg_reward'] = avg_reward
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    # Cleanup
    env_pool.close()
    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('multi-turn-grpo-final')


if __name__ == '__main__':
    main()
