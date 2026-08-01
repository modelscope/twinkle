"""Multi-turn GRPO on MBPP with a remote OpenEnv code-interpreter server.

Server-mode counterpart of ``cookbook/rl/agentenv/agentenv_grpo.py``: the same
code-writing task, but every trajectory gets a **session on a remote OpenEnv
server** instead of a Firecracker microVM. Compare the two to see what the
execution backend does and does not change.

Architecture:
  - No EnvPool / @remote_class. An OpenEnvClient is a WebSocket client and the
    session lives server-side, so Ray sharding buys nothing. The driver creates
    one client per trajectory and resets them on a thread pool, because each
    connect+reset is a blocking network call.
  - One server process hosts many sessions (``MAX_CONCURRENT_ENVS``). Capacity
    must cover BATCH_SIZE x NUM_GENERATIONS concurrent trajectories.

Per-trajectory flow:
  1. reset() opens a session; the server builds a fresh Python interpreter.
  2. MultiTurnRollout drives tool calls: run_python executes code in the
     session (state persists across turns); submit_solution records the final
     source client-side.
  3. Reward = MBPP unit-test pass rate, measured by replaying the hidden tests
     in the same session after the rollout (see ``tools.run_tests``).
  4. GRPO advantages are group-relative across NUM_GENERATIONS rollouts of the
     same problem.

Prerequisites:
  1. Start the environment server (any reachable host, no GPU/KVM needed):
       pip install openenv && pip install -e /path/to/OpenEnv/envs/coding_env
       sh serve.sh
  2. On the training side: pip install openenv

Usage:
  OPENENV_BASE_URL=http://<server-host>:8000 sh openenv_code_grpo.sh
"""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.envs import EnvTool, OpenEnvClient
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager
from tools import SYSTEM_PROMPT, TOOL_SCHEMA, register_tools, run_tests

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'

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

# The OpenEnv server. Behind a load balancer this is the LB address; the client
# never needs to know which backend serves its session.
OPENENV_BASE_URL = os.environ.get('OPENENV_BASE_URL', 'http://127.0.0.1:8000')
OPENENV_ENV_NAME = os.environ.get('OPENENV_ENV_NAME', 'coding_env')
# Parallelism for the blocking connect/reset/score calls in the driver.
ENV_CONCURRENCY = int(os.environ.get('ENV_CONCURRENCY', '16'))


# ========== Dataset (MBPP) ==========
def load_mbpp() -> List[Dict[str, Any]]:
    """Load MBPP as [{'prompt', 'test_list', 'test_setup_code'}]."""
    from modelscope.msdatasets import MsDataset
    raw = MsDataset.load('opencompass/mbpp', subset_name='full', split='train')
    samples = []
    for row in raw:
        test_list = list(row['test_list'] or [])
        if not test_list:
            continue
        # The first assertion is handed to the model as a signature hint: MBPP
        # descriptions alone do not pin the function name, and the hidden tests
        # call it by name.
        prompt = (f"{row['text']}\n\n"
                  f'Your function must satisfy this call:\n{test_list[0]}')
        samples.append({
            'prompt': prompt,
            'test_list': test_list,
            'test_setup_code': row.get('test_setup_code') or '',
        })
    logger.info(f'MBPP loaded: {len(samples)} samples')
    return samples


# ========== Environment Setup ==========
def make_env() -> OpenEnvClient:
    """One session per trajectory; only the two task tools are exposed."""
    env = OpenEnvClient(
        env_name=OPENENV_ENV_NAME,
        base_url=OPENENV_BASE_URL,
        tools=[TOOL_SCHEMA[0]],  # run_python; submit_solution is registered below
        # Code execution can be slow; keep the per-message timeout generous.
        message_timeout_s=120.0,
    )
    return register_tools(env)


def prepare_trajectories(
    samples: List[Dict[str, Any]],
    pool: ThreadPoolExecutor,
) -> Tuple[List[Dict[str, Any]], List[ToolManager], List[OpenEnvClient]]:
    """Open one session per trajectory (in parallel) and build trajectories."""
    envs = [make_env() for _ in samples]
    # reset() blocks on the WebSocket handshake plus server-side env creation;
    # run them concurrently so a batch of 32 does not serialize.
    list(pool.map(lambda env: env.reset(), envs))

    trajectories = []
    tool_managers = []
    for sample, env in zip(samples, envs):
        tool_managers.append(ToolManager(EnvTool.from_env(env)))
        trajectories.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': sample['prompt']},
            ],
            'tools': TOOL_SCHEMA,
        })
    return trajectories, tool_managers, envs


def close_envs(envs: List[OpenEnvClient], pool: ThreadPoolExecutor) -> None:
    """Close all sessions so their server-side slots are freed immediately."""
    list(pool.map(lambda env: env.close(), envs))


def extract_rewards(
    envs: List[OpenEnvClient],
    samples: List[Dict[str, Any]],
    pool: ThreadPoolExecutor,
) -> Tuple[List[float], List[float]]:
    """Score submitted solutions against the hidden MBPP tests.

    Shaped so that a full pass dominates, while a submission that passes some
    tests still beats one that passes none — which beats never submitting.

    Returns:
        ``(rewards, pass_rates)``.
    """

    def score(pair) -> Tuple[float, float]:
        env, sample = pair
        passed, total = run_tests(env, sample['test_list'], sample['test_setup_code'])
        rate = passed / total if total else 0.0
        if total and passed == total:
            return 1.0, rate
        if getattr(env, 'submitted_code', None):
            return 0.1 + 0.4 * rate, rate
        return 0.0, rate

    scored = list(pool.map(score, zip(envs, samples)))
    return [s[0] for s in scored], [s[1] for s in scored]


# ========== Main ==========
def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )

    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
    )
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
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

    rollout_template = Qwen3_5Template(MODEL_ID, max_length=8192, enable_thinking=False)
    rollout_template.truncation_strategy = 'delete'

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

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
    dataset = load_mbpp()
    env_pool = ThreadPoolExecutor(max_workers=ENV_CONCURRENCY)

    optim_step = 0
    sample_cursor = 0
    logger.info(f'Starting OpenEnv (server mode) GRPO: base_url={OPENENV_BASE_URL}, env={OPENENV_ENV_NAME}')
    logger.info(f'Concurrent sessions needed: {BATCH_SIZE * NUM_GENERATIONS} '
                f'(server capacity = workers x MAX_CONCURRENT_ENVS)')
    logger.info(get_device_placement())

    while optim_step < MAX_STEPS:
        metrics.reset()

        # BATCH_SIZE problems x NUM_GENERATIONS contiguous copies (GRPO groups).
        batch = [dataset[(sample_cursor + i) % len(dataset)] for i in range(BATCH_SIZE)]
        sample_cursor += BATCH_SIZE
        expanded = [s for s in batch for _ in range(NUM_GENERATIONS)]
        n_traj = len(expanded)

        # 1. Open sessions and build initial trajectories
        logger.info(f'[Step {optim_step}] Opening {n_traj} sessions...')
        expand_prompts, tool_managers, envs = prepare_trajectories(expanded, env_pool)

        try:
            # 2. Sync model weights to sampler
            ckpt_manager.sync_weights(merge_and_sync=False)
            sampler.reset_prefix_cache()

            # 3. Multi-turn rollout with per-trajectory ToolManagers
            all_trajectories: List[Dict[str, Any]] = rollout(
                expand_prompts,
                tool_manager=tool_managers,
            )

            # 4. Score solutions by replaying the hidden tests in each session
            total_rewards, pass_rates = extract_rewards(envs, expanded, env_pool)
        finally:
            # Sessions occupy server capacity; always release them.
            close_envs(envs, env_pool)

        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []
        n_turns_per_rollout: List[int] = []
        for traj in all_trajectories:
            logprobs = traj.get('logprobs') or []
            all_old_logps.append([lp[0][1] for lp in logprobs] if logprobs else [])
            labels = traj.get('labels') or []
            all_completion_lengths.append(sum(1 for l in labels if l != -100))
            n_turns_per_rollout.append(int(traj.get('turns') or 0))

        # 5. Group-relative advantages
        advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group',
        ).tolist()

        # 6. Log metrics
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={'total': total_rewards},
        )
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        solve_rate = sum(1 for r in total_rewards if r >= 1.0) / max(len(total_rewards), 1)
        avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
        avg_turns = sum(n_turns_per_rollout) / len(n_turns_per_rollout) if n_turns_per_rollout else 0.0
        logger.info(f'[Step {optim_step}] avg_reward={avg_reward:.3f}, solve_rate={solve_rate:.3f}, '
                    f'test_pass_rate={avg_pass_rate:.3f}, avg_turns={avg_turns:.1f}')

        # 7. Filter and train (same recipe as the AgentENV example)
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

        total_completions = len(all_input_data)
        logger.info(f'[Step {optim_step}] {total_completions}/{n_traj} trajectories '
                    f'passed length filter (max_len={max_len})')

        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            model.forward_backward(
                inputs=all_input_data[mb_start:mb_end],
                old_logps=filtered_old_logps[mb_start:mb_end],
                advantages=filtered_advantages[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'openenv-code-grpo-checkpoint-{optim_step}')

        # 8. Step summary
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict['avg_turns'] = avg_turns
        log_dict['avg_reward'] = avg_reward
        log_dict['solve_rate'] = solve_rate
        log_dict['test_pass_rate'] = avg_pass_rate
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    env_pool.shutdown(wait=False)
    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('openenv-code-grpo-final')


if __name__ == '__main__':
    main()
