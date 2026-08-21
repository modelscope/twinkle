"""Agentic RSI: GRPO on multi-turn tool-using episodes, scored by program checks.

The solver is an ms-agent agent with a real tool line-up (shell, filesystem,
python, notebook sandbox, web search, todo list) working in its own directory.
It explores for as many turns as it needs; when it stops calling tools the
episode ends and the task's checks are run against what it left behind. The
checks are ordinary programs -- file exists, file content, command exit status,
final answer match -- so the same trajectory always earns the same reward.

Everything framework-specific lives here and in ``rsi_agent.yaml``: the tool
line-up, the sandbox settings, the task file. ``src/twinkle_agentic`` stays
generic -- ``MsAgentHarness`` shapes messages, ``MsAgentToolEnv`` executes
tools, ``result_check`` scores outcomes, and none of them know about RSI.

Layout mirrors cookbook/rl/multi_turn/multi_turn_grpo.py; the differences are
the harness (ms-agent owns the system prompt and message evolution) and the
reward (program checks over the end state instead of an env-emitted scalar).

Usage:
    RSI_TASKS=cookbook/rl/rsi_agentic/tasks.example.jsonl \\
        python cookbook/rl/rsi_agentic/rsi_agentic_grpo.py

Task file: one JSON object per line, with ``id``, ``query`` and ``checks``
(see tasks.example.jsonl and twinkle_agentic.verifier.result_check.Check).
"""
import json
import os
import shutil
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
from twinkle.template import Template
from twinkle_agentic.envs import EnvTool, MsAgentToolEnv
from twinkle_agentic.harness import MsAgentHarness
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.verifier.result_check import (CheckContext, checks_from_dicts,
                                                   run_checks)

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3-4B'

MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 4096
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 1000
BATCH_SIZE = args.training.batch_size or 4
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
ADAPTER_NAME = args.lora.adapter_name or 'default'
SAVE_STEPS = args.training.save_steps or 500
LORA_RANK = args.lora.lora_r or 16

# A tool-using episode needs room for observations on top of its own tokens.
MAX_TRAJECTORY_TOKENS = int(os.environ.get('RSI_MAX_TRAJ_TOKENS', 32768))
MAX_TURNS = int(os.environ.get('RSI_MAX_TURNS', 20))

TASKS_PATH = os.environ.get('RSI_TASKS', 'cookbook/rl/rsi_agentic/tasks.example.jsonl')
AGENT_CONFIG = os.environ.get('RSI_AGENT_CONFIG', 'cookbook/rl/rsi_agentic/rsi_agent.yaml')
RUN_DIR = os.environ.get('RSI_RUN_DIR', 'output/rsi_agentic/run')
# 'fraction' gives partial credit per check; 'all_or_nothing' is stricter and
# produces a cleaner pass/fail signal at the cost of a sparser reward.
SCORE_MODE = os.environ.get('RSI_SCORE_MODE', 'fraction')
# Keep each episode's workspace after scoring. Useful while debugging tasks,
# expensive over a long run.
KEEP_WORKSPACES = os.environ.get('RSI_KEEP_WORKSPACES', '0') == '1'


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """Read the task file and fail loudly on a task that can never be scored."""
    tasks = []
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not task.get('query'):
                raise ValueError(f'{path}:{lineno} has no query')
            if not task.get('checks'):
                # An unchecked task scores 0 for every rollout, so the whole
                # group has zero advantage and contributes no gradient.
                raise ValueError(f'{path}:{lineno} ({task.get("id")}) declares no checks')
            task['_checks'] = checks_from_dicts(task['checks'])
            tasks.append(task)
    if not tasks:
        raise ValueError(f'{path} contains no tasks')
    return tasks


def build_episode(task: Dict[str, Any], slot: int, step: int) -> Tuple[Any, Any, Any, Dict]:
    """Create one episode: harness + isolated workspace + bound tool manager.

    The harness and the Env share one ms-agent runtime, so the tools named in
    the prompt are exactly the tools that will run. Each episode gets its own
    ``output_dir`` -- that directory is both the sandbox root and what the
    checks will later inspect.
    """
    from omegaconf import OmegaConf, open_dict

    workspace = os.path.join(RUN_DIR, f'step{step:06d}', f'slot{slot:03d}')
    os.makedirs(workspace, exist_ok=True)

    cfg = OmegaConf.load(AGENT_CONFIG)
    with open_dict(cfg):
        cfg.output_dir = os.path.abspath(workspace)

    harness = MsAgentHarness(config=cfg)
    # ms-agent merges the config above over its own default agent.yaml, which
    # declares an `llm:` section; FileSystemTool then builds a remote LLM client
    # from it and asserts on a missing api key. Generation here comes from the
    # vLLM sampler, so drop that section before any tool is constructed.
    with open_dict(harness.agent.config):
        harness.agent.config.pop('llm', None)
    harness.prepare()

    env = MsAgentToolEnv(agent=harness.agent, workspace=workspace)
    # Same schema list on both sides: prompt and executor cannot drift apart.
    tool_manager = ToolManager(EnvTool.from_schemas(env, harness.tool_schemas()))

    trajectory = harness.start(task['query'])
    return harness, env, tool_manager, trajectory


def score_episode(task: Dict[str, Any], env: MsAgentToolEnv, trajectory: Dict[str, Any]) -> float:
    """Run the task's checks against the state this episode left behind."""
    final_answer = ''
    for msg in reversed(trajectory.get('messages') or []):
        if msg.get('role') == 'assistant' and (msg.get('content') or '').strip():
            final_answer = msg['content']
            break

    ctx = CheckContext(
        workspace=env.workspace,
        final_answer=final_answer,
        # Route shell/python checks back through the episode's own sandbox so
        # they see the filesystem the agent actually wrote to.
        runner=env.runner(),
    )
    report = run_checks(task['_checks'], ctx, mode=SCORE_MODE)
    if not report.all_passed:
        logger.debug(f'[{task["id"]}] {report.n_passed}/{report.n_total} checks: '
                     f'{report.failures()}')
    return report.score


def main():
    tasks = load_tasks(TASKS_PATH)
    logger.info(f'Loaded {len(tasks)} tasks from {TASKS_PATH}')

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups,
                       lazy_collect=False)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )

    model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config,
                               gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Template', model_id=MODEL_ID, enable_thinking=True)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': MAX_TRAJECTORY_TOKENS,
            'max_lora_rank': 32,
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Template', model_id=MODEL_ID, enable_thinking=True)

    rollout_template = Template(MODEL_ID, max_length=MAX_TRAJECTORY_TOKENS, enable_thinking=True)
    rollout_template.truncation_strategy = 'delete'

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
                                     temperature=1.0, top_p=0.95)
    rollout = MultiTurnRollout(
        sampler=sampler,
        template=rollout_template,
        sampling_params=sampling_params,
        max_turns=MAX_TURNS,
        max_trajectory_tokens=MAX_TRAJECTORY_TOKENS,
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    optim_step = 0
    task_cursor = 0
    logger.info(f'Starting agentic RSI GRPO (max_turns={MAX_TURNS}, score={SCORE_MODE})')
    logger.info(get_device_placement())

    while optim_step < MAX_STEPS:
        metrics.reset()

        # Each prompt is repeated NUM_GENERATIONS times; GRPO needs a group of
        # rollouts on the SAME task to have anything to compare against.
        batch_tasks = [tasks[(task_cursor + i) % len(tasks)] for i in range(BATCH_SIZE)]
        task_cursor = (task_cursor + BATCH_SIZE) % len(tasks)
        episode_tasks = [t for t in batch_tasks for _ in range(NUM_GENERATIONS)]

        harnesses, envs, tool_managers, trajectories = [], [], [], []
        for slot, task in enumerate(episode_tasks):
            h, env, tm, traj = build_episode(task, slot, optim_step)
            harnesses.append(h)
            envs.append(env)
            tool_managers.append(tm)
            trajectories.append(traj)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        try:
            outs: List[Dict[str, Any]] = rollout(
                trajectories, harness=harnesses, tool_manager=tool_managers)

            rewards = [score_episode(task, env, traj)
                       for task, env, traj in zip(episode_tasks, envs, outs)]
        finally:
            # Sandboxes are a finite resource; a step that raises must still
            # give them back or the next step starts short.
            for env in envs:
                env.close()
            if not KEEP_WORKSPACES:
                shutil.rmtree(os.path.join(RUN_DIR, f'step{optim_step:06d}'), ignore_errors=True)

        all_old_logps, completion_lengths, turns = [], [], []
        for traj in outs:
            logprobs = traj.get('logprobs') or []
            all_old_logps.append([lp[0][1] for lp in logprobs] if logprobs else [])
            labels = traj.get('labels') or []
            completion_lengths.append(sum(1 for label in labels if label != -100))
            turns.append(int(traj.get('turns') or 0))

        advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS,
                                  scale='group').tolist()
        metrics.accumulate(completion_lengths=completion_lengths, rewards={'total': rewards})

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        solved = sum(1 for r in rewards if r >= 1.0)
        logger.info(f'[Step {optim_step}] avg_reward={avg_reward:.3f} '
                    f'fully_solved={solved}/{len(rewards)} '
                    f'avg_turns={sum(turns)/max(1,len(turns)):.1f}')

        # Drop episodes the template refused (too long) or that produced no
        # trainable tokens; feeding those in would corrupt the logp alignment.
        inputs, kept_logps, kept_adv = [], [], []
        for i, traj in enumerate(outs):
            if not completion_lengths[i]:
                continue
            if len(traj.get('input_ids') or []) > MAX_TRAJECTORY_TOKENS:
                continue
            inputs.append(traj)
            kept_logps.append(all_old_logps[i])
            kept_adv.append(advantages[i])

        if len(inputs) < MODEL_GPUS:
            logger.warning(f'[Step {optim_step}] only {len(inputs)} usable trajectories '
                           f'(need >= {MODEL_GPUS}); skipping batch')
            continue

        for mb_start in range(0, len(inputs), MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, len(inputs))
            model.forward_backward(
                inputs=inputs[mb_start:mb_end],
                old_logps=kept_logps[mb_start:mb_end],
                advantages=kept_adv[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1
            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'rsi-agentic-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict['avg_reward'] = avg_reward
        log_dict['fully_solved'] = solved
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('rsi-agentic-final')


if __name__ == '__main__':
    main()
