"""Agentic RSI: GRPO on multi-turn tool-using episodes, scored by program checks.

The solver is an ms-agent agent with a real tool line-up (shell, filesystem,
python, notebook sandbox, todo list) working in its own microVM. It explores for
as many turns as it needs; when it stops calling tools the episode ends and the
task's checks are run against what it left behind. The checks are ordinary
programs -- file exists, file content, command exit status, final answer match --
so the same trajectory always earns the same reward.

The two halves are split by where they run, not by what they know:

  * On the training host, ``MsAgentHarness`` shapes messages -- system prompt,
    tool-result formatting, ms-agent's own message evolution. Its ``llm`` and
    ``tools`` sections are dropped before it prepares, so it constructs no tool
    and nothing the model emits can execute next to the trainer.
  * In the sandbox, ``sandbox_server/tool_server.py`` holds the real ms-agent
    ``ToolManager``. It also *supplies the tool schemas*, which the prompt then
    advertises verbatim -- the contract the model is trained against is read off
    the code that will honour it, so the two cannot drift apart.

Usage:
    AENV_API_URL=http://127.0.0.1:8000 \\
    RSI_TASKS=cookbook/rsi/agentic/tasks.example.jsonl \\
        python cookbook/rsi/agentic/rl.py

See README.md for building the template and starting the sandbox server.

Task file: one JSON object per line, with ``id``, ``query`` and either
``check_script`` (a python script, from ``challenge.py``) or ``checks``
(structured, see twinkle_agentic.verifier.result_check.Check).
"""
import os
import shutil
from typing import Any, Dict, List

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
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout

# Same directory as this script, which python puts on sys.path when it is run as
# a file. Episode construction and scoring are shared with eval.py so the two
# cannot drift: an eval measuring episodes built differently from training would
# not be measuring the training.
from episode import (SandboxConfig, boot_episodes, load_tasks,  # noqa: I100,I202
                     score_episodes)

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3-4B'

MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 8
# Per turn, and it has to match the challenger and eval side (both 8192): a
# trajectory generated with less room than the tasks were built with is trained on
# truncated attempts. At 4096, replies ran out mid-<think> before dispatching any
# tool -- 3 of 12 episodes on the generation side, 15 of 50 solver attempts.
MAX_NEW_TOKENS = args.sampling.max_tokens or 8192
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

TASKS_PATH = os.environ.get('RSI_TASKS', 'cookbook/rsi/agentic/tasks.example.jsonl')
RUN_DIR = os.environ.get('RSI_RUN_DIR', 'output/rsi_agentic/run')

# Where episodes run, and how they are scored. Shared with eval.py.
SANDBOX = SandboxConfig.from_env()

# Keep each episode's downloaded files after scoring. Useful while debugging
# tasks, expensive over a long run.
KEEP_WORKSPACES = os.environ.get('RSI_KEEP_WORKSPACES', '0') == '1'



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

    # torch_dtype=float32: load master weights in fp32. Training precision is still
    # governed by `mixed_precision` (bf16 autocast); fp32 params avoid the numerically
    # fragile bf16 optimizer state on this Blackwell + CUDA 13 box.
    model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
                              torch_dtype='float32')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config,
                               gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    # Both templates get the trajectory budget explicitly. The default is far
    # smaller than a tool-using episode: leaving it out makes the sampler refuse
    # the trajectory mid-run with `Input length N exceeds max_length 8192`, after
    # the step it happened in has already booted its sandboxes.
    model.set_template('Template', model_id=MODEL_ID, enable_thinking=True,
                       max_length=MAX_TRAJECTORY_TOKENS)

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
    sampler.set_template('Template', model_id=MODEL_ID, enable_thinking=True,
                         max_length=MAX_TRAJECTORY_TOKENS)

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
    logger.info(f'Starting agentic RSI GRPO (max_turns={MAX_TURNS}, '
                f'score={SANDBOX.score_mode})')
    logger.info(f'Sandboxes: template={SANDBOX.template} api={SANDBOX.api_url} '
                f'concurrency={SANDBOX.concurrency}')
    logger.info(get_device_placement())

    while optim_step < MAX_STEPS:
        metrics.reset()

        # Each prompt is repeated NUM_GENERATIONS times; GRPO needs a group of
        # rollouts on the SAME task to have anything to compare against.
        batch_tasks = [tasks[(task_cursor + i) % len(tasks)] for i in range(BATCH_SIZE)]
        task_cursor = (task_cursor + BATCH_SIZE) % len(tasks)
        episode_tasks = [t for t in batch_tasks for _ in range(NUM_GENERATIONS)]

        harnesses, envs, tool_managers, trajectories = [], [], [], []
        try:
            episodes = boot_episodes(episode_tasks, SANDBOX)
        except Exception as e:  # noqa
            # A sandbox that never came up answers every call with an error, so
            # the group would score a uniform zero and look like a hard task
            # rather than a broken environment. Skip the batch and say so.
            logger.warning(f'[Step {optim_step}] {e}; skipping batch')
            continue
        for harness, env, tool_manager, trajectory in episodes:
            harnesses.append(harness)
            envs.append(env)
            tool_managers.append(tool_manager)
            trajectories.append(trajectory)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        try:
            outs: List[Dict[str, Any]] = rollout(
                trajectories, harness=harnesses, tool_manager=tool_managers)

            rewards = score_episodes(episode_tasks, envs, outs,
                                     os.path.join(RUN_DIR, f'step{optim_step:06d}'), SANDBOX)
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

        # One optimizer step per batch, not per mini-batch. ``forward_backward``
        # neither steps nor zeroes, so the mini-batches below simply add their
        # gradients together; ``clip_grad_and_step`` afterwards divides by the
        # token count accumulated across all of them, so every trajectory in the
        # batch carries the same weight regardless of how the mini-batches split.
        # Stepping inside the loop instead -- which is what this used to do --
        # made each step see only MINI_BATCH_SIZE trajectories, so a group of
        # NUM_GENERATIONS could be torn across two updates.
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
