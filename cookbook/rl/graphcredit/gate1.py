"""Matched-budget Gate 1 training for GRPO versus StateGraphAdvantage.

The task is a deterministic two-step grid world. A wrong move ends the episode
with reward 0, while reaching the goal returns 1. GRPO therefore only sees the
final episode result. StateGraphAdvantage can additionally distinguish a useful
intermediate move from a wrong move even when the episode later fails.

This is a mechanism study, not a production training entry point. It uses a
standalone Transformers + PEFT loop so both arms share the same rollout,
policy-gradient update and token budget.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from twinkle.advantage import GRPOAdvantage, StateGraphAdvantage


ACTIONS = ('up', 'down', 'left', 'right')
MOVES = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}
ACTION_PATTERN = re.compile(r'<action>\s*(up|down|left|right)\s*</action>', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--mode', choices=('grpo', 'graphcredit'), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--eval-tasks', type=int, default=24)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--distance', type=int, default=2)
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--group-retries', type=int, default=12)
    parser.add_argument('--max-step-attempts', type=int, default=6)
    parser.add_argument('--beta-episode', type=float, default=1.0)
    parser.add_argument('--beta-graph', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--min-state-support', type=int, default=2)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, 'npu'):
        torch.npu.manual_seed_all(seed)


def ensure_npu(device_name: str) -> None:
    if device_name.startswith('npu'):
        import torch_npu  # noqa: F401


def reset_peak_hbm() -> None:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats()
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_hbm_gb() -> float:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return float(torch.npu.max_memory_allocated() / 2**30)
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 2**30)
    return 0.0


def synchronize(device: torch.device) -> None:
    if device.type == 'npu':
        torch.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


@dataclass(frozen=True)
class GridTask:
    task_uid: int
    grid_size: int
    distance: int
    start: tuple[int, int]
    goal: tuple[int, int]

    @property
    def task_key(self) -> str:
        return f'task-{self.task_uid}'

    def state_key(self, position: tuple[int, int]) -> str:
        return f'{self.task_key}:{position[0]}:{position[1]}'

    @property
    def dead_state_key(self) -> str:
        return f'{self.task_key}:dead'


def make_task(task_uid: int, *, grid_size: int, distance: int) -> GridTask:
    if distance < 1:
        raise ValueError('distance must be positive')
    if grid_size <= distance:
        raise ValueError('grid_size must be larger than distance')
    rng = random.Random(task_uid)
    for _ in range(10_000):
        start = (rng.randrange(grid_size), rng.randrange(grid_size))
        axis = rng.choice(('row', 'col'))
        sign = rng.choice((-1, 1))
        goal = (
            start[0] + sign * distance if axis == 'row' else start[0],
            start[1] + sign * distance if axis == 'col' else start[1],
        )
        if 0 <= goal[0] < grid_size and 0 <= goal[1] < grid_size:
            return GridTask(task_uid, grid_size, distance, start, goal)
    raise RuntimeError(f'could not generate task {task_uid}')


def format_action(action: str) -> str:
    return f'<action>{action}</action>'


def parse_action(text: str) -> str | None:
    matches = ACTION_PATTERN.findall(text)
    if len(matches) == 1:
        return matches[0].lower()
    stripped = text.strip().lower()
    return stripped if stripped in ACTIONS else None


def render_prompt(task: GridTask, position: tuple[int, int], history: Sequence[str]) -> str:
    row, col = position
    goal_row, goal_col = task.goal
    distance = abs(goal_row - row) + abs(goal_col - col)
    history_text = ' -> '.join(history) if history else '<empty>'
    return (
        'You control a robot on a rectangular grid. Do not explain or reason aloud. '
        'Reply with exactly one action tag: <action>up</action>, <action>down</action>, '
        '<action>left</action>, or <action>right</action>.\n'
        'Coordinates are zero-based (row, column). Up decreases row, down increases row, '
        'left decreases column and right increases column. A move that does not reduce '
        'the Manhattan distance to the goal immediately ends the episode.\n'
        f'Grid size: {task.grid_size}x{task.grid_size}.\n'
        f'Current position: ({row}, {col}).\n'
        f'Goal position: ({goal_row}, {goal_col}).\n'
        f'Remaining Manhattan distance: {distance}.\n'
        f'Action history: {history_text}.\n'
        'Choose the next action.'
    )


def render_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': 'You are a precise grid-navigation policy.'
        },
        {
            'role': 'user',
            'content': prompt
        },
    ]


@dataclass
class EpisodeTurn:
    prompt: str
    action: str
    pre_state_key: str
    post_state_key: str
    action_token_start: int
    action_token_end: int
    terminal_success: bool | None


@dataclass
class Episode:
    task: GridTask
    uid: str
    position: tuple[int, int]
    done: bool = False
    success: bool = False
    reward: float = 0.0
    invalid_action: bool = False
    input_ids: list[int] = field(default_factory=list)
    turns: list[EpisodeTurn] = field(default_factory=list)

    @property
    def history(self) -> list[str]:
        return [turn.action for turn in self.turns]

    def to_trajectory(self, policy_version: int) -> dict[str, Any]:
        records = [
            {
                'turn_id': index,
                'action_token_start': turn.action_token_start,
                'action_token_end': turn.action_token_end,
                'observation_token_start': None,
                'observation_token_end': None,
                'pre_state_key': turn.pre_state_key,
                'post_state_key': turn.post_state_key,
                'tool_events': [],
                'policy_version': policy_version,
                'terminal_success': turn.terminal_success,
            }
            for index, turn in enumerate(self.turns)
        ]
        return {
            'input_ids': list(self.input_ids),
            'turn_credits': records,
            'turn_credit_status': {
                'available': bool(records) and not self.invalid_action,
                'fallback_reason': None if records else 'turn_credit_empty',
            },
            'episode_reward': self.reward,
            'terminal_success': self.success,
        }


def step_episode(episode: Episode, action: str) -> None:
    if episode.done:
        raise RuntimeError('cannot step a finished episode')
    if action not in MOVES:
        raise ValueError(f'unsupported action: {action}')

    row, col = episode.position
    delta_row, delta_col = MOVES[action]
    next_position = (row + delta_row, col + delta_col)
    current_distance = abs(row - episode.task.goal[0]) + abs(col - episode.task.goal[1])
    next_distance = (
        abs(next_position[0] - episode.task.goal[0])
        + abs(next_position[1] - episode.task.goal[1])
    )
    out_of_bounds = not (
        0 <= next_position[0] < episode.task.grid_size
        and 0 <= next_position[1] < episode.task.grid_size
    )
    if out_of_bounds or next_distance >= current_distance:
        episode.done = True
        episode.success = False
        episode.reward = 0.0
        episode.position = next_position
        return

    episode.position = next_position
    if next_position == episode.task.goal:
        episode.done = True
        episode.success = True
        episode.reward = 1.0


def _append_prompt_action(
    episode: Episode,
    prompt_ids: Sequence[int],
    action_ids: Sequence[int],
    *,
    eos_token_id: int | None,
) -> tuple[int, int]:
    before = len(episode.input_ids)
    episode.input_ids.extend(int(token) for token in prompt_ids)
    action_start = before + len(prompt_ids) - 1
    episode.input_ids.extend(int(token) for token in action_ids)
    action_end = action_start + len(action_ids)
    if eos_token_id is not None:
        episode.input_ids.append(int(eos_token_id))
    return action_start, action_end


def rollout_group(
    policy: Any,
    tokenizer: Any,
    task: GridTask,
    *,
    group_size: int,
    max_new_tokens: int,
    max_length: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    sample: bool,
) -> list[Episode]:
    episodes = [
        Episode(task=task, uid=f'{task.task_uid}:{index}', position=task.start)
        for index in range(group_size)
    ]
    for _turn in range(1, task.distance + 1):
        active = [episode for episode in episodes if not episode.done]
        if not active:
            break
        prompts = [render_prompt(task, episode.position, episode.history) for episode in active]
        outputs = generate_actions(
            policy,
            tokenizer,
            prompts,
            device,
            sample=sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for episode, prompt, raw_output in zip(active, prompts, outputs):
            action = parse_action(raw_output)
            if action is None:
                episode.invalid_action = True
                episode.done = True
                episode.reward = 0.0
                continue
            prompt_text = render_chat_prompt(tokenizer, prompt)
            prompt_ids = encode_text(tokenizer, prompt_text)
            action_ids = encode_text(tokenizer, format_action(action))
            if len(episode.input_ids) + len(prompt_ids) + len(action_ids) > max_length:
                episode.invalid_action = True
                episode.done = True
                episode.reward = 0.0
                continue
            pre_state_key = task.state_key(episode.position)
            action_start, action_end = _append_prompt_action(
                episode,
                prompt_ids,
                action_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
            step_episode(episode, action)
            post_state_key = (
                task.state_key(episode.position) if episode.success or not episode.done else task.dead_state_key
            )
            episode.turns.append(
                EpisodeTurn(
                    prompt=prompt,
                    action=action,
                    pre_state_key=pre_state_key,
                    post_state_key=post_state_key,
                    action_token_start=action_start,
                    action_token_end=action_end,
                    terminal_success=True if episode.success else (False if episode.done else None),
                ))
    for episode in episodes:
        if not episode.done:
            episode.done = True
            episode.reward = 0.0
    return episodes


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        render_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])


def generate_actions(
    policy: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
    *,
    sample: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    rendered = [render_chat_prompt(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(
        rendered,
        padding=True,
        return_tensors='pt',
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    policy.eval()
    policy.config.use_cache = True
    kwargs: dict[str, Any] = {
        'max_new_tokens': max_new_tokens,
        'do_sample': sample,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
    }
    if sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        generated = policy.generate(**encoded, **kwargs)
    width = encoded['input_ids'].shape[1]
    return tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)


def compute_advantages(
    mode: str,
    episodes: Sequence[Episode],
    *,
    policy_version: int,
    group_size: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rewards = [episode.reward for episode in episodes]
    trajectories = [episode.to_trajectory(policy_version) for episode in episodes]
    max_length = max((len(episode.input_ids) for episode in episodes), default=0)
    if mode == 'grpo':
        episode_advantage = GRPOAdvantage()(rewards, num_generations=group_size, scale='group')
        advantages = episode_advantage.unsqueeze(1).repeat(1, max_length)
        diagnostics = {'mode': 'grpo'}
        return advantages, diagnostics

    graph = StateGraphAdvantage(
        beta_episode=args.beta_episode,
        beta_graph=args.beta_graph,
        gamma=args.gamma,
        min_state_support=args.min_state_support,
    )
    advantages = graph(rewards, trajectories, num_generations=group_size)
    diagnostics = {
        'mode': 'graphcredit',
        'transition_count': graph.last_diagnostics.transition_count,
        'graph_node_count': graph.last_diagnostics.graph_node_count,
        'graph_edge_count': graph.last_diagnostics.graph_edge_count,
        'comparable_state_group_count': graph.last_diagnostics.comparable_state_group_count,
        'fallback_counts': graph.last_diagnostics.fallback_counts,
    }
    return advantages, diagnostics


def pad_episode_batch(
    episodes: Sequence[Episode],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(episode.input_ids) for episode in episodes)
    rows = []
    masks = []
    for episode in episodes:
        padding = max_length - len(episode.input_ids)
        rows.append(episode.input_ids + [pad_token_id] * padding)
        masks.append([1] * len(episode.input_ids) + [0] * padding)
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def policy_gradient_update(
    policy: Any,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[Episode],
    advantages: torch.Tensor,
    *,
    device: torch.device,
    pad_token_id: int,
    max_grad_norm: float,
) -> tuple[float, int]:
    policy.train()
    policy.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    input_ids, attention_mask = pad_episode_batch(episodes, pad_token_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    advantages = advantages.to(device=device, dtype=token_logps.dtype)

    loss = input_ids.new_zeros((), dtype=torch.float32)
    action_tokens = 0
    for row, episode in enumerate(episodes):
        for turn in episode.turns:
            start = turn.action_token_start
            end = turn.action_token_end
            if start < 0 or end <= start or end > token_logps.shape[1]:
                raise RuntimeError(
                    f'invalid action span for {episode.uid}: [{start}, {end}) '
                    f'with sequence width {token_logps.shape[1]}')
            loss = loss - (
                token_logps[row, start:end] * advantages[row, start:end]
            ).sum()
            action_tokens += end - start
    if action_tokens == 0:
        raise RuntimeError('episode group contains no trainable action tokens')
    loss = loss / action_tokens
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        max_grad_norm,
    )
    optimizer.step()
    return float(loss.detach().cpu()), action_tokens


def evaluate(
    policy: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    *,
    split_offset: int,
) -> dict[str, float]:
    successes = 0
    invalid = 0
    turns = 0
    rewards = []
    started = time.perf_counter()
    for index in range(args.eval_tasks):
        task = make_task(
            split_offset + index,
            grid_size=args.grid_size,
            distance=args.distance,
        )
        episode = rollout_group(
            policy,
            tokenizer,
            task,
            group_size=1,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            sample=False,
        )[0]
        successes += int(episode.success)
        invalid += int(episode.invalid_action)
        turns += len(episode.turns)
        rewards.append(episode.reward)
    total = max(1, len(rewards))
    return {
        'success_rate': successes / total,
        'invalid_rate': invalid / total,
        'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
        'mean_turns': turns / total,
        'eval_seconds': time.perf_counter() - started,
    }


def sample_training_group(
    policy: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    *,
    step: int,
) -> tuple[GridTask, list[Episode]] | None:
    for attempt in range(args.group_retries):
        task_uid = args.seed + 10_000 + step * 100 + attempt
        task = make_task(task_uid, grid_size=args.grid_size, distance=args.distance)
        episodes = rollout_group(
            policy,
            tokenizer,
            task,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            sample=True,
        )
        if any(episode.invalid_action for episode in episodes):
            continue
        rewards = [episode.reward for episode in episodes]
        if not any(rewards) or all(rewards):
            continue
        return task, episodes
    return None


def load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def load_policy(model_path: str, args: argparse.Namespace, device: torch.device) -> Any:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    if hasattr(policy, 'gradient_checkpointing_enable'):
        policy.gradient_checkpointing_enable()
    policy = get_peft_model(
        policy,
        LoraConfig(
            task_type='CAUSAL_LM',
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
        ),
    )
    policy.config.use_cache = False
    return policy


def train(policy: Any, tokenizer: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    initial_eval = evaluate(
        policy,
        tokenizer,
        args,
        device,
        split_offset=args.seed + 500_000,
    )
    reset_peak_hbm()
    history = []
    completed_steps = 0
    attempted_steps = 0
    max_attempts = args.steps * args.max_step_attempts
    while completed_steps < args.steps and attempted_steps < max_attempts:
        attempted_steps += 1
        step_started = time.perf_counter()
        sampled = sample_training_group(
            policy,
            tokenizer,
            args,
            device,
            step=attempted_steps - 1,
        )
        if sampled is None:
            history.append({
                'attempt': attempted_steps,
                'completed_steps': completed_steps,
                'skipped': True,
            })
            continue
        task, episodes = sampled
        synchronize(device)
        rollout_seconds = time.perf_counter() - step_started
        rewards = [episode.reward for episode in episodes]
        advantages, diagnostics = compute_advantages(
            args.mode,
            episodes,
            policy_version=completed_steps + 1,
            group_size=args.group_size,
            args=args,
        )
        loss, action_tokens = policy_gradient_update(
            policy,
            optimizer,
            episodes,
            advantages,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=args.max_grad_norm,
        )
        synchronize(device)
        record = {
            'step': completed_steps + 1,
            'attempt': attempted_steps,
            'task_uid': task.task_uid,
            'reward_mean': float(np.mean(rewards)),
            'success_rate': float(np.mean([episode.success for episode in episodes])),
            'mean_actions': float(np.mean([len(episode.turns) for episode in episodes])),
            'loss': loss,
            'action_tokens': action_tokens,
            'rollout_seconds': rollout_seconds,
            'step_seconds': time.perf_counter() - step_started,
            'peak_hbm_gb': peak_hbm_gb(),
            **diagnostics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        completed_steps += 1
        if args.eval_every > 0 and completed_steps % args.eval_every == 0:
            heldout = evaluate(
                policy,
                tokenizer,
                args,
                device,
                split_offset=args.seed + 500_000,
            )
            record['heldout_eval'] = heldout
            print(
                json.dumps(
                    {
                        'step': completed_steps,
                        'heldout_success_rate': heldout['success_rate'],
                        'heldout_invalid_rate': heldout['invalid_rate'],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    final_eval = evaluate(
        policy,
        tokenizer,
        args,
        device,
        split_offset=args.seed + 500_000,
    )
    return {
        'mode': args.mode,
        'history': history,
        'completed_steps': completed_steps,
        'attempted_steps': attempted_steps,
        'skipped_steps': attempted_steps - completed_steps,
        'initial_eval': initial_eval,
        'final_eval': final_eval,
        'peak_hbm_gb': peak_hbm_gb(),
    }


def main() -> None:
    args = parse_args()
    ensure_npu(args.device)
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_path)
    policy = load_policy(args.model_path, args, device)
    started = time.perf_counter()
    result = train(policy, tokenizer, args, device)
    result['wall_seconds'] = time.perf_counter() - started
    result['args'] = vars(args)
    policy.save_pretrained(output_dir / 'adapter')
    (output_dir / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
