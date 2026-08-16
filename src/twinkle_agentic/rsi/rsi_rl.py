# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI step 3 — multi-LoRA GRPO where each round of a standard flow becomes its
own training query.

Idea (confirmed): a multi-turn standard flow is decomposed into one training
query PER key round. For a tool round i the model is shown the fixed prior key
nodes (their tool calls + results) and must roll out {reasoning + the tool call}
for round i. The reward is whether the GENERATED tool call matches the recorded
standard call — name exact + every standard-call argument key/value present in
the generated call (extra args / order ignored). No sandbox is needed because
the standard call is the reference answer. Only the reasoning ("思路") varies
across rollouts; the key node is the target.

v1 scope: only TOOL rounds are trained. Code rounds (no tool call) still appear
in the prior context but are not turned into training queries yet (their reward
is rubric-based, wired in a later iteration).

RL data-flow discipline (verified): train ONLY on ``sequence.new_input_feature``
and use ``sequence.logprobs`` as old_logps — never decode-then-re-encode. The
generated tool call is already parsed into ``new_input_feature['messages'][-1]``
by the template, and the reference call rides along in ``user_data``.

Structure mirrors cookbook/rl/grpo/short_math_grpo_multi_lora.py (MultiLoRA
Megatron + filesystem LoRA sync to vLLM). RSI-specific paths come from env vars
so the standard CLI (model/infra/rl knobs) stays identical to the reference:

    RSI_STD_FLOWS   standard_flows.jsonl from rsi_refine.py (default output/rsi/standard_flows.jsonl)
    RSI_TEMPLATE    template name, must match the model (default Qwen3_5Template)
    RSI_LORA_SYNC   dir for filesystem LoRA sync (default output/rsi/lora_sync)
    RSI_ADAPTER     executor adapter name (default executor)
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import MultiLoraMegatronModel
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

logger = get_logger()
args = CLI.from_args()

# ── RSI-specific paths (env) ───────────────────────────────────────────────
STD_FLOWS = os.environ.get('RSI_STD_FLOWS', 'output/rsi/standard_flows.jsonl')
TEMPLATE = os.environ.get('RSI_TEMPLATE', 'Qwen3_5Template')
LORA_SYNC_DIR = os.environ.get('RSI_LORA_SYNC', 'output/rsi/lora_sync')
ADAPTER_NAME = os.environ.get('RSI_ADAPTER', 'executor')
REWARD_TOOL_RESULT = 'tool_result'  # matches rsi_refine.attach_reward_method

# ── standard CLI knobs (same shape as the reference script) ────────────────
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.6-35B-A3B'
MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 2
SAMPLER_TP = args.sampler.tensor_parallel_size or 2
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS
NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 4096
LEARNING_RATE = args.optimizer.learning_rate or 5e-5
MAX_STEPS = args.training.max_steps or 1000
BATCH_SIZE = args.training.batch_size or 4
MINI_BATCH_SIZE = args.training.mini_batch_size or 4
MICRO_BATCH_SIZE = args.training.micro_batch_size or 1
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
SAVE_STEPS = args.training.save_steps or 1000
LORA_RANK = args.lora.lora_r or 16

import swanlab
swanlab.init(project='twinkle-rsi')


# ── tool-call matching (name exact + standard-call arg subset) ─────────────
def _as_args(a: Any) -> Dict[str, Any]:
    if isinstance(a, str):
        try:
            return json.loads(a)
        except (ValueError, TypeError):
            return {}
    return a or {}


def tool_call_matches(gen_call: Optional[Dict[str, Any]], ref_call: Dict[str, Any]) -> bool:
    """True iff name matches and every reference arg (key+value) is present."""
    if not gen_call or gen_call.get('name') != ref_call.get('name'):
        return False
    gen_args = _as_args(gen_call.get('arguments'))
    ref_args = _as_args(ref_call.get('arguments'))
    for k, v in ref_args.items():
        if k not in gen_args or gen_args[k] != v:
            return False
    return True


class ToolMatchReward(Reward):
    """1.0 when the generated tool call matches the recorded standard call."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            gen_call = None
            for m in reversed(traj.get('messages', []) or []):
                if m.get('role') == 'assistant':
                    tcs = m.get('tool_calls') or []
                    if tcs:
                        gen_call = tcs[0].get('function')
                    break
            ref_call = None
            for item in (traj.get('user_data') or []):
                if item[0] == 'ref_tool_call':
                    try:
                        ref_call = json.loads(item[1])
                    except (ValueError, TypeError):
                        ref_call = None
                    break
            rewards.append(1.0 if (ref_call and tool_call_matches(gen_call, ref_call)) else 0.0)
        return rewards


# ── decompose standard flows into per-round training trajectories ──────────
def _openai_tool_call(call: Dict[str, Any], idx: int) -> Dict[str, Any]:
    args_ = call.get('arguments', {})
    return {
        'id': f'call_{idx}',
        'type': 'function',
        'function': {
            'name': call.get('name', ''),
            'arguments': json.dumps(args_, ensure_ascii=False) if isinstance(args_, dict) else str(args_),
        },
    }


def _render_prior_round(r: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    """Render a completed prior round as fixed context messages."""
    result = r.get('result', '')
    if r.get('tool_call'):
        tc = _openai_tool_call(r['tool_call'], idx)
        return [
            {'role': 'assistant', 'content': '', 'tool_calls': [tc]},
            {'role': 'tool', 'content': str(result), 'tool_call_id': tc['id']},
        ]
    # Code round: no tool_call_id exists, so keep it template-agnostic.
    return [
        {'role': 'assistant', 'content': r.get('code', '') or ''},
        {'role': 'user', 'content': f'[execution result]\n{result}'},
    ]


def build_round_trajectories(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One training trajectory per TOOL round; prior rounds become fixed context."""
    trajs: List[Dict[str, Any]] = []
    for rec in records:
        prefix: List[Dict[str, Any]] = []
        if rec.get('system'):
            prefix.append(rec['system'])
        if rec.get('query'):
            prefix.append(rec['query'])
        tools = rec.get('tools') or []
        rounds = rec.get('rounds') or []
        for i, r in enumerate(rounds):
            if r.get('reward_method') != REWARD_TOOL_RESULT or not r.get('tool_call'):
                continue  # v1: train tool rounds only
            messages = list(prefix)
            for j in range(i):
                messages.extend(_render_prior_round(rounds[j], j))
            trajs.append({
                'messages': messages,
                'tools': tools,
                'user_data': [('ref_tool_call', json.dumps(r['tool_call'], ensure_ascii=False))],
            })
    return trajs


def create_rsi_dataset():
    records = Dataset(DatasetMeta(dataset_id=STD_FLOWS)).dataset.to_list()
    trajs = build_round_trajectories(records)
    logger.info(f'[rsi_rl] {len(records)} standard flows -> {len(trajs)} per-round tool queries')
    dataset = Dataset(DatasetMeta(data=trajs))
    # enable_thinking=True: we train the reasoning that precedes the tool call.
    dataset.set_template(TEMPLATE, model_id=MODEL_ID, max_length=8192,
                         truncation_strategy='delete', enable_thinking=True)
    dataset.encode(add_generation_prompt=True)
    return dataset


def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU',
                    gpus_per_worker=SAMPLER_TP),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, tp_size=2, ep_size=2, pp_size=2, sequence_parallel=True)
    sampler_dp_size = SAMPLER_GPUS // SAMPLER_TP
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=sampler_dp_size, tp_size=SAMPLER_TP)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(target_modules='all-linear', r=LORA_RANK, lora_alpha=LORA_RANK * 2, lora_dropout=0.05)

    model = MultiLoraMegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
                                   mixed_precision='bf16')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer('default', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    model.set_loss('GRPOLoss', epsilon=0.2, adapter_name=ADAPTER_NAME)
    model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    model.set_template(TEMPLATE, model_id=MODEL_ID, enable_thinking=True, adapter_name=ADAPTER_NAME)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'tensor_parallel_size': SAMPLER_TP,
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': LORA_RANK,
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE, model_id=MODEL_ID, enable_thinking=True)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(dataset=create_rsi_dataset, batch_size=GLOBAL_BATCH_SIZE,
                            min_batch_size=GLOBAL_BATCH_SIZE, device_mesh=model_mesh, remote_group='model')

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    reward_fn = ToolMatchReward()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)

    optim_step = 0
    logger.info('Starting RSI per-round GRPO (MultiLoraMegatron, filesystem LoRA sync)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        lora_sync_path = model.save(f'lora-sync-step-{optim_step}', output_dir=LORA_SYNC_DIR, adapter_name=ADAPTER_NAME)
        sampler.reset_prefix_cache()
        sample_responses = sampler.sample(expand_prompts, sampling_params, adapter_path=lora_sync_path)

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []
        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        rewards = reward_fn(all_input_data)
        metrics.accumulate(completion_lengths=all_completion_lengths, rewards={'tool_match': rewards})
        advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        total = len(all_input_data)
        for mb_start in range(0, total, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total)
            model.forward_backward(
                inputs=all_input_data[mb_start:mb_end],
                old_logps=all_old_logps[mb_start:mb_end],
                advantages=advantages[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE,
                adapter_name=ADAPTER_NAME,
            )
            model.clip_grad_and_step(adapter_name=ADAPTER_NAME)
            optim_step += 1
            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'rsi-executor-checkpoint-{optim_step}', adapter_name=ADAPTER_NAME)

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME))
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('rsi-executor-final', adapter_name=ADAPTER_NAME)


if __name__ == '__main__':
    main()
