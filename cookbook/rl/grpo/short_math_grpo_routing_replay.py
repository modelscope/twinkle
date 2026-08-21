"""GRPO training with routing replay for MoE models (Only FSDP backend is supported currently).

Supports three routing replay modes for MoE expert routing consistency:
- disabled: No routing replay (default behavior)
- R2: Record routing during a forward_only RECORD pass, then replay during training
- R3: vLLM returns routed_experts data (requires vLLM >= 0.14.0), training replays directly
"""
import copy
import re
from typing import List, Tuple, Dict, Any

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import GRPOMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.model.transformers.moe.router_replay import RouterReplayAction

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.6-35B-A3B'
USE_MEGATRON = args.model.strategy != 'native_fsdp'
ROUTER_REPLAY_MODE = args.rl.router_replay_mode or 'R3'

MODEL_GPUS = args.infra.model_gpus or 4
MODEL_FSDP = args.infra.fsdp_size or 4
MODEL_DP = args.infra.dp_size or 1
MODEL_EP = args.infra.ep_size or 4
MODEL_TP = args.infra.tp_size or 1
MODEL_PP = args.infra.pp_size or 1

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
ADAPTER_NAME = args.lora.adapter_name or 'default'
SAVE_STEPS = args.training.save_steps or 1000
LORA_RANK = args.lora.lora_r or 16

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')

# Validate configuration
if ROUTER_REPLAY_MODE not in ('disabled', 'R2', 'R3'):
    raise ValueError(f'Invalid ROUTER_REPLAY_MODE: {ROUTER_REPLAY_MODE}. '
                     f"Must be one of 'disabled', 'R2', 'R3'")
if ROUTER_REPLAY_MODE != 'disabled' and USE_MEGATRON:
    raise ValueError('Routing replay requires USE_MEGATRON=0 (Only FSDP backend is supported currently)')
if ROUTER_REPLAY_MODE == 'R3':
    logger.info('R3 mode: vLLM will return routed_experts data (requires vLLM >= 0.14.0)')
elif ROUTER_REPLAY_MODE == 'R2':
    logger.info('R2 mode: Recording routing during forward_only, replaying during training')

# ========== Reward Functions ==========
class GSM8KBrevityReward(Reward):
    """Brevity reward: rewards shorter completions that contain a valid answer.

    Returns 0.0 if no valid answer format (\\boxed{} or ####).
    Otherwise returns higher score for shorter completions (1.0 at <=200 chars).
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
            )

            if not has_answer:
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= 200:
                    rewards.append(1.0)
                else:
                    rewards.append(max(0.0, 1.0 - (length - 200) / 3000))
        return rewards


# ========== Dataset ==========
def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=4096, truncation_strategy='delete', enable_thinking=False)
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_reward_fn = GSM8KAccuracyReward()
    brevity_reward_fn = GSM8KBrevityReward()

    accuracy_rewards = accuracy_reward_fn(trajectories)
    brevity_rewards = brevity_reward_fn(trajectories)
    total_rewards = [a + b for a, b in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, brevity_rewards, accuracy_rewards


# ========== Main ==========
def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU', gpus_per_worker=SAMPLER_TP),
    ]
    if USE_MEGATRON:
        dp_size = MODEL_GPUS //  (MODEL_TP * MODEL_PP)
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=dp_size, tp_size=MODEL_TP, pp_size=MODEL_PP, ep_size=MODEL_EP, sequence_parallel=True)
    else:
        model_mesh = DeviceMesh.from_sizes(fsdp_size=MODEL_FSDP, dp_size=MODEL_DP, ep_size=MODEL_EP)
    sampler_dp_size = SAMPLER_GPUS //  (SAMPLER_TP)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=sampler_dp_size, tp_size=SAMPLER_TP)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    enable_ep = MODEL_EP > 1
    if enable_ep and not USE_MEGATRON:
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            target_modules='all-linear',
            target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
        )
    else:
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            target_modules='all-linear',
        )

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            enable_router_replay=(ROUTER_REPLAY_MODE != 'disabled'),
            strategy='native_fsdp',
            fsdp_config={
                'expert_parallel': {
                    'enabled': enable_ep,
                    'router_dtype': 'fp32',
                    'keep_router_logits': False,
                }
            },
        )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    # Configure sampler: R3 mode enables routed_experts return from vLLM
    engine_args = {
        'tensor_parallel_size': SAMPLER_TP,
        'gpu_memory_utilization': 0.7,
        'max_model_len': 10000,
        'max_lora_rank': LORA_RANK,
        'enable_lora': True,
        'enable_tower_connector_lora': True,
    }
    if ROUTER_REPLAY_MODE == 'R3':
        engine_args['enable_return_routed_experts'] = True

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args=engine_args,
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = GRPOMetric()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)

    optim_step = 0
    logger.info(f'Starting GSM8K GRPO training with router replay mode={ROUTER_REPLAY_MODE}')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        metrics.reset()
        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)
        # enable_lora=True used with ckpt_manager.sync_weights(merge_and_sync=False)
        # meaning only sync lora weights, if merge_and_sync=True,
        # lora will be merged into the base model and sync all weights to vLLM
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        sample_responses = sampler.sample(
            expand_prompts,
            sampling_params,
        )
        if sample_responses and sample_responses[0].sequences:
            first_decoded = sample_responses[0].sequences[0].decoded
            if isinstance(first_decoded, str):
                logger.info('[sample_debug] first_generation=%r', first_decoded[:512])

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        total_rewards, _, _ = compute_rewards(all_input_data)

        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        total_completions = len(all_input_data)

        # compute old logps and routed_experts(R2)
        # R2: forward_only RECORD pass → get routing data → inject into inputs
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            recompute_output = model.forward_only(
                    inputs=copy.deepcopy(mb_inputs),
                    router_replay_action={'R2': RouterReplayAction.RECORD,'R3': RouterReplayAction.REPLAY_FORWARD}.get(ROUTER_REPLAY_MODE),
                    micro_batch_size=MICRO_BATCH_SIZE,
                )
            old_logps = recompute_output.get('logps')
            assert old_logps.shape[0] == len(mb_inputs)
            for i, mb in enumerate(mb_inputs):
                mb['old_logps'] = old_logps[i]
            routed_experts = recompute_output.get('routed_experts')
            if routed_experts is not None:
                assert routed_experts.shape[0] == len(mb_inputs)
                for i, mb in enumerate(mb_inputs):
                    mb['routed_experts'] = routed_experts[i]

        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]
            for input in mb_inputs:
                input.pop('old_logps', None)

            mb_output = model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
                router_replay_action=RouterReplayAction.REPLAY_FORWARD
            )
            model.clip_grad_and_step()
            optim_step += 1

            logps = mb_output.get('logps')
            mb_output['logps'] = [logps[i:i+1] for i in range(logps.size(0))]
            metrics.accumulate(
                mb_inputs,
                mb_output,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
            )
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'math-grpo-checkpoint-{optim_step}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-gsm8k-checkpoint')


if __name__ == '__main__':
    main()
