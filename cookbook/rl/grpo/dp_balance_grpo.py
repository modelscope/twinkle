"""GRPO on GSM8K with DP workload balancing (`--balance-batch`).

`forward_backward` dispatches every mini-batch with `slice_dp`: each model DP rank trains on a contiguous
slice, and the step waits for the slowest rank. Thinking completions vary a lot in length, so those
slices are often uneven. `plan_dp_balance` reorders the mini-batch before dispatch so every rank keeps its
sample count but gets a similar workload. It estimates each rank's cost by running the model's own
micro-batch planner on that rank's slice, so it follows both the input layout (padded or packed) and the
micro-batch options (fixed or dynamic). With `GRPOLoss` (a per-sequence mean) the update does not change;
only the waiting shrinks. Training metrics gain `dp_balance/*` entries: the workload imbalance (busiest
rank over the mean), per-rank tokens, micro-batches per rank and, for padded inputs, the padding ratio,
before and after balancing.

Packed inputs (`--padding-free`) of linear-attention models such as Qwen3.5 need flash-linear-attention:
the native GatedDeltaNet does not reset its state between packed sequences. On NPU that means
flash-linear-attention >= 0.5.2, which ships the Ascend Triton backend; the script installs its kernels on
the model workers.

Usage:
    sh dp_balance_grpo.sh                         # padded inputs, dynamic batching, FSDP2
    sh dp_balance_grpo.sh --padding-free          # packed inputs
    sh dp_balance_grpo.sh --no-balance-batch      # the same run without balancing, for comparison
"""
import argparse
from functools import partial
from typing import Any, Dict, List

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.hub import HubOperation
from twinkle.metric import CompletionRewardMetric
from twinkle.model import MicroBatchConfig, TransformersModel, plan_dp_balance
from twinkle.model.dp_balance import estimate_workload
from twinkle.patch import Patch
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.sampler import vLLMSampler

logger = get_logger()

SYSTEM_PROMPT = 'You are a helpful math assistant. Reason step by step and put your final answer within \\boxed{}.'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-id', default='ms://Qwen/Qwen3.5-4B')
    parser.add_argument('--dataset-id', default='ms://modelscope/gsm8k')
    parser.add_argument('--subset-name', default='main')
    parser.add_argument('--strategy', choices=('native_fsdp', 'accelerate'), default='native_fsdp')
    parser.add_argument('--model-gpus', type=int, default=2)
    parser.add_argument('--sampler-gpus', type=int, default=2)
    parser.add_argument('--num-generations', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=2048, help='max new tokens per completion')
    parser.add_argument('--batch-size', type=int, default=8, help='prompts per rollout batch')
    parser.add_argument('--mini-batch-size', type=int, default=32, help='samples per optimizer step')
    parser.add_argument('--micro-batch-size', type=int, default=16, help='row limit per micro-batch and rank')
    parser.add_argument('--dynamic-batching', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-tokens-per-micro-batch', type=int, default=8192)
    parser.add_argument('--padding-free', action='store_true', help='packed inputs instead of padded ones')
    parser.add_argument('--balance-batch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--attn-implementation', default='flash_attention_2')
    return parser.parse_args()


class NPULinearAttentionKernel(Patch):
    """Install the flash-linear-attention GatedDeltaNet kernels on NPU.

    Transformers reports them as available only on CUDA-like devices; the twinkle 'fla' kernel op installs
    them on NPU. Under Ray the model lives on the workers, so this runs there through `apply_patch`.
    """

    def __call__(self, module, *args, **kwargs):
        from twinkle.kernel import kernelize
        from twinkle.kernel.config import DEFAULT_KERNEL_CONFIG
        kernelize(module, {'fla': DEFAULT_KERNEL_CONFIG['fla']})


def main():
    args = parse_args()
    device_type = Platform.get_platform().__name__
    config_dir = HubOperation.download_model(args.model_id, ignore_model=True)
    hf_config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    is_qwen35 = hf_config.model_type.startswith('qwen3_5')
    template = 'Qwen3_5Template' if is_qwen35 else 'Template'
    num_gpus = args.model_gpus + args.sampler_gpus

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(args.model_gpus)), device_type=device_type),
        DeviceGroup(name='sampler', ranks=list(range(args.model_gpus, num_gpus)), device_type=device_type),
    ]
    if args.strategy == 'native_fsdp':
        model_mesh = DeviceMesh.from_sizes(world_size=args.model_gpus, fsdp_size=args.model_gpus)
    else:
        model_mesh = DeviceMesh.from_sizes(world_size=args.model_gpus, dp_size=args.model_gpus)
    sampler_mesh = DeviceMesh.from_sizes(world_size=args.sampler_gpus, dp_size=args.sampler_gpus)
    twinkle.initialize(mode='ray', nproc_per_node=num_gpus, groups=device_groups, lazy_collect=False)
    # slice_dp splits by data_world_size (dp x fsdp), not dp_size: FSDP2 alone still has fsdp_size data ranks.
    dp_size = model_mesh.data_world_size

    # Must match what forward_backward receives: the plan simulates the model's own micro-batch planner.
    micro_batch_kwargs = {
        'micro_batch_size': args.micro_batch_size,
        'dynamic_batching': args.dynamic_batching,
        'max_tokens_per_micro_batch': args.max_tokens_per_micro_batch if args.dynamic_batching else None,
        'packing_algorithm': 'ffd',
    }
    micro_batch_config = MicroBatchConfig(**micro_batch_kwargs)
    # The linear term of the workload estimate grows with the hidden size.
    hidden_size = (getattr(hf_config, 'text_config', None) or hf_config).hidden_size
    workload_fn = partial(estimate_workload, hidden_size=hidden_size)

    model_kwargs = {'attn_implementation': args.attn_implementation} if args.attn_implementation else {}
    if is_qwen35:
        from transformers import Qwen3_5ForConditionalGeneration
        model_kwargs['model_cls'] = Qwen3_5ForConditionalGeneration
    model = TransformersModel(
        model_id=args.model_id, device_mesh=model_mesh, remote_group='model', strategy=args.strategy, **model_kwargs)
    if device_type == 'NPU' and args.padding_free:
        model.apply_patch(NPULinearAttentionKernel)
    # Language-model projections only (full and linear attention, MLP); 'all-linear' would add the vision tower.
    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'in_proj_qkv', 'in_proj_z',
            'in_proj_a', 'in_proj_b', 'out_proj'
        ],
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05)
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=args.lr)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=args.max_steps, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=args.padding_free)
    model.set_template(template, model_id=args.model_id, enable_thinking=True)

    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': max(16, args.lora_r),
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(template, model_id=args.model_id, enable_thinking=True)
    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler, platform=device_type)

    def create_gsm8k_dataset():
        dataset = Dataset()
        dataset.add_dataset(DatasetMeta(args.dataset_id, subset_name=args.subset_name, split='train'))
        # Thinking on and no length penalty: completion lengths differ a lot between prompts.
        dataset.set_template(
            template, model_id=args.model_id, max_length=4096, truncation_strategy='delete', enable_thinking=True)
        dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
        dataset.encode(add_generation_prompt=True)
        return dataset

    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=args.batch_size,
        min_batch_size=args.batch_size,
        device_mesh=model_mesh,
        remote_group='model',
    )
    accuracy_reward = GSM8KAccuracyReward()
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)

    logger.info(get_device_placement())
    logger.info(f'strategy={args.strategy}, padding_free={args.padding_free}, balance_batch={args.balance_batch}, '
                f'model DP={dp_size}, micro-batch {micro_batch_kwargs}')
    optim_step = 0
    for batch in dataloader:
        if optim_step >= args.max_steps:
            break
        metrics.reset()
        expand_prompts = [prompt for prompt in batch for _ in range(args.num_generations)]
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()
        sample_responses = sampler.sample(expand_prompts, sampling_params)

        all_inputs: List[Dict[str, Any]] = []
        trajectories: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        completion_lengths: List[int] = []
        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_inputs.append(sequence.new_input_feature)
                # new_input_feature carries tokens only; the conversation, which rewards score, is in new_state.
                trajectories.append({**sequence.new_input_feature, 'messages': list(sequence.new_state.messages)})
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                completion_lengths.append(len(sequence.tokens))

        rewards = accuracy_reward(trajectories)
        metrics.accumulate(completion_lengths=completion_lengths, rewards={'accuracy': rewards})
        advantages = advantage_fn(rewards, num_generations=args.num_generations, scale='group').tolist()

        balance_metrics: Dict[str, List[float]] = {}
        for mb_start in range(0, len(all_inputs), args.mini_batch_size):
            mb_inputs = all_inputs[mb_start:mb_start + args.mini_batch_size]
            mb_old_logps = all_old_logps[mb_start:mb_start + args.mini_batch_size]
            mb_advantages = advantages[mb_start:mb_start + args.mini_batch_size]
            if args.balance_batch:
                plan = plan_dp_balance(
                    mb_inputs,
                    dp_size,
                    padding_free=args.padding_free,
                    micro_batch_config=micro_batch_config,
                    workload_fn=workload_fn,
                )
                # Every per-sample list is reordered with the same plan so they stay aligned.
                mb_inputs, mb_old_logps, mb_advantages = (
                    plan.apply(values) for values in (mb_inputs, mb_old_logps, mb_advantages))
                for key, value in plan.metrics.items():
                    balance_metrics.setdefault(key, []).append(value)

            model.forward_backward(
                inputs=mb_inputs, old_logps=mb_old_logps, advantages=mb_advantages, **micro_batch_kwargs)
            model.clip_grad_and_step()
            optim_step += 1
            if optim_step >= args.max_steps:
                break

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict.update({key: sum(values) / len(values) for key, values in balance_metrics.items()})
        logger.info(f'[Step {optim_step}/{args.max_steps}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('dp-balance-grpo-final')


if __name__ == '__main__':
    main()
