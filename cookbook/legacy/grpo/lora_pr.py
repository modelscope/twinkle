import gc
import time
from typing import List, Tuple
from peft import LoraConfig
import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, Trajectory, InputFeature
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import VLLMSampler
from twinkle.template import Template
from twinkle import torch_util

logger = get_logger()


def create_countdown_dataset():
    from twinkle.preprocessor import CountdownProcessor
    dataset = Dataset(DatasetMeta("ms://zouxuhong/Countdown-Tasks-3to4", data_slice=range(50000)))
    dataset.set_template("Template", model_id='ms://Qwen/Qwen2.5-3B-Instruct', max_length=8192)
    dataset.map(CountdownProcessor())
    dataset.encode()
    return dataset


def compute_rewards(trajectories: List[Trajectory]) -> Tuple[List[float], List[float], List[float]]:
    from twinkle.reward import CountDownAccuracy, FormatReward
    format_rewards = FormatReward()(trajectories, [])
    accuracy_rewards = CountDownAccuracy()(trajectories, [])
    total_rewards = [a+b for a, b in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards

def main():
    device_groups = [
        DeviceGroup(name='model', ranks=4, device_type='GPU', gpus_per_worker=1),
        DeviceGroup(name='sampler', ranks=4, device_type='GPU', gpus_per_worker=1),
    ]
    model_mesh = DeviceMesh.from_sizes(dp_size=4)
    sampler_mesh = DeviceMesh.from_sizes(dp_size=4)
    twinkle.initialize(mode='ray', nproc_per_node=8, groups=device_groups, lazy_collect=False)
    logger.info(get_device_placement())
    lora_config = LoraConfig(target_modules="all-linear", r=8, lora_alpha=32, lora_dropout=0.05)
    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-3B-Instruct', device_mesh=model_mesh, remote_group='model')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=4,)
    sampler = VLLMSampler(
        model_id='ms://Qwen/Qwen2.5-3B-Instruct',
        engine_args={
            'load_format': 'dummy',
            'gpu_memory_utilization': 0.7,
            'max_model_len': 2048,
            'enforce_eager': True,
            'enable_sleep_mode': False,
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id='ms://Qwen/Qwen2.5-3B-Instruct')

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    dataloader = DataLoader(
        dataset=create_countdown_dataset, batch_size=4, min_batch_size=4,
        device_mesh=model_mesh, remote_group='model', num_workers=0,
    )
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(max_tokens=1024, temperature=1.0, top_p=0.95)
    step = 0
    model.set_optimizer('AdamW', lr=1e-5)
    model.set_lr_scheduler(scheduler_cls='CosineWarmupScheduler', num_warmup_steps=500, num_training_steps=2000)
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
    model.set_processor(InputProcessor)
    model.set_template('Template', model_id='ms://Qwen/Qwen2.5-3B-Instruct')

    for batch in dataloader:
        if step >= 2000:
            break

        metrics.reset()
        prompts = batch if isinstance(batch, list) else [batch]
        if step % 1 == 0:
            ckpt_manager.sync_weights(adapter_name='default')
        sample_response = sampler.sample(prompts, sampling_params, num_samples=8)
        trajectories: List[Trajectory] = []
        input_features: List[InputFeature] = []
        old_logps_list: List[List[float]] = []
        completion_lengths: List[int] = []

        for sequence in sample_response.sequences:
            input_features.append(sequence.new_input_feature)
            trajectories.append(sequence.new_input_feature)
            old_logps_list.append(sequence.logprobs)
            completion_lengths.append(len(sequence.tokens))

        if not trajectories:
            logger.warning(f"Step {step}: No valid samples, skipping")
            step += 1
            continue

        total_rewards, format_rewards, accuracy_rewards = compute_rewards(trajectories)
        metrics.accumulate(None, None,
                           completion_lengths=completion_lengths,
                           rewards={
                               'total': total_rewards,
                               'format': format_rewards,
                               'accuracy': accuracy_rewards,
                           })

        advantages = advantage_fn(total_rewards, num_generations=8, scale='group').tolist()
        frac_zero_std = 1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0
        if frac_zero_std == 1.0:
            logger.info(f"Step {step}: All advantages are zero, skipping training")
            step += 1
            continue

        model.forward_backward(
            inputs=input_features,
            advantages=advantages,
            old_logps=old_logps_list,
        )
        model.clip_grad_and_step()
        gc.collect()
        torch_util.empty_cache()
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric())
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        logger.info(log_dict)
        step += 1

    model.save('grpo-countdown-checkpoint')


if __name__ == '__main__':
    main()
