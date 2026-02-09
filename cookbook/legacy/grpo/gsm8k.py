"""
GSM8K GRPO training demo for Twinkle.

This demo trains a model on Grade School Math (GSM8K) using GRPO.
Reward = accuracy_reward (1.0 if answer correct, 0.0 otherwise)
       + format_reward  (1.0 if output contains <think>...</think>, 0.0 otherwise)

Expected to show reward improvement within ~30-80 steps.

Reference configs:
  - Verl: run_qwen2_5-3b_gsm8k_grpo_lora.sh (lr=3e-6, n=5, max_resp=1024)
  - TRL:  accuracy_reward (math_verify based)
  - Swift: countdown demo (lr=1e-5, n=8, gas=8)
"""
import gc
import os
import re
import time
from typing import List, Tuple, Dict, Any

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, SampleResponse
from twinkle.data_format import Trajectory, InputFeature, Message
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.metric import CompletionRewardMetric

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen2.5-3B-Instruct')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '0')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 2))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
GRPO_EPSILON = float(os.environ.get('GRPO_EPSILON', 0.2))
GRPO_BETA = float(os.environ.get('GRPO_BETA', 0.0))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 200))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 1.0))
WEIGHT_SYNC_INTERVAL = int(os.environ.get('WEIGHT_SYNC_INTERVAL', 1))
ADAPTER_NAME = 'default'
DATA_NUM = int(os.environ.get('DATA_NUM', 7473))  # GSM8K train split has 7473 samples

# SwanLab experiment tracking
USE_SWANLAB = bool(int(os.environ.get('USE_SWANLAB', '1')))
if USE_SWANLAB:
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)
    swanlab.init(project="twinkle-gsm8k", config={
        'model_id': MODEL_ID,
        'num_gpus': NUM_GPUS,
        'model_gpus': MODEL_GPUS,
        'sampler_gpus': SAMPLER_GPUS,
        'num_generations': NUM_GENERATIONS,
        'max_new_tokens': MAX_NEW_TOKENS,
        'learning_rate': LEARNING_RATE,
        'grpo_epsilon': GRPO_EPSILON,
        'grpo_beta': GRPO_BETA,
        'batch_size': BATCH_SIZE,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    })


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "Show your reasoning in <think> </think> tags, then give the final "
    "numerical answer after ####.\n"
    "For example:\n<think> ... reasoning ... </think>\n#### 42"
)


class GSM8KProcessor(Preprocessor):
    """Preprocessor for GSM8K dataset.

    GSM8K fields: question (str), answer (str ending with '#### <number>')
    Extracts the ground truth number and stores it in user_data for reward.
    """

    @staticmethod
    def extract_ground_truth(answer_str: str) -> str:
        """Extract the number after '####' from GSM8K answer."""
        match = re.search(r'####\s*([\-\d,\.]+)', answer_str)
        if match:
            return match.group(1).replace(',', '').strip()
        return ''

    def __call__(self, row) -> Trajectory:
        question = row['question']
        answer = row.get('answer', '')
        ground_truth = self.extract_ground_truth(answer)

        messages = [
            Message(role='system', content=SYSTEM_PROMPT),
            Message(role='user', content=question),
        ]
        return Trajectory(
            messages=messages,
            user_data=[('ground_truth', ground_truth)],
        )


# ========== GSM8K Reward Functions ==========
class GSM8KAccuracyReward(Reward):
    """Accuracy reward for GSM8K: checks if the model's answer matches ground truth.

    Extracts the last '#### <number>' from model output and compares with ground truth.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    @staticmethod
    def extract_answer(completion: str) -> str:
        """Extract the last #### answer from model completion."""
        # Only check last 500 chars for efficiency
        text = completion[-500:] if len(completion) > 500 else completion
        matches = re.findall(r'####\s*([\-\d,\.\s]+)', text)
        if matches:
            return matches[-1].replace(',', '').replace(' ', '').strip()
        return ''

    def __call__(
        self, trajectories: List[Trajectory], ground_truths: List[Trajectory]
    ) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            # Get model completion (last assistant message)
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            # Get ground truth from user_data
            gt = ''
            user_data = trajectory.get('user_data', [])
            if isinstance(user_data, list):
                for item in user_data:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        if item[0] == 'ground_truth':
                            gt = str(item[1])
                            break

            predicted = self.extract_answer(completion)

            # Numeric comparison
            correct = False
            if predicted and gt:
                try:
                    correct = abs(float(predicted) - float(gt)) < 1e-5
                except (ValueError, OverflowError):
                    correct = predicted == gt

            rewards.append(1.0 if correct else 0.0)
        return rewards


class GSM8KFormatReward(Reward):
    """Format reward: checks if output contains <think>...</think> tag.

    Returns 1.0 if format is correct, 0.0 otherwise.
    """

    def __call__(
        self, trajectories: List[Trajectory], ground_truths: List[Trajectory]
    ) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_think = bool(
                re.search(r'<think>.*?</think>', completion, re.DOTALL)
            )
            has_answer = bool(re.search(r'####\s*[\-\d,\.]+', completion))
            rewards.append(1.0 if (has_think and has_answer) else 0.0)
        return rewards


def create_gsm8k_dataset():
    """Create GSM8K dataset."""
    meta = DatasetMeta(
        "ms://modelscope/gsm8k",
        subset_name='main', split='train',
        data_slice=range(DATA_NUM),
    )
    dataset = Dataset(meta)
    dataset.set_template("Template", model_id=MODEL_ID, max_length=2048)
    dataset.map(GSM8KProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Trajectory],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute accuracy and format rewards for GSM8K."""
    accuracy_reward_fn = GSM8KAccuracyReward()
    format_reward_fn = GSM8KFormatReward()

    accuracy_rewards = accuracy_reward_fn(trajectories, [])
    format_rewards = format_reward_fn(trajectories, [])
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards


# ========== Main ==========
def main():
    device_groups = [
        DeviceGroup(
            name='model',
            ranks=list(range(MODEL_GPUS)),
            device_type='GPU',
            gpus_per_worker=1,
        ),
        DeviceGroup(
            name='sampler',
            ranks=list(range(MODEL_GPUS, NUM_GPUS)),
            device_type='GPU',
            gpus_per_worker=1,
        ),
    ]
    if USE_MEGATRON:
        model_mesh = DeviceMesh.from_sizes(
            dp_size=MODEL_GPUS, tp_size=1, pp_size=1,
        )
    else:
        model_mesh = DeviceMesh.from_sizes(
            world_size=MODEL_GPUS, dp_size=MODEL_GPUS,
        )
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS,
    )
    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
        lazy_collect=False,
    )
    logger.info(get_device_placement())

    lora_config = LoraConfig(
        target_modules="all-linear",
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # ── Model ─────────────────────────────────────────────────────────
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            recompute_granularity='selective',
            recompute_num_layers=None,
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(
        ADAPTER_NAME,
        lora_config,
        gradient_accumulation_steps=1,
    )
    if USE_MEGATRON:
        model.set_optimizer(
            'default', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME,
        )
        model.set_lr_scheduler(
            'default',
            lr_decay_steps=MAX_STEPS,
            max_lr=LEARNING_RATE,
            adapter_name=ADAPTER_NAME,
        )
    else:
        model.set_optimizer(
            'AdamW', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME,
        )
        model.set_lr_scheduler(
            'CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0,
            adapter_name=ADAPTER_NAME,
        )
    model.set_loss(
        'GRPOLoss',
        adapter_name=ADAPTER_NAME,
        epsilon=GRPO_EPSILON,
        beta=GRPO_BETA,
    )
    model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    model.set_template('Template', model_id=MODEL_ID, adapter_name=ADAPTER_NAME)

    # ── Sampler (load real weights for meaningful generation) ─────────
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.7,
            'max_model_len': 4096,
            'max_lora_rank': 64,
            'enforce_eager': True,
            'enable_sleep_mode': False,
            'enable_lora': True,
            "logprobs_mode": "processed_logprobs",
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    # Global batch = prompts for one full gradient accumulation cycle
    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
        num_workers=0,
    )
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    # ── Training loop ────────────────────────────────────────────────
    optim_step = 0

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        step_start = time.perf_counter()
        metrics.reset()
        timings: Dict[str, float] = {
            'weight_sync': 0.0,
            'generate': 0.0,
            'reward': 0.0,
            'advantage': 0.0,
            'train': 0.0,
            'total': 0.0,
        }

        global_prompts = batch if isinstance(batch, list) else [batch]

        t0 = time.perf_counter()
        if optim_step % WEIGHT_SYNC_INTERVAL == 0:
            ckpt_manager.sync_weights(adapter_name=ADAPTER_NAME)
            sampler.reset_prefix_cache()
        timings['weight_sync'] = time.perf_counter() - t0

        t1 = time.perf_counter()
        sample_response = sampler.sample(
            global_prompts,
            sampling_params,
            num_samples=NUM_GENERATIONS,
        )
        timings['generate'] = time.perf_counter() - t1

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sequence in sample_response.sequences:
            all_input_data.append(sequence.new_input_feature)
            all_old_logps.append(sequence.logprobs)
            all_completion_lengths.append(len(sequence.tokens))

        if not all_input_data:
            logger.warning(
                f"Optim step {optim_step}: No valid samples, skipping"
            )
            continue

        # ========== 3. Rewards ==========
        t2 = time.perf_counter()
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(
            all_input_data
        )
        timings['reward'] = time.perf_counter() - t2

        metrics.accumulate(
            None,
            None,
            generate_time=timings['generate'],
            weight_sync_time=timings['weight_sync'],
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        # ========== 4. Advantages ==========
        t3 = time.perf_counter()
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        )
        advantages = advantages.tolist()
        timings['advantage'] = time.perf_counter() - t3

        frac_zero_std = (
            1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0
        )

        # ========== 5. Training ==========
        t4 = time.perf_counter()

        if all(abs(a) < 1e-8 for a in advantages):
            logger.info(
                f"Optim step {optim_step}: "
                f"All advantages zero, skipping training"
            )
        else:
            model.forward_backward(
                inputs=all_input_data,
                adapter_name=ADAPTER_NAME,
                advantages=advantages,
                old_logps=all_old_logps,
            )

        model.clip_grad_and_step(adapter_name=ADAPTER_NAME)
        timings['train'] = time.perf_counter() - t4

        gc.collect()
        from twinkle import torch_util
        torch_util.empty_cache()

        timings['total'] = time.perf_counter() - step_start
        optim_step += 1

        # ========== 6. Log ==========
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        log_dict['train/optim_step'] = optim_step
        for k, v in timings.items():
            log_dict[f'time/{k}'] = round(v, 2)

        if USE_SWANLAB:
            swanlab.log(log_dict)
        logger.info(f"[Step {optim_step}/{MAX_STEPS}] {log_dict}")

    logger.info(f"Training completed. optim_steps={optim_step}")
    model.save('grpo-gsm8k-checkpoint', adapter_name=ADAPTER_NAME)


if __name__ == '__main__':
    main()
