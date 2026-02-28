# Quick Start

## ✨ What is Twinkle?

A component library for large model training. Based on PyTorch, simpler, more flexible, production-ready.

🧩 <b>Loosely Coupled Architecture</b> · Standardized Interfaces<br>
🚀 <b>Multiple Runtime Modes</b> · torchrun / Ray / HTTP<br>
🔌 <b>Multi-Framework Compatible</b> · Transformers / Megatron<br>
👥 <b>Multi-Tenant Support</b> · Single Base Model Deployment

## Twinkle Compatibility

Twinkle and [ms-swift](https://github.com/modelscope/ms-swift) are both model training frameworks, but they have very different characteristics. Developers can choose based on their needs.

### When to Choose Twinkle

- If you are a beginner in large models and want to better understand model mechanisms and training methods
- If you are a large model researcher who wants to customize models or training methods
- If you are good at writing training loops and want to customize the training process
- If you want to provide enterprise-level or commercial training platforms

### When to Choose ms-swift

- If you don't care about the training process and just want to provide a dataset to complete training
- If you need more model support and dataset varieties
- If you need various types of training such as Embedding, Reranker, Classification
- If you need other capabilities like inference, deployment, quantization
- If you are sensitive to new model training support, Swift guarantees day-0 update capability

## Usage Patterns

### Using Only Partial Components

Developers can use only a portion of Twinkle's components, combining them with their own existing code to complete training work. For example, using only Dataset & DataLoader:

```python
from twinkle.dataset import PackingDataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.preprocessor import SelfCognitionProcessor

def train():
    dataset_meta = DatasetMeta(
        dataset_id='ms://swift/self-cognition',
    )

    dataset = PackingDataset(dataset_meta)
    dataset.map(SelfCognitionProcessor(model_name='Twinkle Model', model_author='ModelScope Community'))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3-4B', max_length=512)
    dataset.encode()
    dataset.pack_dataset()

    dataloader = DataLoader(dataset, batch_size=8)
    for data in dataloader:
        print(data)
        """
        {
            "input_ids": [...],
            "position_ids": [...],
            ...
        }
        """
        break

if __name__ == '__main__':
    train()
```
In the code above, we use PackingDataset to load a dataset called `swift/self-cognition`. PackingDataset can be used to bin-pack data, ensuring that each batch has a length similar to the configured maximum length.
In the loop, we simply used print to display the output. In actual use, you can continue writing your custom training code below.

All of Twinkle's components support being used separately. Please refer to the component list in the sections below.

### Single GPU

Twinkle supports running training on a single GPU. Here is an example:

```python
from peft import LoraConfig

from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()


def train():
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3-4B')
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle LLM', 'ModelScope Community'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # Use a TransformersModel
    model = TransformersModel(model_id='ms://Qwen/Qwen3-4B')

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % 20 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
```

In this training code, we constructed a dataset and loaded the Qwen/Qwen3-4B model, used LoRA with the all-linear approach, and completed one training run. In the logs, you can observe the process of loss gradually converging.

### torchrun

Twinkle supports running training in torchrun mode. In this scenario, Ray-related dependencies do not need to be installed.

```python
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

# Construct a device_mesh, fsdp=4, dp=2
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def train():
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3-4B')
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle LLM', 'ModelScope Community'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # Use a TransformersModel
    model = TransformersModel(model_id='ms://Qwen/Qwen3-4B')

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % 20 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
```

In the code above, we constructed a hybrid parallel mode combining FSDP2 and DP, and used 8 GPUs for training. You can see that it is basically the same as the single-GPU training code, except that `DeviceMesh` is used to declare the model layout.

When running, you need to launch training like this:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py
```

### Ray Training

[Ray](https://github.com/ray-project/ray) is a commonly used scheduling middleware framework for multi-machine model training and inference scenarios. It provides additional optimizations for multi-model, multi-device execution and resource management, and supports integration with Kubernetes systems for production deployment. These characteristics make it particularly suitable for complex training scenarios such as RL and GKD.

Twinkle supports using Ray for training and sampling, and its code is almost identical to the training API above:

```python
import os
from typing import List, Tuple, Dict, Any
from peft import LoraConfig
import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model.megatron import MegatronModel
from twinkle.metric import CompletionRewardMetric
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3-4B')
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS',4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 200))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16)) # global prompt-level, global completion-level batch size = BATCH_SIZE * num_generations * dp_size
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 16)) # global completion-level mini-batch-size
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2)) # per-device-micro-batch-size (completion-level), batch_size in forward_backward
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'

def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=2048)
    dataset.map(GSM8KProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset

def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_reward_fn = GSM8KAccuracyReward()
    format_reward_fn = GSM8KFormatReward()
    accuracy_rewards = accuracy_reward_fn(trajectories)
    format_rewards = format_reward_fn(trajectories)
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards

def main():
    # set sampler and model separate to use different gpus
    device_groups = [
        DeviceGroup(name='model',ranks=list(range(MODEL_GPUS)),device_type='GPU'),
        DeviceGroup(name='sampler',ranks=list(range(MODEL_GPUS, NUM_GPUS)),device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(target_modules='all-linear', r=32, lora_alpha=64, lora_dropout=0.05)
    model = MegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model', mixed_precision='bf16')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('default', lr=LEARNING_RATE)
    model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Template', model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 4096,
            'max_lora_rank': 32, # save as lora_config
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)
    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        min_batch_size=BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        device_mesh=model_mesh,
        remote_group='model',
    )
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS)
    optim_step = 0
    print(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        global_prompts = batch if isinstance(batch, list) else [batch]
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()
        sample_response = sampler.sample(
            global_prompts*NUM_GENERATIONS,
            sampling_params,
            num_samples=1,
        )
        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sequence in sample_response.sequences:
            all_input_data.append(sequence.new_input_feature)
            all_old_logps.append(sequence.logprobs)
            all_completion_lengths.append(len(sequence.tokens))
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(
            all_input_data
        )
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            },
        )
        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
        # Split completions into mini-batches and run one optim step per mini-batch.
        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]

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
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            print(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    print(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-gsm8k-checkpoint')

if __name__ == '__main__':
    main()
```

In the code above, we provide an RL training example. We can clearly see in the code how data is constructed, how the sampler/model are declared and parameterized, and the construction process for advantage and loss.
There is no explicit reference to `ray` anywhere in this process. We only declared Ray mode during initialization:

```python
twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)
```

Developers can customize the construction and invocation methods of components like models. All Transformers and Megatron model parameters can be passed in when constructing the model.

All subsequent Ray calls and data distribution are performed implicitly. Running this script requires having Ray installed beforehand. Then run it like this:

```shell
python train.py
```

### Remote Training

A major feature of Twinkle is support for multi-tenant mixed training. Specifically, multiple users can use a single base model for LoRA training, which can greatly reduce server-side deployment costs.

Suppose we start a service using eight GPUs. First, we need to start the Ray cluster:

```shell
CUDA_VISIBLE_DEVICES=0,1 ray start --head --port=6379 --num-gpus=2
CUDA_VISIBLE_DEVICES=2,3 ray start --address=127.0.0.1:6379 --num-gpus=2
CUDA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0
```

We started a Ray cluster containing three nodes:
- GPUs 0 and 1 as one node
- GPUs 2 and 3 as one node
- CPU resources as one node

For production environments, you can start more nodes and deploy more replicas to accommodate larger user volumes. Here we only use four GPUs as an example.

Next, start the server:
```shell

cd cookbook/client/twinkle/transformer
python server.py
```

The server will start three services: a sampler cluster, a model cluster, and a utility cluster.

Now you can perform client-side training:
```python
import dotenv
dotenv.load_dotenv('.env')
import re
from twinkle.data_format import Trajectory
from twinkle.reward.base import Reward
import gc
from peft import LoraConfig
from typing import List, Tuple

from twinkle import get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.dataset import DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle_client import init_twinkle_client
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = 'ms://Qwen/Qwen3-4B'
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 1024
LEARNING_RATE = 1e-5
MAX_STEPS = 10
BATCH_SIZE = 2
TEMPERATURE = 1.0
SYNC_INTERVAL = 1  # Save weights for sampler every N steps
GRADIENT_ACCUMULATION_STEPS = 4


def create_countdown_dataset():
    """Create Countdown Game dataset for GRPO training."""

    dataset = Dataset(dataset_meta=DatasetMeta('ms://zouxuhong/Countdown-Tasks-3to4', data_slice=range(500)))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=8192)
    dataset.map('CountdownProcessor')
    dataset.encode(add_generation_prompt=True, batched=True)
    return dataset


class CountDownAccuracy(Reward):

    @staticmethod
    def countdown_accuracy_reward(completion: str, target: int, nums: List[int]) -> float:
        """Accuracy reward: checks if equation is correct."""
        try:
            match = re.search(r'<answer>(.*?)<\/answer>', completion)
            if match is None:
                return 0.0
            equation = match.group(1).strip()
            if '=' in equation:
                equation = equation.split('=')[0]
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
            if sorted(used_numbers) != sorted(nums):
                return 0.0
            if not re.match(r'^[\d+\-*/().\s]+$', equation):
                return 0.0
            result = eval(equation, {'__builtins__': None}, {})
            return 1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0
        except Exception:  # noqa
            return 0.0

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            user_data = trajectory.get('user_data', [{}])
            data = user_data[0] if isinstance(user_data, list) and user_data else {}
            target = data.get('target', 0)
            nums = data.get('nums', [])
            acc_reward = self.countdown_accuracy_reward(completion, target, nums)
            rewards.append(acc_reward)
        return rewards


def compute_rewards(trajectories: List[dict], ) -> Tuple[List[float], List[float], List[float]]:
    """Compute format and accuracy rewards for Countdown game."""
    from twinkle.reward import FormatReward
    format_rewards = FormatReward()(trajectories, [])
    accuracy_rewards = CountDownAccuracy()(trajectories, [])
    total_rewards = [a + b for a, b in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards


def train():
    # Step 1: Initialize the Twinkle client
    client = init_twinkle_client(
        base_url='http://localhost:8000',
        api_key='',
    )

    # Step 2: Prepare dataset and dataloader
    dataset = create_countdown_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    # Step 3: Configure the training model
    model = MultiLoraTransformersModel(model_id=MODEL_ID)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model.add_adapter_to_model(
        'default',
        lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    # Set GRPO loss (the key difference from SFT training)
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)

    # Set optimizer and LR scheduler
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler(
        'CosineWarmupScheduler',
        num_warmup_steps=500,
        num_training_steps=MAX_STEPS,
    )

    # Set processor and template for encoding inputs
    model.set_processor('InputProcessor')
    model.set_template('Template', model_id=MODEL_ID)

    # Step 4: Configure the sampler
    sampler = vLLMSampler(model_id=MODEL_ID)
    sampler.set_template('Template', model_id=MODEL_ID)

    # Step 5: Setup metrics and advantage function
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = {
        'max_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'top_p': 0.95,
    }

    # Track the current adapter path for sampling
    current_adapter_uri = None

    step = 0
    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        metrics.reset()
        prompts = batch if isinstance(batch, list) else [batch]

        # ========== 1. Save weights and update adapter_uri ==========
        # Instead of sync_weights, save the model checkpoint and pass
        # the resulting path to the sampler as adapter_uri
        if step % SYNC_INTERVAL == 0:
            logger.info(f'Step {step}: Saving weights for sampler...')
            twinkle_path = model.save(
                name=f'grpo-sampler-step-{step}',
                save_optimizer=False,
            )
            current_adapter_uri = twinkle_path
            logger.info(f'Step {step}: Saved weights to {current_adapter_uri}')

        # ========== 2. Sample completions ==========
        sample_response = sampler.sample(
            inputs=prompts,
            sampling_params=sampling_params,
            adapter_uri=current_adapter_uri,
            num_samples=NUM_GENERATIONS,
        )

        input_features = []
        old_logps_list = []
        completion_lengths = []

        sequences = sample_response.get('sequences', [])
        for seq in sequences:
            input_features.append(seq.get('new_input_feature', seq))
            old_logps_list.append(seq.get('logprobs', []))
            completion_lengths.append(len(seq.get('tokens', [])))

        if not input_features:
            logger.warning(f'Step {step}: No valid samples, skipping')
            step += 1
            continue

        # ========== 3. Compute rewards ==========
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(input_features)
        metrics.accumulate(
            None,
            None,
            completion_lengths=completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            })

        # ========== 4. Compute advantages ==========
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        ).tolist()

        frac_zero_std = (1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0)
        if frac_zero_std == 1.0:
            logger.info(f'Step {step}: All advantages are zero, skipping training')
            step += 1
            continue

        # ========== 5. Training step (GRPO) ==========
        # forward_backward with GRPO loss: passes advantages and old_logps
        # to the server-side GRPOLoss for proper policy optimization
        model.forward_backward(
            inputs=input_features,
            advantages=advantages,
            old_logps=old_logps_list,
        )

        # Gradient clipping and optimizer step
        model.clip_grad_norm(1.0)
        model.step()
        model.zero_grad()
        model.lr_step()

        gc.collect()

        # ========== 6. Log ==========
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric())
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        logger.info(f'Step {step}: {log_dict}')
        step += 1

    # Save final checkpoint
    twinkle_path = model.save(name='grpo-countdown-final', save_optimizer=True)
    logger.info(f'Saved final checkpoint: {twinkle_path}')


if __name__ == '__main__':
    train()
```

Multiple developers can use a single base model from this service for parallel training and sampling. Furthermore, the training methods they use are allowed to differ. For example, User A can perform SFT, User B can perform RL, and User C can perform sampling. Similarly, Twinkle also supports Tinker-like APIs for remote training:

>[!Note]
> One important note: in the current Twinkle implementation, the client-side Twinkle API and Tinker API cannot be used simultaneously on the same server. When you need to provide the Tinker API, you need to start the service under cookbook/client/tinker.
> This issue will be addressed with high priority in upcoming iterations.

```python
from tinker import types
from tqdm import tqdm
from tinker import ServiceClient
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum

# The base model to fine-tune / evaluate
base_model = 'ms://Qwen/Qwen3-4B'


def train():
    # Step 1: Prepare the dataset

    # Load the self-cognition dataset from ModelScope (first 500 examples)
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))

    # Apply the chat template matching the base model (max 256 tokens per sample)
    dataset.set_template('Template', model_id=f'ms://{base_model}', max_length=256)

    # Replace placeholder names with custom model/author identity
    dataset.map(SelfCognitionProcessor('twinkle model', 'twinkle team'), load_from_cache_file=False)

    # Tokenize and encode the dataset into model-ready input features
    dataset.encode(batched=True, load_from_cache_file=False)

    # Wrap the dataset into a DataLoader that yields batches of size 8
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # Step 2: Initialize the training client
    # Connect to the Twinkle server running locally
    service_client = ServiceClient(base_url='http://localhost:8000', api_key='your-api-key')
    # Create a LoRA training client for the base model (rank=16 for the LoRA adapter)
    training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

    # Step 3: Run the training loop
    for epoch in range(3):
        print(f'Epoch {epoch}')
        for step, batch in tqdm(enumerate(dataloader)):
            # Convert each InputFeature into a Datum for the Tinker API
            input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

            # Send data to server: forward + backward pass (computes gradients)
            fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')

            # Optimizer step: update model weights with Adam
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

            # Wait for both operations to complete
            fwdbwd_future.result()
            optim_result = optim_future.result()
            print(f'Training Metrics: {optim_result}')

        # Save a checkpoint after each epoch
        save_future = training_client.save_state(f'twinkle-lora-{epoch}')
        save_result = save_future.result()
        print(f'Saved checkpoint to {save_result.path}')


if __name__ == '__main__':
    train()
```

### Using ModelScope Community's TaaS Training Service

Concurrent with the open-source release of the Twinkle framework, we also provide a hosted Training as a Service (TaaS) powered by ModelScope's backend services. Developers can experience Twinkle's training API for free through this service.
This service shares the same code as the Tinker API section described above. The only difference is that the Endpoint and Token need to use the official ModelScope information. For details on how to use the official service, please refer to the detailed description in [Training Service](./Train-as-a-Service.md).

## Using Hugging Face models

Switch the prefix.

```text
ms://Qwen/Qwen3-4B -> hf://Qwen/Qwen3-4B
```

## 🛠️ Twinkle✨ Modular Ecosystem

<div align="center">
  <table style="width: 100%; border-collapse: separate; border-spacing: 8px;">
    <tr>
      <td width="20%" bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Dataset</b><br><sub>Data loading and preprocessing</sub></p>
      </td>
      <td width="20%" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Template</b><br><sub>Encoding and decoding</sub></p>
      </td>
      <td width="20%" bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>DataLoader</b><br><sub>Data distribution and batching</sub></p>
      </td>
      <td width="20%" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Preprocessor</b><br><sub>Data ETL</sub></p>
      </td>
      <td width="20%" bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>InputProcessor</b><br><sub>Task-specific input processing</sub></p>
      </td>
    </tr>
    <tr>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Model</b><br><sub>Large models, supports multiple frameworks</sub></p>
      </td>
      <td style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Sampler</b><br><sub>Sampler logic</sub></p>
      </td>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Loss</b><br><sub>Loss functions</sub></p>
      </td>
      <td style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Metric</b><br><sub>Training metrics collection</sub></p>
      </td>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Reward</b><br><sub>Reward function</sub></p>
      </td>
    </tr>
    <tr>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Advantage</b><br><sub>Advantage function</sub></p>
      </td>
      <td style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>CheckpointEngine</b><br><sub>Weight synchronization</sub></p>
      </td>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Patch</b><br><sub>Patches for model fixes</sub></p>
      </td>
      <td style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Module</b><br><sub>Components, e.g., Optimizer</sub></p>
      </td>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Kernel</b><br><sub>Operators</sub></p>
      </td>
    </tr>
    <tr>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Server</b><br><sub>Start backend cluster</sub></p>
      </td>
      <td style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Client</b><br><sub>Client code</sub></p>
      </td>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Infra</b><br><sub>Isolate ray and torchrun differences</sub></p>
      </td>
      <td style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Plugin</b><br><sub>Use hub components</sub></p>
      </td>
      <td bgcolor="#f6f8fa" style="border: 1px solid #d0d7de; border-radius: 8px; padding: 12px;">
        <p align="center"><b>Hub</b><br><sub>Interface with HF/MS libraries</sub></p>
      </td>
    </tr>
  </table>
</div>

## Twinkle's Customizable Components

In Twinkle's design, training using torchrun, Ray, and HTTP uses the same API and shares the same components and input/output structures. Therefore, many of its components can be customized by developers to implement new algorithm development.

Below is a list of recommended components for customization:

| Component Name        | Base Class                                 | Description                                                    |
| --------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| Loss                  | twinkle.loss.Loss                          | Used to define loss functions for model training               |
| Metric                | twinkle.metric.Metric                      | Used to define evaluation systems for model training           |
| Optimizer/LRScheduler | Based on PyTorch                           | Used to define optimizers and LR schedulers for model training |
| Patch                 | twinkle.patch.Patch                        | Used to fix issues during model training                       |
| Preprocessor          | twinkle.preprocessor.Preprocessor          | Used for data preprocessing (ETL) and returns standard format usable by Template |
| Filter                | twinkle.preprocessor.Filter                | Used to filter raw data for reasonableness                     |
| Task Data Processor   | twinkle.processor.InputProcessor           | Used to convert model inputs to data required by each task and add extra fields |
| Model                 | twinkle.model.TwinkleModel                 | The large model itself                                         |
| Sampler               | twinkle.sampler.Sampler                    | Sampler, e.g., vLLM                                            |
| Reward                | twinkle.reward.Reward                      | Used to implement rewards for different RL training            |
| Advantage             | twinkle.advantage.Advantage                | Used to implement advantage estimation for different RL training |
| Template              | twinkle.template.Template                  | Used to process standard inputs and convert them to tokens required by the model |
| Weight Synchronization | twinkle.checkpoint_engine.CheckpointEngine | Used for weight synchronization in RL training                 |

> Components not listed in the above table, such as Dataset, DataLoader, etc., can also be customized, just follow the base class API design.

## DeviceGroup and DeviceMesh

DeviceGroup and DeviceMesh are the core of Twinkle's architecture. All code construction is based on these two designs.

```python
import twinkle
from twinkle import DeviceMesh, DeviceGroup
device_group = [
        DeviceGroup(
            name='default',
            ranks=8,
            device_type='cuda',
        )
    ]

device_mesh = DeviceMesh.from_sizes(pp_size=2, tp_size=2, dp_size=2)
twinkle.initialize(mode='ray', nproc_per_node=8, groups=device_group)
```

After defining the device_group, you need to use `twinkle.initialize` to initialize resources.

DeviceGroup: Define how many resource groups are needed for this training session. Once defined, components can run themselves remotely by selecting resource groups:

```python
from twinkle.model import TransformersModel
model = TransformersModel(model_id='Qwen/Qwen3-4B', remote_group='default', device_mesh=device_mesh)
# Or
from twinkle.model import MegatronModel
model = MegatronModel(model_id='Qwen/Qwen3-4B', remote_group='default', device_mesh=device_mesh)
```

DeviceMesh specifies the topology of components like models within the resource group. It can be understood as how to perform parallelization. This affects a series of framework decisions, such as data acquisition, data consumption, data return, etc.

## Usage Example

```python
from peft import LoraConfig
import twinkle
from twinkle import DeviceMesh, DeviceGroup
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

device_group = [DeviceGroup(name='default',ranks=8,device_type='cuda')]
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
# local for torchrun
twinkle.initialize(mode='ray', groups=device_group, global_device_mesh=device_mesh)


def train():
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id='Qwen/Qwen3-4B')
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle LLM', 'ModelScope Community'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8, min_batch_size=8)
    # Use a TransformersModel
    model = TransformersModel(model_id='Qwen/Qwen3-4B', remote_group='default')

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules='all-linear'
    )

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5,
                           num_training_steps=len(dataloader))
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % 20 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            print(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
```

Start training like this:

```shell
python3 train.py
```

## Supported Large Language Models List

| Model Type          | Model ID Example                                                                                                              | Requires             | Support Megatron | HF Model ID                                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------|----------------------|------------------|---------------------------------------------------------------------------------------------------------------|
| qwen2 series        | [Qwen/Qwen2-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-0.5B-Instruct)                                             | transformers>=4.37   | ✔                | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                                   |
|                     | [Qwen/Qwen2-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-72B-Instruct)                                               | transformers>=4.37   | ✔                | [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)                                     |
|                     | [Qwen/Qwen2-1.5B](https://modelscope.cn/models/Qwen/Qwen2-1.5B)                                                               | transformers>=4.37   | ✔                | [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)                                                     |
|                     | [Qwen/Qwen2-7B](https://modelscope.cn/models/Qwen/Qwen2-7B)                                                                   | transformers>=4.37   | ✔                | [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)                                                         |
|                     | [Qwen/Qwen2-72B](https://modelscope.cn/models/Qwen/Qwen2-72B)                                                                 | transformers>=4.37   | ✔                | [Qwen/Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B)                                                       |
|                     | [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)                                         | transformers>=4.37   | ✔                | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                               |
|                     | [Qwen/Qwen2.5-1.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct)                                         | transformers>=4.37   | ✔                | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                               |
|                     | [Qwen/Qwen2.5-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct)                                           | transformers>=4.37   | ✔                | [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)                                 |
|                     | [Qwen/Qwen2.5-0.5B](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B)                                                           | transformers>=4.37   | ✔                | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)                                                 |
|                     | [Qwen/Qwen2.5-32B](https://modelscope.cn/models/Qwen/Qwen2.5-32B)                                                             | transformers>=4.37   | ✔                | [Qwen/Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)                                                   |
| qwen2_moe series    | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://modelscope.cn/models/Qwen/Qwen1.5-MoE-A2.7B-Chat)                                       | transformers>=4.40   | ✔                | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)                             |
|                     | [Qwen/Qwen1.5-MoE-A2.7B](https://modelscope.cn/models/Qwen/Qwen1.5-MoE-A2.7B)                                                 | transformers>=4.40   | ✔                | [Qwen/Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)                                       |
| qwen3 series        | [Qwen/Qwen3-0.6B-Base](https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base)                                                     | transformers>=4.51   | ✔                | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)                                           |
|                     | [Qwen/Qwen3-14B-Base](https://modelscope.cn/models/Qwen/Qwen3-14B-Base)                                                       | transformers>=4.51   | ✔                | [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base)                                             |
|                     | [Qwen/Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B)                                                               | transformers>=4.51   | ✔                | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                                                     |
|                     | [Qwen/Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B)                                                               | transformers>=4.51   | ✔                | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)                                                     |
|                     | [Qwen/Qwen3-32B](https://modelscope.cn/models/Qwen/Qwen2.5-32B)                                                               | transformers>=4.51   | ✔                | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)                                                       |
| qwen3_moe series    | [Qwen/Qwen3-30B-A3B-Base](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Base)                                               | transformers>=4.51   | ✔                | [Qwen/Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)                                     |
|                     | [Qwen/Qwen3-30B-A3B](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B)                                                         | transformers>=4.51   | ✔                | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)                                               |
|                     | [Qwen/Qwen3-235B-A22B](https://modelscope.cn/models/Qwen/Qwen3-235B-A22B)                                                     | transformers>=4.51   | ✔                | [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)                                           |
| chatglm2 series     | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b)                                                       | transformers<4.42    | ✘                | [zai-org/chatglm2-6b](https://huggingface.co/zai-org/chatglm2-6b)                                             |
|                     | [ZhipuAI/chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k)                                               | transformers<4.42    | ✘                | [zai-org/chatglm2-6b-32k](https://huggingface.co/zai-org/chatglm2-6b-32k)                                     |
| chatglm3 series     | [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b)                                                       | transformers<4.42    | ✘                | [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b)                                             |
|                     | [ZhipuAI/chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base)                                             | transformers<4.42    | ✘                | [zai-org/chatglm3-6b-base](https://huggingface.co/zai-org/chatglm3-6b-base)                                   |
|                     | [ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k)                                               | transformers<4.42    | ✘                | [zai-org/chatglm3-6b-32k](https://huggingface.co/zai-org/chatglm3-6b-32k)                                     |
|                     | [ZhipuAI/chatglm3-6b-128k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-128k)                                             | transformers<4.42    | ✘                | [zai-org/chatglm3-6b-128k](https://huggingface.co/zai-org/chatglm3-6b-128k)                                   |
| chatglm4 series     | [ZhipuAI/glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)                                                   | transformers>=4.42   | ✘                | [zai-org/glm-4-9b-chat](https://huggingface.co/zai-org/glm-4-9b-chat)                                         |
|                     | [ZhipuAI/glm-4-9b](https://modelscope.cn/models/ZhipuAI/glm-4-9b)                                                             | transformers>=4.42   | ✘                | [zai-org/glm-4-9b](https://huggingface.co/zai-org/glm-4-9b)                                                   |
|                     | [ZhipuAI/glm-4-9b-chat-1m](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)                                             | transformers>=4.42   | ✘                | [zai-org/glm-4-9b-chat-1m](https://huggingface.co/zai-org/glm-4-9b-chat-1m)                                   |
|                     | [ZhipuAI/LongWriter-glm4-9b](https://modelscope.cn/models/ZhipuAI/LongWriter-glm4-9b)                                         | transformers>=4.42   | ✘                | [zai-org/LongWriter-glm4-9b](https://huggingface.co/zai-org/LongWriter-glm4-9b)                               |
| glm_edge series     | [ZhipuAI/glm-edge-1.5b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat)                                         | transformers>=4.46   | ✘                | [zai-org/glm-edge-1.5b-chat](https://huggingface.co/zai-org/glm-edge-1.5b-chat)                               |
|                     | [ZhipuAI/glm-edge-4b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-4b-chat)                                             | transformers>=4.46   | ✘                | [zai-org/glm-edge-4b-chat](https://huggingface.co/zai-org/glm-edge-4b-chat)                                   |
| internlm2 series    | [Shanghai_AI_Laboratory/internlm2-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b)                   | transformers>=4.38   | ✘                | [internlm/internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)                                     |
|                     | [Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft) | transformers>=4.38   | ✘                | [internlm/internlm2-chat-1_8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft)                   |
|                     | [Shanghai_AI_Laboratory/internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b)             | transformers>=4.38   | ✘                | [internlm/internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b)                               |
|                     | [Shanghai_AI_Laboratory/internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b)                       | transformers>=4.38   | ✘                | [internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b)                                         |
|                     | [Shanghai_AI_Laboratory/internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b)             | transformers>=4.38   | ✘                | [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)                               |
| deepseek_v1         | [deepseek-ai/deepseek-vl-7b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat)                               | transformers>=4.39.4 | ✔                |                                                                                                               |
|                     | [deepseek-ai/DeepSeek-V2-Lite](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite)                                     | transformers>=4.39.3 | ✔                | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)                           |
|                     | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite-Chat)                           | transformers>=4.39.3 | ✔                | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)                 |
|                     | [deepseek-ai/DeepSeek-V2](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2)                                               | transformers>=4.39.3 | ✔                | [deepseek-ai/DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)                                     |
|                     | [deepseek-ai/DeepSeek-V2-Chat](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Chat)                                     | transformers>=4.39.3 | ✔                | [deepseek-ai/DeepSeek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)                           |
|                     | [deepseek-ai/DeepSeek-V2.5](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2.5)                                           | transformers>=4.39.3 | ✔                | [deepseek-ai/DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)                                 |
|                     | [deepseek-ai/DeepSeek-Prover-V2-7B](https://modelscope.cn/models/deepseek-ai/DeepSeek-Prover-V2-7B)                           | transformers>=4.39.3 | ✔                | [deepseek-ai/DeepSeek-Prover-V2-7B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B)                 |
|                     | [deepseek-ai/DeepSeek-R1](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1)                                               | transformers>=4.39.3 | ✔                | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)                                     |
| deepSeek-r1-distill | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)           | transformers>=4.37   | ✔                | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
|                     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)               | transformers>=4.37   | ✔                | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)     |
|                     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)             | transformers>=4.37   | ✔                | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)   |
|                     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)             | transformers>=4.37   | ✔                | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)   |
