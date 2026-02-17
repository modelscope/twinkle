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
model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', remote_group='default', device_mesh=device_mesh)
# Or
from twinkle.model import MegatronModel
model = MegatronModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', remote_group='default', device_mesh=device_mesh)
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
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle LLM', 'ModelScope Community'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8, min_batch_size=8)
    # Use a TransformersModel
    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', remote_group='default')

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

| Model Type          | Model ID Example                                                                                           | Requires             | Support Megatron | HF Model ID                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------- |
| qwen2 series        | [Qwen/Qwen2-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-0.5B-Instruct)                             | transformers>=4.37   | ✔               | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                                   |
|                     | [Qwen/Qwen2-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-72B-Instruct)                               | transformers>=4.37   | ✔               | [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)                                     |
|                     | [Qwen/Qwen2-1.5B](https://modelscope.cn/models/Qwen/Qwen2-1.5B)                                               | transformers>=4.37   | ✔               | [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)                                                     |
|                     | [Qwen/Qwen2-7B](https://modelscope.cn/models/Qwen/Qwen2-7B)                                                   | transformers>=4.37   | ✔               | [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)                                                         |
|                     | [Qwen/Qwen2-72B](https://modelscope.cn/models/Qwen/Qwen2-72B)                                                 | transformers>=4.37   | ✔               | [Qwen/Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B)                                                       |
|                     | [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)                         | transformers>=4.37   | ✔               | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                               |
|                     | [Qwen/Qwen2.5-1.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct)                         | transformers>=4.37   | ✔               | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                               |
|                     | [Qwen/Qwen2.5-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct)                           | transformers>=4.37   | ✔               | [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)                                 |
|                     | [Qwen/Qwen2.5-0.5B](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B)                                           | transformers>=4.37   | ✔               | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)                                                 |
|                     | [Qwen/Qwen2.5-32B](https://modelscope.cn/models/Qwen/Qwen2.5-32B)                                             | transformers>=4.37   | ✔               | [Qwen/Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)                                                   |
| qwen2_moe series    | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://modelscope.cn/models/Qwen/Qwen1.5-MoE-A2.7B-Chat)                       | transformers>=4.40   | ✔               | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)                             |
|                     | [Qwen/Qwen1.5-MoE-A2.7B](https://modelscope.cn/models/Qwen/Qwen1.5-MoE-A2.7B)                                 | transformers>=4.40   | ✔               | [Qwen/Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)                                       |
| qwen3 series        | [Qwen/Qwen3-0.6B-Base](https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base)                                     | transformers>=4.51   | ✔               | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)                                           |
|                     | [Qwen/Qwen3-14B-Base](https://modelscope.cn/models/Qwen/Qwen3-14B-Base)                                       | transformers>=4.51   | ✔               | [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base)                                             |
|                     | [Qwen/Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B)                                               | transformers>=4.51   | ✔               | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                                                     |
|                     | [Qwen/Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B)                                               | transformers>=4.51   | ✔               | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)                                                     |
|                     | [Qwen/Qwen3-32B](https://modelscope.cn/models/Qwen/Qwen2.5-32B)                                               | transformers>=4.51   | ✔               | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)                                                       |
| qwen3_moe series    | [Qwen/Qwen3-30B-A3B-Base](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Base)                               | transformers>=4.51   | ✔               | [Qwen/Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)                                     |
|                     | [Qwen/Qwen3-30B-A3B](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B)                                         | transformers>=4.51   | ✔               | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)                                               |
|                     | [Qwen/Qwen3-235B-A22B](https://modelscope.cn/models/Qwen/Qwen3-235B-A22B)                                     | transformers>=4.51   | ✔               | [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)                                           |
| chatglm2 series     | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b)                                       | transformers<4.42    | ✘               | [zai-org/chatglm2-6b](https://huggingface.co/zai-org/chatglm2-6b)                                             |
|                     | [ZhipuAI/chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k)                               | transformers<4.42    | ✘               | [zai-org/chatglm2-6b-32k](https://huggingface.co/zai-org/chatglm2-6b-32k)                                     |
| chatglm3 series     | [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b)                                       | transformers<4.42    | ✘               | [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b)                                             |
|                     | [ZhipuAI/chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base)                             | transformers<4.42    | ✘               | [zai-org/chatglm3-6b-base](https://huggingface.co/zai-org/chatglm3-6b-base)                                   |
|                     | [ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k)                               | transformers<4.42    | ✘               | [zai-org/chatglm3-6b-32k](https://huggingface.co/zai-org/chatglm3-6b-32k)                                     |
|                     | [ZhipuAI/chatglm3-6b-128k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-128k)                             | transformers<4.42    | ✘               | [zai-org/chatglm3-6b-128k](https://huggingface.co/zai-org/chatglm3-6b-128k)                                   |
| chatglm4 series     | [ZhipuAI/glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)                                   | transformers>=4.42   | ✘               | [zai-org/glm-4-9b-chat](https://huggingface.co/zai-org/glm-4-9b-chat)                                         |
|                     | [ZhipuAI/glm-4-9b](https://modelscope.cn/models/ZhipuAI/glm-4-9b)                                             | transformers>=4.42   | ✘               | [zai-org/glm-4-9b](https://huggingface.co/zai-org/glm-4-9b)                                                   |
|                     | [ZhipuAI/glm-4-9b-chat-1m](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)                             | transformers>=4.42   | ✘               | [zai-org/glm-4-9b-chat-1m](https://huggingface.co/zai-org/glm-4-9b-chat-1m)                                   |
|                     | [ZhipuAI/LongWriter-glm4-9b](https://modelscope.cn/models/ZhipuAI/LongWriter-glm4-9b)                         | transformers>=4.42   | ✘               | [zai-org/LongWriter-glm4-9b](https://huggingface.co/zai-org/LongWriter-glm4-9b)                               |
| glm_edge series     | [ZhipuAI/glm-edge-1.5b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat)                         | transformers>=4.46   | ✘               | [zai-org/glm-edge-1.5b-chat](https://huggingface.co/zai-org/glm-edge-1.5b-chat)                               |
|                     | [ZhipuAI/glm-edge-4b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-4b-chat)                             | transformers>=4.46   | ✘               | [zai-org/glm-edge-4b-chat](https://huggingface.co/zai-org/glm-edge-4b-chat)                                   |
| internlm2 series    | [Shanghai_AI_Laboratory/internlm2-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b)                   | transformers>=4.38   | ✘               | [internlm/internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)                                     |
|                     | [Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft) | transformers>=4.38   | ✘               | [internlm/internlm2-chat-1_8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft)                   |
|                     | [Shanghai_AI_Laboratory/internlm2-base-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b)             | transformers>=4.38   | ✘               | [internlm/internlm2-base-7b](https://huggingface.co/internlm/internlm2-base-7b)                               |
|                     | [Shanghai_AI_Laboratory/internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b)                       | transformers>=4.38   | ✘               | [internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b)                                         |
|                     | [Shanghai_AI_Laboratory/internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b)             | transformers>=4.38   | ✘               | [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)                               |
| deepseek_v1         | [deepseek-ai/deepseek-vl-7b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat)                               | transformers>=4.39.4 | ✔               |                                                                                                            |
|                     | [deepseek-ai/DeepSeek-V2-Lite](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite)                                     | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)                           |
|                     | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite-Chat)                           | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)                 |
|                     | [deepseek-ai/DeepSeek-V2](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2)                                               | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)                                     |
|                     | [deepseek-ai/DeepSeek-V2-Chat](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Chat)                                     | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)                           |
|                     | [deepseek-ai/DeepSeek-V2.5](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2.5)                                           | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)                                 |
|                     | [deepseek-ai/DeepSeek-Prover-V2-7B](https://modelscope.cn/models/deepseek-ai/DeepSeek-Prover-V2-7B)                           | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-Prover-V2-7B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B)                 |
|                     | [deepseek-ai/DeepSeek-R1](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1)                                               | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)                                     |
| deepSeek-r1-distill | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)           | transformers>=4.37   | ✔               | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
|                     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)               | transformers>=4.37   | ✔               | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)     |
|                     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)             | transformers>=4.37   | ✔               | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)   |
|                     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)             | transformers>=4.37   | ✔               | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)   |
