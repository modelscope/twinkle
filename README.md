<h1 align="center">Twinkle: Training workbench to make your model glow</h1>

<p align="center">
    <img src="assets/slogan.png" width="200"/>
<p>
<p align="center">
by <a href="https://modelscope.cn/home">ModelScope</a>
<br>
        English&nbsp ｜ &nbsp<a href="README_ZH.md">中文</a>&nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.11-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://pypi.org/project/twinkle/"><img src="https://badge.fury.io/py/twinkle.svg"></a>
<a href="https://github.com/modelscope/twinkle/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/twinkle"></a>
<a href="https://pepy.tech/project/twinkle-kit"><img src="https://pepy.tech/badge/twinkle-kit"></a>
<a href="https://github.com/modelscope/twinkle/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

<p align="center">
        <a href="https://twinkle-kit.readthedocs.io/en/latest/">English Documentation</a> &nbsp ｜ &nbsp <a href="https://twinkle-kit.readthedocs.io/zh-cn/latest/">中文文档</a> &nbsp
</p>

## ✨ What is Twinkle?
Twinkle✨ is a lightweight, client-server training framework engineered 
with modular, high-cohesion interfaces. Whether you are executing locally 
with `torchrun`, or scaling training across Ray clusters,
Twinkle✨ eliminates infrastructure friction by encapsulating 
complex training logic into standardized APIs. Beyond simple 
abstraction, Twinkle✨ serves as a robust backend and gateway to enable serverless Training-as-a-Service (TaaS).
It offers interfaces that constitute a _superset_ of  [Tinker](https://thinkingmachines.ai/tinker/) APIs, 
thereby making is possible to access a Twinkle✨ training service via Tinker client or native Twinkle✨-client
which offers more functionalities.

🧩 <b>Decoupled Architecture</b>: Standardized Interfaces, backward compatible with Tinker APIs.<br>
🚀 <b>Multiple Runtime Modes</b>: torchrun / Ray / HTTP.<br>
🔌 <b>Multi-Training Backends</b>: Transformers / Megatron.<br>
👥 <b>Multi-Tenancy Training Service</b>: Train multiple LoRAs that share one base model deployment.<br>

Note: Twinkle✨is built by the team behind [ms-swift](https://github.com/modelscope/ms-swift), and
we expect the two projects to evolve together. We expect some fundamental components in Twinkle✨will likely
be reused in [ms-swift](https://github.com/modelscope/ms-swift).

## Installation

### Install with package:

```shell
pip install 'twinkle-kit'
```

### Installation from Source:

```shell
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e . --no-build-isolation
```

## Tutorials

| Training Type                     | Model Framework | Cookbook Path                            |
| --------------------------------- | --------------- | ---------------------------------------- |
| FSDP finetuning                   | transformers    | [Script](cookbook/transformers/fsdp2.py)      |
| FSDP MoE finetuning               | transformers    | [Script](cookbook/transformers/fsdp2_moe.py)  |
| EP MoE finetuning                 | transformers    | [Script](cookbook/transformers/ep_fsdp_qwen3_moe.py) |
| pp/tp/cp finetuning               | megatron        | [Script](cookbook/megatron/tp.py)             |
| pp/tp/cp MoE finetuning           | megatron        | [Script](cookbook/megatron/tp_moe.py)         |
| tinker client finetuning          | megatron        | [Script](cookbook/client/tinker/megatron)     |
| tinker client finetuning/sampling | transformers    | [Script](cookbook/client/tinker/transformer)  |
| twinkle client finetuning         | megatron        | [Script](cookbook/client/twinkle/megatron)    |
| twinkle client finetuning         | transformer     | [Script](cookbook/client/twinkle/transformer) |

## Changelog

- 🎉2026-02-10 Initial version of Twinkle✨ released, including SFT/PT/RL for text models and serverless training capabilities on [ModelScope](https://modelscope.cn).

## Supported Hardware

| Hardware Environment | Notes                                                           |
|----------------------|-----------------------------------------------------------------|
| Nvidia GPUs          | ✅ Support for BF16/Flash-Attn may be incomplete in earlier GPUs |
| Ascend NPU           | ✅ Some operators may not supported                              |
| PPU                  | ✅                                                               |
| CPU                  | Supports partial components like dataset, dataloader            |

## Supported Models
We will be adding support for more models as new models are released. The following table lists current models
supported on Twinkle✨ framework. However, the models supported on our serverless training backend may be a
much smaller subset. Please refer to the [doc](link) section for more information.

| Model Type          | Model ID on [ModelScope](https://modelscope.cn)                                                                                               | Requires             | Megatron Support | HF Model ID                                                                                                |
| ------------------- |--------------------------------------------------------------------------------------------------------------------------| -------------------- |-----------| ---------------------------------------------------------------------------------------------------------- |
| qwen3 series        | [Qwen/Qwen3-0.6B-Base](https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base)~32B                                            | transformers>=4.51   | ✅         | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)                                           |
| qwen3_moe series    | [Qwen/Qwen3-30B-A3B-Base](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Base)                                          | transformers>=4.51   | ✅         | [Qwen/Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)                                     |
|                     | [Qwen/Qwen3-30B-A3B](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B)~235B                                               | transformers>=4.51   | ✅         | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)                                               |
| qwen2 series        | [Qwen/Qwen2-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-0.5B-Instruct) ~72B                                   | transformers>=4.37   | ✅         | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                                   |
|                     | [Qwen/Qwen2-72B](https://modelscope.cn/models/Qwen/Qwen2-72B)~72B                                                        | transformers>=4.37   | ✅         | [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)                                                     |
|                     | [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)~72B                                | transformers>=4.37   | ✅         | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                               |
|                     | [Qwen/Qwen2.5-0.5B](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B)~72B                                                  | transformers>=4.37   | ✅         | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)                                                 |
| qwen2_moe series    | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://modelscope.cn/models/Qwen/Qwen1.5-MoE-A2.7B-Chat)                                  | transformers>=4.40   | ✅         | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)                             |
| chatglm2 series     | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b)                                                  | transformers<4.42    | ✘         | [zai-org/chatglm2-6b](https://huggingface.co/zai-org/chatglm2-6b)                                             |
|                     | [ZhipuAI/chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k)                                          | transformers<4.42    | ✘         | [zai-org/chatglm2-6b-32k](https://huggingface.co/zai-org/chatglm2-6b-32k)                                     |
| chatglm3 series     | [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b)                                                  | transformers<4.42    | ✘         | [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b)                                             |
|                     | [ZhipuAI/chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base)                                        | transformers<4.42    | ✘         | [zai-org/chatglm3-6b-base](https://huggingface.co/zai-org/chatglm3-6b-base)                                   |
|                     | [ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k)~128k                                     | transformers<4.42    | ✘         | [zai-org/chatglm3-6b-32k](https://huggingface.co/zai-org/chatglm3-6b-32k)                                     |
| chatglm4 series     | [ZhipuAI/glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)                                              | transformers>=4.42   | ✘         | [zai-org/glm-4-9b-chat](https://huggingface.co/zai-org/glm-4-9b-chat)                                         |
|                     | [ZhipuAI/LongWriter-glm4-9b](https://modelscope.cn/models/ZhipuAI/LongWriter-glm4-9b)                                    | transformers>=4.42   | ✘         | [zai-org/LongWriter-glm4-9b](https://huggingface.co/zai-org/LongWriter-glm4-9b)                               |
| glm_edge series     | [ZhipuAI/glm-edge-1.5b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat)                                    | transformers>=4.46   | ✘         | [zai-org/glm-edge-1.5b-chat](https://huggingface.co/zai-org/glm-edge-1.5b-chat)                               |
|                     | [ZhipuAI/glm-edge-4b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-4b-chat)                                        | transformers>=4.46   | ✘         | [zai-org/glm-edge-4b-chat](https://huggingface.co/zai-org/glm-edge-4b-chat)                                   |
| internlm2 series    | [Shanghai_AI_Laboratory/internlm2-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b)              | transformers>=4.38   | ✘         | [internlm/internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)                                     |
|                     | [Shanghai_AI_Laboratory/internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b)        | transformers>=4.38   | ✘         | [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)                               |
| deepseek_v1         | [deepseek-ai/deepseek-vl-7b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat)                          | transformers>=4.39.4 | ✔         | ——                                                                                                       |
|                     | [deepseek-ai/DeepSeek-V2-Lite](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite)                                | transformers>=4.39.3 | ✔         | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)                           |
|                     | [deepseek-ai/DeepSeek-V2.5](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2.5)                                      | transformers>=4.39.3 | ✔         | [deepseek-ai/DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)                                 |
|                     | [deepseek-ai/DeepSeek-R1](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1)                                          | transformers>=4.39.3 | ✔         | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)                                     |
| deepSeek-r1-distill | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) ~32B | transformers>=4.37   | ✔         | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |

For a more detailed model support list 👉  [Quick Start.md](https://github.com/modelscope/twinkle/blob/dev/docs/source/%E4%BD%BF%E7%94%A8%E6%8C%87%E5%BC%95/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.md)

## Sample Code

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

Launch training:

```shell
python3 train.py
```

## Architecture Design

<img src="assets/framework.jpg" style="max-width: 500px; width: 100%;" />

 **Twinkle✨** features a decoupled **Client-Server architecture** designed for maximum flexibility. 
 The client-side provides two distinct integration paths:
 * **Twinkle✨ Native:** A conforming API that mirrors the server-side interface for seamless end-to-end integration.
 * **Tinker Compatibility:** Full support for the native Tinker API, enabling developers to leverage Twinkle✨’s backend using Tinker client.

This dual-path design ensures access to Twinkle✨’s training services using Tinker API, with a simple modification of the Tinker base URL.

## Multi-Tenancy

**Twinkle✨** supports simultaneous multi-tenant training on a shared base model. Leveraging a **LoRA Pool + Tenant Application** architecture, Twinkle enables up to **N tenants** to train in parallel with complete isolation. This design offers unprecedented flexibility: from the model's perspective, each tenant's session is distinct, supporting heterogeneous configurations including unique **data padding strategies, optimizers, and loss functions**—all running concurrently on the same base model. 

*Note: This feature is currently optimized for [LoRA](https://github.com/huggingface/peft).*

<img src="assets/multi_lora.png" style="max-width: 500px; width: 100%;" />

For example:

- Tenant A: Load local private dataset locally, LoRA rank=8, using base model for SFT
- Tenant B: Load open-source dataset from Hub remotely, LoRA rank=32, using base model for PT
- Tenant C: Use base model for GRPO loss calculation, using Sampler for sampling
- Tenant D: Use base model for logps inference

These processes are executed concurrently on a single base model because the **Model and Sampler** 
are integrated as **task-agnostic components** within the Twinkle✨ ecosystem. 
Upon completion, checkpoints are automatically pushed to **ModelScope** or **HuggingFace**  repositories 
(private by default). On the server side, Twinkle✨  provides a robust multi-tenant suite 
featuring **automated cluster management** and **dynamic scaling**, making it the 
foundation for building customizable, enterprise-grade training services.

> As a modular framework, Twinkle✨ also supports remote temporary exclusive training, i.e., training in full-parameter mode.


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

## Community Components

| Component Type | Component Link                                                                                           | Component Function                                                                  | Author           |
| -------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------- |
| Patch          | [qwen3_moe_transformers4_patch](https://www.modelscope.cn/models/twinkle-kit/qwen3_moe_transformers4_patch) | Fixes Qwen3 MoE model hang issue during FSDP2 training, effective for transformers==4.x | ModelScope Official |

