<h1 align="center">Twinkle: <b>T</b>raining <b>W</b>orkbench for <b>I</b>ndustrial <b>N</b>eural-network <b>K</b>it & <b>L</b>LM <b>E</b>ngineering</h1>

<p align="center">
    <br>
    <img src="assets/slogan.png"/>
    <br>
<p>
<p align="center">
<a href="https://modelscope.cn/home">The Modelscope Community</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
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

<div align="center">

## ✨ Twinkle是什么？

大模型训练组件库。基于 PyTorch，更简洁、更灵活、生产就绪。

<p align="center">
🧩 <b>松耦合架构</b> · 标准化接口<br>
🚀 <b>多运行模式</b> · torchrun / Ray / HTTP<br>
🔌 <b>多框架兼容</b> · Transformers / Megatron<br>
👥 <b>多租户支持</b> · 单基座模型部署
</p>

</div>

## 安装

使用pip安装：

```shell
pip install 'twinkle-kit'
```

## 源代码安装

```shell
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e . --no-build-isolation
```

## 示例教程

| 训练类型                          | 模型框架     | cookbook地址                             |
| --------------------------------- | ------------ | ---------------------------------------- |
| FSDP finetuning                   | transformers | [脚本](cookbook/transformers/fsdp2.py)      |
| FSDP MoE finetuning               | transformers | [脚本](cookbook/transformers/fsdp2_moe.py)  |
| EP MoE finetuning                 | transformers | [脚本](cookbook/transformers/ep_fsdp_qwen3_moe.py) |
| pp/tp/cp finetuning               | megatron     | [脚本](cookbook/megatron/tp.py)             |
| pp/tp/cp MoE finetuning           | megatron     | [脚本](cookbook/megatron/tp_moe.py)         |
| tinker client finetuning          | megatron     | [脚本](cookbook/client/tinker/megatron)     |
| tinker client finetuning/sampling | transformers | [脚本](cookbook/client/tinker/transformer)  |
| twinkle client finetuning         | megatron     | [脚本](cookbook/client/twinkle/megatron)    |
| twinkle client finetuning         | transformer  | [脚本](cookbook/client/twinkle/transformer) |

## 更新日志

- 🎉2026-02-10 twinkle-kit第一版编写完成，包含纯文本模型SFT/PT/RL和远程训练能力，并支持了[魔搭官方免费资源]()

## 支持的硬件

| 硬件环境                    | 备注                              |
| --------------------------- | --------------------------------- |
| GPU A10/A100/H100/RTX系列等 |                                   |
| GPU T4/V100等               | 不支持bfloat16、Flash-Attention   |
| Ascend NPU                  | 部分算子不支持                    |
| PPU                         | 支持                              |
| CPU                         | 支持dataset、dataloader等部分组件 |

## 支持的大语言模型

| Model Type          | Model ID 举例                                                                                                          | Requires             | Support Megatron | HF Model ID                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------- |
| qwen2 全系列        | [Qwen/Qwen2-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-0.5B-Instruct) ～72B                                  | transformers>=4.37   | ✔               | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                                   |
|                     | [Qwen/Qwen2-72B](https://modelscope.cn/models/Qwen/Qwen2-72B)～72B                                                        | transformers>=4.37   | ✔               | [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)                                                     |
|                     | [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)～72B                                | transformers>=4.37   | ✔               | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                               |
|                     | [Qwen/Qwen2.5-0.5B](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B)～72B                                                  | transformers>=4.37   | ✔               | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)                                                 |
| qwen2_moe 全系列    | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://modelscope.cn/models/Qwen/Qwen1.5-MoE-A2.7B-Chat)                                   | transformers>=4.40   | ✔               | [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)                             |
| qwen3 全系列        | [Qwen/Qwen3-0.6B-Base](https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base)～32B                                            | transformers>=4.51   | ✔               | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)                                           |
| qwen3_moe 全系列    | [Qwen/Qwen3-30B-A3B-Base](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Base)                                           | transformers>=4.51   | ✔               | [Qwen/Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)                                     |
|                     | [Qwen/Qwen3-30B-A3B](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B)～235B                                               | transformers>=4.51   | ✔               | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)                                               |
| chatglm2 全系列     | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b)                                                   | transformers<4.42    | ✘               | [zai-org/chatglm2-6b](https://huggingface.co/zai-org/chatglm2-6b)                                             |
|                     | [ZhipuAI/chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k)                                           | transformers<4.42    | ✘               | [zai-org/chatglm2-6b-32k](https://huggingface.co/zai-org/chatglm2-6b-32k)                                     |
| chatglm3 全系列     | [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b)                                                   | transformers<4.42    | ✘               | [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b)                                             |
|                     | [ZhipuAI/chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base)                                         | transformers<4.42    | ✘               | [zai-org/chatglm3-6b-base](https://huggingface.co/zai-org/chatglm3-6b-base)                                   |
|                     | [ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k)~128k                                      | transformers<4.42    | ✘               | [zai-org/chatglm3-6b-32k](https://huggingface.co/zai-org/chatglm3-6b-32k)                                     |
| chatglm4 全系列     | [ZhipuAI/glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)                                               | transformers>=4.42   | ✘               | [zai-org/glm-4-9b-chat](https://huggingface.co/zai-org/glm-4-9b-chat)                                         |
|                     | [ZhipuAI/LongWriter-glm4-9b](https://modelscope.cn/models/ZhipuAI/LongWriter-glm4-9b)                                     | transformers>=4.42   | ✘               | [zai-org/LongWriter-glm4-9b](https://huggingface.co/zai-org/LongWriter-glm4-9b)                               |
| glm_edge 全系列     | [ZhipuAI/glm-edge-1.5b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat)                                     | transformers>=4.46   | ✘               | [zai-org/glm-edge-1.5b-chat](https://huggingface.co/zai-org/glm-edge-1.5b-chat)                               |
|                     | [ZhipuAI/glm-edge-4b-chat](https://modelscope.cn/models/ZhipuAI/glm-edge-4b-chat)                                         | transformers>=4.46   | ✘               | [zai-org/glm-edge-4b-chat](https://huggingface.co/zai-org/glm-edge-4b-chat)                                   |
| internlm2 全系列    | [Shanghai_AI_Laboratory/internlm2-1_8b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b)               | transformers>=4.38   | ✘               | [internlm/internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)                                     |
|                     | [Shanghai_AI_Laboratory/internlm2-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b)         | transformers>=4.38   | ✘               | [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)                               |
| deepseek_v1         | [deepseek-ai/deepseek-vl-7b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat)                           | transformers>=4.39.4 | ✔               | ——                                                                                                       |
|                     | [deepseek-ai/DeepSeek-V2-Lite](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite)                                 | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)                           |
|                     | [deepseek-ai/DeepSeek-V2.5](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2.5)                                       | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)                                 |
|                     | [deepseek-ai/DeepSeek-R1](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1)                                           | transformers>=4.39.3 | ✔               | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)                                     |
| deepSeek-r1-distill | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) ~32B | transformers>=4.37   | ✔               | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |

更详细的模型支持列表 👉  [快速开始.md](https://github.com/modelscope/twinkle/blob/dev/docs/source/%E4%BD%BF%E7%94%A8%E6%8C%87%E5%BC%95/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.md)

## 示例代码

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
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
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

这样启动训练:

```shell
python3 train.py
```

## 架构设计

<img src="assets/framework.jpg" style="max-width: 500px; width: 100%;" />

twinkle的架构由client和server两部分构成，其中client端包含两个使用可能性：

1. 符合twinkle调用API的客户端，其API和server端完全相同
2. 对原生Tinker API的兼容

这使得开发者可以直接使用Tinker API调用twinkle部署起来的后端训练服务。

## 多租户支持

Twinkle支持多个租户同时使用一个基模型进行训练。这一行为目前仅限于[LoRA](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py#L323)。
Twinkle采用了LoRA池+租户申请的技术方案。这个方案可以支持最大N个租户并行训练互不干扰，并且在模型角度来看，不同租户的训练流程可能不同，在基模中的数据padding方式、optimizer、Loss类型也可以不同。

<img src="assets/multi_lora.png" style="max-width: 500px; width: 100%;" />

例如：

- 租户A：本机加载本地私有数据集，loRA rank=8，使用基模进行SFT
- 租户B：使用远端加载Hub端开源数据集，LoRA rank=32，使用基模进行PT
- 租户C：使用基模进行GRPO Loss计算，使用Sampler采样
- 租户D：使用基模进行logps推理

这些过程可以同时发生在一个基模上，因为模型、Sampler本质上也是twinkle组件的一部分，可以做到任务无关。训练完成后，支持checkpoint推送HuggingFace/ModelScope的模型仓库，默认为私有。twinkle提供了完整的多租户训练解决方案，在server端支持集群化管理和动态扩缩容，可以进行简单定制化后作为企业级服务。

> 作为模块化框架，twinkle本身也可以支持远端临时的独占式训练，即全参数方式。


## 支持的组件

<table>
  <tr>
    <td align="center"><b>Dataset</b><br><sub>数据加载和预处理</sub></td>
    <td align="center"><b>Template</b><br><sub>编码和解码</sub></td>
    <td align="center"><b>DataLoader</b><br><sub>数据分发和batch化</sub></td>
    <td align="center"><b>Preprocessor</b><br><sub>数据ETL</sub></td>
    <td align="center"><b>InputProcessor</b><br><sub>处理任务特定输入</sub></td>
  </tr>
  <tr>
    <td align="center"><b>Model</b><br><sub>大模型，支持多种框架</sub></td>
    <td align="center"><b>Sampler</b><br><sub>采样器</sub></td>
    <td align="center"><b>Loss</b><br><sub>残差</sub></td>
    <td align="center"><b>Metric</b><br><sub>训练指标集合</sub></td>
    <td align="center"><b>Reward</b><br><sub>奖励函数</sub></td>
  </tr>
  <tr>
    <td align="center"><b>Advantage</b><br><sub>优势函数</sub></td>
    <td align="center"><b>CheckpointEngine</b><br><sub>权重同步</sub></td>
    <td align="center"><b>Patch</b><br><sub>补丁，用于模型修复</sub></td>
    <td align="center"><b>Module</b><br><sub>组件，例如Optimizer</sub></td>
    <td align="center"><b>Kernel</b><br><sub>算子</sub></td>
  </tr>
  <tr>
    <td align="center"><b>Server</b><br><sub>开启后端集群</sub></td>
    <td align="center"><b>Client</b><br><sub>客户端代码</sub></td>
    <td align="center"><b>Infra</b><br><sub>隔离ray和torchrun差异</sub></td>
    <td align="center"><b>Plugin</b><br><sub>使用hub端组件</sub></td>
    <td align="center"><b>Hub</b><br><sub>对接HF/MS网络库</sub></td>
  </tr>
</table>

## 社区组件

| 组件类型 | 组件链接                                                                                                 | 组件作用                                                          | 作者           |
| -------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | -------------- |
| Patch    | [qwen3_moe_transformers4_patch](https://www.modelscope.cn/models/twinkle-kit/qwen3_moe_transformers4_patch) | 修复Qwen3 MoE模型在FSDP2训练时Hang的问题，对transformers==4.x生效 | ModelScope官方 |
