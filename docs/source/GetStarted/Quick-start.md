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

## twinkle适配性

twinkle和[ms-swift](https://github.com/modelscope/ms-swift)都是模型训练框架，但二者的特性有很大不同，开发者可以根据自己的需求选择。

### 何时选择twinkle

- 如果你是大模型的初学者，希望更好地了解模型机制和模型训练方法
- 如果你是大模型研究者，希望定制模型或者训练方法
- 如果你善于编写training loop，希望定制训练过程
- 如果你是B端，希望提供商业化训练平台

### 何时选择ms-swift

- 如果你不关心训练过程，希望仅提供数据集便可完成训练
- 如果你需要更多的模型支持和数据集种类
- 如果你需要推理、部署、量化等其他能力
- 如果你对新模型的训练支持敏感，swift会保证day-0的更新能力

## twinkle的可定制组件

在twinkle的设计中，torchrun、ray、http的训练使用同样的API，并分享相同的组件和输入输出结构。因此其很多组件可以由开发者自定义来实现新的算法开发。

下面我们列出推荐定制的组件列表：

| 组件名称                  | 基类                                 | 说明                                 |
|-----------------------|------------------------------------|------------------------------------|
| 损失                    | twinkle.loss.Loss                  | 用于定义模型训练后的损失函数                     |
| 指标                    | twinkle.metric.Metric              | 用于定义模型训练的评价体系                      |
| Optimizer/LRScheduler | 基于PyTorch                          | 用于定义模型训练的优化器和LR衰减器                 | 
| 补丁                    | twinkle.patch.Patch                | 用于修复模型训练过程的补丁                      |
| 预处理器                  | twinkle.preprocessor.Preprocessor  | 用于对数据进行预处理(ETL)，并返回template可用的标准格式 |
| 过滤器                   | twinkle.preprocessor.Filter        | 用于对原始数据进行合理性过滤                     |
| 任务数据处理器               | twinkle.processor.InputProcessor   | 用于对模型输入转为各任务需要的数据，并添加额外字段          |
| 模型                    | twinkle.model.TwinkleModel         | 大模型本身                              |
| 采样器                   | twinkle.sampler.Sampler            | 采样器，例如vLLM                         |
| 奖励                    | twinkle.reward.Reward              | 用于实现不同RL训练的奖励                      |
| 优势                    | twinkle.advantage.Advantage        | 用于实现不同RL训练的优势估计                    |
| 模板                    | twinkle.template.Template          | 用于处理标准输入，并转换成模型需要的token            |
| 权重同步                  | twinkle.weight_loader.WeightLoader | 用于RL训练中的权重同步                       |

> 未在上表中列出的组件，如Dataset、DataLoader等也可以实现定制，只需要跟随基类API设计即可。

## DeviceGroup和DeviceMesh

DeviceGroup和DeviceMesh是twinkle架构的核心。所有的代码构建均基于这两个设计。

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

当device_group定义完成后，需要使用`twinkle.initialize`来初始化资源。

DeviceGroup：定义本次训练需要多少个资源组。定义后，组件可以通过选择资源组的方式将自己运行在远端：

```python
from twinkle.model import TransformersModel
model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', remote_group='default', device_mesh=device_mesh)
# 或者
from twinkle.model import MegatronModel
model = MegatronModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', remote_group='default', device_mesh=device_mesh)
```

DeviceMesh给出了模型等组件在资源组中的构型。可以理解为如何进行并行。这会影响一系列的框架决策，例如取数据、消费数据、数据返回等。

## 使用样例

