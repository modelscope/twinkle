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

## 使用样例

