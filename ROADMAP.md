# 0.1版本release

## 中文

### 基础能力

- [x] 支持transformers模型
- [x] 支持megatron模型
- [x] 支持vLLM采样器
- [x] 支持dataset、dataloader、reward、advantage、权重同步等基本组件
- [x] 支持数据集packing、padding_free、流式数据集
- [x] 支持纯文本模型的PT/SFT
- [x] 支持纯文本模型的GRPO
- [x] 支持kernels
- [x] 兼容NPU生态

### 网络能力

- [x] 支持多LoRA租户
- [x] 支持twinkle client训练
- [x] 支持tinker API的兼容性
- [x] 支持租户资源控制、水位控制
- [x] 支持checkpoint的保存上传、下载
- [x] 支持魔搭免费训练集群

## English

### Core Capabilities

- [x] Support for Transformers models
- [x] Support for Megatron models
- [x] Support for vLLM sampler
- [x] Support for basic components including dataset, dataloader, reward, advantage, and weight synchronization
- [x] Support for dataset packing, padding-free, and streaming datasets
- [x] Support for PT/SFT of text-only models
- [x] Support for GRPO of text-only models
- [x] Support for kernels
- [x] Compatibility with NPU ecosystem

### Networking Capabilities

- [x] Support for multi-LoRA tenants
- [x] Support for Twinkle client training
- [x] Support for Tinker API compatibility
- [x] Support for tenant resource control and watermark control
- [x] Support for checkpoint saving, uploading, and downloading
- [x] Support for ModelScope free training cluster


# 0.2版本待开发

## 中文

### 基础能力

- [ ] 支持多模态模型
- [ ] 支持megatron VPP
- [ ] 支持liger kernel
- [ ] 支持transformers模型的ulysses/ring-attention
- [ ] 兼容transformers v5的tp、pp
- [ ] 支持多轮RL
- [ ] 支持gym训练
- [ ] 支持GAPO、GSPO算法
- [ ] 支持GKD、on-policy-distill等蒸馏算法
- [ ] 支持DPO对齐训练
- [ ] 支持colocate RL训练
- [ ] Preprocess支持batched
- [ ] 对多replica的支持和粘滞路由

### 网络能力

## English

### Core Capabilities

- [ ] Support for multimodal models
- [ ] Support for Megatron VPP
- [ ] Support for Liger kernel
- [ ] Support for Ulysses/Ring-Attention for Transformers models
- [ ] Compatibility with Transformers v5 TP and PP
- [ ] Support for multi-turn RL
- [ ] Support for Gym training
- [ ] Support for GAPO and GSPO algorithms
- [ ] Support for distillation algorithms such as GKD and on-policy distillation
- [ ] Support for DPO alignment training
- [ ] Support for colocate RL training
- [ ] Support for batched preprocessing
- [ ] Support for multiple replicas and sticky routing

### Networking Capabilities
