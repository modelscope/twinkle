# DSV4 EP Multi-LoRA Target Parameters 设计

## 背景

DeepSeek-V4 的 EP LoRA 训练目前使用单个 PEFT adapter，并通过
`LoraConfig.target_parameters` 训练 packed MoE expert 参数：

- `mlp.experts.gate_up_proj`
- `mlp.experts.down_proj`

这些 packed expert 权重是裸 `nn.Parameter`。保留这种结构可以维持
DSV4 原生 expert 布局，也能继续兼容 Twinkle 当前的 EP/FSDP 训练路径。

Twinkle 已有 `MultiLoraTransformersModel` 和 `MultiLora`，可以在同一个
模型中常驻多个 tenant adapter。它的做法是预分配物理 adapter slot
（`lora_0`、`lora_1` 等），再把用户侧 adapter 名映射到这些 slot。
每次 forward/backward/optimizer step 只激活并训练一个 tenant adapter。
当前这套机制支持普通 `target_modules` LoRA，但还不支持 PEFT
`target_parameters`。

PEFT 0.19.x 以及当前 PEFT main 文档都还不支持通过 `target_parameters`
给裸参数添加或加载多个 adapter。因此，Twinkle 需要在本地实现一套
target-parameter multi-slot 路径。

## 目标

在 `MultiLoraTransformersModel` 中支持 DSV4 EP multi-LoRA 训练，并满足
以下语义：

- 同一个 DSV4 EP 模型中可以常驻多个 adapter。
- 每次 forward/backward/optimizer step 只激活并训练一个 adapter。
- 本轮只保证 SFT 训练路径可用。
- DSV4 expert 结构继续保持 packed `nn.Parameter`，不转换为逐 expert 的
  `nn.Linear` 模块。
- 保存出的 adapter checkpoint 可以直接被原始 Transformers + PEFT
  `target_parameters` 推理路径加载。

## 非目标

- 不支持同一次 forward 中同时训练多个 adapter。
- 不 fork 或 patch PEFT site-packages。
- 不把 DSV4 packed experts 转成 `nn.Linear` 作为默认路径。
- 本阶段不修改、不验证 sampler、vLLM、权重同步或 RL/GRPO 训练路径。

## 架构

在 Twinkle 的 `MultiLora` 系统中新增 target-parameter LoRA 路径。

现有 regular module 路径继续处理 `target_modules`。新增路径处理包含
`target_parameters` 的 config。两条路径共享 tenant 到 slot 的映射、adapter
激活、optimizer group 选择，以及 checkpoint save/load API。

实现上引入一个面向裸参数的小型 wrapper/manager。概念上，每个目标参数
包含：

- owning module 和 parameter name 的引用。
- 预分配的物理 adapter slots（`lora_0`、`lora_1` 等）。
- 每个 slot 一组按 `max_r` 分配的可训练 `lora_A` 和 `lora_B` 权重。
- 由 `MultiLora.activate_adapter` 控制的当前 active slot。

forward 期间，只有 active slot 贡献 delta weight。wrapper 会临时把原参数
变成 `W + delta(active_slot)`，执行 base module forward，然后移除临时
parametrization。

## 注入顺序

对于 EP + target-parameters adapter，顺序很关键：

1. 加载 DSV4 base model。
2. 应用 expert parallelism，使 packed expert 参数先 shard 到本 rank。
3. 在已 shard 的 expert 参数上安装 target-parameter multi-LoRA slots。
4. 为 tenant adapters 构建 optimizer groups。
5. 最后再执行 native FSDP2 wrap。

这个顺序能保证 `find_moe_blocks` 和 EP sharding 仍作用在原始模型结构上，
同时 EP forward 读取 `gate_up_proj` 和 `down_proj` 时能拿到 parametrized
之后的权重。

## Delta 计算

target-parameter delta 计算应与 PEFT `ParamWrapper` 语义一致。

对于 2D 目标参数，使用标准 LoRA delta：

```text
delta = (B @ A) * scaling
```

对于 3D expert 参数，需要为每个 local expert 计算 delta，并生成与目标参数
shape 完全一致的 tensor。实现必须保持 PEFT 对 packed MoE 参数使用的方向
约定，包括 owning experts module 上可能存在的 transposed 变体。

slot tensor 按 `max_r` 分配，但每个 tenant config 只使用前 `r` 行/列。
scaling 与 PEFT 保持一致：

- 标准 LoRA：`lora_alpha / r`
- rsLoRA：`lora_alpha / sqrt(r)`

以下 PEFT target-parameter 选项需要尽早报清晰错误，约束与 PEFT 尽量保持
一致：

- `lora_dropout != 0`
- `fan_in_fan_out=True`
- `use_dora=True`
- `lora_bias=True`

## 激活与训练

`MultiLora.adapter(tenant_adapter_name)` 将 tenant adapter 映射到一个物理
slot，并同时为 regular module LoRA 和 target-parameter LoRA 激活该 slot。

`_get_trainable_parameters(adapter_name)` 只能返回当前 tenant 对应 slot 的
参数。这样可以保持现有独立 optimizer group 行为，并避免一个 adapter 的
optimizer step 更新另一个 adapter。

disabled-LoRA context 需要同时禁用 regular module LoRA 和 target-parameter
LoRA。

## Checkpoint 格式

保存出的 adapter checkpoint 必须兼容 PEFT `target_parameters`。

对于某个 tenant adapter，`get_state_dict` 和 `save` 应只导出该 tenant 对应
的物理 slot，去掉 key 中的物理 slot 名，并输出 PEFT 对相应
`target_parameters` 期望的 logical key 和 shape。

在 EP/FSDP2 模式下，保存必须走 strategy 的 full-state 路径，使 local expert
shards 在写入 `adapter_model.safetensors` 前按原始 global expert 顺序 gather。

加载时应接受同样的 PEFT adapter checkpoint，把保存的实际 rank 拷贝到该
tenant 的物理 slot 中，并像现有 regular module LoRA 一样，把剩余
`max_r - r` 容量置零。

## 错误处理

以下情况需要抛出清晰错误：

- config 请求 `target_parameters`，但当前没有可用的 multi-LoRA
  target-parameter 路径。
- 找不到目标参数。
- 目标参数维度不受支持。
- tenant rank 超过 `max_r`。
- save/load 遇到无法映射回目标参数的 checkpoint key 或 shape。

## 测试

按集成深度递增添加测试：

1. 使用包含 3D expert 参数的 fake module 做单元测试。注册两个 tenant
   adapters，交替执行 forward/backward/optimizer step，验证只有 active
   tenant 被更新。
2. 使用 fake module 做保存/加载兼容性测试。从 Twinkle multi-slot 路径保存
   tenant adapter，再用 PEFT 原始 `LoraConfig(target_parameters=...)` 加载，
   验证输出一致。
3. 基于现有 MoE EP 测试添加 EP/FSDP smoke test。让两个 adapters 常驻，交替
   各训练一步，分别保存，并验证保存出的 checkpoint 能被未修改的
   Transformers + PEFT 模型加载。

前两个测试不应依赖 DSV4 权重。EP/FSDP smoke test 可以沿用现有多 GPU 和
模型可用性 gate。

## 未决风险

- EP/FSDP2 下 target-parameter LoRA 的 full-state gather 必须精确保持 global
  expert 顺序。
- PEFT 后续可能原生支持 `target_parameters` multi-adapter；Twinkle 实现需要
  保持隔离，方便未来替换为上游支持路径。
- DSV4 packed expert 的方向约定必须完全匹配。fake-module 测试应尽量覆盖
  normal 和 transposed 两种 3D layout。
