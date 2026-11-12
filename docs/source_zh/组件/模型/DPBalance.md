# 数据并行负载均衡

公共配置、规划及 local 样本交换由 `TrainableModel` 负责。TransformersModel、MegatronModel 及其 multi-LoRA 变体默认 `dp_balance=True`。子类覆盖 `slice_dp` 的 forward_only/forward_backward 时自动继承调度钩子；独立构造函数调用 `_init_dp_balance`。新后端声明 `_dp_balance_backend`，并为不同的 micro-batch 布局提供对应的代价估算。

模型在 DP 切分前统一规划：保持各副本样本数相同，同步重排所有按样本对齐的 list、Tensor 和嵌套字段，收集后还原逐样本输出顺序。同一个 DP 副本的 TP/PP/CP peers 接收相同分片。local 模式仅在需要重排时交换 CPU 序列化的样本及输出，通信开销需要实测。嵌套调用训练入口只均衡一次。

Transformers 复用固定/动态 micro-batch planner 估算代价。Megatron 按连续固定 micro-batch、variable_seq_lengths 对应的每 micro-batch/整 rank padding，以及 CP/SP 长度对齐规则估算；保持原 micro-batch 数和流水线调度。动态 batching、不完整 micro-batch 自动跳过。成组 loss 还要求 micro-batch 布局不打散组内样本。

仅允许梯度权重满足重排不变性的 loss 配置参与：普通 GRPO、PPO sequence-mean、CE sum、BNPO global 及部分成对/分组目标。按 rank 局部 token 求均值、依赖 batch 统计或未声明支持的自定义 loss 保持原顺序。新 loss 需声明满足该条件的 `dp_balance_group_size`。该功能是计算量启发式，不替代 MoE 专家路由均衡，也不是精确的多模态代价模型。DP=1、rank 样本数不齐、缺少安全长度信息的原始/分组输入、原始 logits 返回及整批远端引用等情况也会跳过。

远端特征以逐样本 ObjectRef 列表传入时，可以附带 `input_lengths=[...]`。控制端只用长度规划，不调用 ray.get 拉回特征；先切分引用列表，每个 worker 只解析自己的分片。缺少元数据的远端输入保持原顺序；隐藏在单个 ObjectRef 后的整批 Tensor 无法确认对齐关系，也会跳过。长度元数据在调度边界消费。训练均衡指标统一由 model.calculate_metric 返回，前缀为 dp_balance/*。

Transformers 不带 micro-batch 的 forward_only 重排后，逐样本 loss 参数应在 forward_only 时传入，calculate_loss 使用相同 loss_key 并复用已对齐参数；新传入的 batch 参数会被拒绝。带 micro-batch 的 forward_only 仅缓存最后一个 micro-batch，重排后禁止单独 calculate_loss，完整训练使用 forward_backward。

Swift 使用同一个默认开启的 `--dp_balance`，`--dp_balance false` 关闭。recipe 只传特征、训练信号和可选长度，不负责规划或重排。
