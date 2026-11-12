# 变长因果融合注意力（Ascend NPU）

本 PR 新增 `twinkle.kernel.ops.fused_varlen_attention`。它把 padding-free/packing
训练所需的分段因果注意力实现为一次 Ascend 融合内核调用：输入是拼接后的
`[T, H, D]` 序列和累积长度 `cu_seqlens`，输出等价于逐文档独立执行因果注意力，
文档之间不可见。

## 为什么需要这个算子

Twinkle 已有 padding-free 数据拼接能力，但 main 分支的 dense Transformers 路径不会为
packed 输入生成 varlen 元数据；部分已有的 packed/SP 路径还直接依赖外部
`flash_attn`，在当前 CANN 运行环境中不可用。NPU 侧现成可用的注意力实现是定长
SDPA，它没有分段边界参数。

Liger-Kernel 的 Ascend 后端已有 attention 相关内核，例如 attention residual 和
fused neighborhood attention，但尚未提供与 `fused_varlen_attention` 等价的、
可直接服务 padding-free 训练的变长因果融合注意力。

因此本 PR 补齐两部分：

- dense packed 输入的分段元数据生成和 ``cu_seq_lens_q/k`` 透传；
- Ascend TND 分段因果融合内核，以及所需的块掩码和缓存。

这是新增能力，不修改现有 padded 训练的默认路径。

## 用法

```python
from twinkle.kernel.ops.fused_varlen_attention import fused_varlen_attention

# 两条文档：512 + 300 token，8 个 query 头，4 个 KV 头
out = fused_varlen_attention(q, k, v, [0, 512, 812])  # [812, 8, D]
loss = out.sum()
loss.backward()                                       # 同时产生 dq/dk/dv
```

该算子不进入 `DEFAULT_KERNEL_CONFIG`，需要显式启用：

```python
from twinkle.kernel import DEFAULT_KERNEL_CONFIG, kernelize
from twinkle.kernel.registry import KernelChoice

kernelize(model, {
    **DEFAULT_KERNEL_CONFIG,
    'npu_varlen': KernelChoice(op='fused_varlen_attention', backends=('npu',)),
})
```

随后将模型注意力实现名设为 `flash_attention_2`，上游模型会在该分发名下透传
`cu_seq_lens_q`。实现通过显式 `kernelize` 安装，不导入或依赖外部 `flash_attn`，
也不会改变默认注意力后端。

使用 `TransformersModel` 时，`InputProcessor(padding_free=True)` 会从重置的
`position_ids` 构造分段边界。只有一个文档时直接使用原生 causal SDPA；多文档才进入
融合 varlen 内核。拿不到分段边界时回退原生 SDPA。

## 语义

- 每个文档只关注本文档中位置不大于自己的 token。
- GQA/MQA 要求 `Hq % Hkv == 0`，KV 头按组共享。
- 反向由融合内核计算；调用方不需要逐文档循环。
- 块掩码使用固定的 `2048 × 2048` 上三角布尔矩阵，并按 `cu_seqlens` 对齐分段边界。

## 适配层边界

同一个适配层同时服务 `npu` 后端（融合内核）和 `torch` 后端（逐文档精确实现），
因此两条路径的语义完全一致：

- 拿不到 `cu_seq_lens_q` 时按 HF 语义合成掩码：方阵输入走原生 causal SDPA；
  `query` 短于 `key`（KV-cache 解码、chunked 解码）时因果掩码按右下角对齐，
  放行全部历史 key。
- 拼接入口会校验边界（从 0 开始、严格递增、末项等于总长），并显式拒绝
  `attention_mask`、非因果请求、`query/key` 长度不一致以及非共享分段边界，
  不再静默忽略这些参数。

## 正确性

```bash
pytest tests/kernel/ops/test_fused_varlen_attention.py -q
```

无 NPU 环境为 `38 passed, 46 skipped`；NPU 环境运行全部用例。

**验收口径（严格，不放宽）**：对当前 Twinkle NPU SDPA baseline（同精度原生实现）逐元素
`assert_close`：

- 13 个形状、FP32/BF16、10 个种子，共 **260/260** 组前向及 `dq/dk/dv` 通过；
- FP32：`rtol=1e-4`，`atol=1e-5`；
- BF16：`rtol=1.6e-2`，`atol=1e-2`；
- 单文档 HF 快路径：**40/40** 组严格前反向通过；
- 连续 token、单 token 文档、跨 2048 块、MHA、GQA 16/4、GQA 16/8 均有覆盖。

**诊断口径（仅用于定位精度来源）**：对“显式浮点掩码 + FP32 精确参考”比对时，FP32 仍是
严格判定；BF16 因为 `dv = P^T @ do` 的上千项求和在两套分块顺序下的舍入差异，会有最多
4/1,048,576 个元素站在 `atol` 边界（最大绝对差 0.0312），因此这类诊断用显式越界元素预算
判定，且不作为验收依据。

BF16 GQA 会在调用融合内核前展开 KV，使反向归约顺序与当前 baseline 的
`repeat_kv` 一致；FP32 保留原生 GQA。

## 单算子性能

相同拼接输入、相同物理 token，只替换注意力内核。NPU Event 计时，3 次预热，
`10 iters × 5 repeats` 取中位数。

| 精度 | 场景 | 原生 SDPA | 融合算子 | 加速比 |
|---|---|---:|---:|---:|
| BF16 | 512/300/1024/77 | 1.136 ms | 0.482 ms | 2.36× |
| BF16 | 2×1024 | 1.068 ms | 0.484 ms | 2.21× |
| BF16 | 2×2048 | 2.908 ms | 1.193 ms | 2.44× |
| BF16 | GQA 16/4，2×1024 | 1.841 ms | 1.088 ms | 1.69× |
| BF16 | 短长混批 8 篇 | 1.381 ms | 0.527 ms | 2.62× |
| BF16 | 长批 4 篇 | 7.131 ms | 2.255 ms | 3.16× |
| FP32 | 512/300/1024/77 | 1.667 ms | 0.569 ms | 2.93× |
| FP32 | 2×1024 | 1.748 ms | 0.590 ms | 2.96× |
| FP32 | 2×2048 | 6.133 ms | 3.076 ms | 1.99× |
| FP32 | GQA 16/4，2×1024 | 3.732 ms | 0.924 ms | 4.04× |
| FP32 | 短长混批 8 篇 | 2.545 ms | 0.693 ms | 3.67× |
| FP32 | 长批 4 篇 | 15.709 ms | 5.286 ms | 2.97× |

复现：

```bash
python cookbook/kernel/fused_varlen_attention/bench_op.py --out result.json
```

## 端到端训练

Qwen3-0.6B，单卡 Ascend 910B4，相同初始权重、固定 batch、SGD `lr=1e-5`、
seed 2026、2 步预热。baseline 使用 padded 布局和现有 NPU SDPA，candidate 使用
packed 布局和本算子。

| 精度 | 场景与步数 | Baseline | Candidate | 加速 | 峰值显存 | 最大 loss 差 |
|---|---|---:|---:|---:|---:|---:|
| BF16 | mixed，200 步 | 727.1 ms | 376.7 ms | **1.930×** | 4979→3424 MB | 0.003743 |
| FP32 | mixed，200 步 | 1507.0 ms | 609.0 ms | **2.474×** | 9947→6833 MB | 1.91e-6 |
| BF16 | 256/1024/4096/512，50 步 | 6539.8 ms | 1246.0 ms | **5.249×** | 16446→6645 MB | 0.001619 |
| FP32 | 256/1024/4096/512，50 步 | OOM | 50 步完成 | 不计算 | padded OOM | 不计算 |

单文档和多文档等长场景没有 padding 收益，端到端分别为 0.989–1.022× 和
0.977–1.029×，基本与 baseline 持平。

### 收益归因

固定相同 packed 输入，逐层增加优化：

| 场景 | 去除 padding | 融合内核（关闭缓存） | 边界列表缓存 | 总加速 |
|---|---:|---:|---:|---:|
| mixed BF16 | 1.498× | 0.996× | 1.364× | 2.002× |
| mixed FP32 | 2.176× | 1.067× | 1.059× | 2.476× |
| 4096 混批 BF16 | 4.516× | 1.150× | 1.013× | 5.275× |

掩码缓存两轮为 0.984× 和 1.055×，未测得稳定的独立收益。相同 packed 输入下，
最终候选相对 packed SDPA 的净加速为 mixed BF16 1.336×、mixed FP32 1.138×、
4096 混批 BF16 1.168×。

padding-free 拼接机制本身是既有能力；端到端提升主要来自消除 padding。
本算子的贡献是让 dense NPU 路径能够正确、高效地执行 packed 分段因果注意力。
完整评测和消融见仓库外的验收报告。

复现：

```bash
python cookbook/kernel/fused_varlen_attention/bench_e2e.py --help
python cookbook/kernel/fused_varlen_attention/e2e_two_checkout.py --help
```

## Ascend 调用注意事项

1. 必须显式传入因果 `atten_mask`。当前 CANN 版本不会仅根据 `sparse_mode` 和
   `next_tockens` 自动生成因果掩码。
2. 变长 TND 前反向可用组合是 `2048 × 2048` 上三角布尔掩码配 `sparse_mode=3`。
   `(T, T)` 掩码配 `sparse_mode=0` 时前向正确但反向会失败。
3. `actual_seq_qlen` 只接受累积长度的 `list[int]`。训练路径会按张量对象和版本号
   缓存转换结果，避免每层触发主机同步。

前两条由 `npu._verify_vendor_semantics` 在设备首次调用时用 8 token、2 文档的
小例子自检，容差 `1e-4`。

## 已知限制

- 只支持因果注意力；非因果使用 `backend='torch'` 的逐文档参考实现。
- 不支持 dropout、滑动窗口和空文档。
- padding-free 输入要求 batch=1，多 batch 需在数据侧拼成单序列。
- 融合路径依赖 CANN `npu_fusion_attention` 的 TND 语义。
- 单段长序列没有跨文档计算可省，2048 块掩码会成为额外开销。
