# GraphCredit Gate 0

该目录只包含离线 Gate 0 探针，不进入 Twinkle 核心训练路径。

Gate 0 使用确定性状态机生成多轮轨迹，验证以下前提：

- 环境显式状态键覆盖率；
- 同一 prompt 下同状态动作的支撑度；
- 状态图 credit 对正确动作和错误动作的排序能力；
- 无成功终态时 graph advantage 是否严格为零；
- 纯 CPU 图构造和 advantage 计算耗时；
- 是否产生额外 policy forward。

运行：

```bash
PYTHONPATH=src python cookbook/rl/graphcredit/gate0.py
```

Gate 0 未通过时，不进入 `TurnCredit` 或 state-graph advantage 的核心实现。
