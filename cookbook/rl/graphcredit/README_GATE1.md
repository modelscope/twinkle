# GraphCredit Gate 1

该实验在保持 rollout、组大小、模型、LoRA 配置、更新步数和评测任务完全一致的条件下，
比较 episode-level GRPO 与 `StateGraphAdvantage`。

## 任务

二维网格只要求连续两步靠近目标。错误方向会立即终止 episode。episode reward 只使用
最终成功或失败：

- 成功：`1.0`
- 失败：`0.0`

因此 GRPO 只能把最终结果广播给 episode 内所有动作，而 GraphCredit 可以根据
`pre_state` 分组，给“当前动作正确但后续失败”的中间动作更高信用。

## A2 运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD/src"

python cookbook/rl/graphcredit/gate1.py \
  --model-path /workspace/shared_assets/models/Qwen/Qwen3-0.6B \
  --mode grpo \
  --steps 20 \
  --eval-tasks 24 \
  --group-size 8 \
  --device npu:0 \
  --output-dir outputs/graphcredit-gate1/grpo

python cookbook/rl/graphcredit/gate1.py \
  --model-path /workspace/shared_assets/models/Qwen/Qwen3-0.6B \
  --mode graphcredit \
  --steps 20 \
  --eval-tasks 24 \
  --group-size 8 \
  --device npu:0 \
  --output-dir outputs/graphcredit-gate1/graphcredit
```

两次运行必须使用相同 `--seed`、模型、任务范围、group size 和训练步数。脚本输出：

- `adapter/`：训练后的 LoRA；
- `result.json`：逐步 reward、loss、耗时、峰值 HBM 和 GraphCredit diagnostics；
- 标准输出：每一步一行 JSON，便于远端日志采集。

## 证据边界

该实验验证 advantage consumer 的训练方向，不替代 Twinkle 完整 `GRPOLoss` 和生产训练
入口。它没有 reference KL、多 epoch PPO 或异步 rollout。只有在 held-out success
稳定优于同预算 GRPO 后，才进入真实 Agentic 任务和多种子验证。
