# RSI Agentic Self-Play

## 前置条件

- 沙箱环境已启动（AgentENV / e2b），模板由 `cookbook/rl/rsi_agentic/sandbox_server/install.sh` 构建
- `AENV_API_URL` 和 `AENV_TEMPLATE` 环境变量已设置
- ms-agent 配置文件就绪（默认 `cookbook/rl/rsi_agentic/rsi_agent.yaml`）

## Step 1: 生成任务

```bash
python cookbook/rsi/agentic/challenge.py \
    --keep-target 200 \
    --sandbox-template $AENV_TEMPLATE \
    --sandbox-api-url $AENV_API_URL \
    --sampler-gpus 4
```

产出：`output/rsi_agentic/challenge_flows.jsonl`

每行一个任务：`{id, query, check_script, n_pass, n_rollouts, keywords, seeded}`

可选参数：
- `--seed-file seeds.jsonl` 用已有 trajectory 做起点
- `--keywords-n 0` 关闭关键词库
- `--solver-rollouts 4` 难度过滤尝试次数
- `--max-turns 20` round1 最大工具调用轮数

## Step 2: 训练（GRPO）

```bash
AENV_API_URL=http://... \
AENV_TEMPLATE=twinkle-rsi-msagent \
RSI_TASKS=output/rsi_agentic/challenge_flows.jsonl \
    python cookbook/rl/rsi_agentic/rsi_agentic_grpo.py \
    --model-id ms://Qwen/Qwen3-4B \
    --model-gpus 4 --sampler-gpus 4
```

训练脚本自动识别 `check_script` 格式，在沙箱中跑检查脚本评分（exit 0 = 1.0）。

## 流程总结

```
challenge.py                    rsi_agentic_grpo.py
┌─────────────────────┐         ┌─────────────────────┐
│ 1. 选方向+关键词     │         │ 1. 读 flows         │
│ 2. 模型在沙箱做事    │         │ 2. 起沙箱           │
│ 3. 模型写检查脚本    │  flows  │ 3. solver 多轮做题  │
│ 4. 跑检查（验证）    │ ──────► │ 4. 跑 check_script  │
│ 5. 模型写题目描述    │         │ 5. GRPO 训练        │
│ 6. 难度过滤         │         └─────────────────────┘
└─────────────────────┘
```
