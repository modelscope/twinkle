# E18 — BigCodeBench 上的拒绝采样 SFT

从 `cookbook/exp/skill2lora/skill_ablate/config.py` L244-252 的 `ExpSpec('E18', 'rejection_sft', ...)`
单独脱离出来的自包含实现，目录布局仿照 `cookbook/human`（E23）。

## 与 E23 的关系

两臂**共用**环境层与教师 judge（直接 import `../human/e23_bcb.py`、`../human/e23_rubric.py`，
不拷贝），所以数据过滤、沙箱单测判分、rubric 诊断三处逐字同源、结果可直接比。差别只在训练方法：

| | E23 | E18（本目录） |
|---|---|---|
| 方法 | GRPO（组内归一化 advantage） | 拒绝采样 SFT（只用正样本） |
| loss | `SEAMBNPOLoss`（PPO clip + KL） | **`CrossEntropyLoss`**（纯交叉熵，无 ratio/clip/KL/advantage） |
| 梯度 | 正负都有；60% 组零方差→零梯度 | 只有正梯度，无零梯度浪费 |
| 每题产出 | 8 个候选全部进 batch | 两道筛后**唯一胜者**进池 |
| 训练轨迹 | 采样 token 直通（query+rubric） | messages 编码（**query-only**） |
| eval | 带 rubric | **query-only**（部署口径） |
| 卡数 | 8（含 ref） | 6（SFT 无需 ref 模型） |

「选择」全部发生在两道筛（只有胜者入池），到 loss 这一层就是普通的「拟合目标文本」，
所以不传 `advantages` —— `CrossEntropyLoss` 不读该参数，传了是静默无效。

## 两道筛（本臂唯一自变量，见 `e18_select.py`）

1. **增量达阈**：`with_pass >= base_pass_rate + MIN_PASS_GAIN`（默认 +2/8）。仅仅「没弄坏」
   （8/8 -> 8/8）不够格 —— 那种样本对「学会写有效 skill」没有监督信号；
2. **不超长**：超 `SKILL_CHAR_LIMIT` 直接丢；
3. **pass_rate 最大 -> 并列内 rubric 相似度**：先按客观效果取最大档，并列内取词频余弦相似度
   最高的。长度只在相似度也并列时做确定性拆平（取较短者）。

⚠️ 原先第 3 道筛是「先按离 `LEN_BUDGET`（400）的距离取前一半」，已删：实测 **66% 的题 8 个
候选全部 `with_pass=1.0`**（天花板打平），此时长度成了事实上的唯一决策依据，而它有系统性
偏差——中文表达同样内容字符数天然更少（357 vs 705），永远更贴近 400，于是「离预算最近」被
翻译成「选中文模板」，把信息量大的长英文候选全部淘汰。

⚠️ 原先有一道泄漏门（`leak_blocks`），已删：BCB 的 `reference_answer` 是 ~2500 字符的 dict，
子串匹配要求 skill 逐字包含整个 dict 的 repr，而 skill 上限 1500 字符 —— 触发概率恒为 0。
真正的泄漏通道在 `test` 的断言期望值与 `canonical_solution`，需另写检测。

⚠️ 存活候选 ≤2 条时相似度那阶形同虚设，胜者由长度决定 —— 详见 `select_winner` 的 docstring。

## 文件

| 文件 | 内容 |
|---|---|
| `e18_rejection_sft.py` | 采集 + SFT 主循环、eval、swanlab |
| `e18_prompts.py` | executor / skill-gen / 训练轨迹三处 prompt |
| `e18_select.py` | 三道拒绝筛（泄漏门 + 词频余弦 + 两阶选择） |

## 跑法

```bash
cd cookbook/human_e18
# 教师 API 必需（rubric 是选择的参照系，不可降级）
export LLM_BACKUP_API_KEY=...  LLM_BACKUP_BASE_URL=...
nohup python -u e18_rejection_sft.py > nohup.e18.$(date +%m%d-%H%M).log 2>&1 &
```

主要环境变量（默认值见文件头）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `ACCUMULATE` | 16 | 攒够多少条胜者 SFT 一次（须为 `TRAIN_DP` 整数倍） |
| `N_SKILLS` | 8 | 每题候选数 = 拒绝采样的池大小 |
| `MAX_UPDATES` | 50 | 总更新数 |
| `SKILL_CHAR_LIMIT` | 1500 | 超过直接丢 |
| `EVAL_SIZE` | 200 | holdout 题数；0 = 不 eval |
| `SWAN_PROJ` | twinkle | swanlab 项目（**勿** export `SWANLAB_PROJECT`） |

## 产物

| 文件 | 内容 |
|---|---|
| `output.e18/e18_sft_dataset.jsonl` | **主产物**：每条胜者 + rubric/相似度/pass 全审计字段，可离线复算、换超参重训而不必重跑 GPU |
| `output.e18/train_log.jsonl` | 逐 chunk 指标（accept_rate / candidate_pass_rate / eval/*） |
| `output.e18/E18-final/` | 最终 skill 模型权重 |
