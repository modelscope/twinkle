# Agentic 预处理器

Agentic 预处理器模块提供了基于流水线的多轮对话数据质量过滤框架，用于 RLHF / Agentic 微调之前的训练数据清洗和过滤。

## QualityPreprocessor

`QualityPreprocessor` 是一个轻量级流水线运行器，接受过滤器列表并按顺序执行。每个步骤接收行列表，返回 `(kept, dropped)`，流水线会记录每步统计信息。

```python
from twinkle_agentic.preprocessor import QualityPreprocessor, HardFilter, DeadLoopFilter

pipeline = [
    HardFilter(min_user_chars=10),
    DeadLoopFilter(),
]
preprocessor = QualityPreprocessor(pipeline, dropped_log_path='dropped.jsonl')

# rows 是列格式的字典（Dataset.map 格式）
cleaned = preprocessor(rows)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `pipeline` | `List[Callable]` | 有序的过滤步骤列表。每个步骤接收 `List[Dict]`，返回 `(kept, dropped)`。 |
| `dropped_log_path` | `str` | 可选的 JSONL 文件路径，用于记录被丢弃的行及步骤名称和原因。 |

## 内置过滤器

### HardFilter

基于硬规则的过滤器，使用确定性规则移除质量差的行。支持多语言检测（EN/ZH/JA/KO）。

```python
from twinkle_agentic.preprocessor import HardFilter

f = HardFilter(
    min_user_chars=10,           # 非 CJK 用户查询最小字符数
    min_user_chars_cjk=6,        # CJK 用户查询最小字符数
    min_assistant_chars_2turn=80, # 两轮对话中助手回复最小长度
    min_thinking_chars=200,      # 思考链最小长度（可豁免过滤）
    system_deny_keywords=['hack', 'exploit'],
    max_chars_per_round=50000,
    max_total_chars=200000,
    max_rounds=50,
)
```

**丢弃原因：** `trivial_single_turn`（平凡单轮）、`shallow_reply`（浅回复）、`all_empty_assistant`（全空助手）、`system_deny_keyword`（系统拒绝关键词）、`round_too_long`（单轮过长）、`total_too_long`（总长过长）、`too_many_rounds`（轮次过多）

### DeadLoopFilter

检测助手消息中的犹豫/死循环模式——重复自我纠正、级联纠正和高 n-gram 重复。

```python
from twinkle_agentic.preprocessor import DeadLoopFilter

f = DeadLoopFilter(
    hesitation_density_threshold=7.0,   # 每 1000 字符犹豫标记数（响应）
    cascade_threshold=5,                 # 窗口内级联标记数
    cascade_window=800,                  # 窗口大小（字符）
    repetition_threshold=0.45,           # N-gram 重复率
    think_hesitation_density_threshold=15.0,  # <think> 块更宽松
    think_repetition_threshold=0.65,
)
```

对 `<think>` 推理块使用更宽松的阈值（允许自由发散），对可见响应使用更严格的阈值。

### DedupFilter

全局最长优先去重。签名由第一个真实用户轮次（首尾）和第一个助手回复推导。

```python
from twinkle_agentic.preprocessor import DedupFilter

f = DedupFilter(prefix_chars=100, asst_chars=100)
kept, dropped = f(all_rows)  # 必须在一次调用中传入整个数据集
```

> **注意：** `DedupFilter` 需要在单次调用中接收完整数据集。**不要**将它放入 `QualityPreprocessor` 中（后者按批处理）。请在流水线之前或之后单独运行。

### RefuseFilter

检测第一条助手回复中的自我引用式拒绝（如"我无法帮助您"）。多语言模式匹配（EN/ZH/JA/KO）。

```python
from twinkle_agentic.preprocessor import RefuseFilter

f = RefuseFilter(check_window=600)  # 仅检查前 N 个字符
```

### TokenSoupFilter

检测乱码/token-soup 输出，检查替换字符、控制字符、私用区 Unicode、泄漏的特殊 token、单字符重复和脚本混乱。

```python
from twinkle_agentic.preprocessor import TokenSoupFilter

f = TokenSoupFilter(
    replacement_char_ratio=0.02,
    special_token_count=20,
    script_chaos_threshold=0.55,
)
```

### PIIPresidioFilter

基于 Microsoft Presidio + spaCy NER + Faker 的多语言 PII 检测和重写。检测并替换个人身份信息（姓名、邮箱、电话号码、地址等）。

```python
from twinkle_agentic.preprocessor import PIIPresidioFilter

f = PIIPresidioFilter(languages=['en', 'zh'])
```

### IntentClassifier

启发式意图分类器，为每行标注检测到的意图。可插拔的检测器管线。

```python
from twinkle_agentic.preprocessor import IntentClassifier

classifier = IntentClassifier()
```

**意图类别：** `tool_call`（工具调用）、`code`（代码）、`math`（数学）、`complex_logic`（复杂逻辑）、`reasoning`（推理）、`user_dissatisfaction`（用户不满）、`other`（其他）

### ScoreFilter

可插拔评分器过滤器，内置字符级指标、语义相似度和代码执行评分器。

```python
from twinkle_agentic.preprocessor import ScoreFilter

f = ScoreFilter()
```

**内置评分器：** `ChrMinScorer`、`SIFDScorer`、`PassNScorer`、`ParaphraseScorer`

### ModelFilter

按模型 ID 白名单过滤行。

```python
from twinkle_agentic.preprocessor import ModelFilter

f = ModelFilter(allowed_models=['qwen3.5-4b', 'qwen3.5-32b'])
```

### MessageNormalizer

三遍消息规范化：心跳剥离、工具调用重写、连续同角色消息合并。

```python
from twinkle_agentic.preprocessor import MessageNormalizer

normalizer = MessageNormalizer()
```

## 完整流水线示例

```python
from twinkle_agentic.preprocessor import (
    QualityPreprocessor,
    HardFilter,
    DeadLoopFilter,
    RefuseFilter,
    TokenSoupFilter,
    MessageNormalizer,
    DedupFilter,
)

# 第一步：全局去重（必须在完整数据集上运行）
dedup = DedupFilter()
rows, _ = dedup(all_rows)

# 第二步：按批流水线
pipeline = [
    HardFilter(min_user_chars=10, max_rounds=30),
    DeadLoopFilter(),
    RefuseFilter(),
    TokenSoupFilter(),
    MessageNormalizer(),
]
preprocessor = QualityPreprocessor(pipeline, dropped_log_path='dropped.jsonl')
cleaned = preprocessor(rows)
```
