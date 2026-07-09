# Preprocessor 审计与整改清单

> 审计范围：`src/twinkle_agentic/preprocessor/` 全部 15 个文件、30 个类。
> 审计方法：逐行只读审阅 + 关键断言代码复核 + cookbook/tests 实际接线核实。
> 三轮视角：(A) 实现问题 (B) 类设计/拆分合并 (C) 功能增删。

## 实施状态（已按本清单完整落地）

> A1 经确认跳过：R1 已把 `score_filter.py` 整体移入 `experimental/`（零 active 使用的死代码），
> 对死代码再做 4 文件拆分只增维护面、零收益，启用前再拆。其余 21 项全部实施。

| 项 | 状态 | 落地位置 |
|----|------|----------|
| A5 | ✅ | `label_schema.py`（`user_data` 信封 + `set_labels`/`get_label` + `pack_value`） |
| P3 | ✅ | `message_normalizer.py` Pass1 重建 assistant 时 `dict(msg)` 透传全字段 |
| P7 | ✅ | `hard_filter._has_tool_calls` / `message_normalizer._strip_heartbeat`+`_is_atomic` 全部走 `normalize_tool_calls` |
| D7 | ✅ | `trajectory_scorer.py`（Segmenter→HardScorer 逐轮→fuse_segment→aggregate_trajectory→写回 `user_data`，mapper 不删） |
| D7c | ✅ | `trajectory_scorer.py` `_segment_confidence`（一致性+voting稳定+决断性）+ `RubricVerifier.score_detail(extra_context=)` 客观注入重评 |
| D6 | ✅ | `outcome_filter.py`（纯读 `traj_score`/`safety_*` 标签比阈值，fail-open） |
| D8 | ✅ | `safety_scorer.py` + `RubricVerifier(fixed_rubric=)` 固定安全 rubric |
| D9 | ✅ | `pii_presidio_filter.py` `regex_only=True`（stub NlpEngine 免 spaCy，REPLACE→MASK 免 faker） |
| D10 | ✅ | `provenance.py`（`ProvenanceStamp`，血缘写入 `user_data`） |
| R1 | ✅ | `experimental/`（`score_filter.py` + `llm_backend.py` git mv 移出主包） |
| R3/R4/R5 | ✅ | `intent_classifier.py`（默认不删；DEFAULT_DETECTORS 精简为 ToolCall/Code/Math；LLM 路径经 R1 已全归 `llm_backup`） |
| P1/P5/P6/P8/P10 | ✅ | trim 后重算 `is_agent`；deadloop agent 行改扫有文本轮；`max_rounds` 按 pair；refuse 扫全 assistant+可选 reasoning；system 多模态保护 |
| A2/A4 | ✅ | `logprob_utils.py` + `message_utils.py`（`utils.py` 保留 shim）；`intents.py` 常量下沉 |
| A3 | ✅ | `twinkle/preprocessor/base.py` 基类返回 `Tuple[List,List]` + `Mapper`/`Filter` 语义基类（`ModelFilter`/`ProvenanceStamp` 已改用） |
| D4/D5 | ✅ | `language_filter.py`（langid 可选，启发式回退）；`structural_noise.py`（关键词无关噪声轮打标） |
| D1/D2 | ✅ | `offline/near_dedup.py`（MinHash-LSH，datasketch 可选+纯 Python 回退）；`offline/decontaminate.py`（13-gram 重叠，drop/tag） |
| A1 | ⏭️ 跳过 | 见上（R1 已隔离为死代码） |

---

## 0. 结论速览

> 本清单已根据 review 意见复核收敛：P2/P4 撤销，P1 降级，P11/P12 归入 R1（死代码，暂不单独修）。

- **共需改动 22 项**：实现问题 6（P1、P3、P5–P10）、结构重构 5（A1–A5）、功能增删 11（R1–R5 + D1/D2/D4/D5 + D6/D7/D7c/D8/D9/D10，D3 废弃）。
- **必须做（会静默损坏训练数据）**：仅 **P3** 一项（工具归一丢 reasoning 字段）。
- **达成「干净 + 每轮评分」最终目标的核心**：**A5**（`user_data` 信封，去 DAG 前置）+ **D7**（接线 verifier，分数写回每轮）+ **D7c**（自动校准 + 客观纠偏主观重评）+ **D6**（读标签滤废案）+ **D8**（安全 rubric）+ **D9**（PII 纯 regex）。
- **一句话结论**：现有清单修的是「清洗器 bug + 基础过滤」；要产出「干净且每轮带可信分」的 trajectory，还差——**A5 统一 `user_data` 标签信封（把评分/过滤解耦成打标 mapper + 末尾读标签 filter，去掉 DAG）+ 每轮评分打标(D7) + 自进化校准(D7c) + 废案过滤(D6) + 安全/PII(D8/D9)**。零件多数已存在（`verifier`+`aggregation`+`RubricVerifier`+`llm_backup`），核心工作是**接线 + 定 `user_data` 契约**。

### review 复核结论（撤销 / 降级项）

| 原项 | review 意见 | 复核结论 | 处置 |
|------|-------------|----------|------|
| **P1** trim 砍 tool 尾 | 不以 assistant 结尾的部分无训练必要，最多用于工具调用打分 | 成立。trim 尾部未闭合 tool 对训练无害；`is_agent` 不更新的副作用仅剩“末尾 `assistant(tool_calls)` 无对应结果”，训练时本应 mask | **降级为中等**，改描述，不再算“数据损坏” |
| **P2** heartbeat 误杀 | 这类数据是 openclaw/OpenHands 常见格式，作者本意就是要删 | 成立。agent 轨迹清洗语境下 heartbeat 轮几乎必为真噪声，误杀率极低；`message_normalizer.py:26-27` 注释确认是**故意**删除 | **撤销**（保留现状；可选加词边界，非必做） |
| **P4** 删 reasoning-only 轮 | 只有 thinking 无工具调用，训练无落点 | 成立。纯 thinking 轮无 target 输出，多轮里是悬空推理，删掉合理 | **撤销** |
| **P11** ParaphraseScorer 崩溃 | 应该没有实际使用 | 成立。`ScoreFilter`/`ParaphraseScorer` **全库零 active 使用**（仅自身定义 + docs 示例 + 注释掉的引用），测试只覆盖 `utils` 数学函数 | **归入 R1**（死代码，启用时再修） |
| **P12** IFD 公式口径 | 同上 | 同上 | **归入 R1** |

---

## 一、实现问题（正确性 / 语义）

### 严重：会静默损坏训练数据

| ID | 位置 | 问题 | 改动 | 预期收益 |
|----|------|------|------|----------|
| **P3** | `message_normalizer.py` Pass 1 重建消息 | 工具归一路径只保留 `role/content/tool_calls/tool_call_id` 四字段，**丢弃 `reasoning_content`/`thinking`/`name`** | 重建时透传全部原字段 | reasoning 蒸馏数据不再被清洗流程静默剥离 |

### 中等：策略漏洞 / 语义错位

| ID | 位置 | 问题 | 改动 | 预期收益 |
|----|------|------|------|----------|
| **P1** | `message_sanity.py:317,324-329` trim + `is_agent` | trim 掉末尾未闭合 tool 结果本身对训练无害，但 `is_agent` 在 trim **前**计算、trim 后不更新，残留“末尾 `assistant(tool_calls)` 无对应结果”，`check_tool_matching`（forward-only）不拦 | trim 后重算 `is_agent`，或末轮 `tool_calls` 无结果时 mask/剥离该 call；**非必做** | 末轮悬空 tool_call 得到一致处理，避免训练时误算 loss |
| **P5** | `dead_loop_filter.py:192-194` | `is_agent_row` 为真则**整行跳过** stuck 检测，agent 恰恰最易死循环 | agent 死循环走 `HardScorer.check_no_repeated_calls`（见 D3） | 覆盖 agent 重复工具调用循环，堵住最大系统性漏检 |
| **P6** | `hard_filter.py` `max_rounds` | 实现是 `len(asst_msgs) > max_rounds`，只数 assistant，注释却写 “user-assistant pairs” | 修正为按 pair 计数或改注释与语义一致 | 轮数过滤阈值语义正确 |
| **P7** | `message_normalizer.py` / `hard_filter.py` / `utils.py` | `tool_calls` 真值判断三处不一致（裸真值 vs `_has_tool_calls` 视 `''`/`'[]'`/`[]` 为空 vs `normalize_tool_calls`） | 全部统一走 `normalize_tool_calls` | 同一数据“是否 agent”判定一致，消除跨 filter 不一致 |
| **P8** | `refuse_filter.py` | 只扫首条 assistant 前 600 字、不读 reasoning，多轮/reasoning 拒答漏检 | 扩到全 assistant + reasoning 字段（可配窗口） | 拒答样本召回上升，减少污染 |
| **P9** | `token_soup.py` | `max_chars>0` 只查头部（cookbook 用 8000），尾部乱码漏检；不扫 reasoning | 全文 + reasoning 扫描或分段抽样 | 乱码样本召回上升 |
| **P10** | `message_sanity.py` `consolidate_system_messages` | 合并 multimodal system 时压成纯字符串，可能丢非 text part | 用 `msg_has_media` 保护多模态 system | 多模态 system 不丢内容 |

---

## 二、架构 / 类设计（拆分与合并）

| ID | 项 | 判断 | 改动 | 预期收益 |
|----|----|------|------|----------|
| **A1**（高） | `score_filter.py` 9 个类 486 行 | 契约与实现混在一起，加 scorer 就改巨型文件 | 拆为 `score/` 子包：`types.py`(RoundContext/ScoreResult/Scorer) + `scorers.py`(ChrMin/SIFD 轻) + `judge.py`(PassN/Paraphrase 重) + `score_filter.py`(编排) | 开闭原则；轻/重依赖分离；新增 scorer 零侵入 |
| **A2**（中） | `utils.py` | logprob 数学 + 消息格式工具两个无关模块塞一起 | 拆为 `logprob_utils.py` + `message_utils.py` | 降耦合；改 score 逻辑不误碰消息工具 |
| **A3**（高） | `twinkle/preprocessor/base.py:39` | 基类 `__call__` 声明返回 `Dict`，所有子类实际返回 `Tuple[kept, dropped]`，类型契约名存实亡 | 基类改 `-> Tuple[List, List]`；可选分 `Mapper`/`Filter` 语义基类 | 类型检查生效；新人不会照错签名写导致解包崩溃 |
| **A4**（低） | intent 常量位置 | `ScoreFilter` 消费 intent，但常量定义在 `intent_classifier.py`，score 独立后形成跨模块依赖 | intent 常量下沉到轻量 `intents.py` | 为 score 子包独立化铺路 |
| **A5**（高，目标前置） | 统一 `user_data` 标签信封 | 评分/安全/血缘无统一落点；D6↔D7 若代码互调会逼出 DAG | 所有标签走 `user_data` 的 `List[Tuple[str, pack_value(v)]]`（PyArrow 稳定，见 D 节前置）；打标 mapper 写、末尾 filter 读，靠列表顺序解耦 | 去 DAG、统一数据契约；A3 返回契约的自然延伸 |

**明确不动**（避免过度设计）：`HardFilter`/`RefuseFilter`/`DeadLoopFilter`/`TokenSoupFilter` 保持独立（合并成上帝类只会更糟）；`data_juicer.py` 4 个薄封装保持一文件；`LLMBackend` 三类保持；`MessageNormalizer` 3 个 pass 不拆（有强顺序依赖）；`IntentDetector` 层级设计是全代码最佳，保持。

---

## 三、功能增删

### 建议去掉 / 降级（死代码与过度设计）

| ID | 项 | 证据 | 改动 | 预期收益 |
|----|----|------|------|----------|
| **R1** | `ScoreFilter` 全家（+4 scorer）**+ `llm_backend.py` 整个文件** | `ScoreFilter` **全库零 active 使用**（仅自身定义 + docs 示例 + `train_cold_start.py:216` 注释态；测试只覆盖底层 `utils` 数学函数）；内含两处死代码 bug —— 原 **P11**（`ParaphraseScorer:452-455` 缺 DP pad，小批量/DP>1 时 `SamplerBackend` raise）、原 **P12**（`utils.py:154` `ifd=exp(-mean_delta)` 是 Superfiltering 差分指数口径而非 Cherry 损失比值，阈值不可互换）。`llm_backend.py`（`LLMBackend`/`OpenAIBackend`/`SamplerBackend`）**唯一消费者就是 `ScoreFilter`**（grep 确认除自身+`__init__` 导出外无他），它专为 score 打分提供 `chat`/`prompt_logprobs`/`prompt_logprobs_ids`/`embeddings` | `ScoreFilter` + `llm_backend.py` 一起移出主包到 `experimental/` 或 `data_selection/`，标记未验证；**启用前**再修 P11（复用 `_pad_batch`）+ 明确 P12 IFD 口径并重标阈值 | 主路径不再拖未验证的重代码 + 未接线的 LLM 后端抽象；bug 修复延后到真正需要时 |
| **R2** | `LLMBackend.embeddings()` | 是 R1 中 `llm_backend.py` 的一部分；preprocessor 侧**零调用者**，`SamplerBackend.embeddings` 直接 raise | 随 R1 一并移出（不单独保留伪抽象） | 去掉未接线接口，等真需要 embedding 去重再加 |
| **R3** | `IntentClassifier` | 产物 `key_rounds/intents` 主消费者是死的 `ScoreFilter`；`dataset_think.py` 只 import 不入 pipeline | 重定位为“标注器”（从不 drop）；短期可移出主 pipeline 省 CPU | 明确职责；省无谓计算 |
| **R4** | `ComplexLogic/Reasoning/UserDissatisfaction` Detector | 仅被 `IntentClassifier.DEFAULT_DETECTORS` 引用，下游死 | 精简 default 到 `ToolCall/Code/Math` | 减少无消费者的启发式维护面 |
| **R5** | LLM 调用抽象统一到 `llm_backup` | preprocessor 里活跃的 LLM 生成需求（summarizer/segment/verifier）**早已全部走 `twinkle_agentic/utils/llm_backup.py`**（置信度路由 student/teacher + 蒸馏数据收集）；只有死代码 `ScoreFilter` 还用独立的 `LLMBackend` | 主路径不再引入独立 `LLMBackend` 抽象，生成类需求统一走 `@llm_backup`。**注意**：`llm_backup` 只提供 chat 生成（返回 content 字符串），**不提供 `prompt_logprobs`/`embeddings`**——IFD/chr_min 类 logprob 数据选择若复活，那部分接口需在 `experimental/` 内单独保留或重写，不能指望 `llm_backup` | 收敛到单一蒸馏路由机制；生成享受 student/teacher 蒸馏；消除重复的推理后端抽象 |

### 建议增加（真正缺失的清洗能力）

#### 已列（清洗器层）

| ID | 项 | 缺口 | 改动 | 预期收益 |
|----|----|------|------|----------|
| **D1**（离线） | 近重复去重 MinHash-LSH/SimHash | `DedupFilter` 只做前缀 md5 精确去重，改一字的近重复全漏 | 扩展 `DedupFilter` 或新增 `NearDupFilter`（`datasketch` 轻依赖）。**限离线批处理阶段**（需全局视图）；实时 per-batch 主路径不启用，否则局部视图导致误杀严重 | 相似轨迹被挡，多样性上升；离线做，避免在线误杀 |
| **D2**（离线） | 基准去污染 decontamination | train/test n-gram overlap **完全没有** | 新增 13-gram 重叠过滤，比对**静态** benchmark n-gram 索引。**限离线**或**只打标不删**，避免实时流误杀正常样本 | 评测不被污染，指标可信 |
| **D4**（中） | 语言识别 langid/fastText | 只有 `cjk_ratio` script 比例，粗糙 | 轻量 langid 过滤 | 中英限定更可靠，混语噪声下降 |
| **D5**（低，可选） | 结构化噪声轮识别 | 现有 heartbeat 靠关键词（对 openclaw/OpenHands 格式已够用，见 P2 撤销）；仅当出现无关键词的结构性噪声轮时才需要 | 极短轮 + 高重复 + embedding 距离（复用 D1 基础设施） | 覆盖无关键词的噪声轮；非当前痛点 |

> **D3（agent 死循环接线）已废弃**：review 指出 `preprocessor` 与 `verifier` 当前**零互相 import**（grep 确认），让 `DeadLoopFilter` 去 import `HardScorer`（一个 RL reward `Verifier`）会破坏模块边界、且职责串（清洗器 vs 打分器）。正确路径并入 **D6/D7**：新增打标/评分 preprocessor，agent 死循环由其中的确定性 check 覆盖。

#### 新增（达成「干净 + 每轮评分」最终目标所需的整段能力）

> 对标业界标准 agent 数据流程（Llama-3 / DeepSeek-V3 / Nemotron / Tulu-3 / AgentInstruct / ToolBench）。目标五属性映射：无不良信息→D8/D9、无废案→D6、无重复冗余→D1+D6(轨迹内)、无心跳→已有、每轮评分→D7。

| ID | 项 | 属性 | 缺口 | 改动 | 预期收益 |
|----|----|------|------|------|----------|
| **D7**（高，核心） | 每轮评分打标 preprocessor（**只打标不过滤**） | 每轮评分 | `verifier`（per-round `HardScorer` + per-segment `RubricVerifier`）+ `aggregation`（round→segment→trajectory）**基础设施现成但未接线**；`aggregation.py:27` 明说编排器 `TrajectoryScorer` 未实现 | **新增 preprocessor**（mapper，从不 drop）：`Segmenter → HardScorer(逐轮) → RubricVerifier(逐段) → aggregation → 分数写回 `user_data``。分数、`score_confidence`、安全标全部作为 `(key, pack_value(v))` 追加进 `user_data`（见架构前置 A5）。`RubricVerifier.score_detail()` 已返回完整 `ScoreDetail`，`__call__(trajectory)` 兼容逐行调用 | 直接产出「每轮评分」的 trajectory；打标与过滤解耦，D6/D8 只读标签 |
| **D7c**（高，核心） | 评分校准探针 + 客观→主观重评（自进化，无人评） | 每轮评分可信度 | 自进化框架**不能靠人评对齐**；未校准的分会系统性放大 judge 偏见 | 用三个**自动**信号合成 per-segment `score_confidence`：①**teacher-student 一致性**（复用 `llm_backup` 已收集的 `(student, teacher, match)`）②**结果锚定**（`HardScorer` 确定性 check 当弱标签探针）③**voting 方差**（`RubricVerifier` 已有 voting，导出方差）。**关键**：当客观（硬 check）与 LLM 主观**不一致**时，**把客观结果注入 rubric 的打分上下文，让 `RubricVerifier` 重新评分**（不是简单降权，是带硬信号修正的二次评分） | 分数可信度自动量化；客观事实纠偏主观判断，闭环收敛；零人工 |
| **D6**（高） | 轨迹成败判定（过滤废案）——**纯读标签 filter** | 无废案 / 轨迹内冗余 | 无 outcome verification：失败/绕圈/工具全错/最终答案错的轨迹留在训练集 | **不自己算分**，只 `user_data_get(row, 'traj_score')` 等标签跟**阈值**比 → 判废案 drop（依赖的是 D7 已写好的**数据标签**，不是 D7 的代码，靠 pipeline 列表顺序保证 D6 在 D7 后）。**阈值先拍默认值，实测回收分布后回调**（不做人评标定） | 废案不进训练集；与打标解耦，无模块依赖 |
| **D8**（高） | 安全/毒性评分（复用 rubric） | 无不良信息 | 只有 `RefuseFilter`（拒答正则）+ 敏感词表，无 toxicity/safety 覆盖暴力/仇恨/成人/越狱成功 | **复用 `RubricVerifier`**：把安全维度作为一组**固定 `RubricItem`**（暴力/仇恨/成人/越狱成功/隐私泄露）注入 stage-2 打分，走现有 `_score_with_voting` + `_aggregate`；低于阈值判不良。**无需新分类器/新依赖** | 安全过滤召回远超敏感词表；与 D7 共用打分基础设施 |
| **D9**（中） | PII 真脱敏（激活现有 Presidio） | 无不良信息 | `PIIPresidioFilter` 存在但曾因**慢**去掉（spaCy NER 是瓶颈） | 加回来但**纯 regex 模式**：现 `IGNORED_ENTITIES` 已忽略全部 NER 实体（PERSON/LOCATION/ORG…），只留 regex 标识符（邮箱/电话/证件/银行卡）→ **可不加载 spaCy**，绕过 NER 瓶颈，速度问题基本消除 | 邮箱/电话/证件等真 PII 脱敏，且不拖慢管线 |
| **D10**（中） | 治理层：provenance（血缘字段） | 可追溯 | 无血缘字段（source/teacher_model/timestamp） | 每条 trajectory 把血缘作为 `(key, pack_value(v))` 写进 `user_data`（蒸馏场景 teacher/student 版本）。**批次归因/数据卡暂缓**（backlog，见「不做」） | 可追溯；对标 Nemotron/Dolma 但先只做血缘字段 |

> **架构前置 A5（去 DAG 的正解）**：所有评分/安全/血缘标签统一写进 **`user_data` 信封**，把「评分」与「过滤」解耦成「**打标 mapper（D7/D8/D10，从不 drop）+ 末尾纯读标签 filter（D6）**」。这样 D6→D7 是**数据依赖**（D7 写标签、D6 读标签），靠**线性 `QualityPreprocessor` 的列表顺序**保证，**无需 DAG、无模块互相 import**。
>
> **PyArrow 硬约束**：`user_data` 必须是 **`List[Tuple[str, str]]`**，**不能用 dict**（HF `datasets` 的 PyArrow 后端对异构/嵌套 dict 序列化有问题）。已核实这是仓库现有官方约定 —— `twinkle/data_format/trajectory.py:18-19`（`user_data: List[Tuple[str, str]]`，注释 "PyArrow-stable encoding: each entry is (key, json.dumps(value))"），写用 `pack_value(v)`（JSON 字符串，值可为任意结构但对外恒为 `(str, str)`），读用 `user_data_get(row, key)`。每轮分数可用 `(f'round_{i}_score', pack_value(...))` 或 `('round_scores', pack_value([...]))` 形态。
>
> 确定性 check 复用：D6/D7/D8 都要 `HardScorer` 的 LLM-free check。可抽到无依赖公共层（如 `twinkle_agentic/agent_checks.py`）供 verifier 与新 preprocessor 各自依赖；用户已确认也可接受 preprocessor→verifier 单向依赖，则公共层后置。

### 明确不做（自进化框架的取舍）

| 项 | 为什么不做 |
|----|-----------|
| 人评校准对齐 | 自进化框架不 scale；改用 D7c 的三信号（teacher 一致性 + 结果锚定 + voting 方差）替代 |
| 批次间质量归因 / 数据集版本 diff | 暂缓（backlog），当前不阻塞可用性 |
| 管线内数据配比 / 分层采样 | 移到**训练时的 sampler** 消费 `user_data` 标签，清洗管线只负责打标 |
| DAG / 阶段化编排引擎 | 用 A5 的「打标 + 末尾读标签」拍平成线性，不引入 DAG |

---

## 四、改动项汇总与优先级

| 优先级 | 项 | 类型 | 是否引入依赖 |
|--------|----|------|--------------|
| P0 前置 | **A5** | `user_data` 信封（去 DAG，其余目标项的地基） | 否 |
| P0 必做 | P3 | 数据损坏（丢 reasoning） | 否 |
| P1 目标核心 | D7, D7c, D6, D8 | 每轮评分 / 校准 / 废案 / 安全（接线 verifier） | 否（复用 verifier） |
| P1 高 | P7, A1, A3 | 一致性 / 结构重构 | 否 |
| P2 中 | D9, D10 | PII 脱敏 / 血缘 | presidio(D9) |
| P2 中 | P1, P5, P6, P8, P10, A2, D4 | 语义/漏检/降耦合 | langid(D4) |
| P2 中（仅离线） | D1, D2 | 去重/去污染（防实时误杀） | `datasketch`(D1) |
| P3 低 | A4, R1, R2, R3, R4, R5, D5 | 清理/重定位/增强 | 否 |

**合计 22 项**：实现问题 6（P1、P3、P5–P10）、架构 5（A1–A5）、功能增删 11（R1–R5 + D1/D2/D4/D5 + D6/D7/D7c/D8/D9/D10）。原 P2/P4 撤销，P11/P12 归入 R1，**D3 废弃**（并入 D6/D7）。

---

## 五、改动后预期整体收益

1. **达成最终目标（干净 + 每轮可信分）**：A5 统一标签信封 → D7 分数写回每轮 → D7c 自动校准 + 客观纠偏主观 → D6 读标签滤废案 → D8 安全 rubric → D9 PII；heartbeat 已有、D1 离线去重。五属性齐活，且分数带 `score_confidence`。
2. **去 DAG**：A5 把「打标 mapper + 末尾读标签 filter」拍平成线性 `QualityPreprocessor`，D6↔D7 只有数据依赖、零模块互调，无需 DAG 引擎。
3. **数据正确性**：P3 消除 reasoning 被静默剥离；P1 末轮悬空 tool_call 一致处理。
4. **复用而非新建**：D7 复用 `verifier`+`aggregation`；D7c 复用 `llm_backup` 一致性 + `RubricVerifier` voting；D8 复用 rubric 固定项；D9 纯 regex 免 spaCy —— **几乎零新依赖**。
5. **可维护性 / 一致性**：A1–A4 契约清晰、依赖分层；R1–R5 砍死代码；P7 统一 `tool_calls` 判定；D10 血缘可追溯。

> 落地顺序建议：
> 1. **A5**（定 `user_data` 标签信封 + list-of-tuple/`pack_value` 约定）→ 所有目标项的地基。
> 2. **P3**（防丢数据，无依赖）。
> 3. **D7**（每轮评分打标 mapper，接线 verifier+aggregation，写回 `user_data`）。
> 4. **D7c**（三信号校准 + 客观→主观重评）→ 让分数可信。
> 5. **D6 + D8**（读标签滤废案 + 安全 rubric，阈值先拍后测）。
> 6. **D9**（PII 纯 regex 加回）→ **D10**（血缘字段）。
> 7. **A1 + A3**（结构地基）→ 其余（P1/P5/P6/P8/P10/D4）视数据分布投入。
> 8. **D1 + D2** 放离线批处理阶段单独跑；配比放训练 sampler。
