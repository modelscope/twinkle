# Memory 线设计（自进化蒸馏框架的第二条线）

> 状态：**设计稿，未实现**。本文件是讨论沉淀，供后续实现参考。
> 与主线（清洗 + 评分 + 在线蒸馏）共用同一批 trajectory 与同一套 verifier / `user_data` 信封 / `llm_backup` / D7c 校准思想，不另起炉灶。

---

## 0. 背景与定位：双线

同一份生产 trajectory，榨出两种产物，固化到不同位置、不同时间尺度：

| | 线 A：训模型（慢记忆） | 线 B：总结 memory（快记忆） |
|---|---|---|
| 固化位置 | 模型权重 | 外部 memory store |
| 生效方式 | 蒸馏/微调后永久具备 | 推理时检索注入 context |
| 时间尺度 | 天/周（攒批再训） | 秒/分钟（写完即用） |
| 改什么 | 参数 | 行为（不改参数） |
| 装什么 | 可泛化的**技能/模式** | 易变的**事实/偏好/近期上下文** |

**line B 的存在理由**：在线蒸馏有延迟（攒数据→训练→部署），空窗期 student 学不到新东西；memory 立刻起作用填这个空窗，等能力被训进权重后再从 memory 退休。

> **line B 有两种载体（同为"不动 base 的外挂补充"，见 §11.6）**：
> 1. **文本 memory**：检索注入 context —— 本文 §1~§9 讨论的主形态，适合**易变事实/实体/偏好**（一条即用即改，天然带"检不到就拒答"信号）。
> 2. **参数化 memory（LoRA，base 冻结）**：把能力挂成可插拔 adapter —— 适合**可泛化的行为模式**（如"边生成边查错"），在已部署 vLLM 里边际显存/进程成本最低、可回滚（换 adapter ≈ 删一条 memory，而非重训 base）。
>
> 二者与 line A 的分界是"**动不动 base**"：line A 改 base 权重（有遗忘风险，故慎重、攒批离线做）；参数化 memory 虽然也是"权重形态"，但**base 全程冻结、只挂 LoRA**，本质仍是 line B（可插拔、可回滚、即挂即用）。参数化 memory 与文本 memory 的取舍见 Substrate Asymmetry（§11 反方证据）：参数化擅长行为/风格，文本检索擅长事实/拒答。

**双线共用的唯一“质检车间”**：现有 `preprocessor + verifier`。它产出的 `traj_score / round_scores / confidence / intent / safety` 同时作为两条线的准入闸门，不重复造。

### 分工判据（什么进 A，什么进 B）
用 **可泛化性 × 稳定性** 分流：
- 进 A（训权重）：高频、可泛化、稳定 —— 能变成“技能”的。
- 进 B（memory）：低频 / 易变 / 实体绑定 —— 只能当“事实/上下文”的。训进权重会过拟合具体实体且会过时。

### 消费者与作用域（已确认）
- **消费者**：两个都要 —— ① 本地 student 推理时检索注入；② 辅助置信度路由。
- **作用域**：两层 —— 用户级（个性化：偏好、历史）+ 全局级（跨用户通用事实/模式）。

---

## 1. 存储与检索选型（基于现有代码，零新增重依赖）

现有可复用：
- **存储** `twinkle/server/state`：memory / file / redis 三后端，带 TTL、`update_atomic`（原子）、`keys(pattern)`（通配）。
- **embedding** `preprocessor/experimental/llm_backend.py`：`OpenAIBackend.embeddings`（走 HTTP，dashscope 有 embedding 接口）。
- **没有**专用向量库（无 Milvus/Qdrant/faiss），但 D1 的 MinHash 说明“近似检索土办法”本项目可接受。

**结论（最小可跑，可后续换库）**：
```
MemoryStore
├── 用户级（结构化 KV/标签）── 复用 StateBackend（FileBackend 默认，可切 Redis）
│      key = mem::user:<uid>::<kind>::<slot>；精确/前缀匹配；零额外依赖
└── 全局级（向量语义召回）── 条目存 StateBackend；embedding 用 OpenAIBackend.embeddings
       召回 = 应用层 cosine top-k（初期 O(n)，涨了再换 faiss/Qdrant）
       抽象出 VectorIndex 接口，后端可替换
```
- 用户级以**结构化 KV** 为主（实体绑定要精确命中，向量会招噪声）。
- 全局级以**向量召回**为主，再用结构化标签（intent/domain/min_score）过滤。

### MemoryItem（统一条目）
```python
@dataclass
class MemoryItem:
    id: str
    scope: str            # 'user:<uid>' | 'global'
    kind: str             # 'fact' | 'preference' | 'action_pattern' | 'anti_pattern'
    key: str              # 结构化槽位（用户级精确匹配）；全局级可空
    content: str          # 供注入 context 的自然语言
    embedding: list       # 全局级语义召回用
    # 毕业机制 / 效用 所需元数据（先埋点）
    hit_count: int
    source_score: float   # 来源轨迹 traj_score（写入门槛）
    created_at: int
    last_hit_at: int
    graduated: bool        # 是否已训进权重 → A→B 退休标记
```

### MemoryStore 接口
```python
class MemoryStore:
    def write(item): ...
    def retrieve(query, scope, *, intent=None, min_score=None, k=5) -> list[MemoryItem]:
        # 用户级：KV/标签精确匹配；全局级：向量召回 + 标签过滤
    def mark_hit(item_id): ...   # 命中打点（喂效用/晋升）
    def retire(item_id): ...     # A→B 退休（标记 graduated）
```

---

## 2. 核心目标：抽取器越来越专业（区别于 mem0/reme）

### mem0 / reme 的做法与天花板
- **mem0**：两步固定 LLM 流水 —— Extraction（固定 prompt 抽事实）+ Update（ADD/UPDATE/DELETE/NOOP 去重消歧）。
- **reme**：分 personal / task memory，带 reflection 从成败轨迹提炼经验，检索时 rerank。
- **共同天花板**：抽取器**静态**（extraction prompt 永远 day-1），**没有下游效用反馈回流到抽取器**，质量靠 LLM 当下自评。→ memory 只会“越攒越多”，不会“越来越专业”。

### 我们的突破口
下游有**真实效用信号**（蒸馏是否受益、路由是否更准）+ 有 **verifier/rubric 打分**。→ 可以让抽取器被“经下游验证过的 memory”反过来训。

### 三个进化层次（由易到难）
1. **meta harness 学会“选对抽取器”**（最快见效）：contextual bandit，按轨迹特征分派抽取器；反馈=各抽取器产出 memory 的效用分。**纯统计即可**，不需训练。
2. **抽取器“知道什么是好 memory”**：用 rubric 给 memory 打分，维度对准**可用性**（自包含 / 可泛化 / 可操作 / 不冗余），当抽取的软靶 + 写入门槛。
3. **抽取器本身被蒸馏得更强**（最终形态，与主线同构）：用**下游效用**给抽取器输出打真标签（不是 rubric 自评），正/负样本微调抽取器（同 `llm_backup` 机制，被蒸馏的是“抽取器”这个角色）。

### 关键：rubric 分 = 训练靶，效用分 = 真值锚
- **rubric 内在分**：即时、便宜 → 抽取软靶 + 写入门槛。
- **下游效用分**：滞后、稀疏、客观 → 校准 rubric、训练抽取器的真值锚。
- 可用效用分**反向校准 rubric**：某维度 rubric 高分但效用低 → 自动降权（复用 D7c“客观校准主观”）。
- ⚠️ 坑：别让 rubric 自评当最终真值 —— “看起来专业 ≠ 用起来有用”，同 “模型不知道自己不知道”。

### 其它有效手段（业界/研究验证）
- **失败/反思挖掘**：失败轨迹信息量常更大（“这么调工具会报错”是高价值 anti_pattern）。见 §4。
- **巩固/合并（consolidation）**：定期把多条相关碎片 memory 合并升华成更抽象的通则 → 更泛化，是 B→A 晋升头号候选。
- **对比式抽取（contrastive）**：`llm_backup` 已收集的 (student 错 / teacher 对) 配对 → 专抽“teacher 做对而 student 做错的那步差异”，精准命中能力缺口。
- **引用计数衰减**：长期不命中的 memory 降权/淘汰，防历史噪声拖累。

---

## 3. 毕业机制（B↔A 梯度；先设计不实现）

memory 是权重的“预备队/退休区”，双线互相喂：
```
生产轨迹 ─(清洗+评分)─┬─► line B: 抽成 memory  → 立刻可检索（快）
                      │        │
                      │        └─ 高频命中 & 已泛化 ──► 晋升为训练样本
                      │                                   ↓
                      └────────────────────────► line A: 攒批蒸馏进权重（慢）
                                                         ↓
                                              权重已掌握 ──► 对应 memory 退休（stale/删）
```

- **B→A 晋升**（同时满足）：`hit_count ≥ N`；`kind==action_pattern` 或被判可泛化（排除实体孤例）；`source_score ≥ 阈值` 且 safety 通过。→ 还原成训练样本进 line A。
- **A→B 退休**：一批训练部署后做**回归检验** —— 让新 student 在**不给该 memory** 时回答，若已能答对（verifier 自动判）→ 标记 `graduated`，移出检索池。
  - ⚠️ 依赖“新权重能否自答”的自动判定，正是“模型不知道自己不知道”；判定靠主线 verifier/置信度，**不能靠模型自评**，否则误退休。
- 价值：普通 memory 系统缺“遗忘/毕业”机制会无限膨胀 + 过时；有 line A 就天然有毕业出口。

---

## 4. 失败挖掘与归因（最硬的一块）

### 现状问题（必须改 pipeline）
按当前 pipeline，**失败轨迹会在打分前/后被丢，不会自动留存**：
- 退化性失败（死循环/复读/心跳/token soup）在 `DeadLoopFilter` 等**硬过滤前置步**就丢了（实测某片 `DeadLoopFilter: 25->2 dropped 23`），**没机会打分**。
- 能力性失败若侥幸过硬过滤，会被尾部 `TrajectoryOutcomeFilter` 按 `low_traj_score` **删除**。

### 归因难题（用户戳中的核心）
一条带 memory 的失败，根因有三，且在最终轨迹上**长得一样**：
1. 模型能力问题（没 memory 也做不对）；
2. memory 缺失（有正确 memory 就能对，但没检索到/库里没有）；
3. **memory 误导**（检索到的 memory 错/过时/不相关，把模型带沟里）。

单看一条失败轨迹**无法区分** —— 缺反事实。若把 memory 误导当“能力缺口”抽成 anti_pattern → 坏 memory 的锅上再叠 memory → **无限污染循环**。这正是 mem0/reme 不敢做 memory 归因的原因。

### 破局：归因在“memory 使用现场”做，不在“挖掘阶段”猜
把 memory 注入当成一次**可对照的干预**。对同一 query 跑带/不带 memory：

| 不带 M | 带 M | 归因 | 动作 |
|---|---|---|---|
| 失败 | 失败 | model_capability | 抽 anti_pattern / 进训练线 |
| 失败 | 成功 | memory_helpful | M 加分 |
| 成功 | 失败 | **memory_harmful** | **退休该 M，不抽 anti_pattern** |
| 成功 | 成功 | memory_redundant | M 中性/可淘汰 |

**归因是推理时记录的，不是挖掘时推断的。**

### 剪枝（用户务实取舍）
- **token soup / 退化性失败**：根因是生成层退化，几乎与 memory 无关 → **直接丢，不留、不归因**（对照必然“两边都烂”，得不到 memory 信息）。
- **归因预算（影子对照 + 留痕）只投在“能力性失败”**。
- 判“垃圾 vs 值得归因”用**现成便宜过滤器**（DeadLoop/TokenSoup/SpecialChars 命中 = 垃圾直接丢）；判“模型问题 vs memory 问题”才用贵的影子对照。两层闸门，贵的只处理少数。
- 放弃项（暂）：“反复调错工具”的死循环表面像 token soup 实为高价值 anti_pattern，但难自动区分、占比不高，先不捞。

### 落地（部署已确认：可做 A/B 影子跑；memory 注入尚未上线 → 从零把留痕设计进去）
注入路径三件套（第一天就装）：
1. **影子对照采样**：每次带 memory 推理以概率 `p`（约 5%）触发“不带 memory”的影子；两条都用 verifier 判成败，填 2×2 表。常规请求只留痕不对照。
2. **注入留痕**（`user_data` 信封加字段）：
   ```
   injected_memory_ids: [...]
   memory_shadow: {ran: bool, without_mem_outcome: 0/1}
   inference_outcome: 0/1        # verifier 判
   attribution: helpful|harmful|redundant|model_capability|unknown
   ```
3. **归因判定**：采样命中 → 按 2×2 表当场出结论；未命中 → 挂 `unknown`，靠历史反事实给**概率性**归因（低置信，仅用于排序“哪些 M 优先做对照验证”）。

### FailureMiner 的干净输入
- 只从 `attribution == model_capability` 的失败抽 anti_pattern（已排除 memory 误导）。
- `memory_harmful` → 触发删 M，不抽新 memory。
- `unknown` 且历史高度怀疑 harmful → 排进影子对照队列验证，不直接抽。
- 用 `round_scores` 定位**失败转折点**（哪轮突然掉），anti_pattern 抽那个点。
- ⚠️ 准确率 > 覆盖率：一条错的 anti_pattern 会主动误导模型 → **宁可不抽，不错抽**；噪声/环境偶发失败不抽。

---

## 5. 下游效用分（memory 飞轮的燃料）

### 定义（已确认：outcome 用主线 verifier；粒度 per-item）
效用 ≠ memory 内在质量，而是**边际因果贡献**：
```
utility(M) ≈ E[ verifier(轨迹) | 注入M ] − E[ verifier(轨迹) | 不注入M ]
```
`verifier(轨迹)` 用主线连续 `traj_score`（比 0/1 更细，能测“0.6→0.85”的边际改善）。它是 §4 归因的**连续版**（2×2 离散表的加权量）。

### 五种信号（弱→强）
- **A 反事实成功率差**（5% 影子对照）：因果最干净、正负都能测；稀疏 → **真值锚**。
- **B teacher 追平差**（`llm_backup` 免费捎带，teacher 不带 memory）：student 带 M 后 match=True 是否上升；覆盖 llm_backup 采样流量。
- **C 归因离散效用**（2×2 → +1/−1/0/0）：A 的离散版，够 bandit 奖励用。
- **D 检索命中×结果**（全流量、便宜、**有选择偏差**）：不能单飞，只当排序候选，靠 A 去偏。
- **E 多轮内即时行为信号**（`round_scores`：工具一次成功率、少走弯路、少重试）：细粒度，尤适 action_pattern。

### 组合（便宜信号打底 + 稀疏真值锚校准，同 D7c）
```
效用分 = f(观测代理 D, teacher 追平 B, 行为 E)  ── 用硬对照 A/C 校准去偏
```
1. 日常：累积 D+B+E 加权分（便宜、全覆盖、有偏）。
2. 校准：5% 对照 A 产出无偏真值，回归；若某类代理分系统性高于真值 → 自动降权 D。
3. 不确定性探索：代理与真值分歧大/方差大的 M → 提高其对照采样率。

### per-item 聚合器（唯一要新写的东西）
```python
MemoryUtility(memory_id):
    n_hits; proxy_sum(按相关度加权); n_controlled; delta_sum
    utility_hat ∈ [-1,1]; confidence(样本量+锚一致性); last_hit_at
```
融合：对照足→用无偏 `delta_sum/n_controlled`；对照少→用 proxy 减去同类已知偏差；分歧/低置信→抬高对照采样率。
per-extractor 分**不单独采**，从 per-item **聚合**上来（某抽取器所有 memory 的 `utility_hat` 均值），零成本供 meta harness。

### 驱动的动作
- 退休/清坏：`utility_hat<0` 且置信够 → 删（memory_harmful 自动闸门）。
- 晋升：`utility_hat` 高 + 高频 + 可泛化 → 进训练池。
- 冷启动保护：新 M 低置信给保底曝光 + 时间衰减防老 M 霸榜。

### 三个坑
1. 选择偏差（D 的病）：必须 A/B 去偏，D 不单飞。
2. 信用分配：一条轨迹多条 M → 初期均摊或按检索相关度加权，别急上 Shapley。
3. 反馈自强化：高效用被检索更多→分更高→更多… 可能锁死 → 时间衰减 + 强制探索低曝光。

### 前提
地基是 **verifier 判 `traj_score` 的质量**。rubric 打分噪声大则整个飞轮歪 → **先坐实 verifier 打分可靠性，再上效用分**（与“修 TrajectoryScorer 打分区分度”同一条线）。

---

## 6. meta harness（管理抽取器；contextual bandit）

### 定性：contextual bandit，不是全 RL；初期连 bandit 都先不上
- 无状态转移（选抽取器不影响下一条外部流量）→ 不需要 Q-learning/PG。
- 是 contextual bandit：轨迹特征=context，选抽取器=action，效用=reward。
- **reward 延迟极大**（几天）→ 先做**离线统计的 bandit**（按 context 分桶统计各抽取器历史效用 + ε 探索），成熟后再升 LinUCB/Thompson。

### 状态（context，全复用 preprocessor 标签）
`intent` / `traj_score` 分桶 / `round_scores` 形状 / 轨迹长度·段数 / 是否 agent(`is_agent_row`) / domain。
- 简版：离散化组合成桶 key，如 `(intent=tool_call, traj_hi, is_agent)`。
- 升级：特征向量 + LinUCB。**初期分桶就够，别急上向量。**

### 动作（两级）
- 一级（必做）：从 `{fact, action_pattern, preference, failure_mine, none}` 选一个/几个。**`none`（不抽任何 memory）是合法动作** —— 省成本 + 避免噪声。
- 二级（成熟后）：抽取参数（粒度/数量/是否跨轮聚合），初期固定默认。
- 约束：**动作空间要小**（个位数~十几个），否则永远冷启动。

### 奖励
```
reward(context, action) = agg{ utility_hat(M) : M 由该 action 产出 }
```
- `agg` 用均值，可选覆盖惩罚（抽太多低效用条目扣分，鼓励精不鼓励多）。`action=none` reward=0 作基线。
- **延迟处理**：批式/异步。抽取时记 `(context_bucket, action, [mem_ids])` 到待结算表；效用成熟后回填、更新桶统计。
- 信用分配已在 per-item 层解决，bandit 直接用聚合值。

### 探索
- 简版 ε-greedy，ε 随桶样本量衰减；新桶/新抽取器强制均匀试几次。
- 升级 Thompson/UCB，用奖励不确定性驱动（与效用分“低置信多做对照”共用信号）。
- **协同**：bandit 探索性选了冷门抽取器产出的 memory，效用最不确定 → 应**优先安排影子对照**去测它，否则探索白探。

### 骨架
```python
MetaHarness:
    policy: dict[context_bucket → dict[action → (reward_mean, n, confidence)]]  # 存 StateBackend
    select(traj_labels) -> action           # ε-greedy over bucket
    log_decision(context, action, mem_ids)  # 待结算表
    settle(mem_id, utility)                  # 效用成熟后回填 policy
```

---

## 7. 稳定性：harness 会漂移，怎么关进笼子

“更新 harness 后不稳定”拆成三种病：
1. **策略震荡**：噪声+延迟反馈做了过自信更新 → 来回摆。
2. **反馈自锁**：偏向某抽取器→只有它有反馈→别人永远没数据→越锁越死（off-policy 经典病）。
3. **非平稳漂移**（本系统独有、最麻烦）：student 在被持续蒸馏、memory 库/流量在变 → 上周最优抽取器这周可能就不对。普通 bandit 假设平稳，这里不平稳。

### 对策：harness 从“决策者”降级为“建议者”，更新慢、可回滚
1. **冻结基线 + 影子上线**（治 2、防炸）：新策略先只“建议”，实际按固定基线执行，记录“若听 harness 会怎样”；证据表明稳定优于基线才灰度切；永远保留基线兜底。
2. **慢更新 + 迟滞带**（治 1）：桶样本 `≥ N` 才更新，否则用先验；只有显著且持续优于当前才切换（margin + 连续几批）。
3. **强制探索地板**（治 2）：每个抽取器保底 `ε_min` 曝光永不归零，防自锁、且能先发现漂移。
4. **滑动窗口/时间衰减**（治 3）：效用统计只用近期窗口/指数衰减，让 harness 跟随漂移重学。

### 更重要：先稳定可用，再进化 —— 分阶段把不稳定源头关掉
- **阶段0 静态 harness（先上，零 bandit）**：固定人写路由表（`tool_call→action_summarizer`、`code→action+fact`、`低分失败→failure_mine`、`其他→fact`）。不学习不更新、完全确定。先把“多抽取器按类型分派”的**结构**跑通、攒效用数据、当 bandit 基线。**无任何不稳定性。**
- **阶段1 离线 bandit，仅离线复盘时更新**：攒够数据后离线做 off-policy 评估，只有证明稳定优于静态表，才把新策略**作为新静态表发布**上线。策略更新发生在**离线、可审、可回滚**节点，不在线漂移。
- **阶段2（远期，可选）**：在线自适应。

### 分层可降级 mode（把“新方案风险”关进笼子）
```
harness.mode = 'static'  # 固定路由表，永远兜底
             | 'shadow'  # bandit 只建议不执行，收集证据
             | 'canary'  # bandit 接管 x% 流量
             | 'live'    # bandit 全量（需离线验证达标才允许）
```
出问题一键降回 `static`。

### 诚实边界
- 阶段0 静态表需人工先验（哪个 intent 配哪个抽取器）—— 这不“自进化”，但是冷启动正确起点。
- 非平稳漂移无完美解，滑动窗口只缓解；靠“离线定期复盘 + 可回滚发布”管理，不追求永远正确的在线策略。

---

## 8. 现有资产映射（实现时几乎不用造轮子）

| 需要的能力 | 复用现有 |
|---|---|
| 存储（用户级 KV、策略、效用状态） | `twinkle/server/state`（memory/file/redis + TTL + update_atomic + keys） |
| embedding（全局向量召回） | `preprocessor/experimental/llm_backend.py::OpenAIBackend.embeddings` |
| 抽取器（fact/action/pattern） | `twinkle_agentic/summarizer/*`（都走 `llm_backup` 蒸馏，可 per-type LoRA）；`pattern_summarizer` 现为空 → 做“可复用模式”抽取，是 B→A 晋升头号候选 |
| 抽取器蒸馏 | `llm_backup`（被蒸馏对象换成“抽取器”角色） |
| outcome / 写入门槛 | 主线 verifier / rubric / hard_scorer 产出的 `traj_score / safety` |
| 转折点定位 | `round_scores`（TrajectoryScorer 已产出） |
| 校准逻辑（客观校主观） | D7c 思想直接搬 |
| 标签信封 | `preprocessor/label_schema.py` + `user_data`（PyArrow 稳定）；新增 `injected_memory_ids / memory_shadow / inference_outcome / attribution / memory_utility` |
| 配对（对比抽取 / teacher 对照） | `llm_backup` 已采 (student, teacher, match) |

**唯一新写**：MemoryStore + VectorIndex + FailureMiner + per-item 效用聚合器 + MetaHarness（含 static/shadow/canary/live mode）。毕业机制先留接口 + docstring + 判据常量，逻辑 `NotImplementedError`。

---

## 9. 待定 / 下一步（未拍板）

- 静态路由表的具体先验规则（intent → 抽取器映射）细化。
- 待结算表 schema、桶设计粒度（先粗按 intent，数据多了再细）。
- 检索/注入策略本身：检索几条、排序、注入到 context 哪个位置。
- 双线去重边界：同一高分轨迹既进训练池又进 memory，B→A 晋升时如何避免重复训练。
- memory 命中对置信度路由的**方向**：命中→更敢自答，还是命中→说明是薄弱区更该路由？（两种逻辑相反，需定。）
- 用户级 memory 的隐私/时效：覆盖写 vs 版本化；TTL。

---

## 10. Related Works（2026 检索，按本设计的轴归类）

> 检索源：arXiv（2026-06 ~ 2026-07 为主）。结论：本设计的**每一条主要思路都能在近半年文献里找到平行工作或验证证据**——这是好事（方向被验证、不孤立），差异化在于**把这些点在一个自进化蒸馏框架里闭环组合**，且共用主线 verifier / `llm_backup` / D7c，而非各做各的。下面按“对应本文哪一节”组织。

### 10.1 双线（context-space + parameter-space）—— 对应 §0

**DuoMem: Dual-Space Distillation**（arXiv 2606.29961）
- 具体做法（分三步离线 + 一步在线，见其 Fig.2）：
  1. **teacher 造料**：用 Qwen2.5-72B teacher 对 3,553 个训练任务各跑 3–4 次（共 11,546 实例，5 次重试内累计成功率 99%），得到 11,434 条成功轨迹；**故意 oversample 同一任务的多条不同解**以增多样性、防过拟合。
  2. **context-space 蒸馏（CD，训练无关）**：让 teacher（而非 student）对每条完成轨迹**离线生成 procedural memory 脚本**，存成文本 bank（整套任务几 MB）。推理时对新任务 d，用 `text-embedding-3-small` 算 d 与各条 memory 的 cosine，取 top-k **prepend** 进 student prompt。不改任何参数。
  3. **parameter-space 蒸馏（LoRA）**：只用**成功 teacher 轨迹**微调 student 的 LoRA（rank 8–32，α/r=2，base 冻结）。
  4. **组合**：先 LoRA 再 CD。ALFWorld 上 Qwen3-4B：No-Mem 4.3% → MemP(student 自产 memory) 55% → +CD 56.4% → +LoRA 72.1% → +DuoMem 77.9%（逼近 72B teacher 87.1%），只加 5.9M 参数、~12MB memory，且比 teacher 快 3×。**关键消融结论：CD 单独收益很小（+1.4），LoRA 才是大头（+17），两者组合还有超加性（>各自之和）。**
- 与我们的区别：DuoMem 的两轴都是**一次性离线固定**（teacher memory 生成一次、LoRA 训一次），**无在线闭环、无毕业机制、只用成功轨迹**。我们要：两轴在线持续、B↔A 毕业梯度、且**失败轨迹也要挖**。另外 DuoMem 的 CD 是"整任务级 memory 检索"，我们是 per-item 效用 + 归因过滤后的 memory。**它是我们双线最直接的可行性背书 + 起点基线**（甚至可先复现 DuoMem 当 line A/B 的 v0）。

**KbSD: Knowledge Boundary aware Self-Distillation**（arXiv 2606.29863）
- 想解决的问题：模型经常**不知道自己知不知道**——该答的时候瞎编（幻觉），不该答的时候硬答，或者明明该去查资料却凭记忆蒙。KbSD 想教会模型"划清知识边界"：**会的直接答、不确定的去检索、真不会的就说不会**。难点是普通 RL 只有一个"最后答对没答对"的稀疏奖励，没法教中间推理过程该怎么走。

- 怎么做（分三步，公式见原文 §3）：

  **① 给每个问题打三个"边界标注"**（这就是你问的"确定性/靠谱度怎么来的"）：
  - **参数化确定性 μ(q)**：拿**冻结模型、不给检索**，对同一问题**独立采样 N 次**，算答对（match 标准答案）的比例 `μ = (1/N)Σ I[match(yᵢ, a)]`。μ 高 = 答案本就在模型脑子里。**注意：这一步需要 ground-truth 答案 a。**
  - **语义稳定性 σ(q)**：这 N 次回答**两两之间的 embedding 余弦相似度求平均** `σ = mean cos(Enc(yᵢ), Enc(yⱼ))`。σ 高 = 每次都说同样的话（信念稳）；σ 低 = 每次瞎蒙都不一样。**这个不需要标准答案。**
  - **检索质量 ρ̂(q)**：对检索返回的证据打一个 retrieval-quality 分（原文没细化打分器，是可替换的相关性模型/reranker）。
  - **映射到四象限**：把 μ、ρ̂ 各卡一个阈值二值化 → 得"内部知识可靠 k / 检索充分 s"，组合成四种该有的行为：都行→**Integrated（整合）**、只检索行→**External（靠检索）**、只内部行→**Internal（信自己）**、都不行→**Refusal（拒答）**。σ 不进象限，只用来筛训练集（已知区留稳定的、未知区留不稳的），让边界更干净。

  **② 用"开小灶的自己"当老师做示范**：把上面标注 `(μ, σ, ρ̂, 目标象限)` 拼成一段 **hint 前缀，只喂给老师**；老师 = **同一个模型** conditioned 在 `[hint; q]` 上（且 stop-gradient，不回传老师）。学生 = 同一个模型但**看不到 hint**。因为老师多看了提示，能写出"知道分寸"的推理示范（该查就查、不会就认怂），学生就跟这个"开了天眼的自己"学——**这就是"信息不对称自蒸馏"，全程不需要更大的外部模型**。hint 只在训练用，推理时不给。

  **③ 学生怎么对齐老师——是 KL 蒸馏，而且按象限切方向**（这就是你问的"用什么 KL"）：token 级两种损失，本质是 KL 的两个方向：
  - **前向 KL（mass-covering，覆盖）**：在老师轨迹上最大化学生似然 → 学生尽量覆盖老师所有说法。
  - **反向 KL（mode-seeking，收敛）**：在**学生自己采样**的轨迹上匹配老师 → 学生收敛到老师主模式、抑制老师不支持的行为。
  - **分象限分配**：Integrated（只有一种正确整合、分布集中）→ **反向 KL**；Refusal（合理拒答说法很多、分布发散）→ **前向 KL**（别塌成一个模板）；External / Internal（要精准又抗噪）→ **前向+反向都要**，用 Pareto 加权自动解一个公共下降方向 α*（闭式解，不用手调系数）。这套蒸馏再和 GRPO 的稀疏 outcome 奖励**联合优化**（稀疏管"最终对不对"，稠密管"推理怎么走"）。

- 对我们有什么用：它正好治我们主线最头疼的病——**"模型不知道自己不知道"**，这直接决定置信度路由该不该把请求甩给 teacher；而且证明了**不用更大模型也能自我校准**（老师=学生+hint），与我们 `llm_backup` 的 student/teacher 同构。可直接借的两处：**训练置信度路由**、**A→B 退休判定**（退休本质就是判"新权重不给 memory 提示、自己能不能答对"，正好对应 μ）。
- **⚠️ 搬过来要改的地方**：KbSD 的 μ **依赖 ground-truth 答案**，而我们主线是**无 ground-truth 的生产流量**。所以不能照搬 μ，要用**主线的 rubric/hard verifier 分数、或 teacher 一致性**来替代"match 标准答案"这一步；σ（自洽度）可原样复用，因为它本就不需要答案。

### 10.2 memory 越来越有用 / 自优化抽取器 —— 对应 §2、§6

**SelfMem: Self-Optimizing Memory**（arXiv 2607.03726）
- 想解决的问题：现有 memory 系统（MemGPT/Mem0/MemoryBank）都是**人写死的 memory 流程**——"存用户画像""到 context 上限就压缩"这种固定规矩，换个任务就不合适、还得手动调。SelfMem 想"授人以渔"：**不给死规矩，让 agent 自己摸索"这个任务下 memory 该怎么攒"。**
- 怎么做（关键：搞清 refine 的到底是什么）：世界里有三样东西——
  1. **原始对话记录（transcript）**：存成 SQLite 表，**只读、永不改**，是事实唯一真相来源。
  2. **memory 工作区（workspace）**：agent 自己维护的一块**可读可写白板**，装什么结构由它自己定（用户画像 / 偏好列表 / 项目笔记 / 时间线，甚至一段压缩策略文字）。
  3. **一组固定的 memory 工具（四类）**：读 transcript（可跑只读 SQL）、读 workspace、写 workspace（加/替换/合并去重/精炼摘要/归档过时/记精确事实）、review（只诊断不改）。
  
  所谓 **"refine 自己的 memory 策略"，refine 的是白板里 memory 的内容与组织方式**（不是改模型权重，也不是改工具本身）：agent 跑一个 **inspect→write→review→revise（查→写→自查→修订）循环**——写完一条就用 review 工具自查"有没有过时/矛盾/没出处/难检索"，拿到诊断后回去改这块 memory。反馈是**多维不塌成单一标量**的（响应质量、token 数、成本、缓存命中），让 agent 在语言层面权衡"多存提升召回但涨成本、压缩省钱但丢细节"。全程**不动模型权重**（与 Reflexion/Self-Refine 同脉，靠语言反馈迭代）。BEAM 上 100K/500K/1M token 比最强基线 official score +0.165/0.141/0.134。
- 与我们的区别：先厘清一个易混点——**工具集是固定的，agent 不能改工具**；它和"普通调工具"的区别在于**调哪个、什么顺序、写什么、要不要重写全由 agent 按反馈自己决定，没有预设 SOP**（MemGPT/Mem0 = 人写好 SOP、工具是执行的手；SelfMem = 人只给工具+评价、让 agent 自己长出 SOP）。而 SelfMem 优化的是"**这一个 agent 怎么攒/用它自己那块 workspace**"（都在 prompt/流程层，权重不动）；我们的 meta harness 优化的是"**用哪个抽取器/参数把轨迹变成 memory**"，反馈是**下游效用（Δverifier）**而非 agent 自评，且更进一步要把好 memory **反向蒸馏回抽取器权重**。相同的是"给工具+反馈让它自进化"这个哲学，它的"inspect→write→review→revise 循环 + 多维不塌缩反馈"可直接借进我们抽取器的自评环节。

**MetaSkill-Evolve: Two-Timescale Recursive Self-Improvement**（arXiv 2607.05297）
- **先说最关键的一句，破除误解**：这篇**完全没有训练、没有梯度、没有 loss**。从头到尾只有**一个冻结的模型**（Gemma-4 31B），"进化"全靠**让这个模型反复读写几个 Markdown 文本文件 + 拿准确率做进化搜索**。所谓"skill/meta-skill"就是几份 `SKILL.md` 文件，不是模型参数。所以你问的"怎么训练、loss 是什么"——答案是**不训练、没有 loss**，它是"改文件 + 挑最好的文件"的搜索过程。

- 几个"模型"其实是同一个冻结模型扮演的**五个角色**（靠不同 prompt 区分，各读一份对应的 `SKILL.md`）：
  - **Analyzer**：看一条失败案例，诊断"为什么错"，打个标签。
  - **Retriever**：从别的分支里捞点"以前类似问题怎么改好的"当灵感。
  - **Allocator**：决定这一轮生几个候选改法（预算）。
  - **Proposer**：根据诊断，具体写出"skill 文件该怎么改"。
  - **Evolver**：把改动写进文件，并验证一下。

- 用什么数据 / 怎么"标注"：**不需要人工标注**。数据就是任务的 (输入, 参考答案) 样本，分成训练批和验证批。所谓"评分"是**自动的**——拿当前 skill 让 agent 去做验证批的题，**答对率就是这份 skill 的分数 `U(s)`**（`r∈[0,1]` 对着参考答案打分）。没有人在中间标任何东西。

- 读还是写：**主要是"写"侧的自我改进**——不断重写 skill 文件让 agent 做题做得更好；"读"只是 Retriever 去翻历史分支当灵感，不涉及检索优化。

- 两个时间尺度到底在进化什么（这是它唯一的新意）：
  - **快环（每轮）**：改 **task skill `s`**（"这个任务该怎么做"的说明书）。拿当前 skill 在训练批上找错得最惨的一题 → 五角色流水诊断+提改法 → 生成几个候选新 skill → 谁在验证批上**答对率更高**就留下（只有严格变好 `ΔU>0` 的才进池子）。
  - **慢环（每 H 轮）**：改 **meta-skill `m`**，也就是"**上面那五个角色自己的说明书**"。关键点：因为五个角色的说明书也是同格式的 `SKILL.md`，所以**用同一套五角色流水去改它们自己**（自己改进自己的改进方法）——这就是"递归自我改进"。判据是 **meta-productivity `P(m|s)`**：这个分支最近 H 个"孩子"平均提升了多少（`= 子代 ΔU 的均值`），衡量"这套改进方法还灵不灵"。
  - **选哪个分支继续进化**：打分 `η₁·当前分 U + η₂·这套方法的产出率 P + η₃·新颖度 N`（N 惩罚被选太多的分支，逼它去探索没试过的路线）。

- 效果：OfficeQA/SealQA/ALFWorld 比 No-Skill +23.5/+16.1/+1.9，比"只进化 s、不进化改进方法"的版本 +6.4/+8.1/+1.9（证明"连改进方法本身也进化"确实有额外收益）。

- 与我们的区别：它的思想（"**快改内容、慢改'改内容的方法'本身**"）正是我们 §6/§7 想要的——我们的 static→shadow→canary→live 就是它慢环的工程化 + 加了可回滚门控。**但它全程改的是文本文件、模型冻结、没有训练**；我们的慢环最终要落到**真的蒸馏进抽取器权重**（把好经验烧进参数，而非只改说明书），且用**下游效用（Δverifier）**而非"验证批答对率"当信号。一句话：**它是"用一个冻结模型玩进化搜索改 prompt"，我们要的是"把搜出来的好东西训进权重"。**

**COMFYCLAW: Self-Evolving Skill Harnesses**（arXiv 2607.01709）
- 领域 / 模型：图像生成工作流（ComfyUI）。**不训练模型**——用现成 LLM 当 agent、现成 VLM 当"验收员"，改的是外部的 skill 文件库。
- 怎么做（一个"边做边攒经验"的闭环，见其 Fig.1）：
  1. 给一个文生图需求，agent 通过**带类型的图编辑**（连节点、调参、加 LoRA）把 ComfyUI 工作流搭起来，跑出一张图。**非法的编辑会被自动撤销**（防止把流程改坏）。
  2. **VLM 验收员**把需求拆成一串"可观察的是非题"（比如"有没有三只手臂""风格对不对"），逐条判过没过 + 给个 0–10 细节分，合成一个标量分数；并把**没过的条目 + 哪里错了 + 具体该怎么改**回吐给 agent，指导下一轮修改。
  3. 跨很多需求跑下来，把"反复成功/失败的经验"提炼成**可复用的 Agent Skill（`SKILL.md` 文件）**，存进 skill 库，下次相关需求时**先只给 skill 摘要、需要时再展开全文**（渐进披露，省 context）。
- 数据 / 标注 / loss：**没有训练、没有 loss**，"分数"来自 VLM 验收员的是非题（自动，无人标）。四个 split×两 backbone×三模型下，比"只有验收员、不进化 skill"的基线高 4 分、比"完全不修改"高 10 分。
- 与我们的区别：它名字和"harness 管理可复用技能"的思路跟我们碎片1直接撞上，但它是**图像生成域实证**、skill 是"给 agent 复用的操作技能"；我们的"skill/抽取器"是**把轨迹变 memory 的工具**、且最终要蒸进权重。它的两个工程点可直接抄：**非法编辑自动回滚**、**验收反馈翻译成"可执行的修改建议"而非只给一个分**。

**UCOB: Credit-Aware On-Policy Bidirectional Self-Distillation**（arXiv 2606.29502）—— **与我们 §4 归因 + §5 效用最像的一篇，务必读透**
- 想解决的问题：检索来的"经验/skill"**不是万能的**——同一个模型，在情形 A 被这条经验帮到、在情形 B 反被它带沟里。所以"把'带经验的回答'当成永远正确的老师去教'不带经验的回答'"这个假设是**脆的、会把坏经验也学进去**。这正是我们担心的"分不清是模型问题还是 memory 问题"的学术版。
- 怎么做（大白话，这是**真·训练，有 RL loss**）：
  1. **同一个在线模型**，同一道题准备**两种输入**：带经验的（`P₊`）和不带经验的（`P₀`）。每道题各采一批 rollout，一半用 `P₊`、一半用 `P₀`，**两边都参与在线 RL 更新**。
  2. **在相同的"局面"上比谁做得好**：把两边 rollout 里**走到同一个中间状态**（论文叫 anchor-state）的记录凑成一组，各自算"从这一步往后的总回报"（return-to-go = 后续奖励的折扣和）。定义**同局面下的差值**`Δ = 带经验的平均回报 − 不带经验的平均回报`。
  3. **谁赢谁当老师，只在这一局面上教对方**：`Δ` 明显为正 → 说明这条经验在这儿确实有用，让"带经验的回答"去教"不带经验的"（把能力吸收进去，以后不给经验也会）；`Δ` 明显为负 → 说明这条经验在这儿是**误导**，反过来让"不带经验的回答"**纠正**"带经验的"（**主动压制坏经验**）。教的方式是 token 级分布对齐 + 置信度门控（只在有把握的位置教）。
  4. 同一个 `Δ` 还顺便用来**更新每条经验的效用分**（配 UCB 决定以后检不检索它）、并训练"写经验"的 reflection 模块。ALFWorld/WebShop 比 SOTA +23.5/+18.0。
- 与我们的区别：UCOB 的"带经验 vs 不带经验、比谁回报高"就是我们 §4 **影子对照的在线 RL 内生版**——它在 rollout 内、按 anchor-state 配对做；我们在**推理服务侧按 ~5% 采样**做 2×3 归因表。UCOB 的产物是**改 policy 权重 + 更新经验效用**；我们把同一个"带/不带 memory 对照"信号接到**三个出口**：失败挖掘、per-item 效用、坏 memory 退休。**它强证了我们方案的核心机制可行且高收益**，它的 `Δ` 就是我们 per-item 效用的一种无偏估计，"按相同中间状态配对"这招可直接借来**降低我们效用估计的方差**。

### 10.3 失败挖掘 / 从成败双向抽 memory —— 对应 §4

**Learning from Failure: Inference-Time Self-Improvement for Computer-Use Agents**（arXiv 2606.31270，**ECCV 2026**）
- 想解决的问题：现在造 agent 训练数据的标准做法是"agent 在有验证器的环境里跑 → **只留成功轨迹**去微调 → 丢掉所有失败"。但失败其实携带了"模型哪里弱"的宝贵信息，全扔了很浪费。
- 怎么做（**不训练**，改的是 agent 的推理时行为，见其 Fig.2）：
  1. agent 跑一批任务，收集**失败**轨迹。
  2. 用一个 LLM 当"分析员"，把 (指令、动作历史、思维链) 喂进去**诊断失败原因**，归成**四类**：定位不准（grounding）、能力缺口（该用某工具却不用）、知识缺失、无脑重复循环。
  3. 对每一类，LLM **提出一个推理时的补救办法并生成一段代码补丁**（分别对应：加视觉搜索、允许走终端执行、注入知识、加重复告警），**人工轻量核对**这段补丁后，并进 agent 的工作流，再重跑。每轮按"当前最主要的失败类型"选一个补丁，补丁跨轮累积。
- 数据 / loss：**零训练、无 loss**，纯粹是"诊断失败→打补丁→重测"的循环。OpenCUA-72B 在 OSWorld 从 42.3%→48.9%（全部补丁叠加 52.74%）。
- 与我们的区别：**直接印证我们"别丢失败轨迹"**。但它把失败变成**给 agent 的代码补丁/工具**（改 harness），而且**完全不区分失败是模型本身弱还是 memory 带偏的**；我们把失败变成 **anti-pattern memory + 训练负样本**，且**先做归因**（排除"是 memory 误导"才抽经验）。它那**四类失败诊断**可直接拿来当我们失败挖掘器的分类初值。

**ISM: Self-Improving Strategy Memory for Continual Math Reasoning**（arXiv 2606.31191，**ICML 2026 AI4Math Workshop**）—— **毕业/退休机制的现成七件套模板**
- 想解决的问题：冻结的 LLM 做连续不同领域的数学题时，学到的经验存哪、怎么不越堆越乱？纯 retrieval 会无限膨胀、纯 reflection 只存散乱文字。
- 怎么做（**不训练模型**，只维护一个外部"策略库"）：
  1. 外挂一个**紧凑的 strategy-schema 库**。每条 schema 拆成两半：**content**（策略描述/解题模板/启发式，用时注入 prompt）+ **feature hook**（结构标签 + embedding，决定"什么时候该检索到它"，且随使用自动微调）——把"这条经验讲什么"和"何时被调出来"解耦。
  2. 检索分两步：先按题型/算子过滤，再 soft 打分选最相关的。
  3. **库由七个自维护机制打理**（这是精华）：①audit 审查 ②merge 合并近重复 ③prune 删无用 ④promote/demote 升降级——管质量和体积；⑤reinforce 从**成功**里抽正向启发式 ⑥antipattern 从**失败**里记"要避开的坑"——**成败双向都学**；⑦rehabilitate 给表现差的 schema 一次翻身机会再决定删不删。**每次改库都要先过符号验证器**（数学能硬校验），防止把错误泛化写进去。
- 数据 / 标注：数据是 300 题的连续流（按域分块），"对不对"由**符号验证器自动判**（数学域独有的硬 verifier），无人工标注。backbone 是 gpt-4.1-mini、temperature=0。
- **准确率数字要拆开看（别被 0.48→0.81 唬到）**（Table 1，MATH-Hard 累计 acc）：
  - Vanilla 48.0 → RAG 57.0 / Reflexion 55.7 → **Static Schema 78.67**（bank=1，一个固定 prompt 模板 + 允许调符号工具）→ Passive 78.67 → ISM 80.67。
  - **+30 点的大头来自 Static Schema——即"好 prompt 模板 + 符号验证工具"，与 memory 无关**；真正属于"memory 自维护机制"的（ISM vs Passive）**只有 +2 点**（300 题净多对 6 道，OlympiadBench 同样 +2）。作者自承单 seed、单 stream 顺序、无逐机制消融，"gains should be interpreted as preliminary"。
  - ISM 的真卖点其实是**省存储 + 抗遗忘**（bank 只有 Passive 的 1/3~1/7、比 RAG 少 23×），不是"memory 让数学变强"。
- **⚠️ 评测可信度存疑（重要）**：全文**没有任何去污染措施**——grep 全文无 `decontamination / n-gram / 13-gram / dedup / held-out / train-test split`。而且它的"记忆积累"和"评测"用的**是同一条 300 题流、没有独立 held-out**：RAG 基线是"**把做过的每道题（含解）全存进库、按 embedding 召回最像的一道注入**"，对 MATH/OlympiadBench 这种"换数字的同型题很常见"的数据，等于系统性地**召回近似题的解 = 流内信息泄漏**，且未做任何近似题过滤。所以 RAG 的 0.57、schema 系的 0.79 都**掺了泄漏水分 + prompt/工具增益**，**这些数字不能当作"memory 在数学上有效"的证据**。
- 与我们的区别：ISM 的**七机制 + 验证门 + content/feature 解耦**几乎就是我们"毕业（promote/demote/retire）+ 效用退休 + 去重合并"的**现成设计模板**，它的"成功→正样本、失败→anti-pattern 对称双向抽取"正是我们要的——**我们借的是这套领域无关的 lifecycle 机制**。但**不采信它的数学有效性结论**：它靠数学专有的硬符号验证器、且评测有泄漏/缺干净对照；我们无此硬 verifier（只有带噪声的 rubric+hard 软 verifier）。而且它这套结论**恰好与我们自己的实验吻合**——我们 200 题数学 memory 只比 direct 多对 1~2 道，而 ISM 剥掉 prompt/工具/泄漏后 memory 机制也就 +2 点：**两边共同印证"数学这类可泛化硬技能应走线 A（训权重），memory（线 B）边际只有 1~2 个点"**。ISM 还是权重全程冻结、纯外部 memory；我们是双线，退休判据还多一条"新权重不给 memory 能不能自答"。

**M2Note: Mistake Notebook Learning**（arXiv 2607.00685）
- 想解决的问题：怎么把失败经验安全地攒成"错题本"，又不会因为写错东西把整体带崩。
- 怎么做（不训练，改外部笔记）：把失败轨迹提炼成**按主题组织的"错题本"note**，检索注入引导 agent 规避同类坑；关键工程点是 **批级后验 + 回滚**——一批笔记编辑先在同一批任务上验证，**只有整批指标真的提升了才提交，否则整批回滚**。支持同模型自我进化，也支持"一个模型的错题本给另一个模型用"（= 我们的 student/teacher）。
- 与我们的区别：它的"批级后验 + 只在变好时提交、否则回滚"与我们 D7c 校准 / harness 发布门的"批级回归检验 + 达标才发布"**几乎一样**，是 §7 稳定性的又一独立佐证；"跨模型进化"正对应我们 teacher→student 的 memory 迁移。差异仍是我们把它接进**归因 + 双线 + 效用退休**的完整闭环，而非只做注入引导。

### 10.4 归因 / 去偏 / "何时不该写 memory" —— 对应 §4 归因难题、§5 选择偏差

**GovMem: When Not to Write Memory — Governing False Promotion from Correlated Traces**（arXiv 2607.02579，**MLISE 2026**）—— **精确命中"分不清是模型还是 memory 问题 + 会污染"**
- 想解决的问题（一句话）：**"重复出现"不等于"多份独立证据"**。五个 agent 说同一句，可能是五次独立发现，也可能是**同一条过时笔记经共享上下文回声了五遍**。如果按"出现次数多就晋升"的朴素规则，长期 memory 会慢慢变成"把相关性错误固化下来的持久层"。
- 怎么做（**不训练**，是一个"该不该写这条 memory"的审计决策，四步 write-path，见其 Fig.1）：
  1. 把说法相近的观测**聚成一条候选 memory**，同时保留每条的**来源信息**（哪个 source、哪套 prompt、哪个父事件、什么环境、可信度）。
  2. **算"去掉相关性后的有效支持度"**：共享同一 prompt/工具/父事件的观测**不算独立的一票**（这是核心——把"回声"折价）。
  3. 检索**反证**、检查"这条经验声称适用的范围"合不合理。
  4. 综合上面判断，输出三选一：**promote（写）/ reject（拒）/ needs-review（送人审）**。
- 数据 / 效果：合成压力测试里把"错误晋升率"从 0.597（按出现次数）降到 **0.040**，同时召回还保 0.960，代价是 15% 送人审。**最刺眼的是真实数据结果**：133 条高影响候选经人裁后，**0/133 可以安全自动晋升，本地门控判过的 11 条全被人否掉**。
- 与我们的区别：GovMem 是个**保守的审计侧策略**，只管"写不写"、不管"写了有没有用"。我们把它当**"晋升到 A 线 / 长期 memory 之前"的闸门**，再叠上我们的 verifier 地基 + 灰度人审；它的四步（留来源→按相关性折价→找反证→三选一）可以直接做我们 promote 的前置检查。它"几乎没有能安全自动晋升的"这个结论，**强烈警示我们：宁可漏抽也别错抽，默认走 needs-review 灰度**。

**MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation**（arXiv 2606.29914）
- 想解决的问题：报告里 memory 系统"比 RAG 强"的结论，常常混进了 LLM/embedding/检索管线本身的变化——到底是**memory 架构真强**，还是只是**换了个更好的 embedding**？
- 怎么做（**一篇测量方法论的论文，不提新架构**）：在 LongMemEval-S（500 题、每人 50+ 会话、三个模型族）上，**一次只变一个变量、其余全冻**，隔离四个隐藏混淆：检索质量、embedding 选择、模型长上下文行为、写路径成本。
- 关键实测（这些数字很能说明问题）：① **agent 自产 memory 只有 42%，反而不如朴素 retrieval**；② 只换 embedding 不动别的，Mem0 就从"比 RAG 基线 +11pp"翻转成"−1.2pp"——**结论被一个变量掀翻**；③ Mem0 只在窄题型上占优，但**写路径成本可占 agent 总执行时间 80%+**。建议：固定 embedding、按模型族分层、把 write 成本当一等公民报告。
- 与我们的区别：**直接给我们 §5 选择偏差 + §4"必须做对照"背书**——它证明了"不做受控对照，memory 的效用数字根本不可信，甚至自产 memory 是负收益"。这正是我们坚持"per-item 效用必须用 **带/不带 memory 反事实对照**去偏、且先把 verifier 地基坐实再上效用分"的理由；它的"固定 embedding + 报告 write 成本"应直接写进我们的效用评估协议。

**Stealthy Memory Injection in Persistent Personal Agents**（arXiv 2607.05189）
- 想解决的问题 / 做法：展示在持久化个人 agent 里，**坏的或恶意的 memory 能被悄悄写入并长期潜伏**——用户看不见、还跨会话反复生效造成危害。
- 与我们的区别：佐证我们需要 `memory_harmful` **自动退休闸门** + 用户级 memory 的隐私/时效治理（§9 待定项）——尤其我们有用户级 scope，更要防"一条坏偏好被写进去后反复检索注入"。

### 10.5 memory 命中的"选择性使用" + memory-on/off 对照 —— 对应 §4 影子对照、§9 路由方向

**ATMem + STR-GRPO: What Memory Do GUI Agents Really Need?**（arXiv 2606.31612）—— **STR-GRPO 就是我们"影子对照"的 RL 化，几乎逐条对应**
- 想解决的问题：GUI agent 做长任务时，光"把过去看到的存下来"不够——它还得知道"这条信息现在**该不该用、用了到底有没有帮助**"。
- 怎么做（两部分，见其 Fig.2；这是**真训练**：先 SFT 再 RL）：
  1. **ATMem（把 memory 从被动存储变成主动的"执行状态"）**：memory 不是流水账，而是一张**结构化的任务进度表**——记着"整体进度 + 约束""每个待办项的内容 + 它的状态（待办 / 已完成 / 跳过）"，由 agent 边做边更新。先用"只保留通过验证器的成功轨迹"造 SFT 数据（120 模板 → 1.1K 实例 → 21,713 步级样本），教会模型**会建、会更新、会引用**这张进度表。
  2. **STR-GRPO（用对照实验学"何时该用 memory"）**：对同一道题采一批 rollout，**刻意一半开 memory、一半关 memory**（只切 memory 这一个开关，其余历史都保留）。奖励 = 最终有没有做成（验证器给 0/1）**减去 memory 的使用成本**（用了 memory 却多走步、没帮上忙，就扣分）。因为同一题的开/关两组**共用同一个打分基准**，所以"开 memory 比关 memory 好多少"就**直接量化成了这条 memory 通道的净贡献**，模型据此学会"该用才用"。
- 与我们的区别：它"同题下开 memory vs 关 memory、比谁做得好"**就是我们 §4 影子对照 2×2 表的 RL 内生版**，它的"memory 使用成本"惩罚正好回答我们 §9 待定的"何时该用 memory、何时 memory 反而是负担"。差异：它在 **GUI 域、RL rollout 内**做、产物是改 policy；我们在**推理服务侧按采样**做、产物接归因+效用+退休三出口，且走双线蒸馏而非纯 RL。**它"开/关对照算净贡献"的思路可直接当我们 per-item 效用的估计量。**

**WorldEvolver: Self-Evolving World Models for LLM Agent Planning**（arXiv 2606.30639）
- 想解决的问题：给 agent 装一个"世界模型"（行动前先预测后果）能帮规划，但**预测不准时反而会把 agent 带偏**。而每次都靠梯度更新去修这个世界模型，在线部署下太贵、还会灾难性遗忘。
- 怎么做（**关键：agent 和世界模型的参数全程冻结，只改外部 memory**）：三个模块——① **情节记忆**：把真实发生过的"状态→动作→结果"存下来，检索出来做"检索式模拟"；② **语义记忆**：把"预测的结果 vs 真实观测"对不上的地方，提炼成可复用的启发式规则；③ **选择性前瞻**：预测在喂给 agent 之前，**先把低置信度的预测过滤掉**，只把有把握的预测注入。
- 与我们的区别："**没把握就别注入**"正是我们 §5 不确定性驱动 + 主线置信度门控的同款直觉（我们把它用在"低效用/高风险 memory 不注入"上）；"从预测-真实的失配里提炼规则"对应我们从失败挖 anti-pattern。差异：它**冻结参数、只改 memory**；我们要把提炼出的规则进一步走 B→A 蒸馏进权重。

### 10.6 记忆巩固 / 晋升与身份稳定 —— 对应 §3 毕业机制、§7 稳定性

**Episodic-to-Semantic Consolidation Without Identity Drift**（arXiv 2607.01988）
- 想解决的问题（一个偏"合规/审计"的场景）：受监管的长期部署 agent（医院、工厂机器人）有一个**加密认证的身份**（对一份 manifest 做哈希）。传统"巩固知识"的做法（微调 / 改 prompt / 蒸馏 / 追加反思）都会改动定义身份的那份东西，于是**每学一点新知识就等于换了个 agent、要重新认证**。矛盾：既要越用越聪明，又要身份字节级不变。
- 怎么做（**v1 完全不用 LLM、是确定性统计规则**）：把"巩固"定义成一个**确定性函数** `f: 情节日志 → 语义层`。情节日志是只追加的原始事件记录；`f` 就是**按 (技能+对象+场景) 分组、数成败、算成功率**，输出一行带 **置信度 + 观测数 + 溯源指针** 的语义事实（例："对玻璃杯、这个环境，建议抓取力 25N，置信 0.83，基于 15 次观测"）。关键设计：**身份哈希在构造上就不读这个语义层** → 无论巩固多少次，身份字节不变；planner 只能**只读**查询语义层、不能改。（用 LLM 的 v2 被明确列为 future work，因为引入不确定性会破坏可审计性。）
- 数据 / loss：**无训练、无 loss**，纯确定性聚合。合成 benchmark（1000 决策）上，对比一个校准过的 Bayesian 基线，planner 的无效尝试降 79.82%，同时身份哈希全程字节相等。
- 与我们的区别：`f: 情节→语义` 正对应我们 **B（快 memory，情节性）→ A（长期语义）** 的毕业方向，"每条带置信度+溯源、可审计"对应我们的血缘记录；"学新知识不改身份"提示我们**毕业/退休不该破坏模型稳定的基础能力/人格**（呼应 §7）。差异：它只在 memory 层做确定性聚合、**完全不碰权重**、且是单向（情节→语义）；我们的 B→A 是**真的权重蒸馏**，还有 A→B 反向退休。

**SEA: Self-Evolving Agents with Anytime-Valid Certificates**（arXiv 2607.00871）—— **几乎就是我们 §7"把进化关进可回滚笼子"的统计化理论**
- 想解决的问题：自进化 agent 有个根本麻烦——它用来学习的数据、评判自己的评估器、用的组件，**全是被它自己更新的策略生产出来的**（自己考自己、自己出题自己判）。这种"闭环自产"下，经典学习理论的保证（收敛、不遗忘、安全改进）**全部失效**。
- 怎么做（四层架构 + 两条铁律，见其 Fig.1；注意它**主要不做权重微调**）：
  - **四层**：L0 = 冻结的底座模型；L1 = 一个很小的 **steering adapter**（在线只调"选哪条指令"的概率分布，不动权重）；L2 = **带版本的 harness**（prompt/工具/预算/技能库，可改可扩，每次改都记一个新版本）；L3 = 在旁边的调度器（不在推理主路径上）。
  - **铁律一：每次自我修改都要过一道"随时有效"的统计门**。因为 agent 每轮都在偷看自己的成绩，普通固定样本量的显著性检验会失效；它改用一种**允许你随时停下来看、结论都成立**的统计量（e-value），对一个固定的"错误预算"发一张**可审计的证书**（通过/暂缓/拒绝/无解），全部记进一个账本。快环每轮调 L1、慢环每隔 K 轮改 L2。
  - **铁律二：门只能在"底座本来就能做出的行为"里挑**（不能凭空造新能力）。所以另配五个"验证器在环"的引擎（best-of-N、微步搜索、自写复现测试、搜索层控制、自修复）来**产生候选行为 + 提供密集的、不靠人打分的信号**。SWE-bench Verified 上 +4/+5。
- 与我们的区别：SEA 用**统计证书 + 错误预算**保证每次进化不把系统带崩；我们用**工程化的发布分层（static→shadow→canary→live + 可回滚）**做同一件事——两者可互补（我们的 canary 达标判定可以升级成它这种"随时有效的统计门"）。它的"L0 冻结 / L1 只 steer / L2 可改 harness"分层，给我们"哪些能在线漂移、哪些必须冻结"划了清晰边界；"门只能挑已有行为"正是我们要把 harness 关进笼子的理由。差异：SEA **基本不微调权重**（只 steer L1）；我们主线恰恰要蒸馏权重，所以更需要它这套门控来兜底。

### 10.7 rubric 作为 reward / 可靠性 —— 对应 §2 rubric 靶、§5 verifier 地基

**RuVerBench: Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?**（arXiv 2606.29920）
- 想解决的问题：现在大家用 LLM 当"裁判"、按 rubric（评分条目）给 agent 打分，但**这个裁判本身靠不靠谱**没人系统测过——尤其 agent 输出又长又复杂（深研报告几千 token、编码轨迹几万 token）时。
- 怎么做（**一个 benchmark，不训练**）：构造 2458 条样本（深研 1615 + 编码 843），每条 = (一段 agent 输出, 一条 rubric, 人工标的"满足没满足")。人标经**双人独立标注 + 裁决**，两组一致率 90.4%、κ=0.808（很高）。然后拿各种前沿模型当裁判去判，测它们和人标的吻合度，并测"多判几次投票""一次判多条"这些策略。
- 结论：**即便最强模型判 rubric 也有明显噪声**；**编码类、尤其涉及 tool-use 的 rubric 判得最差**；多数投票有效但**收益递减**；一次判多条省钱但掉准确率。
- 与我们的区别：直接支撑我们的**执行顺序**——"**先坐实 verifier 打分可靠，再拿它去算 memory 效用分**"，否则效用分建在噪声地基上（和 MemDelta 的警告叠加）。它"编码/tool-use 类判得最差"正戳中我们 agentic 场景，它的投票收益递减曲线能帮我们定"什么时候值得多投几票"。

**MRRG: Many Voices, One Reward — Multi-Role Rubric Generation**（arXiv 2607.01830）
- 想解决的问题：现在"自动生成 rubric"多是**一个通用评估器一口气列所有标准**，容易漏维度（论文叫"维度盲点"），进而导致判分看领域、还能被"只优化被覆盖的标准"刷分（rubric hacking）。
- 怎么做（**训练无关、无需参考答案**）：让同一个 LLM **轮流扮演多个角色**（用户、领域专家、教育者、AI 研究员、语言学家…），每个角色从自己视角产一批**原子、可验证**的 rubric 条目，再**汇总去重**成一个可审计的打分器。这个打分器既能做偏好判定，也能**直接当 GRPO 类 RLVR 的 reward**。
- 效果：RewardBench-2 / JudgeBench / PPE 上比单角色基线 +3.1~16.4pp；用作 RLVR reward 时 +1.7 / +3.4。
- 与我们的区别：可直接增强我们 `RubricVerifier` 的 **rubric 生成质量**——把当前"单 prompt 生成 rubric"升级成"**多角色生成再合并**"，且它天然兼容我们主线 verifier（同一套 verifier 既供 memory 效用、也供训练奖励）。

### 10.8 memory 系统工程形态（存储/检索）—— 对应 §1 选型

**MOSS: Auditable Agentic Memory**（arXiv 2607.04391）
- 想解决的问题：主流 RAG 用向量相似度检索，**不透明、难审计、有理论上限**——在长期/个人/受监管场景尤其致命。
- 怎么做：由 **agent 分析查询意图 → 参数化一条结构化检索 → 在关系库上用 SQL 确定性地取数**，**检索环里没有 LLM**（一旦查询定好，执行完全可复现）；词表从语料自动归纳、不外加本体；每一步从建索引到出答案全部可审计。已**真实生产部署约一年**（约 4400 万 token 语料、每天当主力工作记忆用）。
- 与我们的区别：**印证我们"检索用结构化 KV + 可审计、LLM 只在写/抽 memory 时参与"** 的选型——检索热路径不放 LLM，省成本、可复现、可审计。

**Mandol: Agglomerative Agent Memory**（arXiv 2606.29778）
- 想解决的问题：现有系统把向量库、图库拆成好几套，**跨库 I/O 慢、信息碎片化**，RAG 式检索又容易招噪声、漏关联、控不住 token 预算。
- 怎么做：用一套 **SemanticMap + SemanticGraph 的内存数据结构，原生融合 KV / 向量 / 图**（不是拼三套系统），提供统一的混合检索算子；检索走"查询自适应路由 → 去噪/消解冲突 → 按 token 预算生成上下文"，**全程不调 LLM**。LoCoMo 92.21% / LongMemEval 88.40%（均最优），10 QPS 下检索延迟比最快基线还低 **5.4×**。
- 与我们的区别：直接支持我们"**混合索引（向量 + 结构化 KV）+ 应用层检索、检索不放 LLM**"的方向，且证明融合式索引比拼装更快更准——是我们 §1 存储层的可参考实现形态。

**HyphaeDB**（arXiv 2606.28781）
- 怎么做：把 HNSW 近邻图当作**多 agent 之间的知识传播网**（gossip 扩散 + 能量衰减 + 自发共识，让高价值知识自然传开、陈旧的自然淡出），并给了 **pgvector 参考实现**。
- 与我们的区别：现在用不上（我们先做单机混合索引），但将来若**全局向量层要升级成多副本/多 agent 协同**，它的"能量衰减 = 自动淘汰陈旧 memory"和我们退休机制思路一致，可作远期参考。

**其它形态（提示 memory 不止"注入文本"一条路）**
- **PLACEMEM**（2607.04089）：按算力预算调度 memory 平面。
- **Neural Procedural Memory**（2606.29824）：用**隐式 activation steering**（直接改模型激活，而不是往 prompt 里拼文本）来承载程序性 memory。
- **Analytic Concept-Centric Memory**（2606.29774）：以概念为中心组织 memory。
- 与我们的区别：后两者提示"注入 memory"未必只有"prepend 文本"这一种——**改激活（activation steering）是一条可选的补充通道**（和 SEA 的 L1 steering adapter 呼应）。我们当前走文本注入，把它列为远期 B 线的可选实现。

### 10.9 对比表：本设计 vs 代表性工作

| 维度 | 本设计（twinkle） | 最接近的工作 | 我们的差异 |
|---|---|---|---|
| 双线（memory + 权重） | context 注入 + `llm_backup` 蒸馏，且有 B↔A 毕业梯度 | DuoMem：CD(teacher memory 检索 prepend)+LoRA(成功轨迹) | DuoMem 两轴**离线一次性、只用成功轨迹**；我们在线闭环 + 毕业 + 失败挖掘 |
| 抽取器自进化 | meta harness（bandit）+ 效用回流 + 抽取器可蒸馏 | MetaSkill(两时间尺度五 agent)/ SelfMem / COMFYCLAW | 它们在 prompt/skill 层递归、权重冻结；我们慢环落到**权重蒸馏** + 分层可回滚笼子 |
| 失败挖掘 | 只挖“能力性失败”，token soup 直接丢 | Learning-from-Failure(四类诊断→patch)/ ISM(七机制)/ M2Note | 它们不区分模型/memory；我们前置**归因**排除 memory 误导后才抽 anti-pattern |
| 归因（模型 vs memory） | 注入现场留痕 + 5% 影子对照 2×2 表 | UCOB(CBSD:anchor-state ΔG)/ ATMem(STR-GRPO 干预 advantage) | 同为 memory-on/off 对照；它们在 RL rollout 内改 policy，我们在服务侧接失败挖掘+效用+退休三出口 |
| 效用分 | per-item 反事实 Δverifier，便宜信号 + 稀疏锚校准 | MemDelta（对照诊断证据）/ ATMem(memory-cost reward) | MemDelta 只做评估诊断；我们把去偏后的效用直接驱动晋升/退休 |
| 何时不写 memory | 门槛（traj_score/safety）+ harmful 退休 | GovMem(provenance→依赖去相关→反证→三路决策) | GovMem 是保守诊断策略、只判写不写；我们与 verifier 地基 + 灰度 + 效用结合 |
| 稳定性/进化治理 | static/shadow/canary/live + 离线可回滚发布 | SEA(四层 + e-value certificate 门)/ M2Note(批级 rollback) | 我们用工程 mode 分层，SEA 用统计 certificate；canary 判定可升级成 anytime-valid gate |
| verifier 地基 | 主线 hard+rubric 融合，先坐实再上效用 | RuVerBench(可靠性有噪声)/ MRRG(多角色 rubric) | 我们直接复用主线 verifier，不为 memory 另造评审；rubric 生成可升级为多角色 |

### 10.10 我们仍然新颖的地方（综合判断）
单点都有平行工作，但**没有一篇把下面这套完整闭环合在一起**：
1. **同一批生产轨迹**同时喂“慢权重蒸馏”和“快 memory”，且两者之间有**显式毕业梯度（B→A 晋升 / A→B 退休）**——退休用“新权重能否自答”自动判定（DuoMem 无毕业；Consolidation 无 A→B 反向）。
2. **失败挖掘前置因果归因**：先用注入留痕 + 影子对照区分“模型能力 / memory 缺失 / memory 误导”，**只在 model_capability 上抽 anti-pattern**，把 GovMem/MemDelta 警示的污染从源头挡掉（Learning-from-Failure 类不做 memory 归因）。
3. **抽取器本身被“经下游效用验证过的 memory”反向蒸馏**，效用分是 per-item 反事实 Δverifier、且**与主线训练奖励同源**（SelfMem/MetaSkill 优化的是策略/prompt，不回流蒸馏抽取器权重）。
4. **进化被工程化为可回滚的发布流程**（static→shadow→canary→live），而非在线自由漂移——把 SEA 的“门控只能 select 已有行为”落成部署 mode。

**一句话定位**：DuoMem 证明了双线值得做，UCOB/ATMem 证明了 memory-on/off 对照能归因，GovMem/MemDelta 证明了不归因会污染/被混淆，SEA 证明了进化要门控——**本设计是把这些已被各自验证的结论，收进一个共用 verifier/`llm_backup`/D7c 的单一自进化蒸馏闭环。**

---

## 11. skill/rubric → 参数化（LoRA）：双 LoRA 方案的文献支撑

> 背景：讨论中提出过一个具体的 line A 实现形态 —— **① 把高分轨迹的 system skills 蒸成一个"技能 LoRA"，推理时先用它产出/注入 skill；② 把 rubric 评价能力蒸成另一个"评价 LoRA"，推理中每隔 N 个 token 用它检验，当过程奖励/memory 提示。** 这一节把 2026 上半年直接对应这三个子命题（skill→LoRA、rubric→LoRA、多 LoRA 推理时切换）的工作按 §10 的详尽风格补全，并标注每篇对方案的取舍启示。**结论先行：三个子命题都各有直接平行工作，MetaClaw 几乎是整套方案 + 本双线设计的镜像；但"评价要不要绕一圈做成 LoRA judge"和"过程监督要不要走 LoRA"这两处，有明确的反方证据，需先决策。**

### 11.1 skill → LoRA（对应子命题 ①：技能 LoRA）

这一类的**共同范式**高度一致：**离线用完整 skill 文本合成"技能引导"的示范 → 训一个 skill 专属 LoRA → 在线丢掉 skill 文本、动态挂 LoRA 激活行为**。动机都是：skill 文本每步注入 context 太贵、且长上下文里关键指令定位不到 / 遵守不了（小模型尤甚，ICL 一贯不如微调、且模型越小差距越大）。

**Skill-to-LoRA (S2L)**（arXiv 2606.16769）—— **最贴近方案 ① 的朴素做法（一 skill 一 LoRA，不用 hypernetwork）**
- 想解决的问题：agent skill 现在以 `SKILL.md`（人读的流程文档：workflow / 工具 / 资源 / 领域约定）分发，可读可复用，但**同一套可复用流程要在每步 runtime context 里反复注入**，费 token 又稀释注意力。
- 怎么做（**离线合成 + 在线换挂**，behavior-centric）：不压缩文档本身，而是**建模"skill 文本诱导的行为改变"**。**离线**：把完整 `SKILL.md` 喂进去，让模型合成一批"skill 引导下的示范轨迹"（demonstrations），用这些示范 SFT 出一个**该 skill 专属的 LoRA**；**在线**：完全省略 `SKILL.md` 全文，按当前需要动态加载对应 LoRA 来"激活"这个技能行为。
- 数据 / loss：标准 SFT（在合成的 skill-guided 示范上做 teacher-forcing 训 LoRA），无 RL。Qwen3.6-27B、SWE-Skills-Bench 21 个 skill 子集：比 no-skill +2.9 pp、比 Full-Text +5.2 pp，每步 token 比 Full-Text prompting −6.6%；18/21 个 skill 追平或超过 Full-Text、15/21 超过 no-skill。**关键对照实验：Wrong-LoRA（挂错 skill 的 LoRA）和 Shared-LoRA（所有 skill 共用一个 LoRA）都掉点** → 收益依赖 **skill-专属对齐**，不是"随便训个 LoRA 就行"。
- 与我们的区别：这就是方案 ① 最省事的落法（不引入 hypernetwork，一 skill 一 LoRA、离线 SFT）。**它的对照实验直接给我们敲定了两条工程铁律**：(a) 技能 LoRA 必须按 skill 对齐、**不能把多 skill 或 query 糊进一个 LoRA**（呼应 §0 分工判据："query 是易变实体，训进权重会过拟合"）；(b) 挂错 LoRA 反而有害 → 上线要有"挂哪个 skill LoRA"的可靠路由（正好复用我们 meta harness / intent 标签）。差异：S2L 是**离线一次性**、skill 集固定；我们要它在自进化闭环里**持续产出新技能 LoRA**，且用主线 `traj_score` 当"哪些高分轨迹够格蒸成 skill LoRA"的门槛。

**LatentSkill**（arXiv 2606.06087）—— **用 hypernetwork 把文本 skill 一次前向转成 LoRA**
- 想解决的问题：同 S2L（per-step 注入 skill 费 context、且 skill 明文暴露），但更进一步想要"**不为每个 skill 单独训 LoRA**"。
- 怎么做：训一个**预训练 hypernetwork**，输入文本 skill、输出即插即用的 LoRA adapter（把 skill 知识存进**权重空间**而非 context 空间）。保留了 LoRA 的模块化：可加载、可缩放（用 LoRA scaling 系数精确调强弱）、可组合（对齐时能在**参数空间做算术**叠加多个 skill）。
- 数据 / loss：hypernetwork 预训练。ALFWorld seen/unseen +21.4 / +13.4 分、prefill token −64.1%；Search-QA EM +3.0、skill-token 开销 −72.2%。分析发现生成的 skill LoRA 形成**结构化语义几何**。
- 与我们的区别：如果我们不想"一个 skill 训一个 LoRA"（S2L 的痛点是 skill 一多 LoRA 就爆炸），LatentSkill 的 hypernetwork 是升级路径 —— **一次前向即出新 skill 的 LoRA、零梯度更新、零 skill 专属数据采集**。但它更复杂、要预训练 hypernetwork，属于方案 ① 的"进阶版"，建议 S2L 跑通、验证 skill LoRA 确有增益后再考虑。它的"参数空间算术组合"对我们"把多条相关碎片 memory 合并升华"（§2 consolidation / B→A 晋升）是权重侧的对应工具。

**ParametricSkills**（arXiv 2606.30015）—— **hypernetwork 同时参数化"skill 内容 + 利用方法"，含自进化/持续学习**
- 想解决的问题：同上两条，外加一个更深的观察 —— **文本空间演化 skill（EvoSkill/SkillOpt 等改写 SKILL.md）和模型本身的学习是解耦的**，模型能力没被优化。
- 怎么做（三阶段，hypernetwork 驱动）：(1) 建 **45.8k skill 库**（网爬 + 从真实 agent 轨迹总结，覆盖 13 领域），用 OpenCode 沙箱围绕这些 skill 合成单/多轮"skill 利用轨迹"；(2) **skill-重建预训练**：三个自监督目标让 hypernetwork 学会把 skill 编码成 LoRA —— **完整重建**（据全文生成能重建全文的 adapter）、**前缀补全**（只给前缀、要补出全文，学 skill 结构）、**段级 cloze 补全**（挖掉一个功能段、据前后文补，学"触发条件/执行步骤/失败处理"如何组织与互相支撑 → 强组合泛化 + 支持局部编辑）；(3) 在 skill-利用轨迹上**多轮 SFT** hypernetwork。
- 数据 / loss：自监督重建 + 多轮 SFT（loss 都 backprop 到 hypernetwork）。6 个 SWE 子任务比 ICL +6.44 分（DeepSeek-V4-Flash 判）、BERTScore +1.17、F1 +5.53%；**注意 text-to-LoRA 基线 SHINE 反而打不过 in-context skill** → hypernetwork 训不好会退化。持续学习：把多条轨迹的经验**不断 merge 成一个全局 parametric skill**。
- 与我们的区别：它把 §2「抽取器越来越专业」和 §3「B→A 晋升」在**权重侧**给出了一个具体形态 —— "文本演化 skill = 直接改进模型"。三个自监督目标（尤其**段级 cloze**）可直接借来当我们"技能 LoRA 抽取器"的预训练任务。但它同样是**离线训 hypernetwork**、且 SHINE 反例提醒**参数化 skill 不保证优于文本注入**，必须带对照验证（呼应 §4 影子对照）。

### 11.2 rubric / 评价 → 参数化（对应子命题 ②：评价 LoRA）—— 两条岔路，务必先选

**支线 A：把 rubric 打分能力做成一个轻量 LoRA judge（= 方案 ② 的原意）**

**Plug-and-Play LLM Judges**（arXiv 2506.05748）—— **"rubric + 小 LoRA = 顶级裁判"的最强直接背书**
- 想解决的问题：RLHF 的奖励模型训练是成本瓶颈（动辄几十亿参数 + 离线偏好微调阶段）。
- 怎么做：**冻结的 instruction-tuned 7B + 一行 JSON rubric + rank-16 LoRA（只动 0.8% 参数）**，就当完整奖励模型用。消融：6 条 in-context 示范贡献了大部分零样本→少样本增益（+2pp），**LoRA 补上剩余差距**（尤其 safety / 对抗性 Chat-Hard 段）。
- 数据 / loss：小 LoRA 微调 + prompt 工程。RewardBench 96.2%，超过 27B~70B 专用奖励网络；配它当在线 PPO 的 reward，7B actor 在 GSM-8K 拿 92% EM、超过 70B DPO 基线；LoRA judge 的解释与人类相似度 ≈9/10（零样本裁判仅 ≈5/10）。
- 与我们的区别：**这是方案 ②「rubric→评价 LoRA」最硬的可行性背书** —— 极小 LoRA + 一条 rubric 就能把通用模型变成高质量、可解释、可调的裁判。直接支持我们把 `RubricVerifier`（现在调 dashscope teacher）蒸成本地评价 LoRA：既复用 `score_lora_path` 现成入口，又能治昨天"每段调远程 LLM 太慢"的痛（本地 LoRA 打分快几个数量级）。

**支线 B：跳过 judge，把 rubric 直接蒸进 policy 的 token 级信号（更省，可能是更优解）**

> ⚠️ 这两篇机制外壳都是"rubric-conditioned 的自己当 teacher、逐 token 蒸给 unconditioned 的自己"，但**要解决的痛点、对标的对手、卖点完全不同**，别当成一篇：**RCSD 的对手是 RL 的标量奖励**（卖点＝把标量 reward 升级成过程级 token 信用分配）；**RGSD 的对手是 rubric 训练里的那个 LLM verifier**（卖点＝把 verifier 从训练回路里彻底删掉）。下面各自只讲其独有点。

**Rubric-Conditioned Self-Distillation / RCSD**（arXiv 2606.19327）—— **卖点：用 rubric 替代 RL 的"标量奖励"，做过程级信用分配**
- 想解决的问题：针对的是**蒸馏与 RLVR 两种 post-training 各自的病**。蒸馏靠 CoT 标注（贵、可能有噪/不全/半错，**哪怕最终答案对，坏 rationale 也会干扰学习**）；RLVR 则把评价**压成一个标量 reward**，看不出"该改推理的哪一步"。它要的是一个**比标量更细的过程监督信号**。
- 怎么做（**独有点＝两阶段 pipeline + 显式过程级信用分配**）：核心是"**不把单一参考 rationale 当唯一监督靶**"，而让 teacher 看 criterion 级 rubric、在 student 自采样轨迹上给 token 级指导。落地成**两阶段**：**阶段① 先训一个"生成 task-specific rubric"的模块**（给任务先产出该任务的评分条目），**阶段② 再用这些 rubric 训"rubric 引导的 reasoner"**。rubric 说明"强回答该满足什么" → 转成**过程级信用分配**，这是它明确对标 GRPO 的地方。
- 数据 / loss：on-policy 自蒸馏（token 级）。科学推理套件上**比 GRPO +1.0、比 OPSD +0.9**（对手是 RL/在线自蒸馏方法，不是 verifier）。
- 与我们的区别：它证明 **rubric 能当"过程级 reward"直接进 policy 训练**，比标量 GRPO 奖励细。对我们的启示落在**训练信号形态**上：如果 line A 想要比 `traj_score` 标量更细的过程监督，RCSD 的"rubric→token 级信用分配"是替代 GRPO 标量奖励的路子；它的**阶段① rubric 生成器**正对应我们 `RubricVerifier` 的 stage-1（可复用）。

**Rubric-Guided Self-Distillation / RGSD**（arXiv 2606.12507）—— **卖点：verifier-free，把 LLM judge 从训练回路里删掉**
- 想解决的问题：针对的是**现有 rubric 训练法都要挂一个 LLM verifier 给每条 rollout 打分**这件事本身 —— 带来三个后果：训练期 verifier 调用**开销大**、优化被**特定 verifier 的偏差**污染、且 rubric 反馈被 verifier 压成**稀疏的轨迹末端信号**（只有一个 end-of-trajectory 分）。它要的是**根本不调 verifier**。
- 怎么做（**独有点＝极简、零 verifier、单 rollout**）：直接拿 rubric-conditioned base policy 当 teacher、unconditioned 当 student 逐 token 蒸 —— 关键在于它把这套做到了**训练回路里完全没有 LLM judge**、且**每 prompt 只需一条 on-policy rollout**（不用像 GRPO 那样一题多采样再让 judge 排序）。
- 数据 / loss：**零 verifier 调用 + 单 rollout/prompt**。Qwen-2.5(3B/7B)、Qwen3-Thinking(4B/8B) 医学/科学域：rubric 满足度**与 judge-based GRPO 相当**（对手是"带 verifier 的 rubric 训练"）。独有消融：**raw rubric 比"自生成参考回答"是更强的 teacher 富化信号**；但**更强的 GRPO judge 在某些设置能反超 RGSD** → 它诚实地把自己定位为"**当 verifier 成本/可靠性是瓶颈时**"的互补方案，而非全面更优。
- 与我们的区别：**这篇最该在决策前读透**。它直击我们现状——verifier 又贵又不稳（RuVerBench 警示 + dashscope 每段调用慢）时，**把 rubric 直接蒸进 policy 比"训一个评价 LoRA 再在线打分"更省更稳**。据此，方案 ② 有两条路：**A 训评价 LoRA judge（Plug-and-Play 背书，产物是可复用的独立评价分）** vs **B 走 RGSD/RCSD 把 rubric 直接蒸进主 policy（不产独立分、只喂 student）**。选 A 还是 B，取决于我们是否真需要一个"能被 memory 效用 / 在线 PRM 复用的独立评价分"——若需要就 A，若只为提升 student 就 B。

### 11.3 多 LoRA 推理时切换 / 评价即一个 LoRA（对应子命题 ③：每 N token 用评价 LoRA 检验）

**VideoMind — Chain-of-LoRA**（arXiv 2503.13444，**ICLR 2026**）—— **方案 ③"主生成 LoRA + 评价 LoRA 交替"的现成机制原型**
- 想解决的问题：视频时序 grounding 推理要多种能力（定位、验证、回答），但为每种能力各开一个完整模型太重。
- 怎么做（两个创新）：(1) **角色化 agent 工作流** —— planner 协调、grounder 时序定位、**verifier 评估候选**、answerer 回答；(2) **Chain-of-LoRA**：一个统一 base model + **多个 LoRA adapter**，推理时**无缝切换角色**（用哪个角色就挂哪个 LoRA），在"每角色一个完整模型"和"纯 prompt 切角色"之间取平衡。
- 数据 / loss：各角色 LoRA 分别训。15 个 benchmark（Grounded VideoQA / 时序 grounding / 通用 VideoQA）验证有效，且利于 test-time scaling / 长视频。
- 与我们的区别：**它的 verifier 就是链条里的一个 LoRA 角色，与主生成角色在同一 base 上按需热切** —— 这正是方案 ③ 想要的"生成 LoRA / 评价 LoRA 交替"的成熟原型，**强烈建议精读**它怎么做角色切换调度。⚠️ 但注意：Chain-of-LoRA 是**在"角色回合"边界切**（planner→grounder→verifier→answerer 各跑一段），**不是"每 N 个 token 打断主流插评价"**；后者需要在 decode 中途暂停、跑旁路评估、再续，当前我们 vLLM sampler 的 `sample`/`sample_stream` 没有这种中途插入钩子（一次前向也只能挂一个 LoRA），要自己写**交错解码调度器**——这是方案 ③ 真正的工程量所在。

**MetaClaw: Just Talk**（arXiv 2603.17187，**有 code**）—— **几乎是整套方案 + 本双线设计的生产级镜像**
- 想解决的问题：部署的 agent 是**静态**的，跟不上用户需求漂移；在 OpenClaw（20+ 渠道、杂负载）上，现有法要么只存原始轨迹不蒸馏、要么静态 skill 库、要么retrain 要停机。
- 怎么做（**双互补机制**）：(1) **skill 驱动的快适应** —— LLM evolver 分析**失败轨迹**合成新 skill，**零停机立刻生效**（= 我们的 line B / 快记忆）；(2) **机会式策略优化** —— **云端 LoRA 微调 + RL-PRM（带过程奖励模型的 RL）** 做梯度更新（= 我们的 line A / 慢权重），由 **OMLS 调度器**在**用户空闲窗口**（监控系统空闲 + 日历）触发。两机制**互相喂**：更好的策略产更好轨迹给 skill 合成，更丰富的 skill 给策略优化更高质数据。用**版本机制**分离 support / query 数据**防污染**。proxy 架构、无需本地 GPU。
- 数据 / loss：SFT/RL-PRM（LoRA）+ 无训练的 skill 合成。skill 快适应相对 +32%；全流程把 Kimi-K2.5 从 21.4%→40.6%、综合鲁棒性 +18.3%。
- 与我们的区别：**这是与本设计 §0 双线 + §3 毕业 + §6 harness 最像的一篇，且是 OpenClaw 生产场景**。相同点几乎逐条对上：skill 快线 + LoRA/PRM 慢线、失败轨迹驱动 skill、空闲窗口触发训练、版本防污染。差异（也是我们的增量）：MetaClaw 的两机制是**并列互喂**，**没有显式的 B→A 晋升 / A→B 退休毕业梯度**，也**不做"失败是模型问题还是 memory 误导"的因果归因**（§4）；我们多了归因前置、毕业出口、和 static→shadow→canary→live 的可回滚门控。**它是本方案最好的对标基线与工程参考（有 code），建议直接研读其 OMLS 调度 + RL-PRM 实现。**

### 11.4 ⚠️ 反方证据：过程监督 / TTT 未必要走 LoRA

**SCATR: Simple Calibrated Test-Time Ranking**（arXiv 2604.16535）
- 想解决的问题：Best-of-N 的效果全看打分函数；学出来的 PRM 强但**训练/推理都贵**，而基于 token logprob 的轻量置信度启发式又**明显偏弱**。
- 怎么做：从**小校准集**学一个**轻量 scorer**，用的是 **base 模型的隐藏表示**（不是训 LoRA、不生成）。
- 数据 / loss：轻量回归头。编码/数学基准上比置信度基线 +最高 9%；**相对在同样校准数据上做 LoRA 微调，用少 8000× 的可训练参数达到相当精度**，训练/推理延迟分别快 150×/1000×；和强 PRM 相当、部分设置数学 +7.8%/代码 +4.2% 且推理快 1000×。
- 与我们的区别：**直接质疑方案 ③"评价/PRM 一定要做成 LoRA"** —— 一个读 base 隐藏层的轻量头，可能比评价 LoRA 更省几个数量级且更快。若方案 ② 的评价只用于"打个过程分排序/门控"（而非要它生成文字理由），**优先考虑 SCATR 式轻量头，而不是 LoRA**。

**Surprisal-Guided Selection**（arXiv 2602.07670）
- 想解决的问题：可验证、密集奖励任务（如 GPU kernel 优化，有确定性 evaluator）下，test-time 到底该"梯度自适应"还是"搜索"？
- 怎么做 / 结论：KernelBench + GPT-OSS-120B(LoRA)：**Best-of-N 搜索（K=64 达 90% 成功）远胜 TTT 梯度自适应（最好 30.6%）**；TTT 会**过度锐化**、把多样性塌成平庸解，"等效 K < 1"（还不如单样本）。零成本妙招：**选 surprisal 最高（最不自信）的正确样本**比选最自信的 +30%。
- 与我们的区别：提醒我们——**有确定性 verifier 时，算力花在"采样多样性 + 聪明选样"常比训 LoRA / 在线梯度自适应更值**。方案 ② / ③ 若目的是"提升生成质量"，先比一比"评价 LoRA + 干预" vs "多采样 + 评价选样"哪个划算，别默认前者。

**VDS-TTT**（arXiv 2505.19475）—— **支持方案：verifier 选样 → 只训 LoRA**
- 怎么做：learned verifier 给一批候选打分，**选高分伪标签**（置信度过阈值）配对成训练数据，**只微调 LoRA adapter** 做 test-time training。
- 数据 / loss：verifier 驱动的自监督 SFT（仅 LoRA）。三 benchmark × 三 LLM，比 base 相对 +最高 32.29%、比"用 verifier 但不 TTT" +6.66%。
- 与我们的区别：这是**"高分轨迹 → LoRA"这条主链的直接同构与背书**（verifier 打分选样 → 只训 LoRA），但它是**离线选样再训**、不是"推理中途干预"。我们 line A 的"用主线 `traj_score` 选高分轨迹蒸 LoRA"和它几乎一样，可当实现参考。

**Beyond Perplexity（TTT memory 审计）**（arXiv 2607.00368）
- 怎么做 / 结论：提出行为层评测框架，审计 TTT/memory 工作。发现一步 LoRA 更新能降 support/answer loss（跨 3 个 Qwen3 规模），但**自由 recall 仍为零** —— **proxy 指标改善 ≠ 部署行为改善**。
- 与我们的区别：警示方案 ① 的技能 LoRA / 方案 ② 的评价能力，**别只看 loss 或 rubric 分下降就宣称有效**，要用**行为层对照**（带/不带、later recall / 下游动作）验证真收益（呼应 §4 影子对照、§10 DuoMem 的 CD 单独只 +1.4 的教训）。

### 11.5 双 LoRA 方案落地建议（综合上面证据）

| 子命题 | 直接背书 | 反方 / 风险 | 建议 |
|---|---|---|---|
| ① 技能 LoRA（skill→LoRA） | S2L / LatentSkill / ParametricSkills / VDS-TTT | Beyond-Perplexity（proxy≠行为）；DuoMem CD 单独仅 +1.4 | **先做 S2L 式（一 skill 一 LoRA、离线 SFT、主线 traj_score 选样）**；必须带**带/不带对照**验证增益；skill 一多再上 hypernetwork（LatentSkill） |
| ② 评价 LoRA（rubric→judge） | Plug-and-Play Judge（rank-16 LoRA 顶 70B） | RGSD/RCSD：verifier 贵/不稳时**直接蒸进 policy 更优**；SCATR：轻量头比 LoRA 省 8000× | **先决策产物是不是"独立可复用的评价分"**：要 → 训评价 LoRA（复用 `score_lora_path`）；只为给 student dense 信号 → 走 RGSD 直蒸；只为排序/门控 → SCATR 轻量头 |
| ③ 每 N token 在线 PRM | Chain-of-LoRA（多 LoRA 热切原型）；MetaClaw（生产 RL-PRM） | 一次前向只挂一个 LoRA、无中途插入钩子；Surprisal/SCATR：搜索选样常更值；TTT 过锐化 | **最后做**；第一版用 `prompt_logprobs` **旁路打分 + 只留痕不干预**；确认 PRM 分与主线 hard verifier 一致性够高再考虑写交错解码调度器 |

**对标基线**：**MetaClaw（有 code、OpenClaw 生产、双线镜像）**是整套方案最该研读与对标的工作；**Chain-of-LoRA** 是子命题 ③ 的机制原型；**RGSD** 是子命题 ② 决策的关键对照。我们相对它们的增量仍是 §10.10 那四条（显式毕业梯度、失败前置归因、抽取器被下游效用反向蒸馏、可回滚发布门控）——双 LoRA 只是把 line A 的"权重"具体化成"技能 LoRA + 评价能力"，不改变整体闭环定位。

### 11.6 定稿：查错 LoRA 作为参数化 memory —— 结论与实验方案

> 本节是 §11 讨论收敛后的**决策记录 + 实验计划**。核心转变：把"评价 LoRA"从"给数据打分的 judge"重新定位成**不动 base 的参数化 memory（line B 第二载体），专门给 base 补一个"在线查错"能力**。技能 LoRA（LoRA-2）**本轮暂缓**（理由见末尾）。

#### 11.6.1 定位（钉死的几条前提）

1. **base 全程冻结，永不训。** 从根上杜绝知识遗忘——这是硬底线。所有能力增量都以**可插拔 LoRA**形式外挂，本质是 line B（参数化 memory），不是 line A。
2. **LoRA-1 = 查错器（过程级 memory）。** 它不改 base 的知识，只给 base 补"边生成边发现自己踩了 rubric 里哪类错"的能力（如"公式第 k 步描述错""工具调用缺参数"）。发现错误后把**具体问题**注回 context，引导 base 改。
3. **可回滚性 ≈ 删一条 memory。** 因为不动 base、LoRA 可插拔，一版 adapter 训坏了直接换/摘，撤销成本远低于"重训 base"——这消解了"权重侧犯错难撤"的顾虑（那顾虑只对"蒸进 base"成立）。
4. **蒸馏数据 = 问题定位 + 打分（不是只给分）。** 要蒸出的是"**可定位、可操作的过程诊断**"能力，所以 teacher rubric 的产出必须到"哪一步/哪个 tool call/缺什么"这一粒度，而非单一 PASS/FAIL 标量（这是能否蒸出"查错"而非"打分习惯"的前提，对应 §10.7 原子化可验证条目 / 多角色 rubric）。

#### 11.6.2 自进化引擎：超大模型 + 本地 LoRA 双端检测

```
超大模型(teacher rubric) ──检测出错误──┐
                                        ├─► 分歧/teacher 独有的错误 = 本地能力缺口 = 训练信号 ─► 升级 LoRA-1
本地查错 LoRA-1          ──检测出错误──┘                                                          │
                                                                                                 ▼
                          下一轮两端一起查 ──► 本地 LoRA-1 持续追平 teacher 的查错能力（追平后 teacher 少调、省成本）
```

- 与主线蒸馏同构：主线是"student 生成能力追平 teacher"，这里是"**本地 LoRA 的查错能力追平 teacher**"，同一套 `llm_backup` (teacher 对/本地错) 配对机制，被蒸对象换成"查错"这一角色。
- **双端分歧同时是"该审上游"的探针**：本地 LoRA 与 teacher **系统性**分歧（成规律、非零星）时，要么本地没学到位（继续训），要么 **teacher/rubric 本身有系统性偏差**——后者是上游数据源/超大模型质量问题，**在上游治**（换更强 teacher、多角色 rubric 交叉、rubric 可靠性审计），不指望下游 LoRA 兜（呼应 §10.7"先坐实 verifier 再用它"）。

#### 11.6.3 三阶段实验路径（每阶段是下阶段的 gate，早止损）

**阶段 0（先做）：不引入任何 LoRA，纯 teacher LLM 在线查错，测"上限"。**
- 数据集：**数学**（对错相对明确、查错信号干净，理想试验田）。
- 机制：base 每隔 N 个 token **暂停** → 用 **teacher LLM（走 `llm_backup`）** 判当前生成有没有踩 rubric 错 → 有则把发现的问题注入 context → 继续生成。
- 目的：**验证"在线过程级查错 + 注入 rubric"这个机制本身能否抬升数学解题上限**。用最强 teacher 代表能力天花板。
- 性质：**纯可行性 gate，不训任何东西**。若最强 teacher 在线查错都提不了分，后面蒸 LoRA 更无意义 —— 先证信号有价值。
- 工程：实验期"每 N token 暂停判断"**可用最粗暴实现**（停→跑一次 teacher 判断→拼 rubric→续），**不追性能**；生产化才需要专门的交错解码调度器（见待解决项）。
- **已有实现**：`cookbook/exp/embedding/eval_dualline_math.py`（`--mode dualline`）。它复用 `eval_gpqa_rag.py` 的 MATH 分层加载 / 采样参数 / `answers_match` 判分，token 级分段生成：每 `--chunk-tokens` 暂停 → `RubricVerifier.diagnose()`（无 student sampler，走 llm_backup teacher）查错 → 命中且分数低于 `DUALLINE_CHECK_FLOOR` 则把 fix/原因作 `[Checker]` 注入续写。对照基线 `--mode baseline`（＝单遍生成，等同 `eval_gpqa_rag --mode direct`），同一 200 条子集直接比较 overall / 分层准确率与 checks/injections 计数。launch 配置：`dualline_math`。

**阶段 1（阶段 0 证明有效后）：把 teacher 的查错能力蒸成 LoRA-1，替换在线 teacher 调用。**
- 用阶段 0 收集的 (完整核验 CoT + 问题定位 + 打分) 数据蒸 LoRA-1（可复用 `score_lora_path` 入口）。**数据须正负配平**（成功段"无错、continue" + 失败段"定位/原因/建议"），构造约束见 11.6.6。
- 目的：验证**本地低秩 LoRA 能否追平 teacher 查错**（＝待验证 b"学不学得动"）。实验要**把 rank × rubric 产出粒度当两个变量扫**（ParametricSkills 的 SHINE 退化反例说明：配置不对会打不过 in-context，学不学得动不是 0/1）。

**阶段 2（远期）：LoRA-2 技能 + 召回**——本轮暂缓，见 11.6.5。

#### 11.6.4 三条反驳 → 回应 → 限定（决策留痕）

| 反驳 | 我们的回应 | 仍需守住的限定 |
|---|---|---|
| **① 廉价 query 路由只解决"该不该召回"，没解决"召回内容本身可能是错的"** | 不走"召回一条可能有错的 memory"，而是把"**查错能力**"蒸进 LoRA，让 base 内生地边做边纠错，**绕开召回可信度问题** | 前提：rubric 产出要到"**可定位错误**"粒度，否则只蒸出打分习惯、蒸不出查错 |
| **③ 权重侧坏数据难撤 / 会不会被污染** | LoRA 对**单条随机坏数据**抗性强于 RAG（梯度统计稀释 + base 低秩先验；RAG 一条即直入 context 无稀释）；且不动 base、可插拔，撤销≈删 memory | 抗的是**单条噪声**，**不抗系统性偏差**——系统性错误会被梯度强化。故 §4 归因质量仍是地基；系统性偏差归上游治（11.6.2） |
| **（Substrate Asymmetry）参数化 memory 在"该缺的要拒答"上惨败** | 承认：这是 LoRA "缺检不到信号"的固有短板，**不是污染问题** | **LoRA 只装可泛化行为/模式（如查错），易变事实仍留文本 memory**（§0 分工判据）；别用 LoRA 装事实 |

#### 11.6.6 查错 LoRA 的训练数据构造（三条硬约束，钉死）

> 复用现有 `RubricVerifier` 的 rubric + `llm_backup` 配对采集管线来攒数据，但现有 scoring prompt 刻意"只出 PASS/FAIL、不出解释"（省 token、便于 comparator 对齐），**直接拿来训会蒸出"打分习惯"而非"查错能力"**。故新增一个"诊断模式"产训练样本，须同时满足：

1. **分数与原因必须一次调用同源产出（不可分两次采样拼）。** 若分数、原因来自两次独立采样，二者可能逻辑矛盾（判 FAIL 但原因说"没问题"），LoRA 学到错位映射。实现上：单次调用同时产 `per-criterion verdict + 每条 FAIL 的定位/原因/建议`，聚合分数由这批 verdict 算得，**原因与分数天然一致**。

2. **正负样本都要产、且要均衡（不能只在低分/被 gate 段跑诊断）。** 只见"错的"会把 LoRA 训成"逢查必报错"的挑刺偏置——它在线运行时每 N token 都硬报一个错，注入噪声反拖垮 base。必须让它见过大量"**看完一段、逐条核验、全部 PASS、结论=本段无过程错、无需干预**"的样本，才有能力在线输出关键的"OK，继续"信号。故诊断要**按比例覆盖高分成功段**，与失败段配平。

3. **CoT 要完整（核验过程，非只报结论）。** 正负两类样本的 target 都写成完整推理链：
   - 成功段：`逐条核验 → 每条为何 PASS → 结论：无过程错误，continue`
   - 失败段：`逐条核验 → 定位到第 k 条 FAIL → 错在哪 / 为什么 / 建议修法`
   完整 CoT 才对齐 LoRA-1 在线"边看前缀边判断"的实际形态；只给"这里错了"的结论会让它学不到核验过程，泛化差（呼应 §11.4 Beyond-Perplexity："proxy 改善 ≠ 部署行为改善"）。

> 落地路径（不改动服务打分/准入的现有 `RubricVerifier` 便宜路径）：新增诊断入口（`explain=True` 变体或独立 `_diagnose_once`，带 `@llm_backup` 自动落配对数据），schema：`input=[segment + rubric]`，`target=[完整核验 CoT + verdict + (FAIL 时) 定位/原因/建议]`。
>
> **采样策略（已定）**：
> - **采集期全量存储、不做平衡、不设成本上限** —— 对每个 segment 都产完整诊断 CoT（正负都存），先把数据完整跑出来。正负配比、降采样等**留到训练采样阶段**再定，避免采集期过早丢信息。
> - **注意训练/部署分布差**：离线诊断看的是"已完成的整段"，LoRA-1 在线看的是"生成中途的前缀"——离线攒数据能省掉阶段 0 的交错解码调度器工程，但**仍需一次"带/不带 LoRA-1 在线注入"的行为层对照**才算真验证（不能只看离线诊断准确率）。

#### 11.6.5 待验证 / 待解决 / 暂缓

- **待验证 a（已定方案）**：蒸 LoRA-1 时数据须含"问题定位 + 打分"，二者**一次调用同源产出**，且**正负样本都产、带完整核验 CoT**（详见 11.6.6 三条硬约束）。
- **待验证 b**：低秩 LoRA 学不学得动查错 —— 阶段 1 跑实验，扫 rank × 信号粒度。
- **待解决（生产化）**：vLLM **一次前向只挂一个 LoRA、无中途插入钩子**，"每 N token 暂停查错"生产化需自写**交错解码调度器**；实验阶段用粗暴实现绕过。
- **待解决**：LoRA-1（旁路查错）与未来 LoRA-2（技能常驻）**同时在线的调度冲突**（单请求单 LoRA 限制）。
- **暂缓：LoRA-2（优秀轨迹 skill 召回）** —— 本轮不做。理由：(1) 它**必须先训 LoRA 才能用，没有"纯 LLM 不训练"的验证捷径**，不适合当前"先用 LLM 测上限"的实验起点；(2) 训它需要**大量 trajectory** 作输入。两个已知前置问题记录备查：**P1 技能可能 per-user、跨用户不可迁移**（长期解＝只把 global 可泛化技能进 LoRA-2，用户专属留文本 memory，归 §0 分工；实验阶段忽略）；**P2 冷启动召回**（首个 query 可能是"你好"无信息量）——解法优先级：**A 延迟召回**（无信息量不召回，用 `intent`/信息量阈值判，"你好"本就不该召回）＞ B 滚动召回（每轮用累积上下文重判）＞ C 用 system/场景先验预挂。
