# Copyright (c) ModelScope Contributors. All rights reserved.
"""Declarative ablation matrix (E1-E12) + run order.

Frozen decisions (skill_quality_analysis.md):
- 12 experiments; run nothink before think, the SFT method (E12) LAST and manually gated.
- Unified knobs: executor frozen T=0; skill-model train T=1.0 × 8 rollouts; eval T=0.5 × 4
  rollouts (query-only, no rubric); skill-max-tokens = 8192 (think) / 4096 (nothink).
- view B = query-only; view A = rubric line, eval still query-only (knowledge-transfer probe).

This module is intentionally dependency-free (pure stdlib) so it can be imported and unit-
smoke-tested without torch / twinkle / a GPU.
"""
from dataclasses import dataclass
from typing import Dict, List

# --- training methods (internal keys) --------------------------------------------------
# bnpo        view B query-only GRPO/BNPO main loop (reuses v2 process_chunk verbatim).
# rl_ab       view A RL, AB split: first bare-problem greedy solve; WRONG problems -> A line
#             (skill sampled under query+rubric), RIGHT problems -> B line (query-only);
#             both go through executor greedy -> reward -> in-group BNPO, trained together.
# rl_err      view A RL, error-only: same as rl_ab but the B line is NOT trained
#             (single-variable contrast vs rl_ab on "does training the right-answer B line help").
# opsd        view A On-Policy Self-Distillation: student(query-only) logps pulled toward
#             teacher(query+rubric) logps per token (loss='opsd'); error problems only.
# improve_sft view A improve-skill + SFT: first-pass 1 skill; correct -> positive pool
#             (no leak, <=4096 chars); wrong -> rubric regen (2-in-8 pick 1) -> negative pool;
#             1:1 accumulate -> SFT.
# sft         view A plain SFT: bare-problem wrong -> rubric -> regen (2-in-8) -> accumulate SFT.
# logp_rl    E14+: query-only skill-gen, executor T>0 samples a correct pseudo-GT solution S,
#             rubric/API audits that executor response, then reward each skill by
#             Δ mean logP_executor(S | problem + skill) with answer-leak penalty.
# logp_gt    E15: same dense executor-logP reward as logp_rl, but the target S is DeepMath's
#             external R1 reference solution (record 'solution'); NO executor rollout and NO
#             rubric audit (view B, query-only) -> runs much faster. Validates the logP path
#             against a strong external target instead of a self-sampled pseudo-GT.
# passrate_hinge E16: view B query-only. Reward = pass_rate(M rollouts, T>0) minus a hinge
#             truncation penalty (only rollouts whose executor output nears the 8192 budget are
#             penalized) minus a leak gate; trained problems are pre-filtered to the danger band
#             (baseline executor output long / not all-pass). Data-driven closure of the reward
#             probe (skill_quality_analysis.md 2026-07-29): pass_rate is the only real signal,
#             trunc is the strongest dense side-signal, base_tok is the strongest problem filter.
METHODS = ('bnpo', 'rl_ab', 'rl_err', 'opsd', 'improve_sft', 'sft', 'logp_rl', 'logp_gt',
           'passrate_hinge', 'reflexion', 'rejection_sft')
VIEW_OF_METHOD = {'bnpo': 'B', 'rl_ab': 'A', 'rl_err': 'A', 'reflexion': 'A',
                  'opsd': 'A', 'improve_sft': 'A', 'sft': 'A', 'logp_rl': 'A', 'logp_gt': 'B',
                  'passrate_hinge': 'B', 'rejection_sft': 'A'}
STYLES = ('narrative', 'pitfall', 'freeform')
THINKINGS = ('on', 'off')
TASKS = ('math', 'code')


@dataclass(frozen=True)
class ExpSpec:
    name: str            # E1..E14
    method: str          # one of METHODS
    thinking: str        # 'on' | 'off'
    style: str           # 'narrative' | 'pitfall' (ignored by align='seam': SEAM prompts bypass style)
    optional: bool = False  # E12(sft): manually gated (RUN_SFT=1), runs last
    align: str = 'v2'    # 'v2' | 'seam' — sets v2._ALIGN_MODE (prompt/判分/executor 嵌套全开关)
    # E14+ 稠密 reward：executor 先按 T>0 采样 K 条找正确伪 GT S；训练 reward 不再 rollout，
    # 而是算 Δ mean logP_executor(S | problem + skill)。默认字段复用 reward_rollouts/temperature。
    reward_rollouts: int = 1
    reward_temperature: float = 0.0
    smt_override: int = 0   # force skill_max_tokens regardless of the think rule; 0 = default rule
    # 任务族（2026-07-31 用户拍板把 E4/E17 换到 BigCodeBench）：
    #   'math' = DeepMath-103K + \boxed{} 数值判分（E1-E16、E18 原样）
    #   'code' = BigCodeBench + 跑 unittest 判分，executor/skill-gen/rubric 三套 prompt 全换
    # 依据：数学域上 rubric 无增量甚至负向（−0.056），BFCL 上为零（+0.002），只有 BigCodeBench
    # 这种"机器给出可定位失败证据（异常/断言/行号）"的任务上 rubric 才有增量（+0.135, p=4e-5，
    # 见 bcb/bcb_eval0_probe.py 与 code_task.py 模块注释）。
    task: str = 'math'
    # executor(base_sampler) 的 thinking。E1-E18 全是 'on'（历史口径，勿动）。
    # 'off' = 2026-07-31 新增的 nothink 对照（E19 math / E20 code）：bcb 探针实测 think 的
    # executor 有 34-50% rollout 撞满预算且是字面死循环（8-gram 重复率 p50=0.835），加预算
    # 到 20000 无效；关掉后截断归零、裸解 0.378 > 0.324、rubric 增量 +0.135 vs +0.080。
    # 注意这是 executor 侧；skill 模型仍由 thinking 字段控制（两个新臂都保持 skill think=on，
    # 因为 v2 实测 skill 侧 nothink 会把完整解答写进 <skills>，等于换标签的泄漏）。
    executor_thinking: str = 'on'

    @property
    def view(self) -> str:
        return VIEW_OF_METHOD[self.method]

    @property
    def needs_rubric(self) -> bool:
        return self.view == 'A'

    @property
    def skill_max_tokens(self) -> int:
        # 显式 per-spec override 优先。
        # ⚠️ 2026-07-29 实测推翻了此前“think 长度与 skill 质量无关，截掉长尾不损失信号”的探针结论：
        # E4 在 8192 与 4096 下的受控 A/B（其余配置全同，比共有的 chunk 0-21）显示 4096 是净损失——
        # parse 率 0.939 -> 0.740，无条件 cand_pass 0.689 -> 0.649，撞 think 顶 0.063 -> 0.268。
        # 4096 臂看起来“exec 更短、trunc 更低”纯属幸存者偏差（pass|parse 0.734 -> 0.877），
        # 因为 26% 的候选连 <skills> 都没写完就掉出了统计口径。
        # 见 .tmp_analysis/think_budget_ab.py。think 模式一律用 8192。
        if self.smt_override:
            return self.smt_override
        # think must have room for <think> + <skills> (4096 truncates to an empty block).
        # seam align: 人工拍板用 8192（不复刻 SEAM 原版 4096：think 模式下 4096 会把大量候选截断在
        # <think> 里、压低 parseable，与“think 模式 skill-max-tokens 必须 8192”的矩阵规范保持一致）。
        if self.align == 'seam':
            return 8192
        return 8192 if self.thinking == 'on' else 4096

    @property
    def loss(self) -> str:
        return 'opsd' if self.method == 'opsd' else 'bnpo'

    @property
    def exp_dir(self) -> str:
        # output.ablate12/E{n}_{method}_{think}_{style}/  (seam align / multi-rollout reward 加后缀区分)
        suffix = '_seam' if self.align == 'seam' else ''
        if self.reward_rollouts > 1:
            suffix += f'_r{self.reward_rollouts}'
        # 换数据集必须换目录：否则新语义的 run 会撞上已跑完的数学 run（DONE.json 直接跳过、
        # 曲线与 gen_records 混在一起、config 指纹对不上）。
        if self.task != 'math':
            suffix += f'_{self.task}'
        # executor 口径也必须进目录名：同一个 ExpSpec 换 executor thinking 后 baseline 缓存
        # （key=problem）与 eval 读数全变，撞同一个目录会把两套语义的曲线混在一起。
        if self.executor_thinking != 'on':
            suffix += '_execnothink'
        return f'{self.name}_{self.method}_{self.thinking}_{self.style}{suffix}'

    @property
    def swanlab_exp(self) -> str:
        return f'ablate12_{self.exp_dir}'


# --- the 12-experiment matrix (declarative; order field below drives execution) --------
MATRIX: List[ExpSpec] = [
    # group 1 — view B BNPO: think × style, no-rubric baseline
    ExpSpec('E1', 'bnpo', 'off', 'pitfall'),
    ExpSpec('E2', 'bnpo', 'off', 'narrative'),
    ExpSpec('E3', 'bnpo', 'on', 'pitfall'),
    # E4/E8 的 smt_override=4096 已被上面 skill_max_tokens 里记录的 A/B 推翻（净损失 4 个点
    # 无条件 pass + 20 个点 parse 率）。E8 保留原值只是为了不改动“已跑完的臂”的可复现配置。
    # ★ E4 于 2026-07-31 换到 code 任务并同时删掉 override（用户拍板）：既然要重跑，就按
    #   think=on 的规范值 8192 跑，与 E17 同口径可比。旧数学 E4 的产物在
    #   output.ablate12/E4_bnpo_on_narrative/（新 run 落在 ..._code/，互不覆盖）。
    ExpSpec('E4', 'bnpo', 'on', 'narrative', task='code'),
    # group 2 — view A RL-AB-mix: same grid as E1-E4, isolates "rubric rescues zero-grad groups"
    ExpSpec('E5', 'rl_ab', 'off', 'pitfall'),
    ExpSpec('E6', 'rl_ab', 'off', 'narrative'),
    ExpSpec('E7', 'rl_ab', 'on', 'pitfall'),
    ExpSpec('E8', 'rl_ab', 'on', 'narrative', smt_override=4096),  # 同 E4：重跑前应删掉此 override
    # group 3 — view A training-method comparison (fixed think+narrative), sft last & optional
    ExpSpec('E9', 'rl_err', 'on', 'narrative'),
    ExpSpec('E10', 'opsd', 'on', 'narrative'),
    ExpSpec('E11', 'improve_sft', 'on', 'narrative'),
    ExpSpec('E12', 'sft', 'on', 'narrative', optional=True),
    # group 4 — SEAM-align ablation: same data pipeline as the rest of the matrix, but ALL
    # prompt/parsing/executor-nesting rules follow SEAM (align='seam' -> v2._ALIGN_MODE):
    # actor uses SEAM EXPERIENCE_PROMPT (<memory_item>), executor sees the nested
    # prompt_text+response_text(+think), lpem-parity greedy scoring, actor budget 4096.
    # Query-only BNPO main loop (= SEAM's training form); eval stays the matrix-unified
    # query-only readout so E13 is directly comparable with E1-E12.
    ExpSpec('E13', 'bnpo', 'on', 'narrative', align='seam', executor_thinking='off'),
    # group 5 — E14+ 稠密 reward：E4 的部署形态（query-only skill-gen / eval 不变），训练时
    # executor 先 T=0.7×16 采样，选本地判分正确且非截断的伪 GT S，并用 rubric/API 产审计诊断；
    # skill reward = Δ mean logP_executor(S | problem + skill)（leak 不进 reward，只做监控）。该臂同时降测量噪声
    # 和抬内容信号，替代旧版 T=0.5×4 多数 rollout 0/1 reward。
    ExpSpec('E14', 'logp_rl', 'on', 'narrative', reward_rollouts=16, reward_temperature=0.7),
    # group 6 — E15 稠密 reward 的 GT 版验证：与 E14 同一套 executor logP reward，但 logP 目标 S
    # 换成 DeepMath 自带的 R1 参考解（record 'solution'），不再 executor rollout、不再 rubric 审计
    # （view B / query-only）。用于验证“ΔlogP(强外部参考解 | 题+skill)”是否走得通；因省掉 K 次
    # executor 采样 + rubric API，单 chunk 比 E14 快很多。
    ExpSpec('E15', 'logp_gt', 'on', 'narrative'),
    # group 7 — E16 数据驱动收敛臂：探针（skill_quality_analysis.md 2026-07-28/29）判死 logP，
    # 确认 pass_rate 是唯一真实信号、trunc 是最强稠密辅助、base_tok>5000 是最强筛题维度。
    # reward = mean_i(correct_i·eff_i) - kappa·mean_i(1-eff_i)，护栏 max(reward, pass_rate-1/M)；
    # eff = (1-alpha·len_pen)·(1-beta·loop_pen)，len_pen 从 5500 起二次凸爬升（数据标定见
    # .tmp_analysis/reward_shape_calib.py）。leak 不进 reward，只做监控。
    # skill_max_tokens=8192（2026-07-29 拍板）：4096 的 A/B 判为净损失，见 skill_max_tokens 注释。
    # 训练题预筛危险带（baseline 输出长）。view B query-only，不用 rubric。
    ExpSpec('E16', 'passrate_hinge', 'on', 'narrative',
            reward_rollouts=8, reward_temperature=0.5, smt_override=8192),
    # group 8 — E17 Reflexion 臂：唯一目的是检验"rubric 注入权重外信息"能否让 skill 变得可学。
    # 与 E16 的三处结构差异（2026-07-29 人工拍板）：
    #  1) 只在【裸 executor 做错】的题上训练与评测，做对的题完全不碰。E16 的 lift 分解
    #     +0.170 = 救回 +0.230 - 破坏 -0.060（skill 挂在裸对的题上会砸掉 10.6% 保持率），
    #     条件化按构造把破坏项归零。见 .tmp_analysis/lift_source_decomp.py。
    #  2) skill-gen 走 view A（query+rubric）。E16 判定 reward 对"该写什么文本"几乎无信息
    #     （结果层 SNR 2.25 -> 文本层 0.046，损失 ~50 倍）；rubric 是外部 API 的诊断，是本臂
    #     唯一的新变量，也是它可能不重演 E16 结局的唯一理由。见 .tmp_analysis/batch_size_math.py。
    #  3) 不用 base_tok 危险带筛题（base_tok_floor=0）。
    #     ⚠️ 当初的理由已被实测证伪，保留在此以免重蹈：本以为"去掉筛选后错题集就是
    #     推理错 + 没写完的混合，才测得到方法修正"。实测（e17_param_calib.py + E16 全 1600
    #     题）：全量题池的 527 道错题里 96.96% 是 base_tok>=8192（没写完），只有 3.04%
    #     （16 道）是写完但答错；而 floor=5000 筛选后是 97.71%。也就是说 floor 不是截断主导的
    #     原因，题目本身是（level>=6 配 8192 预算），去掉筛选只把可用的"方法错"样本从
    #     2.3% 提到 3.0%（每 chunk 24 道错题里平均 0.73 道）。保留 floor=0 只是因为它严格
    #     不差于 floor=5000，不要再把它当成本臂能测到 reflexion 的理由。
    #     后果：rubric 在 ~97% 的题上只能说"你超预算了"，signal/wrong_trunc_frac 会直接
    #     开在 0.97。想真正测"方法修正"必须先把 executor 预算提到 16384。
    #     见 .tmp_analysis/e16_redteam5.py、e17_param_calib.py。
    # 批量对齐：chunk_size=128 裸解后取 --reflexion-k=24 道错题，每次更新恒定 24 题 x 8 候选。
    # K=24 不是拍的：E16 每次更新实际只有 23.46 组（全程 1173 组），而累积证据 ∝sqrt(N)
    # 且 E16 在文本层只累到 1.9 sigma，再砍组数就没判别力了。chunk=128 是为了把
    # P(凑不满 24) 压到 0.012%（全量题池裸错率实测 0.329，不是筛选后子集的 0.446）。
    # 见 .tmp_analysis/e17_param_calib.py。
    # reward 形状沿用 E16 的 passrate_hinge（死区 5500/二次凸/leak 只监控），但 rollout 数改为
    # M=1（2026-07-30 拍板）：每个 skill 只让 executor 推理一次，reward = 这一次对/错。
    # ⭐两个连带后果（改前必读）：
    #  1) pass_rate 从 9 档（0,1/8,...,1）退化为 2 档（0/1），组内无方差的概率大升
    #     —— 这正是 E4(M=1) 62% 零梯度组的成因，M=8 当初就是为了绕开它。
    #  2) methods.py 的不可反转护栏 pen_cap = 1/M - 1e-6：M=8 时是 0.125（惩罚只能微调
    #     同档内排序），M=1 时变成 ~1.0，长度/兜圈惩罚能把做对的候选拉到贴近 0。
    #     形式保证（做对过永远排在没做对前）仍成立，但 reward 量级上从“以正确率为主”
    #     变成“以长度惩罚为主”。
    # T=0（对齐 SEAM 的 grm.rollout.temperature=0 与 E4/E13）：打分时 executor 贪心确定性解一次，
    # 同一个 skill 重跑 reward 不变 —— 消掉 executor 采样噪声，reward 只反映 skill 本身的差异。
    # （skill 模型自己的采样温度是另一个参数 skill_gen_temperature=1.0，不受此影响。）
    # ★ 2026-07-31 换数据集（用户拍板）：task='code' —— BigCodeBench + 跑 unittest 判分。
    # 换的理由是上面 (2)(3) 两条在数学域已被实测封死：97% 的裸失败是"没写完"，rubric 只能
    # 说"你超预算了"，三个数据集横比下来 rubric 增量 −0.056(deepmath) / +0.002(BFCL) /
    # **+0.135(BigCodeBench, p=4e-5)**，唯一的差别是 judge 手上有没有机器给出的可定位失败
    # 证据（异常类型/断言差异/失败用例名）。code 分支把这份证据喂进 judge（code_task.diag_segment）。
    # 连带变化：裸错率从数学的 0.329 变成 0.5625（2026-07-31 dry run 实测 18/32，think=on/8192；
    # ⚠️ 不是 probe 的 0.715，那是 nothink+4096 的读数），所以 K=24 需要 chunk=64
    # （P(凑不满 24)=0.001；chunk 48 会有 15.4% 的更新组数不足）；题池 908 题，chunk64×50
    # ≈4.5 个 epoch 重复题（用户同意）。--min-level/--eval-min-level 在 code 下自动失效。
    ExpSpec('E17', 'reflexion', 'on', 'narrative', task='code',
            reward_rollouts=1, reward_temperature=0.0, smt_override=8192),
    # group 9 — E18 拒绝采样 SFT（2026-07-30 用户拍板）：采集与 E17 同源（裸解错题 -> rubric ->
    # rubric 条件化 skill-gen think 模式，executor greedy T=0 单次判分），但不做 RL：每题在做对的
    # 候选里按 leak 过滤 -> 长度贴近 len_budget -> 与原始 rubric 相似度最高 三道筛取唯一胜者，
    # 写本地数据集文件，攒够 16 条 SFT 一次（weight=1，nothink 布局响应；2026-07-30 从 128 改小，
    # 使 50 次更新在 ~80 chunk 内可达），训完同步权重到 vLLM。
    # eval：同一个 skill_sampler vLLM 临时切 nothink 模板跑 query-only greedy（trainer 侧实现）。
    # reward_rollouts/temperature 对本臂无效（判分固定 greedy 单次），填 1/0.0 只为指纹如实。
    ExpSpec('E18', 'rejection_sft', 'on', 'narrative',
            reward_rollouts=1, reward_temperature=0.0, smt_override=8192),
    # group 10 — E19/E20 executor-nothink 对照（2026-07-31 用户拍板）：与 E17 逐字同构的
    # reflexion 臂，唯一变量是 **executor 关 thinking**（skill 侧仍 think=on）。
    # 起因：bcb 探针在 4096/12288/20000 三档 think 预算下测到 rubric 增量 +0.047/+0.058/+0.080，
    # 而 nothink 一档是 +0.135（p=1e-4）；增量与"裸失败里 no_code 的占比"严格反向
    # （69%/54%/50%/0%）。截断样本经查是字面死循环（8-gram 重复率 p50=0.835、同一长句重复
    # 92 次），所以加预算无解、只能关 think。两个臂分别回答：
    #   E19（math）：deepmath + 数学 prompt/rubric。数学域此前的 rubric 增量是 −0.056，而
    #     那次 97% 的裸失败是"没写完"；关掉 executor think 后失败会变成"写完但答错"，
    #     这是第一次能在数学域上把"rubric 是否有用"与"截断"分开测。
    #   E20（code）：BigCodeBench + code prompt/rubric，直接把探针里 +0.135 那一档搬到训练。
    # ⚠️ 规模不能照抄：裸错率随 executor 口径变（code think 0.5625 -> nothink 约 0.62；
    #    math nothink 未测，预期远高于 think 的 0.329），chunk 由 run_ablate12.sh 的
    #    EXEC_NOTHINK_* 单独给，dry run 后按实测复核。
    ExpSpec('E19', 'reflexion', 'on', 'narrative', executor_thinking='off',
            reward_rollouts=1, reward_temperature=0.0, smt_override=8192),
    ExpSpec('E20', 'reflexion', 'on', 'narrative', task='code', executor_thinking='off',
            reward_rollouts=1, reward_temperature=0.0, smt_override=8192),
    # group 11 — E21 freeform 文体（2026-08-01 用户拍板）：以 E2 为模板的 query-only BNPO（view B、
    # 无 rubric），唯一变量是 skill-gen system 换成 SKILL_GEN_FREEFORM“招式菜单”：不锁 narrative/
    # pitfall/toy 固定文体，让模型按题自选最有用的形态（分析/概念/预判纠错/迷你示范/直白执行
    # 指令，甚至 “let's think step by step”），T=1.0×8 自然铺开、组内择优。动机：固定 hint
    # 消融实测 hint 内容语义贡献≈0、增益几乎全来自“有个 skill 块 + 催答案收尾”（fixed_hint_probe.py：
    # A9_wrapperonly/A4_garbage 与有义 hint 打平、A7_budget 最高 +0.16），故放开文体看模型能否自选出更优组合。
    # ★ 刷 thinking='on'（非照 E2 的 off）：freeform prompt 依赖“先私下想再选形态”，nothink 下无处
    #   思考会把推理直接写进 <skills>（line 758 记录的泄漏失败模式）。其余同 E2：bnpo/math/narrative-长度预算。
    ExpSpec('E21', 'bnpo', 'on', 'freeform'),
]

# execution order: all nothink first, then think; E13 (seam-align baseline) right after E6;
# E14 (rubric pseudo-GT + executor logP dense reward) right after E7 per 2026-07-28 人工拍板;
# E15 (GT-target logP validation) right after E14; the data-hungry SFT method dead last.
# 2026-07-31 用户拍板：E19/E20（executor nothink）排在 E4/E17 之前先跑 —— 探针已判定
# nothink 是 rubric 增量最大且唯一无截断混杂的口径，先拿这两个臂的结论。
RUN_ORDER: List[str] = ['E1', 'E2', 'E5', 'E6', 'E13', 'E3', 'E7', 'E14', 'E15', 'E19', 'E20', 'E21', 'E4', 'E8', 'E16', 'E17', 'E18', 'E9', 'E10', 'E11', 'E12']

BY_NAME: Dict[str, ExpSpec] = {e.name: e for e in MATRIX}


def get_spec(name: str) -> ExpSpec:
    key = name.strip().upper()
    if key not in BY_NAME:
        raise KeyError(f'unknown experiment {name!r}; valid: {sorted(BY_NAME)}')
    return BY_NAME[key]


def ordered_specs(include_optional: bool = True) -> List[ExpSpec]:
    specs = [BY_NAME[n] for n in RUN_ORDER]
    return specs if include_optional else [s for s in specs if not s.optional]


def _self_check() -> None:
    """Invariants that guard against typos when editing the matrix."""
    assert set(BY_NAME) == set(RUN_ORDER), 'RUN_ORDER must cover every matrix entry exactly once'
    assert len(RUN_ORDER) == len(set(RUN_ORDER)) == len(MATRIX), 'duplicate / missing names'
    for e in MATRIX:
        assert e.method in METHODS, f'{e.name}: bad method {e.method}'
        assert e.thinking in THINKINGS and e.style in STYLES, f'{e.name}: bad think/style'
        assert e.align in ('v2', 'seam'), f'{e.name}: bad align {e.align}'
        assert e.task in TASKS, f'{e.name}: bad task {e.task}'
        assert e.executor_thinking in THINKINGS, \
            f'{e.name}: bad executor_thinking {e.executor_thinking}'
        assert not (e.task == 'code' and e.align == 'seam'), \
            f'{e.name}: seam align is math-only (SEAM prompts/parsing are \\boxed 数值口径)'
        assert e.reward_rollouts >= 1, f'{e.name}: bad reward_rollouts {e.reward_rollouts}'
    # nothink-before-think ordering within contiguous runs is a soft convention, not asserted.


if __name__ == '__main__':
    import sys
    _self_check()
    if '--plan' in sys.argv:
        # machine-readable run plan for the launcher:
        # name<TAB>exp_dir<TAB>think<TAB>smt<TAB>optional<TAB>task<TAB>executor_thinking
        for e in ordered_specs():
            print(f'{e.name}\t{e.exp_dir}\t{e.thinking}\t{e.skill_max_tokens}\t'
                  f'{int(e.optional)}\t{e.task}\t{e.executor_thinking}')
        sys.exit(0)
    print(f'{len(MATRIX)} experiments; run order: {" -> ".join(RUN_ORDER)}')
    hdr = f'{"name":<4} {"view":<4} {"method":<12} {"think":<6} {"style":<10} {"align":<5} {"smt":<5} {"loss":<5} opt'
    print(hdr)
    print('-' * len(hdr))
    for e in ordered_specs():
        print(f'{e.name:<4} {e.view:<4} {e.method:<12} {e.thinking:<6} {e.style:<10} '
              f'{e.align:<5} {e.skill_max_tokens:<5} {e.loss:<5} {"Y" if e.optional else ""}')
