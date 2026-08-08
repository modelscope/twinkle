"""E18 的拒绝筛：增量达阈 -> 不超长 -> pass_rate 最大（并列内长度/rubric 相似度拆平局）。

这是本臂**唯一的自变量**。E12 靠「rubric 重生成 + 2-in-8 验证」入池，没有排序；E18 在做对的
候选里再排一次序，只留唯一胜者，并把打分全过程写进数据集文件供事后审计。

`_rubric_similarity` 与 skill_ablate/methods.py 逐字同源（含那条踩坑注释），搬过来是为了让
本目录自包含、不再 import 那棵 2 万行的 methods.py。同源搬过来的泄漏门已删，理由见下。
"""
import re
from collections import Counter
from typing import Any, Dict, List, Optional

# --- 第二道筛已删：泄漏检测对 coding 任务恒为 False ------------------------------------
# ⭐ 原 `answer_leaked` / `leak_blocks` 从 skill_ablate/methods.py 搬来，但那边是**数学**任务：
# `reference_answer` 是 `'42'` 这样的答案字符串，`str(ref) in skill` 的子串匹配有意义。
# BCB 的 `reference_answer` 是个 ~2500 字符的 dict（task_id / entry_point / test / code_prompt /
# doc_struct / canonical_solution），该匹配要求 skill 逐字包含整个 dict 的 Python repr
# （含 `{'task_id': 'BigCodeBench/598', ...}`）—— 而 skill 上限 SKILL_CHAR_LIMIT=1500 字符，
# 物理上装不进。实测：只有把整个 dict 原样粘进去才判 True，截断 50 字符加前缀就已为 False，
# 触发概率恒为 0。
# 真正的泄漏通道在 `test`（隐藏单测的断言期望值）与 `canonical_solution`，需要另写检测；
# 留着一个恒 False 的门只会给人「已经防住了」的假安全感，所以整体删除。


# --- 第二道筛：skill 与 rubric 诊断的词频余弦 ----------------------------------------------
# 定位：在「executor 已做对」的候选里挑与诊断内容对得上的那条，压掉两类假赢家 ——
# 与诊断无关的碰巧做对（含泄漏式速通的残余），以及与谁都不像的空泛套话。
# 刻意用去停用词的词频余弦而不是 tfidf/语义模型：可迁移性判别器一节实测「仅 tfidf」
# in-sample 0.983 / OOS 0.541 是纯过拟合；词频余弦纯 stdlib、确定性、可离线复算。
_SIM_STOPWORDS = frozenset(
    'the a an and or of to in is are be for with that this it on as by from at not no was '
    'were will would can could should may might do does did have has had you your we they '
    'he she its if then than so but into over under out up down when where which what how '
    'why all any each more most other some such only own same very'.split())
_SIM_WORD_RE = re.compile(r"[a-z][a-z'\-]{2,}")


def rubric_similarity(skill: str, rubric: str) -> float:
    """内容词词频余弦 ∈ [0,1]；任一侧无内容词返回 0。"""
    ca = Counter(w for w in _SIM_WORD_RE.findall((skill or '').lower())
                 if w not in _SIM_STOPWORDS)
    cb = Counter(w for w in _SIM_WORD_RE.findall((rubric or '').lower())
                 if w not in _SIM_STOPWORDS)
    if not ca or not cb:
        return 0.0
    dot = float(sum(v * cb[k] for k, v in ca.items() if k in cb))
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# --- 两道筛合体 -----------------------------------------------------------------------------
def select_winner(cands: List[Dict[str, Any]], rubric: str, reference: Any = None, *,
                  len_budget: int = 0, skill_char_limit: int,
                  base_pass_rate: float = 0.0,
                  min_pass_gain: float = 0.0) -> Optional[Dict[str, Any]]:
    """按 **pass_rate 增量**选胜者；增量不达 min_pass_gain 就返回 None。

    ⭐ 为何不能用旧的「做对就入池」（M=1/T=0）：那时 with_pass 只有 0/1，易题上「加任何
    skill 都对」，8 个候选全部 with_pass=1 —— 此时「挑哪条」完全由长度决定（易题 rubric 为空，
    相似度恒 0），等于往数据集里灌随机 skill。现在 executor 每个 skill 跑 M=8 次，
    with_pass 是连续通过率，同一题的不同 skill 之间才有方差（如 8/8 vs 5/8）。

    筛选顺序：
      a. **增量达阈**：`with_pass >= base_pass_rate + min_pass_gain`。
         min_pass_gain>0 时，仅仅「没弄坏」（tie，如 8/8 -> 8/8）**不够格**：那种样本对
         「学会写有效 skill」没有监督信号。拉低通过率的更是直接淘汰。
      b. 超 skill_char_limit 过滤。（原本还有一道泄漏门，已删 —— 它在 coding 任务上恒为 False，
         详见文件头部注释。）
      c. 取 **pass_rate 最大**的一档（允许并列）；并列内部按 **rubric 相似度**取高。

    ⭐ c 的层次顺序很关键：pass_rate 是**客观效果**，rubric 相似度是**内容对齐**，
    长度只是**形式偏好**。长度已彻底移出择优路径（只在相似度也并列时做确定性拆平，
    取较短者）：旧版先按 `abs(len - len_budget)` 砍掉一半候选，而实测 66% 的题 8 个候选全部
    with_pass=1.0，于是长度成了事实上的唯一决策依据 —— 它把信息量大的长候选系统性淘汰。

    ⭐ base_pass_rate 接近 1 时 a 几乎不可满足（天花板效应）：8/8 的题永远拿不到 +2/8，
    因此会被成建制排除在训练集外 —— 这是调用方想要的行为（只训真正有提升空间的题），
    但意味着池子会偏向中等难度题，看 `signal/n_accepted_easy` 确认。

    胜者会被标上 `pass_gain`（= with_pass - base_pass_rate）与 `gain_kind`：
      * 'improve'：pass_gain > 0；'tie'：== 0（仅当 min_pass_gain=0 时才可能返回）。

    rubric 为空（易题无诊断）时相似度恒 0，并列内退化成取较短者 —— 此时已经是
    「效果完全相同」的候选，拿什么拆平局都不影响效果，只是个确定性要求。

    `reference` 已不再使用（泄漏门删除后唯一的消费方消失），`len_budget` 同样已废弃；
    两个形参保留只为不打断现有调用方的写法。
    """
    need = base_pass_rate + min_pass_gain
    ok = [c for c in cands
          if c.get('parseable') and (c.get('with_pass') or 0.0) >= need - 1e-9]
    survivors = [c for c in ok if len(c['skills']) <= skill_char_limit]
    if not survivors:
        return None
    # a/c：先按客观效果取最大档，再在并列内按 rubric 相似度拆平局。
    top = max((c.get('with_pass') or 0.0) for c in survivors)
    tied = [c for c in survivors if (c.get('with_pass') or 0.0) >= top - 1e-9]
    for c in tied:
        c['rubric_similarity'] = rubric_similarity(c['skills'], rubric)
    # 长度只作**最后的确定性拆平**（相似度也并列时），不再参与择优：
    # 实测 66% 的题 8 个候选全部 with_pass=1.0，此时 `abs(len - LEN_BUDGET)` 事实上成了唯一
    # 决策依据，而它有系统性偏差 —— 中文表达同样内容的字符数天然更少（357 vs 705），永远
    # 更贴近预算，于是「离 400 最近」被翻译成了「选中文模板」，把信息量大的长英文候选全部
    # 淘汰。长度是形式偏好，不该越过效果与内容对齐，故彻底移出择优路径。
    best = max(tied, key=lambda c: (c['rubric_similarity'], -len(c['skills'])))
    best['kept'] = True
    best['pass_gain'] = round((best.get('with_pass') or 0.0) - base_pass_rate, 6)
    best['gain_kind'] = 'improve' if best['pass_gain'] > 1e-9 else 'tie'
    return best


def gain_stats(cands: List[Dict[str, Any]], base_pass_rate: float) -> Dict[str, int]:
    """候选级增量计数（相对裸解 pass_rate），给 train_log 做监控。

    `degraded` 是关键项：skill 把通过率拉低了。它完全不会反映在 accept_rate 上
    （那些候选只是默默被汰掉），持续偏高就是 skill-gen 在写有害提示的直接证据。
    """
    out = {'improved': 0, 'tied': 0, 'degraded': 0}
    for c in cands:
        if not c.get('parseable'):
            continue
        wp = c.get('with_pass')
        if wp is None:
            continue
        if wp > base_pass_rate + 1e-9:
            out['improved'] += 1
        elif wp < base_pass_rate - 1e-9:
            out['degraded'] += 1
        else:
            out['tied'] += 1
    return out


def filter_stats(passers: int, survivors: int) -> Dict[str, float]:
    """第一道筛后的超长丢弃率，进 train_log。

    指标名保留 `leak_or_overlength_dropped_fraction` 不改：泄漏门删除前它的触发率恒为 0，
    所以新旧 run 的这个数值本来就完全可比（一直只在统计超长），改名反而会断掉曲线。
    """
    return {'train/leak_or_overlength_dropped_fraction':
            ((passers - survivors) / passers) if passers else 0.0}
