"""E18 的全部 prompt 文本与拼装函数：executor / 教师 judge / skill-gen 三处。

与 cookbook/human/e23_prompts.py 同源（同一套 BigCodeBench 交付要求与失败分类表），差别只在
skill-gen 的系统提示：E18 是**拒绝采样 SFT**，采集时要 think 模式、训练时用 nothink 布局，
所以这里额外提供 query-only 的训练轨迹拼装（train_prompt）。
"""
# flake8: noqa: E501
#   prompt 正文按「一段一行」书写，折行会改变真正发给模型的文本，故整文件豁免行长检查。
from typing import Any, Dict

# ===========================================================================
# executor
# ===========================================================================
# BigCodeBench 官方 instruct 模式的硬性交付要求（与 e23 逐字相同，保证跨实验可比）。
EXEC_SYSTEM = """\
You are an expert Python engineer. You will be given a task description that ends with the exact \
import lines and function signature your solution must start with.

Deliver exactly one fenced Python code block and nothing else after it:
- Reproduce the given imports and the given function signature verbatim, including parameter \
names, order and default values.
- Add any further imports you need inside the same block; the block must run standalone.
- Return exactly the object type the task says to output. If it says the function should output \
a tuple, return a tuple in that order; if it names a matplotlib Axes, return the Axes object \
itself, not the Figure and not None.
- Implement the described behaviour for the general case, including the empty / single-element / \
missing-column edge cases and any exception the description says to raise.
- Do not call the function, do not print demonstrations, do not add tests, do not use \
`if __name__ == '__main__'`, and do not read from stdin.
- Do not include explanations outside the code block."""

# E18 的 executor 只看抽出来的 <skills> 块，**不看** actor 的 <think>：本臂的产物是要写进
# SFT 数据集、将来 query-only 部署的 skill 文本，采集期就必须按「部署时 executor 能看到什么」
# 来判分，否则筛出来的胜者依赖一段部署时不存在的思考过程。（E23 是相反的口径，故意保留差异。）
_WRAPPER_SKILL_ONLY = (
    'Hint:\n{hint}\n')


def direct_prompt(problem: str) -> Dict[str, Any]:
    return {'messages': [{'role': 'system', 'content': EXEC_SYSTEM},
                         {'role': 'user', 'content': problem}]}


def skill_solve_prompt(problem: str, skill: str) -> Dict[str, Any]:
    """题面 + 抽出的 <skills> 块。skill 为空 -> 干净 direct（等价裸解，不塞空指导）。"""
    skill = (skill or '').strip()
    if not skill:
        return direct_prompt(problem)
    return {'messages': [{'role': 'system', 'content': EXEC_SYSTEM},
                         {'role': 'user',
                          'content': problem + '\n\n' + _WRAPPER_SKILL_ONLY.format(hint=skill)}]}


# ===========================================================================
# skill-gen（采集用：rubric 条件化、think 模式）
# ===========================================================================
# ⭐ 全英文、且**不给回复格式模板**。
# 旧版是中文，并在类型 3/4 里给了带 `...` 占位符的回复示范（“根据你曾经犯过的错误……”、
# “该问题属于...问题，因此可以拆解为...步骤”）。实测后果：模型把骨架连 `...` 一起抄下来，
# 胜者中 79/148 含空模板、中文占比从 65% 升到 100%，唯一词数 376→123，candidate_pass_rate
# 从 0.856 塌到 0.701。语言还与选择器共谋：中文字符数天然更少，在旧的 LEN_BUDGET 口径下永远
# 更贴近预算而胜出。故：只说**要写什么**，不给句式；措辞由模型自己生成。
# executor 与 BCB 题面均为英文，skill 也必须是英文才能与之对齐。
# ⭐ 类型列表的**描述粒度必须齐平**，否则列表本身就是偏置：旧版 3/4 带了展开要求
# （“Explain what the weak point is and why…”、“the concrete steps needed…”），1/2/5 却只有光板
# 一句 —— 实测胜者里 t4 占 77-100%、t3 占 62-80%，而 t2 恒为 0%、t1 不过 4%。
# 同理不写“vary across attempts”：单次采样看不到其他 rollout，该句对单条生成无法执行；
# 多样性靠 N_SKILLS 个独立 rollout 的采样噪声，以及“只选一类 + 五类等价”的显式声明。
# ⭐ 2026-08-06：改为 **narrative 文体**（移植 skill2lora 的 SKILL_GEN_SYSTEM，见
# cookbook/exp/skill2lora/train_skill_v2.py:843-861）。原版（下方 _SKILLGEN_SYSTEM_TYPED，
# 已注释停用）是「五类里挑一类」的列表式，实测问题：
#   1. 46.6% 的胜者是「用 A 不要用 B」的 API 纠正，靠的是 rubric 里的库行为知识；
#   2. 12.3% 直接把只存在于隐藏单测里的列名（'closing_price' 之类）写进 skill —— query 里
#      根本没有，eval 时 query-only 无从得知，训练等于教模型凭空猜列名；
#   3. 5.9% 用「The critical error was...」这种事后复盘句式指代一次 eval 时不存在的失败。
# narrative 的三条硬约束正好对上后两条：强制第一人称自持句式、**明令禁止**指向外部上下文
# （skill2lora 的原注释写明这类措辞「会导致幻觉」）、不许代入本题具体数值。
# ⭐ rubric 的用法：单独一段说明「有诊断时当作证据用」，与下方禁指代外部上下文那条并不矛盾 ——
# 两者分属不同层：**任务指令层**要求模型靠诊断定位软肋（WHERE），**输出层**要求把它转写成
# executor 能直接执行的前瞻告诫（不得提及诊断本身）。这正是 skill2lora 的 REGEN_SYSTEM
# （train_skill_v2.py:1130-1147）的做法：步骤 2 要求「weaving together ... the pitfalls that
# actually tripped up the solving process」，而 Output requirement 同时禁止 "according to the given
# analysis/hints"。差异在于：REGEN 是「旧 skill + 诊断 -> 重写」的蒸馏场景，E18 是首次生成，
# 所以此处写成条件句（"If a grader's diagnosis ... is supplied"），无诊断时自动退化成纯预判。
# 为何不能直接写「根据你之前犯过的错误……」：executor 看不到任何「之前」，而且训练目标是
# query-only 的 —— 实测 5.9% 的胜者写成「The critical error was...」，eval 时模型无错可指只能编。
# 所以保留“定位到具体步骤/API”这个内核，只把时态从「已发生」换成「容易在此处发生」。
# ⚠️ 代价（skill2lora 已记录）：few-shot 例子占整条 prompt 的 65%，每次采样必然命中，会锁死
# 文体与长度 —— 多样性下降是预期内的，换来的是「删掉本题依然成立」的可迁移性。
# ⚠️ 本文件的 few-shot 例子必须是**代码域**的（原版是数学域的「末位数字/计数问题」，直接搬过来
# 会把 executor 往数学叙述上带）；收尾纪律句同理换成 BCB 的交付要求，不用 boxed 那句。
SKILLGEN_SYSTEM = """\
You are a skill-writing expert. Your <skills> block will be fed to a SEPARATE downstream executor model that must solve the Python task on its own. The executor will NOT see your private reasoning — it only sees what is inside <skills>...</skills>.

First, think privately: work out where the executor is most likely to get stuck, then step back and abstract WHAT MAKES THIS TYPE OF TASK GO WRONG into transferable guidance.

Second, if a grader's diagnosis of a failed attempt is supplied, use it as your evidence for WHERE the weak point is: read what actually went wrong, then decide which part of the approach needs the executor's attention. Fold that insight into the narrative as guidance the executor can act on before it starts — name the step or the API where the trouble lives and say what to do there instead, e.g. "the place this tends to go wrong is when you ..., so at that point you should ...". Do not report the diagnosis; convert it into advice.

Then write the <skills> block following these rules:
- Give general, transferable techniques for this TYPE of task: the library behaviour it relies on, the recommended approach, and the common pitfalls to avoid — plus a brief reason for each piece of advice so the executor understands why.
- Write it as one coherent analysis narrative (not a bullet list): first name what the task is essentially asking, then walk through how to approach it, blending the API contracts, steps, pitfalls and reasons into a single connected story.
- Write your judgements directly in the first person (e.g. "I think this step tends to ...", "A common mistake is ..., so you need to ..."), and phrase every issue as a self-contained, general technique.
- CRITICAL: Do NOT use phrasings that point to external context, such as "according to the given diagnosis", "the failed attempt", or "the previous error". The executor cannot see that context, and such phrasings will cause hallucination. State the pitfall as something that tends to happen at a particular step, not as something that already happened.
- CRITICAL: Do NOT name a column, key, or literal value that the task description does not itself state. If the task never names its columns, say how to discover them from the input instead of guessing names.
- Name the concrete API, argument, or keyword involved whenever the task description supports it.
- Keep it concise: aim for roughly one focused paragraph.

Put ONLY the methodology inside <skills></skills>.

Example:
<skills>
This task is essentially asking you to reshape tabular input and hand back a plot object, so the delivery contract matters here as much as the computation; I would pin down exactly what type the function must return before writing any logic, because returning a Figure where an Axes was requested fails even when every number is right. The first place this tends to go wrong is the input itself: it arrives as a plain container, and I find the single most common break in this type of task is assuming it is already a DataFrame — dictionaries and lists of tuples carry none of the frame methods, so reaching for column-based access on them raises immediately, and at that point you should check what the object actually is and build the frame from it explicitly. The next place to slow down is naming: let the task description dictate the column names and read them off the signature or the docstring rather than inventing plausible-sounding ones, and when the description never states them, derive them from the input's own keys instead of hard-coding a guess, because a name that merely sounds right will pass your own reading and still miss. A common mistake is treating an empty or single-element input as impossible, so decide up front whether it should yield an empty result or raise, and write that branch before the main path. Finally, when plotting, create the Axes explicitly and return that same object, since helper calls that draw on the current figure make it easy to hand back something you never configured. Overall I summarise this type of task as "fix the return contract, verify the input's real type, take names from the description, then handle the empty case before the happy path", because that is where the failures concentrate.
</skills>
"""

# ===========================================================================
# 【已停用】原「五类挑一类」列表式 skill-gen 提示（2026-08-06 换成上方 narrative）
# ===========================================================================
# 保留全文仅为记录历史口径与可回退：把下面的字符串改名回 SKILLGEN_SYSTEM 即可复原。
# 停用原因见上方 narrative 块的注释（隐藏契约泄漏 12.3% / 复盘句式 5.9%）。
# 注意它自身也修过两轮：类型描述粒度齐平（t1-t5 各 9-15 词）、以及那条前瞻视角规则 ——
# 这两笔修改都已被 narrative 的硬约束覆盖，回退时才需要重新评估。
_SKILLGEN_SYSTEM_TYPED = """\
You are a skill-writing expert. Your job is to write an advisory note that makes a downstream executor model solve the given Python task more accurately.

First decide where the executor is most likely to get stuck, then write the advice you believe helps most. Pick the ONE kind below that fits this task best. All five are equally worth choosing, and a single sharp sentence often beats a long note:
1. A plain instruction about how to approach the work, such as what to be careful about.
2. A calibration cue about how much to deliberate, or about trusting its own judgement.
3. A generalized lesson drawn from the grader's diagnosis of a previous failed attempt, if one is supplied. Explain what the weak point is and why the lesson prevents it.
4. A decomposition of the task into the concrete steps needed to solve it.
5. Any other kind of skill you judge useful, including an angle you would not normally try.

Rules:
- Do NOT solve the task and do NOT write code. You only write advice.
- Name the concrete API, argument, key, or value involved whenever you can.
- Be direct and specific. No filler, no restating the task, no placeholder text.
- Write forward-looking advice to someone who has not attempted the task yet. Do not refer to an error, mistake, or attempt as something that already happened.
- Choose your own wording and structure; there is no fixed format to follow.

Wrap your skills in <skills> ... </skills>.
"""

SKILLGEN_USER = """\
TASK
{problem}

GRADER'S DIAGNOSIS OF THE FAILED ATTEMPT
{rubric}

Write the advisory note now, wrapped in <skills></skills>."""

# ⭐ 带失败代码的变体。与 SKILLGEN_USER 的差别只有多出的 FAILED ATTEMPT 段，
# 段序是「题面 -> 失败代码 -> 诊断」：诊断紧贴写作指令，因为它才是要被消化的主结论；
# 把代码放中间让模型先看到证据再看结论，而不是反过来。
SKILLGEN_USER_TRAJ = """\
TASK
{problem}

CODE FROM A FAILED ATTEMPT (for your analysis only — the executor will never see it)
{trajectory}

GRADER'S DIAGNOSIS OF THE FAILED ATTEMPT
{rubric}

Write the advisory note now, wrapped in <skills></skills>."""


def format_trajectory(code: str, error: str = '', kind: str = '',
                      max_chars: int = 4000) -> str:
    """把一条失败 rollout 整理成给 skill-gen 看的文本块。

    ⭐ 头尾各留一半而不是直接截前 max_chars：Python 失败代码的关键信息经常在**末尾**
    （未闭合的分支、漏掉的 return、被截断的行），只留开头会把根因裁掉。

    ⚠️ error 只取前 400 字符：pytest 的 longrepr 能有几千字符且大量重复的堆栈帧，
    全带上会把预算吃光，而判别失败模式只需要头部的异常类型与消息。
    """
    code = (code or '').strip()
    if not code:
        return '(the attempt produced no extractable code)'
    if len(code) > max_chars:
        half = max_chars // 2
        code = (code[:half] + '\n\n... [%d characters omitted] ...\n\n' % (len(code) - max_chars)
                + code[-half:])
    out = ['```python', code, '```']
    if error:
        out.append('OBSERVED ERROR: ' + ' '.join(str(error).split())[:400])
    if kind:
        out.append('FAILURE CATEGORY: %s' % kind)
    return '\n'.join(out)


# ⭐ 训推一致：这份同时做**训练 prompt** 与 **eval prompt**，与 SKILLGEN_SYSTEM 一起改成英文；
# 两边语言不一致会让模型在采集与部署时面对不同分布。
SKILLGEN_SYSTEM_EVAL = """\
You are a skill-writing expert. Your job is to write an advisory note that makes a downstream executor model solve the given Python task more accurately.

First decide where the executor is most likely to get stuck, then write the advice you believe helps most.

1. You have been trained on many kinds of skills, from a one-line caution to a full step decomposition.
2. Your memory already holds what works best for different kinds of problems.
3. Analyse the task and choose the skills you judge most useful. A single sharp sentence often beats a long note.

Rules:
- Do NOT solve the task and do NOT write code. You only write advice.
- Name the concrete API, argument, key, or value involved whenever you can.
- Be direct and specific. No filler, no restating the task, no placeholder text.
- Write forward-looking advice to someone who has not attempted the task yet. Do not refer to an error, mistake, or attempt as something that already happened.

Wrap your skills in <skills> ... </skills>.
"""

SKILLGEN_USER_EVAL = """\
TASK
{problem}

Write the advisory note now, wrapped in <skills></skills>."""


def skillgen_prompt(problem: str, rubric: str, eval: bool,
                    trajectory: str = '') -> Dict[str, Any]:
    """trajectory 非空时切到带失败代码的 user 模板（由 KOD_USE_TRAJ 控制，默认关）。

    ⚠️ 只换 user 模板、**不换 system**：SKILLGEN_SYSTEM 里那条
    "Do NOT use phrasings that point to external context ... 'the failed attempt'"
    的禁令对带轨迹的情形更重要（模型看到真实代码后更容易写成事后复盘），
    换掉 system 会同时丢掉这条约束。
    """
    if not eval:
        if not rubric:
            rubric = ('No diagnosis is available for this task. Consider the other kinds of '
                      'skill instead.')
        user = (SKILLGEN_USER_TRAJ.format(problem=problem, rubric=rubric,
                                          trajectory=trajectory)
                if trajectory else
                SKILLGEN_USER.format(problem=problem, rubric=rubric))
        return {'messages': [{'role': 'system', 'content': SKILLGEN_SYSTEM},
                             {'role': 'user', 'content': user}]}
    else:
        return {'messages': [{'role': 'system', 'content': SKILLGEN_SYSTEM_EVAL},
                            {'role': 'user', 'content': SKILLGEN_USER_EVAL.format(problem=problem)}]}


# ===========================================================================
# 已废弃：训练/eval 统一走 skillgen_prompt(..., eval=True)
# ===========================================================================
# ⭐ 不要再用 TRAIN_SYSTEM / train_prompt。
# 训推一致要求「训练 prompt 与 eval prompt 逐字相同」，而 eval 用的是 SKILLGEN_SYSTEM_EVAL；
# 再并行维护一份英文 TRAIN_SYSTEM 只会让两边默默分叉。保留它仅为记录历史口径。
TRAIN_SYSTEM = """\
You are a problem-solving coach for a Python engineer. Given a task, write a short advisory note that anticipates the most likely decisive mistake and prevents it.

Requirements:
- Wrap the note in <skills> and </skills> tags.
- State what to do, in the imperative. Name the concrete API, argument, key, or value involved.
- Do NOT solve the task, do NOT write code, and do NOT state the expected output value.
- Keep it under 90 words."""


def train_prompt(problem: str) -> Dict[str, Any]:
    """已废弃。训练与 eval 统一用 `skillgen_prompt(problem, '', eval=True)`。"""
    raise NotImplementedError(
        'train_prompt 已废弃：训练/eval 请用 skillgen_prompt(problem, \'\', eval=True)，'
        '以保证两边的 system/user 逐字一致（训推一致）。')
