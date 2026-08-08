"""E23 的全部 prompt 文本与拼装函数：executor / 教师 judge / skill-gen 三处。"""
# flake8: noqa: E501
#   prompt 正文按「一段一行」书写，折行会改变真正发给模型的文本，故整文件豁免行长检查。
import json
from typing import Any, Dict

# ===========================================================================
# executor
# ===========================================================================
# BigCodeBench 官方 instruct 模式的硬性交付要求。
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

# ⭐ E23 的定义性特征：executor 看到 actor 的**完整产出**（<think> + <skills>），不是只看抽出来的
# <skills> 块。第一句必须说明「接下来这坨是什么」，否则 executor 会把 <think> 当成任务描述的一
# 部分。第二句与「只给 skills」的对照臂逐字相同，使两臂只差「能否看到推理过程」这一个变量。
_WRAPPER_WITH_THINK = (
    'Guidance model transcript:\nFor this task, a separate problem-solving guidance model was '
    'asked to analyse it and produce advisory skills. Its full output follows, including its '
    'private reasoning:\n{hint}\n'
    'Prefer using its techniques when they fit, but if you have a clearly better '
    'implementation, you may diverge. Be concise and accurate.\n')


def direct_prompt(problem: str) -> Dict[str, Any]:
    return {'messages': [{'role': 'system', 'content': EXEC_SYSTEM},
                         {'role': 'user', 'content': problem}]}


def skill_solve_prompt(problem: str, skill: str, raw_response: str = '') -> Dict[str, Any]:
    """题面 + actor 全文（含 <think>）。

    skill 为空（actor 只写了 <think>，或写到上限没闭合）-> 干净 direct，而不是把一坨无结论的思考
    过程当指导塞进去。故意**不补 <|im_end|>**：那会让 advisory 落在非法 ChatML 位置并改变紧邻
    token 的 BPE 切分，只为「与 SEAM 逐 token 对齐」才值得，这里的代价是多一个变量。
    """
    skill = (skill or '').strip()
    if not skill:
        return direct_prompt(problem)
    hint = (raw_response or '').strip() or skill
    return {'messages': [{'role': 'system', 'content': EXEC_SYSTEM},
                         {'role': 'user',
                          'content': problem + '\n\n' + _WRAPPER_WITH_THINK.format(hint=hint)}]}


# ===========================================================================
# 教师 judge（rubric 诊断）
# ===========================================================================
DIAG_SYSTEM = """\
You are a code-failure classifier. You are given a Python task description (with the required signature), a list of FAILURE CLASSES, one attempted implementation, and the REAL error that attempt produced when the task's unit tests were run.

Your job is NOT to review the code. It is to (a) name the SINGLE decisive failure class, (b) quote the evidence for it out of the test error, and (c) state the prior knowledge that would have PREVENTED it.

Output STRICT JSON (no prose outside it), either

{"addressable": true, "class": "<CLASS CODE>", "evidence": "<verbatim fragment of the test error that proves this class>", "required_value": "<the exact value, name or behaviour the tests demand -- a title, a label, a dictionary key, a numeric bound, an exception that must be raised -- or null if the failure is not about matching a demanded value>", "required_value_source": "<verbatim fragment of the TASK section that states that value or demands that behaviour, or null if the task never states it>", "reason": "<one short sentence: which assumption was wrong, at which step>", "prior": "<one sentence of transferable knowledge>", "secondary": ["<CLASS CODE>"], "independent_causes": 1}

or, when you cannot ground a decisive class in the test error:

{"addressable": false, "why": "<one short sentence>"}

Rules:
- "evidence" MUST be copied verbatim out of the test error you were given. If the error does not let you single out ONE class, output addressable=false instead of guessing. Never invent evidence.
- The error tells you WHICH assertion fired first, not WHAT caused it. A failing shape, type or value assertion is normally the last link of the chain, not the class. Ask "which wrong belief produced this?" and classify THAT. Classify by the assertion itself only when nothing upstream is wrong: when the computation is right and merely the kind of object handed back is not what the caller reads.
- "class" is the one class the evidence implicates. List other classes that are also off in "secondary"; leave it empty if there are none.
- "independent_causes" counts mutually independent root causes, not symptoms of one. Three or more means no single short warning could have saved this attempt.
- "prior" is the hard part. It MUST STAY TRUE AND USEFUL IF THIS TASK IS DELETED: a fact about a library, an API default, a format directive, or a testing convention. It must NOT contain any identifier, literal, column name, file name or number taken from this task, and must NOT describe the steps of this task's solution. Write "pandas writes JSON Lines rather than one JSON document when the lines flag is set", NOT "pass lines=False here".
- "prior" must say WHY the wrong result arose -- the rule the attempt had backwards -- and NOT what the correct result should look like. "The result must carry one row per input key" merely restates the assertion that failed and teaches nothing, because the reader already has the task description; "a merge defaults to an inner join and silently drops keys missing on either side" is the belief that was actually absent.
- For class TESTCONTRACT and class EXCEPTION, "required_value" MUST NOT be null. Those two classes are BY DEFINITION about failing to match something the caller demands -- a title, a label, a key, an exception -- so name that thing. If you find yourself unable to name it, you have the wrong class.
- "required_value" and "required_value_source" are a CITATION, not an opinion. Whenever the failure comes down to matching something the tests demand, put that demanded thing in "required_value", then go back to the TASK section and copy out the fragment that states it into "required_value_source". You may only fill "required_value_source" with text you can actually see in the TASK section -- it is checked against it verbatim. If the task never states it, write null, and the diagnosis will be discarded, which is the correct outcome: an engineer reading only the task could not have known it either.
- That check is where a plausible-looking diagnosis does the most damage. "Tests assert exact string equality on titles" is a true and transferable sentence, and it is still worthless when the title itself appears nowhere in the task -- the engineer learns that the string matters but not what it is. Do not let a good-sounding prior talk you out of the citation.
- Set addressable=false rather than writing a vacuous prior such as "implement the description carefully". A sentence that merely says validation must be written, or that a parameter must satisfy the tests, is vacuous however factual it sounds; a usable prior names a library, function, format or convention that the reader could look up.
- Output only the JSON object."""

DIAG_USER = """\
## Task
{query}

## Failure classes
{rubric}

## Attempted implementation and its test error
{segment}

Now output the classification JSON object."""

# ⭐ 判据按「什么先验知识能预测这个失败」切分，而不是按「代码哪里错了」。
# 依据是 E23.t1 的 480 组实测：旧表第 3 条只管**签名合法性**（调用能否被接受），于是所有「调用被
# 接受、但行为/默认值不符合假设」的失败（to_json(lines=True)、mkdir 不带 parents、glob('*') 含
# 目录、strftime('%Z') 多后缀）全掉进兜底项「核心计算错」——它在 76% 的题上 FAIL，且 leak 率
# 0.106 是全体 0.021 的 5 倍：它救得回题，靠的正是让 skill 把解法写出来。BEHAVIOUR 就是补这个洞，
# 占旧兜底 FAIL 的 53%。每条都按**报错里的可观测特征**定义，这才能既判得准又分得开。
#
# ⚠️ 「按可观测特征定义」的代价是教师容易**照最先响的那条断言分类**，于是 SHAPE 会接手本属
# BEHAVIOUR 的题（BCB/441：einsum 输入下标 ikl 应为 jkl，症状是 shape 断言先挂 -> 判 SHAPE ->
# prior 只讲输出下标决定形状 -> 8 个候选一起只改输出下标，形状对了数值仍错，整组 reward 0）。
# SHAPE 的定义因此显式排除「算错导致形状/数值不对」，DIAG_SYSTEM 里也有一条反症状规则兜着。
# (code, 给 skill-gen 的短标签, 给教师判定用的完整定义)
FAILURE_CLASSES = [
    ('BEHAVIOUR', 'a library call behaves differently from what was assumed',
     'The call is accepted, but the function\'s real behaviour, default value or precondition '
     'differs from what the code assumed: a flag that changes the output format, a default that '
     'does not do what its name suggests, a required preparation step, a half-open range.'),
    ('SIGNATURE', 'the call itself is not accepted',
     'The function, attribute or module does not exist, or it does not take the argument names, '
     'positions or count that were passed.'),
    ('SHAPE', 'the object handed back is not the one the task asks for',
     'The KIND of object is wrong irrespective of the values inside it: wrong container or element '
     'type, wrong nesting, an unconsumed lazy object where a value was expected, a figure handed '
     'back where the caller reads an axes. NOT for a result whose shape or values came out wrong '
     'because the computation was wrong -- that belongs to whichever class names the wrong '
     'assumption in the computation, usually BEHAVIOUR.'),
    ('TESTCONTRACT', 'what the caller inspects was never set, or a required side effect never ran',
     'The returned object exists but does not carry what the tests read off it, or a side effect '
     'the task requires was never performed: labels and titles left unset, a resource not cleaned '
     'up, a call the task says to make never made.'),
    ('DETERMINISM', 'the result is not reproducible or not exactly comparable',
     'Unseeded randomness, reliance on iteration order, missing or wrong sorting, rounding or '
     'precision other than stated.'),
    ('NORMALISATION', 'a text or boundary convention is wrong',
     'Case sensitivity, surrounding whitespace, regex anchoring, separator handling, inclusive '
     'versus exclusive bounds, off-by-one.'),
    ('DEGENERATE', 'a degenerate input crashes instead of being handled',
     'Empty, single-element, all-equal or missing-key / missing-column input.'),
    ('EXCEPTION', 'the exception contract is not met',
     'The exception the task specifies is not raised, is raised as a different type, or an '
     'unrelated exception escapes.'),
]


def render_classes() -> str:
    """完整定义只给教师；skill-gen 那侧只看短标签，避免它照着举例去写不相干的失败模式。"""
    return '\n'.join(f'- {code}: {desc}' for code, _short, desc in FAILURE_CLASSES)


def diag_query(problem: str, payload: Dict[str, Any]) -> str:
    """题面 + 任务自身声明的硬约定（签名、必需库、返回规格、应抛异常、文档示例）。
    只用题面信息、不含参考解答 —— 训练时同样拿得到，所以是「可得且非泄漏」的判据依据。"""
    try:
        doc = payload['doc_struct']
        doc = json.loads(doc) if isinstance(doc, str) else (doc or {})
    except Exception:
        doc = {}
    lines = [f"- required signature (must be reproduced verbatim):\n"
             f"{payload['code_prompt'].strip()}"]
    for key, label in (('reqs', 'must use these libraries'), ('returns', 'must return'),
                       ('raises', 'must raise'), ('params', 'parameters')):
        vals = [str(x).strip() for x in (doc.get(key) or []) if str(x).strip()]
        if vals:
            lines.append(f'- {label}: ' + '; '.join(vals))
    ex = [str(x) for x in (doc.get('examples') or [])]
    if ex:
        lines.append('- documented example calls:\n    ' + '\n    '.join(ex))
    return problem + '\n\nHard requirements declared by the task:\n' + '\n'.join(lines)


def diag_segment(roll: Dict[str, Any]) -> str:
    """★ rubric 路线唯一真正有效的一处：给 judge 的不是「输出全文」，而是**提交的代码 + 单测真实
    报错**（<think> 已被 extract_code 切掉）。报错是客观事实且不含参考解答 —— 这正是 code 域
    rubric 有增量（+0.135, p=4e-5）而数学 / BFCL 域没有的原因。"""
    return (f"### Submitted code\n```python\n{roll.get('code') or '(no parseable code block)'}\n```"
            f"\n\n### Result of running the task's unit tests\n"
            f"outcome: {roll.get('kind') or 'unknown'}\n"
            f"{roll.get('error') or '(no error output)'}")


# ===========================================================================
# skill-gen（pitfall 文体 + rubric 条件化）
# ===========================================================================
# 与 narrative 文体的对照变量是**广度 vs 聚焦**：narrative 穷尽所有失败模式（~300 词），pitfall
# 只挑决定性的那一个（<90 词）。E23 选 pitfall 是因为 executor 已经能看到 actor 的 <think>，
# narrative 会与思考过程大面积重复，pitfall 让两者的分工是「过程 vs 结论」。
SKILLGEN_SYSTEM = """\
You are a skill-generation model. Your <skills> block will be fed to a SEPARATE downstream engineer model that must implement the function on its own. The engineer sees the same task description and the same required signature, but NOT your private reasoning or the analysis below — it only sees what is inside <skills>...</skills>.

A diagnosis of a failed attempt at THIS task is provided to you: the class of the decisive failure, and the prior knowledge that would have prevented it. That prior is your material — it is a fact that stays true for other tasks too. Your job is to land it at the exact point in THIS task where it bites.

Then, inside <skills></skills>, write under 90 words:
- WARNING: name where the failure strikes — the operation or the hand-off point it happens at — and the wrong assumption behind it, in the terms of the failure class you were given.
- INSTEAD: one or two sentences naming what to guarantee at that exact point, phrased as the general rule rather than as this task's answer.
- End by telling the engineer to deliver one single fenced code block that reproduces the given imports and signature verbatim, with no explanation, no demonstration call and no tests around it.

Hard rules:
- Write about the failure class you were given. Do NOT substitute a different one and do NOT add a second: a made-up warning sends the engineer after something that is not the problem.
- Do NOT write the solution, and do NOT copy identifiers, column names, file names or literal values out of this task. Name the operation and describe the shape of what it returns instead of writing the call out.
- Self-contained: NEVER reference "the diagnosis", "the analysis", "the review" or "the previous attempt" — the engineer cannot see them, such phrasings cause hallucination. Address the engineer directly.

Put ONLY the guidance inside <skills></skills>.

Example:
<skills>
WARNING: the call doing the real work is accepted but does not behave as its name suggests: in the mode this task nudges you towards, it emits one record per row instead of one whole document, so the reader rejects it.
INSTEAD: confirm what that call emits in the mode you pick, and choose the mode whose output the consumer on the other side expects.
Deliver one fenced code block reproducing the given imports and signature verbatim, with no explanation, demonstration call or tests.
</skills>
"""

SKILLGEN_USER = """\
Task:
{problem}

Diagnosis of a failed attempt (for your eyes only; do NOT reference it in the skill):
{rubric}

Now write the <skills> guidance:"""


def skillgen_prompt(problem: str, rubric: str) -> Dict[str, Any]:
    return {'messages': [{'role': 'system', 'content': SKILLGEN_SYSTEM},
                         {'role': 'user', 'content': SKILLGEN_USER.format(problem=problem,
                                                                          rubric=rubric)}]}
