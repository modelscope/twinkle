# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout / prompt primitives for the ablation package.

Reuses v2 verbatim (imported, never edited):
- ``_run_samples`` sampler pad / take-first-N (Ray dp size rule),
- ``_skillgen_prompt`` query-only skill-gen (view B + all eval),
- ``_train_trajectory`` query-only train trajectory (view B, SFT samples, OPSD student),
- ``build_skill_solve_prompt`` / ``build_direct_prompt`` executor prompts,
- ``_parse_seq`` / ``_extract_skill`` / ``_clean_text`` / ``_answer_leaked`` / ``_empty_roll``,
- ``_regen_prompt`` improve-skill regeneration (has orig skill; view A improve_sft),
- style/thinking globals ``_SKILL_STYLE`` / ``SKILL_GEN_SYSTEM`` etc.

Adds ONLY the view-A rubric-conditioned pieces that v2 lacks:
- ``rubric_skillgen_prompt(problem, rubric)``: skill-gen conditioned on query + rubric
  diagnosis, NO prior skill (matches the RL flow step 3 "输入 query+rubric ... skillmodel
  rollout"). Style-matched (narrative / pitfall) to the main line.
- ``rubric_train_trajectory(rec)``: train trajectory that REBUILDS the query+rubric prompt +
  response, for view-A RL training only (train-with-rubric; the trajectory must match the
  prompt the skills were SAMPLED under). The SFT side keeps the query-only
  ``_train_trajectory`` so the SFT prompt口径 stays query-only (skill_quality_analysis.md #6).
- ``opsd_teacher_trajectory(rec)``: the OPSD teacher — EXACTLY the student's query-only
  prompt with the rubric APPENDED TO THE SYSTEM prompt (设计 871 行 "将 rubric 信息额外加入到
  system prompt 中"). Teacher and student therefore differ ONLY by the appended rubric block
  and score the SAME response tokens. (``rubric_skillgen_prompt`` is NOT used here: its
  system prompt differs wholesale from ``SKILL_GEN_SYSTEM``, which would confound the
  distillation signal with a prompt-style shift.)
"""
from typing import Any, Dict

import train_skill_v2 as v2
from train_skill_v2 import (  # noqa: F401  (re-exported for methods.py convenience)
    _answer_leaked,
    _clean_text,
    _empty_roll,
    _extract_skill,
    _parse_seq,
    _regen_prompt,
    _run_samples,
    _skillgen_prompt,
    _train_trajectory,
    build_direct_prompt,
    build_skill_solve_prompt,
)

# --- view-A rubric-conditioned skill-gen system prompts --------------------------------
# 中文注释：view-A 的 rubric 条件 skill-gen 提示词。仿照 skill_quality_analysis.md 的
# "rubric & skill" 模板，但去掉"你已经生成过一个 skill"的指涉（RL 线首步没有旧 skill，只有
# query + 一个失败尝试的 rubric 诊断）。文体与主链路一致（narrative / pitfall），且强制"自持、
# 不指向外部上下文"，因为下游 executor 看不到 rubric——指涉会导致幻觉。改进skill+sft 线另有旧
# skill，走 v2 的 _regen_prompt（含 orig_skill 字段，即 777-797 模板的逐字英文版），不用这里的提示词。
# narrative 版末尾拼入与 REGEN_SYSTEM 同一个 <skills> few-shot 例子（设计 793-796），锁定文体/长度
# 分布与主链路一致；程序化提取而非复制，保证与冻结的 v2 逐字相同。
# 2026-07-30：narrative 版补上收束纪律句的**指令**（pitfall 版一直有，v2 的 SKILL_GEN_SYSTEM /
# REGEN_SYSTEM 也有，只有这里漏了）。few-shot 例子里本来就带这句，但光靠示范不管用——E17 首个
# run 实测出现率 0.000-0.006，而同期 E16（有指令）是 1.000。上一轮把这句评为干预优先级 #1，
# 详见 skill_quality_analysis.md「E17 reflexion 臂截断漂移归因」第四节。
#
# 2026-07-30 第二次改（用户拍板）：从“连贯叙述”改为“问题-根因-规避”结构。
# ★ 命名提醒：`--skill-style narrative` 这个名字从此名不副实（已不再要求叙事体），
#   但没改：它进了 exp_dir / swanlab_exp / config 指纹，改名会让旧 run 数据对不上。
# ★ 只改 _RUBRIC_SKILLGEN_NARRATIVE（E17 专用，训练与 eval 同一个）；v2.REGEN_SYSTEM /
#   SKILL_GEN_SYSTEM 未动，所以 E1-E16 的 query-only 链路不受影响。
_RUBRIC_SKILLGEN_NARRATIVE = """\
You are a skill-generation model. Your <skills> block will be fed to a SEPARATE downstream executor model that must solve the problem on its own. The executor will NOT see your private reasoning or the analysis below — it only sees what is inside <skills>...</skills>.

An expert rubric analysis of a failed attempt on THIS problem is provided to you. Work from it to derive a COMPLETE, CONCRETE and ACTIONABLE set of instructions for how to avoid going wrong on this problem, and put your full line of thinking inside the <skills> block.

Then write the <skills> block following these rules:
- Structure it as problem -> location -> root cause -> countermeasure. Start by naming the failure mode(s) that ACTUALLY occurred on this problem. For each one: state WHERE it happens (which step, which formula or which theorem is being applied), WHAT goes wrong there, WHY it goes wrong (a misunderstanding of a specific concept / a wrong value substituted in / a missing precondition / ...), and WHAT to guarantee in order to avoid it ("to avoid this, make sure that when <condition>, you <action>").
- Cover exactly the failures that are real — no more. If only ONE thing went wrong (e.g. the mathematics was sound and the attempt merely failed to finish), write about that one thing in depth and stop. NEVER invent extra failure modes, and never pad the list to match a pattern: a made-up warning plants a wrong formula in the executor's head.
- If there are several failure modes, walk them in the order the executor will meet them, with ordinals ("First, ...; Second, when moving on to <step>, ..."), so it reads as a checklist; with a single failure mode, no ordinals are needed.
- Be concrete and executable. Every countermeasure must name the object it applies to (the formula, the quantity, the case being split). Do NOT write advice that would read the same on any other problem.
- CRITICAL: Do NOT solve the problem, reveal/compute the final answer, or substitute the problem's specific given numbers. Leave ALL concrete numbers for the executor to compute.
- Self-contained: NEVER reference "the analysis", "the rubric", or "the previous attempt" — the executor cannot see them, such phrasings cause hallucination. Address the solver directly.
- Close by telling the solver to commit and emit the answer in one pass without hesitating, naming the failure that hesitating causes. End the block with this exact sentence: "Avoid re-checking loops; box a bare number as soon as it is computed."
- Keep it under about 300 words.

Put ONLY the guidance inside <skills></skills>.

Example (several real failure modes):
<skills>
The recurring problems on this problem are: miscounting because symmetric configurations are treated as distinct, applying a permutation formula where the objects are actually indistinguishable, and dropping the division that removes duplicates. First, when you set up the count, the failure appears at the moment you choose between a permutation and a combination: the wrong branch is taken because "distinguishable" is read off the surface wording instead of from whether swapping two objects yields a genuinely different configuration. To avoid this, before writing any formula, state explicitly for each set of objects whether swapping two of its members changes the configuration, and only then pick the formula. Second, when moving on to the total count, compute it as if every object were ordered and distinguishable, because that quantity is unambiguous; the error to guard against here is folding the symmetry correction into this step, which makes the correction impossible to audit later. Third, when you apply the symmetry correction, the failure is using the wrong duplication factor — it comes from counting how many objects look alike rather than how many orderings map to the same configuration. To avoid this, derive the factor by asking how many distinct orderings of the interchangeable choices give the identical configuration, and divide the total by exactly that. Finally, commit to the result and emit the answer in one pass without hesitating; hesitating here restarts the case split and burns the budget before any answer is produced. Avoid re-checking loops; box a bare number as soon as it is computed.
</skills>

Example (a single real failure mode — the mathematics was sound, so nothing is invented):
<skills>
The one recurring problem on this problem is not mathematical: the derivation stays on the right track, but a correct intermediate result gets questioned instead of used, the same quantity is re-derived to double-check it, and the attempt is cut off before any answer is written. The failure appears after the setup is complete, at the moment the first candidate value is in hand; it happens because re-verifying feels safer than committing, yet every re-check repeats the same computation and produces nothing new. To avoid this, once an intermediate quantity is computed, treat it as settled and build the next step directly on it; choose one method at the start, stay on it, and write the final line as soon as the last quantity is evaluated. Avoid re-checking loops; box a bare number as soon as it is computed.
</skills>
"""

_RUBRIC_SKILLGEN_PITFALL = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block, NOT the analysis below.

An expert rubric analysis of a failed attempt on THIS problem is provided. From it, pinpoint the single decisive way a solver goes wrong on this type of problem. Then, inside <skills></skills>, write under 90 words:
- WARNING: name that decisive mistake concretely, in self-contained first person (e.g. "I think the step most likely to go wrong is ..."), and say why it is wrong.
- INSTEAD: one or two sentences pointing to the correct turn (technique name + where to apply it), without solving the problem or revealing any numeric result.
- End with: "Avoid re-checking loops; box a bare number as soon as it is computed."
Hard rules: the block must be self-contained — never reference "the analysis", "the rubric" or "the previous attempt"; the executor cannot see them.
"""

_RUBRIC_SKILLGEN_USER = """\
Problem:
{problem}

Expert rubric analysis of a failed attempt (for your eyes only; do NOT reference it in the skill):
{rubric}

Now write the improved <skills> guidance:"""

# --- code 任务（BigCodeBench）的 rubric 条件化 skill-gen -----------------------------------
# 与数学版同一个骨架（问题→位置→根因→规避 / 只写真实发生的失败 / 自持不指涉 rubric），
# 把"公式、定理、代入数字、boxed 裸数"换成"该调哪个 API、参数与返回形状、边界与异常、
# 交付一个 code block"。收尾纪律句也换掉：代码域的对应失败不是"兜圈不给答案"，而是
# "反复重写实现 / 输出解释与演示代码 / 改动给定签名"。
# ★ 这一版的信息优势来自 rubric 里带**单测真实报错**（code_task.diag_segment）：
#   bcb_eval0_probe 实测 rubric skill 0.513 vs query-only 0.382（+0.135, p=4e-5）。
_CODE_RUBRIC_SKILLGEN = """\
You are a skill-generation model. Your <skills> block will be fed to a SEPARATE downstream engineer model that must implement the function on its own. The engineer sees the same task description and the same required signature, but NOT your private reasoning or the analysis below — it only sees what is inside <skills>...</skills>.

An expert review of a failed attempt at THIS task is provided to you, including the real error its unit tests produced. Work from it to derive a COMPLETE, CONCRETE and ACTIONABLE set of instructions for how to avoid going wrong on this task, and put your full line of thinking inside the <skills> block.

Then write the <skills> block following these rules:
- Structure it as problem -> location -> root cause -> countermeasure. Start by naming the failure mode(s) that ACTUALLY occurred. For each one: state WHERE it happens (which step of the implementation, which library call, which returned object), WHAT goes wrong there, WHY it goes wrong (a wrong assumption about what an API returns / a keyword the API does not accept / an unhandled empty or missing-column input / the wrong object handed back to the caller / ...), and WHAT to guarantee in order to avoid it ("to avoid this, make sure that when <condition>, you <action>").
- Cover exactly the failures that are real — no more. If only ONE thing went wrong, write about that one thing in depth and stop. NEVER invent extra failure modes: a made-up warning sends the engineer after an API that is not the problem.
- If there are several failure modes, walk them in the order the engineer will meet them, with ordinals ("First, ...; Second, when building the return value, ..."), so it reads as a checklist.
- Be concrete and executable. Every countermeasure must name the object it applies to: the library function, the argument, the returned type, the edge case, the exception. Do NOT write advice that would read the same on any other task.
- CRITICAL: Do NOT write the solution code and do NOT paste concrete literal values from this task. Name the API and describe the shape of the value it returns instead of writing the call out.
- Self-contained: NEVER reference "the analysis", "the review", "the test error" or "the previous attempt" — the engineer cannot see them, such phrasings cause hallucination. Address the engineer directly.
- Close by telling the engineer to deliver one single fenced code block that reproduces the given imports and signature verbatim, with no explanation, no demonstration call and no tests around it.
- Keep it under about 300 words.

Put ONLY the guidance inside <skills></skills>.

Example (several real failure modes):
<skills>
The recurring problems on this type of task are: handing back the wrong object to the caller, assuming a grouping call returns a plain container when it returns an indexed one, and crashing instead of returning a well-defined result when the input is empty. First, when you build the return value, the failure appears at the very last line: a plotting helper is asked for and the figure is returned instead of the axes it drew on, or a tuple is required and only its first element comes back. To avoid this, re-read the sentence in the task that names the output, and make the last line return exactly that many objects in exactly that order, taking the axes object from the plotting call itself rather than from the figure. Second, when you aggregate, the failure is treating the result of the grouping call as a list: it is an indexed object whose labels are the group keys, so positional access silently reads the wrong group. To avoid this, convert it explicitly with the accessor the library provides before you index into it, and sort by the key the task names rather than relying on insertion order. Third, when the input has no rows or the named column is absent, the failure is an exception escaping from the aggregation. To avoid this, decide up front which of the two the task demands — a defined empty result or a specific raised exception — and write that branch before the main computation. Finally, deliver one single fenced code block that reproduces the given imports and signature verbatim, with no explanation, no demonstration call and no tests around it.
</skills>

Example (a single real failure mode — nothing is invented):
<skills>
The one recurring problem on this type of task is a keyword that the library function does not accept: the call is the right one for the job, but it is invoked with an argument name borrowed from a similar function in another module, so it raises before any of the logic runs. The failure appears at the single line that does the real work, and it happens because the argument list is recalled from memory instead of from the function being called. To avoid this, when you reach that call, pass only the arguments you are certain that exact function declares, prefer positional arguments for the ones the task names explicitly, and if a behaviour you need is not available as a keyword there, achieve it with a following operation instead of inventing a parameter. Everything else in this task is straightforward once the call succeeds. Deliver one single fenced code block that reproduces the given imports and signature verbatim, with no explanation, no demonstration call and no tests around it.
</skills>
"""

_CODE_RUBRIC_SKILLGEN_USER = """\
Task:
{problem}

Expert review of a failed attempt, with the real unit-test error (for your eyes only; do NOT \
reference it in the skill):
{rubric}

Now write the <skills> guidance:"""


def rubric_skillgen_prompt(problem: str, rubric: str) -> Dict[str, Any]:
    """View-A skill-gen conditioned on (problem + rubric diagnosis), NO prior skill.

    Style-matched to the main line via v2's ``_SKILL_STYLE`` global (set by main() from
    ``--skill-style``). narrative -> narrative rubric prompt; pitfall -> pitfall rubric prompt.
    code 任务只有 narrative 一版（E4/E17 都是 narrative；pitfall 未移植，落到同一个 prompt）。
    """
    if v2._TASK == 'code':
        return {'messages': [
            {'role': 'system', 'content': _CODE_RUBRIC_SKILLGEN},
            {'role': 'user', 'content': _CODE_RUBRIC_SKILLGEN_USER.format(
                problem=problem, rubric=rubric)}]}
    sys_p = _RUBRIC_SKILLGEN_PITFALL if v2._SKILL_STYLE == 'pitfall' else _RUBRIC_SKILLGEN_NARRATIVE
    return {'messages': [
        {'role': 'system', 'content': sys_p},
        {'role': 'user', 'content': _RUBRIC_SKILLGEN_USER.format(problem=problem, rubric=rubric)}]}


def rubric_train_trajectory(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Train trajectory whose PROMPT is the query+rubric skill-gen prompt + the response.

    Used by view-A RL (rl_ab / rl_err) only: train WITH rubric in the prompt (knowledge-
    transfer probe; eval is still query-only via v2 ``_skillgen_prompt``). The rebuilt prompt
    matches the prompt the skills were SAMPLED under (on-policy consistency).
    ``rec`` must carry 'problem', 'rubric' and 'response'. ``key_rounds`` marks the final
    assistant turn as the only trainable span (identical convention to v2 ``_train_trajectory``).
    """
    msgs = rubric_skillgen_prompt(rec['problem'], rec.get('rubric', ''))['messages']
    return {'messages': msgs + [{'role': 'assistant', 'content': rec['response']}],
            'user_data': {'key_rounds': [len(msgs)]}}


# 中文注释：OPSD teacher 的特权信息块——按设计 871 行要求放进 SYSTEM prompt，且只做“追加”，
# 保证 teacher 与 student 的 prompt 仅差这一段 rubric（最小差异，蒸馏信号不混入提示词风格漂移）。
_OPSD_TEACHER_SUFFIX = """

[Privileged context — an expert rubric analysis of a failed attempt on this problem. \
It is visible ONLY to you in this forward pass; the downstream executor never sees it. \
Use it to judge which guidance actually helps, but do NOT reference it explicitly.]
{rubric}"""


def opsd_teacher_trajectory(rec: Dict[str, Any]) -> Dict[str, Any]:
    """OPSD teacher trajectory: student's query-only prompt + rubric appended to the SYSTEM
    prompt + the SAME response. With an empty rubric the teacher degenerates to the student
    (zero distillation pull), which is the safe behaviour on rubric API failure."""
    msgs = [dict(m) for m in _skillgen_prompt(rec['problem'])['messages']]
    rubric = (rec.get('rubric') or '').strip()
    if rubric and msgs[0]['role'] == 'system':
        msgs[0]['content'] = msgs[0]['content'] + _OPSD_TEACHER_SUFFIX.format(rubric=rubric)
    return {'messages': msgs + [{'role': 'assistant', 'content': rec['response']}],
            'user_data': {'key_rounds': [len(msgs)]}}


def query_only_train_trajectory(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for v2's query-only train trajectory (view B, SFT samples, OPSD student)."""
    return _train_trajectory(rec)
