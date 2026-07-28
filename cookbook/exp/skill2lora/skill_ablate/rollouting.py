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
_REGEN_EXAMPLE = v2.REGEN_SYSTEM.split('Example:\n', 1)[1]
assert _REGEN_EXAMPLE.startswith('<skills>') and _REGEN_EXAMPLE.rstrip().endswith('</skills>'), \
    'REGEN_SYSTEM example extraction broke; check v2.REGEN_SYSTEM formatting'

_RUBRIC_SKILLGEN_NARRATIVE = """\
You are a skill-generation model. Your <skills> block will be fed to a SEPARATE downstream executor model that must solve the problem on its own. The executor will NOT see your private reasoning or the analysis below — it only sees what is inside <skills>...</skills>.

An expert rubric analysis of a failed attempt on THIS problem is provided to you. Use it to understand where solving this type of problem tends to break down, then think privately and abstract WHAT MAKES THIS TYPE OF PROBLEM SOLVABLE into transferable methodology.

Then write the <skills> block following these rules:
- Give general, transferable solving techniques for this TYPE of problem as one coherent analysis narrative: first name what the problem is essentially asking, then walk through how to approach it, blending the key concepts, the recommended steps, the pitfalls to avoid (informed by the analysis) and a brief reason for each into a single connected story.
- CRITICAL: Do NOT solve the problem, reveal/compute the final answer, or substitute the problem's specific given numbers. Leave ALL concrete numbers for the executor to compute.
- Self-contained: write in the first person (e.g. "I think the step most likely to go wrong is ..."). NEVER reference "the analysis", "the rubric", or "the previous attempt" — the executor cannot see them, such phrasings cause hallucination.
- Keep it concise: aim for roughly one focused paragraph.

Put ONLY the methodology inside <skills></skills>.

Example:
""" + _REGEN_EXAMPLE

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


def rubric_skillgen_prompt(problem: str, rubric: str) -> Dict[str, Any]:
    """View-A skill-gen conditioned on (problem + rubric diagnosis), NO prior skill.

    Style-matched to the main line via v2's ``_SKILL_STYLE`` global (set by main() from
    ``--skill-style``). narrative -> narrative rubric prompt; pitfall -> pitfall rubric prompt.
    """
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
