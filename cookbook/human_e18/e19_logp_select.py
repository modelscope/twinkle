# -*- coding: utf-8 -*-
"""E19：能否不靠 8 次 executor rollout 就选出最好的 skill？（RLT-style logp/熵 打分）

映射自 skill_quality_analysis.md 第 18 节的「用法2（只搬 reward）」。核心问题：
现在选 skill 要给每个候选跑 8 次 executor + 判分（8 候选 = 64 次解码），太贵。
能不能用 **frozen executor 上的一次 teacher-forcing forward**（无解码）替代？

流程（与 e18 采集逐字同源，**不用 GT 造 prompt**）：
  1. 裸解 BARE_ROLLOUTS 次 -> 取**失败的** trajectory（错误代码 + 报错）
  2. 失败 traj + rubric 一起喂给 skillmodel -> 生成 N_SKILLS 个 narrative skill
  3. 每个 skill 让 executor rollout EXEC_ROLLOUTS 次 -> with_pass_rate（**这是 ground truth，
     只用于离线评估选择器，不参与任何打分特征**）
  4. 同时对每个 skill 算 cheap 特征（一次 forward，见下）-> 落盘
  5. 离线比：cheap selector 的命中率 vs 8-rollout oracle

⭐ 为什么 GT 不进 prompt：用户明确要求。RLT 的 teacher 是**开卷**的（输入含标准答案），
因此它必须靠 r_KL 压泄漏，且 teacher 不能直接部署。本实验的 skillmodel 保持**闭卷**
（输入只有题面 + 自己的失败 traj + rubric），GT 只在两个地方出现：
  (a) 判分（pytest 跑测试）—— 本来就在判分侧，不进生成器；
  (b) r_SS 的打分目标 —— 只流经 reward 计算的 forward，不进生成器上下文。
所以本实验**结构上无泄漏**，不需要 RLT 的 r_KL 来压——但仍然实现了 leak 监控项，
因为 rubric 里可能夹带答案片段（见 leak_frac）。

⭐ 打分目标选 `canonical_solution` 而不是「修对后的代码」：RLT 的 r_SS 是
logp(标准答案 | 讲解, 题)。我们没有「修对后的代码」这种东西（那要先跑通才知道），
所以用数据集自带的参考解。代价：参考解的写法风格与 executor 的自然写法不同，
logp 会偏低且带常数偏移 —— 但我们只在**同题的候选之间**比较排序，常数偏移会被抵消。

cheap 特征（全部来自一次 prompt_logprobs forward，零解码）：
  * r_ss      : mean logp(参考解 token | 题 + skill)    <- RLT 主项，越高越好
  * r_ss_min  : min-k 平均（最难的 10% token）           <- RLT 的 α·min 项
  * ppl       : exp(-r_ss)，困惑度
  * ent_*     : 参考解位置上的预测熵（需要 topk）
  * skill 自身的 logp/熵（生成时顺带拿到）
"""
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, '..', 'human'))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e18_kodcode import (clean_text, extract_code, judge_seqs,  # noqa: E402
                        load_records)
from e18_prompts import direct_prompt, skill_solve_prompt  # noqa: E402
from e18_multidiag import MultiDiagCache  # noqa: E402
from e23_rubric import build_checker  # noqa: E402

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3-4B')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(_HERE, 'output.e19.logp'))
SEED = int(os.environ.get('SEED', 42))

SKILL_GPUS = int(os.environ.get('SKILL_GPUS', 4))
EXEC_GPUS = int(os.environ.get('EXEC_GPUS', 4))
NUM_GPUS = SKILL_GPUS + EXEC_GPUS
GPU_MEM = float(os.environ.get('GPU_MEM', 0.85))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 24000))

N_TASKS = int(os.environ.get('N_TASKS', 24))        # 「几组错误的」——先小规模看效果
N_SKILLS = int(os.environ.get('N_SKILLS', 8))       # 每题 8 个 skill 候选
BARE_ROLLOUTS = int(os.environ.get('BARE_ROLLOUTS', 4))
EXEC_ROLLOUTS = int(os.environ.get('EXEC_ROLLOUTS', 8))   # ground truth 用
SKILL_MAX_TOKENS = int(os.environ.get('SKILL_MAX_TOKENS', 8192))
EXEC_MAX_TOKENS = int(os.environ.get('EXEC_MAX_TOKENS', 15000))
EXEC_TEMPERATURE = float(os.environ.get('EXEC_TEMPERATURE', 0.6))
SKILL_TEMPERATURE = float(os.environ.get('SKILL_TEMPERATURE', 1.0))
TOPK = int(os.environ.get('TOPK', 20))              # 算熵用的 top-k
RUN_ID = time.strftime('%m%d-%H%M%S')


# ===========================================================================
# prompt：失败 traj + rubric -> narrative skill（闭卷，无 GT）
# ===========================================================================
SKILLGEN_SYSTEM = (
    'You are helping a Python programmer who is about to attempt a coding task. '
    'You have seen this programmer fail this exact task before, and you have a '
    'diagnosis of what went wrong.\n\n'
    'Write a short piece of guidance (a "skill") that would have prevented that '
    'failure. Requirements:\n'
    '- Write flowing prose, not bullet points or headings.\n'
    '- Name the concrete API, argument, keyword, or edge case involved.\n'
    '- Refer to the past failure as something that already happened, and say what '
    'to do instead.\n'
    '- Do NOT include any code block, and do NOT write a full solution.\n'
    '- Keep it under 200 words.\n'
    'Wrap your guidance in <skills> and </skills> tags.')


def skillgen_prompt(problem: str, failed_code: str, error: str, rubric: str) -> Dict[str, Any]:
    """闭卷 skill 生成 prompt：题面 + 自己的失败代码 + 报错 + rubric 诊断。

    ⭐ 这里**没有** canonical_solution / reference answer。这是与 RLT teacher 的关键区别：
    RLT 开卷（输入含答案）所以必须用 r_KL 压泄漏；我们闭卷，泄漏在结构上不可能发生。
    """
    user = (f'Task the programmer was given:\n{problem}\n\n'
            f'The code they wrote (it failed):\n```python\n{failed_code}\n```\n\n'
            f'How it failed:\n{error}\n\n'
            f'Diagnosis:\n{rubric}\n\n'
            'Write the guidance that would have prevented this failure.')
    return {'messages': [{'role': 'system', 'content': SKILLGEN_SYSTEM},
                         {'role': 'user', 'content': user}]}


# ===========================================================================
# 采样 / 判分 工具（与 e18 同源）
# ===========================================================================
def run_samples(sampler, prompts, num_samples, max_tokens, gen_dp,
                temperature=None, top_p=None, logprobs=None):
    if not prompts:
        return []
    import copy
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6 if temperature is None else temperature,
        top_p=0.95 if top_p is None else top_p,
        num_samples=num_samples,
        **({} if logprobs is None else {'logprobs': logprobs}))
    padded = prompts
    if gen_dp > 1 and 0 < len(prompts) < gen_dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(gen_dp - len(prompts))]
    resp = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in resp]


def seq_text(seq) -> str:
    return clean_text(getattr(seq, 'decoded', '') or '') if seq is not None else ''


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def extract_skill(text: str) -> str:
    if '<skills>' not in text:
        return ''
    body = text.split('<skills>', 1)[1]
    return body.split('</skills>', 1)[0].strip() if '</skills>' in body else ''


# ===========================================================================
# ⭐ 核心：teacher-forcing 打分（一次 forward，零解码）
# ===========================================================================
def score_teacher_forcing(exec_sampler, problem: str, skill: str, target_code: str,
                          gen_dp: int) -> Dict[str, float]:
    """算 logp(target_code | 题 + skill) —— RLT 的 r_SS，用 prompt_logprobs 实现。

    ⭐ 机制：把「题+skill」当 prompt、把 target_code **拼在 prompt 末尾**，然后
    `max_tokens=1` + `prompt_logprobs=TOPK` 采样。vLLM 会回传**每个 prompt token 的
    logprob**（vllm_engine.py:324-343 取的是实际 token 的 logprob），于是我们免费拿到
    了 target_code 每个 token 在该上下文下的条件概率 —— 这就是 teacher forcing，
    且**不解码任何 token**，比 8 次 rollout 便宜两个量级。

    ⭐ 必须用 continue_final_message 让模板把 target_code 编成 **assistant 内容**而不是
    新一轮 user：编错角色会导致 logp 分布完全不同（模型在算「用户会说这段代码的概率」）。
    这里靠传入 assistant 角色的消息 + 模板拼接实现。

    ⭐ 只取 target 段的 logprob，必须切掉 prompt 段。切点靠**两次编码求长度差**确定：
    先编「题+skill」得 n_ctx，再编「题+skill+target」得 n_all，则 target 占
    [n_ctx, n_all)。不能用固定偏移或字符串查找 —— tokenizer 会在边界合并 token。
    """
    return {}


def _entropy_from_topk(topk: List[Optional[List[Tuple[int, float]]]]) -> List[float]:
    """由 top-k logprob 估计每位置的熵。

    ⚠️ 这是**截断熵**（只看 top-k），不是真熵：尾部质量被忽略，所以系统性偏低。
    但我们只做同题候选间的排序比较，偏差方向一致，可用。k=TOPK=20 时通常覆盖 >90% 概率质量。
    """
    import math
    out = []
    for lps in topk or []:
        if not lps:
            out.append(0.0)
            continue
        ps = [math.exp(lp) for _, lp in lps]
        z = sum(ps) or 1.0
        out.append(-sum((p / z) * math.log(max(p / z, 1e-12)) for p in ps))
    return out
