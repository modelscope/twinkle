# -*- coding: utf-8 -*-
"""E18 冷启动数据采集（KodCode 域，**无 SFT**）。

与 `e18_rejection_sft.py` 的关系：把它的采集半部分原样搬过来，删掉训练/eval/权重同步。
所以 `collect_chunk` 的逻辑、三道筛、指标口径、落盘字段全部逐字一致 —— 这样采出来的
冷启动数据集与在线 run 的样本同分布，后续接 SFT 时不需要再对齐一次。

**删掉了什么，以及为什么**
* `TransformersModel` / `set_loss` / `set_optimizer` / `train_batch`：不训练。
* `CheckpointEngineManager` / `_sync_trained_to_sampler` / `_restore_base_weights`：
  没有训练权重要推给 sampler，skill_sampler 全程是初始模型。
* `run_eval` / `EVAL_SIZE`：eval 衡量的是「训练后的 skill 模型在部署口径下的能力」，
  不训练时它恒等于 baseline，跑它纯浪费 GPU。故 `load_records(eval_size=0)`，题全进采集池。

**8 卡全部给 rollout**：原来 train 占 2 张、skill_sampler 2 张、base_sampler 4 张。
现在 train 那 2 张转给两个 sampler。base_sampler 仍拿大头（默认 6）——它是唯一瓶颈：
每 chunk 它要跑裸解 CHUNK_SIZE*BARE_ROLLOUTS + 重解 CHUNK_SIZE*N_SKILLS*EXEC_ROLLOUTS，
序列数比 skill_sampler 多一个量级，且 EXEC_MAX_TOKENS 远大于 SKILL_MAX_TOKENS。

产物（都在 OUTPUT_DIR 下，append-only）：
* `e18_sft_dataset.jsonl`：胜者，字段与在线 run 完全一致，直接可喂 SFT。
* `e18_candidates.jsonl`：**全部** skill 候选（含落选与解析失败的），靠 `kept` 区分选与未选。
  用于离线重算阀值、判断胜者是真更好还是拆平局选出来的。
* `collect_log.jsonl`：逐 chunk 指标，用于监控 accept_rate / degrade_rate 是否异常。
"""
import json
import os
import shutil
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

_HERE = os.path.dirname(os.path.abspath(__file__))
_COOKBOOK = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_HERE, os.path.join(_COOKBOOK, 'human')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e23_rubric import build_checker, class_metrics  # noqa: E402

from e18_kodcode import (clean_text, empty_roll, extract_skill,  # noqa: E402
                         judge_seqs, load_records)
from e18_multidiag import MultiDiagCache, multidiag_metrics  # noqa: E402
from e18_prompts import (direct_prompt, format_trajectory,  # noqa: E402
                         skill_solve_prompt, skillgen_prompt)
from e18_select import gain_stats, select_winner  # noqa: E402

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3-4B')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(_HERE, 'output.e18.kod'))

# ⭐ 8 卡全给 rollout：不训练，所以没有 train 组。base_sampler 拿大头（瓶颈见文件头注释）。
SKILL_SAMPLER_GPUS = int(os.environ.get('SKILL_SAMPLER_GPUS', 2))
BASE_SAMPLER_GPUS = int(os.environ.get('BASE_SAMPLER_GPUS', 6))
NUM_GPUS = SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS
GPU_MEM = float(os.environ.get('GPU_MEM', 0.8))

SEED = int(os.environ.get('SEED', 42))
# ⭐ 续跑开关。为何默认关：开着会把旧 run 的样本接着往同一批数据里添，而跨 run 的
# 模型/prompt 可能已经变了 —— 默认开启等于默认允许污染，违反归档机制的初衷。
# KOD_RESUME=1 时做三件事（见 archive_output_dir / resume_done_ids）：
#   1. 归档**之前**先从 e18_candidates.jsonl 读出已跑过的 data_id；
#   2. 三个 jsonl 复制回新目录（而不是只搬 broken_tasks 缓存），让计数接着累积；
#   3. 把已采题从题池里 filter 掉。
# 为何必须有它：DataLoader 用固定 SEED+shuffle，重启后取题顺序逐字相同，不过滤就会
# 把前 N 个 chunk 的题原样重跑一遂（实测 65 chunk ≈ 11 小时 8 卡），纯浪费。
KOD_RESUME = os.environ.get('KOD_RESUME', '0') == '1'
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 64))
# ⭐ 4 而不是 8：skill 候选池只用来「邀出」候选，最终只有 1 条胜者入池。实测 66% 的题
# 8 个候选的 with_pass 全部并列，多出来的 4 条几乎不改变胜者，却要占掉一半 executor 序列。
# ⭐ N_SKILLS 现在是**两阶段的总上限**：先生成 N_SKILLS_STAGE1 个，全部不达标才补到 N_SKILLS。
# 实测（4224 题）：39.7% 的题前 2 个候选就能出胜者 -> 平均只花 3.21 个候选，
# 产出与固定 4 个**完全相同**（1134 题），executor 序列从 32 降到 25.6。
N_SKILLS = int(os.environ.get('N_SKILLS', 4))
N_SKILLS_STAGE1 = int(os.environ.get('N_SKILLS_STAGE1', 2))
# ⭐ 天花板短路：base_pass_rate 已打满的题直接跳过，不生成任何 skill 候选。
# 依据：实测 1750 道 base=1.0 的题，入池 **0 条** —— 因为门槛是
# pass_gain >= MIN_PASS_GAIN(0.25)，而 base=1.0 时 with_pass 最大也是 1.0，gain 恒为 0。
# 所以这 41% 的题在**本离线采集脚本**里是纯浪费（占 37% 算力、零产出）。
# ⚠️ 与 collect_chunk 原注释「全量 rollout」的设计意图相反：那条理由（避免 skill 模型
# 只学会救难题）适用于**在线 RL**（每步需要 reward 信号，包括平局组）；
# 本脚本是纯离线采集，产物只有 e18_sft_dataset.jsonl，天花板题从不进该文件。
# 若日后要拿这份代码回到在线 RL，必须把本开关置 0。
SKIP_CEILING = int(os.environ.get('SKIP_CEILING', 1))
# ⭐ 粗筛 rollout 次数：先给每个候选跑 PROBE_ROLLOUTS 次，只对**并列最高**者补到
# EXEC_ROLLOUTS 次。实测（1515 题）：M=2 的 top-1 与 8 次一致率 68.9%、M=4 为 77.1%；
# 而 51% 的题 4 候选 with_pass 全同分，前 2 次就能看出打平并提前停手。
# 0 = 关闭粗筛（所有候选直接跑足 EXEC_ROLLOUTS 次，与历史 run 逐字一致）。
PROBE_ROLLOUTS = int(os.environ.get('PROBE_ROLLOUTS', 2))
# 采够多少条胜者就停。0 = 把题池跑完。
TARGET_SAMPLES = int(os.environ.get('TARGET_SAMPLES', 20000))
MAX_CHUNKS = int(os.environ.get('MAX_CHUNKS', 0))          # 0 = 不限

# ⭐ 多机并行分片。SHARD_ID 取值 0..SHARD_N-1，每台机器只跑
#   `crc32(data_id) % SHARD_N == SHARD_ID` 的题。
# 为何用哈希而不是「切片取前后一半」：
#   1. 无状态 —— 两台机器不需要任何通信/共享盘就能保证不重叠（NAS 不通用时的必要条件）
#   2. 对题池变化稳健 —— 万一两边题池大小不一致（版本/缓存差异），按下标切会错位重叠，
#      而哈希绑定在 data_id 上，永远不会
#   3. 难度分布无偏 —— crc32 与 gpt_pass_percentage 无相关，两片难度同分布
# ❗ 两台机器必须用**相同的 SHARD_N**，否则分片不构成划分（会既重叠又遗漏）。
# ❗ TARGET_SAMPLES 是**本分片的**目标：想总共 20000 就两边各填 10000。
SHARD_N = int(os.environ.get('SHARD_N', 1))
SHARD_ID = int(os.environ.get('SHARD_ID', 0))

SKILL_MAX_TOKENS = int(os.environ.get('SKILL_MAX_TOKENS', 8192))
# ⭐ 把失败轨迹（裸解时做错的那份代码）一并给 skillmodel。默认**关**：
# 开了就换了 prompt 口径，与已采的 1750 条不同源，不能默认静默切换。
# 长度安全性已实测（.tmp_analysis/len_budget.py，400 条真实数据）：
#   prompt 预算 = MAX_MODEL_LEN(16000) - SKILL_MAX_TOKENS(8192) = 7808 token
#   现状不带轨迹   中位 1802 / 最大 3192
#   带 1 条(3x 最坏) 中位 2710 / 最大 6162  -> 仍在预算内，**无需改 MAX_MODEL_LEN**
#   带 2 条(3x 最坏) 最大 9132        -> 1.5% 超预算，所以默认只给 1 条
# 超预算的后果不是报错而是 vLLM 返回空序列 -> parseable=False -> 该候选白跑，
# 难以从日志发现，所以宁可保守。
USE_TRAJ = int(os.environ.get('KOD_USE_TRAJ', 0))
TRAJ_N = int(os.environ.get('KOD_TRAJ_N', 1))
# 单条轨迹的字符上限。4000 字符 ≈ 1540 token（代码 2.6 chars/token），
# 加上现状最大 3192 仍不到 7808。超长者由 format_trajectory 头尾各留一半。
TRAJ_MAX_CHARS = int(os.environ.get('KOD_TRAJ_MAX_CHARS', 4000))
SKILL_GEN_TEMPERATURE = float(os.environ.get('SKILL_GEN_TEMPERATURE', 1.0))
SKILL_GEN_TOP_P = float(os.environ.get('SKILL_GEN_TOP_P', 1.0))
SKILL_GEN_TOP_K = int(os.environ.get('SKILL_GEN_TOP_K', -1))
EXEC_MAX_TOKENS = int(os.environ.get('EXEC_MAX_TOKENS', 15000))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16000))

EXEC_ROLLOUTS = int(os.environ.get('EXEC_ROLLOUTS', 8))
# ⭐ 裸解单独用 4 次，而带 skill 重解仍是 EXEC_ROLLOUTS(8)。两侧刻意**不对称**：
# 裸解只承担「这题有没有提升空间」的粗判（rate<1 即进诊断），4 次足够；而 with_pass 要
# 在候选之间排序、还要减去 base 算增量，精度需求高得多，降它会直接动摇入池门槛的语义。
# ⚠️ 代价（已知并接受）：base_pass_rate 只有 5 档（0,.25,.5,.75,1），与 with_pass 的 9 档
# 不同分母。于是 pass_gain = with_pass - base_pass_rate 的零点变粗，base 的采样标准误从
# 0.177 升到 0.25 —— 判「有没有空间」够用，但别再把 pass_gain 的绝对值当精密量看。
# 另外 n_ceiling（base 打满 4/4）会比 8/8 更容易达成，天花板题占比会上升。
BARE_ROLLOUTS = int(os.environ.get('BARE_ROLLOUTS', 4))
EXEC_TEMPERATURE = float(os.environ.get('EXEC_TEMPERATURE', 0.6))
EXEC_TOP_P = float(os.environ.get('EXEC_TOP_P', 0.95))

SKILL_CHAR_LIMIT = int(os.environ.get('SKILL_CHAR_LIMIT', 1500))
# 门槛仍按 EXEC_ROLLOUTS(8) 算 = +2/8 = +0.25，与历史 run 逐字可比。
# 不能拿 BARE_ROLLOUTS 做分母：with_pass 是 8 次采的，两边分母必须是同一个。
MIN_GAIN_ROLLOUTS = int(os.environ.get('MIN_GAIN_ROLLOUTS', 2))
MIN_PASS_GAIN = MIN_GAIN_ROLLOUTS / max(1, EXEC_ROLLOUTS)

# ⭐ 多机并行时 RUN_ID 必须带机器标识：原来只有时间戳，两台机器同一秒启动会撞成
# 同一个 run，合并后就再也分不出某条数据是哪台机器产的（排查单机异常时必须能分开）。
# 分片号是天然的机器标识，比 hostname 稳（容器重建后 hostname 会变），所以
# SHARD_N>1 时后缀 `.sN` —— 单机跑时 RUN_ID 保持原格式，历史 run 的比对不受影响。
RUN_ID = time.strftime('%m%d-%H%M%S') + (f'.s{SHARD_ID}' if SHARD_N > 1 else '')


@dataclass
class Runtime:
    skill_sampler: Any
    base_sampler: Any
    checker: Any
    rubric_cache: MultiDiagCache


# ===========================================================================
# 采样工具（与 e18_rejection_sft 逐字一致）
# ===========================================================================
def run_samples(sampler, prompts, num_samples, max_tokens, gen_dp,
                temperature=None, top_p=None, top_k=None, logprobs=None):
    """采样。prompts 少于 dp 时补齐再截回 —— Ray 的 dp 切分要求每个 rank 至少一条。

    ⭐ 与 e18_rejection_sft.run_samples 逐字一致。三处踩过的坑：
    1. 字段名是 `num_samples` 不是 `n`（SamplingParams 没有 `n`，传了直接 TypeError）。
    2. 走 `sampler.sample(prompts, params)`，不是 pack_user_data + generate_sequences。
    3. dp 补齐不能省：最后一个 chunk 不满、或 flat 很少时，条数 < dp 会直接报错。
    """
    if not prompts:
        return []
    import copy
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6 if temperature is None else temperature,
        top_p=0.95 if top_p is None else top_p,
        num_samples=num_samples,
        **({} if top_k is None else {'top_k': top_k}),
        **({} if logprobs is None else {'logprobs': logprobs}))
    padded = prompts
    if gen_dp > 1 and 0 < len(prompts) < gen_dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(gen_dp - len(prompts))]
    responses = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in responses]


def first_seq(seqs):
    return seqs[0] if seqs else None


def seq_text(seq) -> str:
    return clean_text(getattr(seq, 'decoded', '') or '') if seq is not None else ''


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


# ===========================================================================
# 采集：裸解 -> 诊断 -> skill-gen -> executor 重解 -> 三道筛
# ===========================================================================
def _pass_rate(rolls) -> float:
    return (sum(1.0 for x in rolls if x['correct']) / len(rolls)) if rolls else 0.0


def bare_solve(rt: Runtime, records, rollouts: int = None) -> List[List[Dict[str, Any]]]:
    """裸题重解 `rollouts` 次，每条记录返回一个 roll 列表（长度 = 实际采到的序列数）。

    返回**嵌套**列表而不是单个 roll：调用方靠 _pass_rate() 取连续值。
    判分全部汇到一次 judge_seqs（它内部按 (task_id, code) 去重，相同代码只跑一次单测）。

    ⭐ M==1 时必须降到 temperature=0.0（与原版一致）：单次采样就不需要多样性，
    带温度只会引入无意义的方差；max(1, ...) 防 rollouts 传 0 时采不到任何序列。
    """
    M = max(1, rollouts if rollouts is not None else EXEC_ROLLOUTS)
    out = run_samples(rt.base_sampler, [direct_prompt(r['problem']) for r in records],
                      M, EXEC_MAX_TOKENS, BASE_SAMPLER_GPUS,
                      temperature=(0.0 if M == 1 else EXEC_TEMPERATURE),
                      top_p=(None if M == 1 else EXEC_TOP_P))
    pairs, spans = [], []
    for r, seqs in zip(records, out):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, r['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs)
    rolls, i = [], 0
    for n in spans:
        rolls.append(judged[i:i + n] if n else [empty_roll()])
        i += n
    return rolls


# ===========================================================================
# 采集（与 e18_rejection_sft.collect_chunk 逐字一致）
# ===========================================================================
def _pick_trajectory(rolls: List[Dict[str, Any]]) -> str:
    """从裸解的 rolls 里挑出最值得给 skill-gen 看的失败代码。

    ⭐ 挑选而不是全给：BARE_ROLLOUTS=4 条里常有 2-3 条是**同一个错**
    （judge_seqs 内部按 (task_id, code) 去重就是因为重复普遍），全给只会重复占预算。

    优先级：有报错信息 > 代码短。
    为何偏好**短**代码：长代码往往是思维链泄到正文里的 no_code / import_or_syntax
    废文，信息密度低；短而完整的错解才能看出逻辑问题。同时也直接压低了长度风险。
    """
    bad = [x for x in (rolls or []) if not x.get('correct') and (x.get('code') or '').strip()]
    if not bad:
        return ''
    bad.sort(key=lambda x: (0 if x.get('error') else 1, len(x.get('code') or '')))
    blocks = [format_trajectory(x.get('code'), x.get('error'), x.get('kind'),
                               max_chars=TRAJ_MAX_CHARS)
              for x in bad[:max(1, TRAJ_N)]]
    return '\n\n'.join(blocks)


def _gen_candidates(rt: Runtime, todo, n_skills: int) -> List[List[Dict[str, Any]]]:
    """给 todo 里每道题生成 n_skills 个 skill 候选（只生成，不判分）。

    todo 元素 = (record, rubric, base_rate, trajectory)；trajectory 在 USE_TRAJ=0 时恒为 ''。
    """
    sg = run_samples(rt.skill_sampler,
                     [skillgen_prompt(r['problem'], d, eval=False, trajectory=tj)
                      for r, d, _br, tj in todo],
                     n_skills, SKILL_MAX_TOKENS, SKILL_SAMPLER_GPUS,
                     temperature=SKILL_GEN_TEMPERATURE, top_p=SKILL_GEN_TOP_P,
                     top_k=SKILL_GEN_TOP_K)
    out = []
    for seqs in sg:
        cands = []
        for s in seqs or []:
            resp = seq_text(s)
            block = extract_skill(resp)
            cands.append({'skills': block, 'response': resp, 'parseable': bool(block),
                          'with_pass': None, 'kept': False,
                          'skillgen_stop': getattr(s, 'stop_reason', None)})
        out.append(cands)
    return out


def _judge_candidates(rt: Runtime, flat, rollouts: int) -> None:
    """对 flat=[(record, cand), ...] 跑 `rollouts` 次重解并原地回写 with_pass。

    ⭐ 原地累加而非覆盖：两阶段粗筛里同一个候选会被判两次（先 PROBE 后补足），
    第二次必须把两次的样本**合并**算 pass_rate，否则前 PROBE_ROLLOUTS 次白扔。
    用 _n_correct/_n_total 累计，with_pass 每次由累计值重算。
    """
    if not flat:
        return
    ws = run_samples(rt.base_sampler,
                     [skill_solve_prompt(r['problem'], c['skills']) for r, c in flat],
                     rollouts, EXEC_MAX_TOKENS, BASE_SAMPLER_GPUS,
                     temperature=EXEC_TEMPERATURE, top_p=EXEC_TOP_P)
    pairs, spans = [], []
    for (r, _c), seqs in zip(flat, ws):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, r['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs)
    i = 0
    for (_r, c), n in zip(flat, spans):
        rr = judged[i:i + n]
        i += n
        c['_n_correct'] = c.get('_n_correct', 0) + sum(1 for x in rr if x['correct'])
        c['_n_total'] = c.get('_n_total', 0) + n
        c['with_pass'] = (c['_n_correct'] / c['_n_total']) if c['_n_total'] else 0.0
        c['n_rollouts'] = c['_n_total']
        if c.get('roll_kind') in (None, 'pass'):
            c['roll_kind'] = (next((x['kind'] for x in rr if not x['correct']),
                                   c.get('roll_kind') or 'pass') if rr else 'empty')


def _resolve_with_probe(rt: Runtime, todo, per_task, base_rates) -> None:
    """两阶段 rollout：先粗筛 PROBE_ROLLOUTS 次，只把**可能是胜者**的候选补到 EXEC_ROLLOUTS。

    补足的判据是「粗筛并列最高」而不是「粗筛最高」：粗筛只有 2 次采样，
    并列极其常见（实测 51% 的题全同分），只补单个最高者会把真胜者漏掉。
    ⚠️ 只有 parseable 的候选参与；不可解析的候选 with_pass 保持 None，与历史行为一致。
    """
    alive = [(r, c) for (r, _d, _br, _tj), cands in zip(todo, per_task)
             for c in cands if c.get('parseable')]
    if not alive:
        return
    if not PROBE_ROLLOUTS or PROBE_ROLLOUTS >= EXEC_ROLLOUTS:
        _judge_candidates(rt, alive, EXEC_ROLLOUTS)
        return
    _judge_candidates(rt, alive, PROBE_ROLLOUTS)
    # 逐题挑「粗筛并列最高」者补足到 EXEC_ROLLOUTS
    need = []
    for (r, _d, base_rate, _tj), cands in zip(todo, per_task):
        ok = [c for c in cands if c.get('parseable')]
        if not ok:
            continue
        top = max(c['with_pass'] for c in ok)
        # 粗筛就已经够不到门槛的题：补足也不可能入池，直接省掉。
        if top + 1e-9 < base_rate + MIN_PASS_GAIN - (1.0 / max(1, EXEC_ROLLOUTS)):
            continue
        need.extend((r, c) for c in ok if c['with_pass'] >= top - 1e-9)
    _judge_candidates(rt, need, EXEC_ROLLOUTS - PROBE_ROLLOUTS)


def collect_chunk(rt: Runtime, chunk, ci: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """一个 chunk 的采集，返回 (胜者列表, 指标)。

    三层省算力（每层只放行需要下一层的题），全部可用环境变量关掉回到历史行为：
      1. SKIP_CEILING : base 打满的题不生成候选（实测入池 0 条，纯浪费）
      2. N_SKILLS_STAGE1: 先 2 个候选，全不达标才补到 N_SKILLS
      3. PROBE_ROLLOUTS: 每候选先 2 次，只对并列最高者补到 EXEC_ROLLOUTS

    诊断（rubric）只对**做错的题**拉，且**每一种失败模式各诊一次后合并**（见 e18_multidiag）。
    无诊断的题不丢弃：skillgen_prompt 内部会填兜底文案。
    """
    base_rolls = bare_solve(rt, chunk, rollouts=BARE_ROLLOUTS)
    base_rates = [_pass_rate(rr) for rr in base_rolls]
    base_acc = _mean(base_rates)
    wrong = [(r, rr) for r, rr, rate in zip(chunk, base_rolls, base_rates) if rate < 1.0]

    before = rt.rubric_cache.stats.copy()
    diags = rt.rubric_cache.diagnose_many(rt.checker, wrong)
    rmetrics = multidiag_metrics(rt.rubric_cache.stats - before)
    diag_by_id = {id(r): d for (r, _rr), d in zip(wrong, diags) if d}
    n_rubric_missing = len(wrong) - len(diag_by_id)
    n_multi = sum(1 for d in diag_by_id.values() if d.count('FAILURE ') > 1)

    # ⭐ todo 元素是四元组 (record, rubric, base_rate, trajectory)。
    # trajectory 只在 USE_TRAJ=1 时非空；关闭时恒为 '' -> skillgen_prompt 走原模板，
    # 与历史 run 逐字一致。轨迹取自**裸解**的 rolls（就是当初拿去要诊断的那批），
    # 所以与 rubric 同源、描述的是同一次失败 —— 这正是 E22 当时做不到的（那次是重采的）。
    traj_by_id = ({id(r): _pick_trajectory(rr) for r, rr in zip(chunk, base_rolls)}
                  if USE_TRAJ else {})
    all_tasks = [(r, diag_by_id.get(id(r), ''), rate, traj_by_id.get(id(r), ''))
                 for r, rate in zip(chunk, base_rates)]
    # ⭐ 天花板短路。n_skipped 单独记账，指标分母用 todo（非天花板题）——
    # 于是 baseline_accuracy / candidate_pass_rate / lift 三个指标的口径变了，
    # **与 SKIP_CEILING=0 的历史 run 不可直接比较**，看趋势时注意这一点。
    if SKIP_CEILING:
        todo = [t for t in all_tasks if t[2] + MIN_PASS_GAIN <= 1.0 + 1e-9]
    else:
        todo = all_tasks
    n_skipped = len(all_tasks) - len(todo)

    per_task = _gen_candidates(rt, todo, min(N_SKILLS_STAGE1, N_SKILLS)) if todo else []
    if per_task:
        _resolve_with_probe(rt, todo, per_task, base_rates)

    # ⭐ 阶段2：阶段1 全部不达标的题，再生成剩余候选。
    # 实测 39.7% 的题阶段1 就出胜者 -> 这批题省掉一半候选；6.1% 的题靠阶段2 救回。
    n_stage2_tasks = n_stage2_saved = 0
    remain = N_SKILLS - min(N_SKILLS_STAGE1, N_SKILLS)
    if per_task and remain > 0:
        idx2 = [i for i, ((_r, _d, br, _tj), cands) in enumerate(zip(todo, per_task))
                if not any(c.get('parseable')
                           and (c.get('with_pass') or 0.0) >= br + MIN_PASS_GAIN - 1e-9
                           for c in cands)]
        if idx2:
            n_stage2_tasks = len(idx2)
            todo2 = [todo[i] for i in idx2]
            extra = _gen_candidates(rt, todo2, remain)
            per_task2 = [[] for _ in todo]
            for i, cands in zip(idx2, extra):
                per_task2[i] = cands
            _resolve_with_probe(rt, todo, per_task2, base_rates)
            for i, cands in zip(idx2, extra):
                per_task[i].extend(cands)
            n_stage2_saved = sum(
                1 for i, cands in zip(idx2, extra)
                if any(c.get('parseable')
                       and (c.get('with_pass') or 0.0) >= todo[i][2] + MIN_PASS_GAIN - 1e-9
                       for c in cands))

    accepted, sims = [], []
    n_pass_cands = n_survivors = 0
    n_acc_hard = n_acc_easy = 0
    gtot = {'improved': 0, 'tied': 0, 'degraded': 0}
    gains = []
    for (r, d, base_rate, _tj), cands in zip(todo, per_task):
        passers = [c for c in cands
                   if c.get('parseable') and (c.get('with_pass') or 0) >= base_rate]
        n_pass_cands += len(passers)
        for k, v in gain_stats(cands, base_rate).items():
            gtot[k] += v
        best = select_winner(cands, d, r['reference_answer'],
                             skill_char_limit=SKILL_CHAR_LIMIT,
                             base_pass_rate=base_rate, min_pass_gain=MIN_PASS_GAIN)
        if best is None:
            continue
        n_survivors += 1
        sims.append(best['rubric_similarity'])
        gains.append(best['pass_gain'])
        if base_rate >= 1.0:
            n_acc_easy += 1
        else:
            n_acc_hard += 1
        accepted.append({
            'problem': r['problem'], 'reference_answer': r['reference_answer'],
            'data_id': r.get('data_id', ''), 'skills': best['skills'],
            'response': f"<skills>\n{best['skills']}\n</skills>",
            'base_pass_rate': base_rate, 'with_pass_rate': best['with_pass'],
            'pass_gain': best['pass_gain'], 'gain_kind': best['gain_kind'],
            'rubric': d, 'chunk': ci, 'run': RUN_ID,
            'rubric_similarity': best['rubric_similarity'],
            'skill_chars': len(best['skills']),
            'n_candidates_passed': len(passers)})

    # ⭐ 分母改成**实际生成的候选总数**：两阶段下每道题的候选数不再是定值 N_SKILLS
    # （阶段1 就达标的题只有 N_SKILLS_STAGE1 个），写死 N_SKILLS 会把分母虚抬、
    # 让 candidate_pass_rate 和 lift 系统性偏低。
    n_cands_total = sum(len(cs) for cs in per_task)
    _cand_pass = (n_pass_cands / n_cands_total) if n_cands_total else 0.0
    # 必须在选择循环之后：kept / with_pass / rubric_similarity 都是 select_winner 原地回写的。
    dump_candidates(todo, per_task, ci)
    metrics = {
        'train/baseline_accuracy': base_acc,
        'train/accept_rate': (len(accepted) / len(todo)) if todo else 0.0,
        'train/candidate_pass_rate': _cand_pass,
        'train/lift': _cand_pass - base_acc,
        'train/selected_rubric_similarity': _mean(sims),
        'train/selected_skill_length_characters': _mean(
            [float(s['skill_chars']) for s in accepted]),
        'signal/n_wrong': float(len(wrong)),
        'signal/n_rubric_missing': float(n_rubric_missing),
        'signal/n_multi_cause': float(n_multi),
        'signal/n_accepted': float(len(accepted)),
        'signal/n_accepted_hard': float(n_acc_hard),
        'signal/n_accepted_easy': float(n_acc_easy),
        'signal/n_ceiling': float(sum(1 for _r, _d, br, _tj in todo
                                      if br + MIN_PASS_GAIN > 1.0 + 1e-9)),
        'signal/min_pass_gain': MIN_PASS_GAIN,
        # 三层省算力的实测记账（用于事后核对真省了多少，而不是只信离线模拟）
        'saving/n_ceiling_skipped': float(n_skipped),
        'saving/n_stage2_tasks': float(n_stage2_tasks),
        'saving/n_stage2_rescued': float(n_stage2_saved),
        'saving/candidates_per_task': (n_cands_total / len(todo)) if todo else 0.0,
        'saving/exec_sequences': float(sum(
            c.get('_n_total', 0) for cs in per_task for c in cs)),
        # ⭐ 轨迹注入的实测记账。traj/rate 远低于 1 就说明很多题拿不到可用失败代码
        # （全对、或全是 no_code 空代码），此时该开关的实际覆盖面比以为的小。
        # traj/chars 盯长度风险：除以 2.6 就是多吃的 token 数。
        'traj/enabled': float(USE_TRAJ),
        'traj/rate': (sum(1 for _r, _d, _br, tj in todo if tj) / len(todo)) if todo else 0.0,
        'traj/chars': _mean([float(len(tj)) for _r, _d, _br, tj in todo if tj]),
        'gain/improved_candidates': float(gtot['improved']),
        'gain/tied_candidates': float(gtot['tied']),
        'gain/degraded_candidates': float(gtot['degraded']),
        'gain/degrade_rate': (gtot['degraded'] / max(1, sum(gtot.values()))),
        'gain/improve_rate': (gtot['improved'] / max(1, sum(gtot.values()))),
        'gain/selected_pass_gain': _mean(gains),
    } | rmetrics | class_metrics([d for _r, d, _br, _tj in todo if d])
    return accepted, metrics


def dump_dataset(accepted) -> None:
    """胜者落盘 append-only 的 SFT 数据集（字段与在线 run 完全一致）。"""
    if not accepted:
        return
    path = os.path.join(OUTPUT_DIR, 'e18_sft_dataset.jsonl')
    with open(path, 'a', encoding='utf-8') as f:
        for s in accepted:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


def dump_candidates(todo, per_task, ci: int) -> None:
    """**全部** skill 候选落盘（选上的、没选上的、甚至解不出 <skills> 的），append-only。

    为何必需：`e18_sft_dataset.jsonl` 只存胜者，而拒绝采样的全部信息量在「同一题的 N 条
    候选之间的差异」里 —— 丢掉落选者就无法回答：胜者是真的更好，还是只是采样噪声
    （候选全部并列时靠拆平局选出来的）？也无法事后重算阀值：改 MIN_PASS_GAIN /
    SKILL_CHAR_LIMIT 后想知道会多收多少条，必须有落选者的 with_pass 才能离线重放。

    `kept` 区分选与未选：`select_winner` 对胜者**原地** 置 True（e18_select.py:107），
    本函数必须在 select_winner 之后调用，否则全部候选都是 False。
    同理 `with_pass` / `pass_gain` / `rubric_similarity` 也是选择阶段回写的。

    不存 `response` 全文（包含 think 链，体量是 skills 的十倍量级），只存抽取后的
    skills 与长度/截断信息；解析失败（parseable=False）时 skills 为空串，靠 skillgen_stop
    判断是撞预算截断还是真的不守格式。
    """
    path = os.path.join(OUTPUT_DIR, 'e18_candidates.jsonl')
    with open(path, 'a', encoding='utf-8') as f:
        for (r, d, base_rate, _tj), cands in zip(todo, per_task):
            for j, c in enumerate(cands):
                wp = c.get('with_pass')
                f.write(json.dumps({
                    'chunk': ci, 'run': RUN_ID,
                    'data_id': r.get('data_id', ''),
                    'task_id': r['reference_answer'].get('task_id', ''),
                    'cand_idx': j,
                    'kept': bool(c.get('kept')),
                    'parseable': bool(c.get('parseable')),
                    'skills': c.get('skills', ''),
                    'skill_chars': len(c.get('skills') or ''),
                    'skillgen_stop': c.get('skillgen_stop'),
                    'base_pass_rate': base_rate,
                    'with_pass_rate': wp,
                    # 未参与重解（解析失败）时 with_pass 是 None，pass_gain 也留 None，
                    # 不能当 0 存 —— 否则离线统计会把「没跑」混成「跑了但零增益」。
                    'pass_gain': (None if wp is None else round(wp - base_rate, 6)),
                    'gain_kind': c.get('gain_kind'),
                    'n_rollouts': c.get('n_rollouts'),
                    'roll_kind': c.get('roll_kind'),
                    'rubric_similarity': c.get('rubric_similarity'),
                    'rubric': d,
                }, ensure_ascii=False) + '\n')


# ===========================================================================
# main
# ===========================================================================
def build_runtime(checker, rubric_cache) -> Runtime:
    """两组卡：skill_sampler / base_sampler(executor)。不训练，所以没有 train 组。"""
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='skill_sampler', ranks=list(range(0, SKILL_SAMPLER_GPUS)),
                    device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(SKILL_SAMPLER_GPUS, NUM_GPUS)),
                    device_type='GPU')])

    def _sampler(group, world, enable_thinking):
        s = vLLMSampler(model_id=MODEL_ID,
                        engine_args={'gpu_memory_utilization': GPU_MEM,
                                     'max_model_len': MAX_MODEL_LEN,
                                     'tensor_parallel_size': 1},
                        device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
                        remote_group=group)
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=enable_thinking,
                       max_length=MAX_MODEL_LEN)
        return s

    # 与在线 run 同口径：skill_sampler 开 think（采集要多样性），executor 也开 think。
    skill_sampler = _sampler('skill_sampler', SKILL_SAMPLER_GPUS, enable_thinking=True)
    base_sampler = _sampler('base_sampler', BASE_SAMPLER_GPUS, enable_thinking=True)
    return Runtime(skill_sampler=skill_sampler, base_sampler=base_sampler,
                   checker=checker, rubric_cache=rubric_cache)


def count_existing_samples() -> int:
    """续跑：已有胜者条数（= e18_sft_dataset.jsonl 的有效行数）。同样要在归档前调。

    这里才该用 sft_dataset 而不是 candidates：它要回答的是「已经凑了多少条胜者」，
    用来接着比 TARGET_SAMPLES；而 resume_done_ids() 要回答的是「哪些题不用再跑」。
    两个问题不同源，切勿合并。
    """
    path = os.path.join(OUTPUT_DIR, 'e18_sft_dataset.jsonl')
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def resume_done_ids() -> set:
    """续跑：读出已经跑过的 data_id。必须在 archive_output_dir() **之前**调用。

    ⭐ 读 `e18_candidates.jsonl` 而不是 `e18_sft_dataset.jsonl`：后者只有**胜者**（实测
    accept_rate 约 22%），拿它去重会把剩下 78%「跑过但没能入池」的题当成未跑、
    下次重新烧一遍 GPU。候选文件是全量落盘的，覆盖面才完整。

    容错：进程被 kill 时最后一行可能是写一半的残行，json.loads 会抛异常 ->
    逐行 try 跳过（丢掉一道题的去重信息只是多跑一题，而整个函数抛异常会直接弄挂启动）。
    """
    path = os.path.join(OUTPUT_DIR, 'e18_candidates.jsonl')
    if not os.path.exists(path):
        return set()
    done = set()
    bad = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)['data_id']
            except Exception:
                bad += 1
                continue
            if d:
                done.add(str(d))
    logger.info(f'[resume] 已跑过 {len(done)} 题'
                + (f'（跳过 {bad} 行残缺/半行）' if bad else ''))
    return done


def archive_output_dir(carry_data: bool = False) -> None:
    """启动时把已存在的 OUTPUT_DIR 整个 mv 走，保证本 run 写入空目录。

    为何必需：`e18_sft_dataset.jsonl` / `collect_log.jsonl` 都是 `open(..., 'a')` 追写。
    没有这一步时，重启一次就把新旧 run 的样本焊在同一个文件里，而且不报错。
    用 mv 而不是删：旧 run 的样本是可复用的分析素材。
    沙箱自检缓存（kod_broken_tasks.json）会搬回新目录 —— 那是纯函数结果，重跑一次很贵。

    carry_data=True（续跑）时额外把三个产物 jsonl 也复制回新目录，于是新 run 接着往后面
    追写、总数连续。用 copy2 而不是 move：归档副本保留完整快照，万一续跑又崩了还能回溯。
    注意：续跑后同一份数据里会存在多个 `run` 值，离线分析要按 run 字段分组看。
    """
    if not os.path.isdir(OUTPUT_DIR) or not os.listdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return
    stamp = time.strftime('%m%d-%H%M%S', time.localtime(os.path.getmtime(OUTPUT_DIR)))
    dst = f'{OUTPUT_DIR}.bak-{stamp}'
    i = 1
    while os.path.exists(dst):
        dst = f'{OUTPUT_DIR}.bak-{stamp}-{i}'
        i += 1
    shutil.move(OUTPUT_DIR, dst)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f'[init] 旧输出目录已归档 -> {dst}')
    cache = os.path.join(dst, 'kod_broken_tasks.json')
    if os.path.exists(cache):
        shutil.copy2(cache, os.path.join(OUTPUT_DIR, 'kod_broken_tasks.json'))
        logger.info('[init] 沙箱自检缓存已搬回新目录（避免重跑）')
    if carry_data:
        for name in ('e18_sft_dataset.jsonl', 'e18_candidates.jsonl', 'collect_log.jsonl'):
            src = os.path.join(dst, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUTPUT_DIR, name))
        logger.info('[resume] 旧产物已复制回新目录，本 run 接着追写')


def main():
    t_start = time.time()
    # ⭐ 顺序强制：读已采 id 必须在归档**之前**，否则 OUTPUT_DIR 已经被 mv 走、读到空集。
    done_ids = resume_done_ids() if KOD_RESUME else set()
    n_done = count_existing_samples() if KOD_RESUME else 0
    archive_output_dir(carry_data=KOD_RESUME)
    checker = build_checker()
    # eval_size=0：不训练就没有 eval 的意义，题全进采集池。
    train_dataset, _ = load_records(SEED, 0, OUTPUT_DIR)
    # ⭐ 分片必须在 resume 过滤**之前**做，且用 crc32(data_id) 而不是下标：
    # 两台机器无共享存储，只能靠纯函数保证不重叠。zlib.crc32 跨进程/跨平台稳定
    # （而 hash() 受 PYTHONHASHSEED 影响，每次启动都不同 —— 用它会造成重叠+遗漏）。
    if SHARD_N > 1:
        before = len(train_dataset)
        train_dataset.filter(
            lambda r: zlib.crc32(str(r['data_id']).encode()) % SHARD_N == SHARD_ID)
        logger.info(f'[shard] {SHARD_ID}/{SHARD_N}: 题池 {before} -> {len(train_dataset)}')
    if done_ids:
        before = len(train_dataset)
        train_dataset.filter(lambda r: r['data_id'] not in done_ids)
        logger.info(f'[resume] 题池剔除已采 {before - len(train_dataset)} 题 -> 剩 {len(train_dataset)}')
    logger.info(f'[data] 采集池 ={len(train_dataset)} 题')
    rt = build_runtime(checker, MultiDiagCache())
    logger.info(f'E18-collect start: chunk={CHUNK_SIZE} n_skills={N_SKILLS} '
                f'bare_rollouts={BARE_ROLLOUTS} exec_rollouts={EXEC_ROLLOUTS} '
                f'min_pass_gain={MIN_PASS_GAIN:.3g} '
                f'use_traj={USE_TRAJ}(n={TRAJ_N},max_chars={TRAJ_MAX_CHARS}) '
                f'shard={SHARD_ID}/{SHARD_N} '
                f'target={TARGET_SAMPLES} gpus={SKILL_SAMPLER_GPUS}+{BASE_SAMPLER_GPUS} '
                f'resume={int(KOD_RESUME)} n_done={n_done} '
                f'output={OUTPUT_DIR}')

    # ⭐ n_total 从已有数量起算，不是 0：它是 TARGET_SAMPLES 的比较对象，从 0 起算会变成
    # 「再采 TARGET_SAMPLES 条」而不是「凑到 TARGET_SAMPLES 条」。
    n_total, ci = n_done, 0
    log_path = os.path.join(OUTPUT_DIR, 'collect_log.jsonl')
    with open(log_path, 'a', encoding='utf-8') as log_fh:
        loader = DataLoader(dataset=train_dataset, batch_size=CHUNK_SIZE, num_workers=0,
                            shuffle=True, drop_last=False,
                            generator=torch.Generator().manual_seed(SEED))
        for chunk in loader:
            if TARGET_SAMPLES and n_total >= TARGET_SAMPLES:
                break
            if MAX_CHUNKS and ci >= MAX_CHUNKS:
                break
            t0 = time.time()
            accepted, metrics = collect_chunk(rt, chunk, ci)
            dump_dataset(accepted)
            n_total += len(accepted)
            row = {'chunk': ci, 'run': RUN_ID, 'seconds': round(time.time() - t0, 1),
                   'n_collected': float(n_total), **metrics}
            log_fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            log_fh.flush()
            logger.info(f'[c{ci}] collected={n_total}'
                        + (f'/{TARGET_SAMPLES}' if TARGET_SAMPLES else '')
                        + ' ' + ' '.join(f'{k}={v:.4g}' for k, v in row.items()
                                         if isinstance(v, float)))
            ci += 1
    logger.info(f'[完成] 共 {n_total} 条，{ci} 个 chunk，'
                f'耗时 {(time.time() - t_start) / 60:.1f} 分钟 -> {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
