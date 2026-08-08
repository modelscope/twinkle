#!/usr/bin/env python3
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, pack_user_data
from twinkle.dataloader import DataLoader
from twinkle.model import TransformersModel
from twinkle.patch.no_split_modules import NoSplitModulesPatch
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

# 环境层与教师 judge 复用 human/ 下的 e23 模块（不拷贝，保证判分/诊断逐字同源）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_HUMAN = os.path.abspath(os.path.join(_HERE, '..', 'human'))
if _HUMAN not in sys.path:
    sys.path.insert(0, _HUMAN)

from e23_bcb import clean_text, empty_roll, extract_skill, judge_seqs, load_records  # noqa: E402
from e23_rubric import build_checker, class_metrics  # noqa: E402
# 多轨迹诊断：每种失败模式各诊一次再合并。不用 e23_rubric.RubricCache 是因为它的缓存键
# 只含 data_id，同一题调 N 次会全部命中第一次的结果 —— 「每条 rollout 各诊一次」在那个键下做不到。
from e18_multidiag import MultiDiagCache, multidiag_metrics  # noqa: E402

from e18_prompts import direct_prompt, skill_solve_prompt, skillgen_prompt  # noqa: E402
from e18_select import gain_stats, select_winner  # noqa: E402

try:
    import swanlab
except ImportError:
    swanlab = None

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3-4B')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(_HERE, 'output.e18'))

TRAIN_GPUS = int(os.environ.get('TRAIN_GPUS', 2))
SKILL_SAMPLER_GPUS = int(os.environ.get('SKILL_SAMPLER_GPUS', 2))
# ⭐ base_sampler 拿 4 张（而不是和其他两组一样的 2）：它是唯一瓶颈。每 chunk 的序列数
# 相差一个量级 —— skill_sampler 只跑 CHUNK_SIZE*N_SKILLS（512），而 base_sampler 要跑
# 裸解 CHUNK_SIZE*EXEC_ROLLOUTS（512）+ 重解 CHUNK_SIZE*N_SKILLS*EXEC_ROLLOUTS（4096）= 4608，
# 且 EXEC_MAX_TOKENS（15000）远大于 SKILL_MAX_TOKENS（8192），token 预算相差约 16 倍。
# 给 skill_sampler 加卡几乎无收益，加在这里才能缩短 wall clock。
BASE_SAMPLER_GPUS = int(os.environ.get('BASE_SAMPLER_GPUS', 4))
# ⭐ 没有 ref 模型：SFT 是纯交叉熵，不需要 KL 参考。E23 的 REF_GPUS 那两张转给了 base_sampler，
# 而不是空着 —— 8 卡机器上「省卡」没有意义，只会让瓶颈环节白白排队。
NUM_GPUS = TRAIN_GPUS + SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS
TRAIN_FSDP = int(os.environ.get('TRAIN_FSDP', 1))
TRAIN_DP = TRAIN_GPUS // TRAIN_FSDP
GPU_MEM = float(os.environ.get('GPU_MEM', 0.8))

SEED = int(os.environ.get('SEED', 42))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 64))       # 每轮裸解多少题去筛错题
N_SKILLS = int(os.environ.get('N_SKILLS', 8))            # 每题采多少 skill 候选（拒绝采样的池）
# 攒够多少条胜者才 SFT 一次。必须是 TRAIN_DP 的整数倍（dp 切分要求），否则末尾会被丢。
ACCUMULATE = int(os.environ.get('ACCUMULATE', 16))
MAX_UPDATES = int(os.environ.get('MAX_UPDATES', 200))
EVAL_SIZE = int(os.environ.get('EVAL_SIZE', 100))
EVAL_EVERY_UPDATES = int(os.environ.get('EVAL_EVERY_UPDATES', 10))
SAVE_EVERY_UPDATES = int(os.environ.get('SAVE_EVERY_UPDATES', 50))   # 0 = 只在结束时存

# skill-gen 采集：think 开、T=1（要多样性才有拒绝采样的意义）
SKILL_MAX_TOKENS = int(os.environ.get('SKILL_MAX_TOKENS', 8192))
SKILL_GEN_TEMPERATURE = float(os.environ.get('SKILL_GEN_TEMPERATURE', 1.0))
SKILL_GEN_TOP_P = float(os.environ.get('SKILL_GEN_TOP_P', 1.0))
SKILL_GEN_TOP_K = int(os.environ.get('SKILL_GEN_TOP_K', -1))
EXEC_MAX_TOKENS = int(os.environ.get('EXEC_MAX_TOKENS', 15000))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16000))

# ⭐ pass@k 评分：executor 不再用 greedy 单次，而是每个 prompt 重解 EXEC_ROLLOUTS 次取通过率。
# 为何必须：M=1/T=0 时 pass_rate 只有 0/1，易题上「加任何 skill 都对」，正例标签与 skill
# 质量无关 —— 等于往数据集里灌随机 skill。跑 8 次取连续 pass_rate 后，同一题的不同 skill 之间
# 才有方差（如 8/8 vs 5/8），能真正排序。代价：executor GPU 时间乘 ~8 倍。
# 温度必须 >0：T=0 下 8 次采样会逐字相同（judge_seqs 还会按 code 去重），pass_rate 退回 0/1。
EXEC_ROLLOUTS = int(os.environ.get('EXEC_ROLLOUTS', 8))
EXEC_TEMPERATURE = float(os.environ.get('EXEC_TEMPERATURE', 0.6))
EXEC_TOP_P = float(os.environ.get('EXEC_TOP_P', 0.95))

LR = float(os.environ.get('LR', 1e-5))                   # 恒定 lr，无 warmup 无 decay
TRAIN_MICRO_BATCH = int(os.environ.get('TRAIN_MICRO_BATCH', max(TRAIN_DP, ACCUMULATE // 2)))

# 三道筛的两个阈值
SKILL_CHAR_LIMIT = int(os.environ.get('SKILL_CHAR_LIMIT', 1500))   # 超过直接丢

# ⭐ 入池门槛：skill 至少要多做对 MIN_GAIN_ROLLOUTS 次（默认 2，即 +2/8 = +0.25）。
# 为何不收 tie（8/8 -> 8/8）：那种样本只能证明 skill 无害，对「学会写有效 skill」没有任何
# 监督信号 —— 易题上加任何 skill 都是 8/8，收它等于往数据集里灌随机文本。
# 为何阈值是 2 而不是 1：8 次采样下 +1/8 在采样噪声量级内（二项分布标准误约 0.17），
# 分不清是真提升还是波动；要求 +2/8 才能把噪声挤出去。用 rollout 数而不是写死 0.25，
# 是为了改 EXEC_ROLLOUTS 时该语义（「多做对几次」）保持不变。
MIN_GAIN_ROLLOUTS = int(os.environ.get('MIN_GAIN_ROLLOUTS', 2))
MIN_PASS_GAIN = MIN_GAIN_ROLLOUTS / max(1, EXEC_ROLLOUTS)

SWAN_PROJ = os.environ.get('SWAN_PROJ', 'twinkle')
RUN_TAG = os.environ.get('RUN_TAG', '').strip()
RUN_ID = time.strftime('%m%d-%H%M%S')


@dataclass
class Runtime:
    skill_model: Any
    skill_sampler: Any
    base_sampler: Any
    ckpt: Any
    checker: Any
    rubric_cache: MultiDiagCache


# ===========================================================================
# 采样 / 小工具
# ===========================================================================
def run_samples(sampler, prompts, num_samples, max_tokens, gen_dp,
                temperature=None, top_p=None, top_k=None, logprobs=None):
    """采样。prompts 少于 dp 时补齐再截回 —— Ray 的 dp 切分要求每个 rank 至少一条。"""
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


def collect_chunk(rt: Runtime, chunk, ci: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """一个 chunk 的采集，返回 (胜者列表, 指标)。

    ⭐ **全量 rollout**：chunk 里每一道题都要采 skill 候选，包括裸解已经做对的。
    理由：部署时 skill 模型面对的是任意题，不知道 executor 会不会做对；只在错题上训会让
    它只学会「救难题」、在简单题上也写一大堆纠错式提示。

    诊断（rubric）只对**做错的题**拉，且**每一种失败模式各诊一次后合并**（见 e18_multidiag）：
    8 次 rollout 往往挂在不同地方，只拿「第一条失败轨迹」去诊等于抛硬币选根因，而且旧缓存键
    只含 data_id，会把那次随机结果**永久写进缓存**。现在按失败签名分桶、按频次排序，
    合并文本里明确要求「Address ALL of them」。
    无诊断（做对的、或 API 失败的）的题**不丢弃**：skillgen_prompt 内部会填兜底文案。
    难易判定改用**连续 pass_rate**：裸解跑 EXEC_ROLLOUTS 次，base_pass_rate < 1 即视为「有提升
    空间」。这比单次贪心稳得多 —— 单次 T=0 的对/错在临界题上换个种子就翻转。
    """
    base_rolls = bare_solve(rt, chunk)
    base_rates = [_pass_rate(rr) for rr in base_rolls]
    base_acc = _mean(base_rates)
    # 诊断目标：没能每次都对的题。整组 rolls 都传进去 —— 由 multidiag 自己按失败签名分桶，
    # 每种模式诊一次（同签名只花一次 API），再合并成一份 rubric。
    wrong = [(r, rr) for r, rr, rate in zip(chunk, base_rolls, base_rates) if rate < 1.0]

    before = rt.rubric_cache.stats.copy()
    diags = rt.rubric_cache.diagnose_many(rt.checker, wrong)
    rmetrics = multidiag_metrics(rt.rubric_cache.stats - before)
    diag_by_id = {id(r): d for (r, _rr), d in zip(wrong, diags) if d}
    n_rubric_missing = len(wrong) - len(diag_by_id)
    # 多根因覆盖率：合并文本里有几段 FAILURE。持续=1 说明多轨迹诊断没带来新信息。
    n_multi = sum(1 for d in diag_by_id.values() if d.count('FAILURE ') > 1)

    # todo 现在是**全量** chunk：(record, rubric_or_empty, base_pass_rate)
    todo = [(r, diag_by_id.get(id(r), ''), rate)
            for r, rate in zip(chunk, base_rates)]

    # skill-gen：think 模式、T=1、每题 N 个候选（这就是拒绝采样的候选池）。
    # eval=False -> 用 SKILLGEN_SYSTEM（thinking 采集口径，允许多类型 skill）。
    sg = run_samples(rt.skill_sampler,
                     [skillgen_prompt(r['problem'], d, eval=False) for r, d, _br in todo],
                     N_SKILLS, SKILL_MAX_TOKENS, SKILL_SAMPLER_GPUS,
                     temperature=SKILL_GEN_TEMPERATURE, top_p=SKILL_GEN_TOP_P,
                     top_k=SKILL_GEN_TOP_K)
    per_task: List[List[Dict[str, Any]]] = []
    flat = []
    for (r, _d, _br), seqs in zip(todo, sg):
        cands = []
        for s in seqs or []:
            resp = seq_text(s)
            block = extract_skill(resp)
            c = {'skills': block, 'response': resp, 'parseable': bool(block),
                 'with_pass': None, 'kept': False,
                 'skillgen_stop': getattr(s, 'stop_reason', None)}
            cands.append(c)
            if block:
                flat.append((r, c))
        per_task.append(cands)

    # executor 带 skill 重解：**每个 skill 跑 EXEC_ROLLOUTS 次**，取连续 pass_rate。
    # 这是本次改造的核心：只有连续值才能在「全部都能做对」的易题上区分 skill 好坏。
    if flat:
        ws = run_samples(rt.base_sampler,
                         [skill_solve_prompt(r['problem'], c['skills']) for r, c in flat],
                         EXEC_ROLLOUTS, EXEC_MAX_TOKENS, BASE_SAMPLER_GPUS,
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
            c['with_pass'] = _pass_rate(rr)
            c['n_rollouts'] = n
            c['roll_kind'] = (next((x['kind'] for x in rr if not x['correct']), 'pass')
                              if rr else 'empty')

    # 三道筛：第一道改成**pass_rate 严格不降 + 取最大**（详见 select_winner）
    accepted, sims = [], []
    n_pass_cands = n_survivors = 0
    n_acc_hard = n_acc_easy = 0
    gtot = {'improved': 0, 'tied': 0, 'degraded': 0}
    gains = []
    for (r, d, base_rate), cands in zip(todo, per_task):
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
            # 审计字段（只进数据集文件，不进训练轨迹）
            # base_pass_rate / with_pass / pass_gain：连续口径下判断“skill 到底有没有用”的依据。
            'base_pass_rate': base_rate, 'with_pass_rate': best['with_pass'],
            'pass_gain': best['pass_gain'], 'gain_kind': best['gain_kind'],
            'rubric': d, 'chunk': ci, 'run': RUN_ID,
            'rubric_similarity': best['rubric_similarity'],
            'skill_chars': len(best['skills']),
            'n_candidates_passed': len(passers)})

    _cand_pass = (n_pass_cands / max(1, len(todo) * N_SKILLS)) if todo else 0.0
    metrics = {
        'train/baseline_accuracy': base_acc,
        'train/accept_rate': (len(accepted) / len(todo)) if todo else 0.0,
        'train/candidate_pass_rate': _cand_pass,
        # ⭐ 采集侧 lift：**全部**候选的平均增量，无选择偏差。不要用
        # `gain/selected_pass_gain` 代替它：后者只统计已通过 +MIN_PASS_GAIN 门槛的胜者，
        # 按定义恒为正，衡量的是「被选中那条有多好」而非「模型平均能写多好」。
        # 与 `eval/lift` 也不可直接比：这里的 prompt 带 rubric（教师诊断），eval 是 query-only，
        # 所以 train/lift 包含了“教师诊断的价值”，两者的差距正是本实验要缩小的东西。
        'train/lift': _cand_pass - base_acc,
        'train/selected_rubric_similarity': _mean(sims),
        'train/selected_skill_length_characters': _mean(
            [float(s['skill_chars']) for s in accepted]),
        'signal/n_wrong': float(len(wrong)),
        'signal/n_rubric_missing': float(n_rubric_missing),
        'signal/n_multi_cause': float(n_multi),
        'signal/n_accepted': float(len(accepted)),
        # 分层接受数：易题 = base_pass_rate 已经 1.0（跑 8 次全对）。
        'signal/n_accepted_hard': float(n_acc_hard),
        'signal/n_accepted_easy': float(n_acc_easy),
        # 天花板题数：base_pass_rate 高到拿不到 +MIN_PASS_GAIN（如 8/8），结构性无法入池。
        # 它与 n_accepted_easy 合看：前者持续很大就说明大量 GPU 花在了注定不入池的题上。
        'signal/n_ceiling': float(sum(1 for _r, _d, br in todo
                                      if br + MIN_PASS_GAIN > 1.0 + 1e-9)),
        'signal/min_pass_gain': MIN_PASS_GAIN,
        # 候选级增量分解（相对裸解 pass_rate）。degraded 最重要：skill 把通过率拉低了，
        # 它不会反映在 accept_rate 上，持续偏高就说明 skill-gen 在写有害提示。
        'gain/improved_candidates': float(gtot['improved']),
        'gain/tied_candidates': float(gtot['tied']),
        'gain/degraded_candidates': float(gtot['degraded']),
        'gain/degrade_rate': (gtot['degraded'] / max(1, sum(gtot.values()))),
        'gain/improve_rate': (gtot['improved'] / max(1, sum(gtot.values()))),
        # 胜者的平均 pass_rate 增量：这才是「入池样本到底有多有用」的直接度量。
        # 持续趋近 0 就说明池子里全是「写了也白写」的 skill。
        'gain/selected_pass_gain': _mean(gains),
    } | rmetrics | class_metrics([d for _r, d, _br in todo if d])
    return accepted, metrics


def dump_dataset(accepted) -> None:
    """胜者落盘 append-only 的 SFT 数据集（含 rubric/相似度/pass 全审计字段）。

    这份文件是 E18 的主产物：它让「筛选器选了什么」可离线复算、可跨 run 复用（不必重跑 GPU
    就能换 SFT 超参再训一遍）。
    """
    if not accepted:
        return
    path = os.path.join(OUTPUT_DIR, 'e18_sft_dataset.jsonl')
    with open(path, 'a', encoding='utf-8') as f:
        for s in accepted:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


# ===========================================================================
# 训练：纯 SFT（query-only 轨迹）
# ===========================================================================
def train_batch(rt: Runtime, samples) -> Tuple[int, Dict[str, float]]:
    """在攒够的胜者上做 1 个 optimizer step。

    ⭐ 训推一致的关键：训练 prompt 段用 **`skillgen_prompt(..., eval=True)`**，与 run_eval /
    部署时用的系统提示逐字相同（SKILLGEN_SYSTEM_EVAL，不带 rubric）。
    采集时用的是带诊断的 thinking 口径（SKILLGEN_SYSTEM）—— 那只是为了**邀出**好 skill，
    不能拿去当训练分布：线上没有诊断可用，拿带诊断的 prompt 去训会学成「看着诊断改写」。

    响应段走 messages 编码而不是拼采样 token：胜者的 `<skills>` 文本是程序合成的（采集时原
    响应带 <think>、且包装不同），本来就没有对应的采样 token 序列。`key_rounds` 标出最后一轮
    （assistant）为唯一可训区，prompt 段全 -100。

    loss 是纯 `CrossEntropyLoss`（不走 GRPO）：不传 advantages。拒绝采样的“选择”已经完成于三道筛
    （只有胜者入池），此处只需拟合目标文本。若要给样本加权，得在 loss 内部乘，而不是传
    advantages —— CrossEntropyLoss 不读这个参数，传了也是静默无效（所以此处根本不传）。
    """
    samples = [s for s in samples if (s.get('response') or '').strip()]
    n = (len(samples) // TRAIN_DP) * TRAIN_DP      # dp 切分要求整倍数
    if n == 0:
        return 0, {}
    samples = samples[:n]
    trajs = []
    for s in samples:
        msgs = skillgen_prompt(s['problem'], '', eval=True)['messages']
        trajs.append({'messages': msgs + [{'role': 'assistant', 'content': s['response']}],
                      'user_data': pack_user_data({'key_rounds': [len(msgs)]})})
    micro = max(TRAIN_DP, min(TRAIN_MICRO_BATCH, n))
    for i in range(0, n, micro):
        rt.skill_model.forward_backward(inputs=trajs[i:i + micro])
    rt.skill_model.clip_grad_and_step()
    # ⭐ 这里**不**同步权重。同步只在 run_eval 前后发生（见 sync_for_eval），以保证
    # skill_sampler 在**采集**阶段永远是初始权重。
    # 为何：采集用 thinking 口径邀出候选，而训练目标是 nothink 的 <skills> 纯文本。
    # 每步同步会把“别推理、直接吐 skills”回灌采集端，而下一轮采集又要求它 thinking
    # —— 两个分布互相拉扯，训练每次都赢。实测（output.e18.en）：4 步内 skill 长度
    # 281->573、出现 `Motor virtue` / `spectral misfire` 这类退化文本，candidate_pass_rate
    # 从 0.801 跌到 0.404、train/lift 转负（-0.055）。采集固定用初始权重能切断这个回路。
    # 代价：训练对采集零反馈，本质上退化成“离线数据生成 + 独立 SFT”。eval 仍用最新
    # 权重，所以 eval/lift 依旧反映 skill_model 的真实进步。

    metrics = {'train/n_samples': float(n)}
    # ⭐ 必须用 float() 尝试转换而不是 isinstance 判数值型：twinkle 的 LossMetric.calculate()
    # 把 loss / grad_norm 格式化成**字符串**后才返回（`f'{avg_loss:.4f}'`），用
    # isinstance(val, (int, float)) 会把这两个最关键的优化指标静默丢弃 —— 且丢在写文件之前，
    # 所以 train_log / swanlab / 日志里全部看不到，事后也无法找回。
    # 转不成的（'total time elapse'='12.3 minutes'、'speed'='1.2 iters/s'）才跳过。
    for k, val in (rt.skill_model.calculate_metric(is_training=True) or {}).items():
        if isinstance(val, bool):
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if k.startswith('learning rate'):
            if 'group 1' in k:
                metrics['train/lr'] = fval
        elif k.startswith('train/'):
            metrics[k.replace(' ', '_')] = fval
        else:
            metrics[f'train/{k.replace(" ", "_")}'] = fval
    return 1, metrics


# ===========================================================================
# eval：query-only（部署口径）
# ===========================================================================
def _sync_trained_to_sampler(rt: Runtime) -> None:
    """把当前（已训练）权重临时推给 skill_sampler，供 eval 使用。

    只是「临时」：eval 一结束就由 _restore_base_weights 把初始权重灌回去。skill_model 侧
    不落盘、不 load，训练权重与优化器状态全程不受影响。
    """
    rt.ckpt.sync_weights(merge_and_sync=True)
    rt.skill_sampler.reset_prefix_cache()


def _restore_base_weights(rt: Runtime) -> None:
    """把 skill_sampler 恢复到**初始**权重，供下一轮采集使用。

    ⭐ 走 `skill_sampler.load_weights_from_path()`（不传参 = sampler 自己的 model_id，即原始
    预训练权重），从磁盘直接流进 vLLM。关键是它**完全不碰 skill_model**：
    不 save、不 load，训练权重和 AdamW 动量都不受影响。

    对比曾经考虑过的「save 训练权重 -> skill_model.load(初始) -> sync -> load 回训练权重」：
    那条路要在训练模型上来回 load 两次，一旦中途失败（OOM / 磁盘满 / 进程被杀），训练端就
    停在初始权重上却带着原来的优化器状态继续跑，训练成果被静默清零且日志上看不出来。
    现在最坏情况只是采集端权重不对（下一次 eval 前的 sync 会覆盖掉），训练端不可能被破坏。

    reset_prefix_cache 必须跟着走：prefix cache 里缓存的是旧权重算出的 KV，换完权重不清就会
    拿旧 KV 拼新权重的输出。
    """
    rt.skill_sampler.load_weights_from_path()
    rt.skill_sampler.reset_prefix_cache()


def run_eval(rt: Runtime, eval_records, base_cache: Dict[str, Dict[str, Any]],
             updates: int = 0) -> Dict[str, float]:
    """部署口径 eval：**nothink + SKILLGEN_SYSTEM_EVAL + 不给 rubric**，与训练分布逐字一致。

    ⭐ skill_sampler 建立时是 enable_thinking=True（采集需要 thinking 多样性），而训练/部署是
    nothink。所以 eval 必须先把同一个引擎的**客户端模板**临时切成 nothink，跑完再切回；
    只换编码模板，引擎本身不动（与 skill_ablate/trainer.py 的做法同源）。
    不切的后果：eval 会带着 thinking 布局生成，与训练的 nothink 布局错配 —— 测出来的数不是
    部署时的真实能力。

    baseline（executor 冻结）整个 run 只算一次，之后走 base_cache（缓存的是 **pass_rate**）。
    eval 也跑 EXEC_ROLLOUTS 次取通过率：与采集口径一致，否则 lift 不可比。

    ⭐ 权重由调用方负责：进来之前已 sync 成最新训练权重，出去之后立刻恢复初始权重
    （见 main 里的 _sync_trained_to_sampler / _restore_base_weights）。本函数只管跑分。
    """
    todo = [r for r in eval_records if r['data_id'] not in base_cache]
    for r, rr in zip(todo, bare_solve(rt, todo) if todo else []):
        base_cache[r['data_id']] = _pass_rate(rr)
    base_rates = [base_cache[r['data_id']] for r in eval_records]
    base_acc = _mean(base_rates)

    # skill 模型：nothink + eval 系统提示、greedy 出一条 skill
    rt.skill_sampler.set_template(Template, model_id=MODEL_ID, enable_thinking=False,
                                  max_length=MAX_MODEL_LEN)
    try:
        sg = run_samples(rt.skill_sampler,
                         [skillgen_prompt(r['problem'], '', eval=True) for r in eval_records],
                         1, SKILL_MAX_TOKENS, SKILL_SAMPLER_GPUS, temperature=0.0)
    finally:
        # 必须切回：下一个 chunk 的采集要 thinking。放 finally 里是为了 eval 中途报错也不会
        # 把采集口径永久卡在 nothink 上。
        rt.skill_sampler.set_template(Template, model_id=MODEL_ID, enable_thinking=True,
                                      max_length=MAX_MODEL_LEN)
    skills = [extract_skill(seq_text(first_seq(s))) for s in sg]
    n_parsed = sum(1 for s in skills if s)
    # executor 也跑 EXEC_ROLLOUTS 次，与 baseline / 采集同口径
    ws = run_samples(rt.base_sampler,
                     [skill_solve_prompt(r['problem'], s) for r, s in zip(eval_records, skills)],
                     EXEC_ROLLOUTS, EXEC_MAX_TOKENS, BASE_SAMPLER_GPUS,
                     temperature=EXEC_TEMPERATURE, top_p=EXEC_TOP_P)
    pairs, spans = [], []
    for r, seqs in zip(eval_records, ws):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, r['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs)
    rates, i = [], 0
    for n in spans:
        rates.append(_pass_rate(judged[i:i + n]))
        i += n
    acc = _mean(rates)
    # ⭐ skill 原文落盘：聚合指标看不出「写成了什么样」，lift 为负时必须能回到原文归因。
    dump_eval_skills(eval_records, skills, base_rates, rates, updates)
    # 难题 = baseline 没能每次都对（pass_rate < 1）；rescue 改成平均 pass_rate 增量。
    hard = [(b, w) for b, w in zip(base_rates, rates) if b < 1.0]
    return {
        'eval/accuracy': acc,
        'eval/baseline_accuracy': base_acc,
        'eval/lift': acc - base_acc,
        'eval/format_rate': n_parsed / max(1, len(eval_records)),
        'eval/hard_rescue_rate': (_mean([w - b for b, w in hard]) if hard else 0.0),
        'eval/n_hard': float(len(hard)),
        'eval/skill_length_characters': _mean([float(len(s)) for s in skills]),
    }


def dump_eval_skills(eval_records, skills, base_rates, rates, updates) -> None:
    """eval 产出的 skill 原文落盘（append-only）。

    为何必需：`run_eval` 以前只回传聚合指标，`skills` 是局部变量、用完即弃，而
    SAVE_EVERY_UPDATES 默认 50，所以 lift 为负时既看不到 skill 原文、也没有 ckpt 可以重跑
    —— 无法区分「内容写得差」和「注入方式不对」。这份产物让部署口径可离线归因。

    逐题存 base/with pass_rate，所以可以直接筛出被 skill 带坏的题（with < base）。
    """
    path = os.path.join(OUTPUT_DIR, 'eval_skills.jsonl')
    with open(path, 'a', encoding='utf-8') as f:
        for r, s, b, w in zip(eval_records, skills, base_rates, rates):
            f.write(json.dumps({
                'updates': updates,
                'data_id': r.get('data_id'),
                'task_id': r['reference_answer'].get('task_id'),
                'base_pass_rate': b,
                'with_pass_rate': w,
                'pass_gain': w - b,
                'skill_chars': len(s),
                'problem': r['problem'],
                'skills': s,
            }, ensure_ascii=False) + '\n')


# ===========================================================================
# main
# ===========================================================================
def build_runtime(checker, rubric_cache) -> Runtime:
    """三组卡：train / skill_sampler / base_sampler(executor)。SFT 不需要 ref 模型。"""
    r0 = TRAIN_GPUS
    r1 = r0 + SKILL_SAMPLER_GPUS
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='train', ranks=list(range(0, r0)), device_type='GPU'),
        DeviceGroup(name='skill_sampler', ranks=list(range(r0, r1)), device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(r1, NUM_GPUS)), device_type='GPU')])

    skill_model = TransformersModel(
        model_id=MODEL_ID, remote_group='train',
        device_mesh=DeviceMesh.from_sizes(world_size=TRAIN_GPUS, dp_size=TRAIN_DP,
                                          fsdp_size=TRAIN_FSDP),
        ddp_config={'find_unused_parameters': False}, torch_dtype='float32')
    skill_model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    # ⭐ 训练模板 enable_thinking=False：训练/部署都是 nothink，必须一致。
    # 采集（skill_sampler）才是 thinking，那只用来邀出候选，不是训练分布。
    # 反例：改成 True 会把 `\n<think>\n\n</think>` 那 4 个固定 token 也纳入可训区，而 run_eval
    # 是 nothink 生成的 —— 两边可训/生成区对不上，就不再是训推一致。
    skill_model.set_template(Template, model_id=MODEL_ID, enable_thinking=False,
                             max_length=MAX_MODEL_LEN, truncation_strategy='delete')
    skill_model.set_processor(InputProcessor, padding_free=False)
    # ⭐ 纯 SFT：交叉熵，不走 GRPO 那一套。
    # CrossEntropyLoss 只看 inputs['labels']（即 key_rounds 圈出的 assistant 段），没有 ratio /
    # clip / KL / advantage，也不需要 old_logps、ref 模型。拒绝采样的“选择”已经全部发生在
    # 三道筛里（胜者才入池），到了 loss 这一层就是普通的“拟合这条目标文本”，不应再有 RL 项。
    # reduction='mean'（默认）-> num_tokens=0，每个 micro 自己 token-mean，梯度按 micro 数归一。
    skill_model.set_loss('CrossEntropyLoss')
    skill_model.set_optimizer('AdamW', lr=LR)

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

    # skill_sampler 开 think：采集要多样性（拒绝采样的前提）。eval 时同一引擎跑 query-only。
    skill_sampler = _sampler('skill_sampler', SKILL_SAMPLER_GPUS, enable_thinking=True)
    # executor 关 think（与 E23 一致）：开 think 有大量 rollout 撞满预算连代码块都写不出来。
    base_sampler = _sampler('base_sampler', BASE_SAMPLER_GPUS, enable_thinking=True)
    return Runtime(
        skill_model=skill_model, skill_sampler=skill_sampler, base_sampler=base_sampler,
        ckpt=CheckpointEngineManager(model=skill_model, sampler=skill_sampler),
        checker=checker, rubric_cache=rubric_cache)


def swan_init():
    if swanlab is None or os.environ.get('SWANLAB_MODE') == 'disabled':
        logger.info('[swanlab] 未启用，只写 train_log.jsonl')
        return None
    name = 'E18_rejection_sft_code' + (f'_{RUN_TAG}' if RUN_TAG else '') + f'_{RUN_ID}'
    swanlab.init(project=SWAN_PROJ, experiment_name=name, config={
        'model_id': MODEL_ID, 'seed': SEED, 'chunk_size': CHUNK_SIZE, 'n_skills': N_SKILLS,
        'accumulate': ACCUMULATE, 'max_updates': MAX_UPDATES, 'lr': LR,
        'loss': 'CrossEntropyLoss',
        'exec_rollouts': EXEC_ROLLOUTS, 'exec_temperature': EXEC_TEMPERATURE,
        'exec_top_p': EXEC_TOP_P, 'min_gain_rollouts': MIN_GAIN_ROLLOUTS,
        'min_pass_gain': MIN_PASS_GAIN,
        'skill_char_limit': SKILL_CHAR_LIMIT, 'skill_max_tokens': SKILL_MAX_TOKENS,
        'exec_max_tokens': EXEC_MAX_TOKENS, 'eval_size': EVAL_SIZE,
        'skill_gen_temperature': SKILL_GEN_TEMPERATURE,
        'run_tag': RUN_TAG, 'run_id': RUN_ID, 'output_dir': OUTPUT_DIR})
    logger.info(f'[swanlab] project={SWAN_PROJ} experiment={name}')
    return swanlab


def swan_log(swan, row: Dict[str, Any], step: int) -> None:
    if swan is None:
        return
    m = {k: float(v) for k, v in row.items()
         if isinstance(v, (int, float)) and not isinstance(v, bool)}
    try:
        swan.log(m, step=step)
    except Exception as e:
        logger.warning(f'[swanlab] log 失败（已忽略）：{e}')


# 不参与污染判定的文件：跑一次要花大量 CPU/沙箱时间（~900 道题跑参考解答），
# 且内容只依赖题池、与哪个 run 无关，所以要从旧目录携带到新目录。
_CARRY_OVER = ('bcb_broken_tasks.json', )


def archive_output_dir() -> None:
    """启动时把已存在的 OUTPUT_DIR 整个 mv 走，保证本 run 写入空目录。

    为何必需：`e18_sft_dataset.jsonl` / `train_log.jsonl` 都是 `open(..., 'a')` 追写。
    没有这一步时，重启一次就把新旧 run 的样本焊在同一个文件里，而且不报错。
    实测后果（output.e18.en）：崩溃 run 的 602 条（硬缺陷 41.4%：ttr<0.45 有 170 条、
    超长 125 条、重复 118 条）与新 run 的 274 条混在一起，旧数据占 69%。
    本 run 的训练不受影响（训练吃的是内存里的 pool，这个文件只写不读），但它的存在
    意义就是“不重跑 GPU 就能换 SFT 超参再训一遍”（见 dump_dataset）—— 那个场景下
    污染会直接进训练，且无任何报错，只是模型更差。

    用 mv 而不是删：旧 run 的诊断数据（eval 曲线、崩溃样本）是可复用的分析素材，
    丢了就得重跑 GPU 才能拿回。
    """
    if not os.path.isdir(OUTPUT_DIR):
        return
    if not os.listdir(OUTPUT_DIR):        # 空目录直接用，不制造无意义的归档
        return
    # 归档名带旧目录的 mtime（而不是当前时间）：同一批旧数据无论何时重启都归到
    # 同一个名字上，看名字就知道里面是哪段时间的 run。
    stamp = time.strftime('%m%d-%H%M%S', time.localtime(os.path.getmtime(OUTPUT_DIR)))
    dst = f'{OUTPUT_DIR}.bak-{stamp}'
    n = 1
    while os.path.exists(dst):            # 同一秒重启两次也不能覆盖已有归档
        dst = f'{OUTPUT_DIR}.bak-{stamp}.{n}'
        n += 1
    shutil.move(OUTPUT_DIR, dst)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    carried = []
    for name in _CARRY_OVER:
        src = os.path.join(dst, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, name))
            carried.append(name)
    logger.warning(f'[output] 已存在的 {OUTPUT_DIR} 已归档到 {dst}'
                   + (f'（携带缓存：{", ".join(carried)}）' if carried else ''))


def main():
    archive_output_dir()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if ACCUMULATE % TRAIN_DP != 0:
        raise ValueError(f'ACCUMULATE({ACCUMULATE}) 必须是 TRAIN_DP({TRAIN_DP}) 的整数倍，'
                         f'否则 dp 切分会丢掉尾部样本')
    checker = build_checker()
    if checker is None:
        raise RuntimeError('没有教师 API（LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL / '
                           'OPENAI_API_KEY 都没设）；rubric 是三道筛的参照系，无法降级运行')
    train_dataset, eval_records = load_records(SEED, EVAL_SIZE, OUTPUT_DIR)
    if len(train_dataset) < CHUNK_SIZE:
        raise ValueError(f'训练池 {len(train_dataset)} 小于 CHUNK_SIZE {CHUNK_SIZE}')
    logger.info(f'[data] train={len(train_dataset)} eval={len(eval_records)}')

    rt = build_runtime(checker, MultiDiagCache())
    swan = swan_init()
    base_cache: Dict[str, Dict[str, Any]] = {}
    pool: List[Dict[str, Any]] = []       # 攒够 ACCUMULATE 条就 SFT 一次
    updates, ci, si, epoch, last_eval = 0, 0, 0, 0, 0
    logger.info(f'E18 start: lr={LR} chunk={CHUNK_SIZE} n_skills={N_SKILLS} '
                f'accumulate={ACCUMULATE} max_updates={MAX_UPDATES} output={OUTPUT_DIR}')

    with open(os.path.join(OUTPUT_DIR, 'train_log.jsonl'), 'a', encoding='utf-8') as log_fh:
        # ⭐ updates=0 的 baseline eval：没有这一点，`eval/*` 曲线的第一个数据点已经是训了
        # EVAL_EVERY_UPDATES 步之后的值，无法区分「训练带来的提升」和「初始就有的能力」。
        # 注意 base_cache 缓存的是 executor 裸解 pass_rate（executor 的基线），不是 skill 模型
        # 的基线 —— 两回事，不能互代。
        # 副作用：这次 eval 会把裸解结果写进 base_cache，所以后续 eval 能直接复用，
        # 整体 GPU 成本并非净增一整次 eval。
        if eval_records:
            row0 = {'step': 0, 'chunk': 0, 'updates': 0, 'epoch': 0,
                    'signal/pool_size': 0.0}
            row0.update(run_eval(rt, eval_records, base_cache, updates=0))
            row0['eval/updates_done'] = 0
            log_fh.write(json.dumps(row0, ensure_ascii=False) + '\n')
            log_fh.flush()
            swan_log(swan, row0, si)
            logger.info('[baseline u0] '
                        + ' '.join(f'{k}={v:.4g}' for k, v in row0.items()
                                   if isinstance(v, float)))
            si += 1

        while updates < MAX_UPDATES:
            loader = DataLoader(dataset=train_dataset, batch_size=CHUNK_SIZE, num_workers=0,
                                shuffle=True, drop_last=True,
                                generator=torch.Generator().manual_seed(SEED + epoch))
            for chunk in loader:
                if updates >= MAX_UPDATES:
                    break
                t0 = time.time()
                accepted, metrics = collect_chunk(rt, chunk, ci)
                dump_dataset(accepted)
                pool.extend(accepted)

                n_upd, tmetrics = 0, {}
                while len(pool) >= ACCUMULATE and updates < MAX_UPDATES:
                    batch, pool = pool[:ACCUMULATE], pool[ACCUMULATE:]
                    k, tm = train_batch(rt, batch)
                    n_upd += k
                    updates += k
                    tmetrics = tm or tmetrics

                row = {'step': si, 'chunk': ci, 'updates': updates, 'epoch': epoch,
                       'seconds': round(time.time() - t0, 1),
                       'signal/pool_size': float(len(pool)),
                       **metrics, **tmetrics}
                if eval_records and (updates - last_eval >= EVAL_EVERY_UPDATES
                                     or updates >= MAX_UPDATES) and updates > 0:
                    # ⭐ eval 前把训练权重临时推给 sampler，跑完立刻把初始权重灌回去。
                    # train_batch 已不再逐步同步，所以采集阶段永远是初始模型；
                    # 只有这一段区间内 sampler 带的是训练权重。finally 保证 eval 中途
                    # 报错也不会把退化权重永久留在采集端。
                    _sync_trained_to_sampler(rt)
                    try:
                        row.update(run_eval(rt, eval_records, base_cache,
                                            updates=updates))
                    finally:
                        _restore_base_weights(rt)
                    row['eval/updates_done'] = updates
                    last_eval = updates
                log_fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                log_fh.flush()
                swan_log(swan, row, si)
                logger.info(f'[s{si} c{ci} u{updates}/{MAX_UPDATES}] '
                            + ' '.join(f'{k}={v:.4g}' for k, v in row.items()
                                       if isinstance(v, float)))
                if SAVE_EVERY_UPDATES and updates and updates % SAVE_EVERY_UPDATES == 0:
                    rt.skill_model.save(f'E18-u{updates}', output_dir=OUTPUT_DIR)
                si += 1
                ci += 1
            epoch += 1

    rt.skill_model.save('E18-final', output_dir=OUTPUT_DIR)
    rt.rubric_cache.close()
    if swan is not None:
        swan.finish()
    logger.info(f'done: updates={updates} chunks={ci} -> {OUTPUT_DIR}/E18-final')


if __name__ == '__main__':
    main()
