# -*- coding: utf-8 -*-
"""E18-KOD 离线 SFT：把 `e18_collect_kod.py` 采到的胜者用 **nothink 口径**训一遍，并在
首尾各跑一次 eval，验证训练是否真的带来提升。

与在线版 `e18_rejection_sft.py` 的关系：**保留训练 + eval，去掉采集**。rubric 诊断、
拒绝采样、chunk 循环全部移除 —— 数据已经在 `e18_sft_dataset.jsonl` 里落好了。

卡位：4 训练 + 2 skill_sampler + 2 base_sampler(executor)。为何不是 8 卡全训练：
eval 要 skillmodel 生成 skill、executor 跑代码，两者都需要 vLLM 引擎常驻。

训推一致的三个不可动点（与在线版逐字对齐，改任何一处就不再是同一个实验）：
1. **prompt 段用 `skillgen_prompt(..., eval=True)`** —— 即 SKILLGEN_SYSTEM_EVAL、不带 rubric。
   采集时用的是带诊断的 thinking 口径（SKILLGEN_SYSTEM），那只是为了「邀出」好 skill；
   线上没有诊断可用，拿带诊断的 prompt 去训会学成「看着诊断改写」。
2. **模板 enable_thinking=False** —— 训练/部署都是 nothink，必须一致。改成 True 会把
   think 那几个固定 token 也纳入可训区，与部署时的生成区对不上。
3. **`key_rounds=[len(msgs)]`** 标出最后一轮（assistant）为唯一可训区，prompt 段全 -100。

loss 是纯 `CrossEntropyLoss`（twinkle 没有 SFTLoss 这个类）：拒绝采样的「选择」已经
发生在采集侧的三道筛里（只有胜者入池），到 loss 这层就是普通的拟合目标文本，
不该再有 ratio / clip / KL / advantage。

⭐ eval 口径与在线版的**唯一差异**：executor 只跑 1 次且 temperature=0（在线版是 8 次
T=0.6 取通过率）。这是按需求指定的 —— 省 GPU、且 greedy 单次可复现。代价写在
run_eval 的注释里：pass_rate 退化成 0/1 二值，单题不可比，只能看 100 题的均值。

产物（OUTPUT_DIR 下）：
* `sft_log.jsonl`：逐 step 的 loss / grad_norm / lr，用来看收敛。
* `eval_log.jsonl`：首尾两次 eval 的聚合指标。
* `eval_skills.jsonl`：eval 生成的 skill 原文 + 逐题 base/with，用来归因。
* `KODSFT-final/`：权重（SAVE_EVERY_STEPS>0 时还有 `KODSFT-s<step>/`）。
"""
import json
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, pack_user_data
from twinkle.model import TransformersModel
from twinkle.patch.no_split_modules import NoSplitModulesPatch
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

_HERE = os.path.dirname(os.path.abspath(__file__))
_COOKBOOK = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (_HERE, os.path.join(_COOKBOOK, 'human')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e18_kodcode import (clean_text, empty_roll, extract_skill,  # noqa: E402
                         judge_seqs, load_records)
from e18_prompts import direct_prompt, skill_solve_prompt, skillgen_prompt  # noqa: E402

logger = get_logger()

# ========== 配置 ==========
MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3-4B')
DATA_PATH = os.environ.get(
    'DATA_PATH', os.path.join(_HERE, 'output.e18.kod', 'e18_sft_dataset.jsonl'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(_HERE, 'output.e18.kod.sft'))

# 8 卡分三组：4 训练 + 2 skillmodel 推理 + 2 executor。
# ⭐ 为何 executor 只需 2 张（采集时是 6 张）：eval 只跑 100 题 × 1 次 = 100 个序列，
# 而采集是每 chunk 2304 个。序列数少两个量级，再加卡只会让训练侧变慢。
TRAIN_GPUS = int(os.environ.get('TRAIN_GPUS', 4))
SKILL_SAMPLER_GPUS = int(os.environ.get('SKILL_SAMPLER_GPUS', 2))
BASE_SAMPLER_GPUS = int(os.environ.get('BASE_SAMPLER_GPUS', 2))
NUM_GPUS = TRAIN_GPUS + SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS
TRAIN_FSDP = int(os.environ.get('TRAIN_FSDP', 1))
TRAIN_DP = TRAIN_GPUS // TRAIN_FSDP
GPU_MEM = float(os.environ.get('GPU_MEM', 0.8))

SEED = int(os.environ.get('SEED', 42))
EPOCHS = float(os.environ.get('EPOCHS', 3))
# ⭐ 必须是 TRAIN_DP 的整倍数：dp 切分会把不足一轮的尾部丢掉。
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16))
MICRO_BATCH = int(os.environ.get('MICRO_BATCH', 8))
LR = float(os.environ.get('LR', 1e-5))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16000))

# ---- eval 口径 ----
EVAL_SIZE = int(os.environ.get('EVAL_SIZE', 100))
SKILL_MAX_TOKENS = int(os.environ.get('SKILL_MAX_TOKENS', 8192))
EXEC_MAX_TOKENS = int(os.environ.get('EXEC_MAX_TOKENS', 15000))
# ⭐ executor 跑 1 次、temperature=0（按需求指定，与在线版的 8×T=0.6 不同）。
# 后果：单题 pass_rate 只能是 0 或 1，所以**不要看单题差异**，只看 100 题均值；
# 也因此 hard_rescue 这类分层指标失去意义（不再计算）。greedy 的好处是可复现。
EXEC_ROLLOUTS = int(os.environ.get('EXEC_ROLLOUTS', 1))
EXEC_TEMPERATURE = float(os.environ.get('EXEC_TEMPERATURE', 0.0))

# 数据清洗门槛（见 load_samples 的注释，都是实测抽检出来的缺陷）
MIN_CHARS = int(os.environ.get('MIN_CHARS', 200))
MAX_CHARS = int(os.environ.get('MAX_CHARS', 1500))
DROP_CJK = os.environ.get('DROP_CJK', '1') == '1'

SAVE_EVERY_STEPS = int(os.environ.get('SAVE_EVERY_STEPS', 0))   # 0 = 只在结束时存
LOG_EVERY_STEPS = int(os.environ.get('LOG_EVERY_STEPS', 1))
RUN_ID = time.strftime('%m%d-%H%M%S')


# ===========================================================================
# 数据
# ===========================================================================
def load_samples() -> List[Dict[str, Any]]:
    """读胜者并做格式清洗。返回 [{'problem', 'response'}]。

    ⭐ 为何要在 SFT 侧再清一遍（采集侧已有 SKILL_CHAR_LIMIT）：抽检 1104 条实测出三类
    残留缺陷，占比虽小但都会被模型逐字学走：
      * CJK 混入 3/1104（'动态规划' 这种孤立中文词）—— 部署口径是纯英文，学走就成了双语输出；
      * 残留 `<skills>` 标签 2/1104 —— response 外层已经由采集侧包了一层，内层再出现就是嵌套；
      * 截断（无终止标点）7/1104 —— 学截断等于学「说半句就停」。
    合计约 1.1%，宁可丢掉也不喂进去。

    `response` 用采集时存的原字段（已是 `<skills>\\n...\\n</skills>` 包装），不重新拼：
    与在线版 `train_batch` 消费的是同一个字段，保持逐字一致。
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'找不到数据集：{DATA_PATH}')
    raw, bad_json = [], 0
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except Exception:
                # 采集进程被 kill 时最后一行可能是半行，跳过而不是让整个训练起不来。
                bad_json += 1
    drop = {'no_response': 0, 'too_short': 0, 'too_long': 0, 'cjk': 0,
            'nested_tag': 0, 'truncated': 0}
    out = []
    for r in raw:
        resp = (r.get('response') or '').strip()
        skills = (r.get('skills') or '').strip()
        if not resp or not skills:
            drop['no_response'] += 1
            continue
        if len(skills) < MIN_CHARS:
            drop['too_short'] += 1
            continue
        if len(skills) > MAX_CHARS:
            drop['too_long'] += 1
            continue
        if DROP_CJK and any('\u4e00' <= ch <= '\u9fff' for ch in skills):
            drop['cjk'] += 1
            continue
        if '<skills' in skills.lower() or '</skills' in skills.lower():
            drop['nested_tag'] += 1
            continue
        if not skills.rstrip().endswith(('.', '!', '?', ':', ')')):
            drop['truncated'] += 1
            continue
        out.append({'problem': r['problem'], 'response': resp,
                    'data_id': r.get('data_id', ''), 'run': r.get('run', '')})
    logger.info(f'[data] 读入 {len(raw)} 条'
                + (f'（跳过 {bad_json} 行半行）' if bad_json else '')
                + f'，清洗后 {len(out)} 条，丢弃明细 {drop}')
    runs = sorted({s['run'] for s in out if s['run']})
    if len(runs) > 1:
        # 续跑（KOD_RESUME=1）后同一份文件里会有多个 run，训练是全量混合 —— 这里只提示，
        # 不自动过滤：要不要分开训是实验设计问题，不该由脚本替你决定。
        logger.warning(f'[data] 数据里含 {len(runs)} 个 run：{runs}（全部混合训练）')
    return out


def load_eval_records(train_data_ids: set):
    """eval 集：从 KodCode 题池里挑 EVAL_SIZE 道**没有生成过 skill** 的题。

    ⭐ 必须排掉采集跑过的题，而且排的是 **candidates 里的全量 id**、不是仅胜者：
    采集时一道题跑了 4 个候选但只有 ~22% 能入池，剩下的题虽然不在训练集里，却已经
    被用来挑过 skill —— 拿它当 eval 会高估（数据选择偏差：那些题本身就是「skill 救不了」
    或「本来就全对」的）。所以传进来的 train_data_ids 应该来自 `e18_candidates.jsonl`。

    load_records 的 seed 与采集侧一致，所以题池顺序可复现；取前 EVAL_SIZE 条未采过的。
    """
    ds, _ = load_records(SEED, 0, OUTPUT_DIR)
    out = []
    for r in ds.dataset:
        if r['data_id'] in train_data_ids:
            continue
        out.append(r)
        if len(out) >= EVAL_SIZE:
            break
    logger.info(f'[eval] 选了 {len(out)} 道未采集过的题（排除已跑 {len(train_data_ids)} 题）')
    return out


def collected_data_ids() -> set:
    """采集阶段跑过的全部 data_id（含未入池的），用来从 eval 集里排除。

    优先读 `e18_candidates.jsonl`（全量）；没有它才退回 sft_dataset（仅胜者，覆盖不全，
    会让 eval 集混进采集过的题）。
    """
    ids = set()
    cand = os.path.join(os.path.dirname(DATA_PATH), 'e18_candidates.jsonl')
    src = cand if os.path.exists(cand) else DATA_PATH
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line)['data_id']))
            except Exception:
                continue
    logger.info(f'[eval] 排除源 {os.path.basename(src)}：{len(ids)} 道已跑题')
    return ids


# ===========================================================================
# 训练
# ===========================================================================
def build_model():
    """三组卡：train(4) / skill_sampler(2) / base_sampler(2)。

    返回 (model, skill_sampler, base_sampler, ckpt)。训练组配置与在线版 build_runtime 一致。

    ⭐ 两个 sampler 都直接建成 **enable_thinking=False**：本脚本没有采集阶段，不需要
    thinking 多样性，而 eval 就是部署口径（nothink）。因此不需要像在线版 run_eval 那样
    每次 eval 前后临时切模板再切回 —— 少一个可能切错的状态。

    ⭐ ckpt 只绑 skill_sampler：训练的是 skillmodel，executor 必须全程冻结，
    否则首尾两次 eval 的 baseline 不可比（分母都变了就无法归因给 skill）。
    """
    r0, r1 = TRAIN_GPUS, TRAIN_GPUS + SKILL_SAMPLER_GPUS
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='train', ranks=list(range(0, r0)), device_type='GPU'),
        DeviceGroup(name='skill_sampler', ranks=list(range(r0, r1)), device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(r1, NUM_GPUS)), device_type='GPU')])
    model = TransformersModel(
        model_id=MODEL_ID, remote_group='train',
        device_mesh=DeviceMesh.from_sizes(world_size=TRAIN_GPUS, dp_size=TRAIN_DP,
                                          fsdp_size=TRAIN_FSDP),
        ddp_config={'find_unused_parameters': False}, torch_dtype='float32')
    model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    # ⭐ enable_thinking=False：训练/部署都是 nothink，必须一致（见文件头注释第 2 点）。
    model.set_template(Template, model_id=MODEL_ID, enable_thinking=False,
                       max_length=MAX_MODEL_LEN, truncation_strategy='delete')
    model.set_processor(InputProcessor, padding_free=False)
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('AdamW', lr=LR)

    def _mk_sampler(group: str, world: int):
        s = vLLMSampler(
            model_id=MODEL_ID, remote_group=group,
            device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
            engine_args={'gpu_memory_utilization': GPU_MEM,
                         'max_model_len': MAX_MODEL_LEN, 'tensor_parallel_size': 1})
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=False,
                       max_length=MAX_MODEL_LEN)
        return s

    skill_sampler = _mk_sampler('skill_sampler', SKILL_SAMPLER_GPUS)
    base_sampler = _mk_sampler('base_sampler', BASE_SAMPLER_GPUS)
    ckpt = CheckpointEngineManager(model=model, sampler=skill_sampler)
    return model, skill_sampler, base_sampler, ckpt


# ===========================================================================
# eval：部署口径（nothink + SKILLGEN_SYSTEM_EVAL + 不给 rubric）
# ===========================================================================
def run_samples(sampler, prompts, num_samples, max_tokens, gen_dp,
                temperature=None, top_p=None):
    """采样。与 e18_collect_kod.run_samples 逐字一致（去掉本脚本用不到的 top_k/logprobs）。

    三处踩过的坑：
    1. 字段名是 `num_samples` 不是 `n`（SamplingParams 没有 `n`，传了直接 TypeError）。
    2. 走 `sampler.sample(prompts, params)`，不是 pack_user_data + generate_sequences。
    3. dp 补齐不能省：条数 < dp 会直接报错（eval 只 100 题、dp=2 虽然安全，
       但 EVAL_SIZE 调小到 1 时就会触发）。
    """
    if not prompts:
        return []
    import copy
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6 if temperature is None else temperature,
        top_p=0.95 if top_p is None else top_p,
        num_samples=num_samples)
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


def _pass_rate(rolls) -> float:
    return (sum(1.0 for x in rolls if x['correct']) / len(rolls)) if rolls else 0.0


def _judge_batch(records, prompts, sampler, gen_dp) -> List[float]:
    """跑 executor + 判分，返回逐题 pass_rate。

    spans 记每题实际拿到几条序列（可能为 0），不能直接按 EXEC_ROLLOUTS 切 judged：
    采样失败的题会返回空列表，按固定步长切会整体错位。
    """
    ws = run_samples(sampler, prompts, EXEC_ROLLOUTS, EXEC_MAX_TOKENS, gen_dp,
                     temperature=EXEC_TEMPERATURE)
    pairs, spans = [], []
    for r, seqs in zip(records, ws):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, r['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs) if pairs else []
    rates, i = [], 0
    for n in spans:
        rates.append(_pass_rate(judged[i:i + n]) if n else 0.0)
        i += n
    return rates


def run_eval(skill_sampler, base_sampler, eval_records,
             base_cache: Dict[str, float], tag: str, step: int) -> Dict[str, float]:
    """部署口径 eval：skillmodel 生成 1 个 skill -> executor 跑 1 次 -> 看提升。

    与训练分布逐字一致：nothink + SKILLGEN_SYSTEM_EVAL + 不给 rubric。

    ⭐ baseline 整个 run 只算一次，之后走 base_cache：executor 权重全程冻结，同一道题的
    裸解结果不会变（T=0 更是确定性的）。重算不仅浪费 GPU，还会因为 vLLM 的
    非确定性引入假的 baseline 漂移，把本来该归因给 skill 的差异污染掉。

    ⭐ 只看 `lift`（= accuracy - baseline）的**首尾差异**。因为 EXEC_ROLLOUTS=1 + T=0，
    单题 pass_rate 只能是 0/1，100 题的均值标准误差约 0.05 —— 所以 lift 变化小于
    约 0.07 时不要当成真实效果（双样本差值的噪声更大）。这是 1 次 rollout 的固有代价。
    """
    t0 = time.time()
    todo = [r for r in eval_records if r['data_id'] not in base_cache]
    if todo:
        rates = _judge_batch(todo, [direct_prompt(r['problem']) for r in todo],
                             base_sampler, BASE_SAMPLER_GPUS)
        for r, rate in zip(todo, rates):
            base_cache[r['data_id']] = rate
    base_rates = [base_cache[r['data_id']] for r in eval_records]

    sg = run_samples(skill_sampler,
                     [skillgen_prompt(r['problem'], '', eval=True) for r in eval_records],
                     1, SKILL_MAX_TOKENS, SKILL_SAMPLER_GPUS, temperature=0.0)
    skills = [extract_skill(seq_text(first_seq(s))) for s in sg]
    n_parsed = sum(1 for s in skills if s)
    with_rates = _judge_batch(
        eval_records,
        [skill_solve_prompt(r['problem'], s) for r, s in zip(eval_records, skills)],
        base_sampler, BASE_SAMPLER_GPUS)

    acc, base_acc = _mean(with_rates), _mean(base_rates)
    m = {'eval/accuracy': acc, 'eval/baseline_accuracy': base_acc,
         'eval/lift': acc - base_acc,
         'eval/format_rate': n_parsed / max(1, len(eval_records)),
         'eval/skill_length_characters': _mean([float(len(s)) for s in skills]),
         'eval/n_improved': float(sum(1 for b, w in zip(base_rates, with_rates) if w > b)),
         'eval/n_hurt': float(sum(1 for b, w in zip(base_rates, with_rates) if w < b)),
         'eval/seconds': round(time.time() - t0, 1)}
    # ⭐ skill 原文落盘：聚合指标看不出「写成了什么样」，lift 为负时必须能回到原文归因。
    with open(os.path.join(OUTPUT_DIR, 'eval_skills.jsonl'), 'a', encoding='utf-8') as f:
        for r, s, b, w in zip(eval_records, skills, base_rates, with_rates):
            f.write(json.dumps({'tag': tag, 'step': step, 'run': RUN_ID,
                                'data_id': r['data_id'], 'base': b, 'with': w,
                                'skill_chars': len(s), 'skill': s},
                               ensure_ascii=False) + '\n')
    return m


# ===========================================================================
# 训练
# ===========================================================================


def make_trajs(batch) -> List[Dict[str, Any]]:
    """样本 -> twinkle 轨迹。与在线版 `train_batch` 的构造逐字相同。"""
    trajs = []
    for s in batch:
        msgs = skillgen_prompt(s['problem'], '', eval=True)['messages']
        trajs.append({'messages': msgs + [{'role': 'assistant', 'content': s['response']}],
                      'user_data': pack_user_data({'key_rounds': [len(msgs)]})})
    return trajs


def step_metrics(model) -> Dict[str, float]:
    """取本 step 的优化指标。

    ⭐ 必须用 float() 试转而不是 isinstance 判数值型：twinkle 的 LossMetric.calculate()
    把 loss / grad_norm 格式化成**字符串**后才返回（`f'{avg_loss:.4f}'`），用
    isinstance(val, (int, float)) 会把这两个最关键的指标静默丢弃 —— 而这个脚本唯一的目的
    就是看收敛，丢了 loss 就什么都看不到了。
    转不成的（'total time elapse'='12.3 minutes'）才跳过。
    """
    out: Dict[str, float] = {}
    for k, val in (model.calculate_metric(is_training=True) or {}).items():
        if isinstance(val, bool):
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if k.startswith('learning rate'):
            if 'group 1' in k:
                out['lr'] = fval
        else:
            out[k.replace(' ', '_')] = fval
    return out


def archive_output_dir() -> None:
    """启动时把已存在的 OUTPUT_DIR 整个 mv 走（`sft_log.jsonl` 是追写的）。

    与 e18_collect_kod 同一套机制：不这么做，重跑一次就把两条 loss 曲线焊在一个文件里，
    而且不报错 —— 看收敛时会看到一条莫名其妙回弹的曲线。
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


def log_eval(m: Dict[str, float], tag: str, step: int) -> None:
    with open(os.path.join(OUTPUT_DIR, 'eval_log.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps({'tag': tag, 'step': step, 'run': RUN_ID, **m},
                           ensure_ascii=False) + '\n')
    logger.info(f'[eval:{tag}] ' + ' '.join(f'{k}={v:.4g}' for k, v in m.items()))


def main():
    t0 = time.time()
    archive_output_dir()
    samples = load_samples()
    if len(samples) < BATCH_SIZE:
        raise RuntimeError(f'可用样本 {len(samples)} 条 < BATCH_SIZE {BATCH_SIZE}')
    if BATCH_SIZE % TRAIN_DP:
        raise RuntimeError(f'BATCH_SIZE({BATCH_SIZE}) 必须是 TRAIN_DP({TRAIN_DP}) 的整倍数')
    # eval 题池先选好（纯 CPU），再开 GPU：选错了就不用白等 vLLM 启动的几分钟。
    eval_records = load_eval_records(collected_data_ids())
    if not eval_records:
        raise RuntimeError('eval 集为空（题池里的题已全部被采集过？）')

    model, skill_sampler, base_sampler, ckpt = build_model()
    steps_per_epoch = len(samples) // BATCH_SIZE
    total_steps = int(steps_per_epoch * EPOCHS)
    logger.info(f'E18-KOD-SFT start: n={len(samples)} bs={BATCH_SIZE} micro={MICRO_BATCH} '
                f'lr={LR} epochs={EPOCHS} steps/epoch={steps_per_epoch} '
                f'total_steps={total_steps} eval_n={len(eval_records)} '
                f'exec_rollouts={EXEC_ROLLOUTS}@T{EXEC_TEMPERATURE} '
                f'gpus={TRAIN_GPUS}+{SKILL_SAMPLER_GPUS}+{BASE_SAMPLER_GPUS} out={OUTPUT_DIR}')

    # ---- 首次 eval（step 0，未训练的初始权重）----
    # ⭐ 不用 sync：skill_sampler 刚建立，拿的就是 MODEL_ID 的原始权重，与训练端同源。
    base_cache: Dict[str, float] = {}
    m_before = run_eval(skill_sampler, base_sampler, eval_records, base_cache, 'before', 0)
    log_eval(m_before, 'before', 0)

    log_path = os.path.join(OUTPUT_DIR, 'sft_log.jsonl')
    rng = random.Random(SEED)
    step = 0
    with open(log_path, 'a', encoding='utf-8') as log_fh:
        epoch = 0
        while step < total_steps:
            order = list(range(len(samples)))
            rng.shuffle(order)          # 每个 epoch 重洗，种子固定所以可复现
            for bi in range(steps_per_epoch):
                if step >= total_steps:
                    break
                batch = [samples[j] for j in order[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]]
                trajs = make_trajs(batch)
                micro = max(TRAIN_DP, min(MICRO_BATCH, len(trajs)))
                t_step = time.time()
                for i in range(0, len(trajs), micro):
                    model.forward_backward(inputs=trajs[i:i + micro])
                model.clip_grad_and_step()
                step += 1
                row = {'step': step, 'epoch': epoch, 'run': RUN_ID,
                       'n_samples': len(batch), 'seconds': round(time.time() - t_step, 2)}
                row.update(step_metrics(model))
                log_fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                log_fh.flush()
                if step % LOG_EVERY_STEPS == 0:
                    logger.info('[s%d/%d ep%d] ' % (step, total_steps, epoch)
                                + ' '.join(f'{k}={v:.4g}' for k, v in row.items()
                                           if isinstance(v, float)))
                if SAVE_EVERY_STEPS and step % SAVE_EVERY_STEPS == 0:
                    # ⭐ API 是 `model.save(tag, output_dir=)`，不是 save_checkpoint(path)
                    # —— 与在线版 e18_rejection_sft.py:762 的用法一致。
                    model.save(f'KODSFT-s{step}', output_dir=OUTPUT_DIR)
                    logger.info(f'[save] KODSFT-s{step}')
            epoch += 1

    model.save('KODSFT-final', output_dir=OUTPUT_DIR)

    # ---- 尾次 eval（训练后权重）----
    # ⭐ 必须先 sync_weights 把训好的权重推给 skill_sampler，否则这次 eval 跑的还是初始
    # 权重 —— 两次结果几乎相同，看起来像「训了没效果」，而真因是权重根本没上去。
    # merge_and_sync=True：全参数训练需要先在 dp 间 merge 再推。
    # reset_prefix_cache 必须跟着走：prefix cache 里是旧权重算出的 KV，不清就会拿旧 KV
    # 拼新权重的输出，得到一个既不是训前也不是训后的嵌合态。
    ckpt.sync_weights(merge_and_sync=True)
    skill_sampler.reset_prefix_cache()
    # base_cache 沿用：executor 全程未动，baseline 不需重算（也不应重算，见 run_eval）。
    m_after = run_eval(skill_sampler, base_sampler, eval_records, base_cache, 'after', step)
    log_eval(m_after, 'after', step)

    d_lift = m_after['eval/lift'] - m_before['eval/lift']
    logger.info('[result] lift %.4f -> %.4f (Δ %+.4f)  accuracy %.4f -> %.4f  baseline %.4f'
                % (m_before['eval/lift'], m_after['eval/lift'], d_lift,
                   m_before['eval/accuracy'], m_after['eval/accuracy'],
                   m_after['eval/baseline_accuracy']))
    # ⭐ 噪声底提醒：EXEC_ROLLOUTS=1 + T=0 下单题 pass_rate 是 0/1，n=100 的均值标准误约
    # 0.05，首尾差值的噪声更大。不把这句写进日志，很容易把 ±0.05 的漂动当成结论。
    if abs(d_lift) < 0.07:
        logger.warning(f'[result] Δlift {d_lift:+.4f} 在噪声量级内（n={len(eval_records)}、'
                       f'rollout=1、T=0 时约 ±0.07），不足以断定有/无效果。')
    logger.info(f'[done] steps={step} 用时 {(time.time() - t0) / 60:.1f} 分钟 -> {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
