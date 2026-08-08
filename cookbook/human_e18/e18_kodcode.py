# -*- coding: utf-8 -*-
"""KodCode-V1 数据域适配：与 `e23_bcb.py` **同契约**的加载 + 沙箱判分。

为什么另起一个文件而不改 e23_bcb：BCB 与 KodCode 的单测框架不同（unittest vs pytest）、
题面/入口点的来源字段也不同，但**对上层的接口必须逐字一致** —— `e18_rejection_sft.py`
只认 `load_records / judge_seqs / empty_roll / run_tests` 这组签名和 `{'data_id',
'problem', 'reference_answer'}` 这个记录形状。保持接口一致，换域时上层零改动。

与 e23_bcb 的对齐点（改任何一处都会让两域的 pass_rate 不可比）：
* `run_tests` 返回 `{'passed', 'kind', 'error'}`，`kind` 取值集合完全相同：
  `pass / no_code / no_entry / timeout / assertion / exception / import_or_syntax`。
  `e18_multidiag._signature` 拿 kind 做失败签名，取值不一致会让诊断缓存串味。
* `judge_seqs` 同 `(task_id, code)` 只判一次、线程池并发、返回 roll 的字段集相同。
* `_trim_err` 只保留失败测试名与异常行，并把随机临时目录名归一化成 `<sandbox>` ——
  否则同一个失败在两次运行里字符串不同，`_signature` 会算出两个签名、缓存永远不命中。
* 参考解答跑不过自己单测的题一律剔除（BCB 实测 ~7.5%，KodCode 实测 ~5%），
  自检结果落盘缓存。这类题不是模型的错，留着会把 base_pass_rate 永久压低。

KodCode 特有的两点：
1. 单测是 pytest 风格且 **181/200 靠 `from solution import X`** 取被测函数，所以沙箱里必须
   把提交代码写成 `solution.py`（而不是 BCB 那样把代码和 test 拼进同一个文件）。
2. 自带 `gpt_pass_percentage`（教师多次尝试的通过率），是**免费的先验难度**。E18 原先要靠
   跑 8 次 bare rollout 才能找出难题，这里可以直接按阈值筛，省掉这部分 GPU。
"""
import ast as _ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from twinkle import get_logger
from twinkle.dataset import Dataset, DatasetMeta

logger = get_logger()

# ========== 配置 ==========
KOD_DATASET = os.environ.get('KOD_DATASET', 'ms://AI-ModelScope/KodCode-V1')
KOD_SUBSET = os.environ.get('KOD_SUBSET', 'default')
KOD_SPLIT = os.environ.get('KOD_SPLIT', 'train')
# 难度窗口：只留「教师也常做错、但并非无解」的题。
# 上界 0.3 -> executor 大概率失败（有诊断可采）；下界 >0 -> 排除疑似不可解。
KOD_MAX_PASS_PCT = float(os.environ.get('KOD_MAX_PASS_PCT', 0.3))
KOD_MIN_PASS_PCT = float(os.environ.get('KOD_MIN_PASS_PCT', 0.0))
KOD_MAX_TASKS = int(os.environ.get('KOD_MAX_TASKS', 0))       # 0 = 不截断
# ⭐ 默认关沙箱自检：全量 73747 题跑一遍参考解答要 ~30 小时（子进程）。
# 关掉的代价：参考解答自己都跑不过单测的坏题（实测 ~12.5%）会留在题池里，
# 但它们在采集时会自然显形 —— base_pass_rate 恒为 0、且任何 skill 都拿不到
# +MIN_PASS_GAIN，于是 select_winner 返回 None、不入池。只浪费 rollout，不污染数据集。
KOD_SELFCHECK = os.environ.get('KOD_SELFCHECK', '0') == '1'
# ⭐ TEST_WORKERS 默认 96，而不是继承 BCB 的 24：判分是子进程，与 GPU 采样**串行**，
# 每 chunk 要跑 CHUNK_SIZE*(BARE_ROLLOUTS + N_SKILLS*EXEC_ROLLOUTS) ≈ 2300 次，并发不够就直接拆 GPU 空转。
# BCB 用 24 是因为它的单测要 import pandas/sklearn/matplotlib（单次 1-3s、内存大）；
# KodCode 是纯算法题，单测 0.05-0.3s、几乎不导包，可以开得高得多。
# 上限卡在 min(96, 核数一半)：留余量给 vLLM 的调度/集合线程，别把宿主打满反而拖慢采样。
TEST_WORKERS = int(os.environ.get(
    'TEST_WORKERS', max(24, min(96, (os.cpu_count() or 24) // 2))))
TEST_TIMEOUT = int(os.environ.get('TEST_TIMEOUT', 60))

_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


# ========== 文本处理（与 e23_bcb 逐字一致） ==========
def after_think(text: str) -> str:
    """只取 </think> 之后的正文；没有闭合标签就原样返回。"""
    idx = text.rfind('</think>')
    return text[idx + len('</think>'):] if idx >= 0 else text


def clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').strip()


def extract_code(text: str) -> str:
    """取最后一个完整的 ``` 代码块；没有围栏时退化成整段正文。

    取**最后一个**而不是第一个：模型常先给一版草稿再给最终版，最后一个才是它的结论。
    """
    body = after_think(text)
    blocks = _FENCE_RE.findall(body)
    if blocks:
        return blocks[-1].strip()
    # 没有围栏：可能是 nothink 直出代码。剔掉明显的自然语言行后返回。
    return body.strip()


def extract_skill(text: str) -> str:
    """取 <skills>...</skills> 里的内容；没有标签时返回空串（视为格式失败）。"""
    body = after_think(text)
    m = re.search(r'<skills>(.*?)</skills>', body, re.S | re.IGNORECASE)
    return m.group(1).strip() if m else ''


# ========== 沙箱判分 ==========
# ⭐ 纯 pytest 驱动，但用插件把 (n_tests, n_fail, n_err) 拿回来 —— 与 e23_bcb 的
# `__BCB__ n f e` 输出契约同形，好让 kind 的判定规则完全一致。
# 只统计 when=='call'：setup/teardown 阶段的失败算 error（多半是 import 不了 solution）。
#
# ⭐ 断言 vs 异常的区分必须走 `report.longrepr.reprcrash.message`，**不能**用
# `'AssertionError' in str(longrepr)`：pytest 默认开断言重写（assertion rewriting），
# 失败摘要长这样 `E  assert -1 == 3`，整段里根本没有 "AssertionError" 这个词，
# 于是所有断言失败都会被误判成 exception。实测踩过：断言不符返回了 kind='exception'。
# kind 错了会让 `e18_multidiag._signature` 的失败签名串味、rubric 缓存失效。
_RUNNER = r"""
import sys, pytest


class _Collect:
    def __init__(self):
        self.n_tests = self.n_fail = self.n_err = 0

    @staticmethod
    def _is_assertion(report):
        crash = getattr(getattr(report, 'longrepr', None), 'reprcrash', None)
        msg = getattr(crash, 'message', '') or ''
        # 断言重写后首行是 "assert ..."；未重写时是 "AssertionError: ..."。
        return msg.startswith('assert') or msg.startswith('AssertionError')

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.n_tests += 1
            if report.failed:
                if self._is_assertion(report):
                    self.n_fail += 1
                else:
                    self.n_err += 1
        elif report.failed:
            self.n_err += 1


c = _Collect()
rc = pytest.main(['-q', '--no-header', '-p', 'no:cacheprovider',
                  '--tb=short', 'test_solution.py'], plugins=[c])
print('__KOD__', c.n_tests, c.n_fail, c.n_err)
sys.exit(0 if int(rc) == 0 else 1)
"""


def _trim_err(err: str, limit: int = 1600) -> str:
    """保留失败测试名与异常行，砍掉冗长 traceback 帧 —— 这是喂给 rubric 的客观证据。
    随机临时目录名换成 <sandbox>，否则同一个失败在两次运行里看起来不一样
    （`e18_multidiag._signature` 会因此算出不同签名，诊断缓存永久不命中）。"""
    err = re.sub(r'/tmp/kod_[A-Za-z0-9_]+', '<sandbox>', err or '')
    lines = [ln for ln in err.splitlines() if ln.strip()]
    keep = [ln for ln in lines
            if ln.startswith(('FAILED', 'FAIL:', 'ERROR:', 'AssertionError', 'Traceback', 'E  '))
            or re.match(r'^\w*(Error|Exception|Warning)\b', ln.strip())
            or ', in ' in ln]
    return '\n'.join(keep or lines[-25:])[-limit:]


def run_tests(code: str, payload: Dict[str, Any], timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """子进程里跑「提交代码(solution.py) + 官方 test(test_solution.py)」。

    -> {'passed', 'kind', 'error'}，kind 取值与 e23_bcb.run_tests 完全一致。

    与 BCB 的唯一实质差异：代码单独落成 `solution.py`，因为 KodCode 的单测靠
    `from solution import X` 取被测函数，拼进同一个文件会 ImportError。
    """
    if not code.strip():
        return {'passed': False, 'kind': 'no_code', 'error': 'no parseable code block'}
    entry = payload.get('entry_point') or ''
    if entry and entry not in code:
        return {'passed': False, 'kind': 'no_entry',
                'error': f'function {entry} is not defined in the submitted code'}
    tmp = tempfile.mkdtemp(prefix='kod_')
    try:
        with open(os.path.join(tmp, 'solution.py'), 'w', encoding='utf-8') as f:
            f.write(code)
        with open(os.path.join(tmp, 'test_solution.py'), 'w', encoding='utf-8') as f:
            f.write(payload['test'])
        with open(os.path.join(tmp, '_run.py'), 'w', encoding='utf-8') as f:
            f.write(_RUNNER)
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        # cwd=tmp 让 `from solution import X` 能找到同目录的 solution.py。
        try:
            p = subprocess.run([sys.executable, '_run.py'], cwd=tmp, env=env, timeout=timeout,
                               capture_output=True, text=True, errors='replace')
        except subprocess.TimeoutExpired:
            return {'passed': False, 'kind': 'timeout',
                    'error': f'the tests did not finish within {timeout}s'}
        n_tests = n_fail = n_err = 0
        for line in (p.stdout or '').splitlines():
            if line.startswith('__KOD__'):
                _, a, b, c = line.split()
                n_tests, n_fail, n_err = int(a), int(b), int(c)
        if p.returncode == 0 and n_tests > 0:
            return {'passed': True, 'kind': 'pass', 'error': ''}
        kind = 'assertion' if n_fail else ('exception' if n_err else 'import_or_syntax')
        merged = ((p.stdout or '') + '\n' + (p.stderr or '')).replace(tmp, '<sandbox>')
        return {'passed': False, 'kind': kind, 'error': _trim_err(merged)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def empty_roll() -> Dict[str, Any]:
    return {'correct': False, 'stop_reason': 'empty', 'gen_tokens': 0, 'text': '', 'code': '',
            'kind': 'no_code', 'error': ''}


def judge_seqs(pairs: List[Tuple[Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """[(采样 sequence 或 None, payload)] -> rolls。所有判分都汇合到这里。

    必须批量：单测是子进程，一个 chunk 几百次判分串行会比同 chunk 的 GPU 时间还长一个量级。
    同 (task_id, code) 只跑一次 —— T=0 的 executor 经常对同一题产出逐字相同的代码。
    """
    rolls: List[Dict[str, Any]] = []
    keys: List[Optional[Tuple[str, str]]] = []
    jobs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for seq, payload in pairs:
        if seq is None:
            rolls.append(empty_roll())
            keys.append(None)
            continue
        text = clean_text(getattr(seq, 'decoded', '') or '')
        code = extract_code(text)
        key = (payload['task_id'], code)
        rolls.append({'correct': False, 'stop_reason': getattr(seq, 'stop_reason', None),
                      'gen_tokens': len(getattr(seq, 'tokens', None) or []),
                      'text': text, 'code': code, 'kind': None, 'error': ''})
        keys.append(key)
        jobs.setdefault(key, payload)
    if jobs:
        todo = list(jobs)
        with ThreadPoolExecutor(max_workers=max(1, min(TEST_WORKERS, len(todo)))) as ex:
            verdicts = dict(zip(todo, ex.map(lambda k: run_tests(k[1], jobs[k]), todo)))
        for roll, key in zip(rolls, keys):
            v = verdicts.get(key) if key is not None else None
            if v is not None:
                roll['correct'] = bool(v['passed'])
                roll['kind'], roll['error'] = v['kind'], v['error']
    return rolls


# ========== 数据 ==========
# reference_answer 的字段集：判分需要的一切。与 e23_bcb 的 _PAYLOAD_KEYS 同名对齐，
# 上层（e18_select / e18_multidiag / dump_dataset）拿到的 key 才一致。
_PAYLOAD_KEYS = ('task_id', 'entry_point', 'test', 'code_prompt', 'doc_struct',
                 'canonical_solution')


def _entry_point(row: Dict[str, Any]) -> str:
    """从 test_info 拿被测函数名。拿不到就回落到 test 里的 `from solution import X`。"""
    ti = row.get('test_info')
    if ti is not None:
        try:
            items = list(ti) if not isinstance(ti, str) else _ast.literal_eval(ti)
            for it in items:
                name = (it or {}).get('function_name')
                if name:
                    return str(name)
        except Exception:
            pass
    m = re.search(r'from\s+solution\s+import\s+([A-Za-z_]\w*)', row.get('test') or '')
    return m.group(1) if m else ''


def _code_prompt(row: Dict[str, Any]) -> str:
    """函数签名，用于给 executor 固定入口点（对齐 BCB 的 code_prompt 语义）。"""
    ti = row.get('test_info')
    if ti is not None:
        try:
            items = list(ti) if not isinstance(ti, str) else _ast.literal_eval(ti)
            for it in items:
                decl = (it or {}).get('function_declaration')
                if decl:
                    return str(decl)
        except Exception:
            pass
    return ''


def _usable(row: Dict[str, Any]) -> bool:
    """能进题池的最低门槛。

    要求 test 通过 `from solution import` 取函数：11.7% 的题直接裸调函数名，
    在「代码写进 solution.py」的沙箱布局下必然 NameError —— 那是 harness 不兼容，
    不是模型的错，留着会把 base_pass_rate 永久压低。
    """
    test = row.get('test') or ''
    if 'def test_' not in test:
        return False
    if not re.search(r'from\s+solution\s+import|import\s+solution\b', test):
        return False
    return bool((row.get('solution') or '').strip()) and bool(_entry_point(row))


# ⭐ 题面尾部必须追加函数签名：实测只有 **8%** 的 KodCode question 提到了被测函数名，
# 而单测靠 `from solution import <name>` 取函数。不补签名的后果：executor 把函数叫成
# 任何名字都算错，92% 的题无论 skill 好坏都是 0 分 —— pass_rate 全城 0、整个采集废掉。
# BCB 不需要这一步是因为它的 instruct_prompt 自带 `def task_func(...)` 骨架。
_SIG_HINT = ('\n\nYou should write self-contained code starting with:\n```\n{decl}\n```')


def _problem_text(row: Dict[str, Any]) -> str:
    """题面 = question + 函数签名（签名已在题面里就不重复追加）。"""
    q = row.get('question') or ''
    decl = _code_prompt(row)
    if not decl:
        return q
    if decl.strip() in q:
        return q
    return q + _SIG_HINT.format(decl=decl.strip())


def _to_record(batch: Dict[str, List]) -> Dict[str, List]:
    """原始 KodCode 列 -> {'data_id', 'problem', 'reference_answer'}。

    Dataset.map 强制 batched=True，所以这里收发的都是列式 batch。
    """
    n = len(batch['question_id'])
    rows = [{k: batch[k][i] for k in batch} for i in range(n)]
    return {
        'data_id': [str(r['question_id']) for r in rows],
        'problem': [_problem_text(r) for r in rows],
        'reference_answer': [{
            'task_id': str(r['question_id']),
            'entry_point': _entry_point(r),
            'test': r['test'],
            'code_prompt': _code_prompt(r),
            'doc_struct': '',
            'canonical_solution': r['solution'],
            # 教师先验难度：保留下来供离线分析（不参与判分）。
            'gpt_pass_percentage': float(r.get('gpt_pass_percentage') or 0.0),
            'gpt_difficulty': r.get('gpt_difficulty') or '',
        } for r in rows],
    }


def _broken_tasks(ds: Dataset, output_dir: str) -> set:
    """参考解答跑不过自己的单测 = 数据缺陷或沙箱不可判定，不是模型的错（实测约 5%）。
    自检一次后落盘缓存，题数不变则复用。必须在 map 之后调用（读的是 reference_answer）。"""
    path = os.path.join(output_dir, 'kod_broken_tasks.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                c = json.load(f)
            if int(c.get('n_tasks', -1)) == len(ds):
                logger.info(f'[data] 复用沙箱自检缓存：剔除 {len(c["broken"])} 道')
                return set(c['broken'])
        except Exception as exc:
            logger.warning(f'[data] 读取 {path} 失败（{exc}），重跑自检')
    logger.info(f'[data] 沙箱自检：{len(ds)} 道题跑参考解答（一次性，之后走缓存）…')
    rows = [ds[i] for i in range(len(ds))]
    jobs = [(r['reference_answer']['canonical_solution'], r['reference_answer']) for r in rows]
    with ThreadPoolExecutor(max_workers=max(1, min(TEST_WORKERS, len(jobs)))) as ex:
        vers = list(ex.map(lambda p: run_tests(p[0], p[1]), jobs))
    broken = {r['data_id'] for r, v in zip(rows, vers) if not v['passed']}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'n_tasks': len(ds), 'broken': sorted(broken)}, f, indent=1)
    logger.info(f'[data] 自检完成：剔除 {len(broken)}/{len(rows)} '
                f'({100.0 * len(broken) / max(1, len(rows)):.1f}%)')
    return broken


def load_records(seed: int, eval_size: int,
                 output_dir: str) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """-> (train_dataset, eval_records)，每条记录是 {'data_id', 'problem', 'reference_answer'}。

    签名与 `e23_bcb.load_records` 逐字一致，上层可直接换 import。

    与 BCB 的差异：KodCode 自带 `gpt_pass_percentage`，所以**先按难度窗口过滤**再自检 ——
    自检要跑一遍全部参考解答（子进程，很贵），先筛掉容易题能省掉大部分开销。
    """
    ds = Dataset(DatasetMeta(KOD_DATASET, subset_name=KOD_SUBSET, split=KOD_SPLIT))
    n_raw = len(ds)

    ds.filter(lambda r: KOD_MIN_PASS_PCT < float(r.get('gpt_pass_percentage') or 0.0)
              <= KOD_MAX_PASS_PCT)
    n_hard = len(ds)
    ds.filter(_usable)
    logger.info(f'[data] KodCode: 全集 {n_raw}，难度窗口 '
                f'({KOD_MIN_PASS_PCT}, {KOD_MAX_PASS_PCT}] 保留 {n_hard}、'
                f'harness 不兼容剔除 {n_hard - len(ds)} -> {len(ds)}')
    if KOD_MAX_TASKS and len(ds) > KOD_MAX_TASKS:
        # ⭐ 必须用 filter 而不是 `ds.dataset = ds.dataset.select(...)`：Dataset.map 内部读的是
        # `self.datasets`（未截断的副本）并回写 `self.dataset`，直接赋值 self.dataset 会在
        # 下一句 map 里被静默覆盖 —— 实测踩过：截断到 40 题后自检仍在跑 73747 题。
        keep = set(ds.dataset.shuffle(seed=seed)['question_id'][:KOD_MAX_TASKS])
        ds.filter(lambda r: r['question_id'] in keep)
        logger.info(f'[data] KOD_MAX_TASKS 截断 -> {len(ds)}')

    ds.map(_to_record, remove_columns=ds.dataset.column_names)

    broken = _broken_tasks(ds, output_dir) if KOD_SELFCHECK else set()
    if broken:
        ds.filter(lambda r: r['data_id'] not in broken)
        logger.info(f'[data] 剔除参考解答自己跑不过单测的题 {len(broken)} 道 -> 可用 {len(ds)}')
    elif not KOD_SELFCHECK:
        logger.info(f'[data] 跳过沙箱自检（KOD_SELFCHECK=0），题池 {len(ds)}；'
                    f'坏题会在采集时因 base_pass_rate=0 自然不入池')

    shuffled = ds.dataset.shuffle(seed=seed)
    n_eval = min(eval_size, len(shuffled)) if eval_size > 0 else 0
    eval_records = list(shuffled.select(range(n_eval)))
    train_dataset = Dataset(DatasetMeta(data=shuffled.select(range(n_eval, len(shuffled)))))
    return train_dataset, eval_records
