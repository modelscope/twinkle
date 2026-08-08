"""E18 的多轨迹诊断：给**每一条失败的 rollout** 各诊一次，再合并成一份 rubric。

为什么需要它（E23 的 `RubricCache.get_or_diagnose` 不能直接复用）：
  1. 它的缓存键是 `RUBRIC_VERSION + _PROMPT_KEY + data_id`，**不含失败轨迹内容**。同一题调
     N 次会全部命中第一次的结果 —— 想「每条 rollout 判一次」在那个键下是做不到的。
  2. 它一题只产出一个决定性根因，且 `_format` 把 secondary 渲染成
     `ALSO OFF (do not write about these)`，**主动禁止** skill-gen 覆盖第二处。
     实测 E23 有 73% 的零 reward 组正是「修对第一处、挂在第二处」。

本模块只做加法，不改 `e23_rubric.py`（它是 E18/E23 共用、必须逐字同源的判分/诊断层）：
复用其 `diag_query` / `diag_segment` / `_validate` / `_CLASS_SHORT`，但换一个**按失败内容**
分桶的缓存键，并自己渲染合并文本。

⭐ 去重按 `(kind, 报错签名)` 而不是按 rollout 逐条：8 次采样里常有 5 次是同一个 assertion，
逐条诊断纯属浪费 API。同一签名只诊一次，但记下它出现了几次（`n_seen`），合并时按频次排序 ——
出现 5 次的根因显然比只出现 1 次的更该先修。
"""
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from twinkle import get_logger

import e23_rubric as R

logger = get_logger()

# 每题最多诊断几种**不同**的失败签名。3 与 e23_rubric 的 MAX_INDEPENDENT_CAUSES 一致：
# 超过 3 个独立根因的题，教师侧本来就判为不可救（诊断会互相矛盾，skill 也写不下）。
MAX_DIAG_PER_TASK = int(os.environ.get('MAX_DIAG_PER_TASK', 3))
# 一个签名至少要出现几次才值得诊断。默认 1 = 全诊；设 2 可以过滤掉只出现一次的偶发错误。
MIN_SIGNATURE_COUNT = int(os.environ.get('MIN_SIGNATURE_COUNT', 1))
# 并行度按**题**算：24 个线程各认领一道题，题内的多个签名仍串行，所以同时在飞的 HTTP
# request 就是 24。默认从 8 提到 24：纯 API 等待不占 GPU也不占 CPU，8 并行下一个
# 64 题 chunk 的诊断阶段实测要 ~26 分钟，而这段时间 8 张卡全部空转。
# 上限受教师侧限流约束，碰到 429 就把这个值调回。
DIAG_WORKERS = int(os.environ.get('RUBRIC_WORKERS', 24))

# 失败签名：只取**结构化字段**（哪个测试挂了 + 什么异常类），不用报错正文。
#
# ⭐ 为何不对报错正文做归一化（前一版的做法）：那靠的是「把数字/路径/引号内容逐个替成
# 占位符」，而异常消息里的变量形式永远枚不完（裸数值、列宽对齐的空白、repr 片段……）；
# 漏一个，同一个 bug 就被当成 N 个不同根因，白花 N 倍教师 API 且 N 条诊断在说同一件事。
# 现在只依赖两个结构化信号，行为可预测，不随报错排版变化。
#
# 两个信号都来自 unittest 的固定输出格式，且 e23_bcb._trim_err 保证保留（它只留
# FAIL:/ERROR:/Traceback/异常类名/带 ', in ' 的帧行）：
#   * 失败的测试方法名 —— 区分「同一异常但挂在不同测试上」（那是不同根因）；
#   * 异常类名 —— 区分 KeyError / AttributeError / AssertionError。
# 只用 kind + 异常类会把前者错误合并，所以两个都要。
_TEST_RE = re.compile(r'^(?:FAIL|ERROR):\s*(\w+)', re.M)
_EXC_RE = re.compile(r'^([A-Za-z_][\w.]*(?:Error|Exception|Warning))\b', re.M)


def _signature(roll: Dict[str, Any]) -> str:
    """失败轨迹 -> 稳定签名 = kind + 失败测试名集合 + 异常类集合。

    集合都排序去重，所以「两个测试挂了」不会因报错顺序不同而分成两个签名。

    ⭐ 拿不到任何结构化信号时（如 kind='timeout' / 'no_code'，根本没跑到 unittest），
    两个集合都为空，签名退化成单独的 kind —— 这正是想要的：同一题的 8 次超时是
    同一件事，只该诊一次。
    """
    kind = str(roll.get('kind') or 'unknown')
    err = str(roll.get('error') or '')
    tests = ','.join(sorted(set(_TEST_RE.findall(err))))
    excs = ','.join(sorted(set(_EXC_RE.findall(err))))
    return f'{kind}\x00{tests}\x00{excs}'


def bucket_failures(rolls: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
    """把一题的 rolls 按失败签名分桶，返回 [(代表 roll, 出现次数), ...]，按次数降序。

    只取失败的 roll。代表 roll 用该桶里第一条 —— 同签名意味着同一组 (测试, 异常类)，
    但报错正文仍可能略有差异（具体数值）；教师看到的是这条代表的完整报错，信息不丢。
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rolls:
        if r.get('correct'):
            continue
        sig = _signature(r)
        b = buckets.get(sig)
        if b is None:
            buckets[sig] = {'roll': r, 'n': 1}
        else:
            b['n'] += 1
    out = [(b['roll'], b['n']) for b in buckets.values()]
    out.sort(key=lambda t: -t[1])          # 高频根因优先
    return out


class MultiDiagCache:
    """按 (data_id, 失败签名) 缓存单条诊断；磁盘格式与 e23_rubric 的缓存文件同构但独立成文件。

    独立文件的理由：键的语义不同（这里含失败签名），混进同一个文件会让 E23 的缓存读取拿到
    对不上的条目。E23 那份缓存**完全不动**，两个实验各自可复现。
    """

    def __init__(self, path: str = None):
        self.path = path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'multidiag_cache_code_v1.jsonl')
        self._idx: Dict[str, Any] = {}
        import collections
        self.stats: collections.Counter = collections.Counter()
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._idx[rec['key']] = rec['value']
                    except Exception:
                        continue
            logger.info(f'[multidiag] 缓存载入 {len(self._idx)} 条：{self.path}')
        self._fh = open(self.path, 'a', encoding='utf-8')
        # ⭐ _put 会被 DIAG_WORKERS 个线程并发调用，write + flush 两步不是原子的：无锁时
        # 两条记录会交错成半行，下次启动 json.loads 解不开就默默丢掉（except: continue），
        # 表现是「明明诊过却反复花 API 钱」且无任何报错。并行度提到 24 后这个概率不再可忽。
        self._lock = threading.Lock()

    def _put(self, key: str, value: Any) -> None:
        line = json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n'
        with self._lock:
            self._idx[key] = value
            self._fh.write(line)
            self._fh.flush()

    def _one(self, checker, record: Dict[str, Any], roll: Dict[str, Any]) -> Optional[Dict]:
        """诊断单条失败轨迹，返回 validate 过的 diag dict（不可救/失败返回 None）。

        与 e23_rubric.get_or_diagnose 的差别只有缓存键：这里把失败签名放进键，所以同一题的
        不同失败模式各占一个槽位。校验逻辑直接复用 R._validate，保持判据完全一致。
        """
        sig = _signature(roll)
        key = hashlib.md5(
            f"{R.RUBRIC_VERSION}\x00{R._PROMPT_KEY}\x00"
            f"{record.get('data_id', '')}\x00{sig}".encode('utf-8')).hexdigest()
        if key in self._idx:
            cached = self._idx[key]
            if not cached:
                self.stats['hit_dropped'] += 1
                return None
            self.stats['hit'] += 1
            return cached
        query = R.diag_query(record['problem'], record['reference_answer'])
        try:
            obj = checker.classify(query, R.diag_segment(roll))
        except Exception as exc:
            logger.warning(f'[multidiag] classify error: {exc}')
            obj = None
        if obj is None:
            # 同 e23_rubric：API 故障**绝不缓存**，否则一次抖动会永久丢掉这个失败模式。
            self.stats['api_fail'] += 1
            return None
        diag = R._validate(obj, query)
        if diag is None:
            self._put(key, None)           # 稳定判决：这个失败模式教师给不出可用分类
            self.stats['dropped_unaddressable'] += 1
            return None
        self.stats['ok'] += 1
        self.stats[f"class_{diag['class']}"] += 1
        self._put(key, diag)
        return diag

    def diagnose_task(self, checker, record: Dict[str, Any],
                      rolls: List[Dict[str, Any]]) -> str:
        """一题的多失败模式诊断 -> 合并后的 rubric 文本（无可用诊断返回 ''）。"""
        buckets = [(r, n) for r, n in bucket_failures(rolls) if n >= MIN_SIGNATURE_COUNT]
        if not buckets:
            return ''
        buckets = buckets[:MAX_DIAG_PER_TASK]
        diags = []
        for roll, n in buckets:
            d = self._one(checker, record, roll)
            if d:
                diags.append((d, n))
        return merge_diags(diags, n_rollouts=len(rolls))

    def diagnose_many(self, checker,
                      jobs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]) -> List[str]:
        """并行版。jobs = [(record, rolls), ...]，返回对齐的 rubric 文本列表。

        并行度按**题**而不是按签名：一个线程认领一道题，题内的最多
        MAX_DIAG_PER_TASK 次 classify 仍串行，所以同时在飞的 request 数 = min(DIAG_WORKERS, 题数)。
        题数通常远大于 DIAG_WORKERS，所以实际并行就是 DIAG_WORKERS。

        不把签名也展平成任务（那会再快 ~3 倍）的原因不是缓存安全 —— _put 已加锁；
        而是签名数预先不知道，展平后难以把结果按题对齐回去，且同题的多条诊断本身
        就要合并。纯 API 等待不占 GPU。
        """
        if not jobs:
            return []
        workers = max(1, min(DIAG_WORKERS, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda j: self.diagnose_task(checker, j[0], j[1]), jobs))

    def close(self):
        self._fh.close()


def merge_diags(diags: List[Tuple[Dict[str, Any], int]], n_rollouts: int = 0) -> str:
    """把多条诊断拼成一份给 skill-gen 的文本。

    ⭐ 与 e23_rubric._format 的两处关键差别：
      1. **不再输出** `ALSO OFF (do not write about these)`。那条禁令是 E23 单根因口径的产物，
         而本模块的全部目的就是让 skill 覆盖多处根因 —— 留着它会自相矛盾。
      2. 带上 `seen k/M times` 频次。skill-gen 据此知道哪个根因更普遍、该先写哪个；
         只翻车 1/8 次的偶发问题不该和翻车 5/8 次的主因同等对待。

    evidence 仍然**不进**文本（与 _format 一致）：它是单测报错原文，断言 diff 里带期望值，
    是最强的答案泄漏通道。
    """
    if not diags:
        return ''
    if len(diags) == 1:
        d, n = diags[0]
        head = [f"DECISIVE FAILURE: {d['class']} — {R._CLASS_SHORT[d['class']]}",
                f"WHAT WENT WRONG: {d['reason']}",
                f"PRIOR THAT WOULD HAVE PREVENTED IT: {d['prior']}"]
        if n_rollouts and n:
            head.insert(1, f'OBSERVED: this failure appeared in {n}/{n_rollouts} attempts.')
        return '\n'.join(head)
    lines = [f'The attempt failed in {len(diags)} distinct ways across {n_rollouts} attempts. '
             f'Address ALL of them — fixing only the first will still fail the tests.']
    for i, (d, n) in enumerate(diags, 1):
        freq = f' (seen {n}/{n_rollouts})' if n_rollouts else ''
        lines.append(
            f"\nFAILURE {i}: {d['class']} — {R._CLASS_SHORT[d['class']]}{freq}"
            f"\n  WHAT WENT WRONG: {d['reason']}"
            f"\n  PRIOR THAT WOULD HAVE PREVENTED IT: {d['prior']}")
    return '\n'.join(lines)


def multidiag_metrics(stats) -> Dict[str, float]:
    """缓存/诊断计数 -> train_log 指标（与 e23_rubric.cache_metrics 同风格）。"""
    tot = max(1, stats['hit'] + stats['ok'] + stats['hit_dropped']
              + stats['dropped_unaddressable'] + stats['api_fail'])
    return {'rubric/hit_rate': (stats['hit'] + stats['hit_dropped']) / tot,
            'rubric/ok': float(stats['ok']),
            'rubric/dropped_unaddressable': float(stats['dropped_unaddressable']),
            'rubric/api_fail': float(stats['api_fail'])}
