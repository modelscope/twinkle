# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rubric double-cache (skill_quality_analysis.md #4, #17).

Two cache scopes, both wrapping v2's ``DiskCache`` (append-only jsonl, in-memory index):

- GlobalRubricCache (key = data_id): the RL / SFT lines diagnose the BARE-PROBLEM greedy
  trajectory. Because the executor is frozen at T=0, that trajectory is deterministic and
  identical across experiments, so its rubric diagnosis can be shared across ALL runs via one
  global file (``rubric_cache_global.jsonl``) — diagnose each problem once, reuse everywhere.
  ⚠️ 该共享前提只在"executor 口径完全相同"时成立。轨迹本身**不在键里**，所以任何改变裸解
  轨迹的开关（task 域、executor thinking 开关、以后若改 executor 模型）都必须体现在**文件名**
  上，否则新口径的 run 会命中旧口径的诊断（键只有 data_id，必然命中）。实测这个文件已经攒了
  2630 条 think 轨迹的数学诊断 —— E19（executor nothink）若共用会几乎全程读到与自己失败无关
  的诊断，而 rubric 内容正是该臂唯一的自变量。见 build_rubric_cache 的 tag 拼装。

- LocalRubricCache (key = md5(data_id + skill)): the improve-skill+SFT / OPSD lines diagnose a
  WITH-SKILL trajectory whose skill evolves with the policy, so the diagnosis is experiment-
  and step-specific and lives only in that experiment's directory.

Both reuse v2's ``_diagnose_entry`` (pure teacher-API call, no GPU) for the actual diagnosis,
so there is a single source of truth for the rubric prompt / parsing.
"""
import os
from typing import Any, Dict, Optional

import train_skill_v2 as v2
from train_skill_v2 import DiskCache, _diagnose_entry


def _version() -> str:
    """动态读 v2._RUBRIC_VERSION —— 不能 from-import：v2.set_task('code') 会在运行时把它换成
    code 判据的版本号，而 from-import 会把加载那一刻的值钉死，导致代码域诊断用数学域的键。"""
    return v2._RUBRIC_VERSION


class _BaseRubricCache:
    """Shared get-or-diagnose logic over a DiskCache; subclasses define the key."""

    def __init__(self, path: str, enabled: bool = True):
        self._cache = DiskCache(path, enabled)

    def _key(self, entry: Dict[str, Any], skill: Optional[str]) -> str:
        raise NotImplementedError

    def get(self, entry: Dict[str, Any], skill: Optional[str] = None) -> Optional[str]:
        return self._cache.get(self._key(entry, skill))

    def get_or_diagnose(self, entry: Dict[str, Any], checker,
                        skill: Optional[str] = None) -> str:
        """Return cached diagnosis, else run the teacher rubric once and cache it.

        ``entry`` must carry the fields ``_diagnose_entry`` needs: ``problem``,
        ``fail_segment``, ``fail_stop_reason`` (and ``reference_answer`` is unused by the
        diagnosis but kept for auditing). Returns '' when there is no checker or on API error
        (never raises), so the caller can treat "no diagnosis" uniformly.
        """
        if checker is None:
            return ''
        key = self._key(entry, skill)
        hit = self._cache.get(key)
        # bugfix #1: 旧版把 API 失败（_diagnose_entry 返回 None）也以 '' 永久写进缓存，
        # 一次瞬时抖动会跨实验毒化全局缓存且永不重试。现在：只缓存真诊断；历史残留的
        # '' 条目视为 miss，下次调用自动重试并用真诊断覆盖。
        if hit:
            return hit
        diag = _diagnose_entry(checker, entry)
        if diag is None:  # transient API failure: do NOT cache, retry on the next call
            return ''
        self._cache.put(key, diag)
        return diag

    def put(self, entry: Dict[str, Any], diag: str, skill: Optional[str] = None) -> None:
        self._cache.put(self._key(entry, skill), diag)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def close(self) -> None:
        self._cache.close()


class GlobalRubricCache(_BaseRubricCache):
    """key = (rubric 版本, data_id): bare-problem trajectory diagnosis, shareable across experiments.

    版本号必须进键：该文件跨实验共享且 append-only，一旦判据表改了而键不变，旧
    taxonomy 的诊断会被静默当成新判据的结果返回（旧版本号仅定义未使用）。
    """

    def _key(self, entry: Dict[str, Any], skill: Optional[str] = None) -> str:
        return DiskCache.key_for('rubric_global', _version(),
                                 str(entry.get('data_id', '')))


class LocalRubricCache(_BaseRubricCache):
    """key = md5(rubric 版本 + data_id + skill): with-skill trajectory diagnosis, per-experiment only."""

    def _key(self, entry: Dict[str, Any], skill: Optional[str] = None) -> str:
        return DiskCache.key_for('rubric_local', _version(),
                                 str(entry.get('data_id', '')), skill or '')


def build_rubric_cache(scope: str, output_dir: str, global_dir: Optional[str] = None,
                       enabled: bool = True, task: str = 'math', executor_thinking: str = 'on'):
    """Factory: scope='global' -> shared file under ``global_dir`` (default output_dir/..);
    scope='local' -> per-experiment file under ``output_dir/cache``.

    ``task`` 进文件名：代码域与数学域的诊断内容完全不同源（判据表、judge prompt、segment 里
    有没有单测报错），版本号已经能隔开键，分文件是第二道保险，也让缓存体积可分别管理。

    ★ ``executor_thinking`` 也必须进文件名（2026-07-31 bugfix）：global 缓存跨实验共享的**唯一
    依据**是"executor 冻结在 T=0，所以同一道题的裸解轨迹在所有实验里逐字相同"。E19/E20 把
    executor 的 thinking 关掉后这个前提就不成立了 —— 裸解轨迹完全变了（think 那边大量是
    "撞预算没写出代码"，nothink 这边是"写完但答错"），而诊断正是对着这条轨迹做的。共用一个
    文件会让 nothink 臂直接读到 think 臂的旧诊断（键只有 data_id，必然命中），
    rubric 内容与本臂的真实失败无关 —— 而 rubric 内容恰恰是这两个臂唯一的自变量。
    """
    tag = '' if task == 'math' else f'_{task}'
    if executor_thinking != 'on':
        tag += '_execnothink'
    if scope == 'global':
        base = global_dir or os.path.dirname(os.path.abspath(output_dir.rstrip('/')))
        os.makedirs(base, exist_ok=True)
        return GlobalRubricCache(os.path.join(base, f'rubric_cache_global{tag}.jsonl'), enabled)
    if scope == 'local':
        cache_dir = os.path.join(output_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return LocalRubricCache(os.path.join(cache_dir, f'rubric_cache_local{tag}.jsonl'), enabled)
    raise ValueError(f"scope must be 'global' or 'local', got {scope!r}")
