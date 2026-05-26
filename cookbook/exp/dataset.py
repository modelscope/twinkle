import hashlib
import httpx
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from modelscope import dataset_snapshot_download

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import Preprocessor

dataset = Dataset()


def _hash_id(prefix: str, content: str) -> str:
    """Stable id from MD5 of content; collision-free for textual datasets."""
    return f'{prefix}__{hashlib.md5(content.encode("utf-8")).hexdigest()[:16]}'


def _register(processor_cls, meta: DatasetMeta, init_args: Optional[Dict[str, Any]] = None) -> None:
    """Add dataset and run preprocessor; auto-strip every input column to enforce
    the universal ``{id, source, messages}`` output schema."""
    dataset.add_dataset(meta)
    cols = list(dataset.datasets[meta.get_id()].column_names)
    dataset.map(
        processor_cls,
        dataset_meta=meta,
        init_args=init_args or {},
        remove_columns=cols,
        load_from_cache_file=True,
    )


# ===== MuSiQue =====
MUSIQUE_REPO = 'voidful/MuSiQue'


class MusiqueProcessor(Preprocessor):
    """MuSiQue raw row → multiple ``{id, source, messages}`` rows, one per paragraph."""

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            if row.get('answerable') is False:
                continue
            parent = str(row.get('id', ''))
            for idx, p in enumerate(row.get('paragraphs') or []):
                text = (p.get('paragraph_text') or '').strip()
                if not text:
                    continue
                out.append({
                    'id': f'musique__{parent}__{idx}',
                    'source': 'musique',
                    'messages': [{'role': 'user', 'content': text}],
                })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


# Repo 仅含原始 JSONL 无 HF 元数据，必须先快照下载再以文件路径注册。
_musique_jsonl = Path(dataset_snapshot_download(MUSIQUE_REPO)) / 'musique_ans_v1.0_train.jsonl'
if not _musique_jsonl.is_file():
    raise FileNotFoundError(f'MuSiQue raw file not found: {_musique_jsonl}')
_register(MusiqueProcessor, DatasetMeta(str(_musique_jsonl), data_slice=range(3000)))


# ===== swift/github-code =====
GITHUB_CODE_REPO = 'ms://swift/github-code'


class GithubCodeProcessor(Preprocessor):
    """github-code row → ``{id, source, messages}``；按代码长度均匀采样。

    把 ``[length_min, length_max)`` 切 ``n_buckets`` 桶，每桶配额 ``target/n_buckets``，
    桶满或超界即丢；近似得到 ``target`` 条且长度均匀分布的样本。
    依赖 batched map 单进程下实例状态跨 batch 共享（``num_proc>1`` 会失效）。
    """

    def __init__(self, target: int = 60000, length_min: int = 500,
                 length_max: int = 40000, n_buckets: int = 30):
        self.length_min = length_min
        self.length_max = length_max
        self.n_buckets = n_buckets
        self.bucket_quota = max(1, target // n_buckets)
        self.bucket_count = [0] * n_buckets

    def _bucket(self, n: int) -> int:
        if n < self.length_min or n >= self.length_max:
            return -1
        idx = int((n - self.length_min) / (self.length_max - self.length_min) * self.n_buckets)
        return min(idx, self.n_buckets - 1)

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            code = row.get('code') or ''
            if not isinstance(code, str):
                continue
            b = self._bucket(len(code))
            if b < 0 or self.bucket_count[b] >= self.bucket_quota:
                continue
            self.bucket_count[b] += 1
            lang = row.get('language') or 'unknown'
            out.append({
                'id': _hash_id(f'github_code__{lang}', code),
                'source': 'github-code',
                'messages': [{'role': 'user', 'content': code}],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


_register(GithubCodeProcessor,
         DatasetMeta(dataset_id=GITHUB_CODE_REPO, subset_name='all-apache-2.0', split='train'))


# ===== modelscope/competition_math =====
COMPETITION_MATH_REPO = 'ms://modelscope/competition_math'


class MathProcessor(Preprocessor):
    """competition_math row → ``{id, source, messages}`` (user/assistant pair)."""

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            problem = (row.get('problem') or '').strip()
            solution = (row.get('solution') or '').strip()
            if not problem or not solution:
                continue
            out.append({
                'id': _hash_id('math', f'{problem}\n{solution}'),
                'source': 'competition_math',
                'messages': [
                    {'role': 'user', 'content': problem},
                    {'role': 'assistant', 'content': solution},
                ],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


_register(MathProcessor,
          DatasetMeta(dataset_id=COMPETITION_MATH_REPO, subset_name='default', split='train'))


# ===== nampdn-ai/tiny-textbooks =====
TINY_TEXTBOOKS_REPO = 'ms://AI-ModelScope/tiny-textbooks'


class TinyTextbooksProcessor(Preprocessor):
    """tiny-textbooks row → ``{id, source, messages}`` (user/assistant pair)."""

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            text = (row.get('text') or '').strip()
            textbook = (row.get('textbook') or '').strip()
            if not text or not textbook:
                continue
            out.append({
                'id': _hash_id('tinytb', f'{text}\n{textbook}'),
                'source': 'tiny-textbooks',
                'messages': [
                    {'role': 'user', 'content': text},
                    {'role': 'assistant', 'content': textbook},
                ],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


_register(TinyTextbooksProcessor,
          DatasetMeta(dataset_id=TINY_TEXTBOOKS_REPO, split='train', data_slice=range(60000)))


# ===== Multi-turn ``messages`` datasets (Toucan, SWE-smith) =====


class MessagesNormalizeProcessor(Preprocessor):
    """Normalize multi-turn ``messages`` row → ``{id, source, messages}``。

    丢弃 system 消息；把 OpenAI 多模态 list-content 拼成纯文本；过滤空消息行。
    """

    def __init__(self, source: str):
        self.source = source

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            messages = row.get('messages')
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (ValueError, TypeError):
                    continue
            if not isinstance(messages, list):
                continue
            normalized: List[Dict[str, str]] = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get('role') or ''
                if role == 'system':
                    continue
                content = m.get('content')
                if isinstance(content, list):
                    content = '\n'.join(p.get('text', '') if isinstance(p, dict) else str(p)
                                        for p in content)
                if content is None:
                    content = ''
                if not isinstance(content, str):
                    content = str(content)
                if not content.strip():
                    continue
                normalized.append({'role': role, 'content': content})
            if not normalized:
                continue
            blob = ''.join(f'{m["role"]}:{m["content"]}' for m in normalized)
            out.append({
                'id': _hash_id(self.source, blob),
                'source': self.source,
                'messages': normalized,
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


_register(MessagesNormalizeProcessor,
          DatasetMeta(dataset_id='ms://Agent-Ark/Toucan-1.5M', subset_name='Kimi-K2', split='train', data_slice=range(30000)),
          init_args={'source': 'toucan'})


_register(MessagesNormalizeProcessor,
          DatasetMeta(dataset_id='ms://SWE-bench/SWE-smith-trajectories', split='tool', data_slice=range(30000)),
          init_args={'source': 'swe-smith'})

dataset.mix_dataset(False)
