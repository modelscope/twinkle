import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import Features, Value
from modelscope import dataset_snapshot_download

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import Preprocessor

_TARGET_FEATURES = Features({
    'id': Value('string'),
    'source': Value('string'),
    'messages': [{'role': Value('string'), 'content': Value('string')}],
})


def _hash_id(prefix: str, content: str) -> str:
    """Stable id from MD5 of content; collision-free for textual datasets."""
    return f'{prefix}__{hashlib.md5(content.encode("utf-8")).hexdigest()[:16]}'


def _register(dataset, processor_cls, meta: DatasetMeta, init_args: Optional[Dict[str, Any]] = None,
              load_from_cache_file: bool = True) -> None:
    """Add dataset and run preprocessor; auto-strip every input column to enforce
    the universal ``{id, source, messages}`` output schema."""
    dataset.add_dataset(meta)
    cols = list(dataset.datasets[meta.get_id()].column_names)
    dataset.map(
        processor_cls,
        dataset_meta=meta,
        init_args=init_args or {},
        remove_columns=cols,
        load_from_cache_file=load_from_cache_file,
        features=_TARGET_FEATURES,
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
                    'messages': [{'role': 'assistant', 'content': text}],
                })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


# Repo 仅含原始 JSONL 无 HF 元数据，必须先快照下载再以文件路径注册。
_musique_jsonl = Path(dataset_snapshot_download(MUSIQUE_REPO)) / 'musique_ans_v1.0_train.jsonl'
if not _musique_jsonl.is_file():
    raise FileNotFoundError(f'MuSiQue raw file not found: {_musique_jsonl}')


# ===== swift/github-code =====
GITHUB_CODE_REPO = 'ms://swift/github-code'


class GithubCodeProcessor(Preprocessor):
    """github-code row → ``{id, source, messages}``；按代码长度均匀采样。

    把 ``[length_min, length_max)`` 切 ``n_buckets`` 桶，每桶配额 ``target/n_buckets``，
    桶满或超界即丢；近似得到 ``target`` 条且长度均匀分布的样本。
    依赖 batched map 单进程下实例状态跨 batch 共享（``num_proc>1`` 会失效）。
    """

    def __init__(self, target: int = 30000, length_min: int = 500,
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
                'messages': [{'role': 'assistant', 'content': code}],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


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
                    {'role': 'assistant', 'content': solution},
                ],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


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
                    {'role': 'assistant', 'content': textbook},
                ],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


# ===== Passage Explosion for Compression Distillation =====
# Each message content >= threshold becomes a standalone row: messages=[{role:user, content:X}]

_MIN_PASSAGE_LEN = 500  # CJK-equivalent units


def _effective_len(text: str) -> int:
    """CJK chars count double; threshold 500 ≈ 500 Chinese chars ≈ 1000 Latin chars."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f')
    return cjk * 2 + (len(text) - cjk)


def _extract_content(msg: dict) -> str:
    """Extract text content from a message dict, handling multimodal list-content."""
    content = msg.get('content')
    if isinstance(content, list):
        content = '\n'.join(
            p.get('text', '') if isinstance(p, dict) else str(p) for p in content)
    if not isinstance(content, str):
        return ''
    return content.strip()


class PassageExplodeProcessor(Preprocessor):
    """Explode multi-turn messages into individual long passages for compression distillation."""

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
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role') or ''
                if role == 'system':
                    continue
                content = _extract_content(msg)
                if not content or _effective_len(content) < _MIN_PASSAGE_LEN:
                    continue
                out.append({
                    'id': _hash_id(self.source, content),
                    'source': self.source,
                    'messages': [{'role': 'assistant', 'content': content}],
                })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


# ===== Reasoning / CoT datasets — explode query and assistant separately =====
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


class CotExplodeProcessor(Preprocessor):
    """Base for CoT datasets: explode query and full assistant content as separate passages."""

    def _extract_rows(self, rows: List[Dict[str, Any]]) -> List[tuple]:
        """Subclass returns list of (query, cot, response) tuples."""
        raise NotImplementedError

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows_list = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for query, cot, response, source in self._extract_rows(rows_list):
            if cot:
                response = _THINK_RE.sub('', response).strip()
            assistant_content = f'<think>{cot}</think>{response}' if cot else response
            for text in (query, assistant_content):
                if not text or _effective_len(text) < _MIN_PASSAGE_LEN:
                    continue
                out.append({
                    'id': _hash_id(source, text),
                    'source': source,
                    'messages': [{'role': 'assistant', 'content': text}],
                })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


# -- Chinese-DeepSeek-R1-Distill-data-110k --
CN_R1_DISTILL_REPO = 'ms://AI-ModelScope/Chinese-DeepSeek-R1-Distill-data-110k'


class ChineseR1DistillProcessor(CotExplodeProcessor):
    """input → query, reasoning_content → cot, content → response."""

    def _extract_rows(self, rows):
        for row in rows:
            query = (row.get('input') or '').strip()
            cot = (row.get('reasoning_content') or '').strip()
            response = (row.get('content') or '').strip()
            if not query or not response:
                continue
            yield query, cot, response, 'Chinese-DeepSeek-R1-Distill-data-110k'


# -- Opus-4.6-Reasoning-3000x-filtered --
OPUS_REASONING_REPO = 'ms://nohurry/Opus-4.6-Reasoning-3000x-filtered'


class OpusReasoningProcessor(CotExplodeProcessor):
    """problem → query, thinking → cot, solution → response."""

    def _extract_rows(self, rows):
        for row in rows:
            query = (row.get('problem') or '').strip()
            cot = (row.get('thinking') or '').strip()
            response = (row.get('solution') or '').strip()
            if not query or not response:
                continue
            yield query, cot, response, 'Opus-4.6-Reasoning-3000x-filtered'


# -- claude-opus-4.6-10000x --
CLAUDE_OPUS_REPO = 'ms://Roman1111111/claude-opus-4.6-10000x'


class ClaudeOpusProcessor(CotExplodeProcessor):
    """messages (OpenAI format) → extract user/assistant, split <think> or reasoning field."""

    def _extract_rows(self, rows):
        for row in rows:
            messages = row.get('messages')
            if not isinstance(messages, list):
                continue
            query = ''
            assistant_text = ''
            reasoning = ''
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role') or ''
                content = msg.get('content') or ''
                if not isinstance(content, str):
                    continue
                if role == 'user' and not query:
                    query = content.strip()
                elif role == 'assistant' and not assistant_text:
                    assistant_text = content.strip()
                    reasoning = (msg.get('reasoning') or '').strip()
                    break
            if not query or not assistant_text:
                continue
            cot = reasoning
            if not cot:
                m = _THINK_RE.search(assistant_text)
                if m:
                    cot = m.group(1).strip()
                    assistant_text = assistant_text[m.end():].strip()
            response = assistant_text if not reasoning else _THINK_RE.sub('', assistant_text).strip()
            if not response:
                continue
            yield query, cot, response, 'claude-opus-4.6-10000x'


# -- angrygiraffe-claude-opus-4.6-4.7-reasoning-8.7k --
ANGRYGIRAFFE_REPO = 'ms://hf/angrygiraffe-claude-opus-4.6-4.7-reasoning-8.7k'


class AngrygiraffeOpusReasoningProcessor(CotExplodeProcessor):
    """messages (OpenAI format) → extract first user/assistant, split <think> tag."""

    def _extract_rows(self, rows):
        for row in rows:
            messages = row.get('messages')
            if not isinstance(messages, list):
                continue
            query = ''
            assistant_text = ''
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role') or ''
                content = msg.get('content') or ''
                if not isinstance(content, str):
                    continue
                if role == 'user' and not query:
                    query = content.strip()
                elif role == 'assistant' and not assistant_text:
                    assistant_text = content.strip()
                    break
            if not query or not assistant_text:
                continue
            m = _THINK_RE.search(assistant_text)
            if m:
                cot = m.group(1).strip()
                response = assistant_text[m.end():].strip()
            else:
                cot = ''
                response = assistant_text
            if not response:
                continue
            yield query, cot, response, 'angrygiraffe-claude-opus-4.6-4.7-reasoning-8.7k'


_BASE_SIZES = {
    'tiny_textbooks': 10000,
    'musique': 1000,
    'github_code': 30000,
    'competition_math': 7500,
    'toucan': 10000,
    'swe_smith': 1000,
    'cn_r1_distill': 10000,
    'opus_reasoning': 3000,
    'claude_opus': 10000,
    'angrygiraffe': 20000,
}


def _scaled_sizes(total: Optional[int]) -> Dict[str, int]:
    if total is None:
        return dict(_BASE_SIZES)
    scale = total / sum(_BASE_SIZES.values())
    return {k: max(1, int(round(v * scale))) for k, v in _BASE_SIZES.items()}


def get_dataset(total: Optional[int] = None, load_from_cache_file: bool = True) -> Dataset:
    """Build the unified compression-distillation dataset.

    If ``total`` is given, every per-source row count in ``_BASE_SIZES`` is
    scaled proportionally so the input-row sum approximates ``total``.
    """
    sizes = _scaled_sizes(total)
    dataset = Dataset()

    _register(dataset, TinyTextbooksProcessor,
              DatasetMeta(dataset_id=TINY_TEXTBOOKS_REPO, split='train',
                          data_slice=range(sizes['tiny_textbooks'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, MusiqueProcessor,
              DatasetMeta(str(_musique_jsonl), data_slice=range(sizes['musique'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, GithubCodeProcessor,
              DatasetMeta(dataset_id=GITHUB_CODE_REPO, subset_name='all-apache-2.0', split='train'),
              init_args={'target': sizes['github_code']},
              load_from_cache_file=load_from_cache_file)

    _register(dataset, MathProcessor,
              DatasetMeta(dataset_id=COMPETITION_MATH_REPO, subset_name='default', split='train',
                          data_slice=range(sizes['competition_math'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, PassageExplodeProcessor,
              DatasetMeta(dataset_id='ms://Agent-Ark/Toucan-1.5M', subset_name='Kimi-K2', split='train',
                          data_slice=range(sizes['toucan'])),
              init_args={'source': 'toucan'},
              load_from_cache_file=load_from_cache_file)

    _register(dataset, PassageExplodeProcessor,
              DatasetMeta(dataset_id='ms://SWE-bench/SWE-smith-trajectories', split='tool',
                          data_slice=range(sizes['swe_smith'])),
              init_args={'source': 'swe-smith'},
              load_from_cache_file=load_from_cache_file)

    _register(dataset, ChineseR1DistillProcessor,
              DatasetMeta(dataset_id=CN_R1_DISTILL_REPO, split='train',
                          data_slice=range(sizes['cn_r1_distill'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, OpusReasoningProcessor,
              DatasetMeta(dataset_id=OPUS_REASONING_REPO, split='train',
                          data_slice=range(sizes['opus_reasoning'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, ClaudeOpusProcessor,
              DatasetMeta(dataset_id=CLAUDE_OPUS_REPO, split='train',
                          data_slice=range(sizes['claude_opus'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, AngrygiraffeOpusReasoningProcessor,
              DatasetMeta(dataset_id=ANGRYGIRAFFE_REPO, split='train',
                          data_slice=range(sizes['angrygiraffe'])),
              load_from_cache_file=load_from_cache_file)

    dataset.mix_dataset(False)
    return dataset


if __name__ == '__main__':
    dataset = get_dataset(load_from_cache_file=True)
    print(len(dataset))
