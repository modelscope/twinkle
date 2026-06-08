import hashlib
import re
from typing import Any, Dict, List, Optional

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import Preprocessor

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def _hash_id(prefix: str, content: str) -> str:
    return f'{prefix}__{hashlib.md5(content.encode("utf-8")).hexdigest()[:16]}'


def _register(dataset, processor_cls, meta: DatasetMeta, init_args: Optional[Dict[str, Any]] = None,
              load_from_cache_file: bool = True) -> None:
    """Add dataset and run preprocessor; auto-strip every input column to enforce
    the universal ``{id, source, query, cot, response}`` output schema."""
    dataset.add_dataset(meta)
    cols = list(dataset.datasets[meta.get_id()].column_names)
    dataset.map(
        processor_cls,
        dataset_meta=meta,
        init_args=init_args or {},
        remove_columns=cols,
        load_from_cache_file=load_from_cache_file,
    )


# ===== Modotte/CodeX-2M-Thinking =====
CODEX_THINKING_REPO = 'ms://Modotte/CodeX-2M-Thinking'


class CodeXThinkingProcessor(Preprocessor):
    """CodeX-2M-Thinking row → ``{id, source, query, cot, response}``。

    输入 schema: ``input``（问题）、``output``（含 ``<think>...</think>`` + 答案）。
    拆分 output 为 cot（think 标签内容）和 response（标签之后的正文）。
    丢弃缺失 input/output 或无法解析 think 标签的行。
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('input') or '').strip()
            output = (row.get('output') or '').strip()
            if not query or not output:
                continue
            m = _THINK_RE.search(output)
            if not m:
                continue
            cot = m.group(1).strip()
            response = output[m.end():].strip()
            if not cot or not response:
                continue
            out.append({
                'id': _hash_id('codex_think', f'{query}\n{response}'),
                'source': 'CodeX-2M-Thinking',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===== open-thoughts/OpenThoughts3-1.2M =====
OPEN_THOUGHTS_REPO = 'ms://open-thoughts/OpenThoughts3-1.2M'


class OpenThoughtsProcessor(Preprocessor):
    """OpenThoughts3 row → ``{id, source, query, cot, response}``。

    输入 schema: ``conversations`` (messages 格式 list[{from/value}])。
    取第一个 human 作 query，第一个 gpt 的 value 按 ``<think>...</think>`` 拆 cot/response。
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            convs = row.get('conversations')
            if not isinstance(convs, list):
                continue
            query = ''
            assistant_text = ''
            for msg in convs:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('from') or msg.get('role') or ''
                value = msg.get('value') or msg.get('content') or ''
                if role in ('human', 'user') and not query:
                    query = value.strip()
                elif role in ('gpt', 'assistant') and not assistant_text:
                    assistant_text = value.strip()
                    break
            if not query or not assistant_text:
                continue
            m = _THINK_RE.search(assistant_text)
            if not m:
                continue
            cot = m.group(1).strip()
            response = assistant_text[m.end():].strip()
            if not cot or not response:
                continue
            out.append({
                'id': _hash_id('openthoughts', f'{query}\n{response}'),
                'source': 'OpenThoughts3-1.2M',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===== GAIR/LIMO-v2 =====
LIMO_REPO = 'ms://GAIR/LIMO-v2'


class LIMOProcessor(Preprocessor):
    """LIMO-v2 row → ``{id, source, query, cot, response}``。

    输入 schema: ``question``、``solution``（含 ``<think>...</think>`` + 答案）。
    拆分 solution 为 cot 和 response。
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('question') or '').strip()
            solution = (row.get('solution') or '').strip()
            if not query or not solution:
                continue
            m = _THINK_RE.search(solution)
            if m:
                cot = m.group(1).strip()
                response = solution[m.end():].strip()
            else:
                # 无 think 标签时，solution 整体作为 response，cot 留空
                cot = ''
                response = solution
            if not response:
                continue
            out.append({
                'id': _hash_id('limo', f'{query}\n{response}'),
                'source': 'LIMO-v2',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===== AI-ModelScope/Chinese-DeepSeek-R1-Distill-data-110k =====
CN_R1_DISTILL_REPO = 'ms://AI-ModelScope/Chinese-DeepSeek-R1-Distill-data-110k'


class ChineseR1DistillProcessor(Preprocessor):
    """Chinese-DeepSeek-R1-Distill row → ``{id, source, query, cot, response}``。

    输入已有三列: ``input`` → query, ``reasoning_content`` → cot, ``content`` → response。
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('input') or '').strip()
            cot = (row.get('reasoning_content') or '').strip()
            response = (row.get('content') or '').strip()
            if not query or not response:
                continue
            if cot:
                response = _THINK_RE.sub('', response).strip()
            if not response:
                continue
            out.append({
                'id': _hash_id('cn_r1_distill', f'{query}\n{response}'),
                'source': 'Chinese-DeepSeek-R1-Distill-data-110k',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===== nohurry/Opus-4.6-Reasoning-3000x-filtered =====
OPUS_REASONING_REPO = 'ms://nohurry/Opus-4.6-Reasoning-3000x-filtered'


class OpusReasoningProcessor(Preprocessor):
    """Opus-4.6-Reasoning-3000x-filtered row → ``{id, source, query, cot, response}``。

    输入已有三列: ``problem`` → query, ``thinking`` → cot, ``solution`` → response。
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('problem') or '').strip()
            cot = (row.get('thinking') or '').strip()
            response = (row.get('solution') or '').strip()
            if not query or not response:
                continue
            if cot:
                response = _THINK_RE.sub('', response).strip()
            if not response:
                continue
            out.append({
                'id': _hash_id('opus_reasoning', f'{query}\n{response}'),
                'source': 'Opus-4.6-Reasoning-3000x-filtered',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===== Roman1111111/claude-opus-4.6-10000x =====
CLAUDE_OPUS_REPO = 'ms://Roman1111111/claude-opus-4.6-10000x'


class ClaudeOpusProcessor(Preprocessor):
    """claude-opus-4.6-10000x row → ``{id, source, query, cot, response}``。

    输入 schema: ``messages`` (OpenAI 格式 list[{role, content}])。
    取首个 user 作 query，首个 assistant 按 ``<think>...</think>`` 拆 cot/response。
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
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
            out.append({
                'id': _hash_id('claude_opus', f'{query}\n{response}'),
                'source': 'claude-opus-4.6-10000x',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


ANGRYGIRAFFE_REPO = 'ms://hf/angrygiraffe-claude-opus-4.6-4.7-reasoning-8.7k'


class AngrygiraffeOpusReasoningProcessor(Preprocessor):
    """angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k row → ``{id, source, query, cot, response}``。

    输入 schema: ``messages`` (OpenAI 格式 list[{role, content}])。
    取首个 user 作 query，首个 assistant 按 ``<think>...</think>`` 拆 cot/response，仅用头一轮。
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
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
            out.append({
                'id': _hash_id('angrygiraffe_opus', f'{query}\n{response}'),
                'source': 'angrygiraffe-claude-opus-4.6-4.7-reasoning-8.7k',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


_BASE_SIZES = {
    'codex_think': 100000,
    'open_thoughts': 400000,
    'cn_r1_distill': 100000,
    'opus_reasoning': 3000,
    'claude_opus': 10000,
    'angrygiraffe': 38000,
}


def _scaled_sizes(total: Optional[int]) -> Dict[str, int]:
    if total is None:
        return dict(_BASE_SIZES)
    scale = total / sum(_BASE_SIZES.values())
    return {k: max(1, int(round(v * scale))) for k, v in _BASE_SIZES.items()}


def _build_dataset(total: Optional[int] = None, load_from_cache_file: bool = True) -> Dataset:
    sizes = _scaled_sizes(total)
    dataset = Dataset()

    _register(dataset, CodeXThinkingProcessor,
              DatasetMeta(dataset_id=CODEX_THINKING_REPO, split='train',
                          data_slice=range(sizes['codex_think'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, OpenThoughtsProcessor,
              DatasetMeta(dataset_id=OPEN_THOUGHTS_REPO, split='train',
                          data_slice=range(sizes['open_thoughts'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, LIMOProcessor,
              DatasetMeta(dataset_id=LIMO_REPO, split='train'),
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


class ToMessagesProcessor(Preprocessor):
    """Convert {query, cot, response} → {id, source, messages}."""

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = row.get('query') or ''
            cot = row.get('cot') or ''
            response = row.get('response') or ''
            if not cot:
                continue
            assistant_content = f'<think>{cot}</think>'
            out.append({
                'id': row.get('id', ''),
                'source': row.get('source', ''),
                'messages': [
                    {'role': 'user', 'content': query},
                    {'role': 'assistant', 'content': assistant_content,
                     'reasoning_content': cot},
                ],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])


def get_dataset(total: Optional[int] = None, dropped_log: Optional[str] = None,
                load_from_cache_file: bool = True) -> Dataset:
    """Build, convert to messages format, and quality-filter the CoT dataset.

    If ``total`` is given, every per-source row count in ``_BASE_SIZES`` is
    scaled proportionally so the input-row sum approximates ``total``.
    """
    from twinkle_agentic.preprocessor import (
        DeadLoopFilter,
        FixUnicodeFilter,
        HardFilter,
        IntentClassifier,
        MessageSanityFilter,
        QualityPreprocessor,
        RefuseFilter,
        RemoveRepeatSentencesFilter,
        TokenNumFilter,
        TokenSoupFilter,
    )

    dataset = _build_dataset(total=total, load_from_cache_file=load_from_cache_file)
    dataset.map(ToMessagesProcessor(), remove_columns=['query', 'cot', 'response'],
                load_from_cache_file=load_from_cache_file)
    qp = QualityPreprocessor(
        pipeline=[
            HardFilter(),
            RefuseFilter(),
            DeadLoopFilter(),
            TokenSoupFilter(),
            MessageSanityFilter(min_turns=1, max_msg_chars=200000),
            FixUnicodeFilter(),
            RemoveRepeatSentencesFilter(),
            TokenNumFilter(max_num=32768),
        ],
        dropped_log_path=dropped_log or '',
    )
    dataset.map(qp, num_proc=32, load_from_cache_file=load_from_cache_file)
    return dataset


if __name__ == '__main__':
    import os
    dropped_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dropped.jsonl')
    if os.path.exists(dropped_log):
        os.remove(dropped_log)
    dataset = get_dataset(load_from_cache_file=False)
    print(len(dataset))
