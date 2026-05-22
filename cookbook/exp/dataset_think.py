import hashlib
import httpx
import re
from typing import Any, Dict, List, Optional

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import Preprocessor

dataset = Dataset()

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def _hash_id(prefix: str, content: str) -> str:
    return f'{prefix}__{hashlib.md5(content.encode("utf-8")).hexdigest()[:16]}'


def _register(processor_cls, meta: DatasetMeta, init_args: Optional[Dict[str, Any]] = None) -> None:
    """Add dataset and run preprocessor; auto-strip every input column to enforce
    the universal ``{id, source, query, cot, response}`` output schema."""
    dataset.add_dataset(meta)
    cols = list(dataset.datasets[meta.get_id()].column_names)
    dataset.map(
        processor_cls,
        dataset_meta=meta,
        init_args=init_args or {},
        remove_columns=cols,
        load_from_cache_file=True,
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


_register(CodeXThinkingProcessor,
          DatasetMeta(dataset_id=CODEX_THINKING_REPO, split='train'))


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


_register(OpenThoughtsProcessor,
          DatasetMeta(dataset_id=OPEN_THOUGHTS_REPO, split='train'))


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


_register(LIMOProcessor,
          DatasetMeta(dataset_id=LIMO_REPO, split='train'))


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
            out.append({
                'id': _hash_id('cn_r1_distill', f'{query}\n{response}'),
                'source': 'Chinese-DeepSeek-R1-Distill-data-110k',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


_register(ChineseR1DistillProcessor,
          DatasetMeta(dataset_id=CN_R1_DISTILL_REPO, split='train'))
